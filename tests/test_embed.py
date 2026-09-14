"""Tests for the embedding gateway.

The central claim these tests defend: a fully cached call loads no model.
Every test replaces `embed._load_model` — by default with a sentinel that
fails the test if it is ever reached — so "no model load" is proven by the
test passing, not by a mock quietly returning a value.

No test here downloads all-MiniLM-L6-v2. The encoder is always faked, so the
suite runs offline and in milliseconds; what is under test is the caching and
ordering logic, not sentence-transformers.
"""

import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src import embed

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    """Every test gets an empty cache dir, no loaded model, and no CACHE_ONLY."""
    monkeypatch.setattr(embed, "CACHE_DIR", tmp_path / "embeddings")
    monkeypatch.setattr(embed, "_model", None)
    monkeypatch.delenv("CACHE_ONLY", raising=False)


def fake_vector(text: str) -> np.ndarray:
    """A deterministic unit vector per text. Stands in for the real encoder.

    Seeded from sha256, not hash() -- Python salts string hashing per process,
    so hash() would give different vectors in the subprocess tests.
    """
    digest = hashlib.sha256(text.encode()).digest()[:4]
    rng = np.random.default_rng(int.from_bytes(digest, "big"))
    vector = rng.normal(size=embed.DIM).astype(np.float32)
    return vector / np.linalg.norm(vector)


def no_model(monkeypatch):
    """Install a _load_model() that fails the test if any code path calls it."""

    def _boom():
        raise AssertionError("model was loaded")

    monkeypatch.setattr(embed, "_load_model", _boom)


def recording_encode(monkeypatch):
    """Install an _encode() that returns fake vectors and records its batches."""
    batches = []

    def _encode(texts):
        batches.append(list(texts))
        return np.stack([fake_vector(t) for t in texts])

    monkeypatch.setattr(embed, "_encode", _encode)
    return batches


# --------------------------------------------------------------------------


def test_cache_dir_is_anchored_to_repo_root_not_cwd(tmp_path):
    """CACHE_DIR must come from embed.py's own location, not the working directory.

    Run in a subprocess from an unrelated cwd: the autouse fixture repoints
    CACHE_DIR for every other test, so the real value can only be observed in
    a fresh interpreter.
    """
    result = subprocess.run(
        [sys.executable, "-c", "from src import embed; print(embed.CACHE_DIR)"],
        cwd=tmp_path,
        env={"PYTHONPATH": str(REPO_ROOT), "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        check=True,
    )

    assert Path(result.stdout.strip()) == REPO_ROOT / "data" / "cache" / "embeddings"


def test_importing_embed_does_not_import_torch(tmp_path):
    """The sentence_transformers import must stay inside _load_model().

    If it ever moves to module top level, a fully cached run pays a multi
    second torch import for nothing.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from src import embed; print('torch' in sys.modules)",
        ],
        cwd=tmp_path,
        env={"PYTHONPATH": str(REPO_ROOT), "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout.strip() == "False"


def test_cache_hit_loads_no_model(monkeypatch):
    key = embed._cache_key("hello")
    embed._write_cache(key, fake_vector("hello"))

    no_model(monkeypatch)

    out = embed.embed(["hello"])
    assert out.shape == (1, embed.DIM)
    np.testing.assert_array_equal(out[0], fake_vector("hello"))


def test_live_call_is_cached_then_replayed(monkeypatch):
    batches = recording_encode(monkeypatch)

    first = embed.embed(["hello"])
    assert first.shape == (1, embed.DIM)
    assert batches == [["hello"]]

    assert embed._cache_path(embed._cache_key("hello")).exists()

    second = embed.embed(["hello"])
    np.testing.assert_array_equal(first, second)
    assert batches == [["hello"]], "second identical call must replay from cache"


def test_empty_input_returns_empty_without_model(monkeypatch):
    no_model(monkeypatch)

    out = embed.embed([])
    assert out.shape == (0, embed.DIM)


def test_duplicate_texts_are_encoded_once(monkeypatch):
    batches = recording_encode(monkeypatch)

    out = embed.embed(["a", "b", "a", "a"])

    assert out.shape == (4, embed.DIM)
    assert batches == [["a", "b"]], "each unique text must reach the encoder exactly once"
    np.testing.assert_array_equal(out[0], out[2])
    np.testing.assert_array_equal(out[0], out[3])
    assert not np.array_equal(out[0], out[1])


def test_row_order_matches_input_under_mixed_hits_and_misses(monkeypatch):
    # Pre-cache the middle text only, so the encode order and the input order
    # genuinely differ and a reassembly bug cannot pass by luck.
    embed._write_cache(embed._cache_key("cached"), fake_vector("cached"))

    batches = recording_encode(monkeypatch)

    out = embed.embed(["miss1", "cached", "miss2"])

    assert batches == [["miss1", "miss2"]]
    np.testing.assert_array_equal(out[0], fake_vector("miss1"))
    np.testing.assert_array_equal(out[1], fake_vector("cached"))
    np.testing.assert_array_equal(out[2], fake_vector("miss2"))


def test_cache_only_raises_on_miss_without_loading_model(monkeypatch):
    monkeypatch.setenv("CACHE_ONLY", "1")
    no_model(monkeypatch)

    with pytest.raises(embed.CacheOnlyError) as exc:
        embed.embed(["uncached"])

    assert embed.MODEL in str(exc.value)
    assert "uncached" in str(exc.value)


def test_cache_only_still_serves_hits(monkeypatch):
    embed._write_cache(embed._cache_key("hello"), fake_vector("hello"))

    monkeypatch.setenv("CACHE_ONLY", "1")
    no_model(monkeypatch)

    out = embed.embed(["hello"])
    np.testing.assert_array_equal(out[0], fake_vector("hello"))


def test_cache_key_depends_on_every_input():
    base = embed._cache_key("text", model="m")

    assert base == embed._cache_key("text", model="m")
    assert base != embed._cache_key("text2", model="m")
    assert base != embed._cache_key("text", model="m2")

    # Field boundaries must not blur: ("ab","c") and ("a","bc") are different keys.
    assert embed._cache_key("c", model="ab") != embed._cache_key("bc", model="a")


def test_wrong_shaped_cache_entry_raises_rather_than_propagating(monkeypatch):
    """A cache written by a different model must fail loudly, not reshape."""
    key = embed._cache_key("hello")
    embed._cache_path(key).parent.mkdir(parents=True, exist_ok=True)
    np.save(embed._cache_path(key), np.zeros(768, dtype=np.float32))

    no_model(monkeypatch)

    with pytest.raises(embed.EmbedError, match="different model"):
        embed.embed(["hello"])


def test_write_leaves_no_tmp_file(monkeypatch):
    recording_encode(monkeypatch)

    embed.embed(["hello"])

    leftovers = list(embed.CACHE_DIR.glob("*.tmp"))
    assert leftovers == [], f"atomic write left {leftovers}"


def test_cache_status_counts_unique_texts_without_embedding(monkeypatch):
    embed._write_cache(embed._cache_key("hello"), fake_vector("hello"))

    no_model(monkeypatch)

    cached, uncached = embed.cache_status(["hello", "hello", "new"])
    assert (cached, uncached) == (1, 1), "duplicates must collapse before counting"
