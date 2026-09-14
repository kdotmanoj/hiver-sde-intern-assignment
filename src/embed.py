"""The single gateway for every embedding in this project.

Deliberately shaped like src/llm.py, so the two read the same way: one cache
file per key, sha256 over the inputs that determine the vector, atomic writes,
and CACHE_ONLY=1 to turn a miss into a loud error. Read that module first and
this one holds no surprises.

Three differences from llm.py, all because the model runs locally:

1. No network, no rate limiting, no backoff. all-MiniLM-L6-v2 runs on CPU.
2. The model is loaded lazily and only when there is at least one miss, so a
   fully cached run never pays the torch import or the model load.
3. Cache records are raw .npy vectors, not JSON. The text is not stored
   alongside them -- the key is the only link back. That is the cost of the
   .npy format choice in CLAUDE.md; a vector with no provenance is fine here
   because the text is always still in data/interim/.

Cache location: data/cache/embeddings/, committed to git exactly like
data/cache/llm/. Not because these vectors need an API key to rebuild -- they
do not -- but because reproducing this project's results from a fresh clone
must not require a ~90MB model download from HuggingFace. 3.2MB of committed
.npy files buys a clone that satisfies CACHE_ONLY=1 immediately and replays
the whole stage offline in under 10s.

One file per vector rather than one matrix, which is what makes "never
re-embed the same text twice" hold across runs and across stages. The cost is
~2,000 git blobs; the benefit is that adding a text to the sample re-embeds
one text, not all of them.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np

# Repo root, derived from this file's location: src/embed.py -> src -> repo
# root. Same anchoring rationale as llm.py: the cache is the same directory no
# matter where a script is run from.
REPO_ROOT = Path(__file__).resolve().parent.parent

# Read through _cache_path() at call time, so tests can repoint it at a tmp dir
# with monkeypatch.setattr(embed, "CACHE_DIR", ...).
CACHE_DIR = REPO_ROOT / "data" / "cache" / "embeddings"

# Fixed by CLAUDE.md. The model id IS part of the cache key, so switching
# models invalidates the cache rather than silently mixing vector spaces.
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DIM = 384

# Vectors are L2-normalized at encode time. That makes euclidean k-means
# equivalent to cosine/spherical k-means, so the clustering and the silhouette
# score are computed in the same geometry. Normalizing here rather than at the
# call site means it can never be forgotten by one caller and not another.
NORMALIZE = True

# Encode in batches so a 2,000-text miss does not build one giant tensor.
BATCH_SIZE = 64

# Populated by _load_model() on first miss; None until then.
_model = None


class EmbedError(RuntimeError):
    """An embedding could not be produced."""


class CacheOnlyError(EmbedError):
    """CACHE_ONLY=1 is set and this text is not in the cache."""


def embed(texts: list[str]) -> np.ndarray:
    """Return an (len(texts), DIM) float32 array, from cache where possible.

    Row i is always the vector for texts[i]. Duplicate texts are encoded once
    and their rows are identical -- "never re-embed the same text twice" holds
    within a call, not only across runs.

    If every text is cached, no model is loaded and nothing is imported from
    sentence_transformers.
    """
    if not texts:
        return np.empty((0, DIM), dtype=np.float32)

    keys = [_cache_key(text) for text in texts]

    # Unique misses only, in first-appearance order so the encode is
    # deterministic. dict preserves insertion order; a set would not.
    hits: dict[str, np.ndarray] = {}
    misses: dict[str, str] = {}
    for text, key in zip(texts, keys):
        if key in hits or key in misses:
            continue
        cached = _read_cache(key)
        if cached is not None:
            hits[key] = cached
        else:
            misses[key] = text

    if misses:
        if os.environ.get("CACHE_ONLY") == "1":
            first_key, first_text = next(iter(misses.items()))
            raise CacheOnlyError(
                f"CACHE_ONLY=1 but {len(misses):,} of {len(set(keys)):,} unique "
                f"texts are not cached.\n"
                f"  model:  {MODEL}\n"
                f"  key:    {first_key}\n"
                f"  expect: {_cache_path(first_key)}\n"
                f"  text:   {first_text[:80]!r}"
            )
        fresh = _encode(list(misses.values()))
        for key, vector in zip(misses.keys(), fresh):
            _write_cache(key, vector)
            hits[key] = vector

    out = np.stack([hits[key] for key in keys])
    assert out.shape == (len(texts), DIM), f"expected {(len(texts), DIM)}, got {out.shape}"
    return out


def cache_status(texts: list[str]) -> tuple[int, int]:
    """(cached, uncached) counts over the UNIQUE texts, without embedding anything.

    Used by the stage summary so it can report "1,998 cached, 2 computed"
    before the work happens rather than instrumenting embed() itself.
    """
    keys = {_cache_key(text) for text in texts}
    cached = sum(1 for key in keys if _cache_path(key).exists())
    return cached, len(keys) - cached


# --------------------------------------------------------------------------
# cache
# --------------------------------------------------------------------------


def _cache_key(text: str, model: str = MODEL) -> str:
    """sha256 over the two inputs that determine the vector.

    The NUL separator means ("ab", "c") and ("a", "bc") cannot hash alike.
    NORMALIZE is not in the key for the same reason llm.py leaves temperature
    out: it is fixed for the whole project, so flipping it makes the cache
    stale by definition and it must be cleared on purpose.
    """
    payload = "\x00".join([model, text])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cache_path(key: str) -> Path:
    return CACHE_DIR / f"{key}.npy"


def _read_cache(key: str) -> np.ndarray | None:
    path = _cache_path(key)
    if not path.exists():
        return None
    vector = np.load(path)
    if vector.shape != (DIM,):
        raise EmbedError(
            f"cached vector {path} has shape {vector.shape}, expected {(DIM,)}. "
            f"The cache was written by a different model; clear it."
        )
    return vector


def _write_cache(key: str, vector: np.ndarray) -> None:
    """Write atomically, so an interrupted run cannot leave a half file.

    np.save is handed an open file object rather than a path: given a path not
    ending in .npy it silently appends the suffix, so saving to "<key>.npy.tmp"
    would write "<key>.npy.tmp.npy" and the rename below would fail.
    """
    path = _cache_path(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".npy.tmp")
    with open(tmp, "wb") as handle:
        np.save(handle, vector, allow_pickle=False)
    os.replace(tmp, path)


# --------------------------------------------------------------------------
# the model
# --------------------------------------------------------------------------


def _load_model():
    """Import and load sentence-transformers. Called only when there is a miss.

    The import is inside the function, not at module top level, so that
    importing src.embed for a fully cached run does not pull in torch.
    """
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(MODEL)
    return _model


def _encode(texts: list[str]) -> np.ndarray:
    """Encode texts that are known to be uncached. Returns (len(texts), DIM) float32."""
    model = _load_model()
    vectors = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=NORMALIZE,
        convert_to_numpy=True,
        show_progress_bar=len(texts) > BATCH_SIZE,
    )
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.shape != (len(texts), DIM):
        raise EmbedError(
            f"{MODEL} returned shape {vectors.shape}, expected {(len(texts), DIM)}."
        )
    return vectors
