"""Tests for the LLM gateway.

The central claim these tests defend: a cache hit performs zero network calls.
Every test replaces `llm.requests.post` — by default with a sentinel that fails
the test if it is ever reached — so "no network" is proven by the test passing,
not by a mock quietly returning a value.
"""

import json

import pytest

from src import llm


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    """Every test gets an empty cache dir, fake keys, and no CACHE_ONLY."""
    monkeypatch.setattr(llm, "CACHE_DIR", tmp_path / "llm")
    monkeypatch.setattr(llm, "_last_gemini_call", None)
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    monkeypatch.delenv("CACHE_ONLY", raising=False)


class FakeResponse:
    def __init__(self, status_code, body=None, headers=None):
        self.status_code = status_code
        self._body = body or {}
        self.headers = headers or {}
        self.text = json.dumps(self._body)

    def json(self):
        return self._body


def gemini_body(text):
    return {"candidates": [{"content": {"parts": [{"text": text}]}, "finishReason": "STOP"}]}


def groq_body(text):
    return {"choices": [{"message": {"content": text}, "finish_reason": "stop"}]}


def no_network(monkeypatch):
    """Install a post() that fails the test if any code path calls it."""

    def _boom(*args, **kwargs):
        raise AssertionError("network call was made")

    monkeypatch.setattr(llm.requests, "post", _boom)


def recording_post(monkeypatch, responses):
    """Install a post() that returns `responses` in order and records its calls."""
    calls = []
    queue = list(responses)

    def _post(url, headers=None, json=None, timeout=None):
        calls.append({"url": url, "headers": headers, "json": json})
        return queue.pop(0)

    monkeypatch.setattr(llm.requests, "post", _post)
    return calls


def no_sleep(monkeypatch):
    """Install a sleep() that records durations instead of actually sleeping."""
    slept = []
    monkeypatch.setattr(llm.time, "sleep", slept.append)
    return slept


# --------------------------------------------------------------------------


def test_cache_hit_makes_no_network_call(monkeypatch):
    key = llm._cache_key(prompt="hello", model="gemini-2.0-flash", system=None)
    llm._write_cache(key, model="gemini-2.0-flash", system=None, prompt="hello", response="cached reply")

    no_network(monkeypatch)

    assert llm.complete(prompt="hello", model="gemini-2.0-flash") == "cached reply"


def test_live_call_is_cached_then_replayed(monkeypatch):
    calls = recording_post(monkeypatch, [FakeResponse(200, gemini_body("fresh reply"))])
    no_sleep(monkeypatch)

    first = llm.complete(prompt="hello", model="gemini-2.0-flash")
    assert first == "fresh reply"
    assert len(calls) == 1

    key = llm._cache_key(prompt="hello", model="gemini-2.0-flash", system=None)
    assert llm._cache_path(key).exists()

    second = llm.complete(prompt="hello", model="gemini-2.0-flash")
    assert second == "fresh reply"
    assert len(calls) == 1, "second identical call must replay from cache"


def test_cache_only_raises_on_miss_without_network(monkeypatch):
    monkeypatch.setenv("CACHE_ONLY", "1")
    no_network(monkeypatch)

    with pytest.raises(llm.CacheOnlyError) as exc:
        llm.complete(prompt="uncached", model="gemini-2.0-flash")

    assert "gemini-2.0-flash" in str(exc.value)
    assert "uncached" in str(exc.value)


def test_cache_only_still_serves_hits(monkeypatch):
    key = llm._cache_key(prompt="hello", model="gemini-2.0-flash", system=None)
    llm._write_cache(key, model="gemini-2.0-flash", system=None, prompt="hello", response="cached reply")

    monkeypatch.setenv("CACHE_ONLY", "1")
    no_network(monkeypatch)

    assert llm.complete(prompt="hello", model="gemini-2.0-flash") == "cached reply"


def test_cache_key_depends_on_every_input():
    base = llm._cache_key(prompt="p", model="m", system="s")

    assert base == llm._cache_key(prompt="p", model="m", system="s")
    assert base != llm._cache_key(prompt="p2", model="m", system="s")
    assert base != llm._cache_key(prompt="p", model="m2", system="s")
    assert base != llm._cache_key(prompt="p", model="m", system=None)

    # Field boundaries must not blur: ("ab","c") and ("a","bc") are different keys.
    assert llm._cache_key(prompt="x", model="ab", system="c") != llm._cache_key(
        prompt="x", model="a", system="bc"
    )


def test_cache_record_stores_the_prompt(monkeypatch):
    recording_post(monkeypatch, [FakeResponse(200, gemini_body("fresh reply"))])
    no_sleep(monkeypatch)

    llm.complete(prompt="hello", model="gemini-2.0-flash", system="be terse")

    key = llm._cache_key(prompt="hello", model="gemini-2.0-flash", system="be terse")
    record = json.loads(llm._cache_path(key).read_text())
    assert record["prompt"] == "hello"
    assert record["system"] == "be terse"
    assert record["response"] == "fresh reply"


def test_retries_with_exponential_backoff_on_429(monkeypatch):
    calls = recording_post(
        monkeypatch,
        [
            FakeResponse(429, {"error": "rate limited"}),
            FakeResponse(429, {"error": "rate limited"}),
            FakeResponse(200, gemini_body("finally")),
        ],
    )
    slept = no_sleep(monkeypatch)

    assert llm.complete(prompt="hello", model="gemini-2.0-flash") == "finally"
    assert len(calls) == 3

    # Backoff sleeps are 4s then 8s; the rest are the 4s inter-call Gemini gap.
    backoffs = [s for s in slept if s in (4.0, 8.0)]
    assert 8.0 in backoffs and backoffs.index(8.0) > backoffs.index(4.0)


def test_retry_after_header_wins_over_backoff_schedule(monkeypatch):
    recording_post(
        monkeypatch,
        [
            FakeResponse(429, {"error": "slow down"}, headers={"Retry-After": "17"}),
            FakeResponse(200, gemini_body("ok")),
        ],
    )
    slept = no_sleep(monkeypatch)

    assert llm.complete(prompt="hello", model="gemini-2.0-flash") == "ok"
    assert 17.0 in slept


def test_gives_up_after_max_retries(monkeypatch):
    calls = recording_post(
        monkeypatch,
        [FakeResponse(429, {"error": "rate limited"})] * llm.MAX_RETRIES,
    )
    no_sleep(monkeypatch)

    with pytest.raises(llm.LLMError):
        llm.complete(prompt="hello", model="gemini-2.0-flash")

    assert len(calls) == llm.MAX_RETRIES


def test_auth_failure_is_not_retried(monkeypatch):
    calls = recording_post(monkeypatch, [FakeResponse(401, {"error": "bad key"})])
    no_sleep(monkeypatch)

    with pytest.raises(llm.LLMError, match="auth failed"):
        llm.complete(prompt="hello", model="gemini-2.0-flash")

    assert len(calls) == 1


def test_consecutive_live_gemini_calls_are_spaced(monkeypatch):
    recording_post(
        monkeypatch,
        [FakeResponse(200, gemini_body("one")), FakeResponse(200, gemini_body("two"))],
    )
    slept = no_sleep(monkeypatch)

    llm.complete(prompt="first", model="gemini-2.0-flash")
    assert slept == [], "the first live call should not wait"

    llm.complete(prompt="second", model="gemini-2.0-flash")
    # Patched sleep means no wall-clock time passes, so the full gap is required.
    assert slept and slept[0] == pytest.approx(llm.GEMINI_MIN_GAP_S, abs=0.1)

    # A cache hit after that must not sleep at all.
    before = len(slept)
    llm.complete(prompt="first", model="gemini-2.0-flash")
    assert len(slept) == before


def test_groq_routing_uses_openai_shaped_request(monkeypatch):
    calls = recording_post(monkeypatch, [FakeResponse(200, groq_body("judged"))])
    slept = no_sleep(monkeypatch)

    result = llm.complete(prompt="grade this", model="openai/gpt-oss-120b", system="be strict")

    assert result == "judged"
    assert calls[0]["url"] == llm.GROQ_URL
    assert calls[0]["headers"]["Authorization"] == "Bearer test-groq-key"
    assert calls[0]["json"]["messages"] == [
        {"role": "system", "content": "be strict"},
        {"role": "user", "content": "grade this"},
    ]
    assert slept == [], "the Gemini pacing gap must not apply to Groq"


def test_gemini_request_shape(monkeypatch):
    calls = recording_post(monkeypatch, [FakeResponse(200, gemini_body("ok"))])
    no_sleep(monkeypatch)

    llm.complete(prompt="hi", model="gemini-2.0-flash", system="be terse")

    assert calls[0]["url"].endswith("/models/gemini-2.0-flash:generateContent")
    assert calls[0]["headers"]["x-goog-api-key"] == "test-gemini-key"
    assert calls[0]["json"]["contents"] == [{"role": "user", "parts": [{"text": "hi"}]}]
    assert calls[0]["json"]["systemInstruction"] == {"parts": [{"text": "be terse"}]}
    assert calls[0]["json"]["generationConfig"]["temperature"] == llm.TEMPERATURE


def test_unknown_model_raises_without_network(monkeypatch):
    no_network(monkeypatch)

    with pytest.raises(llm.LLMError, match="Unknown model id"):
        llm.complete(prompt="hello", model="gpt-4o")


def test_missing_api_key_raises_without_network(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    no_network(monkeypatch)

    with pytest.raises(llm.LLMError, match="GEMINI_API_KEY"):
        llm.complete(prompt="hello", model="gemini-2.0-flash")


def test_empty_completion_is_not_cached(monkeypatch):
    recording_post(monkeypatch, [FakeResponse(200, {"candidates": [{"finishReason": "MAX_TOKENS"}]})])
    no_sleep(monkeypatch)

    with pytest.raises(llm.LLMError, match="empty text"):
        llm.complete(prompt="hello", model="gemini-2.0-flash")

    key = llm._cache_key(prompt="hello", model="gemini-2.0-flash", system=None)
    assert not llm._cache_path(key).exists()
