"""The single gateway for every LLM call in this project.

Nothing else in the codebase is allowed to talk to an LLM API directly. Routing
that through here buys three things:

1. Every response is cached to disk by sha256(model + system + prompt), so an
   evaluation run replays from committed cache with no API key and no network.
2. Free-tier rate limits are respected in exactly one place (4s between live
   calls to each provider, on its own clock, plus exponential backoff on 429).
3. Setting CACHE_ONLY=1 turns any cache miss into a loud error instead of a
   surprise API call.

Provider calls are plain REST via `requests` rather than vendor SDKs, so the
retry and pacing behaviour is the code below and nothing else.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

# A real environment variable beats .env, so CI and one-off overrides work.
load_dotenv(override=False)

# Repo root, derived from this file's location: src/llm.py -> src -> repo root.
# Anchoring here means the cache is the same directory whether a script is run
# from the repo root, from scripts/, or from anywhere else.
REPO_ROOT = Path(__file__).resolve().parent.parent

# Where cache records live. Read through _cache_path() at call time, so tests
# can repoint it at a tmp dir with monkeypatch.setattr(llm, "CACHE_DIR", ...).
CACHE_DIR = REPO_ROOT / "data" / "cache" / "llm"

# Generation settings are deliberately NOT part of the cache key: they are fixed
# for the whole project. If one of these ever changes, the cache is stale by
# definition and must be cleared on purpose.
TEMPERATURE = 0.0
MAX_OUTPUT_TOKENS = 1024

# Gemini free tier is ~15 requests/minute. 4s between live calls keeps us under
# it without any token-bucket machinery.
GEMINI_MIN_GAP_S = 4.0

# Groq's free tier allows 30 requests/minute but only 8,000 tokens/minute, and
# TPM binds first: a judge call is ~520 tokens, so 8000/520 is ~15 calls/minute,
# half the RPM allowance. 4s between live calls keeps us under the tokens/minute
# ceiling, which the request counter alone would sail straight past.
GROQ_MIN_GAP_S = 4.0

# The gap each provider is paced at, looked up by _wait_for_slot().
MIN_GAP_S = {"gemini": GEMINI_MIN_GAP_S, "groq": GROQ_MIN_GAP_S}

MAX_RETRIES = 5
BACKOFF_BASE_S = 4.0
BACKOFF_CAP_S = 64.0
TIMEOUT_S = 60

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# Monotonic timestamp of the last live request PER PROVIDER, for the gap above.
# Keyed by provider rather than global: a Gemini call should not make the judge
# wait, and vice versa -- the two quotas are entirely separate.
_last_call: dict[str, float] = {}


class LLMError(RuntimeError):
    """An LLM call could not be completed (bad model id, bad key, retries exhausted)."""


class CacheOnlyError(LLMError):
    """CACHE_ONLY=1 is set and this prompt is not in the cache."""


def cached_response(
    prompt: str, model: str, system: str | None = None, variant: int = 0
) -> str | None:
    """The cached completion for these inputs, or None if there is no cache entry.

    A read-only probe: it never calls a provider, never sleeps, and never writes.
    Exists so a caller can count how many live calls a planned batch would need
    BEFORE spending any of them -- see src/agent.py's preflight. Without this,
    the only way to find out is to start the run and hit the quota wall partway
    through, which is exactly the failure it is there to prevent.
    """
    return _read_cache(_cache_key(prompt=prompt, model=model, system=system, variant=variant))


def complete(prompt: str, model: str, system: str | None = None, variant: int = 0) -> str:
    """Return the model's completion for `prompt`, from cache when possible.

    A cache hit returns before any network code is reached: no key lookup, no
    sleep, no HTTP.

    `variant` asks the same question again instead of reading the last answer.
    It changes the CACHE KEY ONLY and never reaches the provider (see
    _cache_key), so variant=1 sends byte-identical model, system and prompt to a
    different cache slot. That is what makes src/judge.py's --repeat a
    measurement: without it, judging identical input twice returns the identical
    cached string and reports perfect self-consistency it has not observed.
    """
    key = _cache_key(prompt=prompt, model=model, system=system, variant=variant)

    cached = _read_cache(key)
    if cached is not None:
        return cached

    if os.environ.get("CACHE_ONLY") == "1":
        raise CacheOnlyError(
            f"CACHE_ONLY=1 but this prompt is not cached.\n"
            f"  model:   {model}\n"
            f"  variant: {variant}\n"
            f"  key:     {key}\n"
            f"  expect:  {_cache_path(key)}\n"
            f"  prompt:  {prompt[:80]!r}"
        )

    provider = _provider(model)
    if provider == "gemini":
        response = _call_gemini(prompt=prompt, model=model, system=system)
    else:
        response = _call_groq(prompt=prompt, model=model, system=system)

    _write_cache(
        key, model=model, system=system, prompt=prompt, response=response, variant=variant
    )
    return response


# --------------------------------------------------------------------------
# cache
# --------------------------------------------------------------------------


def _cache_key(prompt: str, model: str, system: str | None, variant: int = 0) -> str:
    """sha256 over the inputs that determine the response.

    The NUL separator means ("ab", "c") and ("a", "bc") cannot hash alike.

    variant=0 hashes exactly the three original components and nothing else.
    That is deliberate and load-bearing: appending "\\x000" unconditionally would
    change every key in the project and invalidate the whole committed cache, so
    a normal call keeps the key it has always had and only an explicit repeat
    (variant >= 1) lands in a new file.
    """
    components = [model, system or "", prompt]
    if variant:
        components.append(str(variant))
    return hashlib.sha256("\x00".join(components).encode("utf-8")).hexdigest()


def _cache_path(key: str) -> Path:
    return CACHE_DIR / f"{key}.json"


def _read_cache(key: str) -> str | None:
    path = _cache_path(key)
    if not path.exists():
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    return record["response"]


def _write_cache(
    key: str, model: str, system: str | None, prompt: str, response: str, variant: int = 0
) -> None:
    """Write the record atomically, so an interrupted run cannot leave a half file."""
    record = {
        "model": model,
        "system": system,
        "prompt": prompt,
        "response": response,
        # Recorded so a repeat is identifiable on disk. Two files with the same
        # prompt and different responses are the self-consistency measurement,
        # not a bug someone should go hunting for.
        "variant": variant,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    path = _cache_path(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


# --------------------------------------------------------------------------
# routing
# --------------------------------------------------------------------------


def _provider(model: str) -> str:
    """Map a model id to a provider. Unknown ids raise rather than defaulting."""
    if model.startswith("gemini"):
        return "gemini"
    if model.startswith("openai/gpt-oss") or model.startswith("llama"):
        return "groq"
    raise LLMError(
        f"Unknown model id {model!r}. Known prefixes: 'gemini' (Gemini), "
        f"'openai/gpt-oss' and 'llama' (Groq)."
    )


def _require_key(env_var: str) -> str:
    key = os.environ.get(env_var)
    if not key:
        raise LLMError(f"{env_var} is not set (put it in .env or the environment).")
    return key


# --------------------------------------------------------------------------
# providers
# --------------------------------------------------------------------------


def _call_gemini(prompt: str, model: str, system: str | None) -> str:
    api_key = _require_key("GEMINI_API_KEY")
    payload: dict = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": TEMPERATURE,
            "maxOutputTokens": MAX_OUTPUT_TOKENS,
        },
    }
    if system:
        payload["systemInstruction"] = {"parts": [{"text": system}]}

    body = _post_with_backoff(
        url=GEMINI_URL.format(model=model),
        headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
        payload=payload,
        provider="gemini",
    )

    candidates = body.get("candidates") or []
    if not candidates:
        raise LLMError(f"Gemini returned no candidates: {json.dumps(body)[:300]}")
    candidate = candidates[0]
    parts = candidate.get("content", {}).get("parts") or []
    text = "".join(part.get("text", "") for part in parts)
    if not text:
        # Empty text means truncation or a safety block. Returning "" here would
        # poison the cache with a permanent blank, so fail loudly instead.
        raise LLMError(
            f"Gemini returned empty text (finishReason="
            f"{candidate.get('finishReason')!r}); nothing cached."
        )
    return text


def _call_groq(prompt: str, model: str, system: str | None) -> str:
    api_key = _require_key("GROQ_API_KEY")
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    body = _post_with_backoff(
        url=GROQ_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        payload={
            "model": model,
            "messages": messages,
            "temperature": TEMPERATURE,
            "max_tokens": MAX_OUTPUT_TOKENS,
        },
        provider="groq",
    )

    choices = body.get("choices") or []
    if not choices:
        raise LLMError(f"Groq returned no choices: {json.dumps(body)[:300]}")
    text = choices[0].get("message", {}).get("content") or ""
    if not text:
        raise LLMError(
            f"Groq returned empty content (finish_reason="
            f"{choices[0].get('finish_reason')!r}); nothing cached."
        )
    return text


# --------------------------------------------------------------------------
# transport: pacing + backoff, in one place
# --------------------------------------------------------------------------


def _post_with_backoff(url: str, headers: dict, payload: dict, provider: str) -> dict:
    """POST with exponential backoff on 429/5xx. Returns the parsed JSON body."""
    for attempt in range(MAX_RETRIES):
        # Both providers are paced, each against its own clock. A retry goes
        # through here too, so a backoff sleep and the inter-call gap compose
        # rather than racing each other.
        _wait_for_slot(provider)
        _last_call[provider] = time.monotonic()

        response = requests.post(url, headers=headers, json=payload, timeout=TIMEOUT_S)

        if response.status_code == 200:
            return response.json()

        if response.status_code in (401, 403):
            # A bad or unauthorized key will not fix itself; retrying burns time.
            raise LLMError(
                f"{provider} auth failed ({response.status_code}): {response.text[:200]}"
            )

        if response.status_code == 429 or response.status_code >= 500:
            if attempt == MAX_RETRIES - 1:
                break
            time.sleep(_backoff_seconds(attempt, response))
            continue

        raise LLMError(f"{provider} returned {response.status_code}: {response.text[:300]}")

    raise LLMError(
        f"{provider} still failing after {MAX_RETRIES} attempts "
        f"(last status {response.status_code}): {response.text[:200]}"
    )


def _backoff_seconds(attempt: int, response) -> float:
    """4, 8, 16, 32, 64 seconds — or the server's Retry-After if it sent one."""
    retry_after = response.headers.get("Retry-After") if response.headers else None
    if retry_after:
        try:
            return float(retry_after)
        except ValueError:
            pass  # Retry-After can be an HTTP date; fall back to our own schedule.
    return min(BACKOFF_BASE_S * (2**attempt), BACKOFF_CAP_S)


def _wait_for_slot(provider: str) -> None:
    """Sleep so consecutive live requests to `provider` are >= its min gap apart."""
    last = _last_call.get(provider)
    if last is None:
        return

    gap = MIN_GAP_S[provider]
    elapsed = time.monotonic() - last
    if elapsed < gap:
        time.sleep(gap - elapsed)
