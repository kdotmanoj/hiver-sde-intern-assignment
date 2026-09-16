"""Stage 8: the LLM-as-judge. Score generated replies on three axes, 1-5.

src/evaluate.py scores the intent and the escalation decision, both of which have
hand labels to score against. The reply does not: there is no golden "correct"
reply, and the one Spotify actually sent is one acceptable answer among many, not
the answer. So the reply is scored by a model against a rubric, on three axes:

  groundedness  is it supported by the retrieved historical replies, or invented?
  correctness   is anything false, or promised that Spotify cannot deliver?
  tone          does it sound like SpotifyCares?

Three things the report has to state plainly about this stage:

1. THE JUDGE IS A DIFFERENT MODEL FAMILY FROM THE GENERATOR. Gemini writes the
   replies (src/agent.py's GENERATOR_MODEL); Groq's openai/gpt-oss-120b scores
   them. A model asked to grade its own output rates it higher than it rates
   identical output attributed to someone else, so scoring Gemini with Gemini
   would measure self-preference and reply quality mixed together, with no way to
   separate them afterwards. The judge also never learns which system wrote the
   draft -- the prompt is identical in shape for all three, and names none of
   them, so there is no system identity to prefer.

2. ALL THREE SYSTEMS ARE SCORED AGAINST THE SAME RETRIEVED SET: the k=3
   neighbours the agent saw, read from agent_golden.jsonl and reused verbatim.
   The systems do not retrieve alike -- the trivial baseline retrieves nothing at
   all and k-NN uses only its top-1 -- so letting each bring its own context would
   make "grounded in the retrieved replies" a different question per system, and
   the resulting column would not be a comparison.

3. THAT SHARED CONTEXT IS THE FAIREST OPTION AVAILABLE AND IT STILL FAVOURS THE
   AGENT ON GROUNDEDNESS. The agent's reply was generated FROM exactly those three
   neighbours; the baselines' replies were not. "Is this supported by this
   context?" is therefore a question the agent was built to answer well and the
   baselines were never asked. The groundedness gap between the agent and the
   baselines is inflated by that, by an amount this design cannot measure. It is a
   limitation for the report to state, not a defect to fix: the alternative scores
   each system on a different question, which is strictly worse. Correctness and
   tone do not depend on the retrieved set in the same way and are not affected.

Budget. Groq's free tier is 30 RPM / 8K TPM / 200K TPD, and TPD binds. A judge
call measures ~450 tokens, so the agent's 150 replies cost ~78K -- comfortable.
All three systems would be ~235K, over the daily cap, which is why a run scores
ONE system and the preflight refuses to start anything that would not fit.

Every call goes through src/llm.py and is cached, so a second run of this stage
is free and needs no key.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

from src import llm
from src.agent import GOLDEN_LABELS, INTERIM
from src.evaluate import PRODUCED_BY, SYSTEMS, read_jsonl
from src.ingest import _display_path

# Groq, per CLAUDE.md's model stack. A DIFFERENT FAMILY from src/agent.py's
# GENERATOR_MODEL -- see point 1 of the module docstring. If this ever has to
# become a Gemini model, the self-preference claim in the report dies with it and
# that is a conversation, not a config change.
JUDGE_MODEL = "openai/gpt-oss-120b"

AXES = ("groundedness", "correctness", "tone")

SCORE_MIN = 1
SCORE_MAX = 5

# Where the shared retrieved context comes from. The agent's file specifically:
# it is the only one whose `retrieved` is the k=3 set the generator actually saw.
CONTEXT_SOURCE = "agent"

JUDGE_SCORES = INTERIM / "judge_scores.jsonl"


# --------------------------------------------------------------------------
# the rubric
# --------------------------------------------------------------------------
#
# One entry per axis: the question, then what a 1 looks like and what a 5 looks
# like. Anchored at both ends on purpose -- an unanchored "rate 1-5" collapses
# toward 4 for everything and stops discriminating between systems, which is the
# only thing this stage exists to do. The middle is left to the judge; inventing
# prose for 2, 3 and 4 would triple the prompt for scale points nobody cites.

RUBRIC = {
    "groundedness": (
        "supported by the past replies above?",
        "invents facts or policies the past replies do not support",
        "every claim is backed by what Spotify said above",
    ),
    "correctness": (
        "anything false, or promised that Spotify cannot deliver?",
        "says something false, or commits to a fix, refund or date support cannot give",
        "nothing false, nothing promised beyond what support can do",
    ),
    "tone": (
        "does it sound like Spotify's support account?",
        "robotic, cold, corporate or overfamiliar",
        "warm, brief, plain, the voice of the past replies",
    ),
}


def judge_system_prompt() -> str:
    """The rubric. Fixed for the whole project -- it is part of the cache key.

    Written tight rather than thorough. CLAUDE.md budgets ~500 tokens per judge
    call TOTAL, and the case itself is ~250 of them, so every line spent
    elaborating the scale is a line not available for the replies being scored.
    """
    lines = [
        "You score draft replies for Spotify's support account on Twitter.",
        "",
        "You see real past conversations, a new customer message, and a draft reply to it.",
        f"Score the DRAFT on three axes, each a whole number {SCORE_MIN}-{SCORE_MAX}.",
        "",
    ]

    for axis in AXES:
        question, low, high = RUBRIC[axis]
        lines += [
            f"{axis} -- {question}",
            f"  {SCORE_MIN} = {low}",
            f"  {SCORE_MAX} = {high}",
        ]

    lines += [
        "",
        "Judge the draft only. Being shorter or longer than the past replies is not a fault.",
        "",
        "Reply with exactly this JSON and nothing else:",
        '{"groundedness": n, "correctness": n, "tone": n}',
    ]
    return "\n".join(lines)


def judge_prompt(text: str, reply: str, retrieved: list[dict]) -> str:
    """The case: precedent, the new message, the draft to score.

    Block layout mirrors agent._reply_prompt so the judge reads the precedent in
    the same shape the generator was given it.

    Similarity scores are deliberately NOT included, though agent._reply_prompt
    shows them. A neighbour labelled "similarity 0.91" is an authority cue: it
    invites the judge to treat the closest precedent as the standard answer and
    mark down anything that diverges from it. The judge's job is to read the text.
    """
    blocks = [
        "\n".join(
            [
                f"--- past conversation {n} ---",
                f"Customer: {item['opening_text']}",
                f"Spotify replied: {item['first_reply_text']}",
            ]
        )
        for n, item in enumerate(retrieved, start=1)
    ]
    precedent = "\n\n".join(blocks) if blocks else "(no similar past conversation was found)"

    return "\n".join(
        [
            precedent,
            "",
            "--- new message ---",
            f"Customer: {text}",
            "",
            "--- draft reply to score ---",
            reply,
            "",
            "Scores:",
        ]
    )


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class JudgeScore:
    """Three scores, or a parse failure with the raw string kept for inspection.

    On failure all three scores are None. Never 0, never the midpoint, never the
    last good value: a malformed response is an absence of evidence, and any
    number put here would be averaged into the report as though the judge had
    said it. Same contract agent.parse_intent sets for intents -- parse failures
    are their own reported number, not absorbed into the scores.
    """

    groundedness: int | None
    correctness: int | None
    tone: int | None
    parse_ok: bool
    raw: str

    @classmethod
    def failed(cls, raw: str) -> JudgeScore:
        return cls(groundedness=None, correctness=None, tone=None, parse_ok=False, raw=raw)

    def as_dict(self) -> dict:
        return {axis: getattr(self, axis) for axis in AXES}


def _strip_fences(text: str) -> str:
    """Drop a ```json ... ``` wrapper if there is one. Models add them unasked."""
    text = text.strip()
    if not text.startswith("```"):
        return text

    # Everything after the first newline, minus a trailing fence. The opening
    # line is ``` or ```json; neither is part of the payload.
    without_open = text.split("\n", 1)[1] if "\n" in text else ""
    end = without_open.rfind("```")
    return (without_open[:end] if end != -1 else without_open).strip()


def _first_json_object(text: str) -> str | None:
    """The first balanced {...} span in `text`, or None.

    For the common "Here are the scores: {...}" case.

    Scanning starts at position 0, not at the first brace, and tracks whether it
    is inside a string the whole way. That matters: a preamble like
    `Note: "{not json}" follows.` has its first `{` INSIDE a quoted string, and a
    scan that began there would lock onto a fragment that is not JSON at all.
    """
    depth = 0
    start = -1
    in_string = False
    escaped = False

    for position, character in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue

        if character == '"':
            in_string = True
        elif character == "{":
            if depth == 0:
                start = position
            depth += 1
        elif character == "}" and depth > 0:
            depth -= 1
            if depth == 0:
                return text[start : position + 1]

    # Ran off the end still open: a truncated response. No object to return.
    return None


def _as_score(value) -> int | None:
    """An int in [SCORE_MIN, SCORE_MAX], or None if `value` is not one.

    Accepts 4, 4.0 and "4" -- all three are a model writing the number four in
    JSON. Rejects True (bool is an int subclass in Python, and `true` is not a
    score), 4.5, "four", "high", and anything out of range. Out of range is a
    rejection and never a clamp: a judge answering 7 has not answered 5.
    """
    if isinstance(value, bool):
        return None

    if isinstance(value, int):
        number = value
    elif isinstance(value, float):
        if not value.is_integer():
            return None
        number = int(value)
    elif isinstance(value, str):
        stripped = value.strip()
        if not (stripped.isdigit() or (stripped.startswith("-") and stripped[1:].isdigit())):
            return None
        number = int(stripped)
    else:
        return None

    return number if SCORE_MIN <= number <= SCORE_MAX else None


def parse_scores(raw: str) -> JudgeScore:
    """Read three scores out of a judge response. Strict, and loud when it fails.

    Tolerant about packaging -- code fences, a sentence of preamble, key order,
    4 vs 4.0 vs "4" -- because those are the model being chatty, not the model
    disagreeing. Intolerant about content: a missing axis, an extra axis, a
    non-numeric score or one outside 1-5 is a parse failure with no scores at all.
    """
    candidate = _strip_fences(raw)

    try:
        parsed = json.loads(candidate)
    except (json.JSONDecodeError, TypeError):
        span = _first_json_object(candidate)
        if span is None:
            return JudgeScore.failed(raw)
        try:
            parsed = json.loads(span)
        except json.JSONDecodeError:
            return JudgeScore.failed(raw)

    if not isinstance(parsed, dict):
        return JudgeScore.failed(raw)

    # Exact key set. A missing axis is obviously a failure; an EXTRA key is one
    # too, because it means the model answered a different question than the one
    # asked and the three keys that happen to match may not mean what they say.
    if set(parsed) != set(AXES):
        return JudgeScore.failed(raw)

    scores = {axis: _as_score(parsed[axis]) for axis in AXES}
    if any(score is None for score in scores.values()):
        return JudgeScore.failed(raw)

    return JudgeScore(**scores, parse_ok=True, raw=raw)


# --------------------------------------------------------------------------
# one call
# --------------------------------------------------------------------------


def judge_one(
    text: str,
    reply: str,
    retrieved: list[dict],
    model: str = JUDGE_MODEL,
    repeat: int = 0,
) -> JudgeScore:
    """Score one draft reply.

    `repeat` is the repeat index, passed to llm.complete as `variant`. It changes
    the cache slot and NOTHING the provider sees, so repeat=1 asks the identical
    question again instead of reading back repeat=0's answer. That is what makes
    --repeat a measurement of the judge rather than of the cache.
    """
    raw = llm.complete(
        prompt=judge_prompt(text, reply, retrieved),
        model=model,
        system=judge_system_prompt(),
        variant=repeat,
    )
    return parse_scores(raw)


# --------------------------------------------------------------------------
# the cases: one system's replies, against the agent's retrieved context
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class JudgeCase:
    """One thing to score: a message, a draft, and the shared precedent."""

    conversation_id: int
    text: str
    reply: str
    retrieved: list[dict]


def load_retrieved_context(interim: Path = INTERIM) -> dict[int, list[dict]]:
    """conversation_id -> the k=3 neighbours the AGENT retrieved.

    Read from the agent's file for every system, which is the whole point: see
    points 2 and 3 of the module docstring for what that buys and what it costs.
    """
    path = interim / f"{CONTEXT_SOURCE}_golden.jsonl"
    if not path.exists():
        sys.exit(
            f"missing {_display_path(path)} -- run `{PRODUCED_BY[CONTEXT_SOURCE]}` first.\n"
            f"Every system is judged against the retrieved context from that file, so it is\n"
            f"required even when scoring a baseline."
        )

    rows = read_jsonl(path)
    context = {int(row["conversation_id"]): row["retrieved"] for row in rows}
    assert len(context) == len(rows), f"duplicate conversation_id in {_display_path(path)}"
    return context


def build_cases(system: str, interim: Path = INTERIM, limit: int | None = None) -> list[JudgeCase]:
    """One system's replies, paired with the agent's retrieved context.

    Sorted by conversation_id, so --limit takes a deterministic, re-runnable
    subset rather than whatever order the file happens to be in.
    """
    path = interim / f"{system}_golden.jsonl"
    if not path.exists():
        sys.exit(f"missing {_display_path(path)} -- run `{PRODUCED_BY[system]}` first.")

    rows = read_jsonl(path)
    context = load_retrieved_context(interim)

    missing_context = {row["conversation_id"] for row in rows} - set(context)
    assert not missing_context, (
        f"{len(missing_context)} of {system}'s conversations have no retrieved context in "
        f"{CONTEXT_SOURCE}_golden.jsonl"
    )

    # A row with no reply has nothing to score. Only reachable when the agent ran
    # without --force-reply, or a baseline hit an empty corpus. Counted and
    # reported rather than dropped silently.
    with_reply = [row for row in rows if row.get("reply")]
    n_no_reply = len(rows) - len(with_reply)

    cases = [
        JudgeCase(
            conversation_id=int(row["conversation_id"]),
            text=row["text"],
            reply=row["reply"],
            retrieved=context[row["conversation_id"]],
        )
        for row in sorted(with_reply, key=lambda row: row["conversation_id"])
    ]

    print(
        f"judge[{system}]: rows {len(rows):,} -> with reply {len(cases):,} "
        f"(lost {n_no_reply:,})"
        + (f" -> limited to {min(limit, len(cases)):,}" if limit is not None else "")
    )
    return cases[:limit] if limit is not None else cases


# --------------------------------------------------------------------------
# preflight: count the spend before making any of it
# --------------------------------------------------------------------------
#
# Same guard as src/agent.py's, against a different wall. Gemini's is a daily
# REQUEST budget; Groq's binding limit is a daily TOKEN budget, so this counts
# both and refuses on either. A run that discovers the cap at call 100 of 150 has
# spent the tokens and produced a partial file.

# 150 golden replies plus slack for retries. A run that wants materially more
# than one system's worth is a bug or a much bigger batch than intended.
MAX_LIVE_CALLS = 165

# CLAUDE.md: warn if a planned judge run would exceed 120K tokens.
TOKEN_WARN = 120_000

# Groq's tokens-per-day ceiling. Nothing is sent past this.
TOKEN_ABORT = 200_000

# Conservative chars-per-token for English tweets, which really run ~4. Low on
# purpose: it makes the estimate read HIGH, and over-estimating the spend fails
# safe against a daily cap while under-estimating walks into it.
CHARS_PER_TOKEN = 3.5

# The judge answers with one short JSON object. Small, but it counts against TPD
# the same as the prompt does, so it is in the estimate rather than ignored.
OUTPUT_TOKENS = 40

# CLAUDE.md budgets ~500 real tokens per judge call. estimate_tokens() reads high
# by design (CHARS_PER_TOKEN is 3.5 where English runs closer to 4), so ~500 real
# tokens is ~570 estimated ones. Comparing the estimate against a flat 500 would
# fire this warning on every run and teach me to ignore it.
PROMPT_TOKEN_BUDGET = 570


class QuotaError(RuntimeError):
    """A planned run needs more live calls or tokens than the budget allows."""


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def preflight(cases: list[JudgeCase], repeat: int = 1, model: str = JUDGE_MODEL) -> dict:
    """Count the uncached calls and tokens a run would need. Makes no calls itself.

    Exact, unlike agent.preflight's upper bound: the judge prompt is built from
    files already on disk, so every prompt -- and therefore every cache key -- is
    known before anything is sent.
    """
    if repeat < 1:
        raise ValueError(f"repeat must be at least 1, got {repeat}")

    system = judge_system_prompt()
    system_tokens = estimate_tokens(system)

    n_uncached = 0
    est_tokens = 0
    per_call = []

    for case in cases:
        prompt = judge_prompt(case.text, case.reply, case.retrieved)
        call_tokens = system_tokens + estimate_tokens(prompt) + OUTPUT_TOKENS
        per_call.append(call_tokens)

        for index in range(repeat):
            if llm.cached_response(prompt, model, system, variant=index) is None:
                n_uncached += 1
                est_tokens += call_tokens

    return {
        "n_cases": len(cases),
        "repeat": repeat,
        "n_calls": len(cases) * repeat,
        "n_uncached": n_uncached,
        "est_tokens": est_tokens,
        # Mean AND max: the max is one long outlier, the mean is what the
        # per-prompt budget conversation is actually about.
        "mean_prompt_tokens": round(sum(per_call) / len(per_call)) if per_call else 0,
        "max_prompt_tokens": max(per_call, default=0),
        "system_tokens": system_tokens,
        "exact": True,
    }


def _report_preflight(plan: dict, max_live_calls: int = MAX_LIVE_CALLS) -> None:
    """Print the budget, warn at TOKEN_WARN, then raise if it does not fit."""
    minutes = plan["n_uncached"] * llm.GROQ_MIN_GAP_S / 60.0

    print(
        f"  preflight:   {plan['n_uncached']:,} live calls needed of "
        f"{plan['n_calls']:,} ({plan['n_cases']:,} replies x {plan['repeat']} repeat"
        f"{'s' if plan['repeat'] != 1 else ''})"
    )
    print(
        f"    ~{plan['est_tokens']:,} tokens, per call mean "
        f"~{plan['mean_prompt_tokens']:,} / max ~{plan['max_prompt_tokens']:,} "
        f"(rubric {plan['system_tokens']:,})"
    )
    print(
        f"    budget {plan['n_uncached']:,}/{max_live_calls:,} calls, "
        f"{plan['est_tokens']:,}/{TOKEN_ABORT:,} tokens/day, "
        f"~{minutes:.0f} min at {llm.GROQ_MIN_GAP_S:.0f}s between calls"
    )

    if plan["max_prompt_tokens"] > PROMPT_TOKEN_BUDGET:
        print(
            f"    NOTE: largest call is ~{plan['max_prompt_tokens']:,} estimated tokens, over "
            f"the {PROMPT_TOKEN_BUDGET:,} that corresponds to CLAUDE.md's ~500-token budget."
        )

    if TOKEN_WARN < plan["est_tokens"] <= TOKEN_ABORT:
        print(
            f"    WARNING: ~{plan['est_tokens']:,} tokens is over the {TOKEN_WARN:,} "
            f"review threshold, and leaves "
            f"{TOKEN_ABORT - plan['est_tokens']:,} of today's {TOKEN_ABORT:,} for anything else."
        )

    if plan["n_uncached"] > max_live_calls:
        raise QuotaError(
            f"This run needs {plan['n_uncached']:,} live calls but MAX_LIVE_CALLS is "
            f"{max_live_calls:,}. Nothing has been sent.\n"
            f"  Options: raise --max-live-calls, use --limit to score a subset, or run "
            f"one system at a time."
        )

    if plan["est_tokens"] > TOKEN_ABORT:
        raise QuotaError(
            f"This run would spend ~{plan['est_tokens']:,} tokens but Groq's daily cap on "
            f"{JUDGE_MODEL} is {TOKEN_ABORT:,}. Nothing has been sent.\n"
            f"  The cap is a DAILY budget: going over does not queue, it fails, and the "
            f"tokens already spent are still spent.\n"
            f"  Judge one system per day, or use --limit."
        )


# --------------------------------------------------------------------------
# the run
# --------------------------------------------------------------------------


def run_system(
    system: str,
    cases: list[JudgeCase],
    repeat: int = 1,
    model: str = JUDGE_MODEL,
) -> list[dict]:
    """Judge every case `repeat` times. One row per (system, conversation, repeat)."""
    rows = []
    for position, case in enumerate(cases, start=1):
        for index in range(repeat):
            score = judge_one(case.text, case.reply, case.retrieved, model=model, repeat=index)
            rows.append(
                {
                    "system": system,
                    "conversation_id": case.conversation_id,
                    "repeat": index,
                    **score.as_dict(),
                    "parse_ok": score.parse_ok,
                    "raw": score.raw,
                    "n_retrieved": len(case.retrieved),
                    "reply": case.reply,
                }
            )

        last = rows[-1]
        shown = (
            "  ".join(f"{axis[:4]} {last[axis]}" for axis in AXES)
            if last["parse_ok"]
            else "PARSE FAILED"
        )
        print(f"  [{position:>3}/{len(cases)}] {case.conversation_id:<12} {shown}")

    assert len(rows) == len(cases) * repeat, f"{len(rows)} rows for {len(cases)}x{repeat}"
    return rows


def summarize(system: str, rows: list[dict]) -> dict:
    """The stage row-count line plus the mean per axis over the rows that parsed."""
    parsed = [row for row in rows if row["parse_ok"]]
    n_failed = len(rows) - len(parsed)

    print("")
    print(
        f"judge[{system}]: judged {len(rows):,} -> parsed {len(parsed):,} "
        f"(lost {n_failed:,} parse failures)"
    )

    means = {}
    for axis in AXES:
        values = [row[axis] for row in parsed]
        means[axis] = sum(values) / len(values) if values else None
        shown = f"{means[axis]:.2f}" if values else "n/a (nothing parsed)"
        print(f"  {axis:<14} mean {shown}  over {len(values):,} scores")

    if n_failed:
        print("")
        print(f"  {n_failed:,} response(s) did not parse. Scores are None, not zero, and are")
        print("  excluded from every mean above. First failing raw response:")
        first = next(row for row in rows if not row["parse_ok"])
        print(f"    [{first['conversation_id']}] {first['raw'][:160]!r}")

    return {"n_rows": len(rows), "n_parsed": len(parsed), "n_failed": n_failed, "means": means}


# --------------------------------------------------------------------------
# --repeat: does the judge agree with itself?
# --------------------------------------------------------------------------
#
# A mean over 150 scores is only worth reporting if the same input scores the
# same twice. TEMPERATURE is 0.0 project-wide, but temperature 0 is not a
# determinism guarantee on a hosted mixture-of-experts model -- batching and
# expert routing can move a token. So this measures it instead of assuming it.


def consistency(rows: list[dict], repeat: int) -> dict | None:
    """Per-axis spread across repeats of the same reply. None if repeat < 2.

    Spread is max - min over a conversation's repeats: with N=2 that is just the
    disagreement, and it stays readable for larger N. A conversation is only
    counted if EVERY repeat parsed -- a parse failure is a missing score, not a
    disagreement, and folding it in either way would misstate the thing measured.
    """
    if repeat < 2:
        return None

    by_conversation: dict[int, list[dict]] = {}
    for row in rows:
        by_conversation.setdefault(row["conversation_id"], []).append(row)

    complete = {
        conversation_id: group
        for conversation_id, group in by_conversation.items()
        if len(group) == repeat and all(row["parse_ok"] for row in group)
    }
    n_excluded = len(by_conversation) - len(complete)

    print("")
    print(f"judge consistency: {repeat} repeats of identical input, temperature 0")
    print(f"{'axis':<16}{'mean spread':>13}{'exact agree':>13}{'worst':>8}  worst id")

    report: dict = {"repeat": repeat, "n_compared": len(complete), "n_excluded": n_excluded}

    for axis in AXES:
        spreads = {
            conversation_id: max(row[axis] for row in group) - min(row[axis] for row in group)
            for conversation_id, group in complete.items()
        }

        if not spreads:
            print(f"{axis:<16}{'n/a':>13}{'n/a':>13}{'n/a':>8}  (nothing comparable)")
            report[axis] = None
            continue

        values = list(spreads.values())
        mean_spread = sum(values) / len(values)
        n_agreed = sum(1 for value in values if value == 0)
        worst_id = max(spreads, key=lambda key: spreads[key])

        report[axis] = {
            "mean_spread": mean_spread,
            "n_agreed": n_agreed,
            "share_agreed": n_agreed / len(values),
            "worst_spread": spreads[worst_id],
            "worst_conversation_id": worst_id,
        }
        print(
            f"{axis:<16}{mean_spread:>13.3f}"
            f"{f'{n_agreed}/{len(values)}':>13}{spreads[worst_id]:>8}  {worst_id}"
        )

    print("")
    print(f"  {len(complete):,} conversations compared; {n_excluded:,} excluded (a repeat")
    print("  failed to parse, so there is no second score to compare against).")
    print("  Mean spread 0.000 on every axis means the judge is reproducible and a single")
    print("  run's mean can be reported as-is. Anything above 0 belongs in the report.")
    return report


# --------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------


def merge_rows(existing: list[dict], fresh: list[dict]) -> list[dict]:
    """Fresh rows replace matching ones; everything else is kept.

    Keyed on (system, conversation_id, repeat) rather than on system alone, so
    `--limit 20 --repeat 2` refreshes forty rows and leaves the other 130 of a
    full run intact. Keying on system would let the cheap consistency probe
    silently delete the expensive run it was checking.
    """

    def key(row: dict) -> tuple:
        return (row["system"], row["conversation_id"], row.get("repeat", 0))

    fresh_keys = {key(row) for row in fresh}
    kept = [row for row in existing if key(row) not in fresh_keys]

    merged = kept + fresh
    assert len({key(row) for row in merged}) == len(merged), "duplicate key after merge"
    return merged


def write_scores(path: Path, rows: list[dict], system: str) -> None:
    existing = read_jsonl(path) if path.exists() else []
    merged = merge_rows(existing, rows)

    # Stable order so the file diffs cleanly between runs.
    merged.sort(key=lambda row: (row["system"], row["conversation_id"], row.get("repeat", 0)))

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in merged:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    n_replaced = len(existing) + len(rows) - len(merged)
    others = sorted({row["system"] for row in merged} - {system})
    print("")
    print(
        f"  wrote {_display_path(path)}: {len(merged):,} rows "
        f"({len(rows):,} written, {n_replaced:,} replaced, "
        f"{len(merged) - len(rows):,} kept from earlier runs"
        + (f": {', '.join(others)}" if others else "")
        + ")"
    )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--system",
        choices=SYSTEMS,
        default="agent",
        help="which system's replies to judge (default agent -- the baselines cost a "
        "second day's token budget)",
    )
    parser.add_argument("--interim", type=Path, default=INTERIM)
    parser.add_argument("--golden-labels", type=Path, default=GOLDEN_LABELS)
    parser.add_argument("--out", type=Path, default=JUDGE_SCORES)
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="judge each reply this many times and report the spread (default 1)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="judge only the first N replies by conversation_id (for a cheap --repeat probe)",
    )
    parser.add_argument(
        "--max-live-calls",
        type=int,
        default=MAX_LIVE_CALLS,
        help=f"abort if the run would need more live calls than this (default {MAX_LIVE_CALLS})",
    )
    args = parser.parse_args()

    if args.repeat < 1:
        parser.error(f"--repeat must be at least 1, got {args.repeat}")

    cases = build_cases(args.system, args.interim, limit=args.limit)

    plan = preflight(cases, repeat=args.repeat)
    try:
        _report_preflight(plan, args.max_live_calls)
    except QuotaError as error:
        # A budget refusal is an expected outcome, not a crash. A traceback here
        # would bury the one thing worth reading.
        print(f"\nABORTED: {error}", file=sys.stderr)
        raise SystemExit(1)

    rows = run_system(args.system, cases, repeat=args.repeat)

    summarize(args.system, rows)
    consistency(rows, args.repeat)
    write_scores(args.out, rows, args.system)


if __name__ == "__main__":
    main()
