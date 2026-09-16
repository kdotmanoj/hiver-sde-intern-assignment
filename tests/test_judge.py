"""Tests for the non-LLM parts of the judge.

What is tested: the response parser, the prompt builder, the shared-context
rule, the budget preflight, the repeat plumbing, and the consistency table.

What is not: whether the judge scores well. That is what the judge IS, and
asserting on it here would be asserting on a cached string.

No test makes a network call and no test loads the embedding model.
"""

import json

import pytest

from src import judge, llm


@pytest.fixture(autouse=True)
def no_llm(monkeypatch):
    """Any test that reaches an LLM provider fails, rather than quietly mocking one."""

    def _boom(*args, **kwargs):
        raise AssertionError("an LLM call was made")

    monkeypatch.setattr(llm.requests, "post", _boom)


def retrieved(n=3):
    """n neighbours in the shape agent.Retrieved serializes to."""
    return [
        {
            "conversation_id": 1000 + index,
            "similarity": 0.9 - index / 100,
            "opening_text": f"past customer {index}",
            "first_reply_text": f"past spotify reply {index}",
        }
        for index in range(n)
    ]


# --------------------------------------------------------------------------
# the parser: tolerant about packaging, strict about content
# --------------------------------------------------------------------------


ACCEPTED = [
    ('{"groundedness": 4, "correctness": 5, "tone": 3}', (4, 5, 3)),
    # Code fences, with and without a language tag. Models add them unasked.
    ('```json\n{"groundedness": 1, "correctness": 2, "tone": 3}\n```', (1, 2, 3)),
    ('```\n{"groundedness": 5, "correctness": 5, "tone": 5}\n```', (5, 5, 5)),
    # A sentence of preamble, and a trailing one.
    ('Here are the scores: {"groundedness": 2, "correctness": 3, "tone": 4}', (2, 3, 4)),
    ('{"groundedness": 2, "correctness": 3, "tone": 4}\nHope that helps!', (2, 3, 4)),
    # Key order is not meaningful in JSON.
    ('{"tone": 1, "groundedness": 2, "correctness": 3}', (2, 3, 1)),
    # Four, written three ways. All three are the model saying four.
    ('{"groundedness": 4.0, "correctness": 4, "tone": "4"}', (4, 4, 4)),
    # Whitespace and newlines inside the object.
    ('{\n  "groundedness" : 3,\n  "correctness" : 3,\n  "tone" : 3\n}', (3, 3, 3)),
]


@pytest.mark.parametrize("raw,expected", ACCEPTED)
def test_parser_accepts_well_formed_responses(raw, expected):
    score = judge.parse_scores(raw)

    assert score.parse_ok is True
    assert (score.groundedness, score.correctness, score.tone) == expected
    assert score.raw == raw, "the raw response is kept verbatim even on success"


REJECTED = [
    # Out of range, both directions. A judge answering 0 or 6 has not answered.
    '{"groundedness": 0, "correctness": 3, "tone": 3}',
    '{"groundedness": 6, "correctness": 3, "tone": 3}',
    '{"groundedness": -1, "correctness": 3, "tone": 3}',
    # Not a number.
    '{"groundedness": "high", "correctness": 3, "tone": 3}',
    '{"groundedness": null, "correctness": 3, "tone": 3}',
    '{"groundedness": true, "correctness": 3, "tone": 3}',
    '{"groundedness": 4.5, "correctness": 3, "tone": 3}',
    # A missing axis.
    '{"groundedness": 4, "correctness": 3}',
    # An extra axis: the model answered a different question than the one asked.
    '{"groundedness": 4, "correctness": 3, "tone": 3, "overall": 4}',
    # Nothing parseable at all.
    "",
    "   ",
    "I would rate this reply quite highly overall.",
    "4, 5, 3",
    # Truncated mid-object -- the response hit a token limit.
    '{"groundedness": 4',
    # Valid JSON, wrong shape.
    "[4, 5, 3]",
    '"4"',
    "null",
]


@pytest.mark.parametrize("raw", REJECTED)
def test_parser_rejects_rather_than_coercing(raw):
    """The contract: no silent coercion. A failure has no scores at all.

    Not clamped to the range, not defaulted to the midpoint, not zero. Any
    number here would be averaged into the report as though the judge had said
    it, which is the specific way a broken judge run looks like a working one.
    """
    score = judge.parse_scores(raw)

    assert score.parse_ok is False
    assert score.groundedness is None
    assert score.correctness is None
    assert score.tone is None
    assert score.raw == raw, "the raw response is kept so the failure can be read"


def test_parser_handles_braces_inside_string_values():
    """Brace counting is string-aware, so a brace in a value cannot unbalance it."""
    raw = 'Note: "{not json}" precedes. {"groundedness": 3, "correctness": 3, "tone": 3}'
    score = judge.parse_scores(raw)

    assert score.parse_ok is True
    assert score.groundedness == 3


def test_failed_scores_serialize_as_null_not_zero():
    """The jsonl row must carry None, which is `null` on disk and not a score."""
    row = judge.JudgeScore.failed("garbage").as_dict()

    assert row == {"groundedness": None, "correctness": None, "tone": None}
    assert json.loads(json.dumps(row))["tone"] is None


# --------------------------------------------------------------------------
# the prompt builder
# --------------------------------------------------------------------------


def test_prompt_contains_message_reply_and_every_retrieved_reply():
    prompt = judge.judge_prompt("my music stopped", "Hey! Try a reinstall.", retrieved(3))

    assert "my music stopped" in prompt
    assert "Hey! Try a reinstall." in prompt
    for item in retrieved(3):
        assert item["opening_text"] in prompt
        assert item["first_reply_text"] in prompt


def test_prompt_omits_similarity_scores():
    """A neighbour labelled "similarity 0.891" is an authority cue, not evidence."""
    prompt = judge.judge_prompt("q", "draft", retrieved(3))

    assert "similarity" not in prompt.lower()
    assert "0.9" not in prompt
    assert "0.89" not in prompt


def test_prompt_never_names_the_system_that_wrote_the_draft():
    """Scoring is blind: there is no system identity for the judge to prefer."""
    prompt = judge.judge_prompt("q", "draft", retrieved(3))
    lowered = prompt.lower()

    for name in ("agent", "trivial", "knn", "baseline", "gemini", "model"):
        assert name not in lowered, f"{name!r} leaks the draft's provenance into the prompt"


def test_prompt_is_byte_identical_across_calls():
    """The prompt is the cache key. An unstable one silently re-spends the budget."""
    first = judge.judge_prompt("q", "draft", retrieved(3))
    second = judge.judge_prompt("q", "draft", retrieved(3))

    assert first == second
    assert judge.judge_system_prompt() == judge.judge_system_prompt()


def test_prompt_handles_an_empty_retrieved_set():
    """Only reachable with an empty corpus, but it must not produce a bare prompt."""
    prompt = judge.judge_prompt("q", "draft", [])

    assert "no similar past conversation" in prompt
    assert "draft" in prompt


def worst_case_tokens():
    """A call with every field at Twitter's 280-char ceiling. The true ceiling."""
    longest = "x" * 280
    neighbours = [
        {
            "conversation_id": index,
            "similarity": 0.5,
            "opening_text": longest,
            "first_reply_text": longest,
        }
        for index in range(3)
    ]
    return (
        judge.estimate_tokens(judge.judge_system_prompt())
        + judge.estimate_tokens(judge.judge_prompt(longest, longest, neighbours))
        + judge.OUTPUT_TOKENS
    )


def test_a_full_run_fits_the_daily_cap_even_at_worst_case():
    """The binding constraint, asserted against the worst input that can exist.

    Not a taste bound -- this is the number that decides whether a 150-reply run
    completes or dies partway through having spent the day's tokens. Even if
    every golden reply and all three of its neighbours were maximum-length
    tweets, 150 calls fit inside Groq's TPD with room to spare.
    """
    assert 150 * worst_case_tokens() < judge.TOKEN_ABORT


def test_the_rubric_is_a_minority_of_the_prompt():
    """The budget goes on the thing being judged, not on instructions about judging.

    A rubric that outgrows the case it is scoring is the failure mode here: it
    costs tokens on every one of 150 calls and crowds out the replies. The
    per-call total is reported by the preflight and warned on there; this pins
    the split.
    """
    rubric = judge.estimate_tokens(judge.judge_system_prompt())

    assert rubric < worst_case_tokens() / 2


def test_preflight_notes_a_prompt_over_the_per_call_budget(monkeypatch, capsys):
    """CLAUDE.md budgets ~500 tokens a call. Going over must be said out loud."""
    monkeypatch.setattr(llm, "cached_response", lambda *args, **kwargs: None)
    plan = judge.preflight(cases(3), repeat=1)
    plan["max_prompt_tokens"] = judge.PROMPT_TOKEN_BUDGET + 1

    judge._report_preflight(plan, max_live_calls=1000)

    assert "NOTE" in capsys.readouterr().out


def test_rubric_anchors_both_ends_of_every_axis():
    """An unanchored scale collapses toward 4 and stops separating the systems."""
    prompt = judge.judge_system_prompt()

    for axis in judge.AXES:
        assert axis in prompt
        question, low, high = judge.RUBRIC[axis]
        assert low in prompt
        assert high in prompt


# --------------------------------------------------------------------------
# the shared retrieved context
# --------------------------------------------------------------------------


def write_system_file(interim, system, rows):
    interim.mkdir(parents=True, exist_ok=True)
    path = interim / f"{system}_golden.jsonl"
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return path


def golden_rows(n=3, reply="a draft reply", with_retrieved=True):
    return [
        {
            "conversation_id": index,
            "text": f"customer message {index}",
            "reply": reply,
            "retrieved": retrieved(3) if with_retrieved else [],
        }
        for index in range(n)
    ]


def test_every_system_is_judged_against_the_agents_retrieved_set(tmp_path, capsys):
    """The fairness rule, and the thing the module docstring claims.

    The trivial baseline's own file carries no retrieved context at all. Its
    cases must still come back with the agent's three neighbours, or the
    groundedness axis is asking a different question of each system.
    """
    interim = tmp_path / "interim"
    write_system_file(interim, "agent", golden_rows(3))
    write_system_file(
        interim, "trivial", [{**row, "retrieved": [], "reply": "canned"} for row in golden_rows(3)]
    )

    agent_cases = judge.build_cases("agent", interim)
    trivial_cases = judge.build_cases("trivial", interim)

    assert [case.retrieved for case in trivial_cases] == [case.retrieved for case in agent_cases]
    assert all(len(case.retrieved) == 3 for case in trivial_cases)
    # Same context, different draft -- which is the only thing that should differ.
    assert [case.reply for case in trivial_cases] == ["canned"] * 3


def test_cases_are_sorted_so_limit_is_deterministic(tmp_path):
    interim = tmp_path / "interim"
    rows = golden_rows(5)
    write_system_file(interim, "agent", list(reversed(rows)))

    cases = judge.build_cases("agent", interim, limit=3)

    assert [case.conversation_id for case in cases] == [0, 1, 2]
    assert judge.build_cases("agent", interim, limit=3) == cases


def test_rows_without_a_reply_are_dropped_and_counted(tmp_path, capsys):
    interim = tmp_path / "interim"
    rows = golden_rows(3)
    rows[1]["reply"] = None
    write_system_file(interim, "agent", rows)

    cases = judge.build_cases("agent", interim)

    assert len(cases) == 2
    assert "with reply 2" in capsys.readouterr().out


# --------------------------------------------------------------------------
# preflight
# --------------------------------------------------------------------------


def cases(n=3):
    return [
        judge.JudgeCase(
            conversation_id=index,
            text=f"message {index}",
            reply=f"draft {index}",
            retrieved=retrieved(3),
        )
        for index in range(n)
    ]


def test_preflight_counts_every_repeat(monkeypatch):
    monkeypatch.setattr(llm, "cached_response", lambda *args, **kwargs: None)

    single = judge.preflight(cases(10), repeat=1)
    double = judge.preflight(cases(10), repeat=2)

    assert single["n_uncached"] == 10
    assert double["n_uncached"] == 20
    assert double["est_tokens"] == 2 * single["est_tokens"]


def test_preflight_skips_cached_calls(monkeypatch):
    """Only variant 0 is cached, so a 2-repeat run still needs the repeats."""
    monkeypatch.setattr(
        llm, "cached_response", lambda *args, variant=0, **kwargs: "cached" if variant == 0 else None
    )

    plan = judge.preflight(cases(10), repeat=2)

    assert plan["n_uncached"] == 10
    assert plan["n_calls"] == 20


def test_preflight_makes_no_calls(monkeypatch):
    """The autouse no_llm fixture is the assertion: preflight must not reach post()."""
    plan = judge.preflight(cases(5), repeat=1)

    assert plan["exact"] is True
    assert plan["n_cases"] == 5


def test_preflight_aborts_over_the_call_budget(monkeypatch):
    monkeypatch.setattr(llm, "cached_response", lambda *args, **kwargs: None)
    plan = judge.preflight(cases(10), repeat=1)

    with pytest.raises(judge.QuotaError, match="MAX_LIVE_CALLS"):
        judge._report_preflight(plan, max_live_calls=5)


def test_preflight_aborts_over_the_daily_token_cap(monkeypatch):
    monkeypatch.setattr(llm, "cached_response", lambda *args, **kwargs: None)
    plan = judge.preflight(cases(10), repeat=1)
    plan["est_tokens"] = judge.TOKEN_ABORT + 1

    with pytest.raises(judge.QuotaError, match="daily cap"):
        judge._report_preflight(plan, max_live_calls=1000)


def test_preflight_warns_but_proceeds_past_the_review_threshold(monkeypatch, capsys):
    monkeypatch.setattr(llm, "cached_response", lambda *args, **kwargs: None)
    plan = judge.preflight(cases(10), repeat=1)
    plan["est_tokens"] = judge.TOKEN_WARN + 1

    judge._report_preflight(plan, max_live_calls=1000)

    assert "WARNING" in capsys.readouterr().out


def test_preflight_rejects_a_nonsense_repeat():
    with pytest.raises(ValueError, match="at least 1"):
        judge.preflight(cases(3), repeat=0)


# --------------------------------------------------------------------------
# the repeat plumbing
# --------------------------------------------------------------------------


def test_repeat_varies_the_cache_slot_and_nothing_else(monkeypatch):
    """The contract --repeat depends on: same prompt, different variant.

    If repeat changed the prompt, a score difference between repeats would be
    the question changing rather than the judge disagreeing with itself, and the
    consistency table would be measuring the wrong thing entirely.
    """
    seen = []

    def fake_complete(prompt, model, system=None, variant=0):
        seen.append({"prompt": prompt, "model": model, "system": system, "variant": variant})
        return '{"groundedness": 3, "correctness": 3, "tone": 3}'

    monkeypatch.setattr(judge.llm, "complete", fake_complete)

    judge.judge_one("q", "draft", retrieved(3), repeat=0)
    judge.judge_one("q", "draft", retrieved(3), repeat=1)

    assert [call["variant"] for call in seen] == [0, 1]
    assert seen[0]["prompt"] == seen[1]["prompt"]
    assert seen[0]["system"] == seen[1]["system"]
    assert seen[0]["model"] == seen[1]["model"] == judge.JUDGE_MODEL


def test_judge_model_is_a_different_family_from_the_generator():
    """The central claim of the stage. If this fails, the report cannot say it."""
    from src.agent import GENERATOR_MODEL

    assert judge.JUDGE_MODEL != GENERATOR_MODEL
    assert llm._provider(judge.JUDGE_MODEL) != llm._provider(GENERATOR_MODEL)


# --------------------------------------------------------------------------
# the consistency table
# --------------------------------------------------------------------------


def score_row(conversation_id, repeat, groundedness, correctness, tone, parse_ok=True):
    return {
        "system": "agent",
        "conversation_id": conversation_id,
        "repeat": repeat,
        "groundedness": groundedness,
        "correctness": correctness,
        "tone": tone,
        "parse_ok": parse_ok,
        "raw": "",
        "n_retrieved": 3,
        "reply": "draft",
    }


def test_consistency_is_none_for_a_single_pass():
    rows = [score_row(1, 0, 4, 4, 4)]

    assert judge.consistency(rows, repeat=1) is None


def test_consistency_reports_hand_computed_spreads(capsys):
    """Three conversations, spreads of 0, 1 and 2 on groundedness.

    mean spread = (0 + 1 + 2) / 3 = 1.0, exact agreement 1 of 3, worst 2 at id 3.
    """
    rows = [
        score_row(1, 0, 4, 5, 3),
        score_row(1, 1, 4, 5, 3),
        score_row(2, 0, 3, 5, 3),
        score_row(2, 1, 4, 5, 3),
        score_row(3, 0, 1, 5, 3),
        score_row(3, 1, 3, 5, 3),
    ]

    report = judge.consistency(rows, repeat=2)

    assert report["groundedness"]["mean_spread"] == pytest.approx(1.0)
    assert report["groundedness"]["n_agreed"] == 1
    assert report["groundedness"]["worst_spread"] == 2
    assert report["groundedness"]["worst_conversation_id"] == 3
    # correctness and tone were identical every time.
    assert report["correctness"]["mean_spread"] == 0.0
    assert report["correctness"]["n_agreed"] == 3


def test_consistency_excludes_parse_failures_rather_than_scoring_them(capsys):
    """A parse failure is a missing score, not a disagreement.

    Counting it as a disagreement would inflate the spread; counting it as
    agreement would hide a broken run. It is excluded and reported separately.
    """
    rows = [
        score_row(1, 0, 4, 4, 4),
        score_row(1, 1, 4, 4, 4),
        score_row(2, 0, 4, 4, 4),
        score_row(2, 1, None, None, None, parse_ok=False),
    ]

    report = judge.consistency(rows, repeat=2)

    assert report["n_compared"] == 1
    assert report["n_excluded"] == 1
    assert report["groundedness"]["mean_spread"] == 0.0


# --------------------------------------------------------------------------
# writing the file
# --------------------------------------------------------------------------


def test_merge_replaces_only_the_rows_it_rewrote():
    """The guard against `make judge-consistency` deleting `make judge`'s work.

    A 20-row repeat probe must refresh its own 40 rows and leave the other 130
    of a full run in place.
    """
    existing = [score_row(index, 0, 4, 4, 4) for index in range(150)]
    fresh = [score_row(index, repeat, 5, 5, 5) for index in range(20) for repeat in (0, 1)]

    merged = judge.merge_rows(existing, fresh)

    assert len(merged) == 150 + 20, "130 untouched + 20 refreshed + 20 new repeats"
    by_key = {(row["conversation_id"], row["repeat"]): row for row in merged}
    assert by_key[(0, 0)]["groundedness"] == 5, "refreshed"
    assert by_key[(0, 1)]["groundedness"] == 5, "added"
    assert by_key[(100, 0)]["groundedness"] == 4, "kept from the earlier full run"


def test_merge_keeps_other_systems_rows():
    """Judging the agent today and a baseline tomorrow must accumulate."""
    existing = [{**score_row(index, 0, 4, 4, 4), "system": "agent"} for index in range(3)]
    fresh = [{**score_row(index, 0, 2, 2, 2), "system": "trivial"} for index in range(3)]

    merged = judge.merge_rows(existing, fresh)

    assert len(merged) == 6
    assert {row["system"] for row in merged} == {"agent", "trivial"}


def test_write_scores_round_trips(tmp_path, capsys):
    path = tmp_path / "judge_scores.jsonl"
    rows = [score_row(index, 0, 4, 5, 3) for index in range(3)]

    judge.write_scores(path, rows, "agent")
    judge.write_scores(path, [score_row(0, 0, 1, 1, 1)], "agent")

    written = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    assert len(written) == 3
    assert written[0]["conversation_id"] == 0
    assert written[0]["groundedness"] == 1, "the rerun replaced conversation 0"
    assert written[1]["groundedness"] == 4, "conversation 1 was left alone"
