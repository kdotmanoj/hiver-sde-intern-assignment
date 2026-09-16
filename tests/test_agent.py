"""Tests for the non-LLM parts of the agent.

What is tested: corpus construction (the leakage exclusion is the experiment,
not housekeeping), brute-force retrieval, intent parsing, and prompt assembly.

What is not: whether the model classifies well or writes a good reply. That is
what the evaluation stage and the judge are for, and asserting it here would be
asserting on a cached string.

No test makes a network call and no test loads the embedding model.
"""

import json

import numpy as np
import pandas as pd
import pytest

from src import agent, embed, llm, ingest


@pytest.fixture(autouse=True)
def no_llm(monkeypatch):
    """Any test that reaches an LLM provider fails, rather than quietly mocking one."""

    def _boom(*args, **kwargs):
        raise AssertionError("an LLM call was made")

    monkeypatch.setattr(llm.requests, "post", _boom)


def unit(*components) -> np.ndarray:
    """A normalized float32 vector from the components given."""
    vector = np.array(components, dtype=np.float32)
    return vector / np.linalg.norm(vector)


def fake_embed(monkeypatch, table: dict[str, np.ndarray]):
    """Replace agent's embed() with a lookup, so no model is ever loaded."""

    def _embed(texts):
        return np.stack([table[text] for text in texts])

    monkeypatch.setattr(agent, "embed", _embed)


# --------------------------------------------------------------------------
# a synthetic corpus on disk
# --------------------------------------------------------------------------
#
# data/interim/ is gitignored, so nothing here may depend on the real parquet.
# draw_sample draws taxonomy.N_SAMPLE (2,000) and that default is bound at
# definition time, so the synthetic frame has to be bigger than that.

N_ROWS = 2_400


def write_fixture(tmp_path, n_golden=50, n_orphan=7, overlap=3):
    """Write a synthetic spotify.parquet and labels.jsonl. Returns the paths.

    `overlap` rows are BOTH golden and orphan, so the test can check that the
    two exclusion counts are reported separately and not double-counted in the
    total.
    """
    frame = pd.DataFrame(
        {
            "conversation_id": np.arange(N_ROWS, dtype=np.int64),
            "source": "customer_root",
            "opening_text": [f"opening number {i}" for i in range(N_ROWS)],
            "first_reply_text": [f"reply number {i}" for i in range(N_ROWS)],
            "first_reply_orphan_part": False,
        }
    )

    # Golden ids first, then orphans chosen to overlap the tail of the golden
    # block by `overlap` rows.
    golden_ids = list(range(n_golden))
    orphan_start = n_golden - overlap
    orphan_ids = list(range(orphan_start, orphan_start + n_orphan))
    frame.loc[frame["conversation_id"].isin(orphan_ids), "first_reply_orphan_part"] = True

    parquet = tmp_path / "spotify.parquet"
    ingest.write_parquet(frame, parquet)

    labels = tmp_path / "labels.jsonl"
    with open(labels, "w", encoding="utf-8") as handle:
        for conversation_id in golden_ids:
            handle.write(
                json.dumps(
                    {
                        "conversation_id": conversation_id,
                        "intent": "app_bug",
                        "escalate": False,
                        "opening_text": f"opening number {conversation_id}",
                    }
                )
                + "\n"
            )
    return parquet, labels


# --------------------------------------------------------------------------
# build_corpus
# --------------------------------------------------------------------------


def test_build_corpus_excludes_every_golden_id(tmp_path):
    parquet, labels = write_fixture(tmp_path)

    corpus = agent.build_corpus(parquet, labels, verbose=False)

    golden_ids = agent.load_golden_ids(labels)
    assert not corpus["conversation_id"].isin(golden_ids).any()


def test_build_corpus_excludes_orphan_first_replies(tmp_path):
    parquet, labels = write_fixture(tmp_path)

    corpus = agent.build_corpus(parquet, labels, verbose=False)

    assert not corpus["first_reply_orphan_part"].any()


def test_build_corpus_row_count_is_the_sample_minus_the_union_of_exclusions(tmp_path):
    """Golden and orphan overlap, so the losses are a union and not a sum."""
    parquet, labels = write_fixture(tmp_path, n_golden=50, n_orphan=7, overlap=3)

    from src.taxonomy import draw_sample, load_eligible

    eligible, _ = load_eligible(parquet)
    sampled = draw_sample(eligible)

    golden_ids = agent.load_golden_ids(labels)
    excluded = sampled["conversation_id"].isin(golden_ids) | sampled["first_reply_orphan_part"]

    corpus = agent.build_corpus(parquet, labels, verbose=False)

    assert len(corpus) == len(sampled) - int(excluded.sum())
    # The union really is smaller than the sum, or this test proves nothing.
    n_golden = int(sampled["conversation_id"].isin(golden_ids).sum())
    n_orphan = int(sampled["first_reply_orphan_part"].sum())
    assert int(excluded.sum()) < n_golden + n_orphan


def test_build_corpus_is_deterministic(tmp_path):
    parquet, labels = write_fixture(tmp_path)

    first = agent.build_corpus(parquet, labels, verbose=False)
    second = agent.build_corpus(parquet, labels, verbose=False)

    pd.testing.assert_frame_equal(first, second)


def test_load_golden_ids_reads_every_line(tmp_path):
    _, labels = write_fixture(tmp_path, n_golden=50)

    assert len(agent.load_golden_ids(labels)) == 50


# --------------------------------------------------------------------------
# retrieve
# --------------------------------------------------------------------------
#
# A four-row corpus with hand-chosen 2-D unit vectors, so every similarity below
# is arithmetic I can do on paper:
#
#   row 0  (1.0, 0.0)   dot query = 1.00
#   row 1  (0.0, 1.0)   dot query = 0.00
#   row 2  (0.6, 0.8)   dot query = 0.60
#   row 3  (1.0, 0.0)   dot query = 1.00   <- deliberate tie with row 0
#
# query = (1.0, 0.0)

QUERY = "the query"
CORPUS_VECTORS = np.stack([unit(1, 0), unit(0, 1), unit(0.6, 0.8), unit(1, 0)])


@pytest.fixture
def small_corpus():
    return pd.DataFrame(
        {
            "conversation_id": [10, 11, 12, 13],
            "opening_text": ["row zero", "row one", "row two", "row three"],
            "first_reply_text": ["reply zero", "reply one", "reply two", "reply three"],
        }
    )


@pytest.fixture
def query_table():
    return {QUERY: unit(1, 0)}


def test_retrieve_returns_hand_computed_similarities(monkeypatch, small_corpus, query_table):
    fake_embed(monkeypatch, query_table)

    results = agent.retrieve(QUERY, small_corpus, CORPUS_VECTORS, k=3)

    assert [r.conversation_id for r in results] == [10, 13, 12]
    assert results[0].similarity == pytest.approx(1.0)
    assert results[1].similarity == pytest.approx(1.0)
    assert results[2].similarity == pytest.approx(0.6)


def test_retrieve_ties_break_on_corpus_order(monkeypatch, small_corpus, query_table):
    """Rows 0 and 3 are identical vectors; the lower index must come first.

    argsort(kind='stable') is what guarantees this. With an unstable sort the
    order would depend on numpy's partitioning and the run would not reproduce.
    """
    fake_embed(monkeypatch, query_table)

    results = agent.retrieve(QUERY, small_corpus, CORPUS_VECTORS, k=2)

    assert [r.conversation_id for r in results] == [10, 13]


def test_retrieve_carries_the_reply_text_not_just_the_opening(
    monkeypatch, small_corpus, query_table
):
    """The first_reply_text is the whole point -- it is what grounds the draft."""
    fake_embed(monkeypatch, query_table)

    results = agent.retrieve(QUERY, small_corpus, CORPUS_VECTORS, k=1)

    assert results[0].opening_text == "row zero"
    assert results[0].first_reply_text == "reply zero"


def test_identical_text_retrieves_itself_at_similarity_one(monkeypatch, small_corpus):
    """Sanity check on the geometry: a unit vector's dot with itself is 1.0.

    This is also why the golden ids are excluded from the corpus -- without that
    exclusion, every golden query would hit this case.
    """
    fake_embed(monkeypatch, {"row two": unit(0.6, 0.8)})

    results = agent.retrieve("row two", small_corpus, CORPUS_VECTORS, k=1)

    assert results[0].conversation_id == 12
    assert results[0].similarity == pytest.approx(1.0)


def test_retrieve_k_larger_than_the_corpus_returns_everything(
    monkeypatch, small_corpus, query_table
):
    fake_embed(monkeypatch, query_table)

    results = agent.retrieve(QUERY, small_corpus, CORPUS_VECTORS, k=99)

    assert len(results) == 4


def test_retrieve_rejects_non_positive_k(monkeypatch, small_corpus, query_table):
    fake_embed(monkeypatch, query_table)

    with pytest.raises(ValueError, match="k must be positive"):
        agent.retrieve(QUERY, small_corpus, CORPUS_VECTORS, k=0)


def test_retrieve_does_not_renormalize(monkeypatch, small_corpus):
    """src.embed guarantees unit vectors; retrieve relies on that and must not hide it.

    Handed a deliberately unnormalized query of length 2, the similarities come
    back doubled. A retrieve() that quietly renormalized would return 1.0 here
    and the guarantee would be untestable from outside.
    """
    fake_embed(monkeypatch, {QUERY: np.array([2.0, 0.0], dtype=np.float32)})

    results = agent.retrieve(QUERY, small_corpus, CORPUS_VECTORS, k=1)

    assert results[0].similarity == pytest.approx(2.0)


def test_corpus_vectors_asserts_one_vector_per_row(monkeypatch, small_corpus):
    """A short embed() result must fail loudly, not silently misalign the corpus."""
    monkeypatch.setattr(agent, "embed", lambda texts: np.stack([unit(1, 0)]))

    with pytest.raises(AssertionError, match="vectors for"):
        agent.corpus_vectors(small_corpus)


def test_corpus_vectors_under_cache_only_never_loads_the_model(monkeypatch, small_corpus):
    """CACHE_ONLY=1 on an uncached text must raise, not fall through to encoding.

    This is what makes `make eval` safe: the offline replay guarantee is only
    real if a cache miss is an error rather than a silent model load.
    """
    monkeypatch.setattr(embed, "CACHE_DIR", tmp_dir := embed.CACHE_DIR.parent / "does-not-exist")
    monkeypatch.setattr(embed, "_model", None)
    monkeypatch.setenv("CACHE_ONLY", "1")

    def _no_model():
        raise AssertionError("the embedding model was loaded")

    monkeypatch.setattr(embed, "_load_model", _no_model)

    with pytest.raises(embed.CacheOnlyError):
        agent.corpus_vectors(small_corpus)
    assert not tmp_dir.exists()


# --------------------------------------------------------------------------
# parse_intent
# --------------------------------------------------------------------------


@pytest.mark.parametrize("intent", agent.INTENTS)
def test_every_taxonomy_label_parses(intent):
    parsed = agent.parse_intent(intent)

    assert parsed.intent == intent
    assert parsed.parse_ok


@pytest.mark.parametrize(
    "raw",
    [
        "  billing_dispute  ",
        "BILLING_DISPUTE",
        "billing dispute",
        "billing-dispute",
        "`billing_dispute`",
        '"billing_dispute."',
    ],
)
def test_formatting_noise_still_parses(raw):
    parsed = agent.parse_intent(raw)

    assert parsed.intent == "billing_dispute"
    assert parsed.parse_ok


def test_a_single_label_inside_prose_parses():
    parsed = agent.parse_intent("The intent here is app_bug, since the app crashes.")

    assert parsed.intent == "app_bug"
    assert parsed.parse_ok


def test_a_response_naming_two_labels_is_a_parse_failure():
    """Picking one would be a coin flip dressed up as a classification."""
    parsed = agent.parse_intent("could be app_bug or playback_library")

    assert parsed.intent == "other"
    assert not parsed.parse_ok


def test_an_invented_label_falls_back_to_other_but_is_flagged():
    """The fallback keeps the pipeline running; parse_ok keeps it honest.

    Without the flag, a model that answers nonsense looks like a model that
    answered `other`, and the `other` class absorbs the failure rate.
    """
    parsed = agent.parse_intent("password_reset")

    assert parsed.intent == "other"
    assert not parsed.parse_ok
    assert parsed.raw == "password_reset"


def test_empty_response_is_a_parse_failure():
    parsed = agent.parse_intent("")

    assert parsed.intent == "other"
    assert not parsed.parse_ok


def test_a_genuine_other_is_not_confused_with_a_parse_failure():
    parsed = agent.parse_intent("other")

    assert parsed.intent == "other"
    assert parsed.parse_ok


# --------------------------------------------------------------------------
# prompts
# --------------------------------------------------------------------------


def test_classify_system_prompt_lists_all_nine_intents():
    prompt = agent._classify_system_prompt()

    for intent in agent.INTENTS:
        assert f"- {intent}:" in prompt


def test_classify_system_prompt_has_a_gloss_for_every_intent():
    """A label listed without its definition would be a silent taxonomy drift."""
    assert set(agent.INTENT_GLOSS) == set(agent.INTENTS)


def test_reply_prompt_includes_every_retrieved_precedent():
    retrieved = [
        agent.Retrieved(1, 0.9, "they asked one", "spotify said one"),
        agent.Retrieved(2, 0.8, "they asked two", "spotify said two"),
        agent.Retrieved(3, 0.7, "they asked three", "spotify said three"),
    ]

    prompt = agent._reply_prompt("the new message", "app_bug", retrieved)

    for item in retrieved:
        assert item.opening_text in prompt
        assert item.first_reply_text in prompt
    assert "the new message" in prompt
    assert "app_bug" in prompt


def test_reply_prompt_survives_an_empty_retrieval():
    """Only reachable with an empty corpus, but it must not produce a bare prompt."""
    prompt = agent._reply_prompt("the new message", "other", [])

    assert "no similar past conversation" in prompt
    assert "the new message" in prompt


@pytest.mark.parametrize(
    "prompt",
    [
        agent._classify_system_prompt(),
        agent.REPLY_SYSTEM_PROMPT,
        agent._classify_prompt("hello"),
        agent._reply_prompt("hello", "app_bug", [agent.Retrieved(1, 0.9, "a", "b")]),
    ],
)
def test_no_prompt_has_an_unformatted_placeholder(prompt):
    """A stray {} or {name} means an f-string was written as a plain string."""
    assert "{" not in prompt and "}" not in prompt


# --------------------------------------------------------------------------
# answer
# --------------------------------------------------------------------------


@pytest.fixture
def stub_pipeline(monkeypatch, query_table):
    """classify and reply replaced with stubs; retrieve and escalate stay real."""
    fake_embed(monkeypatch, query_table)
    monkeypatch.setattr(
        agent, "classify", lambda text, **kw: agent.Classification("billing_dispute", True, "x")
    )
    monkeypatch.setattr(agent, "reply", lambda *args, **kwargs: "a generated reply")


def test_escalated_case_generates_no_reply_by_default(stub_pipeline, small_corpus):
    result = agent.answer(QUERY, small_corpus, CORPUS_VECTORS)

    assert result["decision"] == "escalate"
    assert result["reply"] is None
    assert result["reply_generated"] is False


def test_force_reply_generates_without_changing_the_decision(stub_pipeline, small_corpus):
    """The eval stage needs a reply for all 150 goldens to score the judge.

    Scoring only the auto-handled subset would make the judge's numbers
    conditional on the escalation rule being right, which is one of the things
    under test. So force_reply must move the reply and nothing else.
    """
    default = agent.answer(QUERY, small_corpus, CORPUS_VECTORS)
    forced = agent.answer(QUERY, small_corpus, CORPUS_VECTORS, force_reply=True)

    assert forced["reply"] == "a generated reply"
    assert forced["reply_generated"] is True

    assert forced["decision"] == default["decision"]
    assert forced["reason"] == default["reason"]
    assert forced["triggered_signals"] == default["triggered_signals"]
    assert forced["signals_by_family"] == default["signals_by_family"]


def test_answer_reports_all_three_families(stub_pipeline, small_corpus):
    result = agent.answer(QUERY, small_corpus, CORPUS_VECTORS)

    assert set(result["signals_by_family"]) == {"intent", "similarity", "text"}


def test_answer_records_the_top_similarity_it_escalated_on(stub_pipeline, small_corpus):
    result = agent.answer(QUERY, small_corpus, CORPUS_VECTORS)

    assert result["top_similarity"] == pytest.approx(1.0)
    assert result["retrieved"][0]["similarity"] == pytest.approx(1.0)


def test_answer_is_json_serializable(stub_pipeline, small_corpus):
    """The golden run writes these straight to jsonl, so numpy floats would break it."""
    result = agent.answer(QUERY, small_corpus, CORPUS_VECTORS, force_reply=True)

    json.dumps(result)


# --------------------------------------------------------------------------
# preflight
# --------------------------------------------------------------------------
#
# The failure this guards against is specific and was paid for once already: a
# golden run that discovers the daily quota wall at call 21 of 300 has spent the
# calls, produced nothing usable, and left a half-written cache. So the count
# has to happen before the first call, and it has to be right.

GOLDEN_TEXTS = ["golden one", "golden two", "golden three"]


@pytest.fixture
def golden_rows():
    return [
        {"conversation_id": 100 + n, "opening_text": text}
        for n, text in enumerate(GOLDEN_TEXTS)
    ]


@pytest.fixture
def isolated_llm_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "CACHE_DIR", tmp_path / "llm")
    return tmp_path / "llm"


@pytest.fixture
def golden_embeds(monkeypatch):
    """Every golden text embeds to the same unit vector as the corpus's row 0."""
    fake_embed(monkeypatch, {text: unit(1, 0) for text in GOLDEN_TEXTS})


def seed_classify(text, intent):
    """Write a classify response into the (isolated) cache."""
    llm._write_cache(
        llm._cache_key(
            prompt=agent._classify_prompt(text),
            model=agent.CLASSIFIER_MODEL,
            system=agent._classify_system_prompt(),
        ),
        model=agent.CLASSIFIER_MODEL,
        system=agent._classify_system_prompt(),
        prompt=agent._classify_prompt(text),
        response=intent,
    )


def seed_reply(text, intent, corpus, vectors):
    """Write a reply response into the (isolated) cache, for the exact prompt."""
    retrieved = agent.retrieve(text, corpus, vectors, k=agent.K)
    prompt = agent._reply_prompt(text, intent, retrieved)
    llm._write_cache(
        llm._cache_key(prompt=prompt, model=agent.GENERATOR_MODEL, system=None),
        model=agent.GENERATOR_MODEL,
        system=None,
        prompt=prompt,
        response="a cached reply",
    )


def test_preflight_on_a_cold_cache_counts_two_calls_per_opening(
    isolated_llm_cache, golden_embeds, golden_rows, small_corpus
):
    plan = agent.preflight(golden_rows, small_corpus, CORPUS_VECTORS, force_reply=True)

    assert plan["classify_misses"] == 3
    assert plan["reply_unknown"] == 3
    assert plan["total"] == 6


def test_preflight_on_a_cold_cache_is_flagged_as_an_upper_bound(
    isolated_llm_cache, golden_embeds, golden_rows, small_corpus
):
    """The reply prompt contains the intent, so it cannot be looked up until
    classify has run. An uncounted-but-possible call must round UP."""
    plan = agent.preflight(golden_rows, small_corpus, CORPUS_VECTORS, force_reply=True)

    assert plan["exact"] is False


def test_preflight_on_a_warm_cache_is_exact_and_zero(
    isolated_llm_cache, golden_embeds, golden_rows, small_corpus
):
    for text in GOLDEN_TEXTS:
        seed_classify(text, "app_bug")
        seed_reply(text, "app_bug", small_corpus, CORPUS_VECTORS)

    plan = agent.preflight(golden_rows, small_corpus, CORPUS_VECTORS, force_reply=True)

    assert plan["total"] == 0
    assert plan["exact"] is True


def test_preflight_counts_only_the_missing_replies_when_classify_is_warm(
    isolated_llm_cache, golden_embeds, golden_rows, small_corpus
):
    for text in GOLDEN_TEXTS:
        seed_classify(text, "app_bug")
    seed_reply(GOLDEN_TEXTS[0], "app_bug", small_corpus, CORPUS_VECTORS)

    plan = agent.preflight(golden_rows, small_corpus, CORPUS_VECTORS, force_reply=True)

    assert plan["classify_misses"] == 0
    assert plan["reply_misses"] == 2
    assert plan["exact"] is True
    assert plan["total"] == 2


def test_preflight_does_not_count_replies_the_escalation_rule_would_skip(
    isolated_llm_cache, golden_embeds, golden_rows, small_corpus
):
    """billing_dispute escalates on intent alone, so the default run skips the reply."""
    for text in GOLDEN_TEXTS:
        seed_classify(text, "billing_dispute")

    plan = agent.preflight(golden_rows, small_corpus, CORPUS_VECTORS, force_reply=False)

    assert plan["skipped_by_escalation"] == 3
    assert plan["reply_misses"] == 0
    assert plan["total"] == 0


def test_force_reply_puts_the_escalated_replies_back_in_the_budget(
    isolated_llm_cache, golden_embeds, golden_rows, small_corpus
):
    for text in GOLDEN_TEXTS:
        seed_classify(text, "billing_dispute")

    default = agent.preflight(golden_rows, small_corpus, CORPUS_VECTORS, force_reply=False)
    forced = agent.preflight(golden_rows, small_corpus, CORPUS_VECTORS, force_reply=True)

    assert default["total"] == 0
    assert forced["total"] == 3


def test_preflight_makes_no_llm_call(
    isolated_llm_cache, golden_embeds, golden_rows, small_corpus
):
    """Guaranteed by the autouse no_llm fixture: any provider call fails the test.

    Stated as its own test because it is the entire point of the function -- a
    preflight that spent a call to find out how many calls to spend would be
    worse than useless.
    """
    agent.preflight(golden_rows, small_corpus, CORPUS_VECTORS, force_reply=True)


# --------------------------------------------------------------------------
# the budget guard
# --------------------------------------------------------------------------


def test_report_preflight_raises_when_the_run_exceeds_the_budget(capsys):
    plan = {
        "n_golden": 150, "classify_misses": 150, "reply_misses": 0,
        "reply_unknown": 150, "skipped_by_escalation": 0, "total": 300, "exact": False,
    }

    with pytest.raises(agent.QuotaError, match="Nothing has been sent"):
        agent._report_preflight(plan, max_live_calls=20)


def test_report_preflight_allows_a_run_inside_the_budget():
    plan = {
        "n_golden": 150, "classify_misses": 0, "reply_misses": 10,
        "reply_unknown": 0, "skipped_by_escalation": 0, "total": 10, "exact": True,
    }

    agent._report_preflight(plan, max_live_calls=320)


def test_report_preflight_names_both_numbers_in_the_error():
    """The message has to say what it needed AND what it was allowed."""
    plan = {
        "n_golden": 150, "classify_misses": 300, "reply_misses": 0,
        "reply_unknown": 0, "skipped_by_escalation": 0, "total": 300, "exact": True,
    }

    with pytest.raises(agent.QuotaError) as caught:
        agent._report_preflight(plan, max_live_calls=20)

    assert "300" in str(caught.value) and "20" in str(caught.value)


def test_default_budget_admits_one_full_cold_golden_run():
    """150 classify + 150 reply = 300. If MAX_LIVE_CALLS ever drops below that,
    the intended run stops being possible and that should be a deliberate act."""
    assert agent.MAX_LIVE_CALLS >= 300


def test_golden_run_aborts_before_the_loop_when_over_budget(
    isolated_llm_cache, golden_embeds, small_corpus, tmp_path, monkeypatch
):
    """The abort must happen before any result is produced, not partway through."""
    golden_path = tmp_path / "golden.jsonl"
    with open(golden_path, "w", encoding="utf-8") as handle:
        for n, text in enumerate(GOLDEN_TEXTS):
            handle.write(json.dumps({"conversation_id": n, "opening_text": text}) + "\n")

    def _boom(*args, **kwargs):
        raise AssertionError("answer() ran despite the budget abort")

    monkeypatch.setattr(agent, "answer", _boom)
    out = tmp_path / "results.jsonl"

    with pytest.raises(agent.QuotaError):
        agent._run_golden(
            small_corpus, CORPUS_VECTORS, force_reply=True,
            out=out, golden_path=golden_path, max_live_calls=1,
        )

    assert not out.exists()
