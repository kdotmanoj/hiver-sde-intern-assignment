"""Tests for the two non-LLM baselines.

What is tested: the shape contract with agent.answer (evaluate.py depends on it),
the leave-one-out exclusion and its tie-break (the part of the k-NN baseline that
is a decision rather than arithmetic), reply verbatimness, and the claim that no
golden id is retrievable.

What is not: whether either baseline scores well. That is the point of them.

No test makes a network call and no test loads the embedding model.
"""

import json

import numpy as np
import pandas as pd
import pytest

from src import agent, baselines, llm
from src.escalate import FAMILIES


@pytest.fixture(autouse=True)
def no_llm(monkeypatch):
    """Any test that reaches an LLM provider fails, rather than quietly mocking one."""

    def _boom(*args, **kwargs):
        raise AssertionError("an LLM call was made")

    monkeypatch.setattr(llm.requests, "post", _boom)


@pytest.fixture(autouse=True)
def no_shared_bank(monkeypatch):
    """No test may see the process-wide bank cache, in either direction."""
    monkeypatch.setattr(baselines, "_BANK", None)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def unit(degrees: float) -> np.ndarray:
    """A normalized 2-D vector at the given angle.

    Angles make the neighbour ORDER readable at the point of definition: with a
    query at 0 degrees, similarity is cos(angle), so smaller angle == nearer.
    """
    radians = np.deg2rad(degrees)
    return np.array([np.cos(radians), np.sin(radians)], dtype=np.float32)


def fake_embed(monkeypatch, table: dict[str, np.ndarray]):
    """Replace embed() in BOTH modules, so no model is ever loaded.

    baselines calls embed directly for the vote; agent.retrieve calls its own
    module-level embed for the reply. Patching one and not the other would load
    the real model in half the tests.
    """

    def _embed(texts):
        return np.stack([table[text] for text in texts])

    monkeypatch.setattr(baselines, "embed", _embed)
    monkeypatch.setattr(agent, "embed", _embed)


# The vector table shared by the shape-contract tests, which care about keys and
# not about which neighbour wins.
SIX_BANK_VECTORS = {
    "query": unit(0.0),
    "near": unit(5.0),
    **{f"b{i}": unit(float(i)) for i in range(6)},
}

# Every text a vote test uses as a QUERY, all at 0 degrees so that the bank
# angles set by make_bank are read directly as neighbour rank. The bank's own
# vectors come from make_bank, so only the query needs to be embeddable here --
# "b0" appears because the leave-one-out tests query with a bank member.
VOTE_QUERY_VECTORS = {"q": unit(0.0), "outside": unit(0.0), "b0": unit(0.0)}


def make_bank(entries: list[tuple[str, str, float]]) -> baselines.GoldenBank:
    """A GoldenBank from (text, intent, angle) triples. Ids are the positions."""
    return baselines.GoldenBank(
        conversation_ids=list(range(len(entries))),
        texts=[text for text, _, _ in entries],
        intents=[intent for _, intent, _ in entries],
        vectors=np.stack([unit(angle) for _, _, angle in entries]),
    )


def make_corpus(entries: list[tuple[int, str, str, float]]):
    """A corpus frame and its vector matrix from (id, opening, reply, angle)."""
    frame = pd.DataFrame(
        {
            "conversation_id": [cid for cid, _, _, _ in entries],
            "opening_text": [opening for _, opening, _, _ in entries],
            "first_reply_text": [reply for _, _, reply, _ in entries],
        }
    )
    vectors = np.stack([unit(angle) for _, _, _, angle in entries])
    return frame, vectors


def real_bank_without_vectors() -> baselines.GoldenBank:
    """The actual 150 labels with placeholder vectors.

    The trivial baseline and majority_intent never touch the vectors, so this
    reads the committed labels without embedding anything.
    """
    with open(agent.GOLDEN_LABELS, encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return baselines.GoldenBank(
        conversation_ids=[row["conversation_id"] for row in rows],
        texts=[row["opening_text"] for row in rows],
        intents=[row["intent"] for row in rows],
        vectors=np.zeros((len(rows), 2), dtype=np.float32),
    )


# --------------------------------------------------------------------------
# the shape contract
# --------------------------------------------------------------------------


def test_both_baselines_return_every_key_agent_answer_returns(monkeypatch):
    """evaluate.py calls all three identically, so all three must return the same keys.

    The expected keys come from a real agent.answer() call rather than a hardcoded
    list, so that adding a key to answer() fails here instead of silently going
    unscored for the baselines.
    """
    corpus, vectors = make_corpus([(10, "near", "the near reply", 5.0)])
    bank = make_bank([(f"b{i}", "app_bug", float(i)) for i in range(6)])
    fake_embed(monkeypatch, SIX_BANK_VECTORS)

    monkeypatch.setattr(
        agent, "classify", lambda text, **kw: agent.Classification("app_bug", True, "app_bug")
    )
    monkeypatch.setattr(agent, "reply", lambda *args, **kwargs: "a draft")
    expected = set(agent.answer("query", corpus, vectors).keys())

    for answer_fn in (baselines.trivial_answer, baselines.knn_answer):
        keys = set(answer_fn("query", corpus, vectors, bank=bank).keys())
        assert expected <= keys, f"{answer_fn.__name__} is missing {expected - keys}"


def test_neither_baseline_ever_escalates(monkeypatch):
    corpus, vectors = make_corpus([(10, "near", "the near reply", 5.0)])
    bank = make_bank([(f"b{i}", "app_bug", float(i)) for i in range(6)])
    fake_embed(monkeypatch, SIX_BANK_VECTORS)

    for answer_fn in (baselines.trivial_answer, baselines.knn_answer):
        result = answer_fn("query", corpus, vectors, bank=bank)
        assert result["decision"] == "auto"
        assert result["triggered_signals"] == []
        # All families present and empty, so the eval stage can count true
        # negatives per family without special-casing the baselines.
        assert set(result["signals_by_family"]) == set(FAMILIES)
        assert all(signals == [] for signals in result["signals_by_family"].values())


def test_force_reply_is_accepted_and_changes_nothing(monkeypatch):
    """It exists to override an escalation gate, and the baselines have no gate."""
    corpus, vectors = make_corpus([(10, "near", "the near reply", 5.0)])
    bank = make_bank([(f"b{i}", "app_bug", float(i)) for i in range(6)])
    fake_embed(monkeypatch, SIX_BANK_VECTORS)

    for answer_fn in (baselines.trivial_answer, baselines.knn_answer):
        off = answer_fn("query", corpus, vectors, bank=bank)
        on = answer_fn("query", corpus, vectors, force_reply=True, bank=bank)
        assert off == on
        assert off["reply_generated"]


# --------------------------------------------------------------------------
# the trivial baseline
# --------------------------------------------------------------------------


def test_majority_intent_of_the_real_golden_set_is_feature_request():
    """Hand-counted from data/golden/labels.jsonl: 32 of 150."""
    bank = real_bank_without_vectors()

    assert len(bank.texts) == 150
    assert bank.majority_intent() == "feature_request"
    assert bank.intents.count("feature_request") == 32


def test_majority_intent_breaks_ties_on_taxonomy_order_not_dict_order():
    """A tie must not depend on which label happened to be labelled first."""
    # app_bug precedes content_issue in agent.INTENTS, and is deliberately the
    # LATER of the two here, so insertion order would give the other answer.
    bank = make_bank(
        [
            ("a", "content_issue", 0.0),
            ("b", "content_issue", 1.0),
            ("c", "app_bug", 2.0),
            ("d", "app_bug", 3.0),
        ]
    )

    assert bank.majority_intent() == "app_bug"


def test_trivial_predicts_the_majority_intent_on_every_one_of_the_150():
    """So its accuracy is exactly the majority-class share, by construction."""
    bank = real_bank_without_vectors()
    corpus, vectors = make_corpus([(10, "near", "the near reply", 5.0)])

    predictions = [
        baselines.trivial_answer(text, corpus, vectors, bank=bank)["intent"] for text in bank.texts
    ]

    assert set(predictions) == {"feature_request"}
    n_correct = sum(1 for p, truth in zip(predictions, bank.intents) if p == truth)
    assert n_correct == 32


def test_trivial_retrieves_nothing_and_returns_the_canned_reply(monkeypatch):
    """No embedding call at all: the fake table is empty, so any lookup would KeyError."""
    fake_embed(monkeypatch, {})
    bank = real_bank_without_vectors()
    corpus, vectors = make_corpus([(10, "near", "the near reply", 5.0)])

    result = baselines.trivial_answer("anything at all", corpus, vectors, bank=bank)

    assert result["retrieved"] == []
    assert result["top_similarity"] == 0.0
    assert result["reply"] == baselines.CANNED_REPLY
    assert result["reply_generated"]


# --------------------------------------------------------------------------
# the k-NN intent vote: leave-one-out
# --------------------------------------------------------------------------


def test_leave_one_out_excludes_the_query_and_that_changes_the_answer(monkeypatch):
    """The exclusion has to be load-bearing, or the test proves nothing.

    b0 is the query. Including it would push b5 out of the top 5 and produce a
    2-2 tie that the tie-break hands to billing_dispute; excluding it lets
    content_issue win 3-2. Different answers, so the mask is doing real work.
    """
    fake_embed(monkeypatch, VOTE_QUERY_VECTORS)
    bank = make_bank(
        [
            ("b0", "playback_library", 0.0),
            ("b1", "billing_dispute", 1.0),
            ("b2", "billing_dispute", 2.0),
            ("b3", "content_issue", 3.0),
            ("b4", "content_issue", 4.0),
            ("b5", "content_issue", 5.0),
        ]
    )

    intent, votes = baselines.knn_intent("b0", bank, k=5)

    assert intent == "content_issue"
    assert 0 not in [vote.conversation_id for vote in votes]
    assert [vote.conversation_id for vote in votes] == [1, 2, 3, 4, 5]


def test_a_query_outside_the_bank_excludes_nothing(monkeypatch):
    """LOO is for golden queries; anything else gets the full 150."""
    fake_embed(monkeypatch, VOTE_QUERY_VECTORS)
    bank = make_bank(
        [
            ("b0", "playback_library", 0.0),
            ("b1", "billing_dispute", 1.0),
            ("b2", "billing_dispute", 2.0),
            ("b3", "content_issue", 3.0),
            ("b4", "content_issue", 4.0),
            ("b5", "content_issue", 5.0),
        ]
    )

    # "outside" sits at 0 degrees, exactly where b0 does, but is not b0.
    intent, votes = baselines.knn_intent("outside", bank, k=5)

    assert 0 in [vote.conversation_id for vote in votes]
    assert intent == "billing_dispute"


def test_the_tie_break_is_the_closest_member_not_the_label_order(monkeypatch):
    """Same 2-2-1 counts, reversed distances, opposite winner."""
    fake_embed(monkeypatch, VOTE_QUERY_VECTORS)
    closer_billing = make_bank(
        [
            ("b1", "billing_dispute", 1.0),
            ("b2", "billing_dispute", 2.0),
            ("b3", "content_issue", 3.0),
            ("b4", "content_issue", 4.0),
            ("b5", "other", 5.0),
        ]
    )
    closer_content = make_bank(
        [
            ("b1", "content_issue", 1.0),
            ("b2", "content_issue", 2.0),
            ("b3", "billing_dispute", 3.0),
            ("b4", "billing_dispute", 4.0),
            ("b5", "other", 5.0),
        ]
    )

    assert baselines.knn_intent("q", closer_billing, k=5)[0] == "billing_dispute"
    assert baselines.knn_intent("q", closer_content, k=5)[0] == "content_issue"


def test_an_outright_majority_wins_regardless_of_distance(monkeypatch):
    """The tie-break must only fire on ties, not quietly become 1-NN."""
    fake_embed(monkeypatch, VOTE_QUERY_VECTORS)
    bank = make_bank(
        [
            ("b1", "how_to_question", 1.0),
            ("b2", "app_bug", 2.0),
            ("b3", "app_bug", 3.0),
            ("b4", "app_bug", 4.0),
            ("b5", "other", 5.0),
        ]
    )

    assert baselines.knn_intent("q", bank, k=5)[0] == "app_bug"


def test_a_bank_too_small_for_the_vote_fails_loudly(monkeypatch):
    fake_embed(monkeypatch, VOTE_QUERY_VECTORS)
    bank = make_bank([("b0", "app_bug", 0.0), ("b1", "other", 1.0)])

    with pytest.raises(AssertionError, match="bank rows available"):
        baselines.knn_intent("b0", bank, k=5)


def test_duplicate_openings_in_the_bank_are_rejected_at_load(tmp_path):
    """LOO matches on text, so a duplicate would vote its own label in under another id."""
    labels = tmp_path / "labels.jsonl"
    with open(labels, "w", encoding="utf-8") as handle:
        for conversation_id in (1, 2):
            handle.write(
                json.dumps(
                    {
                        "conversation_id": conversation_id,
                        "intent": "app_bug",
                        "opening_text": "the same opening twice",
                    }
                )
                + "\n"
            )

    with pytest.raises(AssertionError, match="duplicate opening_text"):
        baselines.GoldenBank.load(labels)


# --------------------------------------------------------------------------
# the k-NN reply
# --------------------------------------------------------------------------


def test_the_reply_is_the_nearest_neighbours_reply_byte_for_byte(monkeypatch):
    """Nearest only, and untouched -- no trimming, no cleaning, no blending."""
    corpus, vectors = make_corpus(
        [
            (10, "far", "the far reply", 40.0),
            (11, "near", "  the NEAR reply /SC  ", 5.0),
            (12, "middling", "the middling reply", 20.0),
        ]
    )
    bank = make_bank([(f"b{i}", "app_bug", float(i)) for i in range(6)])
    fake_embed(
        monkeypatch,
        {
            "query": unit(0.0),
            "far": unit(40.0),
            "near": unit(5.0),
            "middling": unit(20.0),
            **{f"b{i}": unit(float(i)) for i in range(6)},
        },
    )

    result = baselines.knn_answer("query", corpus, vectors, k=3, bank=bank)

    assert result["reply"] == "  the NEAR reply /SC  "
    assert result["retrieved"][0]["conversation_id"] == 11
    assert result["top_similarity"] == pytest.approx(np.cos(np.deg2rad(5.0)), abs=1e-6)


def test_the_vote_pool_and_the_reply_pool_are_different(monkeypatch):
    """The intent comes from the golden bank, the reply from the corpus. Never crossed."""
    corpus, vectors = make_corpus([(10, "near", "the near reply", 1.0)])
    bank = make_bank([(f"b{i}", "billing_dispute", float(i + 1)) for i in range(5)])
    fake_embed(
        monkeypatch,
        {"query": unit(0.0), "near": unit(1.0), **{f"b{i}": unit(float(i + 1)) for i in range(5)}},
    )

    result = baselines.knn_answer("query", corpus, vectors, bank=bank)

    # The only corpus row carries no intent at all; the label can only have come
    # from the bank. The only bank row carries no reply; the reply can only have
    # come from the corpus.
    assert result["intent"] == "billing_dispute"
    assert result["reply"] == "the near reply"
    assert [vote["conversation_id"] for vote in result["votes"]] == [0, 1, 2, 3, 4]


# --------------------------------------------------------------------------
# leakage
# --------------------------------------------------------------------------


def test_report_leakage_fails_if_a_golden_id_is_retrievable():
    """build_corpus already guarantees this; the baseline re-checks rather than trusts."""
    bank = baselines.GoldenBank(
        conversation_ids=[10],
        texts=["b0"],
        intents=["app_bug"],
        vectors=np.stack([unit(0.0)]),
    )
    corpus, _ = make_corpus([(10, "near", "the near reply", 5.0)])

    with pytest.raises(AssertionError, match="retrievable from the corpus"):
        baselines._report_leakage(bank, corpus, [{"top_similarity": 0.5}])


def test_the_real_golden_set_has_150_unique_ids_and_unique_openings():
    """Both are preconditions for the LOO mask; build_corpus's exclusion is
    tested in tests/test_agent.py and not re-tested here."""
    bank = real_bank_without_vectors()

    assert len(set(bank.conversation_ids)) == 150
    assert len(set(bank.texts)) == 150
