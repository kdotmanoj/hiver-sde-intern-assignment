"""Two non-LLM baselines: the floor the agent has to clear.

A macro-F1 of 0.6 means nothing on its own. These two exist so every number in
the report has something to be compared against, and both are callable exactly
like `agent.answer()` so `evaluate.py` scores all three through one path.

  trivial   always the majority intent, always one canned reply, never escalate
  knn       5-NN majority vote for the intent, the nearest neighbour's actual
            reply verbatim, never escalate

Neither makes an LLM call. Both read only cached embeddings, so they cost no
Gemini or Groq quota and run under CACHE_ONLY=1 from a fresh clone.

Two things about the k-NN baseline that the report has to state plainly:

1. The corpus has no intent labels -- only the 150 golden examples are
   labelled -- so the intent vote has to happen over the golden set itself.
   That is leave-one-out: predicting golden item i uses the other 149. It is
   the correct handling, but it is not an even contest. The agent's classifier
   is zero-shot; this baseline gets 149 labelled examples from the same 150-item
   distribution, hand-labelled in one sitting against one guide. Worse, the
   taxonomy was defined by reading MiniLM clusters, so the intents are partly
   defined to be separable in exactly the space this vote happens in. Pulling
   the other way: ~16 examples per class, and playback_library has 6, which a
   5-vote will under-predict and macro-F1 will punish. Net: treat the k-NN
   intent score as an optimistic ceiling on what a label bank extracts from 150
   labels, not as a deployable system. Its bootstrap CI also understates
   uncertainty, because the 150 predictions share an overlapping vote pool.

2. The reply side has no such problem. It retrieves from the corpus, which
   `agent.build_corpus` already strips of every golden id and asserts empty of
   them (src/agent.py). A golden message cannot retrieve its own reply. What it
   CAN retrieve is a near-duplicate under a different id -- `agent.retrieve`
   notes duplicate openings exist -- so main() reports how many golden queries
   land above NEAR_DUPLICATE_FLOOR rather than leaving that assumed away. The
   agent draws from the same corpus, so that effect is shared and does not tilt
   the comparison.

Neither baseline escalates. escalate=False is 113 of the 150 golden labels, so
"never escalate" IS the majority-class null for the escalation task, matching
what "always the majority intent" is for the intent task. The alternative --
letting the k-NN baseline escalate on a similarity threshold -- would import
src/escalate.py's own rule into its own control group and measure nothing.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.agent import (
    GOLDEN_LABELS,
    INTENTS,
    K,
    SPOTIFY_PARQUET,
    build_corpus,
    corpus_vectors,
    retrieve,
)
from src.embed import embed
from src.escalate import signals_by_family
from src.ingest import _display_path

# The intent vote's k, which is NOT the reply retrieval's k. Kept as separate
# named constants because they answer different questions and will drift apart
# if anyone tunes one of them.
VOTE_K = 5

# The reply is the single nearest neighbour's, verbatim. Averaging or blending
# the top few would produce text no human ever sent, which is the one thing this
# baseline is for: showing what pure copy-the-closest-precedent scores.
REPLY_K = 1

# Only used for the leakage report in main(). Above this, a retrieved neighbour
# is close enough to the query to be a near-duplicate conversation rather than a
# precedent, and the reply score for that row is closer to a lookup than to
# generalization. Reported, not filtered -- the agent sees the same corpus.
NEAR_DUPLICATE_FLOOR = 0.95

# Chosen from the corpus, not invented: "backstage" appears in 5,039 of the
# first replies in data/interim/spotify.parquet and "DM with your account" in
# 910, so this is close to the single most common thing SpotifyCares actually
# says. That is the point. A canned deflection asking for a DM is what a lazy
# production system really would ship, so the trivial baseline should score
# however a generic DM request scores against the judge and against the real
# replies -- and that score is the number the agent has to beat. A deliberately
# bad strawman here would flatter the agent.
CANNED_REPLY = (
    "Hey there! Sorry to hear that. Could you send us a DM with your account's "
    "email address? We'll take a look backstage /SC"
)

# Both baselines hand this to the eval stage in place of escalate()'s reason.
NEVER_ESCALATE_REASON = "baseline: never escalates (the majority-class null)"


# --------------------------------------------------------------------------
# the golden bank: the only labelled data that exists
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class GoldenBank:
    """The 150 hand-labelled examples, plus their cached vectors.

    This is the k-NN baseline's training set AND the evaluation's test set --
    there is no other labelled data. See the module docstring for what that does
    to the comparison. Read-only: nothing here writes to data/golden/.
    """

    conversation_ids: list[int]
    texts: list[str]
    intents: list[str]
    vectors: np.ndarray

    @classmethod
    def load(cls, path: Path = GOLDEN_LABELS) -> GoldenBank:
        with open(path, encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]

        texts = [row["opening_text"] for row in rows]

        # The leave-one-out exclusion below matches on text, so a duplicate
        # opening would let one row vote its own label in under another id.
        # There are none today; fail loudly rather than silently leak if the
        # golden set is ever re-drawn.
        assert len(set(texts)) == len(texts), "duplicate opening_text in the golden set"

        vectors = embed(texts)
        assert len(vectors) == len(rows), f"{len(vectors)} vectors for {len(rows)} labels"

        return cls(
            conversation_ids=[int(row["conversation_id"]) for row in rows],
            texts=texts,
            intents=[row["intent"] for row in rows],
            vectors=vectors,
        )

    def majority_intent(self) -> str:
        """The modal intent. Computed, never hardcoded.

        Ties break on INTENTS order, so this cannot silently flip to a different
        label if the counts move. Currently feature_request, 32 of 150.
        """
        counts = {intent: self.intents.count(intent) for intent in set(self.intents)}
        return min(counts, key=lambda intent: (-counts[intent], INTENTS.index(intent)))


# A process-wide cache so a 150-row run loads and embeds the bank once rather
# than 150 times. Tests pass their own bank explicitly and never touch this.
_BANK: GoldenBank | None = None


def get_bank(path: Path = GOLDEN_LABELS) -> GoldenBank:
    global _BANK
    if _BANK is None:
        _BANK = GoldenBank.load(path)
    return _BANK


# --------------------------------------------------------------------------
# the k-NN intent vote, leave-one-out
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Vote:
    """One neighbour's ballot, kept so a wrong prediction can be audited."""

    conversation_id: int
    similarity: float
    intent: str


def knn_intent(query_text: str, bank: GoldenBank, k: int = VOTE_K) -> tuple[str, list[Vote]]:
    """Majority intent of the k nearest golden examples, excluding the query itself.

    The leave-one-out step is the mask below: any bank row whose opening is
    byte-identical to the query is pushed to -inf and can never be drawn. For a
    golden query that removes exactly one row (openings are asserted unique in
    GoldenBank.load), leaving 149 to vote. For a message from outside the golden
    set it removes none and all 150 vote.

    Written out as an explicit loop rather than a Counter, because the tie-break
    is a decision I have to be able to defend, not a library default.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    query = embed([query_text])[0]
    similarities = bank.vectors @ query

    is_self = np.array([text == query_text for text in bank.texts])
    similarities = np.where(is_self, -np.inf, similarities)

    n_available = len(bank.texts) - int(is_self.sum())
    assert n_available >= k, f"only {n_available} bank rows available for a {k}-vote"

    # Stable sort, matching agent.retrieve: exact similarity ties break on bank
    # order (the labelling order in labels.jsonl) rather than on numpy's
    # partition order, so the same query always gets the same five neighbours.
    order = np.argsort(-similarities, kind="stable")[:k]

    votes = [
        Vote(
            conversation_id=bank.conversation_ids[index],
            similarity=float(similarities[index]),
            intent=bank.intents[index],
        )
        for index in order
    ]

    counts: dict[str, int] = {}
    closest: dict[str, float] = {}
    for vote in votes:
        counts[vote.intent] = counts.get(vote.intent, 0) + 1
        # votes is already similarity-descending, so the first sighting of an
        # intent is its closest member. max() says so without relying on that.
        closest[vote.intent] = max(closest.get(vote.intent, -1.0), vote.similarity)

    # Ties are the common case, not the edge case: five votes over nine classes
    # splits 2-2-1 often. The rule is "the tied class whose single nearest
    # member is closest" -- i.e. fall back from the vote to the one neighbour
    # the embedding is most confident about. INTENTS order is a third key so
    # that even an exact float tie is deterministic, with no RNG anywhere.
    top_count = max(counts.values())
    tied = [intent for intent, count in counts.items() if count == top_count]
    winner = min(tied, key=lambda intent: (-closest[intent], INTENTS.index(intent)))

    return winner, votes


# --------------------------------------------------------------------------
# the two baselines
# --------------------------------------------------------------------------
#
# Both return the dict agent.answer() returns, key for key, so evaluate.py can
# call all three identically. The extra `method` (and `votes`) keys are
# provenance for debugging; the scorer reads the shared keys only.
#
# `force_reply` is accepted and ignored on purpose. It exists on answer() to
# override the escalation gate, and neither baseline has a gate to override --
# they always produce a reply. Dropping the parameter would break the shared
# call signature; honouring it would be a no-op with extra branches.


def _never_escalate() -> dict:
    """The escalation half of both baselines' result, in escalate()'s own shape."""
    return {
        "decision": "auto",
        "reason": NEVER_ESCALATE_REASON,
        "triggered_signals": [],
        # Reused rather than written as {family: [] for ...}: if a fourth family
        # is ever added to src/escalate.py, the baselines pick it up for free
        # instead of quietly reporting three.
        "signals_by_family": signals_by_family([]),
    }


def trivial_answer(
    text: str,
    corpus: pd.DataFrame,
    vectors: np.ndarray,
    k: int = K,
    force_reply: bool = False,
    *,
    bank: GoldenBank | None = None,
) -> dict:
    """Majority intent, canned reply, no retrieval, no escalation.

    `corpus`, `vectors` and `k` are accepted and unused: this baseline retrieves
    nothing, and `retrieved` comes back empty with top_similarity 0.0 rather
    than absent, so the scorer never has to special-case a missing key. They stay
    in the signature because the whole point is that evaluate.py calls all three
    functions with the same arguments.
    """
    bank = bank if bank is not None else get_bank()
    intent = bank.majority_intent()

    return {
        "text": text,
        "intent": intent,
        "parse_ok": True,
        # No LLM output to parse, so there is no raw string. The predicted label
        # itself keeps the key a str, matching answer()'s type.
        "raw_intent": intent,
        "retrieved": [],
        "top_similarity": 0.0,
        **_never_escalate(),
        "reply": CANNED_REPLY,
        "reply_generated": True,
        "method": "trivial",
    }


def knn_answer(
    text: str,
    corpus: pd.DataFrame,
    vectors: np.ndarray,
    k: int = K,
    force_reply: bool = False,
    *,
    bank: GoldenBank | None = None,
    vote_k: int = VOTE_K,
) -> dict:
    """5-NN intent vote over the golden bank; the nearest corpus reply, verbatim.

    Two retrievals over two different pools, which is the part to keep straight:
    the INTENT vote runs over the 150 labelled golden examples (leave-one-out,
    because that is the only labelled data there is), while the REPLY comes from
    the 1,850-row corpus, which has every golden id removed upstream. So the
    reply side cannot leak and the intent side is leave-one-out rather than
    leak-free-by-construction.

    `k` is the reply retrieval depth, passed through to agent.retrieve so the
    returned neighbours match what the agent would have seen. Only the first is
    used for the reply.
    """
    bank = bank if bank is not None else get_bank()

    intent, votes = knn_intent(text, bank, k=vote_k)
    retrieved = retrieve(text, corpus, vectors, k=k)

    # An empty corpus is the only way to get no neighbours. Return no reply
    # rather than crashing, and let reply_generated say so.
    nearest = retrieved[0] if retrieved else None

    return {
        "text": text,
        "intent": intent,
        "parse_ok": True,
        "raw_intent": intent,
        "retrieved": [
            {
                "conversation_id": item.conversation_id,
                "similarity": item.similarity,
                "opening_text": item.opening_text,
                "first_reply_text": item.first_reply_text,
            }
            for item in retrieved
        ],
        "top_similarity": nearest.similarity if nearest else 0.0,
        **_never_escalate(),
        # Verbatim. Not trimmed, not cleaned of the /SC sign-off, not stripped
        # of @mentions -- this baseline is "send what was sent last time", and
        # tidying it up would be doing some of the generator's job for it.
        "reply": nearest.first_reply_text if nearest else None,
        "reply_generated": nearest is not None,
        "method": "knn",
        "votes": [
            {
                "conversation_id": vote.conversation_id,
                "similarity": vote.similarity,
                "intent": vote.intent,
            }
            for vote in votes
        ],
    }


BASELINES = {"trivial": trivial_answer, "knn": knn_answer}


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _report_leakage(bank: GoldenBank, corpus: pd.DataFrame, results: list[dict]) -> None:
    """The two things the report claims about leakage, as measured numbers.

    The first is a guarantee (no golden id is retrievable, enforced in
    build_corpus). The second is not a guarantee and must not be presented as
    one: a near-identical conversation under a different id is still reachable,
    which inflates the reply score for those rows. Count it, don't assume it away.
    """
    overlap = set(bank.conversation_ids) & set(corpus["conversation_id"].tolist())
    assert not overlap, f"{len(overlap)} golden ids are retrievable from the corpus"

    similarities = [result["top_similarity"] for result in results]
    n_near = sum(1 for value in similarities if value >= NEAR_DUPLICATE_FLOOR)

    print(
        f"  leakage:     golden ids in corpus 0 of {len(bank.conversation_ids)} "
        f"(enforced in build_corpus)"
    )
    print(
        f"  near-dupes:  {n_near:,} of {len(results):,} golden queries retrieve a "
        f"neighbour at similarity >= {NEAR_DUPLICATE_FLOOR} "
        f"(max {max(similarities):.3f}) -- shared with the agent, not a baseline artefact"
    )


def _run_golden(
    name: str,
    bank: GoldenBank,
    corpus: pd.DataFrame,
    vectors: np.ndarray,
    k: int,
    out: Path | None,
) -> list[dict]:
    answer_fn = BASELINES[name]

    results = []
    for conversation_id, text in zip(bank.conversation_ids, bank.texts):
        result = answer_fn(text, corpus, vectors, k=k, bank=bank)
        result["conversation_id"] = conversation_id
        results.append(result)

    assert len(results) == len(bank.texts), f"{len(results)} results for {len(bank.texts)} golden"

    n_escalated = sum(1 for r in results if r["decision"] == "escalate")
    n_replied = sum(1 for r in results if r["reply_generated"])
    n_correct = sum(1 for r, truth in zip(results, bank.intents) if r["intent"] == truth)

    print("")
    print(
        f"baselines[{name}]: golden {len(results):,} -> escalated {n_escalated:,} "
        f"-> replied {n_replied:,}"
    )
    # Raw accuracy only, as a smoke signal that the wiring is right. The real
    # numbers -- macro-F1, per-class, kappa, bootstrap CI -- are src/metrics.py's
    # job, and quoting accuracy in the report on a 9-class problem with a 21%
    # majority would be misleading.
    print(f"  intent:      {n_correct:,}/{len(results):,} exact match (smoke signal, not a metric)")

    if name == "knn":
        _report_leakage(bank, corpus, results)

    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as handle:
            for result in results:
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
        print(f"  wrote {_display_path(out)}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--parquet", type=Path, default=SPOTIFY_PARQUET)
    parser.add_argument("--golden-labels", type=Path, default=GOLDEN_LABELS)
    parser.add_argument("--k", type=int, default=K)
    parser.add_argument(
        "--baseline",
        choices=sorted(BASELINES),
        help="run only one (default: both)",
    )
    parser.add_argument("--out-dir", type=Path, help="write {name}_golden.jsonl per baseline")
    args = parser.parse_args()

    corpus = build_corpus(args.parquet, args.golden_labels)
    vectors = corpus_vectors(corpus)
    bank = GoldenBank.load(args.golden_labels)

    print(f"  bank:        {len(bank.texts):,} labelled, majority {bank.majority_intent()}")

    for name in [args.baseline] if args.baseline else sorted(BASELINES):
        out = args.out_dir / f"{name}_golden.jsonl" if args.out_dir else None
        _run_golden(name, bank, corpus, vectors, args.k, out)


if __name__ == "__main__":
    main()
