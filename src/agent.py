"""Stage 6: the agent. Classify an opening message, retrieve precedent, draft a reply.

Three parts and a composer:

  classify(text)                  -> one of the nine intents from the taxonomy
  retrieve(text, corpus, vectors) -> the k most similar past conversations
  reply(text, intent, retrieved)  -> a draft grounded in what Spotify actually said
  answer(...)                     -> all three, plus the escalation decision

The retrieval corpus is the same stratified 2,000 openings src/taxonomy.py
clustered, minus the 150 golden ids and any conversation whose first reply is an
orphan_part. Reusing that draw rather than taking a new one is the whole reason
this stage needs no new embeddings: every one of those vectors is already in
data/cache/embeddings/ and already committed, so the stage runs offline from a
fresh clone. The alternative -- retrieving over all ~27,400 eligible
conversations -- would mean committing ~27k new .npy blobs (~45MB) purely to
keep `make eval` key-free. The cost of the choice is a thinner retrieval pool,
and it belongs in the report's limitations rather than being quietly enjoyed.

Excluding the golden ids is not hygiene, it is the experiment: leaving them in
would let a golden message retrieve itself, and every reply metric would then be
measuring a lookup table.

Escalation lives in src/escalate.py, not here. It is one of the modules CLAUDE.md
requires to be plain explicit code, and keeping it out of this file keeps the
rule separable from the LLM plumbing.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src import llm
from src.embed import cache_status, embed
from src.escalate import escalate, signals_by_family
from src.ingest import REPO_ROOT, _display_path, read_parquet
from src.taxonomy import SEED, draw_sample, load_eligible

INTERIM = REPO_ROOT / "data" / "interim"
SPOTIFY_PARQUET = INTERIM / "spotify.parquet"
GOLDEN_LABELS = REPO_ROOT / "data" / "golden" / "labels.jsonl"

# Fixed by CLAUDE.md's model stack. The classifier and the generator are the
# same model on purpose -- they are both "the system under test". The judge is a
# different family (Groq openai/gpt-oss-120b) and lives in the eval stage, which
# is where self-preference bias would actually bite.
#
# gemini-3.5-flash-lite, not gemini-3.6-flash: the free tier gives 3.6 Flash
# only 20 requests PER DAY, which a single golden run exceeds in its first
# fifteen conversations. 3.5-flash-lite allows 500/day at 15 RPM. A full cold
# golden run is 300 calls, so it fits with room to spare.
CLASSIFIER_MODEL = "gemini-3.5-flash-lite"
GENERATOR_MODEL = "gemini-3.5-flash-lite"

# Retrieved neighbours used to ground a reply. Three is enough to show a pattern
# without letting one outlier dominate, and keeps the prompt small enough to
# audit turn by turn.
K = 3

# The nine intents, verbatim from data/golden/labelling_guide.md. This module
# does not name or define the taxonomy -- it consumes it. Order is the guide's.
INTENTS = (
    "playback_library",
    "app_bug",
    "content_issue",
    "account_access",
    "billing_dispute",
    "subscription_query",
    "how_to_question",
    "feature_request",
    "other",
)

# One line each, condensed from the guide's section headers. Condensed, not
# rewritten: if these ever disagree with the guide, the guide wins and these are
# the bug.
INTENT_GLOSS = {
    "playback_library": (
        "Content the user expects is missing, wrong, or will not play on their own "
        "account or device, while Spotify's catalogue itself is fine."
    ),
    "app_bug": (
        "The app misbehaves regardless of content: crashes, freezes, blank screens, "
        "failed installs, device-specific breakage, platform outages."
    ),
    "content_issue": (
        "Spotify's catalogue is wrong for everyone: a track or album removed, the "
        "wrong version uploaded, a playlist discontinued, a release missing, or a "
        "request to add a specific artist or album."
    ),
    "account_access": (
        "Cannot get into an account, or cannot manage who is on it and what it is "
        "called: login failures, hacked accounts, family plan invites, username "
        "changes, account deletion, wrong country on the account."
    ),
    "billing_dispute": (
        "Money has already been taken wrongly, or they paid and did not get what "
        "they paid for: unexpected or duplicate charges, wrong amount, still seeing "
        "ads on premium, refund requests."
    ),
    "subscription_query": (
        "Questions about getting, paying for or keeping the service where no charge "
        "is disputed: what plans exist, student discounts, how to cancel or switch, "
        "payment methods, gift cards, promo codes."
    ),
    "how_to_question": (
        "Nothing is broken. They want to know how something works or whether it "
        "exists."
    ),
    "feature_request": (
        "They want Spotify to build, change or extend something that does not exist "
        "yet, including country availability and device or platform support."
    ),
    "other": (
        "No actionable support request: non-English, pure abuse with no request, "
        "content-free continuations, jokes, praise, or marketing copy posing as a "
        "customer message."
    ),
}


# --------------------------------------------------------------------------
# corpus
# --------------------------------------------------------------------------


def load_golden_ids(path: Path = GOLDEN_LABELS) -> set[int]:
    """The conversation_ids of the hand-labelled set, which must not be retrievable."""
    with open(path, encoding="utf-8") as handle:
        ids = {json.loads(line)["conversation_id"] for line in handle if line.strip()}
    return ids


def build_corpus(
    path: Path = SPOTIFY_PARQUET,
    golden_path: Path = GOLDEN_LABELS,
    verbose: bool = True,
) -> pd.DataFrame:
    """The retrievable past conversations: the taxonomy sample minus leakage.

    Reuses src.taxonomy's load_eligible and draw_sample unchanged, so the 2,000
    drawn here are byte-identical to the 2,000 that were clustered and every
    vector is already cached.
    """
    eligible, n_all = load_eligible(path)
    sampled = draw_sample(eligible)
    n_sampled = len(sampled)

    golden_ids = load_golden_ids(golden_path)

    # Two named masks rather than a chained filter, so each exclusion can be
    # counted and reported on its own line. They overlap (a golden conversation
    # may also have an orphan first reply), which is why the two counts below
    # will not generally sum to the total lost.
    is_golden = sampled["conversation_id"].isin(golden_ids)
    is_orphan = sampled["first_reply_orphan_part"].fillna(False)

    corpus = sampled[~(is_golden | is_orphan)].reset_index(drop=True)

    # load_eligible already filtered to notna(); assert rather than re-filter so
    # that a change in that function's contract fails loudly here.
    assert corpus["first_reply_text"].notna().all(), "a corpus row has no first reply"
    assert corpus["conversation_id"].is_unique, "duplicate conversation_id in corpus"
    assert not corpus["conversation_id"].isin(golden_ids).any(), "golden id survived the filter"

    if verbose:
        _print_corpus_summary(n_all, eligible, n_sampled, is_golden, is_orphan, corpus, golden_ids)
    return corpus


def _print_corpus_summary(
    n_all: int,
    eligible: pd.DataFrame,
    n_sampled: int,
    is_golden: pd.Series,
    is_orphan: pd.Series,
    corpus: pd.DataFrame,
    golden_ids: set[int],
) -> None:
    """The stage row-count row, plus what the exclusions cost."""
    n_golden = int(is_golden.sum())
    n_orphan = int(is_orphan.sum())
    n_both = int((is_golden & is_orphan).sum())

    print(
        f"agent: conversations {n_all:,} -> with reply {len(eligible):,} "
        f"(lost {n_all - len(eligible):,}) -> sampled {n_sampled:,} -> "
        f"corpus {len(corpus):,} (lost {n_sampled - len(corpus):,})"
    )
    print(
        f"  excluded:    {n_golden:,} golden of {len(golden_ids):,} labelled "
        f"({len(golden_ids) - n_golden:,} of the golden set were never in the sample), "
        f"{n_orphan:,} orphan first reply, {n_both:,} both"
    )

    cached, uncached = cache_status(corpus["opening_text"].tolist())
    print(
        f"  embeddings:  {cached + uncached:,} unique texts "
        f"({cached:,} cached, {uncached:,} to compute)"
    )


def corpus_vectors(corpus: pd.DataFrame) -> np.ndarray:
    """Embed the corpus openings. A pure cache read when the cache is warm."""
    vectors = embed(corpus["opening_text"].tolist())
    assert len(vectors) == len(corpus), f"{len(vectors)} vectors for {len(corpus)} rows"
    return vectors


# --------------------------------------------------------------------------
# retrieve
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Retrieved:
    """One neighbour: what they asked, what Spotify replied, and how close it is."""

    conversation_id: int
    similarity: float
    opening_text: str
    first_reply_text: str


def retrieve(
    query_text: str,
    corpus: pd.DataFrame,
    vectors: np.ndarray,
    k: int = K,
) -> list[Retrieved]:
    """The k most similar corpus openings, most similar first. Brute force.

    Brute force on purpose: ~1,850 x 384 is one matrix-vector product, so an
    index would add a dependency and an approximation in exchange for
    microseconds.

    src.embed L2-normalizes every vector at encode time (see embed.NORMALIZE),
    so the dot product below IS cosine similarity. Nothing is renormalized here
    -- doing it again would be a silent no-op that implies the guarantee is not
    trusted.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    query = embed([query_text])[0]
    similarities = vectors @ query

    # Stable sort so that exact ties break on corpus order (conversation_id
    # ascending, inherited from draw_sample) rather than on numpy's partition
    # order. Ties are real here: duplicate openings exist in the corpus.
    order = np.argsort(-similarities, kind="stable")[:k]

    return [
        Retrieved(
            conversation_id=int(corpus.at[index, "conversation_id"]),
            similarity=float(similarities[index]),
            opening_text=str(corpus.at[index, "opening_text"]),
            first_reply_text=str(corpus.at[index, "first_reply_text"]),
        )
        for index in order
    ]


# --------------------------------------------------------------------------
# classify
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Classification:
    """The parsed intent, plus enough to audit a parse that went wrong."""

    intent: str
    parse_ok: bool
    raw: str


def _classify_system_prompt() -> str:
    lines = [
        "You label customer support messages sent to Spotify on Twitter.",
        "",
        "Label what the customer WANTS DONE, not what they are talking about.",
        "Two messages that would get the same kind of reply have the same label.",
        "",
        "The nine labels:",
        "",
    ]
    lines += [f"- {intent}: {INTENT_GLOSS[intent]}" for intent in INTENTS]
    lines += [
        "",
        "Rules:",
        "- If the message has several intents, label the one they LEAD WITH.",
        "- Anger is not an intent. Rage with a real complaint underneath gets the",
        "  complaint's label. Rage with no request gets other.",
        "- Vagueness is not other. A problem report with no detail is still that",
        "  problem. Use other only when there is NO request at all.",
        "- A message asking how to fix something broken takes the broken thing's",
        "  label, not how_to_question.",
        "",
        "Reply with exactly one label from the list and nothing else.",
    ]
    return "\n".join(lines)


def _classify_prompt(text: str) -> str:
    return f"Message:\n{text}\n\nLabel:"


def parse_intent(raw: str) -> Classification:
    """Map a model response to one of the nine labels.

    Deliberately strict, then deliberately loud. An unrecognised response falls
    back to `other` so the pipeline keeps running, but carries parse_ok=False and
    the raw string, so the eval stage reports parse failures as their own number
    instead of absorbing them into the `other` class and flattering the model.
    """
    cleaned = raw.strip().strip("`\"'.").strip().lower().replace(" ", "_").replace("-", "_")

    if cleaned in INTENTS:
        return Classification(intent=cleaned, parse_ok=True, raw=raw)

    # A model that answers "Label: billing_dispute" or "the intent is app_bug"
    # has still answered. Accept it only when exactly one label is present, so
    # a response naming two labels is a parse failure rather than a coin flip.
    found = [intent for intent in INTENTS if intent in cleaned]
    if len(found) == 1:
        return Classification(intent=found[0], parse_ok=True, raw=raw)

    return Classification(intent="other", parse_ok=False, raw=raw)


def classify(text: str, model: str = CLASSIFIER_MODEL) -> Classification:
    """Classify one opening message into one of the nine intents."""
    raw = llm.complete(
        prompt=_classify_prompt(text),
        model=model,
        system=_classify_system_prompt(),
    )
    return parse_intent(raw)


# --------------------------------------------------------------------------
# reply
# --------------------------------------------------------------------------

REPLY_SYSTEM_PROMPT = "\n".join(
    [
        "You write replies for Spotify's customer support account on Twitter.",
        "",
        "You are shown past conversations: a customer message, and the reply Spotify",
        "actually sent. Write a reply to the new message in the same voice and, more",
        "importantly, within the same SCOPE.",
        "",
        "Rules:",
        "- Never state account-specific facts. You cannot see their charges, their",
        "  subscription state, their listening history or who is on their plan.",
        "- Never promise anything the past replies do not promise. If none of them",
        "  commit to a fix or a date, you do not either.",
        "- If the past replies ask the customer for information, ask for it too.",
        "- Match their length. These are tweets, not emails. Two or three sentences.",
        "- Write only the reply. No preamble, no quotation marks, no signature.",
    ]
)


def _reply_prompt(text: str, intent: str, retrieved: list[Retrieved]) -> str:
    blocks = [
        "\n".join(
            [
                f"--- past conversation {n} (similarity {item.similarity:.3f}) ---",
                f"Customer: {item.opening_text}",
                f"Spotify replied: {item.first_reply_text}",
            ]
        )
        for n, item in enumerate(retrieved, start=1)
    ]
    precedent = "\n\n".join(blocks) if blocks else "(no similar past conversation was found)"

    return "\n".join(
        [
            precedent,
            "",
            f"--- new message (classified as {intent}) ---",
            f"Customer: {text}",
            "",
            "Your reply:",
        ]
    )


def reply(
    text: str,
    intent: str,
    retrieved: list[Retrieved],
    model: str = GENERATOR_MODEL,
) -> str:
    """Draft a reply grounded in the retrieved past replies."""
    raw = llm.complete(
        prompt=_reply_prompt(text, intent, retrieved),
        model=model,
        system=REPLY_SYSTEM_PROMPT,
    )
    return raw.strip()


# --------------------------------------------------------------------------
# compose
# --------------------------------------------------------------------------


def answer(
    text: str,
    corpus: pd.DataFrame,
    vectors: np.ndarray,
    k: int = K,
    force_reply: bool = False,
) -> dict:
    """classify -> retrieve -> escalate -> (conditionally) reply.

    By default an escalated ticket gets no draft: `reply` is None and
    `reply_generated` is False. That saves generator quota on exactly the cases
    where the draft would be discarded, and it makes the escalation decision
    load-bearing instead of decorative.

    force_reply=True generates regardless. Evaluation needs a reply for all 150
    goldens to score the judge -- scoring only the auto-handled subset would make
    the judge's numbers conditional on the escalation rule being correct, which
    is one of the things under test. The decision itself is computed and recorded
    in full either way; force_reply changes only whether the reply call happens.
    """
    classification = classify(text)
    retrieved = retrieve(text, corpus, vectors, k=k)

    # No neighbours can only happen with an empty corpus. Treat it as maximal
    # distance rather than crashing, so the similarity signal fires and a human
    # picks it up.
    top_similarity = retrieved[0].similarity if retrieved else 0.0

    decision = escalate(
        intent=classification.intent,
        top_similarity=top_similarity,
        text=text,
    )

    should_reply = force_reply or decision["decision"] == "auto"
    draft = reply(text, classification.intent, retrieved) if should_reply else None

    return {
        "text": text,
        "intent": classification.intent,
        "parse_ok": classification.parse_ok,
        "raw_intent": classification.raw,
        "retrieved": [asdict(item) for item in retrieved],
        "top_similarity": top_similarity,
        "decision": decision["decision"],
        "reason": decision["reason"],
        "triggered_signals": decision["triggered_signals"],
        "signals_by_family": signals_by_family(decision["triggered_signals"]),
        "reply": draft,
        "reply_generated": draft is not None,
    }


# --------------------------------------------------------------------------
# preflight: count the live calls before spending any of them
# --------------------------------------------------------------------------
#
# The free tier is a daily budget, and a run that discovers the wall at call 21
# of 300 has burned the day for nothing: the calls are spent, the results are
# partial, and the cache is half-written. So the golden path counts what it is
# about to spend first, and refuses to start if the number is larger than it is
# allowed to be.

# A full cold golden run is 150 classify + 150 reply = 300 calls. The 500/day
# limit on gemini-3.5-flash-lite therefore permits exactly one full run per day
# with 200 to spare. 320 allows that one run plus a little slack for retries,
# and refuses anything bigger -- which at this point in the project could only
# be a bug or a much larger batch than intended. Lower it to make the guard
# tighter; raise it only with the daily quota in hand.
MAX_LIVE_CALLS = 320


class QuotaError(RuntimeError):
    """A planned run needs more live calls than MAX_LIVE_CALLS allows."""


def preflight(
    golden: list[dict],
    corpus: pd.DataFrame,
    vectors: np.ndarray,
    k: int = K,
    force_reply: bool = False,
) -> dict:
    """Count the uncached LLM calls a golden run would need. Makes no calls itself.

    Exact when the classify cache is warm, and a correct UPPER BOUND when it is
    not. The asymmetry is unavoidable: the reply prompt contains the classified
    intent and the retrieved precedents, so it cannot be constructed -- and
    therefore cannot be looked up -- until the intent is known. When classify is
    a miss, the matching reply is counted as a miss too, which is the safe
    direction to be wrong in.
    """
    system = _classify_system_prompt()

    classify_misses = 0
    reply_misses = 0
    reply_unknown = 0
    skipped_by_escalation = 0

    for row in golden:
        text = row["opening_text"]

        raw = llm.cached_response(_classify_prompt(text), CLASSIFIER_MODEL, system)
        if raw is None:
            # Intent unknown, so the reply prompt cannot be built. Count both.
            classify_misses += 1
            reply_unknown += 1
            continue

        intent = parse_intent(raw).intent
        retrieved = retrieve(text, corpus, vectors, k=k)
        top_similarity = retrieved[0].similarity if retrieved else 0.0
        decision = escalate(intent=intent, top_similarity=top_similarity, text=text)

        if not force_reply and decision["decision"] == "escalate":
            skipped_by_escalation += 1
            continue

        if llm.cached_response(_reply_prompt(text, intent, retrieved), GENERATOR_MODEL, None) is None:
            reply_misses += 1

    total = classify_misses + reply_misses + reply_unknown
    return {
        "n_golden": len(golden),
        "classify_misses": classify_misses,
        "reply_misses": reply_misses,
        "reply_unknown": reply_unknown,
        "skipped_by_escalation": skipped_by_escalation,
        "total": total,
        "exact": reply_unknown == 0,
    }


def _report_preflight(plan: dict, max_live_calls: int) -> None:
    """Print the budget, then raise if the run would exceed it."""
    qualifier = "" if plan["exact"] else " (upper bound)"
    minutes = plan["total"] * llm.GEMINI_MIN_GAP_S / 60.0

    print(
        f"  preflight:   {plan['total']:,} live calls needed{qualifier} for "
        f"{plan['n_golden']:,} golden openings"
    )
    print(
        f"    classify {plan['classify_misses']:,} uncached, "
        f"reply {plan['reply_misses']:,} uncached, "
        f"{plan['reply_unknown']:,} reply calls unknowable until classify runs"
    )
    if plan["skipped_by_escalation"]:
        print(
            f"    {plan['skipped_by_escalation']:,} replies skipped by the escalation "
            f"decision (use --force-reply to generate them)"
        )
    print(
        f"    budget {plan['total']:,}/{max_live_calls:,}, "
        f"~{minutes:.0f} min at {llm.GEMINI_MIN_GAP_S:.0f}s between calls"
    )

    if plan["total"] > max_live_calls:
        raise QuotaError(
            f"This run needs {plan['total']:,} live calls but MAX_LIVE_CALLS is "
            f"{max_live_calls:,}. Nothing has been sent.\n"
            f"  The free tier on {GENERATOR_MODEL} is a DAILY budget: going over it "
            f"does not queue, it fails, and the calls already made are still spent.\n"
            f"  Options: raise --max-live-calls if you know the quota is there, "
            f"run a subset, or warm the cache in batches across days."
        )


# --------------------------------------------------------------------------
# diagnose: the evidence for SIMILARITY_FLOOR
# --------------------------------------------------------------------------

PERCENTILES = (0, 10, 25, 50, 75, 90, 100)
HISTOGRAM_EDGES = (0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01)


def diagnose(corpus: pd.DataFrame, vectors: np.ndarray, golden_path: Path = GOLDEN_LABELS) -> None:
    """Print the top-1 similarity distribution over the golden openings.

    This exists to set escalate.SIMILARITY_FLOOR from evidence rather than from
    taste. It makes no API calls -- embeddings only -- so it runs under
    CACHE_ONLY=1 with no key, which is also what proves the corpus is fully
    cached before any live run is attempted.
    """
    with open(golden_path, encoding="utf-8") as handle:
        golden = [json.loads(line) for line in handle if line.strip()]

    queries = embed([row["opening_text"] for row in golden])
    # (n_golden, n_corpus) similarities in one product; both sides are already
    # L2-normalized, so this is cosine.
    similarities = queries @ vectors.T
    top = similarities.max(axis=1)

    print(f"  top-1 similarity over {len(golden):,} golden openings vs {len(corpus):,} corpus")
    spread = "  ".join(f"p{p}={np.percentile(top, p):.3f}" for p in PERCENTILES)
    print(f"    {spread}")
    print(f"    mean={top.mean():.3f}  std={top.std():.3f}")
    print("")

    counts, _ = np.histogram(top, bins=list(HISTOGRAM_EDGES))
    for low, high, count in zip(HISTOGRAM_EDGES, HISTOGRAM_EDGES[1:], counts):
        share = 100.0 * count / len(top)
        bar = "#" * int(round(share / 2))
        print(f"    [{low:.2f}, {high:.2f})  {count:>4}  {share:>5.1f}%  {bar}")
    print("")
    print("  Escalation rate if SIMILARITY_FLOOR were set to each edge:")
    for edge in HISTOGRAM_EDGES[1:-1]:
        fired = int((top < edge).sum())
        print(f"    floor {edge:.2f} -> {fired:>4} of {len(top):,} ({100.0 * fired / len(top):.1f}%)")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _print_answer(result: dict) -> None:
    print("")
    print(f"  message:   {' '.join(result['text'].split())}")
    print(f"  intent:    {result['intent']}" + ("" if result["parse_ok"] else "  (PARSE FAILED)"))
    print(f"  decision:  {result['decision']} -- {result['reason']}")
    if result["triggered_signals"]:
        print(f"  signals:   {', '.join(result['triggered_signals'])}")
        for family, signals in result["signals_by_family"].items():
            print(f"    {family:<11} {', '.join(signals) if signals else '-'}")
    print(f"  retrieved:")
    for item in result["retrieved"]:
        opening = " ".join(item["opening_text"].split())
        print(f"    {item['similarity']:.3f}  [{item['conversation_id']}]  {opening[:80]}")
    if result["reply_generated"]:
        print(f"  reply:     {' '.join(result['reply'].split())}")
    else:
        print("  reply:     (not generated -- escalated; use --force-reply to generate anyway)")


def _run_golden(
    corpus: pd.DataFrame,
    vectors: np.ndarray,
    force_reply: bool,
    out: Path | None,
    golden_path: Path = GOLDEN_LABELS,
    k: int = K,
    max_live_calls: int = MAX_LIVE_CALLS,
) -> None:
    with open(golden_path, encoding="utf-8") as handle:
        golden = [json.loads(line) for line in handle if line.strip()]

    # Count the spend before making any of it. Raises QuotaError rather than
    # starting a run that cannot finish.
    plan = preflight(golden, corpus, vectors, k=k, force_reply=force_reply)
    _report_preflight(plan, max_live_calls)

    results = []
    for n, row in enumerate(golden, start=1):
        result = answer(row["opening_text"], corpus, vectors, force_reply=force_reply)
        result["conversation_id"] = row["conversation_id"]
        results.append(result)
        print(f"  [{n:>3}/{len(golden)}] {result['intent']:<19} {result['decision']}")

    n_escalated = sum(1 for r in results if r["decision"] == "escalate")
    n_parse_failed = sum(1 for r in results if not r["parse_ok"])
    n_replied = sum(1 for r in results if r["reply_generated"])
    print("")
    print(
        f"  golden {len(results):,} -> escalated {n_escalated:,} "
        f"({100.0 * n_escalated / len(results):.1f}%) -> replied {n_replied:,}"
        + ("  [--force-reply]" if force_reply else "")
    )
    print(f"  intent parse failures: {n_parse_failed:,}")

    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as handle:
            for result in results:
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
        print(f"  wrote {_display_path(out)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--parquet", type=Path, default=SPOTIFY_PARQUET)
    parser.add_argument("--golden-labels", type=Path, default=GOLDEN_LABELS)
    parser.add_argument("--k", type=int, default=K)

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--text", help="run the agent on one message")
    mode.add_argument("--golden", action="store_true", help="run all 150 golden openings")
    mode.add_argument(
        "--diagnose",
        action="store_true",
        help="print the top-1 similarity distribution (no LLM calls) and exit",
    )

    parser.add_argument(
        "--force-reply",
        action="store_true",
        help="generate a reply even when the decision is escalate (use for the golden run)",
    )
    parser.add_argument("--out", type=Path, help="write golden results as jsonl")
    parser.add_argument(
        "--max-live-calls",
        type=int,
        default=MAX_LIVE_CALLS,
        help=f"abort --golden if it would need more live calls than this (default {MAX_LIVE_CALLS})",
    )
    args = parser.parse_args()

    corpus = build_corpus(args.parquet, args.golden_labels)
    vectors = corpus_vectors(corpus)

    if args.diagnose:
        diagnose(corpus, vectors, args.golden_labels)
    elif args.golden:
        try:
            _run_golden(
                corpus,
                vectors,
                args.force_reply,
                args.out,
                args.golden_labels,
                k=args.k,
                max_live_calls=args.max_live_calls,
            )
        except QuotaError as error:
            # A budget refusal is an expected outcome, not a crash. A traceback
            # here would bury the one thing worth reading.
            print(f"\nABORTED: {error}", file=sys.stderr)
            raise SystemExit(1)
    else:
        _print_answer(answer(args.text, corpus, vectors, k=args.k, force_reply=args.force_reply))


if __name__ == "__main__":
    main()
