"""Map reconstructed threads to customer conversations. Stage 3 of the pipeline.

A thread is not a conversation. The largest thread in this dataset is 1,390
tweets: `ATVIAssist` posted once and 22 unrelated customers replied to it.
Treating that as one unit of customer support is wrong in ~20 different ways
at once. This module is the interpretation layer that splits it.

`src/threads.py` is not touched. It owns graph reconstruction — parents,
depths, cycles — and this file is a separate view on top of its output.

The rule, keyed on what each thread is rooted in:

- customer_root                -> the whole thread is one conversation
- brand_root                   -> broadcast; split (below)
- orphan with an inbound root  -> like customer_root  (label orphan_customer)
- orphan with an outbound root -> like brand_root     (label orphan_broadcast)
- cycle                        -> no conversation; counted as unassigned

Orphans carry their own labels rather than being folded into the main two, so
they cannot silently inflate the customer_root count and can be dropped
downstream with one filter.

Splitting a broadcast: a tweet starts a conversation when it is INBOUND and
has NO INBOUND ANCESTOR in the thread. It then carries its subtree.

That rule began as "depth-1 inbound replies only". Measuring what that would
strand found 3,212 inbound tweets at depth >= 2 with no inbound ancestor —
3.6% of all broadcast tweets — sitting under a brand's reply to its own
broadcast. They are not noise; they are complaints with brand replies beneath
them ("I am paying you 40$ a month and I can't even use the service. FIX IT!").
They are promoted to conversation roots under the separate `*_deep` labels, so
the count stays visible and separable. The cost, which the report must state:
broadcast conversation roots are NOT uniformly at depth 1.

What is left unassigned falls out of the rule rather than being special-cased:
tweets on an all-outbound path from a broadcast root — the brand talking to
itself, including the broadcast post. Those are counted and reported, never
dropped.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.ingest import REPO_ROOT, _display_path, load_cached_or_raw, read_parquet, write_parquet
from src.threads import build_threads

THREADS_PARQUET = REPO_ROOT / "data" / "interim" / "threads.parquet"
ASSIGNMENT_PARQUET = REPO_ROOT / "data" / "interim" / "conversation_assignment.parquet"
CONVERSATIONS_PARQUET = REPO_ROOT / "data" / "interim" / "conversations.parquet"

# Which conversation source a thread's members can take, by the thread's root.
# The `_deep` variants are assigned per tweet, not per thread — see _assign().
CUSTOMER_SOURCES = {"customer_root": "customer_root", "orphan": "orphan_customer"}
BROADCAST_SOURCES = {"brand_root": "brand_broadcast", "orphan": "orphan_broadcast"}

SOURCES = (
    "customer_root",
    "brand_broadcast",
    "brand_broadcast_deep",
    "orphan_customer",
    "orphan_broadcast",
    "orphan_broadcast_deep",
)


def _merge_inbound(threads: pd.DataFrame, tweets: pd.DataFrame) -> pd.DataFrame:
    """threads.py does not carry `inbound`, and the split rule needs it."""
    merged = threads.merge(
        tweets[["tweet_id", "inbound"]], on="tweet_id", how="left", validate="one_to_one"
    )
    assert len(merged) == len(threads), (
        f"merge changed row count: {len(threads):,} -> {len(merged):,}"
    )
    assert merged["inbound"].notna().all(), "a tweet in threads has no row in tweets"
    return merged


def assign_conversations(threads: pd.DataFrame, tweets: pd.DataFrame) -> pd.DataFrame:
    """Return one row per tweet with its conversation id, or NA for unassigned.

    Columns: tweet_id, thread_id, parent_id, depth, root_kind, conversation_id,
    conversation_source, depth_in_conversation.
    """
    merged = _merge_inbound(threads, tweets)
    assigned = _assign(merged, _broadcast_threads(merged))
    assert len(assigned) == len(threads), (
        f"assignment changed row count: {len(threads):,} -> {len(assigned):,}"
    )
    return assigned


def _broadcast_threads(merged: pd.DataFrame) -> set[int]:
    """Thread ids whose root is outbound, i.e. a brand post rather than a customer's.

    This is what separates the two rules, and it is the root's `inbound` flag
    rather than its root_kind — an orphan thread is a broadcast or not for
    exactly the same reason a brand_root thread is.
    """
    roots = merged[merged["depth"] == 0]
    return set(roots.loc[~roots["inbound"].astype(bool), "thread_id"].tolist())


def _assign(merged: pd.DataFrame, is_broadcast: set[int]) -> pd.DataFrame:
    """One pass in depth order, assigning each tweet its conversation.

    Depth order is what makes a single pass correct: a tweet's parent always
    has a strictly lower depth, so the parent's conversation is already decided
    by the time the child is reached. Cycle-quarantined tweets have depth -1
    and sort first, which is harmless — they are never assigned.
    """
    ordered = merged.sort_values(["depth", "tweet_id"], kind="stable")

    conversation_of: dict[int, int | None] = {}
    source_of: dict[int, str | None] = {}
    root_depth_of: dict[int, int] = {}  # conversation id -> depth of its root tweet

    for row in ordered.itertuples():
        tweet_id = int(row.tweet_id)
        thread_id = int(row.thread_id)

        if row.depth < 0:  # cycle-quarantined; no root, so no conversation
            conversation_of[tweet_id] = None
            source_of[tweet_id] = None
            continue

        if thread_id not in is_broadcast:
            # Customer-rooted: the thread is the conversation, whole and entire.
            conversation_of[tweet_id] = thread_id
            source_of[tweet_id] = CUSTOMER_SOURCES[
                "orphan" if row.root_kind == "orphan" else "customer_root"
            ]
            continue

        parent_conversation = (
            None if pd.isna(row.parent_id) else conversation_of[int(row.parent_id)]
        )

        if bool(row.inbound) and parent_conversation is None:
            # Inbound with no inbound ancestor: this customer starts a
            # conversation. Depth 1 is the ordinary case; deeper means it sits
            # under the brand replying to its own broadcast, and gets the
            # separate `_deep` label.
            base = BROADCAST_SOURCES["orphan" if row.root_kind == "orphan" else "brand_root"]
            conversation_of[tweet_id] = tweet_id
            source_of[tweet_id] = base if row.depth == 1 else f"{base}_deep"
            root_depth_of[tweet_id] = int(row.depth)
        else:
            # Inherit. A null parent_conversation propagates down, which is
            # how the brand-talking-to-itself path stays unassigned without
            # any special case.
            conversation_of[tweet_id] = parent_conversation
            source_of[tweet_id] = (
                None if parent_conversation is None else source_of[int(row.parent_id)]
            )

    out = merged[["tweet_id", "thread_id", "parent_id", "depth", "root_kind"]].copy()
    ids = out["tweet_id"].tolist()
    out["conversation_id"] = pd.array([conversation_of[t] for t in ids], dtype="Int64")
    out["conversation_source"] = [source_of[t] for t in ids]
    out["depth_in_conversation"] = pd.array(
        [
            None
            if conversation_of[t] is None
            else depth - root_depth_of.get(conversation_of[t], 0)
            for t, depth in zip(ids, out["depth"].tolist())
        ],
        dtype="Int64",
    )
    return out


# --------------------------------------------------------------------------
# per-conversation summary
# --------------------------------------------------------------------------


def summarize_conversations(assigned: pd.DataFrame, tweets: pd.DataFrame) -> pd.DataFrame:
    """One row per conversation: size, customer/brand split, and time span."""
    joined = assigned.merge(
        tweets[["tweet_id", "inbound", "created_at"]],
        on="tweet_id",
        how="left",
        validate="one_to_one",
    )
    members = joined[joined["conversation_id"].notna()]

    # dropna=False on the groupby key is unnecessary here (nulls are filtered
    # above) but stated anyway so a future change cannot silently drop a group.
    grouped = members.groupby("conversation_id", dropna=False)
    summary = grouped.agg(
        source=("conversation_source", "first"),
        thread_id=("thread_id", "first"),
        n_tweets=("tweet_id", "size"),
        n_customer_tweets=("inbound", "sum"),
        started_at=("created_at", "min"),
        ended_at=("created_at", "max"),
        max_depth=("depth_in_conversation", "max"),
    ).reset_index()
    summary["n_brand_tweets"] = summary["n_tweets"] - summary["n_customer_tweets"]

    assert summary["conversation_id"].is_unique, "conversation ids are not unique"
    # Conversation ids come from two namespaces — thread_id for customer-rooted
    # conversations, tweet_id for broadcast roots. Both are tweet ids (a
    # thread_id IS its root's tweet_id), and tweet_id is unique, so they cannot
    # collide. Asserted rather than assumed, so a future change cannot quietly
    # break it.
    assert summary["conversation_id"].isin(set(tweets["tweet_id"])).all(), (
        "a conversation_id is not an existing tweet_id; the two id namespaces "
        "have diverged"
    )
    assert (
        summary["n_customer_tweets"] + summary["n_brand_tweets"] == summary["n_tweets"]
    ).all(), "customer + brand tweet counts do not sum to the conversation size"
    assert int(summary["n_tweets"].sum()) == len(members), (
        "per-conversation sizes do not sum back to the number of assigned tweets"
    )
    return summary


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------


def _size_distribution(sizes: pd.Series) -> str:
    """min / quartiles / tail percentiles / max / mean, on one line."""
    if sizes.empty:
        return "(none)"
    q = sizes.quantile([0.25, 0.5, 0.75, 0.9, 0.99])
    return (
        f"min {int(sizes.min()):>3}  p25 {int(q[0.25]):>3}  med {int(q[0.5]):>3}  "
        f"p75 {int(q[0.75]):>3}  p90 {int(q[0.9]):>3}  p99 {int(q[0.99]):>4}  "
        f"max {int(sizes.max()):>5}  mean {sizes.mean():.1f}"
    )


def _print_summary(
    assigned: pd.DataFrame, summary: pd.DataFrame, is_broadcast: set[int]
) -> None:
    n_total = len(assigned)
    n_assigned = int(assigned["conversation_id"].notna().sum())
    n_unassigned = n_total - n_assigned

    by_source = summary.groupby("source", dropna=False).size()
    broadcast_sources = [s for s in SOURCES if "broadcast" in s]
    deep_sources = [s for s in SOURCES if s.endswith("_deep")]

    unassigned = assigned[assigned["conversation_id"].isna()]
    n_cycle = int((unassigned["depth"] < 0).sum())
    # An unassigned tweet at depth 0 is a broadcast post itself.
    n_broadcast_roots = int((unassigned["depth"] == 0).sum())
    n_broadcast_tweets = int(assigned["thread_id"].isin(is_broadcast).sum())
    n_deep_convs = int(by_source.reindex(deep_sources).fillna(0).sum())
    n_deep_tweets = int(assigned["conversation_source"].isin(deep_sources).sum())
    deep_pct = (100.0 * n_deep_tweets / n_broadcast_tweets) if n_broadcast_tweets else 0.0

    customer_sizes = summary.loc[~summary["source"].isin(broadcast_sources), "n_tweets"]
    broadcast_sizes = summary.loc[summary["source"].isin(broadcast_sources), "n_tweets"]

    print(
        f"conversations: tweets {n_total:,} -> {len(summary):,} conversations, "
        f"assigned {n_assigned:,} tweets (lost {n_total - n_assigned - n_unassigned:,}, "
        f"unassigned {n_unassigned:,})"
    )
    print(
        "  by source:  "
        + " | ".join(f"{s} {int(by_source.get(s, 0)):,}" for s in SOURCES[:3])
    )
    print(
        "              "
        + " | ".join(f"{s} {int(by_source.get(s, 0)):,}" for s in SOURCES[3:])
    )
    print(
        f"  unassigned: {n_unassigned - n_cycle:,} tweets on an all-outbound path from a "
        f"broadcast root (of which {n_broadcast_roots:,} are the broadcast posts "
        f"themselves); {n_cycle:,} cycle-quarantined"
    )
    print(
        f"  deep roots: {n_deep_convs:,} conversations rooted below depth 1, covering "
        f"{n_deep_tweets:,} tweets ({deep_pct:.1f}% of {n_broadcast_tweets:,} broadcast tweets)"
    )
    print("  size (tweets/conversation):")
    print(f"    customer-rooted  {_size_distribution(customer_sizes)}")
    print(f"    broadcast        {_size_distribution(broadcast_sizes)}")


def build_conversations(
    threads: pd.DataFrame, tweets: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign, summarize, and report. Returns (per-tweet, per-conversation)."""
    # Merged once here rather than calling assign_conversations(), which would
    # repeat the merge over 2.8M rows just to recompute is_broadcast.
    merged = _merge_inbound(threads, tweets)
    is_broadcast = _broadcast_threads(merged)

    assigned = _assign(merged, is_broadcast)
    assert len(assigned) == len(threads), (
        f"assignment changed row count: {len(threads):,} -> {len(assigned):,}"
    )
    summary = summarize_conversations(assigned, tweets)
    _print_summary(assigned, summary, is_broadcast)
    return assigned, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--threads", type=Path, default=THREADS_PARQUET)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    tweets = load_cached_or_raw()
    if args.threads.exists():
        threads = read_parquet(args.threads)
        print(f"threads: loaded {len(threads):,} rows from cached {args.threads.name}")
    else:
        threads = build_threads(tweets)

    assigned, summary = build_conversations(threads, tweets)

    if not args.no_write:
        for frame, path in ((assigned, ASSIGNMENT_PARQUET), (summary, CONVERSATIONS_PARQUET)):
            out = write_parquet(frame, path)
            print(f"  wrote {_display_path(out)} ({len(frame):,} rows)")


if __name__ == "__main__":
    main()
