"""Tests for the thread -> conversation mapping.

Each fixture is a thread small enough to trace by hand, run through the real
build_threads() first so the tests exercise the same depths and parent links
production does rather than a hand-faked thread frame.

The case that matters most is test_deep_inbound_reply_is_promoted: it pins the
3,212-tweet finding — a customer complaint sitting under a brand's reply to
its own broadcast — to a specific, named behaviour.
"""

import pandas as pd
import pytest

from src.conversations import assign_conversations, build_conversations, summarize_conversations
from src.threads import build_threads
from tests.test_threads import tweets


def convert(rows):
    """Build tweets + threads + the conversation assignment from raw tuples."""
    tw = tweets(rows)
    threads = build_threads(tw)
    return tw, assign_conversations(threads, tw)


def by_id(assigned):
    return {int(r.tweet_id): r for r in assigned.itertuples()}


# --------------------------------------------------------------------------
# 1. customer-rooted threads
# --------------------------------------------------------------------------


def test_customer_rooted_thread_is_exactly_one_conversation():
    """Branching does not split a customer-rooted thread.

        10 customer (root)
        |-- 12 brand
        |   `-- 13 customer
        `-- 11 brand
    """
    _, out = convert(
        [(10, None, True, 0), (11, 10, False, 2), (12, 10, False, 1), (13, 12, True, 3)]
    )

    assert out["conversation_id"].nunique() == 1
    assert set(out["conversation_id"]) == {10}, "keyed on the root tweet"
    assert set(out["conversation_source"]) == {"customer_root"}
    assert out["conversation_id"].notna().all(), "nothing in a customer thread is unassigned"

    rows = by_id(out)
    assert rows[10].depth_in_conversation == 0
    assert rows[13].depth_in_conversation == 2


# --------------------------------------------------------------------------
# 2. broadcasts split
# --------------------------------------------------------------------------


def test_broadcast_splits_into_one_conversation_per_customer():
    """A brand posts; two unrelated customers reply; each reply gets an answer.

        20 brand (root, a broadcast)
        |-- 21 customer  -> conversation 21
        |   `-- 22 brand
        `-- 23 customer  -> conversation 23
            `-- 24 brand
    """
    _, out = convert(
        [
            (20, None, False, 0),
            (21, 20, True, 1),
            (22, 21, False, 2),
            (23, 20, True, 3),
            (24, 23, False, 4),
        ]
    )
    rows = by_id(out)

    assert out["conversation_id"].nunique() == 2, "one thread, two conversations"
    assert rows[21].conversation_id == 21 and rows[22].conversation_id == 21
    assert rows[23].conversation_id == 23 and rows[24].conversation_id == 23
    assert rows[21].conversation_source == "brand_broadcast"

    # The broadcast post itself belongs to neither customer's conversation.
    assert pd.isna(rows[20].conversation_id)
    assert pd.isna(rows[20].conversation_source)

    # Depth is measured from the conversation root, not the thread root.
    assert rows[21].depth_in_conversation == 0
    assert rows[22].depth_in_conversation == 1


# --------------------------------------------------------------------------
# 3. the 3,212-tweet case
# --------------------------------------------------------------------------


def test_deep_inbound_reply_is_promoted_with_its_own_label():
    """A customer replying under the brand's reply to its own broadcast.

        30 brand (root, a broadcast)
        `-- 31 brand      (self-reply, depth 1, outbound -> no conversation)
            `-- 32 customer  (depth 2, no inbound ancestor -> promoted)
                `-- 33 brand (depth 3, carried with it)

    Without promotion, 32 and 33 would be silently unassigned. On the full
    dataset that is 3,212 real complaints.
    """
    _, out = convert(
        [(30, None, False, 0), (31, 30, False, 1), (32, 31, True, 2), (33, 32, False, 3)]
    )
    rows = by_id(out)

    assert rows[32].conversation_id == 32, "the deep customer reply starts a conversation"
    assert rows[32].conversation_source == "brand_broadcast_deep", "flagged separately"
    assert rows[33].conversation_id == 32, "and carries its subtree"
    assert rows[33].conversation_source == "brand_broadcast_deep"
    assert rows[32].depth_in_conversation == 0, "depth is relative to the promoted root"
    assert rows[33].depth_in_conversation == 1

    # The all-brand path above it stays unassigned.
    assert pd.isna(rows[30].conversation_id)
    assert pd.isna(rows[31].conversation_id)


def test_depth_one_inbound_is_not_labelled_deep():
    """The `_deep` label must mean 'below depth 1', not 'in a broadcast'."""
    _, out = convert([(30, None, False, 0), (31, 30, True, 1)])
    rows = by_id(out)

    assert rows[31].conversation_source == "brand_broadcast"
    assert not rows[31].conversation_source.endswith("_deep")


def test_customer_reply_under_another_customer_does_not_start_a_conversation():
    """Only a tweet with NO inbound ancestor is a root; otherwise broadcasts
    would shatter into one conversation per customer message.

        30 brand (broadcast)
        `-- 31 customer      -> conversation 31
            `-- 32 brand
                `-- 33 customer  (same customer conversation, NOT a new one)
    """
    _, out = convert(
        [(30, None, False, 0), (31, 30, True, 1), (32, 31, False, 2), (33, 32, True, 3)]
    )
    rows = by_id(out)

    assert rows[33].conversation_id == 31, "an inbound ancestor exists, so it is not a root"
    assert out["conversation_id"].nunique() == 1


# --------------------------------------------------------------------------
# 4. orphans
# --------------------------------------------------------------------------


def test_orphans_are_routed_by_their_roots_inbound_flag_with_own_labels():
    """Orphans follow the same rule but must never be labelled customer_root."""
    # Inbound-rooted orphan: one conversation.
    _, inbound_out = convert([(40, 99, True, 0), (41, 40, False, 1)])
    assert set(inbound_out["conversation_source"]) == {"orphan_customer"}
    assert inbound_out["conversation_id"].nunique() == 1
    assert set(inbound_out["conversation_id"]) == {40}

    # Outbound-rooted orphan: treated as a broadcast and split.
    _, outbound_out = convert([(50, 99, False, 0), (51, 50, True, 1), (52, 50, True, 2)])
    rows = by_id(outbound_out)
    assert outbound_out["conversation_id"].nunique() == 2
    assert rows[51].conversation_source == "orphan_broadcast"
    assert rows[52].conversation_source == "orphan_broadcast"
    assert pd.isna(rows[50].conversation_id)


def test_orphan_labels_never_collide_with_customer_root():
    """The point of the separate labels: one filter drops every orphan."""
    _, out = convert([(10, None, True, 0), (40, 99, True, 1)])
    sources = set(out["conversation_source"])

    assert sources == {"customer_root", "orphan_customer"}
    kept = out[~out["conversation_source"].fillna("").str.startswith("orphan")]
    assert set(kept["tweet_id"]) == {10}


# --------------------------------------------------------------------------
# 5. conservation
# --------------------------------------------------------------------------


def test_every_tweet_is_either_assigned_or_counted_as_unassigned():
    """All the shapes at once. Nothing may be lost between stages."""
    tw, out = convert(
        [
            (10, None, True, 0),  # customer root
            (11, 10, False, 1),
            (20, None, False, 0),  # broadcast
            (21, 20, True, 1),  # -> conversation
            (31, 20, False, 2),  # brand self-reply, unassigned
            (32, 31, True, 3),  # -> promoted, deep
            (40, 99, True, 0),  # inbound orphan
            (50, 51, True, 0),  # cycle
            (51, 50, False, 1),  # cycle
        ]
    )

    assert len(out) == len(tw), "one row per input tweet"
    assert out["tweet_id"].is_unique
    n_assigned = int(out["conversation_id"].notna().sum())
    n_unassigned = int(out["conversation_id"].isna().sum())
    assert n_assigned + n_unassigned == len(tw)

    # Cycle-quarantined tweets have no root, so no conversation.
    rows = by_id(out)
    assert pd.isna(rows[50].conversation_id) and pd.isna(rows[51].conversation_id)

    # Every non-null source is one we declared.
    sources = out["conversation_source"].dropna()
    assert sources.isin(
        [
            "customer_root",
            "brand_broadcast",
            "brand_broadcast_deep",
            "orphan_customer",
            "orphan_broadcast",
            "orphan_broadcast_deep",
        ]
    ).all()


# --------------------------------------------------------------------------
# 6. summary arithmetic
# --------------------------------------------------------------------------


def test_summary_rows_and_counts_reconcile():
    rows = [
        (10, None, True, 0),
        (11, 10, False, 1),
        (12, 10, True, 2),
        (20, None, False, 0),
        (21, 20, True, 1),
        (22, 21, False, 2),
        (23, 20, True, 3),
    ]
    tw, out = convert(rows)
    summary = summarize_conversations(out, tw)

    assert len(summary) == out["conversation_id"].nunique()
    assert (summary["n_customer_tweets"] + summary["n_brand_tweets"] == summary["n_tweets"]).all()
    assert summary["n_tweets"].sum() == int(out["conversation_id"].notna().sum())

    conversation_10 = summary[summary["conversation_id"] == 10].iloc[0]
    assert conversation_10["n_tweets"] == 3
    assert conversation_10["n_customer_tweets"] == 2  # tweets 10 and 12
    assert conversation_10["n_brand_tweets"] == 1  # tweet 11
    assert conversation_10["started_at"] < conversation_10["ended_at"]


def test_conversation_ids_are_unique_across_both_namespaces():
    """Customer-rooted ids come from thread_id, broadcast ids from tweet_id.

    Both are tweet ids, so they cannot collide — this asserts it rather than
    trusting the reasoning to survive a future change.
    """
    tw, out = convert(
        [
            (10, None, True, 0),  # conversation id 10, from thread_id
            (20, None, False, 0),  # broadcast
            (21, 20, True, 1),  # conversation id 21, from tweet_id
        ]
    )
    summary = summarize_conversations(out, tw)

    assert summary["conversation_id"].is_unique
    assert summary["conversation_id"].isin(set(tw["tweet_id"])).all()


def test_build_conversations_returns_both_frames(capsys):
    tw = tweets([(10, None, True, 0), (20, None, False, 0), (21, 20, True, 1)])
    threads = build_threads(tw)
    assigned, summary = build_conversations(threads, tw)

    assert len(assigned) == len(tw)
    assert len(summary) == 2
    printed = capsys.readouterr().out
    assert "conversations: tweets 3" in printed
    assert "by source:" in printed
