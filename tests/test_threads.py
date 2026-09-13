"""Tests for thread reconstruction.

Every fixture is small enough to trace by hand, and every assertion is against
an answer derived on paper rather than against whatever the code happens to
produce. The four structural hazards each get their own test: branching, a
missing parent, a cycle, and a root whose tweet_id is higher than its reply's.
"""

import pandas as pd
import pytest

from src.threads import build_parent_map, build_threads, path_to_root


def tweets(rows):
    """Build an input frame from (tweet_id, parent_id, inbound, minute) tuples.

    `minute` is the offset from a fixed base time, so ordering is explicit in
    each test rather than hidden in a timestamp string.
    """
    base = pd.Timestamp("2017-10-31 22:00:00", tz="UTC")
    return pd.DataFrame(
        {
            "tweet_id": [r[0] for r in rows],
            "author_id": [("cust" if r[2] else "BrandSupport") for r in rows],
            "inbound": [r[2] for r in rows],
            "created_at": [base + pd.Timedelta(minutes=r[3]) for r in rows],
            "text": [f"tweet {r[0]}" for r in rows],
            "response_tweet_id": [None] * len(rows),
            "in_response_to_tweet_id": [r[1] for r in rows],
        }
    )


def rows_by_id(out):
    return {int(r.tweet_id): r for r in out.itertuples()}


# --------------------------------------------------------------------------
# 1. branching
# --------------------------------------------------------------------------


def test_branching_keeps_siblings_in_one_thread_ordered_by_time():
    """Root 10 has two replies, 11 and 12. 12 is a minute EARLIER than 11.

        10 (t+0)
        |-- 12 (t+1)   <- earlier, so branch 0
        |   `-- 13 (t+3)
        `-- 11 (t+2)   <- later, so branch 1

    Sorting by tweet_id would put 11 first. Sorting by created_at puts 12
    first, which is what the assertion below pins down.
    """
    df = tweets([(10, None, True, 0), (11, 10, False, 2), (12, 10, False, 1), (13, 12, True, 3)])
    out = build_threads(df)
    by_id = rows_by_id(out)

    # One thread, one root, four members.
    assert out["thread_id"].nunique() == 1
    assert set(out["thread_id"]) == {10}
    assert all(by_id[t].root_kind == "customer_root" for t in (10, 11, 12, 13))

    assert by_id[10].depth == 0
    assert by_id[11].depth == 1
    assert by_id[12].depth == 1
    assert by_id[13].depth == 2

    # 12 is earlier than 11, so 12's path is explored first and is branch 0.
    assert by_id[12].branch_id == 0
    assert by_id[13].branch_id == 0
    assert by_id[11].branch_id == 1

    assert by_id[10].n_children == 2
    assert by_id[11].is_leaf and by_id[13].is_leaf
    assert not by_id[12].is_leaf


def test_branch_id_is_not_a_dialogue_filter():
    """The documented trap: filtering by branch_id loses the shared ancestors.

    Same fixture as above. Branch 1 is just tweet 11 — the root 10 is on
    branch 0 — so a branch_id filter gives a dialogue with no opening message.
    path_to_root is the thing that gets it right.
    """
    df = tweets([(10, None, True, 0), (11, 10, False, 2), (12, 10, False, 1), (13, 12, True, 3)])
    out = build_threads(df)

    branch_one = set(out.loc[out["branch_id"] == 1, "tweet_id"])
    assert branch_one == {11}, "branch 1 excludes the root, which is the whole point"
    assert path_to_root(out, 11) == [10, 11], "path_to_root keeps the shared ancestor"


def test_path_to_root_on_both_branches():
    df = tweets([(10, None, True, 0), (11, 10, False, 2), (12, 10, False, 1), (13, 12, True, 3)])
    out = build_threads(df)
    parent_map = build_parent_map(out)

    left = path_to_root(out, 13, parent_map)
    right = path_to_root(out, 11, parent_map)

    assert left == [10, 12, 13]  # root-first
    assert right == [10, 11]
    assert left[0] == right[0] == 10, "both dialogues share the opening message"
    assert left[1:] != right[1:], "and diverge after it"


# --------------------------------------------------------------------------
# 2. missing parent
# --------------------------------------------------------------------------


def test_missing_parent_becomes_a_flagged_orphan_root():
    """Tweet 21 replies to 99, which is not in the file.

    21 must survive as its own thread, labelled `orphan` rather than
    `customer_root`: it looks like an opening message but is a fragment.
    """
    df = tweets([(20, None, True, 0), (21, 99, True, 1), (22, 21, False, 2)])
    out = build_threads(df)
    by_id = rows_by_id(out)

    assert len(out) == 3, "the orphan must not be dropped"
    assert by_id[21].root_kind == "orphan"
    assert by_id[21].depth == 0
    assert pd.isna(by_id[21].parent_id), "the dangling parent reference is cleared, not kept"

    # Its own reply comes with it, inheriting the orphan label.
    assert by_id[22].thread_id == 21
    assert by_id[22].root_kind == "orphan"
    assert by_id[22].depth == 1

    # And it is a separate thread from the well-formed one.
    assert by_id[20].thread_id == 20
    assert by_id[20].root_kind == "customer_root"


def test_brand_root_is_distinguished_from_customer_root():
    """A null parent on an outbound tweet is not a customer-initiated thread."""
    df = tweets([(30, None, True, 0), (31, None, False, 1)])
    out = build_threads(df)
    by_id = rows_by_id(out)

    assert by_id[30].root_kind == "customer_root"
    assert by_id[31].root_kind == "brand_root"


# --------------------------------------------------------------------------
# 3. cycle
# --------------------------------------------------------------------------


def test_cycle_is_quarantined_not_dropped_and_terminates():
    """40 -> 41 -> 40 is a parent cycle, with 42 hanging off it.

    No root can reach any of them. All three must still appear in the output,
    labelled `cycle`, with depth -1 because they have no root to measure from.
    """
    df = tweets([(40, 41, True, 0), (41, 40, False, 1), (42, 41, True, 2), (43, None, True, 3)])
    out = build_threads(df)
    by_id = rows_by_id(out)

    assert len(out) == 4, "cycle members must not be dropped"
    for tweet_id in (40, 41, 42):
        assert by_id[tweet_id].root_kind == "cycle"
        assert by_id[tweet_id].depth == -1
    # Named after the smallest id on the cycle, so the name is deterministic.
    assert by_id[40].thread_id == 40
    assert by_id[41].thread_id == 40
    assert by_id[42].thread_id == 40, "a tweet hanging off the cycle is quarantined with it"

    # The unrelated well-formed thread is untouched.
    assert by_id[43].root_kind == "customer_root"


def test_path_to_root_terminates_inside_a_cycle():
    df = tweets([(40, 41, True, 0), (41, 40, False, 1), (42, 41, True, 2)])
    out = build_threads(df)

    chain = path_to_root(out, 42)
    assert chain[-1] == 42
    assert len(chain) == len(set(chain)), "the walk must not repeat a tweet"
    assert len(chain) <= 3


# --------------------------------------------------------------------------
# 4. root id higher than its reply's
# --------------------------------------------------------------------------


def test_root_id_higher_than_reply_id():
    """The documented case: tweet 1 has parent 3 and reply 2, so order is 3 -> 1 -> 2.

    Any code that sorted by tweet_id would report 1 as the root and get the
    depths backwards. This is the test that catches that.
    """
    df = tweets([(1, 3, False, 1), (2, 1, True, 2), (3, None, True, 0)])
    out = build_threads(df)
    by_id = rows_by_id(out)

    assert by_id[3].depth == 0, "3 is the root despite having the highest id"
    assert by_id[1].depth == 1
    assert by_id[2].depth == 2
    assert set(out["thread_id"]) == {3}
    assert path_to_root(out, 2) == [3, 1, 2]


# --------------------------------------------------------------------------
# conservation
# --------------------------------------------------------------------------


def test_every_tweet_appears_exactly_once_across_all_categories():
    """One frame containing all four hazards at once. Nothing may be lost."""
    df = tweets(
        [
            (10, None, True, 0),  # customer root
            (11, 10, False, 2),  # branch
            (12, 10, False, 1),  # branch
            (21, 99, True, 1),  # orphan
            (31, None, False, 1),  # brand root
            (40, 41, True, 0),  # cycle
            (41, 40, False, 1),  # cycle
            (42, 41, True, 2),  # off the cycle
        ]
    )
    out = build_threads(df)

    assert len(out) == len(df)
    assert out["tweet_id"].is_unique
    assert set(out["tweet_id"]) == set(df["tweet_id"])
    assert out["root_kind"].isin(["customer_root", "brand_root", "orphan", "cycle"]).all()

    # thread_ids partition the tweets: the per-thread sizes sum back to the input.
    assert out.groupby("thread_id", dropna=False).size().sum() == len(df)


def test_path_to_root_rejects_an_unknown_tweet():
    out = build_threads(tweets([(10, None, True, 0)]))
    with pytest.raises(KeyError):
        path_to_root(out, 12345)
