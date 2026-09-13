"""Tests for the SpotifyCares working dataset.

Every fixture is small enough to trace by hand and is run through the REAL
build_threads() -> assign_conversations() first, so the tests exercise the same
parent links, depths and conversation ids production does rather than a
hand-faked assignment frame.

The two that matter most:

- test_replies_to_different_customers_do_not_merge pins the 106-pair trap. The
  brand answering two different customers seconds apart under one broadcast
  looks exactly like a split reply to any time-adjacency rule, and merging them
  would splice two unrelated answers together.
- test_part_markers_beat_created_at pins the ordering precedence. It currently
  fires on 0 of 1,258 real groups, so without this test the rule would be
  unexercised code that nobody would notice breaking.
"""

import pandas as pd
import pytest

from src.conversations import assign_conversations
from src.ingest import read_parquet, write_parquet
from src.sample import (
    BRAND,
    MAX_TWEETS,
    build_turns,
    extract_part_numbers,
    strip_signatures,
    transform,
)
from src.threads import build_threads

CUSTOMER = "cust"


def tweets(rows):
    """Build an input frame from (tweet_id, parent_id, inbound, minute, text) tuples.

    An optional 6th element overrides author_id, which is how a second brand or
    a second customer gets into a fixture. `minute` is an offset from a fixed
    base so ordering is explicit in the test rather than hidden in a timestamp.
    """
    base = pd.Timestamp("2017-10-31 22:00:00", tz="UTC")
    return pd.DataFrame(
        {
            "tweet_id": [r[0] for r in rows],
            "author_id": [
                r[5] if len(r) > 5 else (CUSTOMER if r[2] else BRAND) for r in rows
            ],
            "inbound": [r[2] for r in rows],
            "created_at": [base + pd.Timedelta(minutes=r[3]) for r in rows],
            "text": [r[4] for r in rows],
            "response_tweet_id": [None] * len(rows),
            "in_response_to_tweet_id": [r[1] for r in rows],
        }
    )


def assigned_frame(rows):
    """Run the real stages 2-3 and return the `assigned` frame stage 4 consumes."""
    tw = tweets(rows)
    threads = build_threads(tw)
    assignment = assign_conversations(threads, tw)
    cols = assignment[
        [
            "tweet_id",
            "conversation_id",
            "conversation_source",
            "parent_id",
            "depth_in_conversation",
        ]
    ]
    tagged = tw.merge(cols, on="tweet_id", how="left", validate="one_to_one")
    return tagged[tagged["conversation_id"].notna()]


def run(rows):
    """(turns, conversations) for a fixture."""
    ordered, turns, out, n_reordered, _ = transform(assigned_frame(rows))
    return turns, out


def turn_texts(turns, role=None):
    selected = turns if role is None else turns[turns["role"].eq(role)]
    return list(selected.sort_values("turn_index")["text"])


# --------------------------------------------------------------------------
# 1. merging split replies
# --------------------------------------------------------------------------


def test_sibling_parts_merge_into_one_turn():
    """The real shape of a split reply: both parts reply to the SAME customer tweet.

        10 customer
        |-- 11 brand "1: first half"   (t+1)
        `-- 12 brand "2: second half"  (t+2)

    Modelled on conversation 1878, where tweets 1880 and 1879 are both children
    of customer tweet 1877.
    """
    turns, out = run(
        [
            (10, None, True, 0, "my music stopped"),
            (11, 10, False, 1, "1: first half"),
            (12, 10, False, 2, "2: second half"),
        ]
    )

    assert len(turns) == 2, "one customer turn, one merged brand turn"
    brand = turns[turns["role"].eq("brand")].iloc[0]
    assert brand["text"] == "first half second half", "joined in created_at order"
    assert brand["n_parts"] == 2
    assert list(brand["tweet_ids"]) == [11, 12]
    assert out.iloc[0]["n_tweets"] == 3 and out.iloc[0]["n_turns"] == 2


def test_replies_to_different_customers_do_not_merge():
    """The 106-pair trap: adjacent in time, same author, DIFFERENT parents.

        20 brand (broadcast)
        |-- 21 customer A
        |   `-- 23 brand "answer for A"  (t+3)
        `-- 22 customer B
            `-- 24 brand "answer for B"  (t+4)

    The two brand replies are one minute apart and would be merged by any rule
    keyed on time-adjacency alone. They are answers to two different people.
    """
    turns, out = run(
        [
            (20, None, False, 0, "new release out now"),
            (21, 20, True, 1, "cannot log in", "custA"),
            (22, 20, True, 2, "billing problem", "custB"),
            (23, 21, False, 3, "answer for A"),
            (24, 22, False, 4, "answer for B"),
        ]
    )

    brand_turns = turns[turns["role"].eq("brand")]
    assert len(brand_turns) == 2, "two separate answers, never spliced together"
    assert set(brand_turns["text"]) == {"answer for A", "answer for B"}
    assert (brand_turns["n_parts"] == 1).all()
    assert len(out) == 2, "and they stay in two separate conversations"


def test_self_reply_chain_merges():
    """The other 18 cases: part 2 replies to part 1 rather than to the customer.

        10 customer
        `-- 11 brand "1: first half"
            `-- 12 brand "2: second half"
    """
    turns, _ = run(
        [
            (10, None, True, 0, "my music stopped"),
            (11, 10, False, 1, "1: first half"),
            (12, 11, False, 2, "2: second half"),
        ]
    )

    brand = turns[turns["role"].eq("brand")]
    assert len(brand) == 1
    assert brand.iloc[0]["text"] == "first half second half"
    assert brand.iloc[0]["n_parts"] == 2


def test_brand_replies_separated_by_a_customer_tweet_do_not_merge():
    """A reply, the customer answering, then another reply is a dialogue, not a split."""
    turns, _ = run(
        [
            (10, None, True, 0, "my music stopped"),
            (11, 10, False, 1, "which device?"),
            (12, 11, True, 2, "an iphone"),
            (13, 12, False, 3, "try reinstalling"),
        ]
    )

    assert len(turns) == 4, "four turns; nothing merges across a customer turn"
    assert turn_texts(turns, "brand") == ["which device?", "try reinstalling"]


def test_two_brands_in_one_conversation_do_not_merge_together():
    """A handoff is two authors, so it is two turns even though both are outbound."""
    turns, _ = run(
        [
            (10, None, True, 0, "my music stopped"),
            (11, 10, False, 1, "over to spotify"),
            (12, 10, False, 2, "we can help", "hulu_support"),
        ]
    )

    brand = turns[turns["role"].eq("brand")]
    assert len(brand) == 2
    assert set(brand["author_id"]) == {BRAND, "hulu_support"}


def test_continuation_mentions_and_markers_are_stripped_but_not_the_first_parts():
    """A merged reply must read as one reply, not as two tweets glued together.

    Modelled on conversation 1878, which before this read
    "...there's info about... @116129 2: Spotify content here...".
    """
    turns, _ = run(
        [
            (10, None, True, 0, "albums are missing"),
            (11, 10, False, 1, "@116129 1: Those albums are unavailable, but there's info about..."),
            (12, 10, False, 2, "@116129 2: Spotify content here. /RH"),
        ]
    )

    brand = turns[turns["role"].eq("brand")].iloc[0]
    assert brand["text"] == (
        "@116129 Those albums are unavailable, but there's info about... "
        "Spotify content here."
    )
    assert brand["text"].startswith("@116129 "), "the first part keeps its @mention"
    assert "1:" not in brand["text"], "and loses its now-stale part marker"
    assert "@116129 2:" not in brand["text"], "the continuation's routing artifacts are gone"
    assert list(brand["tweet_ids"]) == [11, 12], "both original tweet ids are still here"


def test_unsplit_reply_keeps_its_mention_and_marker():
    """Stripping applies to continuations only, so a lone reply is untouched."""
    turns, _ = run(
        [
            (10, None, True, 0, "albums are missing"),
            (11, 10, False, 1, "@116129 2: and disable shuffle play /LO"),
        ]
    )

    brand = turns[turns["role"].eq("brand")].iloc[0]
    assert brand["text"] == "@116129 2: and disable shuffle play"
    assert bool(brand["orphan_part"]), "still an orphan part, just not rewritten"


def test_continuation_that_is_only_a_mention_and_marker_is_left_alone():
    """Stripping must never blank out a part. Empty is worse than redundant."""
    turns, _ = run(
        [
            (10, None, True, 0, "albums are missing"),
            (11, 10, False, 1, "1: here is the answer"),
            (12, 10, False, 2, "@116129 2:"),
        ]
    )

    brand = turns[turns["role"].eq("brand")].iloc[0]
    assert brand["text"] == "here is the answer @116129 2:", "kept rather than emptied"
    assert brand["n_parts"] == 2


# --------------------------------------------------------------------------
# 2. ordering: part markers outrank created_at
# --------------------------------------------------------------------------


def test_part_markers_beat_created_at():
    """Part "2" is timestamped EARLIER than part "1". The marker wins.

    src/threads.py established that timestamps never run backwards along a
    parent-child edge, but these two are siblings and so are not on an edge
    with each other. That guarantee does not reach this case, which is why the
    marker is treated as the stronger evidence.
    """
    turns, _ = run(
        [
            (10, None, True, 0, "my music stopped"),
            (11, 10, False, 2, "1: first half"),  # later timestamp
            (12, 10, False, 1, "2: second half"),  # earlier timestamp
        ]
    )

    brand = turns[turns["role"].eq("brand")].iloc[0]
    assert brand["text"] == "first half second half", "marker order, not clock order"
    assert list(brand["tweet_ids"]) == [11, 12]


def test_marker_reordering_is_counted():
    """The re-order count is reported, so a disagreement can never be silent."""
    in_order = assigned_frame(
        [
            (10, None, True, 0, "my music stopped"),
            (11, 10, False, 1, "1: first half"),
            (12, 10, False, 2, "2: second half"),
        ]
    )
    out_of_order = assigned_frame(
        [
            (10, None, True, 0, "my music stopped"),
            (11, 10, False, 2, "1: first half"),
            (12, 10, False, 1, "2: second half"),
        ]
    )

    assert transform(in_order)[3] == 0
    assert transform(out_of_order)[3] == 1


def test_part_numbers_are_extracted_from_every_marker_style():
    """"1:", "2.", "(1/3)", "2 of 3", and the leading @mentions Twitter adds."""
    frame = pd.DataFrame(
        {
            "text": [
                "1: first",
                "2. second",
                "(1/3) first",
                "2 of 3 second",
                "@116129 1: with a mention",
                "no marker here",
                "24/7 support is not a marker",
            ]
        }
    )
    parts = extract_part_numbers(frame)

    assert list(parts[:5]) == [1, 2, 1, 2, 1]
    assert pd.isna(parts[5]) and pd.isna(parts[6]), "neither of the last two is a part marker"


# --------------------------------------------------------------------------
# 3. signatures
# --------------------------------------------------------------------------


def test_signature_is_extracted_not_deleted():
    """Both real positions: at the very end, and before a single trailing URL."""
    frame = pd.DataFrame(
        {
            "inbound": [False, False, False, True],
            "text": [
                "a clean reinstall should help /LS",
                "take a look backstage /CH https://t.co/ldFdZRiNAt",
                "no signature on this one",
                "my playlist is broken and/or empty",
            ],
        }
    )
    out = strip_signatures(frame)

    assert list(out["signature"][:2]) == ["LS", "CH"]
    assert out["text_clean"][0] == "a clean reinstall should help"
    assert out["text_clean"][1] == "take a look backstage https://t.co/ldFdZRiNAt", (
        "the trailing URL survives; only the signature is lifted out"
    )

    assert pd.isna(out["signature"][2]) and out["text_clean"][2] == "no signature on this one"
    assert pd.isna(out["signature"][3]), "customer tweets are never signature-stripped"
    assert out["text_clean"][3] == "my playlist is broken and/or empty"


def test_merged_reply_has_no_interior_mention_or_part_marker():
    """A merged reply must read as ONE reply, not two tweets glued together.

    Modelled on conversation 1878, which before this step read
    "...there's info about... @116129 2: Spotify content here...". The mention
    and the marker are Twitter routing artifacts, not content.
    """
    turns, _ = run(
        [
            (10, None, True, 0, "albums are missing"),
            (11, 10, False, 1, "@116129 1: Those albums are unavailable, info here:"),
            (12, 10, False, 2, "@116129 2: Spotify content lives here. /RH"),
        ]
    )

    text = turns[turns["role"].eq("brand")].iloc[0]["text"]

    assert text == (
        "@116129 Those albums are unavailable, info here: Spotify content lives here."
    )
    assert text.startswith("@116129 "), "the first part keeps its @mention"
    assert text.count("@") == 1, "no interior @mention survives the join"
    assert "2:" not in text, "no interior part marker survives the join"


def test_single_part_reply_keeps_its_mention_and_marker():
    """Only the split introduces the artifact, so only a continuation is stripped.

    A reply that was never split must come through byte-for-byte, including an
    orphan "2:" whose sibling is missing -- editing that would hide the fact
    that it is half an answer.
    """
    turns, _ = run(
        [
            (10, None, True, 0, "my music stopped"),
            (11, 10, False, 1, "@116129 we can help out /LS"),
            (12, 11, True, 2, "thanks"),
            (13, 12, False, 3, "@116129 2: and disable shuffle play /LO"),
        ]
    )

    brand = turns[turns["role"].eq("brand")].sort_values("turn_index")
    assert list(brand["text"]) == [
        "@116129 we can help out",
        "@116129 2: and disable shuffle play",
    ]
    assert bool(brand.iloc[1]["orphan_part"]), "still visibly a half-answer"


def test_merged_reply_takes_the_signature_of_its_last_signed_part():
    """Part 1 is usually unsigned; the agent signs the last part."""
    turns, _ = run(
        [
            (10, None, True, 0, "my music stopped"),
            (11, 10, False, 1, "1: first half"),
            (12, 10, False, 2, "2: second half /LS"),
        ]
    )

    brand = turns[turns["role"].eq("brand")].iloc[0]
    assert brand["agent_signature"] == "LS"
    assert brand["text"] == "first half second half", "no signature left mid-text"


# --------------------------------------------------------------------------
# 4. orphan parts
# --------------------------------------------------------------------------


def test_orphan_part_is_flagged_and_kept():
    """A "2:" whose "1:" is not in the corpus: permanently half an answer."""
    turns, out = run(
        [
            (10, None, True, 0, "my music stopped"),
            (11, 10, False, 1, "2: and disable shuffle play /LO"),
        ]
    )

    brand = turns[turns["role"].eq("brand")].iloc[0]
    assert bool(brand["orphan_part"]), "carries a marker but found no sibling"
    assert brand["n_parts"] == 1
    assert brand["text"] == "2: and disable shuffle play", "kept, not dropped"
    assert len(out) == 1, "and its conversation survives"


def test_properly_merged_parts_are_not_orphans():
    turns, _ = run(
        [
            (10, None, True, 0, "my music stopped"),
            (11, 10, False, 1, "1: first half"),
            (12, 10, False, 2, "2: second half"),
        ]
    )

    assert not bool(turns[turns["role"].eq("brand")].iloc[0]["orphan_part"])


def test_customer_tweet_starting_with_a_number_is_not_an_orphan_part():
    """orphan_part is a statement about brand replies. "1. it crashes" is not one."""
    turns, _ = run(
        [
            (10, None, True, 0, "1. it crashes 2. it is slow"),
            (11, 10, False, 1, "we can help /LS"),
        ]
    )

    assert not bool(turns[turns["role"].eq("customer")].iloc[0]["orphan_part"])


# --------------------------------------------------------------------------
# 5. the size cap
# --------------------------------------------------------------------------


def _ladder(n_tweets):
    """A straight alternating conversation of exactly n_tweets tweets."""
    rows = [(10, None, True, 0, "opening complaint")]
    for i in range(1, n_tweets):
        inbound = i % 2 == 1
        text = f"turn {i}" if not inbound else f"reply {i}"
        rows.append((10 + i, 9 + i, not inbound, i, text))
    return rows


def test_conversation_over_the_cap_is_dropped_and_at_the_cap_is_kept():
    _, kept = run(_ladder(MAX_TWEETS))
    assert len(kept) == 1, f"{MAX_TWEETS} tweets is not 'more than {MAX_TWEETS}'"

    _, dropped = run(_ladder(MAX_TWEETS + 1))
    assert len(dropped) == 0


# --------------------------------------------------------------------------
# 6. opening message and first substantive reply
# --------------------------------------------------------------------------


def test_opening_message_is_the_inbound_root():
    _, out = run(
        [
            (10, None, True, 0, "my playlist vanished"),
            (11, 10, False, 1, "we can help /LS"),
        ]
    )

    row = out.iloc[0]
    assert row["opening_text"] == "my playlist vanished"
    assert row["opening_tweet_id"] == 10
    assert row["opening_author_id"] == CUSTOMER
    assert row["source"] == "customer_root"


def test_first_substantive_reply_skips_a_bare_deflection():
    """A pure "DM us" is not the reply to learn from; the troubleshooting is."""
    _, out = run(
        [
            (10, None, True, 0, "my music stopped"),
            (11, 10, False, 1, "Could you DM us your account's email address? /LS"),
            (12, 11, True, 2, "sent"),
            (13, 12, False, 3, "Hold Sleep/Wake and Volume Down for 10 seconds /CH"),
        ]
    )

    row = out.iloc[0]
    assert row["first_reply_text"] == "Hold Sleep/Wake and Volume Down for 10 seconds"
    assert row["first_reply_tweet_ids"] == [13]
    assert row["first_reply_signature"] == "CH"
    assert row["first_reply_turn_index"] == 3


def test_deflection_only_conversation_is_kept_with_a_null_first_reply():
    """Dropping these would quietly change what the dataset is a sample OF."""
    _, out = run(
        [
            (10, None, True, 0, "my music stopped"),
            (11, 10, False, 1, "Could you DM us your account's email address? /LS"),
        ]
    )

    assert len(out) == 1, "kept"
    assert pd.isna(out.iloc[0]["first_reply_text"])


def test_first_reply_prefers_the_merged_text_not_a_half_answer():
    """The point of the whole stage: the first reply is the WHOLE first reply."""
    _, out = run(
        [
            (10, None, True, 0, "albums are missing"),
            (11, 10, False, 1, "1: Those albums are currently unavailable."),
            (12, 10, False, 2, "2: More info on Spotify content here. /RH"),
        ]
    )

    row = out.iloc[0]
    assert row["first_reply_text"] == (
        "Those albums are currently unavailable. More info on Spotify content here."
    )
    assert row["first_reply_tweet_ids"] == [11, 12]


# --------------------------------------------------------------------------
# 7. selection and conservation
# --------------------------------------------------------------------------


def test_conversations_without_the_brand_are_excluded():
    """Another brand's conversation is not in this dataset at all."""
    _, out = run(
        [
            (10, None, True, 0, "spotify problem"),
            (11, 10, False, 1, "we can help /LS"),
            (20, None, True, 0, "apple problem", "custB"),
            (21, 20, True, 1, "still broken", "custB"),
        ]
    )

    assert len(out) == 1
    assert out.iloc[0]["conversation_id"] == 10


def test_other_brands_tweets_inside_a_kept_conversation_are_retained():
    """81 such tweets exist. Dropping them leaves a hole in the turn list."""
    turns, out = run(
        [
            (10, None, True, 0, "my music stopped"),
            (11, 10, False, 1, "we can help /LS"),
            (12, 11, True, 2, "thanks"),
            (13, 12, False, 3, "chiming in", "hulu_support"),
        ]
    )

    assert len(out) == 1
    assert out.iloc[0]["n_tweets"] == 4
    assert "hulu_support" in set(turns["author_id"])


def test_turns_reconcile_with_tweets_and_are_ordered():
    """Nothing may be lost or duplicated between tweets and turns."""
    rows = [
        (10, None, True, 0, "my music stopped"),
        (11, 10, False, 1, "1: first half"),
        (12, 10, False, 2, "2: second half /LS"),
        (13, 12, True, 3, "still broken"),
        (14, 13, False, 4, "try a clean reinstall /CH"),
    ]
    turns, out = run(rows)

    assert len(out) == 1, "one row per surviving conversation"
    row = out.iloc[0]
    assert row["n_tweets"] == 5
    assert row["n_turns"] == 4
    assert int(turns["n_parts"].sum()) == 5, "every tweet lands in exactly one turn"

    nested = row["turns"]
    assert [t["turn_index"] for t in nested] == [0, 1, 2, 3]
    assert [t["role"] for t in nested] == ["customer", "brand", "customer", "brand"]
    assert [t["text"] for t in nested] == [
        "my music stopped",
        "first half second half",
        "still broken",
        "try a clean reinstall",
    ]


def test_output_round_trips_through_the_projects_own_parquet_helpers(tmp_path):
    """The nested `turns` column must survive write -> read.

    This is a regression test for a real defect: duckdb converts a top-level
    TIMESTAMP WITH TIME ZONE to pandas unaided but needs pytz for one nested
    inside a STRUCT, so a tz-aware created_at in `turns` produced a parquet
    that read_parquet() could not open at all. Nothing downstream would have
    worked, and nothing else in the suite would have noticed.
    """
    _, out = run(
        [
            (10, None, True, 0, "my music stopped"),
            (11, 10, False, 1, "1: first half"),
            (12, 10, False, 2, "2: second half /LS"),
        ]
    )

    path = write_parquet(out, tmp_path / "spotify.parquet")
    back = read_parquet(path)

    assert len(back) == len(out)
    turns = back.iloc[0]["turns"]
    assert [t["role"] for t in turns] == ["customer", "brand"]
    assert turns[1]["text"] == "first half second half"
    assert turns[1]["agent_signature"] == "LS"
    assert list(turns[1]["tweet_ids"]) == [11, 12]
    assert turns[1]["n_parts"] == 2


def test_turn_index_is_unique_within_a_conversation():
    turns, _ = run(
        [
            (10, None, True, 0, "opening"),
            (11, 10, False, 1, "reply /LS"),
            (20, None, True, 0, "another opening", "custB"),
            (21, 20, False, 1, "another reply /CH"),
        ]
    )

    counts = turns.groupby("conversation_id", dropna=False)["turn_index"].nunique()
    sizes = turns.groupby("conversation_id", dropna=False).size()
    assert (counts == sizes).all()
