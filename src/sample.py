"""Cut the corpus to SpotifyCares and repair split replies. Stage 4 of the pipeline.

Stages 1-3 produce 813,446 conversations across 108 brands. Decision 31 picked
SpotifyCares. This stage emits the one narrow table everything downstream
(retrieval, generation, judging) actually reads.

The defect it repairs, from decision 32: 6.27% of SpotifyCares replies are split
across tweets, so a single tweet is often half an answer. Indexing raw tweets
means retrieval can return half a sentence as a "historical resolution".

Two things about how those replies are split drive the whole design, and both
were measured rather than assumed:

1. MULTI-PART REPLIES ARE SIBLINGS, NOT SELF-REPLY CHAINS. In conversation 1878,
   the tweets "1: Thanks! Looks like those albums..." and "2: Spotify content
   here:..." are both children of the SAME customer tweet. Of 1,483 adjacent
   same-author outbound pairs, 1,359 share a parent and only 18 are parent ->
   child.

   So merging cannot key on the reply chain. But it also cannot key on
   time-adjacency alone: the remaining 106 pairs are the brand answering TWO
   DIFFERENT CUSTOMERS under one broadcast, seconds apart. Merging those would
   splice two unrelated answers into one reply and corrupt the dataset
   silently. The rule below therefore requires a shared parent OR a direct
   chain, which is exactly what separates the 1,377 real cases from the 106
   traps.

2. PART MARKERS OUTRANK created_at. src/threads.py established that timestamps
   never run backwards along a parent-child edge, but siblings are not on an
   edge with each other, so that guarantee does not reach this case. Ordering
   within a merge group is therefore by created_at and then CHECKED against the
   "1:" / "2:" markers; where they disagree the marker wins and the count is
   reported. Currently 0 of 1,258 fully-marked groups disagree — the check
   exists so a future disagreement is loud rather than silent.

When parts are joined, the artifacts of the split are removed so a merged reply
reads as one reply rather than as two tweets glued together: continuation parts
lose their leading @mentions and part marker, and the first part loses its now
meaningless "1:" (there is no part N once they are joined) while keeping its
@mention, which every reply has. A reply that was NOT merged is left exactly as
it was -- an unmerged "2:" is an orphan_part and must stay visible. See
strip_continuation_prefixes() and strip_stale_first_part_marker().

Two further decisions visible in the output:

- The >20-tweet cap is a size guard doing double duty. The largest thing it
  removes is conversation 2812, rooted at @115888 -- a Spotify MARKETING handle
  that carries inbound=True. A promo post is therefore indistinguishable from a
  customer's opening message, and ~130 replies pile up beneath it. `@<number>`
  does not mean "a customer".

- 68 replies carry a part marker but have no sibling to merge with: a "2:"
  whose "1:" is not in the corpus. They are permanently half-answers. They are
  FLAGGED (`orphan_part`) and kept, not dropped, so that excluding them happens
  visibly at the retrieval index rather than invisibly here.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.deflection import (
    LEADING_MENTIONS,
    MAX_RESIDUAL_WORDS,
    has_redirect,
    normalize,
    residual_word_count,
)
from src.ingest import REPO_ROOT, _display_path, read_parquet, write_parquet

INTERIM = REPO_ROOT / "data" / "interim"
SPOTIFY_PARQUET = INTERIM / "spotify.parquet"

BRAND = "SpotifyCares"

# Conversations longer than this are dropped. See the module docstring: this is
# a size guard that also removes the brand-handle-as-customer artifacts.
MAX_TWEETS = 20

# Nothing here shuffles or samples. Declared because the project pins seed=42
# everywhere and a reader should not have to grep to confirm this stage is
# deterministic; deliberately unused rather than fed to a call that does not
# need it.
SEED = 42

# A part marker at the START of a reply: "1:", "2.", "(1/3)", "2 of 3",
# tolerating the leading @mentions that Twitter puts on every reply. The
# capture groups are why this is not scripts/brand_survey.py's MULTIPART, which
# only needs a boolean -- ordering needs the number itself.
#
# The part number is a SINGLE digit, deliberately tighter than MULTIPART's
# \d{1,2}. Two digits makes "24/7 support is available..." parse as part 24 and
# silently mis-sort the reply it opens. Over the real data this costs nothing:
# SpotifyCares part numbers only ever run 1-4 (1,322 / 1,339 / 36 / 1), and the
# "x of y" form never appears at all -- the house style is bare "1:" / "2:".
PART_MARKER = re.compile(
    r"^\s*(?:@\w+\s+)*\(?(\d)\s*(?:/|of)\s*\d{1,2}\)?[\s:.\-]"
    r"|^\s*(?:@\w+\s+)*\(?(\d)\)?\s*[:.](?:\s|(?=[A-Za-z]))"
)

# The brand's agent signature: "/LS", occasionally one letter ("/T"). Anchored
# to the very end of the reply OR to just before a single trailing URL, because
# those are the only two places it appears (30,099 and 9,752 respectively).
# The anchoring is what keeps "24/7", "w/", and URL slashes out: verified zero
# false positives over all 43,203 SpotifyCares replies.
SIGNATURE = re.compile(r"\s*/([A-Za-z]{1,2})(?=\s*$|\s+https?://\S+\s*$)")

# The part marker on its own, with no leading-@mention allowance, for stripping
# it off a continuation part once its mentions are already gone. Same shapes as
# PART_MARKER above; the trailing delimiter is consumed with it so "2: text"
# leaves "text" rather than ": text".
# The delimiter after the marker may be end-of-string: a continuation part that
# is nothing but routing ("@116129 2:") must strip to empty so the guard in
# strip_continuation_prefixes() can spot it and keep the original instead of
# leaving a bare "2:" behind.
PART_MARKER_ONLY = re.compile(
    r"^\s*(?:\(?\d\s*(?:/|of)\s*\d{1,2}\)?(?:[\s:.\-]|$)|\(?\d\)?\s*[:.](?:\s|$|(?=[A-Za-z])))"
)

# The same marker, but sitting AFTER the leading @mentions rather than instead
# of them, so the mentions can be captured and put back. Used on the first part
# of a merged reply, where the mention is kept and only the marker goes.
FIRST_PART_MARKER = re.compile(
    r"^((?:\s*@\w+)*\s*)"
    r"(?:\(?\d\s*(?:/|of)\s*\d{1,2}\)?(?:[\s:.\-]|$)|\(?\d\)?\s*[:.](?:\s|$|(?=[A-Za-z])))"
)


# --------------------------------------------------------------------------
# selection
# --------------------------------------------------------------------------


def load_assigned(interim: Path = INTERIM) -> pd.DataFrame:
    """Every assigned tweet, carrying the conversation columns from stage 3."""
    tweets = read_parquet(interim / "tweets.parquet")
    assignment = read_parquet(interim / "conversation_assignment.parquet")

    cols = assignment[
        ["tweet_id", "conversation_id", "conversation_source", "parent_id", "depth_in_conversation"]
    ]
    tagged = tweets.merge(cols, on="tweet_id", how="left", validate="one_to_one")
    assert len(tagged) == len(tweets), f"merge changed row count: {len(tweets):,} -> {len(tagged):,}"

    assigned = tagged[tagged["conversation_id"].notna()]
    return assigned


def select_brand_conversations(assigned: pd.DataFrame, brand: str = BRAND) -> pd.DataFrame:
    """Every tweet of every conversation in which `brand` authored an outbound tweet.

    Tweets by OTHER brands inside those conversations are kept -- 81 of them, from
    handoffs and customers tagging two companies. They are part of the dialogue,
    and dropping them would leave a turn list with a hole in it.
    """
    authored = assigned["author_id"].eq(brand) & ~assigned["inbound"]
    brand_conversations = set(assigned.loc[authored, "conversation_id"])
    selected = assigned[assigned["conversation_id"].isin(brand_conversations)]
    return selected


def drop_long_conversations(selected: pd.DataFrame, max_tweets: int = MAX_TWEETS) -> pd.DataFrame:
    """Drop conversations with more than `max_tweets` tweets.

    Applied to the RAW tweet count, before merging: "more than 20 tweets" is
    read literally as tweets, not as post-merge turns.
    """
    sizes = selected.groupby("conversation_id", dropna=False)["tweet_id"].transform("size")
    capped = selected[sizes <= max_tweets]
    return capped


# --------------------------------------------------------------------------
# merging split replies
# --------------------------------------------------------------------------


def assign_merge_groups(capped: pd.DataFrame) -> pd.DataFrame:
    """Number each run of tweets that together form one logical turn.

    Returns the frame sorted into conversation order with a `merge_group`
    column. Every tweet gets a group; most groups have exactly one member.

    A tweet CONTINUES the previous group when it is the same outbound author in
    the same conversation AND it either shares the previous tweet's parent
    (sibling parts) or replies directly to it (a self-reply chain). The parent
    condition is the whole point -- see trap 1 in the module docstring.
    """
    ordered = capped.sort_values(["conversation_id", "created_at", "tweet_id"], kind="stable")
    within = ordered.groupby("conversation_id", dropna=False)

    same_author = ordered["author_id"].eq(within["author_id"].shift())
    both_outbound = ~ordered["inbound"] & within["inbound"].shift().eq(False)
    shares_parent = ordered["parent_id"].eq(within["parent_id"].shift())
    replies_to_previous = ordered["parent_id"].eq(within["tweet_id"].shift())

    # fillna(False) resolves the pd.NA that Int64 comparisons produce at a
    # conversation root, where parent_id is NA. NA means "cannot show these are
    # the same reply", so it must not merge -- False is the safe direction.
    continues = (
        (same_author & both_outbound & (shares_parent | replies_to_previous))
        .fillna(False)
        .astype(bool)
    )

    ordered = ordered.assign(merge_group=(~continues).cumsum())
    assert ordered["merge_group"].is_monotonic_increasing, (
        "merge groups must follow conversation order; a group cannot span a gap"
    )
    return ordered


def extract_part_numbers(ordered: pd.DataFrame) -> pd.Series:
    """The leading "1:" / "2/3" part number per tweet, or NA where there is none."""
    captured = ordered["text"].str.extract(PART_MARKER)
    combined = captured[0].fillna(captured[1])
    return pd.to_numeric(combined, errors="coerce").astype("Int64")


def apply_marker_order(ordered: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Re-sort any merge group whose part markers contradict its timestamps.

    Returns the frame in final turn order and the number of groups re-sorted.

    created_at is the default order. Where a marker sequence decreases along it
    the markers win, because sibling ordering is not covered by the
    parent-child timestamp guarantee that src/threads.py established.
    """
    within_group = ordered.groupby("merge_group", dropna=False)
    previous_part = within_group["part_number"].shift()
    # .lt on Int64 yields pd.NA against a null previous part; those are groups
    # with nothing to compare, not disagreements.
    disagrees = ordered["part_number"].lt(previous_part).fillna(False)

    reordered_groups = set(ordered.loc[disagrees, "merge_group"])
    if not reordered_groups:
        return ordered, 0

    # Sort the offending groups by marker, leave every other group on its
    # created_at order (position within the group).
    position = within_group.cumcount()
    sort_key = np.where(
        ordered["merge_group"].isin(reordered_groups),
        ordered["part_number"].fillna(0).to_numpy(),
        position.to_numpy(),
    )
    resorted = ordered.assign(_sort_key=sort_key).sort_values(
        ["merge_group", "_sort_key"], kind="stable"
    )
    return resorted.drop(columns="_sort_key"), len(reordered_groups)


def strip_signatures(ordered: pd.DataFrame) -> pd.DataFrame:
    """Split the trailing /XX agent signature out of the text into its own column.

    Done per tweet, BEFORE parts are joined: part 1 of a split reply is usually
    unsigned and only the last part carries the signature, so stripping per part
    cannot leave one stranded mid-text.

    Only outbound tweets are touched. A customer writing "and/or" at the end of
    a tweet is not signing it.
    """
    outbound = ~ordered["inbound"]
    captured = ordered["text"].str.extract(SIGNATURE)[0]

    signature = captured.where(outbound)
    text_clean = ordered["text"].mask(
        outbound, ordered["text"].str.replace(SIGNATURE, "", regex=True)
    )

    return ordered.assign(signature=signature, text_clean=text_clean.str.strip())


def strip_continuation_prefixes(ordered: pd.DataFrame) -> pd.DataFrame:
    """Remove the leading @mentions and part marker from CONTINUATION parts only.

    A split reply is one reply, and should read as one. Before this step
    conversation 1878's merged reply read:

        "...there's info about... @116129 2: Spotify content here..."

    The "@116129" and the "2:" are Twitter routing artifacts -- the mention is
    how a reply is addressed and the marker is how a 140-character limit was
    worked around. Neither is content, and both land mid-sentence once the
    parts are joined.

    The FIRST part is never touched. Its mention and its "1:" are the reply's
    real opening, and stripping them would edit a tweet that was not split at
    all in any way the reader can see. So this only ever removes text that the
    split itself introduced.

    Nothing is lost: every original tweet id stays in the turn's `tweet_ids`,
    and the raw text is still in data/interim/tweets.parquet.

    Position is taken AFTER apply_marker_order, so "first part" means first in
    the reply's true order rather than first by timestamp.
    """
    position = ordered.groupby("merge_group", dropna=False).cumcount()
    is_continuation = position > 0

    # Mentions come off first, so the marker is then at the very front and
    # PART_MARKER_ONLY (which allows no leading mentions) is the exact pattern.
    stripped = (
        ordered["text_clean"]
        .str.replace(LEADING_MENTIONS, "", regex=True)
        .str.replace(PART_MARKER_ONLY, "", regex=True)
        .str.strip()
    )
    # Only overwrite where stripping left something behind. A continuation part
    # that is nothing but a mention and a marker would otherwise become an
    # empty string and silently blank out half a reply.
    safe = is_continuation & stripped.str.len().gt(0)

    return ordered.assign(text_clean=ordered["text_clean"].mask(safe, stripped))


def strip_stale_first_part_marker(ordered: pd.DataFrame) -> pd.DataFrame:
    """Drop the "1:" from the first part of a reply that was actually merged.

    "1:" means "part 1 of N". Once the parts are joined there is no part N, so
    the marker describes a split that no longer exists in the text. It is
    stale, and leaving it makes a whole reply look like a fragment.

    The leading @mention is KEPT. Every reply on Twitter opens with one, so
    keeping it is uniform across the dataset; the marker is the only part that
    is an artifact of the split.

    Only applies where the reply was genuinely merged (more than one part). A
    single-part reply keeps its marker, because there the marker is real
    evidence: an unmerged "2:" is an orphan_part, a permanently half answer,
    and editing it would hide exactly the thing the flag exists to surface.
    """
    group = ordered.groupby("merge_group", dropna=False)
    is_first = group.cumcount() == 0
    was_merged = group["tweet_id"].transform("size") > 1

    # \1 puts the captured leading mentions back; only the marker is removed.
    stripped = (
        ordered["text_clean"].str.replace(FIRST_PART_MARKER, r"\1", regex=True).str.strip()
    )
    # Same guard as above: never let stripping empty a part out.
    safe = is_first & was_merged & stripped.str.len().gt(0)

    return ordered.assign(text_clean=ordered["text_clean"].mask(safe, stripped))


# --------------------------------------------------------------------------
# turns
# --------------------------------------------------------------------------


def build_turns(ordered: pd.DataFrame) -> pd.DataFrame:
    """Collapse each merge group into one turn. One row per turn.

    `created_at` is made tz-NAIVE here, unlike everywhere else in the pipeline.
    duckdb can convert a top-level TIMESTAMP WITH TIME ZONE to pandas on its
    own, but needs pytz to convert one nested inside a STRUCT, so a tz-aware
    timestamp in `turns` produces a parquet that this project's own
    read_parquet() cannot open. Dropping the tz rather than adding pytz loses
    nothing: ingest parses every timestamp with utc=True, so the whole corpus
    is UTC and these values are UTC too. test_sample.py pins the round-trip.
    """
    ordered = ordered.assign(
        role=np.where(ordered["inbound"], "customer", "brand"),
        created_at_naive=ordered["created_at"].dt.tz_localize(None),
        # ffill inside the group so "last" picks up the last agent who signed,
        # rather than a null from an unsigned trailing part.
        signature_filled=ordered.groupby("merge_group", dropna=False)["signature"].ffill(),
    )

    grouped = ordered.groupby(["conversation_id", "merge_group"], dropna=False)
    turns = grouped.agg(
        tweet_ids=("tweet_id", list),
        role=("role", "first"),
        author_id=("author_id", "first"),
        created_at=("created_at_naive", "min"),
        text=("text_clean", " ".join),
        agent_signature=("signature_filled", "last"),
        n_parts=("tweet_id", "size"),
        n_marked=("part_number", "count"),
    ).reset_index()

    assert int(turns["n_parts"].sum()) == len(ordered), (
        f"turn parts do not sum back to the tweets that went in: "
        f"{int(turns['n_parts'].sum()):,} vs {len(ordered):,}"
    )

    # A brand reply carrying a part marker that found no sibling to merge with:
    # a "2:" whose "1:" is not in the corpus. Permanently half an answer.
    turns["orphan_part"] = (
        turns["role"].eq("brand") & turns["n_parts"].eq(1) & turns["n_marked"].ge(1)
    )
    turns["turn_index"] = turns.groupby("conversation_id", dropna=False).cumcount()
    return turns


def find_first_substantive_reply(turns: pd.DataFrame, brand: str = BRAND) -> pd.DataFrame:
    """The first merged brand reply that is not a pure deflection, per conversation.

    Reuses src/deflection.py rather than restating the rule, so this introduces
    no new threshold -- it is decision 26's rule applied to merged replies. A
    conversation whose every reply is a bare "DM us" yields no row here and is
    kept with null first_reply_* fields, because dropping it would quietly
    change what the dataset is a sample OF.

    orphan_part deliberately does NOT disqualify a reply. The flag is carried
    through instead, so excluding half-answers stays a visible downstream choice.
    """
    brand_turns = turns[turns["author_id"].eq(brand)].sort_values(
        ["conversation_id", "turn_index"], kind="stable"
    )
    normalized = brand_turns["text"].map(normalize)
    deflection = has_redirect(normalized) & (residual_word_count(normalized) <= MAX_RESIDUAL_WORDS)

    substantive = brand_turns[~deflection]
    first = substantive.groupby("conversation_id", dropna=False).first().reset_index()

    return first[
        ["conversation_id", "tweet_ids", "text", "agent_signature", "turn_index", "orphan_part"]
    ].rename(
        columns={
            "tweet_ids": "first_reply_tweet_ids",
            "text": "first_reply_text",
            "agent_signature": "first_reply_signature",
            "turn_index": "first_reply_turn_index",
            "orphan_part": "first_reply_orphan_part",
        }
    )


# --------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------

TURN_FIELDS = [
    "turn_index",
    "tweet_ids",
    "role",
    "author_id",
    "created_at",
    "text",
    "agent_signature",
    "n_parts",
    "orphan_part",
]


def build_conversation_rows(ordered: pd.DataFrame, turns: pd.DataFrame) -> pd.DataFrame:
    """One row per conversation: opening message, ordered turns, first reply."""
    openings = ordered[ordered["depth_in_conversation"].eq(0)]
    assert openings["conversation_id"].is_unique, "a conversation has two depth-0 tweets"
    assert bool(openings["inbound"].all()), (
        "a conversation opens with an outbound tweet; every conversation root "
        "is supposed to be inbound by construction in stage 3"
    )

    base = openings[
        ["conversation_id", "conversation_source", "tweet_id", "author_id", "created_at", "text"]
    ].rename(
        columns={
            "conversation_source": "source",
            "tweet_id": "opening_tweet_id",
            "author_id": "opening_author_id",
            "created_at": "opening_created_at",
            "text": "opening_text",
        }
    )

    # An empty groupby().apply() returns a DataFrame rather than a Series, so
    # the nesting is built explicitly for that case. It happens whenever every
    # conversation is dropped by the cap, which a test covers.
    if turns.empty:
        nested = pd.DataFrame({"conversation_id": pd.Series(dtype="Int64"), "turns": []})
    else:
        nested = (
            turns.sort_values(["conversation_id", "turn_index"], kind="stable")
            .groupby("conversation_id", dropna=False)[TURN_FIELDS]
            .apply(lambda frame: frame.to_dict("records"))
            .rename("turns")
            .reset_index()
        )

    counts = pd.DataFrame(
        {
            "n_tweets": ordered.groupby("conversation_id", dropna=False)["tweet_id"].size(),
            "n_turns": turns.groupby("conversation_id", dropna=False)["turn_index"].size(),
        }
    ).reset_index()

    out = base.merge(nested, on="conversation_id", how="left", validate="one_to_one")
    out = out.merge(counts, on="conversation_id", how="left", validate="one_to_one")
    first = find_first_substantive_reply(turns)
    out = out.merge(first, on="conversation_id", how="left", validate="one_to_one")

    assert len(out) == openings["conversation_id"].nunique(), (
        f"assembly changed the conversation count: "
        f"{openings['conversation_id'].nunique():,} -> {len(out):,}"
    )
    assert (out["n_turns"] <= out["n_tweets"]).all(), "a conversation has more turns than tweets"
    return out


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------


def _print_summary(
    assigned: pd.DataFrame,
    selected: pd.DataFrame,
    ordered: pd.DataFrame,
    turns: pd.DataFrame,
    out: pd.DataFrame,
    n_reordered: int,
) -> None:
    """The stage row-count row, plus the counts that qualify what it means."""
    n_all = int(assigned["conversation_id"].nunique())
    n_brand = int(selected["conversation_id"].nunique())
    n_kept = len(out)

    merged = turns[turns["n_parts"] > 1]
    n_merged_convs = int(merged["conversation_id"].nunique())
    merged_pct = (100.0 * n_merged_convs / n_kept) if n_kept else 0.0

    n_orphan = int(turns["orphan_part"].sum())
    n_orphan_first = int(out["first_reply_orphan_part"].fillna(False).sum())

    n_signed = int(ordered["signature"].notna().sum())
    multi_signed = ordered[ordered["signature"].notna()].groupby("merge_group", dropna=False)[
        "signature"
    ].nunique()
    n_multi_signed = int((multi_signed > 1).sum())

    n_substantive = int(out["first_reply_text"].notna().sum())

    print(
        f"sample: conversations {n_all:,} -> {BRAND} {n_brand:,} -> "
        f"<={MAX_TWEETS} tweets {n_kept:,} (lost {n_brand - n_kept:,})"
    )
    print(
        f"  tweets:    {len(selected):,} -> {len(ordered):,} after cap -> "
        f"{len(turns):,} turns after merging {len(merged):,} multi-part groups"
    )
    print(
        f"  multipart: {len(merged):,} groups merged in {n_merged_convs:,} conversations "
        f"({merged_pct:.2f}%); {n_reordered:,} re-ordered by part marker over created_at"
    )
    print(
        f"  orphan:    {n_orphan:,} replies carry a part marker with no sibling to merge "
        f"(half-answers, flagged orphan_part); {n_orphan_first:,} are a first_reply"
    )
    print(
        f"  signature: {n_signed:,} replies carried a /XX signature, extracted to "
        f"agent_signature ({n_multi_signed:,} merged replies signed by >1 agent)"
    )
    print(
        f"  first reply: {n_substantive:,} conversations have a substantive first reply "
        f"({n_kept - n_substantive:,} are deflection-only, kept with nulls)"
    )


def transform(assigned: pd.DataFrame, brand: str = BRAND) -> tuple:
    """Every step from assigned tweets to conversation rows. No I/O, no printing.

    Returns (ordered, turns, out, n_reordered, selected) so tests and the
    summary can both see the intermediates rather than only the nested output.
    """
    selected = select_brand_conversations(assigned, brand)
    capped = drop_long_conversations(selected)

    ordered = assign_merge_groups(capped)
    ordered = ordered.assign(part_number=extract_part_numbers(ordered))
    ordered, n_reordered = apply_marker_order(ordered)
    ordered = strip_signatures(ordered)
    # After apply_marker_order, so "continuation" means continuation in the
    # FINAL turn order rather than in clock order.
    ordered = strip_continuation_prefixes(ordered)
    ordered = strip_stale_first_part_marker(ordered)

    turns = build_turns(ordered)
    out = build_conversation_rows(ordered, turns)
    return ordered, turns, out, n_reordered, selected


def build_sample(interim: Path = INTERIM) -> pd.DataFrame:
    """Run the whole stage and report. Returns one row per conversation."""
    assigned = load_assigned(interim)
    ordered, turns, out, n_reordered, selected = transform(assigned)
    _print_summary(assigned, selected, ordered, turns, out, n_reordered)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=SPOTIFY_PARQUET)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    out = build_sample()
    if not args.no_write:
        path = write_parquet(out, args.out)
        print(f"  wrote {_display_path(path)} ({len(out):,} rows)")


if __name__ == "__main__":
    main()
