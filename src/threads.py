"""Reconstruct conversations by walking the reply graph. Stage 2 of the pipeline.

This module is written as plain, explicit code — dicts, lists and an explicit
stack, no graph library. The traversal rules below are decisions I have to be
able to derive from scratch, so they are spelled out rather than delegated.

The two structural facts everything rests on:

1. `in_response_to_tweet_id` is a SINGLE id, so every tweet has at most one
   parent. The parent edges therefore form a forest: no tweet can be claimed
   by two roots, and threads partition the data exactly. That is what makes
   "one row per input tweet, lost 0" an assertable invariant.

2. `response_tweet_id` is a comma-separated LIST and is NOT used to build
   edges. It can disagree with the parent column (A lists C as a response
   while C's parent is B), and drawing edges from it would give C two parents
   and break the partition. The parent column is the single source of truth;
   `response_tweet_id` is only cross-checked, and disagreements are counted
   and reported.

Consequently a reconstructed thread is always a tree, never a DAG. Branches
cannot reconverge in this view, and the reported disagreement count is the
honest error bar on that claim.

Tweet ids are NOT chronological — tweet 1 can have parent 3 and reply 2, so
the real order is 3 -> 1 -> 2. Nothing here sorts by id. Siblings are ordered
by `created_at`, with `tweet_id` only as a deterministic tiebreak for
identical timestamps. Because that ordering depends entirely on timestamps,
the summary reports how many edges run backwards in time.

Nothing is dropped silently. Every tweet lands in exactly one `root_kind`
bucket: customer_root, brand_root, orphan (parent absent from the file) or
cycle (unreachable from any root).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.ingest import REPO_ROOT, _display_path, load_cached_or_raw, write_parquet

# Categories a tweet's thread can be rooted in. Every row gets exactly one.
ROOT_KINDS = ("customer_root", "brand_root", "orphan", "cycle")


def build_threads(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per input tweet, annotated with its thread structure.

    Columns: tweet_id, thread_id, root_id, root_kind, parent_id, depth,
    branch_id, n_children, is_leaf.

    `depth` is the tweet's distance from its root, so it is also its position
    in the root-to-tweet path. It is -1 for cycle-quarantined tweets, which
    have no root and therefore no well-defined depth.

    `branch_id` is a LEAF-PATH COUNTER, not a filter key. Each tweet carries
    exactly one branch_id — the number of the first root-to-leaf path that
    reached it — so a shared ancestor belongs to only one branch_id even
    though it is part of several dialogues. Selecting `df.branch_id == b`
    therefore does NOT give a linear dialogue: for every branch except the
    first, it silently omits the shared ancestors. Use `path_to_root()` for
    that. branch_id answers only "how many distinct endings does this thread
    have, and which leaf's path first claimed this tweet".
    """
    ids = df["tweet_id"].tolist()
    parents_raw = df["in_response_to_tweet_id"].tolist()
    inbound_raw = df["inbound"].tolist()

    id_set = set(ids)

    # parent_of maps every tweet to its parent id, or None when the parent
    # column is null OR names a tweet that is not in the file. Collapsing both
    # to None here is deliberate: in both cases the tweet starts a thread. The
    # two are told apart by root_kind below, not by the edge structure.
    parent_of: dict[int, int | None] = {}
    n_orphans = 0
    for tweet_id, parent in zip(ids, parents_raw):
        # pd.isna covers None, NaN and pd.NA in one check. The column is
        # nullable Int64 (pinned in ingest), so missing values are pd.NA,
        # whose truthiness raises rather than being falsy.
        if pd.isna(parent):
            parent_of[tweet_id] = None
        elif int(parent) not in id_set:
            parent_of[tweet_id] = None
            n_orphans += 1
        else:
            parent_of[tweet_id] = int(parent)

    # Invert into child lists. This is the only edge set used for traversal.
    children: dict[int, list[int]] = {}
    for tweet_id, parent in parent_of.items():
        if parent is not None:
            children.setdefault(parent, []).append(tweet_id)

    _sort_siblings(children, df)

    root_kind_of = _classify_roots(ids, parents_raw, inbound_raw, parent_of, id_set)
    rows = _walk_from_roots(root_kind_of, children)
    cycle_rows, n_cycles = _quarantine_cycles(ids, parent_of, visited=set(rows))
    rows.update(cycle_rows)

    assert len(rows) == len(df), (
        f"thread assignment covers {len(rows):,} tweets but the input has "
        f"{len(df):,}. Every tweet must land in exactly one thread."
    )

    out = _assemble(df, rows, parent_of, children)
    _print_summary(
        out,
        df=df,
        parent_of=parent_of,
        children=children,
        id_set=id_set,
        n_orphans=n_orphans,
        n_cycles=n_cycles,
    )
    return out


# --------------------------------------------------------------------------
# traversal
# --------------------------------------------------------------------------


def _sort_siblings(children: dict[int, list[int]], df: pd.DataFrame) -> None:
    """Order each child list by (created_at, tweet_id), in place.

    created_at comes first because tweet ids are not chronological. tweet_id
    breaks ties so two runs never disagree.
    """
    timestamps = _timestamp_map(df)
    for parent, kids in children.items():
        kids.sort(key=lambda tweet_id: (timestamps[tweet_id], tweet_id))


def _timestamp_map(df: pd.DataFrame) -> dict[int, int]:
    """tweet_id -> created_at as an integer, for cheap comparison.

    The absolute value (epoch nanoseconds or microseconds, depending on the
    column's unit) is never used — only the ordering — so the unit does not
    matter as long as it is the same for every row, which it is.
    """
    as_ints = df["created_at"].astype("int64")
    return dict(zip(df["tweet_id"].tolist(), as_ints.tolist()))


def _classify_roots(
    ids: list[int],
    parents_raw: list,
    inbound_raw: list,
    parent_of: dict[int, int | None],
    id_set: set[int],
) -> dict[int, str]:
    """Find every traversal seed and label why it is one.

    A tweet is a root when parent_of says None. That happens two ways, and the
    difference matters downstream:

    - the parent column was null      -> customer_root if inbound else brand_root
    - the parent column named a tweet
      that is not in the file         -> orphan

    An orphan looks like a customer's opening message but is actually a
    fragment starting mid-conversation, which is why it gets its own label
    rather than being folded into customer_root.
    """
    root_kind_of: dict[int, str] = {}
    for tweet_id, parent, inbound in zip(ids, parents_raw, inbound_raw):
        if parent_of[tweet_id] is not None:
            continue
        if not pd.isna(parent):
            # parent_of said None but the column is not null, so the named
            # parent is simply absent from the file.
            root_kind_of[tweet_id] = "orphan"
        elif bool(inbound):
            root_kind_of[tweet_id] = "customer_root"
        else:
            root_kind_of[tweet_id] = "brand_root"
    return root_kind_of


def _walk_from_roots(
    root_kind_of: dict[int, str], children: dict[int, list[int]]
) -> dict[int, tuple]:
    """Iterative depth-first walk from every root. Returns tweet_id -> row tuple.

    The stack is explicit because the data is 2.8M tweets deep in the worst
    case and recursion would overflow. `visited` makes a cycle impossible to
    loop on even if one were somehow reachable from a root.

    branch_id increments when a leaf is reached, so the first root-to-leaf
    path is branch 0, the next sibling subtree is branch 1, and so on. A tweet
    keeps the branch_id of the first path that reached it — see the caveat in
    build_threads' docstring.
    """
    rows: dict[int, tuple] = {}

    # Roots in id order only so that runs are reproducible; this is a
    # deterministic tiebreak, not a claim that ids mean anything.
    for root_id in sorted(root_kind_of):
        kind = root_kind_of[root_id]
        branch_id = 0
        stack = [(root_id, 0)]
        while stack:
            tweet_id, depth = stack.pop()
            if tweet_id in rows:
                continue
            rows[tweet_id] = (root_id, root_id, kind, depth, branch_id)

            kids = children.get(tweet_id, ())
            if not kids:
                branch_id += 1  # finished a root-to-leaf path
                continue
            # Reversed, because a LIFO stack pops the last item first and we
            # want the earliest sibling explored first.
            for child in reversed(kids):
                stack.append((child, depth + 1))

    return rows


def _quarantine_cycles(
    ids: list[int], parent_of: dict[int, int | None], visited: set[int]
) -> tuple[dict[int, tuple], int]:
    """Assign the tweets no root can reach. Returns their rows and a cycle count.

    Every tweet has at most one parent, so walking up from any tweet either
    ends at a root or revisits a tweet. Anything not reached from a root is
    therefore on a parent cycle (A -> B -> A) or hanging off one. Both are
    quarantined into a thread named after the smallest id on their cycle,
    with depth -1, and counted — never dropped.
    """
    rows: dict[int, tuple] = {}
    thread_of: dict[int, int] = {}
    n_cycles = 0

    for start in sorted(ids):
        if start in visited or start in rows:
            continue

        # Walk up until we repeat a tweet on this path or reach one already
        # assigned. `position` lets us slice the cycle out of the path.
        path: list[int] = []
        position: dict[int, int] = {}
        node = start
        while node is not None and node not in visited and node not in position:
            if node in thread_of:
                break
            position[node] = len(path)
            path.append(node)
            node = parent_of[node]

        if node is not None and node in position:
            # Closed a loop: everything from where we re-entered is the cycle.
            cycle = path[position[node] :]
            thread_id = min(cycle)
            n_cycles += 1
        elif node is not None and node in thread_of:
            thread_id = thread_of[node]  # hanging off an already-found cycle
        else:
            # Reached a root or an already-visited tweet without closing a
            # loop. The root walk should have covered this, so it is a bug.
            raise AssertionError(
                f"tweet {start} is unreachable from any root but its parent "
                f"chain ends at {node!r} without a cycle."
            )

        for tweet_id in path:
            thread_of[tweet_id] = thread_id
            rows[tweet_id] = (thread_id, thread_id, "cycle", -1, 0)

    return rows, n_cycles


# --------------------------------------------------------------------------
# reading a thread back
# --------------------------------------------------------------------------


def build_parent_map(threads: pd.DataFrame) -> dict[int, int | None]:
    """tweet_id -> parent_id, for repeated path_to_root calls."""
    return {
        int(tweet_id): (None if pd.isna(parent) else int(parent))
        for tweet_id, parent in zip(threads["tweet_id"], threads["parent_id"])
    }


def path_to_root(
    threads: pd.DataFrame, tweet_id: int, parent_map: dict[int, int | None] | None = None
) -> list[int]:
    """Return the root-first chain of tweet ids ending at `tweet_id`.

    This is the correct way to get a linear dialogue — it keeps the ancestors
    a tweet shares with its siblings, which filtering on branch_id would drop.

    Pass `parent_map` from build_parent_map() when calling this in a loop;
    otherwise the map is rebuilt from the frame on every call.

    A quarantined cycle has no root, so the walk stops as soon as it would
    revisit a tweet and returns the chain collected so far.
    """
    if parent_map is None:
        parent_map = build_parent_map(threads)
    if tweet_id not in parent_map:
        raise KeyError(f"tweet {tweet_id} is not in this frame")

    chain: list[int] = []
    seen: set[int] = set()
    node: int | None = tweet_id
    while node is not None and node not in seen:
        seen.add(node)
        chain.append(node)
        node = parent_map.get(node)
    chain.reverse()
    return chain


# --------------------------------------------------------------------------
# assembly and reporting
# --------------------------------------------------------------------------


def _assemble(
    df: pd.DataFrame,
    rows: dict[int, tuple],
    parent_of: dict[int, int | None],
    children: dict[int, list[int]],
) -> pd.DataFrame:
    """Turn the per-tweet tuples into a frame, in the input's row order."""
    ordered = [rows[tweet_id] for tweet_id in df["tweet_id"].tolist()]
    out = pd.DataFrame(
        ordered,
        columns=["thread_id", "root_id", "root_kind", "depth", "branch_id"],
        index=df.index,
    )
    out.insert(0, "tweet_id", df["tweet_id"].to_numpy())
    tweet_ids = out["tweet_id"].tolist()
    # Int64 (nullable) rather than float64: a root's missing parent is NA, and
    # tweet ids should never round-trip through a float.
    out["parent_id"] = pd.array([parent_of[t] for t in tweet_ids], dtype="Int64")
    out["n_children"] = [len(children.get(t, ())) for t in tweet_ids]
    out["is_leaf"] = out["n_children"] == 0

    assert len(out) == len(df), f"assembly changed row count: {len(df)} -> {len(out)}"
    return out


def _count_edge_disagreements(
    df: pd.DataFrame, parent_of: dict[int, int | None], id_set: set[int]
) -> tuple[int, int]:
    """Cross-check response_tweet_id against the parent column.

    Returns (claims naming a tweet not in the file, claims whose target has a
    different parent). Neither is used to build edges — see the module
    docstring — they are the error bar on treating the parent column as truth.
    """
    n_absent = 0
    n_mismatched = 0
    for tweet_id, claims in zip(df["tweet_id"].tolist(), df["response_tweet_id"].tolist()):
        if pd.isna(claims) or not claims:
            continue
        for claim in str(claims).split(","):
            claim = claim.strip()
            if not claim:
                continue
            child = int(claim)
            if child not in id_set:
                n_absent += 1
            elif parent_of[child] != tweet_id:
                n_mismatched += 1
    return n_absent, n_mismatched


def _count_backwards_edges(df: pd.DataFrame, parent_of: dict[int, int | None]) -> tuple[int, int]:
    """Return (edges where the child is older than its parent, total edges).

    Sibling ordering depends entirely on created_at. If a meaningful share of
    edges run backwards in time, that ordering is not trustworthy and the
    summary needs to say so before anything downstream relies on it.
    """
    timestamps = _timestamp_map(df)
    n_backwards = 0
    n_edges = 0
    for tweet_id, parent in parent_of.items():
        if parent is None:
            continue
        n_edges += 1
        if timestamps[tweet_id] < timestamps[parent]:
            n_backwards += 1
    return n_backwards, n_edges


def _print_summary(
    out: pd.DataFrame,
    df: pd.DataFrame,
    parent_of: dict[int, int | None],
    children: dict[int, list[int]],
    id_set: set[int],
    n_orphans: int,
    n_cycles: int,
) -> None:
    """The stage row-count row, plus the counts that qualify what it means."""
    # dropna=False so an unexpected null kind would show up rather than vanish.
    by_kind = out.groupby("root_kind", dropna=False).size()
    roots = out[out["depth"] == 0]
    roots_by_kind = roots.groupby("root_kind", dropna=False).size()
    n_cycle_tweets = int(by_kind.get("cycle", 0))
    n_threads = int(out["thread_id"].nunique())

    n_orphan_threads = int(roots_by_kind.get("orphan", 0))
    n_orphan_tweets = int(by_kind.get("orphan", 0))
    n_absent, n_mismatched = _count_edge_disagreements(df, parent_of, id_set)
    n_backwards, n_edges = _count_backwards_edges(df, parent_of)
    backwards_pct = (100.0 * n_backwards / n_edges) if n_edges else 0.0

    n_branching = sum(1 for kids in children.values() if len(kids) > 1)
    max_depth = int(out["depth"].max())
    max_branches = int(out.groupby("thread_id", dropna=False)["branch_id"].max().max()) + 1

    print(
        f"threads: tweets {len(df):,} -> roots {len(roots):,} "
        f"(customer {int(roots_by_kind.get('customer_root', 0)):,} / "
        f"brand {int(roots_by_kind.get('brand_root', 0)):,} / "
        f"orphan {n_orphan_threads:,}) "
        f"-> threaded tweets {len(out):,} (lost {len(df) - len(out):,})"
    )
    print(f"  threads:   {n_threads:,}")
    print(
        f"  orphans:   {n_orphan_tweets:,} tweets in {n_orphan_threads:,} threads start "
        f"mid-conversation (parent absent from the file), so they look like an "
        f"opening message but are fragments"
    )
    print(f"  cycles:    {n_cycle_tweets:,} tweets in {n_cycles:,} distinct cycles (quarantined)")
    print(
        f"  edges:     {n_absent:,} response_tweet_id claims name a tweet not in the file; "
        f"{n_mismatched:,} name a tweet whose parent disagrees"
    )
    print(
        f"             {n_backwards:,} of {n_edges:,} ({backwards_pct:.2f}%) run backwards "
        f"in time (child older than parent)"
    )
    print(
        f"  branching: {n_branching:,} tweets with >1 child; max depth {max_depth:,}; "
        f"max branches/thread {max_branches:,}"
    )
    if backwards_pct > 1.0:
        print(
            f"  WARNING: {backwards_pct:.2f}% of edges run backwards in time. Sibling "
            f"ordering is by created_at, so branch order is unreliable at this rate."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "interim" / "threads.parquet")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    df = load_cached_or_raw()
    threads = build_threads(df)
    if not args.no_write:
        out = write_parquet(threads, args.out)
        print(f"  wrote {_display_path(out)} ({len(threads):,} rows)")


if __name__ == "__main__":
    main()
