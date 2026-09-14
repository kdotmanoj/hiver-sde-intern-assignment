"""Hand-labelling CLI for the golden set. Not a pipeline stage.

Draws 150 conversation openings and asks, one at a time, for an intent, an
escalate/auto decision, and an optional note. Every answer is appended to
data/golden/labels.jsonl immediately, so the session can be killed at any
point and resumed without losing or repeating work.

Where the 150 come from
-----------------------

The k=8 clusters exist only over the 2,000-opening sample that
src/taxonomy.py draws -- those are the only texts in the embedding cache.
So this script reuses that exact draw (same seed, same stratification),
re-runs k-means at k=8 on the cached vectors, and samples 150 from it,
stratified across the eight clusters so no message type is missed.

The exemplars already exported to notes/clusters_k*.md are excluded: those
are the openings that were read to write the taxonomy, so a rule in the
guide may have been written from the very message it would then be scored
on. Ids come from the notes files themselves rather than from re-deriving
the draw, because what was read is what those files say.

Two consequences worth knowing before quoting numbers off the result:

1. The exclusion removes the openings that were read, not the dependence on
   the pool they were drawn from. The 150 still come from the same 2,000.
2. It is stratified by cluster, not proportional to intent frequency, so
   the label distribution is NOT an estimate of the population's. It is
   built for per-class coverage, which is what macro-F1 needs.

Openings under 5 words are oversampled to SHORT_SHARE of the set (~4x their
natural rate). They carry the least intent signal and are where the
classifier is most likely to fail, so the golden set should hold more of
them than chance would give.
"""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.embed import embed
from src.ingest import REPO_ROOT, _display_path
from src.taxonomy import SHORT_WORDS, allocate, cluster, draw_sample, load_eligible

LABELS_PATH = REPO_ROOT / "data" / "golden" / "labels.jsonl"
GUIDE_PATH = REPO_ROOT / "data" / "golden" / "labelling_guide.md"
NOTES = REPO_ROOT / "notes"

# Exemplar lines in notes/clusters_k*.md look like:
#   - `130876` [customer_root] Spotify has removed all my tracks it appears
# Anchored at the start of the line so a backticked id inside a tweet's own
# text could never be mistaken for an exemplar id.
EXEMPLAR_LINE = re.compile(r"^- `(\d+)`")

SEED = 42
N_LABEL = 150
K = 8

# Share of the 150 reserved for openings under SHORT_WORDS words. They are
# ~2.25% of the sample, so this is roughly a 4x oversample. Shortfall in any
# one cluster is backfilled from that cluster's longer openings.
SHORT_SHARE = 0.10

# Hotkey order is the guide's order. The keys are part of the on-disk record
# only via the name, so renumbering later would not corrupt old labels.
INTENTS = {
    "1": "playback_library",
    "2": "app_bug",
    "3": "content_issue",
    "4": "account_access",
    "5": "billing_dispute",
    "6": "subscription_query",
    "7": "how_to_question",
    "8": "feature_request",
    "9": "other",
}

OTHER_FLAGS = {"n": "nonenglish", "b": "brandpromo", "x": "noask"}

ESCALATION_RULE = """\
ESCALATE if: needs account-specific data Spotify can't see from a tweet
  (charges, account state, verification status) / customer is abusive or
  threatening to cancel / money is actually in dispute / legal or safety
  language.
AUTO otherwise: general how-to, feature requests, known bugs with public
  workarounds, catalogue questions, anything answerable from Spotify's own
  past public replies."""


# --------------------------------------------------------------------------
# selection
# --------------------------------------------------------------------------


def read_exemplars(notes_dir: Path | None = None) -> set[int]:
    """conversation_ids exported to notes/clusters_k*.md -- the ones read.

    Missing files are an error, not an empty set: silently excluding nothing
    would hand back a contaminated draw that looks clean.
    """
    notes_dir = notes_dir or NOTES
    paths = sorted(notes_dir.glob("clusters_k*.md"))
    if not paths:
        raise FileNotFoundError(
            f"no clusters_k*.md in {_display_path(notes_dir)} -- run `make taxonomy` first, "
            "otherwise there is no record of which openings were read"
        )

    ids: set[int] = set()
    for path in paths:
        found = {
            int(match.group(1))
            for line in path.read_text(encoding="utf-8").splitlines()
            if (match := EXEMPLAR_LINE.match(line))
        }
        if not found:
            raise ValueError(f"{_display_path(path)} has no exemplar lines -- has its format changed?")
        ids |= found
    return ids


def build_pool() -> pd.DataFrame:
    """The 2,000-opening taxonomy sample, with its k=8 cluster id attached."""
    eligible, _ = load_eligible()
    sampled = draw_sample(eligible)

    vectors = embed(sampled["opening_text"].tolist())
    labels, _, _ = cluster(vectors, K)
    assert len(labels) == len(sampled), f"{len(labels)} labels for {len(sampled)} rows"

    pool = sampled.assign(
        cluster=labels,
        n_words=sampled["opening_text"].str.split().str.len(),
    )
    pool["is_short"] = pool["n_words"] < SHORT_WORDS

    # Clustering runs on the full 2,000 first: the clusters are defined by
    # that draw, and dropping rows before k-means would move the centroids
    # and stop these being the clusters that were read.
    exemplars = read_exemplars()
    pool["is_exemplar"] = pool["conversation_id"].isin(exemplars)

    stray = exemplars - set(pool["conversation_id"])
    if stray:
        raise ValueError(
            f"{len(stray)} exemplar ids are not in the sample, e.g. {sorted(stray)[:3]} -- "
            "notes/clusters_k*.md was generated from a different draw than this one"
        )
    return pool


def select(pool: pd.DataFrame, total: int = N_LABEL) -> pd.DataFrame:
    """Stratify `total` across the clusters, oversampling short openings.

    Per cluster: take round(quota * SHORT_SHARE) short openings if that many
    exist, then fill the rest from the longer ones. If a cluster is short of
    long openings instead, the shorts absorb the remainder.
    """
    counts = pool.groupby("cluster", dropna=False).size().sort_index()
    quotas = allocate(counts, total)

    parts = []
    for cluster_id, quota in quotas.items():
        members = pool[pool["cluster"] == cluster_id]
        short = members[members["is_short"]]
        long = members[~members["is_short"]]

        n_short = min(len(short), round(quota * SHORT_SHARE))
        n_long = min(len(long), quota - n_short)
        n_short = quota - n_long  # cluster with too few long openings
        assert n_short <= len(short), f"cluster {cluster_id} is smaller than its quota of {quota}"

        parts.append(short.sample(n_short, random_state=SEED))
        parts.append(long.sample(n_long, random_state=SEED))

    selected = pd.concat(parts)
    assert len(selected) == total, f"selected {len(selected)}, wanted {total}"
    assert selected["conversation_id"].is_unique, "the same conversation was drawn twice"

    # Shuffled, so the labeller does not meet the clusters in blocks and start
    # labelling by position instead of by text.
    return selected.sample(frac=1, random_state=SEED).reset_index(drop=True)


# --------------------------------------------------------------------------
# the append-only label file
# --------------------------------------------------------------------------


def read_done(path: Path | None = None) -> dict[int, dict]:
    """Already-labelled records, keyed by conversation_id. Last write wins."""
    path = path or LABELS_PATH
    if not path.exists():
        return {}

    done = {}
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            # A half-written final line is the expected shape of a crash.
            print(f"  warning: {_display_path(path)}:{line_no} is not valid JSON, ignoring it")
            continue
        done[record["conversation_id"]] = record
    return done


def append(record: dict, path: Path | None = None) -> None:
    """Append one record and force it to disk before the next prompt.

    LABELS_PATH is read here rather than bound as a default, so a dry run can
    point the module constant somewhere harmless -- same reason src/embed.py
    resolves its cache dir at call time.
    """
    path = path or LABELS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


# --------------------------------------------------------------------------
# prompting
# --------------------------------------------------------------------------


def ask(prompt: str) -> str:
    """input() where EOF (ctrl-D) reads as a quit rather than a traceback."""
    try:
        return input(prompt).strip()
    except EOFError:
        print()
        return "q"


def show(row, position: int, total: int) -> None:
    hotkeys = "  ".join(f"[{key}] {name}" for key, name in INTENTS.items())

    print()
    print("=" * 78)
    print(f"  {position}/{total}   conversation {row.conversation_id}   "
          f"cluster {row.cluster}   {row.n_words} words"
          + ("   SHORT" if row.is_short else ""))
    print("=" * 78)
    print()
    print("  " + "\n  ".join(str(row.opening_text).splitlines()))
    print()
    print("-" * 78)
    print(ESCALATION_RULE)
    print("-" * 78)
    print(hotkeys)
    print("[s] skip  [q] quit")


def ask_intent() -> str | None:
    """Intent name, or None to skip. 'q' exits the whole session."""
    while True:
        key = ask("intent> ").lower()
        if key in INTENTS:
            return INTENTS[key]
        if key == "s":
            return None
        if key == "q":
            raise KeyboardInterrupt
        print(f"  ? expected 1-9, s or q -- got {key!r}")


def ask_escalate() -> bool:
    while True:
        key = ask("[e] escalate  [a] auto-handle > ").lower()
        if key in ("e", "a"):
            return key == "e"
        if key == "q":
            raise KeyboardInterrupt
        print(f"  ? expected e or a -- got {key!r}")


def ask_other_flags() -> list[str]:
    """Sub-flags for an 'other' label. Multiple letters allowed, e.g. 'nb'."""
    menu = "  ".join(f"[{key}] {name}" for key, name in OTHER_FLAGS.items())
    while True:
        keys = ask(f"other sub-flags -- {menu}  (enter for none) > ").lower().replace(",", "")
        if not keys:
            return []
        unknown = sorted(set(keys) - set(OTHER_FLAGS))
        if unknown:
            print(f"  ? not a sub-flag: {''.join(unknown)!r}")
            continue
        # dict order, not keystroke order, so the same set always serializes
        # the same way.
        return [name for key, name in OTHER_FLAGS.items() if key in keys]


def ask_note_and_hard() -> tuple[str, bool]:
    """Free-text note. Entering 'h' alone toggles the hard flag and re-asks."""
    hard = False
    while True:
        text = ask(f"note (enter to skip, 'h' = hard{', HARD set' if hard else ''}) > ")
        if text.lower() == "h":
            hard = not hard
            continue
        return text, hard


# --------------------------------------------------------------------------


def label(selected: pd.DataFrame, done: dict[int, dict]) -> int:
    """Prompt for every unlabelled row in order. Returns how many were added."""
    total = len(selected)
    added = 0

    for row in selected.itertuples():
        if row.conversation_id in done:
            continue

        show(row, position=len(done) + added + 1, total=total)
        intent = ask_intent()
        if intent is None:
            print("  skipped (it will come back next run)")
            continue

        escalate = ask_escalate()
        flags = ask_other_flags() if intent == "other" else []
        note, hard = ask_note_and_hard()

        record = {
            "conversation_id": int(row.conversation_id),
            "intent": intent,
            "escalate": escalate,
            "other_flags": flags,
            "hard": hard,
            "note": note,
            "cluster": int(row.cluster),
            "n_words": int(row.n_words),
            "opening_text": str(row.opening_text),
            "labelled_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        append(record)
        added += 1

        flag_text = (" +" + ",".join(flags)) if flags else ""
        print(
            f"  saved {intent}{flag_text} / {'ESCALATE' if escalate else 'auto'}"
            f"{' / HARD' if hard else ''}   {record['labelled_at']}   "
            f"{len(done) + added}/{total}"
        )

    return added


def main() -> None:
    pool = build_pool()
    candidates = pool[~pool["is_exemplar"]]
    selected = select(candidates)
    done = read_done()

    # Labels from an older draw would make the progress count lie, so say so
    # rather than quietly ignoring them.
    stray = sorted(set(done) - set(selected["conversation_id"]))
    if stray:
        print(f"  warning: {len(stray)} labelled ids are not in the current draw, e.g. {stray[:3]}")

    in_draw = {cid: record for cid, record in done.items() if cid in set(selected["conversation_id"])}
    short = int(selected["is_short"].sum())
    per_cluster = selected.groupby("cluster", dropna=False).size().sort_index()

    n_exemplars = int(pool["is_exemplar"].sum())
    print(
        f"label_cli: pool {len(pool):,} -> unread {len(candidates):,} "
        f"(dropped {n_exemplars} read exemplars) -> selected {len(selected)} "
        f"(labelled {len(in_draw)}, remaining {len(selected) - len(in_draw)})"
    )
    print(f"  clusters:    " + " / ".join(f"c{c} {n}" for c, n in per_cluster.items()))
    print(f"  length:      {short} under {SHORT_WORDS} words ({100.0 * short / len(selected):.1f}%)")
    print(f"  guide:       {_display_path(GUIDE_PATH)}")
    print(f"  writing to:  {_display_path(LABELS_PATH)}")

    try:
        added = label(selected, in_draw)
    except KeyboardInterrupt:
        print("\n  stopped. Everything answered so far is saved; rerun to continue.")
        sys.exit(0)

    print(f"\n  done. {added} labelled this session, {len(in_draw) + added}/{len(selected)} total.")


if __name__ == "__main__":
    main()
