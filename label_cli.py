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

--relabel: the second pass
--------------------------

`--relabel` draws 50 of the already-labelled conversations and asks for them
again, blind, writing to data/golden/labels_pass2.jsonl. `--score` then joins
the two files on conversation_id and prints Cohen's kappa, raw agreement and
the disagreements; it writes nothing. The two flags cannot be combined --
--score displays pass-1 labels, which is exactly what a --relabel session must
not do.

Self-agreement is a number the report can put beside the classifier's kappa:
the agent is being scored against labels whose own author reproduces them only
so often, and the gap between those two figures is the part of the agent's
error that is not the agent's.

Blind means blind. The pass-1 intent is read exactly once, to stratify the
draw so all nine labels are represented, and is dropped before the session
starts -- see select_relabel(), which asserts that no pass-1 answer survives
into the frame the session iterates. Nothing about pass 1 is printed at any
point, including the per-intent quota breakdown that the pass-1 summary line
prints for clusters.

The draw is seeded 43, not 42, so it is not the same subset order the first
pass was labelled in.

What pass 2 deliberately does NOT change is the information on screen: the
same text, cluster id, word count and escalation rule. Showing less would
measure the effect of removing context, and showing more would measure the
effect of adding it; neither is label stability.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from src import metrics
from src.embed import embed
from src.ingest import REPO_ROOT, _display_path
from src.taxonomy import SHORT_WORDS, allocate, cluster, draw_sample, load_eligible

LABELS_PATH = REPO_ROOT / "data" / "golden" / "labels.jsonl"
PASS2_PATH = REPO_ROOT / "data" / "golden" / "labels_pass2.jsonl"
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

# The second pass. A different seed from SEED on purpose: reusing 42 would draw
# the 50 in the same relative order the first pass met them in, and order is one
# of the things a second pass is supposed to vary.
RELABEL_SEED = 43
N_RELABEL = 50

# Keys that carry a pass-1 ANSWER. None of these may reach the session frame --
# enforced by assert_blind(), not by care. The failure being guarded against is
# silent: a leaked column does not crash, it just quietly turns the kappa into a
# measurement of how well I can copy.
PASS1_ANSWER_KEYS = ("intent", "escalate", "other_flags", "hard", "note", "labelled_at")

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

# The nine names in the guide's order, for anything that needs a label set --
# kappa and the confusion table both do. Derived from INTENTS rather than
# retyped, so a renamed intent cannot end up with two spellings in one file.
INTENT_NAMES = tuple(INTENTS.values())

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
# selection: the blind second pass
# --------------------------------------------------------------------------


def read_intents(path: Path, what: str) -> dict[int, str]:
    """conversation_id -> intent, from a labels file.

    Returns intents rather than whole records: on the relabel path, the escalate
    flag, the sub-flags, the hard flag and the note are pass-1 answers that have
    no business being loaded at all.
    """
    records = read_done(path)
    if not records:
        raise FileNotFoundError(f"no labels in {_display_path(path)} -- {what}")
    return {cid: record["intent"] for cid, record in records.items()}


def read_pass1_intents(path: Path | None = None) -> dict[int, str]:
    """The only read of pass 1 on the RELABEL path, called before the session.

    (--score reads pass 1 too, but that is a separate invocation which runs no
    session; argparse makes the two flags mutually exclusive so a scoring run
    can never put pass-1 labels on screen while there is labelling to do.)
    """
    return read_intents(
        path or LABELS_PATH,
        "there is nothing to re-label. Run the first pass before the second.",
    )


def assert_blind(selected: pd.DataFrame) -> None:
    """Fail loudly if any pass-1 answer survived into the session frame.

    Cheap, and the only thing standing between a leaked column and a kappa that
    silently measures recall of my own earlier answers instead of agreement
    with them.
    """
    leaked = [column for column in selected.columns if column in PASS1_ANSWER_KEYS]
    assert not leaked, f"pass-1 answers leaked into the relabel frame: {leaked}"


def select_relabel(
    pool: pd.DataFrame,
    intents: dict[int, str],
    total: int = N_RELABEL,
    seed: int = RELABEL_SEED,
) -> pd.DataFrame:
    """`total` already-labelled conversations, stratified by their pass-1 intent.

    Stratifying by the pass-1 label is the point: a uniform draw of 50 from 150
    would take ~2 of the rare classes and the per-class agreement would rest on
    almost nothing. It does mean the pass-1 labels decide WHICH conversations
    come back, which is unavoidable -- and harmless, because what is measured is
    the label given to each one, not which ones were chosen.

    The intent column exists inside this function and nowhere outside it. The
    returned frame is shuffled, so the strata are not presented in blocks: a
    labeller who met nine playback_library messages in a row would infer the
    grouping and, from it, the pass-1 answer.
    """
    labelled = pool[pool["conversation_id"].isin(intents)]

    missing = set(intents) - set(labelled["conversation_id"])
    assert not missing, (
        f"{len(missing)} labelled conversations are not in the pool, e.g. "
        f"{sorted(missing)[:3]} -- labels.jsonl came from a different draw than this one"
    )

    # Materialized rather than chained: `stratum` is a pass-1 answer living in a
    # frame, and it must be visible where it is added and where it is dropped.
    with_intent = labelled.assign(stratum=labelled["conversation_id"].map(intents))
    counts = with_intent.groupby("stratum", dropna=False).size().sort_index()
    quotas = allocate(counts, total)

    parts = [
        with_intent[with_intent["stratum"] == stratum].sample(quota, random_state=seed)
        for stratum, quota in quotas.items()
        if quota > 0
    ]

    selected = pd.concat(parts).drop(columns="stratum")
    assert len(selected) == total, f"selected {len(selected)}, wanted {total}"
    assert selected["conversation_id"].is_unique, "the same conversation was drawn twice"

    shuffled = selected.sample(frac=1, random_state=seed).reset_index(drop=True)
    assert_blind(shuffled)
    return shuffled


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


def label(selected: pd.DataFrame, done: dict[int, dict], path: Path, pass_number: int = 1) -> int:
    """Prompt for every unlabelled row in order. Returns how many were added.

    `path` is required and never defaulted. append() falls back to LABELS_PATH
    when given None, so a label() that forgot to pass one through would write
    the second pass's answers into the first pass's file and destroy the thing
    being measured -- with no error, and no way back except git.
    """
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
            # On every record, both passes, so a merged file is never ambiguous
            # about which answer came from which sitting.
            "pass": pass_number,
        }
        append(record, path)
        added += 1

        flag_text = (" +" + ",".join(flags)) if flags else ""
        print(
            f"  saved {intent}{flag_text} / {'ESCALATE' if escalate else 'auto'}"
            f"{' / HARD' if hard else ''}   {record['labelled_at']}   "
            f"{len(done) + added}/{total}"
        )

    return added


# --------------------------------------------------------------------------
# scoring the two passes against each other
# --------------------------------------------------------------------------


def disagreements(pass1: list[str], pass2: list[str], ids: list[int]) -> list[dict]:
    """Every (pass-1 label, pass-2 label) pair that differs, with its count.

    A full 9x9 matrix over 50 rows would be 81 cells with ~8 non-zero, so this
    lists only the off-diagonal pairs that actually occurred. Sorted by count
    descending, then by label order, so a rerun prints the same table.
    """
    pairs: dict[tuple[str, str], list[int]] = {}
    for first, second, conversation_id in zip(pass1, pass2, ids):
        if first != second:
            pairs.setdefault((first, second), []).append(conversation_id)

    names = list(INTENT_NAMES)
    return sorted(
        (
            {"pass1": first, "pass2": second, "n": len(found), "ids": found}
            for (first, second), found in pairs.items()
        ),
        key=lambda row: (-row["n"], names.index(row["pass1"]), names.index(row["pass2"])),
    )


def score_passes(labels_path: Path | None = None, pass2_path: Path | None = None) -> dict:
    """Cohen's kappa between the two passes. Prints only; writes nothing.

    What this measures is SELF-agreement: one labeller, the same guide, two
    sittings. That is not inter-annotator agreement, and it is the more
    forgiving of the two -- a second person would not share my particular
    reading of the guide's edge cases.
    """
    labels_path = labels_path or LABELS_PATH
    pass2_path = pass2_path or PASS2_PATH

    pass1 = read_intents(labels_path, "run the first pass before scoring.")
    pass2 = read_intents(
        pass2_path, "run `python label_cli.py --relabel` before scoring."
    )

    # Inner join. A partial second pass is the normal case mid-way through, so
    # scoring what exists beats refusing until all 50 are done.
    common = sorted(set(pass1) & set(pass2))
    if not common:
        sys.exit(
            f"no conversation_id appears in both {_display_path(labels_path)} and "
            f"{_display_path(pass2_path)} -- nothing to compare."
        )

    orphans = sorted(set(pass2) - set(pass1))
    if orphans:
        print(
            f"  warning: {len(orphans)} pass-2 ids are not in pass 1, e.g. {orphans[:3]} "
            f"-- excluded from the join"
        )

    y1 = [pass1[cid] for cid in common]
    y2 = [pass2[cid] for cid in common]

    kappa = metrics.cohens_kappa(y1, y2, INTENT_NAMES)
    agreement = metrics.accuracy(y1, y2)
    n_agreed = sum(1 for first, second in zip(y1, y2) if first == second)

    print(
        f"label_cli --score: pass 1 {len(pass1)} -> pass 2 {len(pass2)} -> "
        f"joined {len(common)} (agreed {n_agreed}, differed {len(common) - n_agreed})"
    )
    print("")
    print(f"  raw agreement:   {agreement:.3f}  ({n_agreed}/{len(common)})")
    print(f"  Cohen's kappa:   {kappa:.3f}")
    print("")
    print("  Raw agreement counts the agreement two coin-flippers would get by chance;")
    print("  kappa subtracts it. Both are over the same 9-class label set.")
    print("  This is one labeller twice, not two labellers -- self-agreement, which is")
    print("  the looser of the two measures.")

    # Both passes using a single label makes expected agreement 1.0, so kappa is
    # 0/0 and src/metrics.py returns 0.0 by documented choice. Printing a bare
    # "agreement 1.000, kappa 0.000" next to each other reads as a bug in the
    # kappa; it is not, and the reason belongs on screen where it happens.
    if len(set(y1) | set(y2)) == 1:
        print("")
        print(
            f"  NOTE: every joined row is {y1[0]!r} in both passes. Expected agreement is"
        )
        print("  then 1.0, kappa is 0/0, and metrics.cohens_kappa returns 0.0 by choice.")
        print("  The kappa above is not meaningful on this subset -- score more rows.")

    rows = disagreements(y1, y2, common)
    if not rows:
        print("")
        print("  No disagreements.")
        return {"n": len(common), "kappa": kappa, "agreement": agreement, "disagreements": rows}

    print("")
    print(f"  the {len(common) - n_agreed} disagreements (pass 1 -> pass 2)")
    print(f"  {'pass 1':<20}{'pass 2':<20}{'n':>3}   conversation ids")
    for row in rows:
        shown = ", ".join(str(cid) for cid in row["ids"][:4])
        more = f", +{len(row['ids']) - 4}" if len(row["ids"]) > 4 else ""
        print(f"  {row['pass1']:<20}{row['pass2']:<20}{row['n']:>3}   {shown}{more}")

    # Which labels I am least stable on. Counted over both directions, because a
    # label that is confused in either direction is one whose boundary is soft.
    involved: dict[str, int] = {}
    for row in rows:
        involved[row["pass1"]] = involved.get(row["pass1"], 0) + row["n"]
        involved[row["pass2"]] = involved.get(row["pass2"], 0) + row["n"]

    ranked = sorted(involved.items(), key=lambda item: (-item[1], INTENT_NAMES.index(item[0])))
    print("")
    print("  labels involved in a disagreement, either direction:")
    print("    " + "  ".join(f"{name} {count}" for name, count in ranked))

    return {"n": len(common), "kappa": kappa, "agreement": agreement, "disagreements": rows}


def run_session(selected: pd.DataFrame, path: Path, pass_number: int) -> None:
    """Resume, report, prompt. Shared by both passes so they behave identically."""
    done = read_done(path)

    # Labels from an older draw would make the progress count lie, so say so
    # rather than quietly ignoring them.
    stray = sorted(set(done) - set(selected["conversation_id"]))
    if stray:
        print(f"  warning: {len(stray)} labelled ids are not in the current draw, e.g. {stray[:3]}")

    in_draw = {cid: record for cid, record in done.items() if cid in set(selected["conversation_id"])}
    print(
        f"  progress:    {len(in_draw)}/{len(selected)} already answered, "
        f"{len(selected) - len(in_draw)} remaining"
    )
    print(f"  writing to:  {_display_path(path)}")

    try:
        added = label(selected, in_draw, path, pass_number=pass_number)
    except KeyboardInterrupt:
        print("\n  stopped. Everything answered so far is saved; rerun to continue.")
        sys.exit(0)

    print(f"\n  done. {added} labelled this session, {len(in_draw) + added}/{len(selected)} total.")


def run_pass1(pool: pd.DataFrame) -> None:
    candidates = pool[~pool["is_exemplar"]]
    selected = select(candidates)

    short = int(selected["is_short"].sum())
    per_cluster = selected.groupby("cluster", dropna=False).size().sort_index()
    n_exemplars = int(pool["is_exemplar"].sum())

    print(
        f"label_cli: pool {len(pool):,} -> unread {len(candidates):,} "
        f"(dropped {n_exemplars} read exemplars) -> selected {len(selected)}"
    )
    print(f"  clusters:    " + " / ".join(f"c{c} {n}" for c, n in per_cluster.items()))
    print(f"  length:      {short} under {SHORT_WORDS} words ({100.0 * short / len(selected):.1f}%)")
    print(f"  guide:       {_display_path(GUIDE_PATH)}")

    run_session(selected, LABELS_PATH, pass_number=1)


def run_relabel(pool: pd.DataFrame) -> None:
    """The blind second pass. Reads pass 1 to stratify, then never again."""
    if PASS2_PATH == LABELS_PATH:
        raise AssertionError("the second pass must not write to the first pass's file")

    intents = read_pass1_intents()
    selected = select_relabel(pool, intents)

    short = int(selected["is_short"].sum())
    per_cluster = selected.groupby("cluster", dropna=False).size().sort_index()

    print(
        f"label_cli --relabel: labelled {len(intents)} -> selected {len(selected)} "
        f"for a blind second pass (seed {RELABEL_SEED})"
    )
    print(f"  clusters:    " + " / ".join(f"c{c} {n}" for c, n in per_cluster.items()))
    print(f"  length:      {short} under {SHORT_WORDS} words ({100.0 * short / len(selected):.1f}%)")
    print(f"  guide:       {_display_path(GUIDE_PATH)}")
    # The per-intent quota is NOT printed, though the per-cluster spread is.
    # Cluster is a property of the text that pass 1 also had on screen; the
    # intent breakdown is a pass-1 answer, and this session shows none of those.
    print("  blind:       pass-1 labels are not read again or shown at any point")

    run_session(selected, PASS2_PATH, pass_number=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])

    # Mutually exclusive, and not merely for tidiness: --score puts pass-1
    # labels on screen, which is the one thing a --relabel session must never
    # do. Making them un-combinable in argparse means no invocation exists that
    # could do both.
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--relabel",
        action="store_true",
        help=f"blind second pass: re-label {N_RELABEL} already-labelled conversations "
        f"into {PASS2_PATH.name} (pass-1 answers are never shown)",
    )
    mode.add_argument(
        "--score",
        action="store_true",
        help="print Cohen's kappa between the two passes and the disagreements "
        "(reads both files, writes nothing)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # A missing labels file is an expected state -- the second pass has not been
    # run yet, or neither has -- and the message already says which command
    # fixes it. A traceback would bury that under a stack nobody needs.
    try:
        # Scoring reads two jsonl files and nothing else: no parquet, no
        # embedding model, no clustering. build_pool() would cost ~10s and load
        # MiniLM for a table that needs neither.
        if args.score:
            score_passes()
            return

        pool = build_pool()

        if args.relabel:
            run_relabel(pool)
        else:
            run_pass1(pool)
    except FileNotFoundError as error:
        sys.exit(f"\n{error}")


if __name__ == "__main__":
    main()
