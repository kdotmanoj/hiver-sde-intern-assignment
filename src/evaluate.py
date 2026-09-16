"""Score the agent and both baselines against the golden set.

Reads the three prediction files written by `make agent` and `make baselines`
and joins them to data/golden/labels.jsonl on conversation_id. It does not call
a model, load an embedding model, or touch data/raw/ -- every number comes from
jsonl already on disk, which is what lets `make eval` run from a fresh clone
with no API key in a couple of seconds.

All scoring goes through src/metrics.py. Nothing is computed inline here, so the
report's numbers and the unit-tested implementations cannot drift apart.

Four things the tables are built to make visible:

1. Macro-F1 with a bootstrap CI, not accuracy. Nine classes with a 21% majority
   means accuracy flatters the trivial baseline; the CI says whether a gap
   between two systems survives the 150-row sample size.

2. The bootstrap resamples are PAIRED -- one set of 1000 index lists, reused for
   every system and every metric. A difference between two intervals is then a
   difference between the systems, not between two random draws.

3. Undefined precision and recall are printed, never hidden. Both baselines
   never escalate, so their escalate precision is 0/0; it shows as 0.0 with the
   class named, which is the honest reading. See src/metrics.py.

4. Per-family escalation scores answer the question src/escalate.py's docstring
   raises and cannot answer alone: do the similarity and text families add
   anything over the intent lookup?
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.agent import GOLDEN_LABELS, INTENTS, INTERIM
from src.escalate import FAMILIES
from src.ingest import _display_path
from src import metrics

# Order matters only for display: the agent first, then the floors it has to
# clear, in the order src/baselines.py introduces them.
SYSTEMS = ("agent", "trivial", "knn")

# Which make target writes each file, so a missing one names its own fix.
PRODUCED_BY = {
    "agent": "make agent",
    "trivial": "make baselines",
    "knn": "make baselines",
}

N_GOLDEN = 150


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_golden(path: Path = GOLDEN_LABELS) -> list[dict]:
    """The hand labels, sorted by conversation_id.

    Sorted, not in file order, because the file is in labelling order and the
    paired bootstrap needs every system's row i to be the same conversation.
    """
    golden = read_jsonl(path)

    ids = [row["conversation_id"] for row in golden]
    assert len(set(ids)) == len(ids), f"{len(ids) - len(set(ids))} duplicate ids in {path}"

    return sorted(golden, key=lambda row: row["conversation_id"])


def load_predictions(name: str, golden: list[dict], interim: Path) -> list[dict]:
    """One system's results, reordered to match `golden` row for row.

    Exits with a message rather than a traceback when the file is absent: the
    agent file only exists after a ~20 minute live run, and "which make target
    do I run" is the only useful thing to say at that point.
    """
    path = interim / f"{name}_golden.jsonl"

    if not path.exists():
        sys.exit(
            f"missing {_display_path(path)} -- run `{PRODUCED_BY[name]}` first.\n"
            f"No scores can be printed without it; evaluate.py never calls a model itself."
        )

    results = read_jsonl(path)
    by_id = {row["conversation_id"]: row for row in results}

    assert len(by_id) == len(results), f"duplicate conversation_id in {_display_path(path)}"

    golden_ids = {row["conversation_id"] for row in golden}
    missing = golden_ids - set(by_id)
    extra = set(by_id) - golden_ids
    assert not missing, f"{_display_path(path)} is missing {len(missing)} golden ids"
    assert not extra, f"{_display_path(path)} has {len(extra)} ids that are not in the golden set"

    aligned = [by_id[row["conversation_id"]] for row in golden]
    assert len(aligned) == len(golden), f"{len(aligned)} aligned rows for {len(golden)} golden"
    return aligned


# --------------------------------------------------------------------------
# pulling the columns out
# --------------------------------------------------------------------------


def intent_columns(golden: list[dict], results: list[dict]) -> tuple[list[str], list[str]]:
    return (
        [row["intent"] for row in golden],
        [row["intent"] for row in results],
    )


def escalate_columns(golden: list[dict], results: list[dict]) -> tuple[list[bool], list[bool]]:
    """Golden escalate flag vs the decision the system actually took."""
    return (
        [bool(row["escalate"]) for row in golden],
        [row["decision"] == "escalate" for row in results],
    )


def family_column(results: list[dict], family: str) -> list[bool]:
    """Would this signal family alone have escalated?

    escalate() is a flat OR over signals, so "family F escalates on its own" is
    exactly "at least one signal of family F fired" -- which is what
    escalate.signals_by_family already split out per row.
    """
    return [bool(row["signals_by_family"][family]) for row in results]


# --------------------------------------------------------------------------
# formatting
# --------------------------------------------------------------------------


def _heading(title: str) -> None:
    print("")
    print(title)
    print("-" * len(title))


def _interval(bounds: tuple[float, float]) -> str:
    low, high = bounds
    return f"[{low:.3f}, {high:.3f}]"


# --------------------------------------------------------------------------
# the tables
# --------------------------------------------------------------------------


def intent_table(scored: dict, indices: list[list[int]]) -> dict:
    """Table 1: one row per system -- accuracy, macro-F1 + CI, kappa + CI."""
    _heading("intent (9 classes, macro-averaged over all nine equally)")
    print(f"{'system':<10}{'accuracy':>10}{'macro-F1':>10}  {'macro-F1 95% CI':<18}{'kappa':>8}")

    summary = {}
    for name in SYSTEMS:
        y_true, y_pred = scored[name]["intent"]

        macro = metrics.macro_f1(y_true, y_pred, INTENTS)
        macro_ci = metrics.bootstrap_ci(
            y_true,
            y_pred,
            lambda truth, pred: metrics.macro_f1(truth, pred, INTENTS),
            indices=indices,
        )
        summary[name] = {
            "accuracy": metrics.accuracy(y_true, y_pred),
            "macro_f1": macro,
            "macro_f1_ci": macro_ci,
            "cohens_kappa": metrics.cohens_kappa(y_true, y_pred, INTENTS),
        }

        print(
            f"{name:<10}{summary[name]['accuracy']:>10.3f}{macro:>10.3f}  "
            f"{_interval(macro_ci):<18}{summary[name]['cohens_kappa']:>8.3f}"
        )

    print("")
    print("  CI: percentile bootstrap, 1000 resamples, seed 42, 2.5th/97.5th.")
    print("  Resamples are paired -- all three systems are scored on the same 1000 draws.")
    return summary


def per_class_table(scored: dict) -> dict:
    """Table 2: per-class F1 for all three systems, side by side, plus support."""
    _heading("intent, per class (F1)")
    print(f"{'class':<20}{'support':>8}" + "".join(f"{name:>10}" for name in SYSTEMS))

    reports = {}
    for name in SYSTEMS:
        y_true, y_pred = scored[name]["intent"]
        reports[name] = metrics.per_class_report(y_true, y_pred, INTENTS)

    for intent in INTENTS:
        support = reports[SYSTEMS[0]]["per_class"][intent]["support"]
        cells = "".join(f"{reports[name]['per_class'][intent]['f1']:>10.3f}" for name in SYSTEMS)
        print(f"{intent:<20}{support:>8}{cells}")

    print("")
    for name in SYSTEMS:
        never_predicted = reports[name]["undefined_precision"]
        if not never_predicted:
            print(f"  {name:<8} predicted every class at least once.")
            continue
        print(f"  {name:<8} never predicted: {', '.join(never_predicted)}")
        print(f"  {'':<8} precision on those is 0/0, reported as 0.000, not 1.000.")

    return reports


def escalate_table(scored: dict, indices: list[list[int]]) -> dict:
    """Table 3: the binary escalate decision, scored on the positive class."""
    _heading("escalate (binary, scored on decision == escalate)")
    print(
        f"{'system':<10}{'P':>8}{'R':>8}{'F1':>8}  {'F1 95% CI':<18}"
        f"{'fired':>7}{'true':>7}"
    )

    summary = {}
    for name in SYSTEMS:
        y_true, y_pred = scored[name]["escalate"]
        report = metrics.per_class_report(y_true, y_pred, [False, True])
        positive = report["per_class"][True]

        f1_ci = metrics.bootstrap_ci(
            y_true,
            y_pred,
            lambda truth, pred: metrics.f1(truth, pred, True),
            indices=indices,
        )
        summary[name] = {
            "precision": positive["precision"],
            "recall": positive["recall"],
            "f1": positive["f1"],
            "f1_ci": f1_ci,
            "n_predicted": positive["n_predicted"],
            "support": positive["support"],
            "undefined_precision": positive["n_predicted"] == 0,
        }

        print(
            f"{name:<10}{positive['precision']:>8.3f}{positive['recall']:>8.3f}"
            f"{positive['f1']:>8.3f}  {_interval(f1_ci):<18}"
            f"{positive['n_predicted']:>7}{positive['support']:>7}"
        )

    print("")
    never = [name for name in SYSTEMS if summary[name]["undefined_precision"]]
    if never:
        print(
            f"  undefined precision (zero predicted positives): {', '.join(never)}. "
            f"Reported as 0.000."
        )
        print("  Both baselines never escalate by construction -- that is the majority-class null.")
    return summary


def family_table(golden: list[dict], results: list[dict], indices: list[list[int]]) -> dict:
    """Table 4: each escalation signal family scored as its own binary rule.

    Agent only. The baselines fire no signals at all, so every family would be
    three rows of zeros with nothing to read.
    """
    _heading("escalate by signal family (agent only)")
    print(
        f"{'family':<12}{'P':>8}{'R':>8}{'F1':>8}  {'F1 95% CI':<18}{'fired':>7}"
    )

    y_true = [bool(row["escalate"]) for row in golden]

    summary = {}
    for family in FAMILIES:
        y_pred = family_column(results, family)
        report = metrics.per_class_report(y_true, y_pred, [False, True])
        positive = report["per_class"][True]

        f1_ci = metrics.bootstrap_ci(
            y_true,
            y_pred,
            lambda truth, pred: metrics.f1(truth, pred, True),
            indices=indices,
        )
        summary[family] = {
            "precision": positive["precision"],
            "recall": positive["recall"],
            "f1": positive["f1"],
            "f1_ci": f1_ci,
            "n_fired": positive["n_predicted"],
            "undefined_precision": positive["n_predicted"] == 0,
        }

        print(
            f"{family:<12}{positive['precision']:>8.3f}{positive['recall']:>8.3f}"
            f"{positive['f1']:>8.3f}  {_interval(f1_ci):<18}{positive['n_predicted']:>7}"
        )

    print("")
    print(f"  {sum(y_true)} of {len(y_true)} golden rows are labelled escalate.")
    print("  Each row is that family alone as the whole rule, not its marginal contribution.")
    print("  Baselines are omitted: they fire zero signals by construction.")
    return summary


def confusion_table(scored: dict, system: str = "agent") -> list[list[int]]:
    """Table 5: the agent's 9x9 confusion matrix, rows true, columns predicted."""
    _heading(f"{system} confusion matrix (rows = true, columns = predicted)")

    y_true, y_pred = scored[system]["intent"]
    matrix = metrics.confusion_matrix(y_true, y_pred, INTENTS)

    for position, intent in enumerate(INTENTS):
        print(f"  {position}  {intent}")
    print("")

    header = "".join(f"{position:>5}" for position in range(len(INTENTS)))
    print(f"{'':<22}{header}{'total':>7}")
    for position, intent in enumerate(INTENTS):
        row = matrix[position]
        print(f"{position} {intent:<20}" + "".join(f"{cell:>5}" for cell in row) + f"{sum(row):>7}")

    total = sum(sum(row) for row in matrix)
    assert total == len(y_true), f"confusion matrix totals {total}, expected {len(y_true)}"
    return matrix


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--golden-labels", type=Path, default=GOLDEN_LABELS)
    parser.add_argument("--interim", type=Path, default=INTERIM)
    parser.add_argument("--json", type=Path, help="also write every number to this path")
    parser.add_argument(
        "--n-resamples",
        type=int,
        default=metrics.N_RESAMPLES,
        help=f"bootstrap resamples (default {metrics.N_RESAMPLES})",
    )
    parser.add_argument("--seed", type=int, default=metrics.SEED)
    args = parser.parse_args()

    golden = load_golden(args.golden_labels)
    assert len(golden) == N_GOLDEN, f"{len(golden)} golden labels, expected {N_GOLDEN}"

    predictions = {name: load_predictions(name, golden, args.interim) for name in SYSTEMS}

    scored = {}
    for name in SYSTEMS:
        scored[name] = {
            "intent": intent_columns(golden, predictions[name]),
            "escalate": escalate_columns(golden, predictions[name]),
        }

    counts = " -> ".join(f"{name} {len(predictions[name]):,}" for name in SYSTEMS)
    print(f"evaluate: golden {len(golden):,} -> {counts} (matched {len(golden):,}, lost 0)")

    # One index set for everything below: paired resamples across systems,
    # metrics and families. See the module docstring.
    indices = metrics.bootstrap_indices(len(golden), n_resamples=args.n_resamples, seed=args.seed)

    # Called in printing order, one table at a time, so the transcript reads
    # top to bottom the way the report cites it.
    report: dict = {
        "n_golden": len(golden),
        "n_resamples": args.n_resamples,
        "seed": args.seed,
    }
    report["intent"] = intent_table(scored, indices)
    report["intent_per_class"] = per_class_table(scored)
    report["escalate"] = escalate_table(scored, indices)
    report["escalate_by_family"] = family_table(golden, predictions["agent"], indices)
    report["agent_confusion_matrix"] = {
        "labels": list(INTENTS),
        "matrix": confusion_table(scored, "agent"),
    }

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, default=str)
        print("")
        print(f"  wrote {_display_path(args.json)}")


if __name__ == "__main__":
    main()
