"""Classification metrics, written out by hand.

One of the three modules CLAUDE.md requires to be plain, explicit code: no
numpy, no pandas, and specifically no sklearn.metrics. Every number here is a
counting loop and a division, because these are the numbers the report makes
claims with and I have to be able to derive each one on a whiteboard.

Nothing in this file knows the taxonomy. Callers pass `labels` explicitly, so
the same functions score the nine-class intent problem and the two-class
escalate decision without a special case.

The 0/0 convention, which is the whole reason this file is careful:

  A class with zero PREDICTED positives has precision 0.0, not 1.0 and not a
  crash. A class with zero TRUE positives has recall 0.0, on the same grounds.
  Both cases are also NAMED, in per_class_report's `undefined_precision` and
  `undefined_recall` lists, because 0.0 read alone is indistinguishable from "we
  predicted this class and got every one of them wrong" -- a very different
  failure. Both baselines predict zero positives on escalate, so this is the
  normal path through the code, not an edge case.

macro_f1 averages over every label passed to it, including labels a system never
predicts. Those score 0.0 and drag the mean down. That is the intended reading
of "all nine classes weighted equally": a system that ignores a class is
penalised for ignoring it.
"""

from __future__ import annotations

import random
from typing import Callable, Sequence

# Defaults for the bootstrap. Seed is CLAUDE.md's project-wide 42; the interval
# is the conventional 95%, i.e. the 2.5th and 97.5th percentiles of the
# resampled metric.
N_RESAMPLES = 1000
SEED = 42
ALPHA = 0.05


# --------------------------------------------------------------------------
# counting
# --------------------------------------------------------------------------


def _check_lengths(y_true: Sequence, y_pred: Sequence) -> int:
    """Both sequences must be the same length and non-empty. Returns that length."""
    if len(y_true) != len(y_pred):
        raise ValueError(f"y_true has {len(y_true)} rows, y_pred has {len(y_pred)}")
    if len(y_true) == 0:
        raise ValueError("cannot score an empty sequence")
    return len(y_true)


def counts(y_true: Sequence, y_pred: Sequence, label) -> tuple[int, int, int]:
    """One-vs-rest (true positives, false positives, false negatives) for `label`.

    tp: predicted label, and it was label.
    fp: predicted label, but it was something else.
    fn: it was label, but we predicted something else.

    True negatives are not returned because precision, recall and F1 do not use
    them. accuracy() and cohens_kappa() count agreement directly instead.
    """
    _check_lengths(y_true, y_pred)

    tp = 0
    fp = 0
    fn = 0
    for truth, prediction in zip(y_true, y_pred):
        if prediction == label and truth == label:
            tp += 1
        elif prediction == label and truth != label:
            fp += 1
        elif prediction != label and truth == label:
            fn += 1
    return tp, fp, fn


def confusion_matrix(y_true: Sequence, y_pred: Sequence, labels: Sequence) -> list[list[int]]:
    """Rows are true labels, columns are predicted labels, both in `labels` order.

    matrix[i][j] is the number of rows whose true label is labels[i] and whose
    predicted label is labels[j]. The diagonal is therefore the correct ones.

    Raises on any value not in `labels`, rather than dropping the row. A label
    the caller forgot to declare would otherwise silently shrink every total.
    """
    _check_lengths(y_true, y_pred)

    index = {label: position for position, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]

    for truth, prediction in zip(y_true, y_pred):
        if truth not in index:
            raise ValueError(f"true label {truth!r} is not in labels")
        if prediction not in index:
            raise ValueError(f"predicted label {prediction!r} is not in labels")
        matrix[index[truth]][index[prediction]] += 1

    return matrix


# --------------------------------------------------------------------------
# per-class scores
# --------------------------------------------------------------------------


def precision(y_true: Sequence, y_pred: Sequence, label) -> float:
    """tp / (tp + fp). Zero predicted positives is 0.0 -- see the module docstring."""
    tp, fp, _ = counts(y_true, y_pred, label)
    if tp + fp == 0:
        return 0.0
    return tp / (tp + fp)


def recall(y_true: Sequence, y_pred: Sequence, label) -> float:
    """tp / (tp + fn). Zero true positives is 0.0 -- see the module docstring."""
    tp, _, fn = counts(y_true, y_pred, label)
    if tp + fn == 0:
        return 0.0
    return tp / (tp + fn)


def f1(y_true: Sequence, y_pred: Sequence, label) -> float:
    """Harmonic mean of precision and recall: 2pr / (p + r).

    When both are 0.0 the harmonic mean is 0/0, which we report as 0.0. That
    covers the class nobody predicted and nobody had, which under this
    convention scores zero rather than being skipped.
    """
    p = precision(y_true, y_pred, label)
    r = recall(y_true, y_pred, label)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def per_class_report(y_true: Sequence, y_pred: Sequence, labels: Sequence) -> dict:
    """Precision, recall, F1, support and prediction count for every label.

    Returns:
        {"per_class": {label: {"precision", "recall", "f1", "support",
                               "n_predicted", "tp", "fp", "fn"}},
         "undefined_precision": [labels with tp + fp == 0],
         "undefined_recall":    [labels with tp + fn == 0]}

    The two lists are the honest half of the 0/0 convention: they say which of
    the 0.0s mean "never predicted" / "never occurred" rather than "always
    wrong". Both are in `labels` order, so the report reads in taxonomy order.
    """
    _check_lengths(y_true, y_pred)

    per_class: dict = {}
    undefined_precision: list = []
    undefined_recall: list = []

    for label in labels:
        tp, fp, fn = counts(y_true, y_pred, label)

        if tp + fp == 0:
            undefined_precision.append(label)
        if tp + fn == 0:
            undefined_recall.append(label)

        per_class[label] = {
            "precision": precision(y_true, y_pred, label),
            "recall": recall(y_true, y_pred, label),
            "f1": f1(y_true, y_pred, label),
            "support": tp + fn,  # how many rows truly are this label
            "n_predicted": tp + fp,  # how many rows we called this label
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

    return {
        "per_class": per_class,
        "undefined_precision": undefined_precision,
        "undefined_recall": undefined_recall,
    }


# --------------------------------------------------------------------------
# aggregate scores
# --------------------------------------------------------------------------


def macro_f1(y_true: Sequence, y_pred: Sequence, labels: Sequence) -> float:
    """Plain unweighted mean of the per-class F1s, over every label given.

    Unweighted is the point. The golden set runs from 6 rows of
    playback_library to 32 of feature_request; a support-weighted average would
    let the big classes hide failures on the small ones, which are exactly the
    ones a support agent would notice.
    """
    if len(labels) == 0:
        raise ValueError("cannot macro-average over zero labels")

    total = 0.0
    for label in labels:
        total += f1(y_true, y_pred, label)
    return total / len(labels)


def accuracy(y_true: Sequence, y_pred: Sequence) -> float:
    """Fraction of rows where the prediction equals the truth."""
    n = _check_lengths(y_true, y_pred)

    n_correct = 0
    for truth, prediction in zip(y_true, y_pred):
        if truth == prediction:
            n_correct += 1
    return n_correct / n


def cohens_kappa(y_true: Sequence, y_pred: Sequence, labels: Sequence) -> float:
    """Agreement corrected for the agreement two random labellers would get anyway.

        kappa = (p_o - p_e) / (1 - p_e)

    p_o is observed agreement, i.e. accuracy. p_e is expected agreement: for
    each label, the chance both sides pick it independently, which is
    (rows truly that label / n) * (rows predicted that label / n), summed.

    On a nine-class problem with a 21% majority, raw accuracy flatters a system
    that just guesses the big class; kappa is what survives that correction.

    Returns 0.0 when p_e == 1.0. That happens only when both sides used exactly
    one label for everything, where "agreement above chance" is not a defined
    quantity -- and 0.0 is the honest reading: no information beyond chance.
    """
    n = _check_lengths(y_true, y_pred)

    p_o = accuracy(y_true, y_pred)

    p_e = 0.0
    for label in labels:
        n_true = 0
        n_pred = 0
        for truth, prediction in zip(y_true, y_pred):
            if truth == label:
                n_true += 1
            if prediction == label:
                n_pred += 1
        p_e += (n_true / n) * (n_pred / n)

    if p_e == 1.0:
        return 0.0
    return (p_o - p_e) / (1 - p_e)


# --------------------------------------------------------------------------
# bootstrap confidence intervals
# --------------------------------------------------------------------------


def percentile(values: Sequence[float], q: float) -> float:
    """The q-th percentile of `values`, by linear interpolation between ranks.

    Sort the values, then the percentile sits at fractional rank

        rank = (q / 100) * (len(values) - 1)

    Example, values [10, 20, 30, 40, 50] and q = 2.5:
        rank  = 0.025 * 4 = 0.1
        lower = index 0 -> 10, upper = index 1 -> 20
        result = 10 + 0.1 * (20 - 10) = 11.0

    This is the same definition numpy uses by default, written out so the
    interval in the report is reproducible with a calculator.
    """
    if len(values) == 0:
        raise ValueError("cannot take a percentile of an empty sequence")
    if not 0.0 <= q <= 100.0:
        raise ValueError(f"q must be in [0, 100], got {q}")

    ordered = sorted(values)
    rank = (q / 100.0) * (len(ordered) - 1)

    lower_index = int(rank)  # floor, since rank >= 0
    upper_index = min(lower_index + 1, len(ordered) - 1)
    fraction = rank - lower_index

    return ordered[lower_index] + fraction * (ordered[upper_index] - ordered[lower_index])


def bootstrap_indices(
    n: int,
    n_resamples: int = N_RESAMPLES,
    seed: int = SEED,
) -> list[list[int]]:
    """`n_resamples` index lists, each n draws from range(n) with replacement.

    Returned rather than consumed so evaluate.py can generate ONE set and score
    every system through it. Paired resamples: system A and system B see the
    same 1000 synthetic golden sets, so their intervals are comparable and a
    difference between them is not an artefact of two different random draws.

    stdlib random.Random, not numpy, so the sequence is reproducible from the
    seed with nothing but Python.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if n_resamples <= 0:
        raise ValueError(f"n_resamples must be positive, got {n_resamples}")

    rng = random.Random(seed)

    all_indices = []
    for _ in range(n_resamples):
        sample = []
        for _ in range(n):
            sample.append(rng.randrange(n))
        all_indices.append(sample)
    return all_indices


def bootstrap_ci(
    y_true: Sequence,
    y_pred: Sequence,
    metric_fn: Callable[[Sequence, Sequence], float],
    n_resamples: int = N_RESAMPLES,
    seed: int = SEED,
    alpha: float = ALPHA,
    indices: list[list[int]] | None = None,
) -> tuple[float, float]:
    """Percentile bootstrap interval for any metric of (y_true, y_pred).

    Resample the 150 rows with replacement, recompute the metric on each
    resample, and take the empirical 2.5th and 97.5th percentiles of the 1000
    values. No normality assumption, which matters because macro-F1 on 150 rows
    is bounded and skewed.

    `metric_fn` takes the two resampled sequences and returns a float, so
    macro-F1, accuracy, kappa and binary F1 all go through here unchanged --
    bind the labels with a lambda at the call site.

    Pass `indices` from bootstrap_indices() to share resamples across systems.
    """
    n = _check_lengths(y_true, y_pred)

    if indices is None:
        indices = bootstrap_indices(n, n_resamples=n_resamples, seed=seed)

    scores = []
    for sample in indices:
        if len(sample) != n:
            raise ValueError(f"resample has {len(sample)} indices, expected {n}")
        true_sample = [y_true[i] for i in sample]
        pred_sample = [y_pred[i] for i in sample]
        scores.append(metric_fn(true_sample, pred_sample))

    assert len(scores) == len(indices), f"{len(scores)} scores for {len(indices)} resamples"

    low = percentile(scores, 100.0 * (alpha / 2.0))
    high = percentile(scores, 100.0 * (1.0 - alpha / 2.0))
    return low, high
