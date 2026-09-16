"""Tests for src/metrics.py.

What is tested: every metric against a value computed by hand, on fixtures small
enough to check on paper. The arithmetic for each assertion is written out in a
comment above it -- that is the point of these tests. They are not regression
snapshots; if one fails, the comment tells you what the number should be and
why.

The 0/0 cases get their own tests, because both baselines predict zero positives
on escalate and on several intent classes. Precision must be 0.0 and the class
must be named in `undefined_precision`. Never 1.0, never a crash.

What is not tested: agreement with sklearn.metrics. CLAUDE.md forbids importing
it, and checking against it would defeat the purpose of deriving these by hand.
"""

from __future__ import annotations

import pytest

from src import metrics

# --------------------------------------------------------------------------
# the worked fixture
# --------------------------------------------------------------------------
#
# Four rows, three classes. Confusion matrix (rows = true, cols = pred):
#
#           pred a   pred b   pred c
#   true a      1        1        0
#   true b      0        1        0
#   true c      1        0        0
#
# class a: tp=1 fp=1 fn=1   -> P = 1/2, R = 1/2, F1 = 1/2
# class b: tp=1 fp=1 fn=0   -> P = 1/2, R = 1/1, F1 = 2(.5)(1)/1.5 = 2/3
# class c: tp=0 fp=0 fn=1   -> P = 0.0 (never predicted), R = 0/1, F1 = 0.0

Y_TRUE = ["a", "a", "b", "c"]
Y_PRED = ["a", "b", "b", "a"]
LABELS = ["a", "b", "c"]


def test_confusion_matrix_matches_the_hand_drawn_table():
    assert metrics.confusion_matrix(Y_TRUE, Y_PRED, LABELS) == [
        [1, 1, 0],
        [0, 1, 0],
        [1, 0, 0],
    ]


def test_confusion_matrix_rows_sum_to_support_and_total_sums_to_n():
    matrix = metrics.confusion_matrix(Y_TRUE, Y_PRED, LABELS)

    # Row sums are how often each class truly occurred: a twice, b once, c once.
    assert [sum(row) for row in matrix] == [2, 1, 1]
    assert sum(sum(row) for row in matrix) == len(Y_TRUE)


def test_confusion_matrix_rejects_a_label_the_caller_did_not_declare():
    # Dropping the row instead would silently shrink every total in the report.
    with pytest.raises(ValueError):
        metrics.confusion_matrix(["a", "z"], ["a", "a"], LABELS)


@pytest.mark.parametrize(
    "label,expected_counts",
    [
        ("a", (1, 1, 1)),
        ("b", (1, 1, 0)),
        ("c", (0, 0, 1)),
    ],
)
def test_counts_are_the_one_vs_rest_tp_fp_fn(label, expected_counts):
    assert metrics.counts(Y_TRUE, Y_PRED, label) == expected_counts


@pytest.mark.parametrize(
    "label,expected",
    [
        ("a", 1 / 2),  # tp 1 / (tp 1 + fp 1)
        ("b", 1 / 2),  # tp 1 / (tp 1 + fp 1)
        ("c", 0.0),  # 0 predicted -> the convention, not 1.0
    ],
)
def test_precision_per_class(label, expected):
    assert metrics.precision(Y_TRUE, Y_PRED, label) == pytest.approx(expected)


@pytest.mark.parametrize(
    "label,expected",
    [
        ("a", 1 / 2),  # tp 1 / (tp 1 + fn 1)
        ("b", 1.0),  # tp 1 / (tp 1 + fn 0)
        ("c", 0.0),  # tp 0 / (tp 0 + fn 1)
    ],
)
def test_recall_per_class(label, expected):
    assert metrics.recall(Y_TRUE, Y_PRED, label) == pytest.approx(expected)


@pytest.mark.parametrize(
    "label,expected",
    [
        ("a", 1 / 2),  # 2(0.5)(0.5) / (0.5 + 0.5) = 0.5
        ("b", 2 / 3),  # 2(0.5)(1.0) / (0.5 + 1.0) = 1/1.5
        ("c", 0.0),  # P + R = 0 -> 0.0
    ],
)
def test_f1_per_class(label, expected):
    assert metrics.f1(Y_TRUE, Y_PRED, label) == pytest.approx(expected)


def test_macro_f1_is_the_plain_mean_of_the_three_class_f1s():
    # (0.5 + 2/3 + 0.0) / 3 = 1.166666... / 3 = 0.388888...
    assert metrics.macro_f1(Y_TRUE, Y_PRED, LABELS) == pytest.approx((0.5 + 2 / 3 + 0.0) / 3)
    assert metrics.macro_f1(Y_TRUE, Y_PRED, LABELS) == pytest.approx(0.3888888888888889)


def test_accuracy_counts_exact_matches():
    # Row 1 (a/a) and row 3 (b/b) agree; rows 2 and 4 do not. 2/4.
    assert metrics.accuracy(Y_TRUE, Y_PRED) == pytest.approx(0.5)


def test_mismatched_lengths_raise_rather_than_zipping_short():
    with pytest.raises(ValueError):
        metrics.accuracy(["a", "b"], ["a"])


def test_empty_input_raises():
    with pytest.raises(ValueError):
        metrics.accuracy([], [])


# --------------------------------------------------------------------------
# the 0/0 convention
# --------------------------------------------------------------------------


def test_a_class_with_zero_predicted_positives_scores_zero_and_is_named():
    report = metrics.per_class_report(Y_TRUE, Y_PRED, LABELS)

    # c is never predicted: tp + fp = 0. Not 1.0, and not an exception.
    assert report["per_class"]["c"]["precision"] == 0.0
    assert report["per_class"]["c"]["n_predicted"] == 0
    assert report["undefined_precision"] == ["c"]

    # a and b were both predicted, so neither is undefined.
    assert "a" not in report["undefined_precision"]
    assert "b" not in report["undefined_precision"]


def test_a_class_with_zero_true_positives_scores_zero_recall_and_is_named():
    # "d" is a declared label that never occurs in the truth: tp + fn = 0.
    labels_with_absent_class = ["a", "b", "c", "d"]
    report = metrics.per_class_report(Y_TRUE, Y_PRED, labels_with_absent_class)

    assert report["per_class"]["d"]["recall"] == 0.0
    assert report["per_class"]["d"]["support"] == 0
    assert report["undefined_recall"] == ["d"]

    # Never predicted either, so it is undefined on both axes and F1 is 0.0.
    assert report["undefined_precision"] == ["c", "d"]
    assert report["per_class"]["d"]["f1"] == 0.0


def test_a_constant_negative_predictor_does_not_crash_or_score_one():
    # This is both baselines on the escalate decision: they never escalate.
    y_true = [True, True, False, False, False]
    y_pred = [False, False, False, False, False]

    report = metrics.per_class_report(y_true, y_pred, [False, True])

    assert report["per_class"][True]["precision"] == 0.0
    assert report["per_class"][True]["recall"] == 0.0  # tp 0 / (tp 0 + fn 2)
    assert report["per_class"][True]["f1"] == 0.0
    assert report["undefined_precision"] == [True]
    assert report["undefined_recall"] == []  # True does occur in the truth


def test_macro_f1_is_dragged_down_by_a_class_the_system_never_predicts():
    # The reason we average over all declared labels: ignoring a class costs you.
    # Perfect on a and b, silent on c -> (1.0 + 1.0 + 0.0) / 3.
    y_true = ["a", "b", "c"]
    y_pred = ["a", "b", "b"]

    assert metrics.macro_f1(y_true, y_pred, ["a", "b"]) == pytest.approx((1.0 + 2 / 3) / 2)
    assert metrics.macro_f1(y_true, y_pred, ["a", "b", "c"]) == pytest.approx((1.0 + 2 / 3) / 3)


# --------------------------------------------------------------------------
# kappa
# --------------------------------------------------------------------------


def test_cohens_kappa_on_the_worked_fixture():
    # p_o = accuracy = 2/4 = 0.5
    # true counts a=2 b=1 c=1; predicted counts a=2 b=2 c=0
    # p_e = (2/4)(2/4) + (1/4)(2/4) + (1/4)(0/4) = 0.25 + 0.125 + 0 = 0.375
    # kappa = (0.5 - 0.375) / (1 - 0.375) = 0.125 / 0.625 = 0.2
    assert metrics.cohens_kappa(Y_TRUE, Y_PRED, LABELS) == pytest.approx(0.2)


def test_kappa_is_one_when_the_two_labellers_agree_everywhere():
    y_true = ["a", "b", "c", "a"]
    # p_o = 1.0, so (1 - p_e) / (1 - p_e) = 1.0 for any p_e < 1.
    assert metrics.cohens_kappa(y_true, y_true, LABELS) == pytest.approx(1.0)


def test_kappa_is_zero_at_chance_agreement():
    # p_o = 2/4 = 0.5. Marginals are 2 and 2 on both sides:
    # p_e = (2/4)(2/4) + (2/4)(2/4) = 0.5. kappa = (0.5 - 0.5) / 0.5 = 0.
    y_true = ["a", "a", "b", "b"]
    y_pred = ["a", "b", "a", "b"]

    assert metrics.cohens_kappa(y_true, y_pred, ["a", "b"]) == pytest.approx(0.0)


def test_kappa_is_negative_when_agreement_is_worse_than_chance():
    # p_o = 0. Marginals 2/2 both sides -> p_e = 0.5. kappa = -0.5 / 0.5 = -1.
    y_true = ["a", "a", "b", "b"]
    y_pred = ["b", "b", "a", "a"]

    assert metrics.cohens_kappa(y_true, y_pred, ["a", "b"]) == pytest.approx(-1.0)


def test_kappa_returns_zero_rather_than_dividing_by_zero_when_both_sides_are_constant():
    # Everyone said "a": p_o = 1.0 and p_e = 1.0, so kappa is 0/0. We report 0.0,
    # because "agreement above chance" is not a defined quantity here.
    y_true = ["a", "a", "a"]
    assert metrics.cohens_kappa(y_true, y_true, ["a", "b"]) == 0.0


# --------------------------------------------------------------------------
# percentile and bootstrap
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "q,expected",
    [
        # rank = (q/100) * (5 - 1) = (q/100) * 4, then interpolate.
        (2.5, 11.0),  # rank 0.1  -> 10 + 0.1 * (20 - 10)
        (50.0, 30.0),  # rank 2.0  -> exactly the middle element
        (97.5, 49.0),  # rank 3.9  -> 40 + 0.9 * (50 - 40)
        (0.0, 10.0),  # rank 0    -> the smallest
        (100.0, 50.0),  # rank 4    -> the largest
    ],
)
def test_percentile_interpolates_between_ranks(q, expected):
    assert metrics.percentile([10, 20, 30, 40, 50], q) == pytest.approx(expected)


def test_percentile_sorts_before_indexing():
    assert metrics.percentile([50, 10, 40, 20, 30], 50.0) == pytest.approx(30.0)


def test_bootstrap_indices_has_the_requested_shape_and_stays_in_range():
    indices = metrics.bootstrap_indices(4, n_resamples=10, seed=42)

    assert len(indices) == 10
    assert all(len(sample) == 4 for sample in indices)
    assert all(0 <= i < 4 for sample in indices for i in sample)


def test_bootstrap_indices_are_reproducible_from_the_seed():
    assert metrics.bootstrap_indices(4, n_resamples=10, seed=42) == metrics.bootstrap_indices(
        4, n_resamples=10, seed=42
    )


def test_bootstrap_indices_resample_with_replacement():
    # With replacement, some resample of 4 draws must repeat an index. Without
    # replacement every sample would be a permutation and this would be 0.
    indices = metrics.bootstrap_indices(4, n_resamples=50, seed=42)
    assert any(len(set(sample)) < 4 for sample in indices)


def test_bootstrap_ci_of_a_perfect_predictor_is_exactly_one():
    # Every resample of a perfect predictor is still perfect, so accuracy is 1.0
    # on all 1000 and both percentiles land on 1.0.
    y_true = ["a", "b", "c", "a", "b"]
    low, high = metrics.bootstrap_ci(y_true, y_true, metrics.accuracy, n_resamples=100)

    assert low == pytest.approx(1.0)
    assert high == pytest.approx(1.0)


def test_bootstrap_ci_of_a_always_wrong_predictor_is_exactly_zero():
    y_true = ["a", "a", "a", "a"]
    y_pred = ["b", "b", "b", "b"]
    low, high = metrics.bootstrap_ci(y_true, y_pred, metrics.accuracy, n_resamples=100)

    assert (low, high) == (0.0, 0.0)


def test_bootstrap_ci_is_deterministic_across_runs():
    y_true = ["a", "a", "b", "c", "b", "a", "c", "b"]
    y_pred = ["a", "b", "b", "c", "b", "a", "a", "b"]

    first = metrics.bootstrap_ci(y_true, y_pred, metrics.accuracy)
    second = metrics.bootstrap_ci(y_true, y_pred, metrics.accuracy)

    assert first == second


def test_bootstrap_ci_brackets_the_point_estimate():
    y_true = ["a", "a", "b", "c", "b", "a", "c", "b"]
    y_pred = ["a", "b", "b", "c", "b", "a", "a", "b"]

    point = metrics.accuracy(y_true, y_pred)
    low, high = metrics.bootstrap_ci(y_true, y_pred, metrics.accuracy)

    assert low <= point <= high


def test_passing_shared_indices_gives_the_same_answer_as_the_default_ones():
    # This is how evaluate.py pairs the three systems: one index set, reused.
    y_true = ["a", "a", "b", "c", "b", "a", "c", "b"]
    y_pred = ["a", "b", "b", "c", "b", "a", "a", "b"]

    shared = metrics.bootstrap_indices(len(y_true), n_resamples=metrics.N_RESAMPLES, seed=42)

    assert metrics.bootstrap_ci(y_true, y_pred, metrics.accuracy, indices=shared) == (
        metrics.bootstrap_ci(y_true, y_pred, metrics.accuracy)
    )


def test_bootstrap_ci_works_for_macro_f1_bound_to_labels():
    # metric_fn is (y_true, y_pred) -> float, so the labels are bound at the
    # call site. Same shape evaluate.py uses.
    low, high = metrics.bootstrap_ci(
        Y_TRUE,
        Y_PRED,
        lambda truth, pred: metrics.macro_f1(truth, pred, LABELS),
        n_resamples=100,
    )

    assert 0.0 <= low <= high <= 1.0


def test_bootstrap_ci_rejects_indices_of_the_wrong_length():
    with pytest.raises(ValueError):
        metrics.bootstrap_ci(Y_TRUE, Y_PRED, metrics.accuracy, indices=[[0, 1]])
