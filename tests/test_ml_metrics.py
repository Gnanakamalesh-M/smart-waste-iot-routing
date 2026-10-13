"""Error and ranking metrics, checked against hand-worked examples.

``average_precision`` is implemented in the stdlib so the rule-based anomaly detector can be
scored without the optional ``ml`` extra, which means it needs testing against known values
rather than against scikit-learn.
"""

import pytest

from smartwaste.ml.metrics import (
    average_precision,
    mae,
    precision_recall,
    rmse,
    score_detector,
)

pytestmark = pytest.mark.unit


def test_mae_and_rmse_on_a_worked_example() -> None:
    actual = [1.0, 2.0, 3.0]
    predicted = [1.0, 4.0, 0.0]
    # errors: 0, 2, 3 -> MAE = 5/3; squared 0, 4, 9 -> RMSE = sqrt(13/3)
    assert mae(actual, predicted) == pytest.approx(5 / 3)
    assert rmse(actual, predicted) == pytest.approx((13 / 3) ** 0.5)


def test_perfect_prediction_scores_zero_error() -> None:
    assert mae([1.0, 2.0], [1.0, 2.0]) == 0.0
    assert rmse([1.0, 2.0], [1.0, 2.0]) == 0.0


def test_regression_metrics_reject_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        mae([1.0], [1.0, 2.0])
    with pytest.raises(ValueError, match="length mismatch"):
        rmse([1.0], [1.0, 2.0])


def test_regression_metrics_reject_empty_input() -> None:
    with pytest.raises(ValueError, match="at least one observation"):
        mae([], [])


def test_precision_recall_on_a_worked_example() -> None:
    labels = [True, True, True, False, False]
    flagged = [True, False, True, True, False]
    # tp = 2, fp = 1, fn = 1
    precision, recall = precision_recall(labels, flagged)
    assert precision == pytest.approx(2 / 3)
    assert recall == pytest.approx(2 / 3)


def test_precision_is_zero_when_nothing_is_flagged() -> None:
    """A degenerate detector must still produce a reportable row, not a crash or a NaN."""
    precision, recall = precision_recall([True, False], [False, False])
    assert precision == 0.0
    assert recall == 0.0


def test_average_precision_is_one_for_a_perfect_ranking() -> None:
    labels = [True, True, False, False]
    scores = [0.9, 0.8, 0.2, 0.1]
    assert average_precision(labels, scores) == pytest.approx(1.0)


def test_average_precision_on_a_worked_example() -> None:
    """One positive ranked second: precision at the only recall step is 1/2."""
    labels = [False, True]
    scores = [0.9, 0.8]
    assert average_precision(labels, scores) == pytest.approx(0.5)


def test_average_precision_of_a_worst_case_ranking() -> None:
    labels = [True, False, False]
    scores = [0.1, 0.9, 0.8]
    # The single positive sits last: recall reaches 1 at precision 1/3.
    assert average_precision(labels, scores) == pytest.approx(1 / 3)


def test_average_precision_is_about_the_prevalence_for_a_useless_ranker() -> None:
    """The reason prevalence is printed beside PR-AUC everywhere in the anomaly tables."""
    labels = [i % 10 == 0 for i in range(1000)]
    constant = [0.5] * 1000
    prevalence = sum(labels) / len(labels)
    assert average_precision(labels, constant) == pytest.approx(prevalence, abs=1e-9)


def test_average_precision_ignores_row_order_within_a_tie() -> None:
    """The rule detector emits few distinct scores, so ties must not be order-sensitive."""
    labels = [True, False, False, True]
    scores = [0.5] * 4
    reordered_labels = [False, True, True, False]
    assert average_precision(labels, scores) == pytest.approx(
        average_precision(reordered_labels, scores)
    )


def test_average_precision_with_no_positives_is_zero_not_a_crash() -> None:
    assert average_precision([False, False], [0.1, 0.2]) == 0.0


def test_score_detector_bundles_counts_and_prevalence() -> None:
    labels = [True, True, False, False]
    scores = [0.9, 0.1, 0.8, 0.2]
    flagged = [True, False, True, False]
    result = score_detector(labels, scores, flagged)

    assert result.n == 4
    assert result.n_positive == 2
    assert result.true_positives == 1
    assert result.false_positives == 1
    assert result.false_negatives == 1
    assert result.prevalence == pytest.approx(0.5)
    assert result.precision == pytest.approx(0.5)
    assert result.recall == pytest.approx(0.5)


def test_module_does_not_implement_accuracy() -> None:
    """CLAUDE.md forbids accuracy on imbalanced problems, so it must not be available at all."""
    import smartwaste.ml.metrics as metrics

    assert not hasattr(metrics, "accuracy")
    assert "def accuracy" not in metrics.__doc__ if metrics.__doc__ else True
