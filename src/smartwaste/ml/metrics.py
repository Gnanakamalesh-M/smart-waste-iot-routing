"""Error and ranking metrics, in the stdlib so no metric needs the optional ``ml`` extra.

Two groups:

- **Regression** (:func:`mae`, :func:`rmse`) for the forecast comparison.
- **Imbalanced classification** (:func:`precision_recall`, :func:`average_precision`) for the
  anomaly comparison.

### Why accuracy appears nowhere in this module

CLAUDE.md forbids it for imbalanced problems, and planted sensor faults are badly imbalanced.
``anomaly.fault_bin_fraction`` is a share of *bins*, and each fault episode covers only part of
the horizon, so faulty *windows* are a few percent of the dataset -- a detector that says "never
faulty" therefore scores well above 90% accuracy while catching nothing. So accuracy is not
implemented here at all; it cannot be reported by accident. Every anomaly table prints precision,
recall, PR-AUC **and the positive-class prevalence**, which is the number PR-AUC has to be judged
against: a useless ranker scores about the prevalence, not 0.5. The exact trivial accuracy is
computed from the real rows at render time rather than asserted anywhere in prose.

:func:`average_precision` is the step-wise average precision, which is what scikit-learn's
``average_precision_score`` computes. It is implemented here rather than imported so the rule
baseline can be scored with sklearn absent, and :mod:`tests.test_ml_metrics` checks it against
hand-worked examples.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


def mae(actual: Sequence[float], predicted: Sequence[float]) -> float:
    """Mean absolute error."""
    if not actual:
        raise ValueError("mae needs at least one observation")
    if len(actual) != len(predicted):
        raise ValueError(f"length mismatch: {len(actual)} actual vs {len(predicted)} predicted")
    return sum(abs(a - p) for a, p in zip(actual, predicted, strict=True)) / len(actual)


def rmse(actual: Sequence[float], predicted: Sequence[float]) -> float:
    """Root mean squared error."""
    if not actual:
        raise ValueError("rmse needs at least one observation")
    if len(actual) != len(predicted):
        raise ValueError(f"length mismatch: {len(actual)} actual vs {len(predicted)} predicted")
    squared = sum((a - p) ** 2 for a, p in zip(actual, predicted, strict=True)) / len(actual)
    return float(squared**0.5)


@dataclass(frozen=True)
class ClassificationScore:
    """Precision / recall / PR-AUC for one detector on one label set.

    ``prevalence`` is carried alongside deliberately: PR-AUC is only interpretable next to the
    positive rate, because a random ranker scores approximately the prevalence.
    """

    n: int
    n_positive: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    average_precision: float

    @property
    def prevalence(self) -> float:
        return self.n_positive / self.n if self.n else 0.0


def precision_recall(labels: Sequence[bool], flagged: Sequence[bool]) -> tuple[float, float]:
    """Precision and recall for a hard yes/no detector.

    Precision is defined as 0.0 when nothing was flagged, and recall as 0.0 when there is nothing
    to find -- rather than raising or returning NaN, so a degenerate detector still produces a
    reportable row instead of crashing the table.
    """
    if len(labels) != len(flagged):
        raise ValueError(f"length mismatch: {len(labels)} labels vs {len(flagged)} flags")
    tp = sum(1 for label, flag in zip(labels, flagged, strict=True) if label and flag)
    fp = sum(1 for label, flag in zip(labels, flagged, strict=True) if flag and not label)
    fn = sum(1 for label, flag in zip(labels, flagged, strict=True) if label and not flag)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return precision, recall


def average_precision(labels: Sequence[bool], scores: Sequence[float]) -> float:
    """PR-AUC as step-wise average precision: ``sum over k of (R_k - R_{k-1}) * P_k``.

    Rows are ranked by descending score. Ties are handled by advancing through the whole tied
    group before taking a measurement, so a detector cannot gain from the input's row order --
    which matters here because the rule baseline emits only a handful of distinct scores and
    would otherwise be flattered or punished by how the dataset happens to be sorted.
    """
    if len(labels) != len(scores):
        raise ValueError(f"length mismatch: {len(labels)} labels vs {len(scores)} scores")
    n_positive = sum(1 for label in labels if label)
    if n_positive == 0:
        # Undefined rather than 0.0 in principle; 0.0 is returned so a family with no planted
        # faults still renders, and the caller prints n_positive beside it.
        return 0.0

    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    total = 0.0
    tp = 0
    seen = 0
    previous_recall = 0.0
    position = 0
    while position < len(order):
        tie_score = scores[order[position]]
        # Consume the entire tied group: within a tie there is no defensible ordering.
        while position < len(order) and scores[order[position]] == tie_score:
            if labels[order[position]]:
                tp += 1
            seen += 1
            position += 1
        recall = tp / n_positive
        precision = tp / seen
        total += (recall - previous_recall) * precision
        previous_recall = recall
    return total


def score_detector(
    labels: Sequence[bool], scores: Sequence[float], flagged: Sequence[bool]
) -> ClassificationScore:
    """Bundle the hard-decision and ranking metrics for one detector."""
    precision, recall = precision_recall(labels, flagged)
    tp = sum(1 for label, flag in zip(labels, flagged, strict=True) if label and flag)
    fp = sum(1 for label, flag in zip(labels, flagged, strict=True) if flag and not label)
    fn = sum(1 for label, flag in zip(labels, flagged, strict=True) if label and not flag)
    return ClassificationScore(
        n=len(labels),
        n_positive=sum(1 for label in labels if label),
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        precision=precision,
        recall=recall,
        average_precision=average_precision(labels, scores),
    )
