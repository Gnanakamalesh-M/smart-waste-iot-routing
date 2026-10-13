"""Sensor anomaly detection: a rule baseline against an Isolation Forest, on planted faults.

### The leakage firewall

The fault generator writes **two** files: the readings and, separately, the planted labels. The
detectors in this module:

- receive :class:`SensorWindow` objects, which carry readings and nothing else;
- have **no parameter** that could accept a label, a label array or a label file path;
- never import or reference the label file name.

``tests/test_anomaly.py`` asserts the last point against this module's own source text, so a
future edit that quietly reads the truth file fails a test rather than producing a flattering
number. Only the scorer in :mod:`smartwaste.anomaly_eval` ever opens the labels.

### Thresholds are chosen on a different city

Both detectors have knobs (the rule's variance tolerance and disagreement limit; the forest's
contamination). Picking them against the scoring seed's planted labels would be exactly the
leakage this project forbids elsewhere, so they are selected on a **validation seed**
(``seed + anomaly.validation_seed_offset``) and applied unchanged to the scoring seed. The
runner prints both seeds so the separation is visible in the output.

### Which families each detector is built for, stated up front

The rule baseline tests for *no variance over a window* and for *3-sensor disagreement*. That is
near enough a definition of the ``stuck`` and ``dropout`` families, so a rule win there is a
**demonstration, not a measurement** -- the Phase 3 results table labels those two families
"favors the rule baseline by construction", in the same spirit as Phase 1's scenario labels.
``drift`` and ``noise_burst`` are fair tests: neither detector is built around them.

### scikit-learn is imported lazily

Only :meth:`IsolationForestDetector.fit` imports sklearn, so the rule baseline runs with the
optional ``ml`` extra uninstalled.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from smartwaste.config import AnomalyConfig

#: The four planted fault families. Names are used in output tables and in the labels file.
FAULT_STUCK = "stuck"
FAULT_DRIFT = "drift"
FAULT_DROPOUT = "dropout"
FAULT_NOISE_BURST = "noise_burst"
FAULT_FAMILIES = (FAULT_STUCK, FAULT_DRIFT, FAULT_DROPOUT, FAULT_NOISE_BURST)

#: How each family should be read, so a built-in advantage is never reported as evidence.
FAMILY_FAIRNESS = {
    FAULT_STUCK: "favors the rule baseline by construction",
    FAULT_DROPOUT: "favors the rule baseline by construction",
    FAULT_DRIFT: "fair test",
    FAULT_NOISE_BURST: "fair test",
}

DETECTOR_KEYS = ("rule", "isolation_forest")
DETECTOR_LABELS = {
    "rule": "rule baseline: zero variance OR 3-sensor disagreement [stdlib]",
    "isolation_forest": "Isolation Forest [ML]",
}
ML_DETECTOR_KEYS = ("isolation_forest",)


class OutlierEstimator(Protocol):
    """The slice of scikit-learn's outlier-detector API this module actually uses.

    A Protocol rather than ``Any``: scikit-learn ships no type information, so without this the
    estimator would be untyped and a renamed method would fail at runtime instead of at
    type-check time.
    """

    def score_samples(self, X: Sequence[Sequence[float]]) -> Sequence[float]:
        """Lower = more anomalous, which is why this module negates it."""
        ...

    def predict(self, X: Sequence[Sequence[float]]) -> Sequence[float]:
        """-1 for an outlier, 1 for an inlier."""
        ...


@dataclass(frozen=True)
class SensorWindow:
    """One bin's readings over a rolling window -- everything a detector is allowed to see.

    ``distances`` is ``[day][sensor]``. There is deliberately no label field, no true-fill field
    and no fault-family field: a detector physically cannot read the truth from this object.
    """

    bin_index: int
    end_day: int
    distances: tuple[tuple[float, ...], ...]
    weights: tuple[float, ...]

    @property
    def n_days(self) -> int:
        return len(self.distances)

    @property
    def n_sensors(self) -> int:
        return len(self.distances[0]) if self.distances else 0

    def sensor_series(self, sensor: int) -> tuple[float, ...]:
        return tuple(row[sensor] for row in self.distances)


def window_features(window: SensorWindow) -> list[float]:
    """Numeric features for one window: per-sensor spread and level, plus disagreement.

    Readings only. Chosen to be the kind of thing either detector could use -- deliberately not
    specialised towards the families the rule baseline happens to catch.
    """
    features: list[float] = []
    for sensor in range(window.n_sensors):
        series = window.sensor_series(sensor)
        features.append(statistics.fmean(series))
        features.append(statistics.pstdev(series) if len(series) > 1 else 0.0)
        features.append(min(series))
        features.append(max(series))
        # Total absolute day-to-day movement: a stuck sensor is 0, a bursty one is large.
        features.append(sum(abs(series[i] - series[i - 1]) for i in range(1, len(series))))
    features.append(max_disagreement(window))
    features.append(mean_disagreement(window))
    weights = window.weights
    features.append(statistics.fmean(weights))
    features.append(statistics.pstdev(weights) if len(weights) > 1 else 0.0)
    return features


def max_disagreement(window: SensorWindow) -> float:
    """Largest spread between the sensors on any single day in the window."""
    worst = 0.0
    for row in window.distances:
        if len(row) > 1:
            worst = max(worst, max(row) - min(row))
    return worst


def mean_disagreement(window: SensorWindow) -> float:
    """Mean per-day spread between the sensors."""
    if not window.distances:
        return 0.0
    spreads = [max(row) - min(row) for row in window.distances if len(row) > 1]
    return statistics.fmean(spreads) if spreads else 0.0


def min_sensor_variation(window: SensorWindow) -> float:
    """The quietest sensor's total movement over the window.

    Near zero means at least one sensor is not changing at all, which is what ``stuck`` and
    ``dropout`` look like.
    """
    if window.n_sensors == 0:
        return 0.0
    totals: list[float] = []
    for sensor in range(window.n_sensors):
        series = window.sensor_series(sensor)
        totals.append(sum(abs(series[i] - series[i - 1]) for i in range(1, len(series))))
    return min(totals)


class Detector:
    """Scores windows for "how anomalous", and makes a hard flag decision.

    Higher score = more anomalous, for both detectors, so PR-AUC is computed the same way for
    each.
    """

    key: str
    label: str
    needs_ml_extra: bool = False

    def fit(self, windows: Sequence[SensorWindow]) -> None:
        """Unsupervised: fit on readings only.

        Never given labels -- there is no parameter one could arrive through.
        """

    def score(self, window: SensorWindow) -> float:
        raise NotImplementedError

    def flag(self, window: SensorWindow) -> bool:
        raise NotImplementedError

    def score_all(self, windows: Sequence[SensorWindow]) -> list[float]:
        """Score many windows at once.

        The default loops, which is right for the stdlib rule baseline. An estimator-backed
        detector **must** override this: a per-row call into scikit-learn costs about 10 ms of
        validation overhead, so scoring a 22,000-window dataset one row at a time takes minutes
        where a single batched call takes milliseconds.
        """
        return [self.score(window) for window in windows]

    def flag_all(self, windows: Sequence[SensorWindow]) -> list[bool]:
        """Hard decisions for many windows at once. Same reasoning as :meth:`score_all`."""
        return [self.flag(window) for window in windows]


class RuleDetector(Detector):
    """The simple baseline the ML detector has to beat. Pure stdlib.

    Flags a window when **either**:

    1. the quietest sensor moved less than ``stuck_tolerance_cm`` in total (stuck / dropout), or
    2. the sensors disagreed by more than ``disagreement_cm`` on some day.

    Both thresholds are invented and uncited, and both are chosen on the validation seed.
    """

    key = "rule"
    label = DETECTOR_LABELS["rule"]

    def __init__(self, stuck_tolerance_cm: float, disagreement_cm: float) -> None:
        self.stuck_tolerance_cm = stuck_tolerance_cm
        self.disagreement_cm = disagreement_cm

    def score(self, window: SensorWindow) -> float:
        """A continuous score so PR-AUC is meaningful, not just the hard flag.

        Combines "how still is the stillest sensor" and "how far apart do they disagree", each
        normalised by its own threshold, and takes the stronger signal.
        """
        variation = min_sensor_variation(window)
        stillness = 0.0
        if self.stuck_tolerance_cm > 0:
            stillness = max(0.0, 1.0 - variation / max(self.stuck_tolerance_cm, 1e-9))
        else:
            stillness = 1.0 if variation == 0.0 else 0.0
        disagreement = 0.0
        if self.disagreement_cm > 0:
            disagreement = max_disagreement(window) / self.disagreement_cm
        return max(stillness, disagreement)

    def flag(self, window: SensorWindow) -> bool:
        return (
            min_sensor_variation(window) <= self.stuck_tolerance_cm
            or max_disagreement(window) > self.disagreement_cm
        )


class IsolationForestDetector(Detector):
    """``sklearn.ensemble.IsolationForest`` over :func:`window_features`.

    Unsupervised, so it genuinely never sees a label -- :meth:`fit` takes windows only.

    ### Determinism is same-machine only

    ``random_state`` is pinned, so re-running on **this machine with these library versions**
    reproduces the fit. scikit-learn promises no bit-identical floats across operating systems,
    BLAS builds or versions, so no test pins an exact score and ``docs/PHASE3_RESULTS.md``
    scopes these figures to same-machine reproducibility.
    """

    key = "isolation_forest"
    label = DETECTOR_LABELS["isolation_forest"]
    needs_ml_extra = True

    def __init__(self, cfg: AnomalyConfig, seed: int) -> None:
        self._cfg = cfg
        self._seed = seed
        self._model: OutlierEstimator | None = None

    @property
    def is_fitted(self) -> bool:
        return self._model is not None

    def fit(self, windows: Sequence[SensorWindow]) -> None:
        if not windows:
            return
        # Lazy, deliberately: keeps the rule baseline usable without the ml extra.
        from sklearn.ensemble import IsolationForest

        model = IsolationForest(
            n_estimators=self._cfg.n_estimators,
            contamination=self._cfg.contamination,
            random_state=self._seed,
        )
        model.fit([window_features(window) for window in windows])
        self._model = model

    def score(self, window: SensorWindow) -> float:
        """Negated ``score_samples``, so larger means more anomalous like the rule detector."""
        return self.score_all([window])[0]

    def flag(self, window: SensorWindow) -> bool:
        return self.flag_all([window])[0]

    def score_all(self, windows: Sequence[SensorWindow]) -> list[float]:
        """One batched ``score_samples`` call for every window. See :meth:`Detector.score_all`."""
        model = self._fitted_model("score")
        if not windows:
            return []
        raw = model.score_samples([window_features(window) for window in windows])
        return [-float(value) for value in raw]

    def flag_all(self, windows: Sequence[SensorWindow]) -> list[bool]:
        """One batched ``predict`` call for every window."""
        model = self._fitted_model("flag")
        if not windows:
            return []
        predicted = model.predict([window_features(window) for window in windows])
        return [bool(value == -1) for value in predicted]

    def _fitted_model(self, what: str) -> OutlierEstimator:
        model = self._model
        if model is None:
            raise RuntimeError(f"IsolationForestDetector.{what} called before fit")
        return model


def build_detector(key: str, cfg: AnomalyConfig, seed: int) -> Detector:
    """One detector by key."""
    if key == "rule":
        return RuleDetector(cfg.stuck_tolerance_cm, cfg.disagreement_cm)
    if key == "isolation_forest":
        return IsolationForestDetector(cfg, seed)
    raise ValueError(f"unknown detector key: {key!r} (known: {DETECTOR_KEYS})")
