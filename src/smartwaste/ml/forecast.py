"""Fill-level forecasting: three naive baselines first, then one gradient-boosting model.

### What is predicted, and why it is a *rate*

Every forecaster here predicts one scalar: a bin's **daily rise in estimated fill percent**.
Strategy (f) then asks the only question a collection planner cares about -- *will this bin hit
RED before its next scheduled visit?* -- as

    projected = Fv_hat_now + rate * days_until_next_visit

Forecasting the rate rather than the level keeps all four methods comparable (a naive baseline
has a rate too) and makes the downstream decision a single line of arithmetic a reader can check.

### Features come ONLY from past noisy sensor estimates

A feature row contains:

- the last ``history_days`` values of ``Fv_hat`` -- the *fused* fill estimate, i.e. the output of
  :func:`smartwaste.fusion.fuse` on a noisy :class:`~smartwaste.noise.SensorReading`;
- the last ``history_days`` noisy load-cell weights;
- the observed day-to-day differences of those estimates;
- days since this bin was last collected;
- area type (one-hot), day of week (one-hot), festival flag.

**No ground truth anywhere.** Not the true fill, not the true weight, not
:func:`smartwaste.fusion.truth_state`. The guarantee is structural rather than a convention: a
:class:`BinHistory` is fed from ``SensorReading`` objects, which by construction carry no
ground-truth field, and the calendar facts come from
:class:`~smartwaste.strategies.RunContext`, which carries no fill or weight. ``tests/test_
forecast.py`` additionally asserts this module never references ``truth_state``.

### The target is the sensors' own future estimate, not the truth

We regress on future ``Fv_hat``, because a forecaster deployed on real hardware has no truth to
regress on -- it only ever sees what its sensors reported. The honest consequence is that **the
model learns to predict its own sensors, including their biases**, and a sensor error that
persists is learned rather than corrected. This is stated as a limitation in
``docs/PHASE3_RESULTS.md`` rather than hidden.

### Time-based split only

Training rows come from days ``0..warmup_days-1``; the model is fitted once and frozen; every
evaluated row is from day ``warmup_days`` onward. There is no random split anywhere in this
module -- a random split of a time series would let tomorrow train on next week.

A training row is **dropped if the bin was collected inside its span**, because a collection
resets the fill and the "rise" across it is an artefact of the truck, not of demand.

### scikit-learn is imported lazily

Only :meth:`GbmForecaster.fit` imports sklearn, so the three naive baselines run with the ``ml``
extra absent. The GBM is also **not bit-reproducible across machines or library versions**; see
the note on :class:`GbmForecaster`.
"""

from __future__ import annotations

from collections.abc import Container, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from smartwaste.config import BinConfig, ForecastConfig
from smartwaste.demand import AREA_TYPE_NAMES, DAYS_PER_WEEK
from smartwaste.fusion import fuse
from smartwaste.noise import SensorReading

#: Forecaster keys, in the order they appear in every results table. Baselines first, on purpose.
FORECASTER_KEYS = ("last_value", "moving_average", "seasonal_naive", "gbm")

FORECASTER_LABELS = {
    "last_value": "last value [naive baseline]",
    "moving_average": "moving average [naive baseline]",
    "seasonal_naive": "same-weekday seasonal naive [naive baseline]",
    "gbm": "gradient boosting (HistGradientBoostingRegressor) [ML]",
}

#: Keys that need the optional ``ml`` extra. Everything else is stdlib.
ML_FORECASTER_KEYS = ("gbm",)


class RegressorEstimator(Protocol):
    """The slice of scikit-learn's regressor API this module actually uses.

    A Protocol rather than ``Any``: scikit-learn ships no type information, so without this the
    fitted model would be untyped.
    """

    def predict(self, X: Sequence[Sequence[float]]) -> Sequence[float]: ...


@dataclass
class BinHistory:
    """One bin's record of what its own sensors reported, newest last.

    Holds **estimates only**. Nothing that enters this object has seen the true fill or weight.
    """

    #: Fused fill estimate per observed day.
    fv_hat: list[float] = field(default_factory=list)
    #: Noisy load-cell weight per observed day.
    weight: list[float] = field(default_factory=list)
    #: Day index of each observation, so gaps cannot be mistaken for consecutive days.
    days: list[int] = field(default_factory=list)
    #: Day this bin was last collected, or ``None`` if it never has been.
    last_collected_day: int | None = None

    def observe(self, day: int, fv_hat: float, weight_kg: float) -> None:
        self.days.append(day)
        self.fv_hat.append(fv_hat)
        self.weight.append(weight_kg)

    def days_since_collection(self, day: int) -> int:
        """How long since the truck came. Capped by the horizon so an uncollected bin is finite."""
        if self.last_collected_day is None:
            return day + 1
        return day - self.last_collected_day

    def rises(self) -> list[float]:
        """Day-to-day changes in the fill estimate, per day elapsed.

        Steps that span a collection are **negative** (the bin was emptied) and callers drop
        them: a drop of 80 points is the truck, not demand.
        """
        out: list[float] = []
        for index in range(1, len(self.fv_hat)):
            elapsed = self.days[index] - self.days[index - 1]
            if elapsed <= 0:
                continue
            out.append((self.fv_hat[index] - self.fv_hat[index - 1]) / elapsed)
        return out

    def positive_rises(self) -> list[float]:
        """Rises with collection resets removed."""
        return [rise for rise in self.rises() if rise >= 0.0]

    def latest_fv(self) -> float | None:
        return self.fv_hat[-1] if self.fv_hat else None


def estimate_fill(reading: SensorReading, bin_cfg: BinConfig) -> float:
    """The fill estimate a forecaster is allowed to see: the paper's fusion of noisy sensors.

    Uses the full 3-sensor mean plus the load cell, i.e. the paper's own rule, so the forecaster
    is built on top of the paper's method rather than on a privileged signal.
    """
    return fuse(reading, bin_cfg, use_load_cell=True, sensor_indices=None).fill_pct


# --------------------------------------------------------------------------------------------
# Feature construction
# --------------------------------------------------------------------------------------------


def feature_names(cfg: ForecastConfig) -> list[str]:
    """Human-readable feature names, same order as :func:`build_features`."""
    names = [f"fv_lag_{lag}" for lag in range(cfg.history_days)]
    names += [f"weight_lag_{lag}" for lag in range(cfg.history_days)]
    names += [f"fv_rise_{lag}" for lag in range(cfg.history_days - 1)]
    names += ["days_since_collection", "mean_rise", "max_rise"]
    names += [f"area_{name}" for name in AREA_TYPE_NAMES]
    names += [f"dow_{index}" for index in range(DAYS_PER_WEEK)]
    names += ["is_festival"]
    return names


def build_features(
    history: BinHistory,
    *,
    day: int,
    area_type: int,
    weekday: int,
    is_festival: bool,
    cfg: ForecastConfig,
) -> list[float] | None:
    """One feature row for one bin on one day, or ``None`` if history is too short.

    Lags are padded by repeating the oldest observation rather than with zeros: a zero would read
    as "the bin was empty", which is a different and wrong claim.
    """
    if len(history.fv_hat) < cfg.history_days:
        return None

    recent_fv = history.fv_hat[-cfg.history_days :]
    recent_weight = history.weight[-cfg.history_days :]

    # Newest first, so lag 0 is always "now" regardless of history length.
    row = list(reversed(recent_fv))
    row += list(reversed(recent_weight))
    row += [recent_fv[index] - recent_fv[index - 1] for index in range(len(recent_fv) - 1, 0, -1)]

    rises = history.positive_rises()
    row.append(float(history.days_since_collection(day)))
    row.append(sum(rises) / len(rises) if rises else 0.0)
    row.append(max(rises) if rises else 0.0)

    row += [1.0 if index == area_type else 0.0 for index in range(len(AREA_TYPE_NAMES))]
    row += [1.0 if index == weekday else 0.0 for index in range(DAYS_PER_WEEK)]
    row.append(1.0 if is_festival else 0.0)
    return row


@dataclass(frozen=True)
class TrainingRow:
    """One supervised example: features from the past, target rate observed later.

    ``day`` is the day the features were built on. Every consumer uses it to assert the
    time-based split, so it is carried rather than discarded.

    ``span`` is how many days the target rate was measured over. It varies because a span that
    crosses a collection is unusable (see :func:`clean_span`), so the target is always a *daily
    rate* rather than a raw difference -- that keeps rows measured over different spans
    comparable.
    """

    day: int
    features: tuple[float, ...]
    target_rate: float
    span: int = 1


def clean_span(
    day: int,
    horizon: int,
    collection_days: Sequence[int],
    available: Container[int],
) -> int:
    """The longest span up to ``horizon`` that contains no collection and has an observation.

    Returns 0 when no usable span exists.

    ### Why a variable span is necessary rather than merely nice

    A collection resets the fill, so a rise measured across one is the truck rather than demand.
    With a fixed 3-day horizon and a 3-day collection calendar, **every** span contains a
    collection -- any three consecutive days include a multiple of three -- so a fixed horizon
    yields zero training rows and the model silently never fits. Taking the longest clean span
    instead keeps the 3-day-backstop strategies trainable.

    ### The off-by-one

    A day's reading is taken BEFORE that day's collection, so a collection on day ``c`` affects
    readings from ``c + 1`` onward. One on ``day`` itself therefore spoils the span, while one on
    the target day does not. The exclusion window is ``[day, day + span)``.
    """
    collections = set(collection_days)
    best = 0
    for span in range(1, horizon + 1):
        if any(c in collections for c in range(day, day + span)):
            break
        if day + span in available:
            best = span
    return best


# --------------------------------------------------------------------------------------------
# Forecasters
# --------------------------------------------------------------------------------------------


class Forecaster:
    """Predicts a bin's daily rise in estimated fill percent.

    Baselines need no fitting and ignore :meth:`fit`; the GBM overrides it.
    """

    key: str
    label: str
    #: True when this forecaster needs the optional ``ml`` extra installed.
    needs_ml_extra: bool = False

    def fit(self, rows: Sequence[TrainingRow]) -> None:
        """Default: nothing to fit. A naive baseline is defined, not trained."""

    def predict_rate(self, history: BinHistory, features: Sequence[float] | None) -> float:
        raise NotImplementedError

    @property
    def is_fitted(self) -> bool:
        return True


class LastValueForecaster(Forecaster):
    """Baseline 1: tomorrow's rise is the most recent rise."""

    key = "last_value"
    label = FORECASTER_LABELS["last_value"]

    def predict_rate(self, history: BinHistory, features: Sequence[float] | None) -> float:
        rises = history.positive_rises()
        return rises[-1] if rises else 0.0


class MovingAverageForecaster(Forecaster):
    """Baseline 2: mean of the last ``w`` rises.

    This is the baseline the gradient-boosting model has to beat to justify itself, so it is
    reported beside the GBM everywhere and never omitted.
    """

    key = "moving_average"
    label = FORECASTER_LABELS["moving_average"]

    def __init__(self, window: int) -> None:
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        self.window = window

    def predict_rate(self, history: BinHistory, features: Sequence[float] | None) -> float:
        rises = history.positive_rises()
        if not rises:
            return 0.0
        recent = rises[-self.window :]
        return sum(recent) / len(recent)


class SeasonalNaiveForecaster(Forecaster):
    """Baseline 3: the rise observed on the same weekday a week ago.

    Falls back to the mean rise when a week of history is not available, which during early days
    is most of the time -- stated rather than hidden, because it means this baseline is partly a
    mean-rise baseline over a 30-day horizon.
    """

    key = "seasonal_naive"
    label = FORECASTER_LABELS["seasonal_naive"]

    def predict_rate(self, history: BinHistory, features: Sequence[float] | None) -> float:
        rises = history.positive_rises()
        if not rises:
            return 0.0
        if len(rises) >= DAYS_PER_WEEK:
            return rises[-DAYS_PER_WEEK]
        return sum(rises) / len(rises)


class GbmForecaster(Forecaster):
    """Gradient boosting on the feature rows above.

    ### Library choice

    ``sklearn.ensemble.HistGradientBoostingRegressor``. SPEC section 5 lists XGBoost; we use
    scikit-learn's histogram gradient booster instead, which is the same model family with one
    fewer heavy dependency. The substitution is recorded in ``docs/ASSUMPTIONS.md``.

    ### Determinism is same-machine only

    ``random_state`` is pinned, so a re-run on **this machine with these library versions**
    reproduces the fit. scikit-learn does **not** promise bit-identical floats across operating
    systems, BLAS builds or versions, so no test pins an exact model output and
    ``docs/PHASE3_RESULTS.md`` scopes the ML figures to same-machine reproducibility. The
    stdlib parts of this project keep their cross-machine byte-identical guarantee.

    ### Lazy import

    sklearn is imported inside :meth:`fit`, never at module scope, so importing this module (and
    therefore using the naive baselines) works with the ``ml`` extra uninstalled.
    """

    key = "gbm"
    label = FORECASTER_LABELS["gbm"]
    needs_ml_extra = True

    def __init__(self, cfg: ForecastConfig, seed: int) -> None:
        self._cfg = cfg
        self._seed = seed
        self._model: RegressorEstimator | None = None
        self._fallback = MovingAverageForecaster(cfg.moving_average_window)

    @property
    def is_fitted(self) -> bool:
        return self._model is not None

    def fit(self, rows: Sequence[TrainingRow]) -> None:
        """Fit once on the warm-up rows, then freeze.

        With no training rows the model stays unfitted and :meth:`predict_rate` falls back to the
        moving average. That is reported when it happens rather than silently scoring the
        fallback as if it were the model.
        """
        if not rows:
            return
        # Lazy, deliberately: see the class docstring.
        from sklearn.ensemble import HistGradientBoostingRegressor

        features = [list(row.features) for row in rows]
        targets = [row.target_rate for row in rows]
        model = HistGradientBoostingRegressor(
            max_iter=self._cfg.gbm_max_iter,
            learning_rate=self._cfg.gbm_learning_rate,
            max_depth=self._cfg.gbm_max_depth,
            random_state=self._seed,
        )
        model.fit(features, targets)
        self._model = model

    def predict_rate(self, history: BinHistory, features: Sequence[float] | None) -> float:
        model = self._model
        if model is None or features is None:
            return self._fallback.predict_rate(history, features)
        predicted = model.predict([list(features)])
        return float(predicted[0])


def build_forecaster(key: str, cfg: ForecastConfig, seed: int) -> Forecaster:
    """One forecaster by key. Baselines first in :data:`FORECASTER_KEYS`."""
    if key == "last_value":
        return LastValueForecaster()
    if key == "moving_average":
        return MovingAverageForecaster(cfg.moving_average_window)
    if key == "seasonal_naive":
        return SeasonalNaiveForecaster()
    if key == "gbm":
        return GbmForecaster(cfg, seed)
    raise ValueError(f"unknown forecaster key: {key!r} (known: {FORECASTER_KEYS})")
