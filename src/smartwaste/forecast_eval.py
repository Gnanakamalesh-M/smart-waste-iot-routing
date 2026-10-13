"""Phase 3 forecast evaluation: do the ML model's predictions beat the naive baselines?

One command (``scripts/run_forecast.py``) produces every forecasting table in
``docs/PHASE3_RESULTS.md``.

### What is measured

Each forecaster predicts a bin's fill ``horizon_days`` ahead. The error is reported in
**percentage points of fill**, which is the unit the RED threshold is expressed in, so an MAE of
5 pp can be read directly against the 80 pp threshold.

All four forecasters are scored on the **identical rows** -- same city, same arrivals, same
sensor noise, same observation history -- so the comparison is paired and the differences are not
confounded by the draw.

### Baselines first, and reported even when they win

:data:`~smartwaste.ml.forecast.FORECASTER_KEYS` puts the three naive baselines ahead of the
gradient-boosting model, and :func:`verdict_lines` states plainly when a baseline matches or beats
it. CLAUDE.md requires that; it is computed here rather than left for a reader to infer.

### Where the observation history comes from

The city is simulated with the ordinary :func:`~smartwaste.city_sim.run_strategy` loop under a
fixed calendar, using a recording strategy that logs each bin's **fused noisy estimate** every
day. Reusing the real simulation loop (rather than writing a second one here) is deliberate: a
private copy of the day order, the sensor model or the overflow rules would silently drift away
from what the benchmark actually does.

The calendar, not a forecast, drives collection during evaluation. That keeps the dataset
identical for all four forecasters -- a forecaster that changed the collection pattern would be
scored on a history of its own making.

### No leakage

The recorder logs only :class:`~smartwaste.noise.SensorReading` values, which carry no
ground-truth field. Targets are future *fused estimates*, never the true fill. The split is
strictly by time: train on days ``< warmup_days``, evaluate on days ``>= warmup_days``.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass, field

from smartwaste.city_sim import build_layout, run_strategy
from smartwaste.config import Config
from smartwaste.demand import SCENARIOS
from smartwaste.ml.forecast import (
    FORECASTER_KEYS,
    BinHistory,
    TrainingRow,
    build_features,
    build_forecaster,
    clean_span,
    estimate_fill,
)
from smartwaste.ml.metrics import mae, rmse
from smartwaste.noise import SensorReading
from smartwaste.routing.cluster import kmeans
from smartwaste.seeds import child_seed
from smartwaste.strategies import FixedSchedule, RunContext

CSV_FIELDS = (
    "scenario",
    "seed",
    "forecaster_key",
    "forecaster_label",
    "n_train_rows",
    "n_eval_rows",
    "mae_pp",
    "rmse_pp",
)


class RecordingSchedule(FixedSchedule):
    """A fixed calendar that also logs every bin's fused noisy estimate each day.

    Subclasses the real strategy so collection timing, and therefore the fill trajectories the
    forecasters see, are exactly a calendar's -- no bespoke simulation loop.
    """

    def __init__(self, every_n_days: int, cfg: Config) -> None:
        super().__init__(every_n_days, key="recorder", label="recorder [forecast evaluation]")
        self._cfg = cfg
        self.histories: list[BinHistory] = []
        self.fv_by_day: list[dict[int, float]] = []
        self.collection_days: list[list[int]] = []
        self.features: list[dict[int, tuple[float, ...]]] = []
        self.context: RunContext | None = None

    def begin_run(self, context: RunContext) -> None:
        self.context = context
        n = context.n_bins
        self.histories = [BinHistory() for _ in range(n)]
        self.fv_by_day = [{} for _ in range(n)]
        self.collection_days = [[] for _ in range(n)]
        self.features = [{} for _ in range(n)]

    def observe_collection(self, day: int, selected: Sequence[int]) -> None:
        for index in selected:
            self.histories[index].last_collected_day = day
            self.collection_days[index].append(day)

    def select(self, day: int, readings: Sequence[SensorReading]) -> list[int]:
        context = self.context
        if context is None:
            raise RuntimeError("begin_run must be called before select; the runner does this")
        weekday = context.calendar.weekday[day]
        is_festival = context.calendar.is_festival[day]
        for index, reading in enumerate(readings):
            fv_hat = estimate_fill(reading, self._cfg.bin)
            self.histories[index].observe(day, fv_hat, reading.weight_kg)
            self.fv_by_day[index][day] = fv_hat
            row = build_features(
                self.histories[index],
                day=day,
                area_type=context.calendar.area_type[index],
                weekday=weekday,
                is_festival=is_festival,
                cfg=self._cfg.forecast,
            )
            if row is not None:
                self.features[index][day] = tuple(row)
        return super().select(day, readings)


@dataclass(frozen=True)
class EvalRow:
    """One supervised example held for evaluation, with the context a prediction needs."""

    bin_index: int
    day: int
    features: tuple[float, ...]
    fv_now: float
    #: The future fused estimate -- the target. Never the true fill.
    fv_future: float
    #: Days between ``fv_now`` and ``fv_future``. Varies because spans crossing a collection are
    #: unusable, so predictions are scaled by it rather than by a fixed horizon.
    span: int = 1


@dataclass
class Dataset:
    """Train and evaluation rows, split strictly by day."""

    train: list[TrainingRow] = field(default_factory=list)
    eval_rows: list[EvalRow] = field(default_factory=list)
    histories: list[BinHistory] = field(default_factory=list)
    warmup_days: int = 0
    horizon_days: int = 1
    festival_days_in_warmup: int = 0
    weekdays_in_warmup: int = 0
    total_festival_days: int = 0

    def latest_train_day(self) -> int | None:
        return max((row.day for row in self.train), default=None)

    def earliest_eval_day(self) -> int | None:
        return min((row.day for row in self.eval_rows), default=None)


def build_dataset(cfg: Config, seed: int, *, scenario: str, n_bins: int, days: int) -> Dataset:
    """Simulate one city under a calendar and split its observation log by time.

    The recorder's own per-bin :class:`BinHistory` objects are reused for prediction, which is
    what lets a naive baseline (which needs the history, not a feature row) and the GBM (which
    needs the feature row) be scored on the same rows.
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"scenario must be one of {SCENARIOS}, got {scenario!r}")

    layout = build_layout(
        cfg,
        seed,
        n_bins=n_bins,
        days=days,
        stuck_bin_fraction=cfg.sensors.stuck_bin_fraction,
        scenario=scenario,
    )
    clusters = kmeans(
        layout.locations, cfg.routing.n_clusters, random.Random(child_seed(seed, "clusters"))
    )
    # The observation calendar must be LONGER than the forecast horizon. With a 3-day calendar
    # and a 3-day horizon every span (day, day+3] contains a collection, so every row would be
    # dropped as a reset and the dataset would come out empty. The longest configured backstop
    # is used, which is also the schedule under which forecasting has the most to contribute.
    observation_period = max(max(cfg.forecast.backstops), cfg.forecast.horizon_days + 1)
    recorder = RecordingSchedule(observation_period, cfg)
    run_strategy(cfg, layout, clusters, recorder, seed)

    warmup = cfg.forecast.warmup_days
    horizon = cfg.forecast.horizon_days
    dataset = Dataset(
        histories=recorder.histories,
        warmup_days=warmup,
        horizon_days=horizon,
        festival_days_in_warmup=layout.calendar.festival_days_in(0, warmup),
        weekdays_in_warmup=layout.calendar.weekdays_in(0, warmup),
        total_festival_days=layout.calendar.n_festival_days,
    )

    for bin_index in range(layout.n_bins):
        fv_by_day = recorder.fv_by_day[bin_index]
        collected = recorder.collection_days[bin_index]
        for day, feature_row in sorted(recorder.features[bin_index].items()):
            # The longest span up to the horizon that contains no collection. A rise measured
            # across a collection is the truck emptying the bin, not demand; see clean_span for
            # the off-by-one on the span boundaries.
            span = clean_span(day, horizon, collected, fv_by_day)
            if span == 0:
                continue
            fv_now = fv_by_day[day]
            fv_future = fv_by_day[day + span]
            if day < warmup:
                dataset.train.append(
                    TrainingRow(
                        day=day,
                        features=feature_row,
                        target_rate=(fv_future - fv_now) / span,
                        span=span,
                    )
                )
            else:
                dataset.eval_rows.append(
                    EvalRow(
                        bin_index=bin_index,
                        day=day,
                        features=feature_row,
                        fv_now=fv_now,
                        fv_future=fv_future,
                        span=span,
                    )
                )
    return dataset


@dataclass(frozen=True)
class ForecastScore:
    """One forecaster's accuracy on one dataset."""

    scenario: str
    seed: int
    key: str
    label: str
    n_train: int
    n_eval: int
    mae_pp: float
    rmse_pp: float


def score_forecasters(
    cfg: Config,
    dataset: Dataset,
    seed: int,
    *,
    scenario: str,
    keys: Sequence[str] = FORECASTER_KEYS,
) -> list[ForecastScore]:
    """Fit each forecaster on the train split and score it on the eval split.

    Predictions are rates; they are converted to a fill level
    (``fv_now + rate * horizon``) before scoring, so the error is in percentage points of fill.

    **Prediction uses a history truncated to the row's own day.** Handing a forecaster the
    bin's complete history would let it average over days that had not happened yet -- leakage
    through the back door. Each row is predicted from a replayed prefix instead.
    """
    scores: list[ForecastScore] = []

    # Replayed prefixes, built once and shared by every forecaster so the comparison stays paired.
    prefixes = _history_prefixes(dataset)

    for key in keys:
        forecaster = build_forecaster(key, cfg.forecast, seed)
        forecaster.fit(dataset.train)

        actual: list[float] = []
        predicted: list[float] = []
        for position, row in enumerate(dataset.eval_rows):
            rate = forecaster.predict_rate(prefixes[position], row.features)
            # Scaled by this row's own span, which is what makes rows measured over different
            # clean spans comparable.
            predicted.append(row.fv_now + rate * row.span)
            actual.append(row.fv_future)

        scores.append(
            ForecastScore(
                scenario=scenario,
                seed=seed,
                key=key,
                label=forecaster.label,
                n_train=len(dataset.train),
                n_eval=len(dataset.eval_rows),
                mae_pp=mae(actual, predicted) if actual else 0.0,
                rmse_pp=rmse(actual, predicted) if actual else 0.0,
            )
        )
    return scores


def _history_prefixes(dataset: Dataset) -> list[BinHistory]:
    """A per-eval-row history containing only observations up to that row's day.

    Without this a baseline would be allowed to see the future: ``BinHistory`` accumulated over
    the whole run, and a moving average over it would include days after the one being predicted.
    """
    prefixes: list[BinHistory] = []
    for row in dataset.eval_rows:
        full = dataset.histories[row.bin_index]
        prefix = BinHistory()
        for position, day in enumerate(full.days):
            if day > row.day:
                break
            prefix.observe(day, full.fv_hat[position], full.weight[position])
        prefix.last_collected_day = full.last_collected_day
        prefixes.append(prefix)
    return prefixes


# --------------------------------------------------------------------------------------------
# Rendering. No timings and no timestamps, so two runs at one seed are byte-identical.
# --------------------------------------------------------------------------------------------


def _fmt(value: float, places: int = 3) -> str:
    return f"{value:.{places}f}"


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def render_accuracy(scores: Sequence[ForecastScore], scenario: str) -> str:
    """MAE / RMSE per forecaster, baselines listed first."""
    chosen = [s for s in scores if s.scenario == scenario]
    rows = [
        [
            s.label,
            str(s.seed),
            str(s.n_train),
            str(s.n_eval),
            _fmt(s.mae_pp),
            _fmt(s.rmse_pp),
        ]
        for s in chosen
    ]
    header = (
        "forecaster",
        "seed",
        "train rows",
        "eval rows",
        "MAE (fill pp)",
        "RMSE (fill pp)",
    )
    title = f"### Forecast accuracy, demand scenario `{scenario}`"
    note = (
        "Error is in **percentage points of fill**, so it reads directly against the RED\n"
        "threshold of 80 pp. All forecasters are scored on the **same rows** (same city, same\n"
        "arrivals, same sensor noise), so the comparison is paired. Baselines are listed first.\n"
        "The target is the **future fused sensor estimate**, not the true fill -- a forecaster on\n"
        "real hardware has no truth to regress on."
    )
    return f"{title}\n\n{note}\n\n{_table(header, rows)}"


def render_warmup_diagnostics(datasets: Sequence[tuple[str, int, Dataset]]) -> str:
    """How much history and seasonality the warm-up actually contained.

    This exists because the warm-up is short by construction (14 days), so the model may see one
    or zero festival days while being scored on days that contain them. Reporting it is the
    honest alternative to lengthening the warm-up until the model looks better.
    """
    rows = [
        [
            scenario,
            str(seed),
            str(dataset.warmup_days),
            str(dataset.weekdays_in_warmup),
            str(dataset.festival_days_in_warmup),
            str(dataset.total_festival_days),
            str(len(dataset.train)),
            str(len(dataset.eval_rows)),
        ]
        for scenario, seed, dataset in datasets
    ]
    header = (
        "scenario",
        "seed",
        "warm-up days",
        "distinct weekdays in warm-up",
        "festival days in warm-up",
        "festival days in horizon",
        "train rows",
        "eval rows",
    )
    title = "### Warm-up diagnostics: how much the model had to learn from"
    note = (
        "**A limitation, stated rather than fixed.** `forecast.warmup_days = 14` gives the model\n"
        "two weekday cycles at most, and it may see **zero** festival days in training while\n"
        "being evaluated on days that contain them. No constant was changed to improve this:\n"
        "lengthening the warm-up until the ML model won would be tuning to the result."
    )
    return f"{title}\n\n{note}\n\n{_table(header, rows)}"


def verdict_lines(scores: Sequence[ForecastScore], scenario: str) -> list[str]:
    """State plainly whether a naive baseline matched or beat the ML model.

    Required by CLAUDE.md: "If a simple baseline matches or beats the ML model, report that
    plainly. Do not hide it."
    """
    chosen = {s.key: s for s in scores if s.scenario == scenario}
    gbm = chosen.get("gbm")
    if gbm is None:
        return [f"  {scenario}: the ML model was not run, so there is nothing to compare"]

    lines: list[str] = []
    for key, score in chosen.items():
        if key == "gbm":
            continue
        if score.mae_pp < gbm.mae_pp:
            lines.append(
                f"  {scenario}: {score.label} BEATS the ML model on MAE "
                f"({_fmt(score.mae_pp)} vs {_fmt(gbm.mae_pp)} pp) -- the ML model does not "
                f"earn its keep against this baseline"
            )
        elif score.mae_pp == gbm.mae_pp:
            lines.append(
                f"  {scenario}: {score.label} TIES the ML model on MAE "
                f"({_fmt(score.mae_pp)} pp each)"
            )
        else:
            lines.append(
                f"  {scenario}: the ML model beats {score.label} on MAE "
                f"({_fmt(gbm.mae_pp)} vs {_fmt(score.mae_pp)} pp)"
            )
    return lines


def csv_rows(scores: Sequence[ForecastScore]) -> list[dict[str, object]]:
    """Flatten for ``csv.DictWriter`` using :data:`CSV_FIELDS`."""
    return [
        {
            "scenario": s.scenario,
            "seed": s.seed,
            "forecaster_key": s.key,
            "forecaster_label": s.label,
            "n_train_rows": s.n_train,
            "n_eval_rows": s.n_eval,
            "mae_pp": _fmt(s.mae_pp),
            "rmse_pp": _fmt(s.rmse_pp),
        }
        for s in scores
    ]
