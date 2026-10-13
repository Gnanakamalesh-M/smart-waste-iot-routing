"""The five collection strategies the benchmark compares.

Each strategy answers one question per day: **which bins do we send a truck to?** It is given the
day number and that day's sensor readings, and nothing else. It cannot see the true fill, the true
weight or any other strategy's decisions -- a strategy only ever receives
:class:`~smartwaste.noise.SensorReading` objects, which structurally carry no ground truth.

| id | label | what it reads |
|---|---|---|
| a | fixed-route, every bin daily | nothing (the paper's baseline) |
| b | fixed schedule, every N days | nothing |
| c | single ultrasonic threshold | centre sensor only, no load cell |
| d | paper's sensor fusion | 3-sensor MEAN + load cell |
| e | median fusion | 3-sensor MEDIAN + load cell -- **extension, not in the paper** |

(a) and (b) are the fair baselines SPEC section 3 asks for: a schedule can be run with no sensors
at all, so if a schedule matches the sensor strategies then the sensors are not earning their
keep. (c) isolates what the extra two ultrasonic sensors and the load cell actually add.

(d) is the paper's rule, unchanged. (e) is ours: Phase 1 measured the mean collapsing when one
sensor sticks (20.85 pp worse than a single sensor), which a median would survive. (e) is expected
to lose slightly to (d) when no sensor is broken, because a median of three is a noisier estimator
than a mean -- see :func:`smartwaste.fusion.median_distance`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from smartwaste.config import BinConfig, Config, ForecastConfig
from smartwaste.demand import DemandCalendar
from smartwaste.fusion import BinState, fuse, mean_distance, median_distance
from smartwaste.ml.forecast import (
    BinHistory,
    Forecaster,
    TrainingRow,
    build_features,
    build_forecaster,
    clean_span,
    estimate_fill,
)
from smartwaste.noise import CENTER_INDEX, SensorReading


@dataclass(frozen=True)
class RunContext:
    """The city structure a strategy is allowed to know before day 0.

    Area type, weekday and festival flag are **operational records**, not ground truth: a real
    utility knows which bins sit in a market and when the public holidays are without any
    sensor. What this object deliberately does **not** carry is any true fill or true weight --
    so a strategy still cannot see the thing it is trying to estimate.

    Phase 2's five strategies ignore it entirely.
    """

    calendar: DemandCalendar
    bin_cfg: BinConfig
    n_bins: int


class Strategy(ABC):
    """A collection policy: given a day's readings, choose bins to visit."""

    #: Stable short key used in CSV output and cache keys.
    key: str
    #: Human-readable label, printed in the results tables exactly as written.
    label: str
    #: True when the strategy consults sensors at all.
    uses_sensors: bool

    @abstractmethod
    def select(self, day: int, readings: Sequence[SensorReading]) -> list[int]:
        """Indices of the bins to collect today, ascending."""

    def begin_run(self, context: RunContext) -> None:
        """Called once before day 0. Stateless strategies ignore it, which is the default.

        Exists so a forecast-driven strategy can learn the calendar and the area types without
        changing :meth:`select`'s signature -- the five Phase 2 strategies are untouched.

        Deliberately a concrete no-op rather than an abstractmethod: making it abstract would
        force Phase 2's five strategies to implement a hook none of them needs.
        """
        return None

    def observe_collection(self, day: int, selected: Sequence[int]) -> None:
        """Called after the day's collection with the bins THIS strategy chose.

        A forecaster needs it to reset a collected bin's history. A strategy is told only its own
        decisions, never what the truck found.

        Concrete no-op by default, for the same reason as :meth:`begin_run`.
        """
        return None


class FixedRoute(Strategy):
    """(a) Visit every bin every day -- the paper's baseline.

    Reads no sensors, so it never misses a full bin and never overflows. Its cost is the price of
    that guarantee, and it is the number every saving in this project is measured against.
    """

    key = "a_fixed_route"
    label = "(a) fixed-route: all bins daily [paper's baseline]"
    uses_sensors = False

    def select(self, day: int, readings: Sequence[SensorReading]) -> list[int]:
        return list(range(len(readings)))


class FixedSchedule(Strategy):
    """(b) Visit every bin every N days.

    The cheapest honest baseline: no sensors, no telemetry, no capital cost. If this matches the
    sensor strategies on overflow and beats them on cost, the sensors are not paying for
    themselves -- and the results doc says so rather than burying it.
    """

    key = "b_fixed_schedule"
    label = "(b) fixed schedule: all bins every N days"
    uses_sensors = False

    def __init__(
        self,
        every_n_days: int,
        *,
        key: str | None = None,
        label: str | None = None,
    ) -> None:
        if every_n_days < 1:
            raise ValueError(f"every_n_days must be >= 1, got {every_n_days}")
        self.every_n_days = every_n_days
        self.label = f"(b) fixed schedule: all bins every {every_n_days} days"
        # Phase 3 adds a second, longer calendar as the fair baseline for the 7-day backstop, and
        # it needs its own key and label in the results tables. Both default to Phase 2's.
        if key is not None:
            self.key = key
        if label is not None:
            self.label = label

    def select(self, day: int, readings: Sequence[SensorReading]) -> list[int]:
        if day % self.every_n_days != 0:
            return []
        return list(range(len(readings)))


class ThresholdStrategy(Strategy):
    """Collect exactly the bins this sensor configuration classifies as RED.

    All three sensor strategies share this: they differ only in which distances they read, whether
    they use the load cell, and how they reduce the distances to ``d_bar``. YELLOW bins are never
    collected, which is the paper's rule and its known weakness (Phase 3 addresses it).
    """

    uses_sensors = True

    def __init__(
        self,
        key: str,
        label: str,
        bin_cfg: BinConfig,
        *,
        use_load_cell: bool,
        sensor_indices: tuple[int, ...] | None,
        reducer: Callable[[Sequence[float]], float] | None = None,
    ) -> None:
        self.key = key
        self.label = label
        self._bin = bin_cfg
        self._use_load_cell = use_load_cell
        self._sensor_indices = sensor_indices
        self._reducer = mean_distance if reducer is None else reducer

    def select(self, day: int, readings: Sequence[SensorReading]) -> list[int]:
        selected: list[int] = []
        for index, reading in enumerate(readings):
            result = fuse(
                reading,
                self._bin,
                use_load_cell=self._use_load_cell,
                sensor_indices=self._sensor_indices,
                reducer=self._reducer,
            )
            if result.state is BinState.RED:
                selected.append(index)
        return selected


class ForecastDriven(Strategy):
    """(f) Collect a bin when its FORECAST reaches RED before the next scheduled visit.

    ### The rule, in full

    On each day, for each bin, collect it if **any** of:

    1. the paper's fusion already classifies it RED today (so (f) is never worse informed than
       (d) about the present);
    2. its backstop visit is due -- the schedule is a floor, not a suggestion, so a forecast that
       is badly wrong cannot abandon a bin indefinitely;
    3. ``Fv_hat_now + rate * days_until_next_visit >= red_threshold`` -- it is predicted to hit
       RED before anyone would next come.

    ### Warm-up, and why the comparison window exists

    Until ``warmup_days`` the forecaster has no fitted model and too little history, so (f) runs
    **its backstop schedule alone** and merely logs what its sensors report. The model is fitted
    once at ``warmup_days`` from those logs and then frozen -- the time-based split required by
    CLAUDE.md.

    This is exactly why the Phase 3 benchmark scores every strategy on days
    ``warmup_days..end`` as well as on the full horizon: a 30-day total for (f) would include 14
    days in which it was simply a calendar, which would flatter or flatten it for no modelling
    reason.

    ### Each (f) is compared against the calendar of its OWN period

    A 7-day-backstop (f) beating a 3-day calendar would prove nothing about forecasting -- it
    collects far less often. So the benchmark pairs ``backstop=3`` against the sensor-free 3-day
    calendar and ``backstop=7`` against the sensor-free 7-day calendar.

    ### No ground truth, by construction

    The only per-bin inputs are :class:`~smartwaste.noise.SensorReading` objects (no truth field
    exists on them) and this strategy's own past decisions. Area type, weekday and festival flag
    come from :class:`RunContext` and are operational records, not measurements of fill.
    """

    uses_sensors = True

    def __init__(
        self,
        key: str,
        label: str,
        bin_cfg: BinConfig,
        forecast_cfg: ForecastConfig,
        forecaster: Forecaster,
        *,
        backstop_days: int,
    ) -> None:
        if backstop_days < 1:
            raise ValueError(f"backstop_days must be >= 1, got {backstop_days}")
        self.key = key
        self.label = label
        self._bin = bin_cfg
        self._fc = forecast_cfg
        self.forecaster = forecaster
        self.backstop_days = backstop_days
        self._context: RunContext | None = None
        self._history: list[BinHistory] = []
        # Feature rows logged during warm-up, awaiting the target that only later days reveal.
        self._pending: list[list[tuple[int, tuple[float, ...]]]] = []
        self._collection_days: list[list[int]] = []
        self._fv_by_day: list[dict[int, float]] = []
        self._fitted = False
        #: Training rows actually used, exposed so the runner can report how thin the warm-up was.
        self.training_rows: int = 0

    # -- lifecycle ---------------------------------------------------------------------------

    def begin_run(self, context: RunContext) -> None:
        self._context = context
        n = context.n_bins
        self._history = [BinHistory() for _ in range(n)]
        self._pending = [[] for _ in range(n)]
        self._collection_days = [[] for _ in range(n)]
        self._fv_by_day = [{} for _ in range(n)]
        self._fitted = False
        self.training_rows = 0

    def observe_collection(self, day: int, selected: Sequence[int]) -> None:
        for index in selected:
            self._history[index].last_collected_day = day
            self._collection_days[index].append(day)

    # -- the daily decision ------------------------------------------------------------------

    def select(self, day: int, readings: Sequence[SensorReading]) -> list[int]:
        context = self._context
        if context is None:
            raise RuntimeError("begin_run must be called before select; the runner does this")

        features_today = self._observe_day(day, readings, context)

        # Fit exactly once, at the warm-up boundary, on warm-up data only.
        if day >= self._fc.warmup_days and not self._fitted:
            self._fit(context)

        warming_up = day < self._fc.warmup_days
        selected: list[int] = []
        for index, reading in enumerate(readings):
            if self._should_collect(
                index, day, reading, features_today[index], warming_up=warming_up
            ):
                selected.append(index)
        return selected

    def _observe_day(
        self, day: int, readings: Sequence[SensorReading], context: RunContext
    ) -> list[list[float] | None]:
        """Log this day's fused estimates and build each bin's feature row."""
        features: list[list[float] | None] = []
        weekday = context.calendar.weekday[day]
        is_festival = context.calendar.is_festival[day]
        for index, reading in enumerate(readings):
            fv_hat = estimate_fill(reading, self._bin)
            self._history[index].observe(day, fv_hat, reading.weight_kg)
            self._fv_by_day[index][day] = fv_hat
            row = build_features(
                self._history[index],
                day=day,
                area_type=context.calendar.area_type[index],
                weekday=weekday,
                is_festival=is_festival,
                cfg=self._fc,
            )
            features.append(row)
            if day < self._fc.warmup_days and row is not None:
                self._pending[index].append((day, tuple(row)))
        return features

    def _should_collect(
        self,
        index: int,
        day: int,
        reading: SensorReading,
        features: Sequence[float] | None,
        *,
        warming_up: bool,
    ) -> bool:
        history = self._history[index]

        # The backstop is the SAME absolute calendar that FixedSchedule uses, not a per-bin
        # "days since last collection" clock. Two things depend on that:
        #
        #  * during warm-up (f) is then *exactly* its sensor-free calendar partner, which is what
        #    makes the post-warm-up window a clean comparison;
        #  * (f)'s visits are a strict SUPERSET of the calendar's, so a wrong forecast can only
        #    ever add a visit, never strand a bin by pushing its next visit later.
        #
        # A relative clock breaks both: an early forecast-driven pickup would reset it and the
        # following scheduled visit would drift, so (f) could end up visiting a bin *less* often
        # than the calendar it is being compared against.
        if day % self.backstop_days == 0:
            return True
        if warming_up:
            # Calendar only: the forecaster is not trusted before it has been fitted.
            return False

        already_red = (
            fuse(reading, self._bin, use_load_cell=True, sensor_indices=None).state is BinState.RED
        )
        if already_red:
            return True

        current = history.latest_fv()
        if current is None:
            return False
        rate = self.forecaster.predict_rate(history, features)
        projected = current + rate * self._days_until_next_visit(day)
        return projected >= self._bin.red_threshold_pct

    def _days_until_next_visit(self, day: int) -> int:
        """Days until the next scheduled (backstop) visit. Always >= 1."""
        next_visit = (day // self.backstop_days + 1) * self.backstop_days
        return next_visit - day

    # -- training ----------------------------------------------------------------------------

    def _fit(self, context: RunContext) -> None:
        """Build training rows from warm-up logs only, then fit and freeze.

        A row is dropped when the bin was collected inside its span: the fill drop across a
        collection measures the truck, not demand, and training on it would teach the model that
        bins spontaneously empty.

        The span boundaries are an off-by-one trap. A day's reading is taken BEFORE that day's
        collection, so a collection on day ``c`` affects readings from ``c + 1`` onward. One on
        ``day`` itself therefore corrupts the span, while one on ``target_day`` does not --
        hence ``day <= c < target_day``.
        """
        horizon = self._fc.horizon_days
        rows: list[TrainingRow] = []
        for index, pending in enumerate(self._pending):
            fv_by_day = self._fv_by_day[index]
            collected = self._collection_days[index]
            for day, feature_row in pending:
                span = clean_span(day, horizon, collected, fv_by_day)
                if span == 0:
                    continue
                rate = (fv_by_day[day + span] - fv_by_day[day]) / span
                rows.append(TrainingRow(day=day, features=feature_row, target_rate=rate, span=span))

        self.forecaster.fit(rows)
        self.training_rows = len(rows)
        self._fitted = True


def build_strategies(bin_cfg: BinConfig, fixed_schedule_days: int) -> tuple[Strategy, ...]:
    """The five strategies, in the order they appear in every results table."""
    return (
        FixedRoute(),
        FixedSchedule(fixed_schedule_days),
        ThresholdStrategy(
            "c_single_ultrasonic",
            "(c) single-ultrasonic threshold: center sensor only, no load cell",
            bin_cfg,
            use_load_cell=False,
            sensor_indices=(CENTER_INDEX,),
        ),
        ThresholdStrategy(
            "d_paper_fusion",
            "(d) paper's sensor fusion: 3-sensor MEAN + load cell [the paper's method]",
            bin_cfg,
            use_load_cell=True,
            sensor_indices=None,
            reducer=mean_distance,
        ),
        ThresholdStrategy(
            "e_median_fusion",
            "(e) median fusion: 3-sensor MEDIAN + load cell [extension, not in the paper]",
            bin_cfg,
            use_load_cell=True,
            sensor_indices=None,
            reducer=median_distance,
        ),
    )


#: Phase 3's forecast-driven keys that need the optional ``ml`` extra installed.
GBM_STRATEGY_KEYS = ("f_gbm_3", "f_gbm_7")


def forecast_strategy_key(forecaster_key: str, backstop_days: int) -> str:
    """Stable key for one (forecaster, backstop) pair, e.g. ``"f_gbm_7"``."""
    short = {"moving_average": "ma", "gbm": "gbm", "last_value": "lv", "seasonal_naive": "sn"}
    return f"f_{short.get(forecaster_key, forecaster_key)}_{backstop_days}"


def build_phase3_strategies(
    cfg: Config,
    seed: int,
    *,
    include_gbm: bool = True,
) -> tuple[Strategy, ...]:
    """Phase 3's full strategy set: Phase 2's five, a 7-day calendar, and the (f) variants.

    ### Why the 7-day calendar is here

    Phase 2 compared everything against (b), a 3-day calendar. A 7-day-backstop (f) collects far
    less often than that, so measuring it against (b) would confuse "forecasting works" with
    "visiting less often is cheaper". Every (f) is therefore paired with the sensor-free calendar
    of its own period, and both calendars are run.

    ### ``include_gbm``

    The gradient-boosting variants are the only strategies that need scikit-learn. Passing
    ``include_gbm=False`` yields the eight pure-stdlib strategies, which is what the
    ``unit`` / ``sim`` / ``routing`` / ``slow`` test chunks use so that none of them requires the
    optional ``ml`` extra. The scripts use the default and run all ten.

    ``build_strategies`` above is left exactly as Phase 2 wrote it, returning its five -- Phase 2
    must keep reproducing its own published numbers.
    """
    strategies: list[Strategy] = list(build_strategies(cfg.bin, cfg.benchmark.fixed_schedule_days))

    backstops = cfg.forecast.backstops
    # A sensor-free calendar for every backstop period that is not already covered by (b).
    for backstop in backstops:
        if backstop == cfg.benchmark.fixed_schedule_days:
            continue
        strategies.append(
            FixedSchedule(
                backstop,
                key=f"b{backstop}_fixed_schedule",
                label=(
                    f"(b{backstop}) fixed schedule: all bins every {backstop} days "
                    f"[sensor-free baseline for the {backstop}-day backstop]"
                ),
            )
        )

    forecaster_keys = ["moving_average"] + (["gbm"] if include_gbm else [])
    for forecaster_key in forecaster_keys:
        for backstop in backstops:
            kind = (
                "moving-average forecast [naive baseline]"
                if forecaster_key == "moving_average"
                else "gradient-boosting forecast [ML]"
            )
            key = forecast_strategy_key(forecaster_key, backstop)
            strategies.append(
                ForecastDriven(
                    key,
                    f"({key}) forecast-driven collection: {kind}, {backstop}-day backstop",
                    cfg.bin,
                    cfg.forecast,
                    build_forecaster(forecaster_key, cfg.forecast, seed),
                    backstop_days=backstop,
                )
            )
    return tuple(strategies)
