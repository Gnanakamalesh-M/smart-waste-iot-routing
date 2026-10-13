"""City-scale simulation: 5000 bins on a 2D grid over 30 days.

**This module removes Phase 1's biggest shortcut.** In Phase 1 every method saw the identical fill
trajectory because collection was triggered by *ground truth*, so a method that missed a full bin
paid no price. Here each strategy acts on **its own noisy classification**: if a strategy fails to
see that a bin is full, that bin is not emptied, it keeps filling, and it overflows. The
trajectories genuinely diverge, which is the only way overflow and missed pickups mean anything.

### Keeping the comparison fair anyway

Letting trajectories diverge creates a trap. If every random number came from one shared stream,
strategy (d) collecting on day 5 would shift the *noise draws* strategy (e) sees, and the two
would be compared on different sensor errors as well as different decisions. The fix is to make
all **exogenous** randomness a pure function of ``(seed, bin, day)`` rather than of call order:

- Daily waste increments are pre-generated per ``(bin, day)`` in :func:`build_layout`, before any
  strategy runs, from the seed label ``"fill:<bin>:<day>"``.
- Each bin's :class:`~smartwaste.noise.SensorArray` is rebuilt per strategy from the same label
  ``"sensors:<bin>"``, and is read exactly once per bin per day **whether or not** the strategy
  looks at the reading. So the noise sequence a bin produces is identical across strategies.
- The fault draw has its own label ``"faults"``, so changing ``stuck_bin_fraction`` does not
  reshuffle the layout or the fills.

Result: all five strategies face the same city, the same waste arrivals and the same sensor
errors. Only their decisions -- and the consequences of those decisions -- differ.

### Overflow model

True fill is capped at 100% (the paper's ``Fv`` is a percentage and its rule assumes
``[0, 100]``), but weight keeps accruing up to ``bin.max_accumulated_weight_kg`` so the load cell
can still escalate a missed bin to RED and eventually rescue it. The cap exists so an abandoned
bin's weight cannot grow without bound; it is an invented, uncited number.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from smartwaste.config import Config
from smartwaste.cost import TruckDay
from smartwaste.demand import (
    SCENARIO_SMOOTH,
    DemandCalendar,
    build_calendar,
    generate_increments,
)
from smartwaste.fusion import BinState, truth_state
from smartwaste.noise import (
    NO_STUCK_SENSOR,
    SKEW_CENTER_PEAK,
    NoiseParams,
    SensorArray,
    SensorReading,
)
from smartwaste.routing.cluster import ClusterResult
from smartwaste.routing.tsp import Point, RouteCache
from smartwaste.seeds import child_seed
from smartwaste.strategies import RunContext, Strategy

#: True fill is a percentage, so 100 is full. At or above this the bin is overflowing.
FULL_PCT = 100.0


@dataclass(frozen=True)
class CityLayout:
    """A city: where the bins are, how dense their waste is, which sensors are broken.

    Fixed for a given ``(seed, n_bins, days, stuck_bin_fraction)``, and shared by every strategy.
    """

    locations: tuple[Point, ...]
    #: kg of waste per 1% of volume, per bin. The spread is what gives the load cell information.
    kg_per_pct: tuple[float, ...]
    #: Index of the stuck ultrasonic sensor per bin, or ``NO_STUCK_SENSOR``.
    stuck_sensor: tuple[int, ...]
    #: ``increments[bin][day]`` kg of waste arriving. Pre-generated, decision-independent.
    increments: tuple[tuple[float, ...], ...]
    depot: Point
    #: Area types, weekdays and festival days. Under the ``"smooth"`` scenario this is neutral
    #: (one area type, no festivals), which is what keeps Phase 2's numbers untouched.
    calendar: DemandCalendar

    @property
    def scenario(self) -> str:
        return self.calendar.scenario

    @property
    def n_bins(self) -> int:
        return len(self.locations)

    @property
    def days(self) -> int:
        return len(self.increments[0]) if self.increments else 0

    @property
    def n_faulted(self) -> int:
        return sum(1 for s in self.stuck_sensor if s != NO_STUCK_SENSOR)


def build_layout(
    cfg: Config,
    seed: int,
    *,
    n_bins: int,
    days: int,
    stuck_bin_fraction: float,
    scenario: str = SCENARIO_SMOOTH,
) -> CityLayout:
    """Generate the city. Nothing here depends on any strategy's behaviour.

    ``scenario`` selects the waste arrival process and **defaults to ``"smooth"``**, which is
    Phase 2's process bit for bit (see :mod:`smartwaste.demand`). The default is deliberate: every
    Phase 2 caller and test omits the argument and must keep getting Phase 2's exact numbers.
    Phase 3's runner passes ``cfg.demand.scenario`` explicitly.

    The depot is the single central depot of the paper's routing model: the middle of the grid.

    Which bins are faulted, and which of their three sensors is stuck, is drawn from a dedicated
    stream so that sweeping ``stuck_bin_fraction`` changes only the faults -- not the bin
    locations, not the densities, not the waste arrivals. That makes the fault sweep a controlled
    comparison instead of three unrelated cities.

    The stuck sensor is chosen **uniformly** over left/centre/right. Phase 1 found the 3-sensor
    mean collapses when a *side* sensor sticks but merely ties when the *centre* one does, so
    hand-picking either would pre-decide the Phase 2 mean-vs-median result.
    """
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")
    if days < 1:
        raise ValueError(f"days must be >= 1, got {days}")
    if not 0 <= stuck_bin_fraction <= 1:
        raise ValueError(f"stuck_bin_fraction must be in [0, 1], got {stuck_bin_fraction}")

    grid = cfg.sim.grid_km
    layout_rng = random.Random(child_seed(seed, "layout"))
    locations = tuple(
        (layout_rng.uniform(0.0, grid), layout_rng.uniform(0.0, grid)) for _ in range(n_bins)
    )

    density_rng = random.Random(child_seed(seed, "density"))
    cv = cfg.bin.density_cv
    kg_per_pct = tuple(
        cfg.bin.density_kg_per_pct * density_rng.uniform(1.0 - cv, 1.0 + cv) for _ in range(n_bins)
    )

    fault_rng = random.Random(child_seed(seed, "faults"))
    stuck: list[int] = []
    for _ in range(n_bins):
        # Both draws happen for every bin regardless of the outcome, so a change in
        # stuck_bin_fraction cannot shift the stream for the bins that follow.
        faulted = fault_rng.random() < stuck_bin_fraction
        which = fault_rng.randrange(cfg.sensors.n_ultrasonic)
        stuck.append(which if faulted else NO_STUCK_SENSOR)

    # One stream per bin, so the arrivals of bin 7 never depend on how many bins precede it.
    # Both the smooth and heterogeneous processes live in smartwaste.demand.
    calendar = build_calendar(cfg, seed, n_bins=n_bins, days=days, scenario=scenario)
    increments = generate_increments(
        cfg, seed, n_bins=n_bins, days=days, scenario=scenario, calendar=calendar
    )

    return CityLayout(
        locations=locations,
        kg_per_pct=kg_per_pct,
        stuck_sensor=tuple(stuck),
        increments=increments,
        depot=(grid / 2.0, grid / 2.0),
        calendar=calendar,
    )


def build_sensor_arrays(cfg: Config, layout: CityLayout, seed: int) -> list[SensorArray]:
    """One sensor array per bin, seeded per bin so every strategy gets identical noise.

    Rebuilt for each strategy: the arrays are stateful (the stuck latch, and the previous reading
    each sensor may repeat), and that state must start clean for every strategy rather than
    carrying over from whichever strategy ran first.

    The noise parameters come straight from ``config/default.yaml`` -- nothing here is tuned per
    strategy. ``center_peak`` skew is used because the paper angles the inner walls to push waste
    towards the centre, which is the uneven-fill shape that geometry implies.
    """
    s = cfg.sensors
    arrays: list[SensorArray] = []
    for bin_index in range(layout.n_bins):
        params = NoiseParams(
            sigma_cm=s.noise_sigma_cm,
            bias_max_cm=s.bias_max_cm,
            skew_cm=s.uneven_fill_skew_cm,
            skew_mode=SKEW_CENTER_PEAK,
            stuck_sensor=layout.stuck_sensor[bin_index],
            stuck_probability=s.stuck_probability,
            load_cell_sigma_kg=s.load_cell_sigma_kg,
            load_cell_bias_max_kg=s.load_cell_bias_max_kg,
        )
        arrays.append(
            SensorArray(
                params,
                cfg.bin,
                cfg.sensors,
                random.Random(child_seed(seed, f"sensors:{bin_index}")),
            )
        )
    return arrays


@dataclass(frozen=True)
class DayState:
    """What one day of the city looks like, handed to a strategy to decide on."""

    day: int
    readings: tuple[SensorReading, ...]


@dataclass
class OutcomeCounters:
    """Consequence counters, accumulated as the simulation runs."""

    bin_visits: int = 0
    missed_pickups: int = 0
    overflow_events: int = 0
    overflow_bin_days: int = 0
    #: Bins that overflowed on at least one day. A count of bins, not of days or transitions.
    overflowed_bins: int = 0
    true_red_bin_days: int = 0


@dataclass
class ScoringWindow:
    """A second set of counters covering only days ``>= start_day``.

    Phase 3 needs this because the forecast-driven strategies spend days 0..13 on their backstop
    schedule while the model is still warming up. Scoring them over all 30 days would credit the
    forecaster with 14 days it had no part in, so every Phase 3 headline claim is made on the
    post-warm-up window instead -- and **every** strategy, including the calendars and the
    paper's fusion, is scored on that identical window.

    The full-horizon :class:`OutcomeCounters` are still accumulated unchanged alongside this, so
    Phase 2's numbers remain directly comparable.
    """

    start_day: int
    counters: OutcomeCounters = field(default_factory=OutcomeCounters)
    #: Window-local "has this bin overflowed yet", so ``overflowed_bins`` counts bins that
    #: overflowed *inside the window* rather than inheriting warm-up overflows.
    ever_overflowed: list[bool] = field(default_factory=list)

    def covers(self, day: int) -> bool:
        return day >= self.start_day


def step_day(
    cfg: Config,
    layout: CityLayout,
    weights: list[float],
    was_overflowing: list[bool],
    ever_overflowed: list[bool],
    day: int,
    *,
    window: ScoringWindow | None = None,
) -> tuple[list[float], OutcomeCounters]:
    """Advance every bin by one day's waste arrivals and measure overflow.

    Returns the day's true fill percentages and the overflow counters for that day only. Called
    *before* any collection, so "the true fill reached 100% before collection" is what is counted.

    ``window``, when given, accumulates the same measurements a second time for days inside the
    scoring window. It is keyword-only and optional so that every Phase 2 caller -- which passes
    six positional arguments and expects a 2-tuple back -- is completely unaffected.
    """
    counters = OutcomeCounters()
    fills: list[float] = []
    cap = cfg.bin.max_accumulated_weight_kg

    # Bound once per day rather than re-tested per bin: inside the window these are the window's
    # counters, outside it they are None and every window update is skipped.
    scored = window.counters if window is not None and window.covers(day) else None
    window_ever = window.ever_overflowed if window is not None else []

    for i in range(layout.n_bins):
        # Weight is capped: an abandoned bin must not accumulate without bound. The cap sits
        # above alpha * W_max, so the load cell can still flag it.
        weights[i] = min(cap, weights[i] + layout.increments[i][day])
        fill = min(FULL_PCT, weights[i] / layout.kg_per_pct[i])
        fills.append(fill)

        overflowing = fill >= FULL_PCT
        if overflowing:
            counters.overflow_bin_days += 1
            if not was_overflowing[i]:
                # A transition into overflow, not a day spent there: one event per episode.
                counters.overflow_events += 1
            if not ever_overflowed[i]:
                ever_overflowed[i] = True
                counters.overflowed_bins += 1

            if scored is not None:
                scored.overflow_bin_days += 1
                if not was_overflowing[i]:
                    scored.overflow_events += 1
                if not window_ever[i]:
                    window_ever[i] = True
                    scored.overflowed_bins += 1
        was_overflowing[i] = overflowing

        if truth_state(fill, weights[i], cfg.bin) is BinState.RED:
            counters.true_red_bin_days += 1
            if scored is not None:
                scored.true_red_bin_days += 1

    return fills, counters


@dataclass(frozen=True)
class StrategyRun:
    """Everything one strategy did over the whole horizon."""

    key: str
    label: str
    truck_days: tuple[TruckDay, ...]
    counters: OutcomeCounters
    route_cache_hits: int
    route_cache_misses: int
    #: Counters and truck-days restricted to the scoring window, or ``None`` when no window was
    #: requested. Phase 2 callers never ask for one, so they see exactly what they always did.
    window_counters: OutcomeCounters | None = None
    window_truck_days: tuple[TruckDay, ...] = ()
    window_start_day: int = 0


def run_strategy(
    cfg: Config,
    layout: CityLayout,
    clusters: ClusterResult,
    strategy: Strategy,
    seed: int,
    *,
    window_start_day: int | None = None,
) -> StrategyRun:
    """Simulate the whole horizon under one strategy, routing its choices each day.

    The loop per day is: waste arrives -> overflow is measured -> every bin is read -> the
    strategy chooses -> the chosen bins are routed per cluster -> the chosen bins are emptied.

    Every bin is read **even when the strategy ignores readings** ((a) and (b) do), which costs a
    few seconds across the whole benchmark and buys the thing that makes the comparison valid:
    the sensor noise stream for a given bin is then identical under all five strategies, so (d)
    and (e) are scored on the same sensor errors and differ only in their decisions.
    """
    arrays = build_sensor_arrays(cfg, layout, seed)
    cache = RouteCache(layout.depot, layout.locations, cfg.routing.two_opt_max_iters)

    # Hand the strategy the city's non-secret structure once, before day 0. Area type, weekday
    # and festival flag are operational facts a real utility knows from its own records -- they
    # are NOT ground truth about fill, and the context carries no fill or weight at all.
    strategy.begin_run(RunContext(calendar=layout.calendar, bin_cfg=cfg.bin, n_bins=layout.n_bins))

    window: ScoringWindow | None = None
    if window_start_day is not None:
        window = ScoringWindow(
            start_day=window_start_day,
            ever_overflowed=[False] * layout.n_bins,
        )
    window_truck_days: list[TruckDay] = []

    # Cluster membership never changes, so invert the labels once rather than per day.
    members: list[list[int]] = [[] for _ in range(clusters.n_clusters)]
    for bin_index, label in enumerate(clusters.labels):
        members[label].append(bin_index)

    weights = [0.0] * layout.n_bins
    was_overflowing = [False] * layout.n_bins
    ever_overflowed = [False] * layout.n_bins
    totals = OutcomeCounters()
    truck_days: list[TruckDay] = []

    for day in range(layout.days):
        fills, day_counters = step_day(
            cfg, layout, weights, was_overflowing, ever_overflowed, day, window=window
        )
        totals.overflow_events += day_counters.overflow_events
        totals.overflow_bin_days += day_counters.overflow_bin_days
        totals.overflowed_bins += day_counters.overflowed_bins
        totals.true_red_bin_days += day_counters.true_red_bin_days

        readings = tuple(arrays[i].read(fills[i], weights[i]) for i in range(layout.n_bins))
        selected = strategy.select(day, readings)
        selected_set = set(selected)

        # A missed pickup is a bin that was genuinely RED today and was not chosen. Ground truth
        # is read here, by the *scorer*, never by the strategy.
        in_window = window is not None and window.covers(day)
        for i in range(layout.n_bins):
            if i in selected_set:
                continue
            if truth_state(fills[i], weights[i], cfg.bin) is BinState.RED:
                totals.missed_pickups += 1
                if in_window and window is not None:
                    window.counters.missed_pickups += 1

        totals.bin_visits += len(selected)
        if in_window and window is not None:
            window.counters.bin_visits += len(selected)

        # One truck per cluster. A cluster with nothing selected still produces a TruckDay, with
        # zero bins, so the fixed-cost bases can tell a parked truck from a rolling one.
        for cluster_id in range(clusters.n_clusters):
            cluster_selected = [i for i in members[cluster_id] if i in selected_set]
            distance_km = cache.distance_for(cluster_id, cluster_selected)
            truck_day = TruckDay(distance_km=distance_km, bins_visited=len(cluster_selected))
            truck_days.append(truck_day)
            if in_window:
                window_truck_days.append(truck_day)

        # Tell the strategy what it actually collected, so a forecaster can reset that bin's
        # history. It is told its OWN decisions only -- never the true fill it just emptied.
        strategy.observe_collection(day, selected)

        for i in selected:
            weights[i] = 0.0
            # Emptied, so it is no longer overflowing; the next episode counts as a new event.
            was_overflowing[i] = False

    return StrategyRun(
        key=strategy.key,
        label=strategy.label,
        truck_days=tuple(truck_days),
        counters=totals,
        route_cache_hits=cache.hits,
        route_cache_misses=cache.misses,
        window_counters=window.counters if window is not None else None,
        window_truck_days=tuple(window_truck_days),
        window_start_day=0 if window is None else window.start_day,
    )
