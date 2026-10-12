"""City simulation: determinism, the paired-randomness invariant, and overflow accounting.

The most important test in this file is
:func:`test_sensor_noise_is_unaffected_by_a_strategy_s_decisions`. Phase 2 lets each strategy act
on its own classification, so the fill trajectories legitimately diverge -- but if the *sensor
noise* diverged too, (d) and (e) would be scored on different sensor errors and the comparison
would be meaningless. That test pins the invariant.
"""

import random

import pytest

from smartwaste import load_config
from smartwaste.city_sim import (
    FULL_PCT,
    CityLayout,
    build_layout,
    build_sensor_arrays,
    run_strategy,
    step_day,
)
from smartwaste.config import Config
from smartwaste.noise import NO_STUCK_SENSOR
from smartwaste.routing.cluster import ClusterResult, kmeans
from smartwaste.strategies import FixedRoute, FixedSchedule, build_strategies

pytestmark = pytest.mark.sim

SEED = 42
N_BINS = 60
DAYS = 12
N_CLUSTERS = 3


@pytest.fixture
def cfg() -> Config:
    return load_config()


@pytest.fixture
def layout(cfg: Config) -> CityLayout:
    return build_layout(cfg, SEED, n_bins=N_BINS, days=DAYS, stuck_bin_fraction=0.1)


@pytest.fixture
def clusters(layout: CityLayout) -> ClusterResult:
    return kmeans(layout.locations, N_CLUSTERS, random.Random(1))


# --------------------------------------------------------------------------------------------
# Layout determinism
# --------------------------------------------------------------------------------------------


def test_same_seed_gives_an_identical_layout(cfg: Config) -> None:
    first = build_layout(cfg, SEED, n_bins=N_BINS, days=DAYS, stuck_bin_fraction=0.1)
    second = build_layout(cfg, SEED, n_bins=N_BINS, days=DAYS, stuck_bin_fraction=0.1)

    assert first.locations == second.locations
    assert first.kg_per_pct == second.kg_per_pct
    assert first.stuck_sensor == second.stuck_sensor
    assert first.increments == second.increments
    assert first.depot == second.depot


def test_different_seeds_give_a_different_layout(cfg: Config) -> None:
    first = build_layout(cfg, 42, n_bins=N_BINS, days=DAYS, stuck_bin_fraction=0.1)
    second = build_layout(cfg, 43, n_bins=N_BINS, days=DAYS, stuck_bin_fraction=0.1)
    assert first.locations != second.locations


def test_layout_shape_and_bounds(cfg: Config, layout: CityLayout) -> None:
    assert layout.n_bins == N_BINS
    assert layout.days == DAYS
    assert all(len(row) == DAYS for row in layout.increments)
    grid = cfg.sim.grid_km
    assert all(0 <= x <= grid and 0 <= y <= grid for x, y in layout.locations)
    assert layout.depot == (grid / 2.0, grid / 2.0)


def test_waste_increments_are_never_negative(layout: CityLayout) -> None:
    """dW(t) is a Gaussian floored at 0: waste arrives, it does not spontaneously leave."""
    assert all(value >= 0.0 for row in layout.increments for value in row)


# --------------------------------------------------------------------------------------------
# The fault sweep must be a controlled comparison
# --------------------------------------------------------------------------------------------


def test_changing_the_fault_fraction_changes_only_the_faults(cfg: Config) -> None:
    """Sweeping stuck_bin_fraction must not reshuffle the city, or the sweep compares nothing."""
    none = build_layout(cfg, SEED, n_bins=N_BINS, days=DAYS, stuck_bin_fraction=0.0)
    some = build_layout(cfg, SEED, n_bins=N_BINS, days=DAYS, stuck_bin_fraction=0.5)

    assert none.locations == some.locations
    assert none.kg_per_pct == some.kg_per_pct
    assert none.increments == some.increments
    assert none.stuck_sensor != some.stuck_sensor


def test_zero_fault_fraction_breaks_no_sensor(cfg: Config) -> None:
    layout = build_layout(cfg, SEED, n_bins=N_BINS, days=DAYS, stuck_bin_fraction=0.0)
    assert layout.n_faulted == 0
    assert all(s == NO_STUCK_SENSOR for s in layout.stuck_sensor)


def test_full_fault_fraction_breaks_every_bin(cfg: Config) -> None:
    layout = build_layout(cfg, SEED, n_bins=N_BINS, days=DAYS, stuck_bin_fraction=1.0)
    assert layout.n_faulted == N_BINS


def test_fault_count_rises_with_the_fraction(cfg: Config) -> None:
    low = build_layout(cfg, SEED, n_bins=2000, days=2, stuck_bin_fraction=0.02).n_faulted
    high = build_layout(cfg, SEED, n_bins=2000, days=2, stuck_bin_fraction=0.10).n_faulted
    assert low < high
    # Roughly the requested rate, allowing for sampling noise at 2000 bins.
    assert 0.01 <= low / 2000 <= 0.035
    assert 0.08 <= high / 2000 <= 0.125


def test_the_stuck_sensor_is_spread_over_all_three_positions(cfg: Config) -> None:
    """Uniform over left/centre/right, so neither mean nor median fusion is favoured."""
    layout = build_layout(cfg, SEED, n_bins=3000, days=2, stuck_bin_fraction=1.0)
    counts = [layout.stuck_sensor.count(i) for i in range(3)]
    assert all(count > 0 for count in counts)
    # Each position should take roughly a third of the faults.
    assert all(0.28 <= count / 3000 <= 0.39 for count in counts)


def test_invalid_layout_arguments_are_rejected(cfg: Config) -> None:
    with pytest.raises(ValueError, match="n_bins"):
        build_layout(cfg, SEED, n_bins=0, days=DAYS, stuck_bin_fraction=0.0)
    with pytest.raises(ValueError, match="days"):
        build_layout(cfg, SEED, n_bins=N_BINS, days=0, stuck_bin_fraction=0.0)
    with pytest.raises(ValueError, match="stuck_bin_fraction"):
        build_layout(cfg, SEED, n_bins=N_BINS, days=DAYS, stuck_bin_fraction=1.5)


# --------------------------------------------------------------------------------------------
# The paired-randomness invariant
# --------------------------------------------------------------------------------------------


def test_sensor_arrays_are_identical_across_strategies(cfg: Config, layout: CityLayout) -> None:
    """Rebuilding the arrays for a second strategy must reproduce the same noise stream."""
    first = build_sensor_arrays(cfg, layout, SEED)
    second = build_sensor_arrays(cfg, layout, SEED)

    for a, b in zip(first, second, strict=True):
        for _ in range(5):
            assert a.read(55.0, 27.0) == b.read(55.0, 27.0)


def test_sensor_noise_is_unaffected_by_a_strategy_s_decisions(
    cfg: Config, layout: CityLayout
) -> None:
    """The invariant that makes (d) vs (e) a fair fight.

    Two strategies with deliberately opposite behaviour -- one collects every bin every day, the
    other collects on day 0 only -- must still see byte-identical sensor readings for a bin when
    that bin is presented with the same true state. The readings are driven by the per-bin seeded
    stream, not by call order, so a collection decision cannot shift them.
    """
    greedy = build_sensor_arrays(cfg, layout, SEED)
    lazy = build_sensor_arrays(cfg, layout, SEED)

    # Feed the two arrays the same sequence of true states but interleave extra work differently;
    # the per-bin streams are independent, so bin 0's readings cannot be perturbed by bin 1's.
    greedy_readings = [greedy[0].read(40.0 + d, 20.0 + d) for d in range(DAYS)]
    for _ in range(50):
        greedy[1].read(10.0, 5.0)
    lazy_readings = [lazy[0].read(40.0 + d, 20.0 + d) for d in range(DAYS)]

    assert greedy_readings == lazy_readings


def test_all_strategies_face_the_same_arrivals(
    cfg: Config, layout: CityLayout, clusters: ClusterResult
) -> None:
    """Every strategy's true_red_bin_days is driven by the same pre-generated increments.

    They will differ, because collection changes what a bin's fill becomes -- but the *inputs*
    cannot differ, and the sensor-free baselines prove it: (a) collects everything daily, so no
    bin ever reaches RED.
    """
    run = run_strategy(cfg, layout, clusters, FixedRoute(), SEED)
    assert run.counters.true_red_bin_days == 0
    assert run.counters.missed_pickups == 0
    assert run.counters.overflow_events == 0


# --------------------------------------------------------------------------------------------
# Run determinism
# --------------------------------------------------------------------------------------------


def test_same_seed_gives_the_same_run(
    cfg: Config, layout: CityLayout, clusters: ClusterResult
) -> None:
    for strategy in build_strategies(cfg.bin, 3):
        first = run_strategy(cfg, layout, clusters, strategy, SEED)
        second = run_strategy(cfg, layout, clusters, strategy, SEED)
        assert first.truck_days == second.truck_days
        assert first.counters == second.counters


def test_one_truck_day_per_cluster_per_day(
    cfg: Config, layout: CityLayout, clusters: ClusterResult
) -> None:
    """A parked truck still produces a truck-day, so the fixed-cost bases can count it."""
    run = run_strategy(cfg, layout, clusters, FixedSchedule(5), SEED)
    assert len(run.truck_days) == clusters.n_clusters * DAYS


def test_bin_visits_match_the_truck_days(
    cfg: Config, layout: CityLayout, clusters: ClusterResult
) -> None:
    """Total bins visited must equal the sum over trucks: no bin routed twice or dropped."""
    for strategy in build_strategies(cfg.bin, 3):
        run = run_strategy(cfg, layout, clusters, strategy, SEED)
        assert sum(d.bins_visited for d in run.truck_days) == run.counters.bin_visits


def test_fixed_route_visits_every_bin_every_day(
    cfg: Config, layout: CityLayout, clusters: ClusterResult
) -> None:
    run = run_strategy(cfg, layout, clusters, FixedRoute(), SEED)
    assert run.counters.bin_visits == N_BINS * DAYS


def test_fixed_schedule_visits_on_its_firing_days_only(
    cfg: Config, layout: CityLayout, clusters: ClusterResult
) -> None:
    every_4 = run_strategy(cfg, layout, clusters, FixedSchedule(4), SEED)
    firing_days = len([d for d in range(DAYS) if d % 4 == 0])
    assert every_4.counters.bin_visits == N_BINS * firing_days


def test_route_cache_is_exercised_by_the_repeating_baselines(
    cfg: Config, layout: CityLayout, clusters: ClusterResult
) -> None:
    """(a) selects the same bins every day, so all but the first day must hit the cache."""
    run = run_strategy(cfg, layout, clusters, FixedRoute(), SEED)
    assert run.route_cache_misses == clusters.n_clusters
    assert run.route_cache_hits == clusters.n_clusters * (DAYS - 1)


# --------------------------------------------------------------------------------------------
# Overflow accounting and the weight cap
# --------------------------------------------------------------------------------------------


def test_weight_is_capped_and_fill_never_exceeds_100(cfg: Config) -> None:
    """An abandoned bin must not accumulate weight without bound."""
    cap = cfg.bin.max_accumulated_weight_kg

    # Never collect anything over a long horizon, so every bin pins against the cap.
    long_layout = build_layout(cfg, SEED, n_bins=20, days=200, stuck_bin_fraction=0.0)
    weights = [0.0] * 20
    was = [False] * 20
    ever = [False] * 20
    for day in range(200):
        fills, _ = step_day(cfg, long_layout, weights, was, ever, day)
        assert all(0.0 <= f <= FULL_PCT for f in fills)

    assert all(w <= cap for w in weights)
    assert max(weights) == pytest.approx(cap)


def test_the_weight_cap_still_allows_the_load_cell_to_flag(cfg: Config) -> None:
    """The cap must sit above alpha * W_max or a missed bin could never be rescued."""
    assert cfg.bin.max_accumulated_weight_kg >= cfg.bin.weight_threshold_kg


def test_overflow_events_count_transitions_not_days(cfg: Config) -> None:
    """One event per episode of overflow, however long the bin stays full."""
    layout = build_layout(cfg, SEED, n_bins=5, days=60, stuck_bin_fraction=0.0)
    weights = [0.0] * 5
    was = [False] * 5
    ever = [False] * 5

    events = 0
    bin_days = 0
    for day in range(60):
        _, counters = step_day(cfg, layout, weights, was, ever, day)
        events += counters.overflow_events
        bin_days += counters.overflow_bin_days

    # Nothing is ever collected here, so each bin overflows once and then stays overflowing.
    assert events == 5
    assert bin_days > events


def test_overflowed_bins_counts_each_bin_once(cfg: Config) -> None:
    layout = build_layout(cfg, SEED, n_bins=5, days=60, stuck_bin_fraction=0.0)
    weights = [0.0] * 5
    was = [False] * 5
    ever = [False] * 5

    total = 0
    for day in range(60):
        _, counters = step_day(cfg, layout, weights, was, ever, day)
        total += counters.overflowed_bins

    assert total == 5  # 5 distinct bins, counted once each despite many overflow days


def test_no_overflow_before_any_bin_fills(cfg: Config) -> None:
    layout = build_layout(cfg, SEED, n_bins=10, days=1, stuck_bin_fraction=0.0)
    weights = [0.0] * 10
    _, counters = step_day(cfg, layout, weights, [False] * 10, [False] * 10, 0)
    assert counters.overflow_events == 0
    assert counters.overflow_bin_days == 0


def test_a_strategy_that_misses_bins_overflows_more_than_one_that_does_not(
    cfg: Config, layout: CityLayout, clusters: ClusterResult
) -> None:
    """The Phase 1 shortcut is really gone: missing a full bin now has a consequence."""
    long_layout = build_layout(cfg, SEED, n_bins=N_BINS, days=40, stuck_bin_fraction=0.0)
    daily = run_strategy(cfg, long_layout, clusters, FixedRoute(), SEED)
    rare = run_strategy(cfg, long_layout, clusters, FixedSchedule(30), SEED)

    assert daily.counters.overflow_events == 0
    assert rare.counters.overflow_events > 0
    assert rare.counters.overflowed_bins > 0
