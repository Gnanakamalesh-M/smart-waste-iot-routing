"""The Phase 3 demand model, and the guarantee that Phase 2 still reproduces.

The most important test in this file is
:func:`test_smooth_scenario_reproduces_the_phase2_numbers_exactly`. Phase 2's published tables
are only honest for as long as the smooth code path is untouched, so the numbers are pinned here
as literals captured from the pre-Phase-3 tree.
"""

import pytest

from smartwaste import load_config
from smartwaste.benchmark import run_city
from smartwaste.city_sim import build_layout
from smartwaste.config import Config, ConfigError, DemandConfig
from smartwaste.demand import (
    DAYS_PER_MONTH,
    DAYS_PER_WEEK,
    SCENARIO_HETEROGENEOUS,
    SCENARIO_SMOOTH,
    DemandCalendar,
    arrival_mean_kg,
    assign_area_types,
    build_calendar,
    day_of_week,
    expected_mean_daily_arrival,
    generate_increments,
    pick_festival_days,
)

pytestmark = pytest.mark.sim

SEED = 42


@pytest.fixture
def cfg() -> Config:
    return load_config()


def _hetero(
    cfg: Config, seed: int, *, n_bins: int, days: int
) -> tuple[DemandCalendar, tuple[tuple[float, ...], ...]]:
    """A heterogeneous calendar and its arrivals, built together for one city."""
    calendar = build_calendar(cfg, seed, n_bins=n_bins, days=days, scenario=SCENARIO_HETEROGENEOUS)
    increments = generate_increments(
        cfg,
        seed,
        n_bins=n_bins,
        days=days,
        scenario=SCENARIO_HETEROGENEOUS,
        calendar=calendar,
    )
    return calendar, increments


# --------------------------------------------------------------------------------------------
# The Phase 2 guarantee
# --------------------------------------------------------------------------------------------

#: Captured from the tree at commit 5a3758a (Phase 2 results), BEFORE any Phase 3 code existed,
#: with: run_city(cfg, 42, 0.02, n_bins=60, days=12, n_clusters=3).
#: (key, distance_km, variable_cost, bin_visits, missed, overflow_events, overflow_bin_days,
#:  overflowed_bins, true_red_bin_days)
PHASE2_GOLDEN = [
    ("a_fixed_route", 8065.168933, 397237.434806, 720, 0, 0, 0, 0, 0),
    ("b_fixed_schedule", 2688.389644, 132412.478269, 240, 0, 0, 0, 0, 0),
    ("c_single_ultrasonic", 2021.814786, 98451.92638, 55, 20, 0, 0, 0, 51),
    ("d_paper_fusion", 1971.630592, 96038.50218, 57, 2, 1, 1, 1, 57),
    ("e_median_fusion", 1958.998729, 95417.488435, 56, 8, 1, 1, 1, 61),
]


def test_smooth_scenario_reproduces_the_phase2_numbers_exactly(cfg: Config) -> None:
    """Phase 2's published tables must survive Phase 3 untouched.

    ``run_city`` is Phase 2's entry point and defaults to the smooth scenario. If this fails,
    ``docs/PHASE2_RESULTS.md`` has been silently invalidated -- fix the code, never this test.
    """
    results = run_city(cfg, SEED, 0.02, n_bins=60, days=12, n_clusters=3)
    actual = [
        (
            r.key,
            round(r.totals.distance_km, 6),
            round(r.totals.variable_cost, 6),
            r.bin_visits,
            r.missed_pickups,
            r.overflow_events,
            r.overflow_bin_days,
            r.overflowed_bins,
            r.true_red_bin_days,
        )
        for r in results
    ]
    assert actual == PHASE2_GOLDEN


def test_build_layout_defaults_to_the_smooth_scenario(cfg: Config) -> None:
    """Phase 2 callers omit the argument, so the default must be the Phase 2 process."""
    layout = build_layout(cfg, SEED, n_bins=5, days=4, stuck_bin_fraction=0.0)
    assert layout.scenario == SCENARIO_SMOOTH


def test_the_smooth_stream_is_independent_of_the_demand_config(cfg: Config) -> None:
    """Changing a ``demand`` constant must not move one smooth number.

    The heterogeneous process uses its own seed labels precisely so that the Phase 2 arrivals
    cannot be perturbed by a Phase 3 knob.
    """
    other = load_config(
        overrides={"demand": {"area_mean_kg": [99.0, 99.0, 99.0], "festival_multiplier": 9.0}}
    )
    first = build_layout(cfg, SEED, n_bins=20, days=10, stuck_bin_fraction=0.0)
    second = build_layout(other, SEED, n_bins=20, days=10, stuck_bin_fraction=0.0)
    assert first.increments == second.increments


# --------------------------------------------------------------------------------------------
# Volume matching: the heterogeneous city must not simply make more waste
# --------------------------------------------------------------------------------------------


def test_heterogeneous_mean_daily_arrival_matches_the_smooth_mean(cfg: Config) -> None:
    """MEASURED, not computed from the formula: the realised mean is within 5% of 4.0 kg.

    If the heterogeneous city made materially more waste than the smooth one, every Phase 3
    improvement would be partly an artefact of extra volume rather than of sensors or
    forecasting. The tolerance is 5% because the realised mean also depends on the per-bin area
    draw and on where the weekend falls in the horizon.
    """
    n_bins, days = 400, 700
    calendar = build_calendar(cfg, SEED, n_bins=n_bins, days=days, scenario=SCENARIO_HETEROGENEOUS)
    increments = generate_increments(
        cfg,
        SEED,
        n_bins=n_bins,
        days=days,
        scenario=SCENARIO_HETEROGENEOUS,
        calendar=calendar,
    )
    total = sum(sum(row) for row in increments)
    realised = total / (n_bins * days)
    target = cfg.sim.daily_fill_mean_kg

    assert realised == pytest.approx(target, rel=0.05), (
        f"heterogeneous demand averages {realised:.4f} kg/bin/day against the smooth "
        f"{target} kg: the two scenarios would no longer be comparable on volume"
    )


def test_the_analytic_mean_agrees_with_the_measured_one(cfg: Config) -> None:
    """The documented formula must match what the generator actually produces."""
    n_bins, days = 400, DAYS_PER_MONTH * 20
    calendar = build_calendar(cfg, SEED, n_bins=n_bins, days=days, scenario=SCENARIO_HETEROGENEOUS)
    increments = generate_increments(
        cfg,
        SEED,
        n_bins=n_bins,
        days=days,
        scenario=SCENARIO_HETEROGENEOUS,
        calendar=calendar,
    )
    realised = sum(sum(row) for row in increments) / (n_bins * days)
    assert realised == pytest.approx(expected_mean_daily_arrival(cfg.demand, days), rel=0.05)


# --------------------------------------------------------------------------------------------
# The heterogeneity actually exists
# --------------------------------------------------------------------------------------------


def test_heterogeneous_demand_is_more_variable_across_bins_than_smooth(cfg: Config) -> None:
    """The whole point of Phase 3: bins must differ from each other.

    Phase 2's null result came from every bin filling at nearly the same rate. If this spread is
    not materially larger than the smooth one, a forecaster still has nothing to learn.
    """
    n_bins, days = 300, 60
    per_bin = {}
    for scenario in (SCENARIO_SMOOTH, SCENARIO_HETEROGENEOUS):
        calendar = build_calendar(cfg, SEED, n_bins=n_bins, days=days, scenario=scenario)
        increments = generate_increments(
            cfg, SEED, n_bins=n_bins, days=days, scenario=scenario, calendar=calendar
        )
        means = [sum(row) / days for row in increments]
        overall = sum(means) / len(means)
        variance = sum((m - overall) ** 2 for m in means) / len(means)
        per_bin[scenario] = variance**0.5

    assert per_bin[SCENARIO_HETEROGENEOUS] > 3 * per_bin[SCENARIO_SMOOTH]


def test_weekends_receive_more_waste_than_midweek(cfg: Config) -> None:
    """The weekday cycle must be visible in the generated arrivals, not just in the config."""
    n_bins, days = 300, 70
    calendar = build_calendar(cfg, SEED, n_bins=n_bins, days=days, scenario=SCENARIO_HETEROGENEOUS)
    increments = generate_increments(
        cfg,
        SEED,
        n_bins=n_bins,
        days=days,
        scenario=SCENARIO_HETEROGENEOUS,
        calendar=calendar,
    )
    # Saturday is index 5 and has the largest multiplier; Tuesday (1) has the smallest.
    saturday = [
        increments[b][d]
        for b in range(n_bins)
        for d in range(days)
        if calendar.weekday[d] == 5 and not calendar.is_festival[d]
    ]
    tuesday = [
        increments[b][d]
        for b in range(n_bins)
        for d in range(days)
        if calendar.weekday[d] == 1 and not calendar.is_festival[d]
    ]
    assert sum(saturday) / len(saturday) > sum(tuesday) / len(tuesday)


def test_festival_days_receive_more_waste(cfg: Config) -> None:
    n_bins, days = 300, 90
    calendar = build_calendar(cfg, SEED, n_bins=n_bins, days=days, scenario=SCENARIO_HETEROGENEOUS)
    increments = generate_increments(
        cfg,
        SEED,
        n_bins=n_bins,
        days=days,
        scenario=SCENARIO_HETEROGENEOUS,
        calendar=calendar,
    )
    festival = [
        increments[b][d] for b in range(n_bins) for d in range(days) if calendar.is_festival[d]
    ]
    normal = [
        increments[b][d] for b in range(n_bins) for d in range(days) if not calendar.is_festival[d]
    ]
    assert festival, "the horizon contained no festival day, so this test proved nothing"
    assert sum(festival) / len(festival) > sum(normal) / len(normal)


def test_area_types_differ_in_mean_arrival(cfg: Config) -> None:
    """A market bin must actually fill faster than a residential one."""
    n_bins, days = 600, 60
    calendar = build_calendar(cfg, SEED, n_bins=n_bins, days=days, scenario=SCENARIO_HETEROGENEOUS)
    increments = generate_increments(
        cfg,
        SEED,
        n_bins=n_bins,
        days=days,
        scenario=SCENARIO_HETEROGENEOUS,
        calendar=calendar,
    )
    by_type: dict[int, list[float]] = {}
    for bin_index, row in enumerate(increments):
        by_type.setdefault(calendar.area_type[bin_index], []).append(sum(row) / days)

    means = {area: sum(values) / len(values) for area, values in by_type.items()}
    # Area 0 (market) has the largest configured mean, area 1 (residential) the smallest.
    assert means[0] > means[2] > means[1]


# --------------------------------------------------------------------------------------------
# Determinism and the seed-stream discipline
# --------------------------------------------------------------------------------------------


def test_same_seed_gives_identical_heterogeneous_arrivals(cfg: Config) -> None:
    first_cal, first_inc = _hetero(cfg, SEED, n_bins=30, days=20)
    second_cal, second_inc = _hetero(cfg, SEED, n_bins=30, days=20)
    assert first_cal == second_cal
    assert first_inc == second_inc


def test_different_seeds_change_the_heterogeneous_arrivals(cfg: Config) -> None:
    _, first = _hetero(cfg, 42, n_bins=30, days=20)
    _, second = _hetero(cfg, 43, n_bins=30, days=20)
    assert first != second


def test_arrivals_are_never_negative(cfg: Config) -> None:
    _, increments = _hetero(cfg, SEED, n_bins=50, days=30)
    assert all(value >= 0.0 for row in increments for value in row)


def test_festival_count_follows_the_configured_rate(cfg: Config) -> None:
    flags = pick_festival_days(cfg.demand, SEED, DAYS_PER_MONTH)
    assert sum(flags) == round(cfg.demand.festival_days_per_month)


def test_smooth_calendar_is_neutral(cfg: Config) -> None:
    """One area type, no festivals: the smooth city has no structure to leak into Phase 2."""
    calendar = build_calendar(cfg, SEED, n_bins=10, days=30, scenario=SCENARIO_SMOOTH)
    assert set(calendar.area_type) == {0}
    assert calendar.n_festival_days == 0


def test_day_of_week_cycles_weekly() -> None:
    assert day_of_week(0) == 0
    assert day_of_week(DAYS_PER_WEEK) == 0
    assert [day_of_week(d) for d in range(8)] == [0, 1, 2, 3, 4, 5, 6, 0]


def test_area_types_are_drawn_from_the_configured_weights(cfg: Config) -> None:
    types = assign_area_types(cfg.demand, SEED, 4000)
    share_residential = types.count(1) / len(types)
    # Residential is weighted 0.5; a 4000-bin draw should land close.
    assert share_residential == pytest.approx(cfg.demand.area_type_weights[1], abs=0.05)


def test_arrival_mean_applies_every_multiplier(cfg: Config) -> None:
    demand = cfg.demand
    plain = arrival_mean_kg(demand, 0, 0, False)
    festival = arrival_mean_kg(demand, 0, 0, True)
    assert festival == pytest.approx(plain * demand.festival_multiplier)
    assert arrival_mean_kg(demand, 0, 5, False) == pytest.approx(
        demand.area_mean_kg[0] * demand.weekday_multipliers[5]
    )


def test_unknown_scenario_is_rejected(cfg: Config) -> None:
    with pytest.raises(ValueError, match="scenario must be one of"):
        build_calendar(cfg, SEED, n_bins=5, days=5, scenario="made_up")


# --------------------------------------------------------------------------------------------
# Config validation
# --------------------------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"scenario": "nope"}, "demand.scenario must be one of"),
        ({"area_mean_kg": [1.0, 2.0]}, "same length"),
        ({"area_type_weights": [0.5, 0.2, 0.2]}, "must sum to 1.0"),
        ({"cv": -0.1}, "demand.cv must be >= 0"),
        ({"weekday_multipliers": [1.0, 1.0]}, "exactly 7 entries"),
        ({"festival_multiplier": -1.0}, "festival_multiplier must be >= 0"),
        ({"festival_days_per_month": -1.0}, "festival_days_per_month must be >= 0"),
    ],
)
def test_invalid_demand_config_is_rejected(override: dict[str, object], message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        load_config(overrides={"demand": override})


@pytest.mark.unit
def test_demand_config_is_frozen(cfg: Config) -> None:
    """Config sections are frozen so a run cannot quietly retune itself mid-flight."""
    with pytest.raises(Exception, match="cannot assign|frozen"):
        cfg.demand.cv = 0.9  # type: ignore[misc]


@pytest.mark.unit
def test_demand_config_defaults_are_the_documented_ones(cfg: Config) -> None:
    """Pins the invented constants, so a casual edit to a published number is caught.

    These are the values ``docs/PHASE3_RESULTS.md`` and ``docs/ASSUMPTIONS.md`` describe. The
    test is a tripwire against silent retuning, not a claim that the values are realistic.
    """
    demand: DemandConfig = cfg.demand
    assert demand.area_type_weights == (0.2, 0.5, 0.3)
    assert demand.area_mean_kg == (6.0, 2.0, 3.4)
    assert demand.cv == 0.35
    assert demand.weekday_multipliers == (1.0, 1.0, 1.0, 1.0, 1.15, 1.4, 1.25)
    assert demand.festival_days_per_month == 2
    assert demand.festival_multiplier == 2.5
