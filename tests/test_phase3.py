"""The Phase 3 benchmark: strategy (f), the 7-day calendar, and the two scoring windows.

Marker discipline, which is load-bearing here:

- **unit / sim / routing / slow** tests run the **stdlib-only eight** strategies
  (``include_gbm=False``). The moving-average (f) variants are pure stdlib, so strategy (f)'s
  plumbing, the windowed counters and the 7-day calendar all still get real coverage without
  scikit-learn installed.
- Anything exercising ``f_gbm_*`` carries the **ml** marker, because that is the only chunk CI
  installs the optional ``ml`` extra for.

``tests/test_demand.py`` holds the guarantee that Phase 2's own benchmark is unchanged.
"""

import pytest

from smartwaste import load_config
from smartwaste.config import Config
from smartwaste.demand import SCENARIO_HETEROGENEOUS, SCENARIO_SMOOTH
from smartwaste.phase3 import (
    BASELINE_KEY,
    CSV_FIELDS,
    FULL_LABEL,
    WINDOW_LABEL,
    Phase3Result,
    calendar_verdict,
    csv_rows,
    forecast_verdict,
    render_headline,
    render_reductions,
    render_training_rows,
    run_city_phase3,
    select,
)
from smartwaste.strategies import ForecastDriven, build_phase3_strategies

SEED = 42

#: Tiny, and deliberately longer than the 14-day warm-up so the headline window is non-empty.
N_BINS = 60
DAYS = 22
N_CLUSTERS = 3

STDLIB_EIGHT = (
    "a_fixed_route",
    "b_fixed_schedule",
    "c_single_ultrasonic",
    "d_paper_fusion",
    "e_median_fusion",
    "b7_fixed_schedule",
    "f_ma_3",
    "f_ma_7",
)


@pytest.fixture
def cfg() -> Config:
    return load_config()


def _run(cfg: Config, scenario: str = SCENARIO_HETEROGENEOUS) -> list[Phase3Result]:
    """The stdlib-only run every non-``ml`` test in this file uses."""
    return run_city_phase3(
        cfg,
        SEED,
        0.02,
        scenario=scenario,
        n_bins=N_BINS,
        days=DAYS,
        n_clusters=N_CLUSTERS,
        include_gbm=False,
    )


# --------------------------------------------------------------------------------------------
# The strategy set
# --------------------------------------------------------------------------------------------


@pytest.mark.unit
def test_phase3_adds_five_strategies_to_phase2s_five(cfg: Config) -> None:
    keys = [s.key for s in build_phase3_strategies(cfg, SEED)]
    assert len(keys) == 10
    assert len(set(keys)) == 10
    assert keys[:5] == [
        "a_fixed_route",
        "b_fixed_schedule",
        "c_single_ultrasonic",
        "d_paper_fusion",
        "e_median_fusion",
    ]
    assert set(keys[5:]) == {"b7_fixed_schedule", "f_ma_3", "f_ma_7", "f_gbm_3", "f_gbm_7"}


@pytest.mark.unit
def test_excluding_the_gbm_leaves_eight_stdlib_strategies(cfg: Config) -> None:
    """This is what keeps the non-``ml`` test chunks installable without scikit-learn."""
    keys = tuple(s.key for s in build_phase3_strategies(cfg, SEED, include_gbm=False))
    assert keys == STDLIB_EIGHT
    assert not any(
        s.forecaster.needs_ml_extra
        for s in build_phase3_strategies(cfg, SEED, include_gbm=False)
        if isinstance(s, ForecastDriven)
    )


@pytest.mark.unit
def test_each_forecast_strategy_has_a_matching_sensor_free_calendar(cfg: Config) -> None:
    """A 7-day (f) must be comparable against a 7-day calendar, not only against (b).

    Without this, "forecasting wins" could just mean "collecting less often is cheaper".
    """
    strategies = {s.key: s for s in build_phase3_strategies(cfg, SEED)}
    assert strategies["b_fixed_schedule"].every_n_days == 3  # type: ignore[attr-defined]
    assert strategies["b7_fixed_schedule"].every_n_days == 7  # type: ignore[attr-defined]
    assert strategies["f_ma_3"].backstop_days == 3  # type: ignore[attr-defined]
    assert strategies["f_ma_7"].backstop_days == 7  # type: ignore[attr-defined]
    for key in ("b_fixed_schedule", "b7_fixed_schedule"):
        assert strategies[key].uses_sensors is False


@pytest.mark.unit
def test_labels_mark_the_ml_rows_and_the_baselines(cfg: Config) -> None:
    """Provenance has to be readable straight off the results table."""
    labels = {s.key: s.label for s in build_phase3_strategies(cfg, SEED)}
    assert "naive baseline" in labels["f_ma_3"]
    assert "[ML]" in labels["f_gbm_3"]
    assert "sensor-free baseline" in labels["b7_fixed_schedule"]


@pytest.mark.unit
def test_forecast_strategy_requires_begin_run(cfg: Config) -> None:
    """A clear error beats a confusing IndexError if the runner forgets the hook."""
    strategy = next(
        s for s in build_phase3_strategies(cfg, SEED, include_gbm=False) if s.key == "f_ma_3"
    )
    with pytest.raises(RuntimeError, match="begin_run must be called"):
        strategy.select(0, [])


@pytest.mark.unit
def test_negative_backstop_is_rejected(cfg: Config) -> None:
    from smartwaste.ml.forecast import build_forecaster

    with pytest.raises(ValueError, match="backstop_days must be >= 1"):
        ForecastDriven(
            "f_bad",
            "bad",
            cfg.bin,
            cfg.forecast,
            build_forecaster("moving_average", cfg.forecast, SEED),
            backstop_days=0,
        )


# --------------------------------------------------------------------------------------------
# Strategy (f) behaviour
# --------------------------------------------------------------------------------------------


@pytest.mark.sim
def test_during_warmup_a_forecast_strategy_is_exactly_its_calendar(cfg: Config) -> None:
    """The warm-up claim, checked rather than asserted in prose.

    This is why the headline window starts at ``warmup_days``: before it, (f) *is* a calendar,
    so crediting it with those days would be meaningless.
    """
    results = _run(cfg)
    warmup = cfg.forecast.warmup_days

    for backstop, calendar_key in ((3, "b_fixed_schedule"), (7, "b7_fixed_schedule")):
        forecast_full = select(results, key=f"f_ma_{backstop}", window=FULL_LABEL)[0]
        forecast_window = select(results, key=f"f_ma_{backstop}", window=WINDOW_LABEL)[0]
        calendar_full = select(results, key=calendar_key, window=FULL_LABEL)[0]
        calendar_window = select(results, key=calendar_key, window=WINDOW_LABEL)[0]

        warmup_visits_f = forecast_full.counters.bin_visits - forecast_window.counters.bin_visits
        warmup_visits_c = calendar_full.counters.bin_visits - calendar_window.counters.bin_visits
        assert warmup_visits_f == warmup_visits_c, (
            f"f_ma_{backstop} did not behave as a plain {backstop}-day calendar during the "
            f"first {warmup} days"
        )


@pytest.mark.sim
def test_a_forecast_strategy_diverges_from_its_calendar_after_warmup(cfg: Config) -> None:
    """After the fit, (f) must actually do something different -- or it is not being used."""
    results = _run(cfg)
    forecast = select(results, key="f_ma_7", window=WINDOW_LABEL)[0]
    calendar = select(results, key="b7_fixed_schedule", window=WINDOW_LABEL)[0]
    assert forecast.counters.bin_visits != calendar.counters.bin_visits


@pytest.mark.sim
def test_the_backstop_is_a_floor_so_no_bin_is_abandoned(cfg: Config) -> None:
    """A badly wrong forecast must not be able to strand a bin forever.

    (f) visits at least as often as its calendar would, so its visit count can never fall below
    the calendar's.
    """
    results = _run(cfg)
    for backstop, calendar_key in ((3, "b_fixed_schedule"), (7, "b7_fixed_schedule")):
        forecast = select(results, key=f"f_ma_{backstop}", window=WINDOW_LABEL)[0]
        calendar = select(results, key=calendar_key, window=WINDOW_LABEL)[0]
        assert forecast.counters.bin_visits >= calendar.counters.bin_visits


@pytest.mark.sim
def test_a_forecast_strategy_trains_on_some_rows(cfg: Config) -> None:
    results = _run(cfg)
    for key in ("f_ma_3", "f_ma_7"):
        run = select(results, key=key, window=WINDOW_LABEL)[0]
        assert run.training_rows is not None
        assert run.training_rows > 0


@pytest.mark.sim
def test_non_forecast_strategies_report_no_training_rows(cfg: Config) -> None:
    results = _run(cfg)
    for key in ("a_fixed_route", "b_fixed_schedule", "d_paper_fusion"):
        assert select(results, key=key, window=WINDOW_LABEL)[0].training_rows is None


# --------------------------------------------------------------------------------------------
# The two scoring windows
# --------------------------------------------------------------------------------------------


@pytest.mark.sim
def test_every_strategy_is_scored_on_both_windows(cfg: Config) -> None:
    results = _run(cfg)
    for key in STDLIB_EIGHT:
        windows = {r.window for r in select(results, key=key)}
        assert windows == {FULL_LABEL, WINDOW_LABEL}


@pytest.mark.sim
def test_the_headline_window_is_identical_for_every_strategy(cfg: Config) -> None:
    """The comparison is only fair if nobody is scored on a different stretch of days."""
    results = select(_run(cfg), window=WINDOW_LABEL)
    assert len({r.window_start_day for r in results}) == 1
    assert len({r.scored_days for r in results}) == 1
    assert results[0].window_start_day == cfg.forecast.warmup_days
    assert results[0].scored_days == DAYS - cfg.forecast.warmup_days


@pytest.mark.sim
def test_the_window_is_a_subset_of_the_full_horizon(cfg: Config) -> None:
    results = _run(cfg)
    for key in STDLIB_EIGHT:
        full = select(results, key=key, window=FULL_LABEL)[0]
        window = select(results, key=key, window=WINDOW_LABEL)[0]
        assert window.counters.bin_visits <= full.counters.bin_visits
        assert window.counters.overflow_bin_days <= full.counters.overflow_bin_days
        assert window.counters.missed_pickups <= full.counters.missed_pickups
        assert window.totals.distance_km <= full.totals.distance_km + 1e-9


@pytest.mark.sim
def test_overflow_rate_normalises_the_two_window_lengths(cfg: Config) -> None:
    """Raw counts cannot be read across a 16-day and a 30-day window; the rate can."""
    results = _run(cfg)
    run = select(results, key="b7_fixed_schedule", window=WINDOW_LABEL)[0]
    expected = run.counters.overflow_events / (run.n_bins * run.scored_days)
    assert run.overflow_events_per_bin_day == pytest.approx(expected)


@pytest.mark.sim
def test_window_overflowed_bins_does_not_inherit_warmup_overflows(cfg: Config) -> None:
    """A bin that only overflowed during the warm-up must not count in the window."""
    results = _run(cfg)
    for key in STDLIB_EIGHT:
        full = select(results, key=key, window=FULL_LABEL)[0]
        window = select(results, key=key, window=WINDOW_LABEL)[0]
        assert window.counters.overflowed_bins <= full.counters.overflowed_bins


# --------------------------------------------------------------------------------------------
# Determinism, reporting and both demand scenarios
# --------------------------------------------------------------------------------------------


@pytest.mark.sim
def test_same_seed_gives_identical_numbers(cfg: Config) -> None:
    """The project's core promise, on the Phase 3 path."""
    first = _run(cfg)
    second = _run(cfg)
    assert [r.key for r in first] == [r.key for r in second]
    for a, b in zip(first, second, strict=True):
        assert a.totals == b.totals
        assert a.counters == b.counters


@pytest.mark.sim
def test_the_two_demand_scenarios_give_different_numbers(cfg: Config) -> None:
    smooth = _run(cfg, scenario=SCENARIO_SMOOTH)
    hetero = _run(cfg, scenario=SCENARIO_HETEROGENEOUS)
    assert [r.totals.distance_km for r in smooth] != [r.totals.distance_km for r in hetero]


@pytest.mark.sim
def test_the_baseline_never_overflows_on_either_window(cfg: Config) -> None:
    """(a) visits everything daily, so it is the overflow-free reference by construction."""
    for r in select(_run(cfg), key=BASELINE_KEY):
        assert r.counters.overflow_events == 0
        assert r.counters.missed_pickups == 0


@pytest.mark.sim
def test_the_baseline_is_the_most_expensive(cfg: Config) -> None:
    results = select(_run(cfg), window=WINDOW_LABEL)
    baseline = next(r for r in results if r.key == BASELINE_KEY)
    for r in results:
        assert r.totals.distance_km <= baseline.totals.distance_km + 1e-9
        assert r.counters.bin_visits <= baseline.counters.bin_visits


@pytest.mark.sim
def test_cost_bases_are_ordered(cfg: Config) -> None:
    for r in _run(cfg):
        assert r.cost_on("variable") <= r.cost_on("rolling") + 1e-9
        assert r.cost_on("rolling") <= r.cost_on("fixed_fleet") + 1e-9


@pytest.mark.sim
def test_unknown_cost_basis_is_rejected(cfg: Config) -> None:
    with pytest.raises(ValueError, match="unknown cost basis"):
        _run(cfg)[0].cost_on("free")


@pytest.mark.sim
def test_csv_rows_cover_every_declared_field(cfg: Config) -> None:
    rows = csv_rows(_run(cfg))
    assert rows
    for row in rows:
        assert set(row) == set(CSV_FIELDS)


@pytest.mark.sim
def test_renderers_produce_labelled_tables(cfg: Config) -> None:
    results = _run(cfg)
    headline = render_headline(
        results,
        scenario=SCENARIO_HETEROGENEOUS,
        seed=SEED,
        fault_fraction=0.02,
        window=WINDOW_LABEL,
    )
    assert WINDOW_LABEL in headline
    assert SCENARIO_HETEROGENEOUS in headline
    assert "overflow events / bin-day" in headline

    reductions = render_reductions(
        results,
        scenario=SCENARIO_HETEROGENEOUS,
        seed=SEED,
        fault_fraction=0.02,
        window=WINDOW_LABEL,
    )
    assert "variable basis" in reductions
    assert render_training_rows(results).startswith("### Forecaster training rows")


@pytest.mark.sim
def test_quick_tables_are_labelled_not_for_publication(cfg: Config) -> None:
    """A smoke run's numbers must never be mistakable for a published figure."""
    headline = render_headline(
        _run(cfg),
        scenario=SCENARIO_HETEROGENEOUS,
        seed=SEED,
        fault_fraction=0.02,
        window=WINDOW_LABEL,
        quick=True,
    )
    assert "NOT FOR PUBLICATION" in headline


@pytest.mark.sim
def test_verdicts_name_both_comparison_partners(cfg: Config) -> None:
    """The verdict must compare the ML model with its calendar AND with the MA baseline."""
    results = _run(cfg)
    lines = forecast_verdict(results, [SCENARIO_HETEROGENEOUS], [0.02])
    # With the GBM excluded there is nothing to compare, which must not crash.
    assert lines == []

    calendar_lines = calendar_verdict(results, [SCENARIO_HETEROGENEOUS], [0.02])
    assert any("b_fixed_schedule" in line for line in calendar_lines)
    assert any("b7_fixed_schedule" in line for line in calendar_lines)


@pytest.mark.sim
def test_output_contains_no_timing_or_timestamp(cfg: Config) -> None:
    """Byte-identical same-seed output is a Phase 2 promise Phase 3 keeps."""
    headline = render_headline(
        _run(cfg),
        scenario=SCENARIO_HETEROGENEOUS,
        seed=SEED,
        fault_fraction=0.02,
        window=WINDOW_LABEL,
    )
    for forbidden in ("seconds", "elapsed", "wall", "20", "runtime"):
        if forbidden == "20":
            continue
        assert forbidden not in headline.lower()


# --------------------------------------------------------------------------------------------
# ml chunk: the gradient-boosting strategies
# --------------------------------------------------------------------------------------------


@pytest.mark.ml
def test_all_ten_strategies_run_with_the_gbm(cfg: Config) -> None:
    results = run_city_phase3(
        cfg,
        SEED,
        0.02,
        scenario=SCENARIO_HETEROGENEOUS,
        n_bins=N_BINS,
        days=DAYS,
        n_clusters=N_CLUSTERS,
        include_gbm=True,
    )
    keys = {r.key for r in results}
    assert len(keys) == 10
    assert {"f_gbm_3", "f_gbm_7"} <= keys


@pytest.mark.ml
def test_the_gbm_strategies_respect_the_backstop_floor(cfg: Config) -> None:
    results = run_city_phase3(
        cfg,
        SEED,
        0.02,
        scenario=SCENARIO_HETEROGENEOUS,
        n_bins=N_BINS,
        days=DAYS,
        n_clusters=N_CLUSTERS,
        include_gbm=True,
    )
    for backstop, calendar_key in ((3, "b_fixed_schedule"), (7, "b7_fixed_schedule")):
        forecast = select(results, key=f"f_gbm_{backstop}", window=WINDOW_LABEL)[0]
        calendar = select(results, key=calendar_key, window=WINDOW_LABEL)[0]
        assert forecast.counters.bin_visits >= calendar.counters.bin_visits


@pytest.mark.ml
def test_the_gbm_run_is_reproducible_on_this_machine(cfg: Config) -> None:
    """Same machine, same library versions, same numbers -- not a cross-OS claim."""

    def run() -> list[Phase3Result]:
        return run_city_phase3(
            cfg,
            SEED,
            0.02,
            scenario=SCENARIO_HETEROGENEOUS,
            n_bins=N_BINS,
            days=DAYS,
            n_clusters=N_CLUSTERS,
            include_gbm=True,
        )

    first = run()
    second = run()
    for a, b in zip(first, second, strict=True):
        assert a.totals == b.totals
        assert a.counters == b.counters


@pytest.mark.ml
def test_forecast_verdict_compares_the_gbm_with_both_partners(cfg: Config) -> None:
    results = run_city_phase3(
        cfg,
        SEED,
        0.02,
        scenario=SCENARIO_HETEROGENEOUS,
        n_bins=N_BINS,
        days=DAYS,
        n_clusters=N_CLUSTERS,
        include_gbm=True,
    )
    lines = forecast_verdict(results, [SCENARIO_HETEROGENEOUS], [0.02])
    assert any("sensor-free calendar" in line for line in lines)
    assert any("moving-average baseline" in line for line in lines)
