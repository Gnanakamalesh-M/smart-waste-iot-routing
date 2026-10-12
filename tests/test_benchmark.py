"""Benchmark orchestration and reporting.

Split deliberately across two markers:

- ``sim``: a tiny 50-bin, 5-day reproducibility check. Small enough to run on **every** CI push,
  so "same command + same seed = same numbers" is enforced continuously rather than only when
  someone remembers to run the slow chunk.
- ``slow``: a larger end-to-end run across two fault fractions, exercising the real report
  assembly. Still far smaller than the 5000-bin benchmark, which lives only in the script.
"""

import pytest

from smartwaste import load_config
from smartwaste.benchmark import (
    BASELINE_KEY,
    COST_BASES,
    CSV_FIELDS,
    EXTENSION_KEY,
    PAPER_KEY,
    baseline_beats_paper_verdict,
    csv_rows,
    mean_vs_median_verdict,
    render_headline,
    render_paired_mean_vs_median,
    render_reductions,
    render_seed_range,
    run_benchmark,
    run_city,
    select,
)
from smartwaste.config import Config

TINY = {"n_bins": 50, "days": 5, "n_clusters": 2}
SMALL = {"n_bins": 200, "days": 10, "n_clusters": 4}


@pytest.fixture
def cfg() -> Config:
    return load_config()


# --------------------------------------------------------------------------------------------
# sim chunk: tiny, and run on every push
# --------------------------------------------------------------------------------------------


@pytest.mark.sim
def test_same_seed_gives_identical_numbers(cfg: Config) -> None:
    """The project's core promise, checked cheaply enough to run on every CI push."""
    first = run_city(cfg, 42, 0.02, **TINY)
    second = run_city(cfg, 42, 0.02, **TINY)

    assert [r.key for r in first] == [r.key for r in second]
    for a, b in zip(first, second, strict=True):
        assert a.totals == b.totals
        assert a.bin_visits == b.bin_visits
        assert a.overflow_events == b.overflow_events
        assert a.overflow_bin_days == b.overflow_bin_days
        assert a.overflowed_bins == b.overflowed_bins
        assert a.missed_pickups == b.missed_pickups


@pytest.mark.sim
def test_rendered_tables_are_byte_identical_for_the_same_seed(cfg: Config) -> None:
    """Determinism has to survive formatting, not just the raw floats."""
    first = run_city(cfg, 42, 0.02, **TINY)
    second = run_city(cfg, 42, 0.02, **TINY)

    assert render_headline(first, 42, 0.02) == render_headline(second, 42, 0.02)
    assert render_reductions(first, 42, 0.02) == render_reductions(second, 42, 0.02)


@pytest.mark.sim
def test_a_different_seed_changes_the_numbers(cfg: Config) -> None:
    """Determinism must come from the seed, not from the simulation ignoring it."""
    a = run_city(cfg, 42, 0.02, **TINY)
    b = run_city(cfg, 43, 0.02, **TINY)
    assert [r.totals.distance_km for r in a] != [r.totals.distance_km for r in b]


@pytest.mark.sim
def test_every_strategy_is_reported_once(cfg: Config) -> None:
    results = run_city(cfg, 42, 0.02, **TINY)
    keys = [r.key for r in results]
    assert len(keys) == 5
    assert len(set(keys)) == 5
    assert keys[0] == BASELINE_KEY


@pytest.mark.sim
def test_baseline_never_overflows_and_never_misses(cfg: Config) -> None:
    """(a) visits everything daily, so it is the overflow-free reference by construction."""
    baseline = select(run_city(cfg, 42, 0.02, **TINY), key=BASELINE_KEY)[0]
    assert baseline.overflow_events == 0
    assert baseline.overflowed_bins == 0
    assert baseline.missed_pickups == 0


@pytest.mark.sim
def test_baseline_is_the_most_expensive_and_longest(cfg: Config) -> None:
    """Sanity on the whole pipeline: skipping bins must cost less than visiting all of them."""
    results = run_city(cfg, 42, 0.02, **TINY)
    baseline = select(results, key=BASELINE_KEY)[0]

    for r in results:
        if r.key == BASELINE_KEY:
            continue
        assert r.totals.distance_km <= baseline.totals.distance_km
        assert r.bin_visits <= baseline.bin_visits
        for basis in COST_BASES:
            assert r.cost_on(basis) <= baseline.cost_on(basis)


@pytest.mark.sim
def test_cost_bases_are_ordered(cfg: Config) -> None:
    """variable <= rolling <= fixed_fleet, for every strategy."""
    for r in run_city(cfg, 42, 0.02, **TINY):
        assert r.cost_on("variable") <= r.cost_on("rolling") + 1e-9
        assert r.cost_on("rolling") <= r.cost_on("fixed_fleet") + 1e-9


@pytest.mark.sim
def test_unknown_cost_basis_is_rejected(cfg: Config) -> None:
    r = run_city(cfg, 42, 0.02, **TINY)[0]
    with pytest.raises(ValueError, match="unknown cost basis"):
        r.cost_on("free")


@pytest.mark.sim
def test_csv_rows_cover_every_declared_field(cfg: Config) -> None:
    rows = csv_rows(run_city(cfg, 42, 0.02, **TINY))
    assert rows
    for row in rows:
        assert set(row) == set(CSV_FIELDS)


# --------------------------------------------------------------------------------------------
# slow chunk: the fuller end-to-end shape
# --------------------------------------------------------------------------------------------


@pytest.mark.slow
def test_end_to_end_across_seeds_and_fault_fractions(cfg: Config) -> None:
    seeds = [42, 43]
    faults = [0.0, 0.2]

    results = run_benchmark(cfg, seeds, faults, **SMALL)

    assert len(results) == len(seeds) * len(faults) * 5
    for seed in seeds:
        for fault in faults:
            assert len(select(results, seed=seed, fault_fraction=fault)) == 5


@pytest.mark.slow
def test_faults_make_the_paper_s_mean_fusion_worse(cfg: Config) -> None:
    """Phase 1's finding, now in collection outcomes: a stuck sensor hurts the plain mean.

    Not asserted of the median: that is the thing under test in the results doc, not a property
    we should bake into a test.
    """
    seeds = [42, 43]
    results = run_benchmark(cfg, seeds, [0.0, 0.3], **SMALL)

    clean = [r for r in results if r.key == PAPER_KEY and r.fault_fraction == 0.0]
    faulted = [r for r in results if r.key == PAPER_KEY and r.fault_fraction == 0.3]

    clean_missed = sum(r.missed_pickups for r in clean)
    faulted_missed = sum(r.missed_pickups for r in faulted)
    assert faulted_missed > clean_missed


@pytest.mark.slow
def test_reports_render_for_every_section(cfg: Config) -> None:
    seeds = [42, 43]
    faults = [0.0, 0.2]
    results = run_benchmark(cfg, seeds, faults, **SMALL)

    headline = render_headline(results, 42, 0.0)
    assert "Headline" in headline
    assert "bins overflowed >=1x" in headline

    reductions = render_reductions(results, 42, 0.0)
    # Every cost column must name its basis, so no percentage is ambiguous.
    assert "variable basis" in reductions
    assert "rolling-fleet basis" in reductions
    assert "fixed-fleet basis" in reductions

    seed_range = render_seed_range(results, seeds, 0.0)
    assert "range over seeds, not a confidence interval" in seed_range
    assert "95%" not in seed_range  # never claim a CI for a 2-5 seed range

    paired = render_paired_mean_vs_median(results, seeds, faults)
    assert "(d)-(e)" in paired
    # One row per (fault fraction, seed) pair.
    assert paired.count("\n| 0 | 42 |") + paired.count("\n| 0 | 43 |") == 2


@pytest.mark.slow
def test_verdicts_state_a_direction_for_every_fault_fraction(cfg: Config) -> None:
    faults = [0.0, 0.2]
    results = run_benchmark(cfg, [42, 43], faults, **SMALL)

    verdicts = mean_vs_median_verdict(results, faults)
    assert len(verdicts) == len(faults)
    for line in verdicts:
        # The verdict must commit to one of three readings, including the unflattering one.
        assert ("BEATS" in line) or ("WORSE" in line) or ("TIES" in line)

    baseline_lines = baseline_beats_paper_verdict(results, faults)
    # Two simple baselines, (b) and (c), at each fault fraction.
    assert len(baseline_lines) == 2 * len(faults)


@pytest.mark.slow
def test_median_and_mean_are_identical_when_nothing_is_broken_except_noise(cfg: Config) -> None:
    """With no stuck sensors the two fusion rules should be close, not wildly different.

    A loose bound: this guards against a wiring mistake (e.g. the median strategy accidentally
    reading one sensor), not against a specific performance claim.
    """
    results = run_benchmark(cfg, [42], [0.0], **SMALL)
    d = select(results, key=PAPER_KEY)[0]
    e = select(results, key=EXTENSION_KEY)[0]

    assert d.bin_visits > 0 and e.bin_visits > 0
    ratio = e.totals.distance_km / d.totals.distance_km
    assert 0.5 < ratio < 2.0
