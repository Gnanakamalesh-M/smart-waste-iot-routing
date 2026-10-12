"""Tests for the Phase 1 evaluation machinery.

Reproducibility is the main thing under test here: every number in docs/PHASE1_RESULTS.md comes
out of this module, so the seed derivation is pinned to literals and the evaluation is checked
to be deterministic.
"""

from __future__ import annotations

import random

import pytest

from smartwaste import load_config
from smartwaste.fusion_eval import (
    FAIRNESS_LABELS,
    METHODS,
    bootstrap_ratio_ci,
    build_scenarios,
    child_seed,
    evaluate_scenario,
    ratio,
    render_markdown,
    verdicts,
)

# --- seed derivation: pinned, because every result number depends on it ----------------------


@pytest.mark.unit
def test_child_seed_is_pinned_to_known_values() -> None:
    """Hardcoded so a refactor of child_seed cannot silently change every published number.

    These are sha256-derived, NOT from hash(), which is salted per process.
    """
    assert child_seed(42, "clean") == 7982154272178045808
    assert child_seed(42, "noisy") == 8238804300110788029
    assert child_seed(42, "clean:bin:0") == 3167219703595189758


@pytest.mark.unit
def test_child_seed_is_stable_across_calls() -> None:
    assert child_seed(42, "clean") == child_seed(42, "clean")


@pytest.mark.unit
def test_child_seed_varies_with_label_and_seed() -> None:
    assert child_seed(42, "clean") != child_seed(42, "noisy")
    assert child_seed(42, "clean") != child_seed(7, "clean")


# --- scenarios -------------------------------------------------------------------------------


@pytest.mark.unit
def test_every_scenario_is_labelled_for_fairness() -> None:
    for scenario in build_scenarios(load_config()):
        assert scenario.fairness in FAIRNESS_LABELS, scenario.name


@pytest.mark.unit
def test_scenario_names_are_unique() -> None:
    names = [s.name for s in build_scenarios(load_config())]
    assert len(names) == len(set(names))


@pytest.mark.unit
def test_the_clean_scenario_has_no_noise() -> None:
    clean = build_scenarios(load_config())[0]
    assert clean.name == "clean"
    assert clean.params.sigma_cm == 0.0
    assert clean.params.bias_max_cm == 0.0
    assert clean.params.stuck_probability == 0.0


@pytest.mark.unit
def test_scenarios_use_config_noise_levels_untouched() -> None:
    """Guard against quietly tuning noise to make fusion win."""
    cfg = load_config()
    noisy = next(s for s in build_scenarios(cfg) if s.name == "noisy")
    assert noisy.params.sigma_cm == cfg.sensors.noise_sigma_cm
    assert noisy.params.bias_max_cm == cfg.sensors.bias_max_cm


@pytest.mark.unit
def test_only_the_load_cell_method_uses_the_load_cell() -> None:
    assert [m.use_load_cell for m in METHODS] == [False, False, True]


# --- rates ----------------------------------------------------------------------------------


@pytest.mark.unit
def test_ratio_is_none_on_an_empty_denominator() -> None:
    """An unmeasurable rate must read as n/a, not as a perfect 0."""
    assert ratio(0, 0) is None
    assert ratio(3, 0) is None


@pytest.mark.unit
def test_ratio_computes_the_obvious_thing() -> None:
    assert ratio(3, 4) == pytest.approx(0.75)


# --- bootstrap -------------------------------------------------------------------------------


@pytest.mark.unit
def test_bootstrap_brackets_the_point_estimate() -> None:
    per_bin = [(2.0, 10.0)] * 40  # every bin at 20%
    ci = bootstrap_ratio_ci(per_bin, random.Random(0), n_resamples=400)
    assert ci is not None
    low, high = ci
    assert low <= 0.2 <= high


@pytest.mark.unit
def test_bootstrap_is_deterministic_for_a_fixed_seed() -> None:
    per_bin = [(float(i % 4), 10.0) for i in range(50)]
    first = bootstrap_ratio_ci(per_bin, random.Random(5), n_resamples=300)
    second = bootstrap_ratio_ci(per_bin, random.Random(5), n_resamples=300)
    assert first == second


@pytest.mark.unit
def test_bootstrap_differs_with_a_different_seed() -> None:
    per_bin = [(float(i % 4), 10.0) for i in range(50)]
    first = bootstrap_ratio_ci(per_bin, random.Random(5), n_resamples=300)
    second = bootstrap_ratio_ci(per_bin, random.Random(6), n_resamples=300)
    assert first != second


@pytest.mark.unit
def test_bootstrap_is_none_when_nothing_contributes_a_denominator() -> None:
    assert bootstrap_ratio_ci([(0.0, 0.0)] * 10, random.Random(0), n_resamples=50) is None


@pytest.mark.unit
def test_bootstrap_is_none_on_empty_input() -> None:
    assert bootstrap_ratio_ci([], random.Random(0), n_resamples=50) is None


@pytest.mark.unit
def test_bootstrap_of_a_degenerate_sample_is_a_point() -> None:
    assert bootstrap_ratio_ci([(10.0, 10.0)] * 20, random.Random(1), n_resamples=100) == (1.0, 1.0)


# --- metric arithmetic, on a hand-checked scenario ------------------------------------------


@pytest.mark.unit
def test_clean_scenario_metrics_are_hand_checkable() -> None:
    """With no noise at all, every method must reproduce the truth exactly."""
    cfg = load_config()
    clean = build_scenarios(cfg)[0]
    rows = evaluate_scenario(clean, cfg, seed=42, n_bins=5, days=6)

    assert len(rows) == len(METHODS)
    for row in rows:
        assert row.n_readings == 30, "5 bins x 6 days"
        assert row.n_red + row.n_not_red == row.n_readings
        assert row.misclass_rate == 0.0
        assert row.false_red_rate == 0.0
        # No noise means no missed full bins either; n/a only if no RED day occurred.
        assert row.false_green_rate in (0.0, None)


@pytest.mark.unit
def test_rates_use_the_right_denominators() -> None:
    cfg = load_config()
    noisy = next(s for s in build_scenarios(cfg) if s.name == "noisy")
    for row in evaluate_scenario(noisy, cfg, seed=42, n_bins=6, days=5):
        assert row.n_readings == 30
        assert row.n_red + row.n_not_red == row.n_readings
        if row.n_red == 0:
            assert row.false_green_rate is None
        if row.n_not_red == 0:
            assert row.false_red_rate is None


# --- reproducibility of the whole evaluation ------------------------------------------------


@pytest.mark.sim
def test_same_seed_gives_identical_metrics() -> None:
    cfg = load_config()
    scenario = next(s for s in build_scenarios(cfg) if s.name == "noisy")
    first = evaluate_scenario(scenario, cfg, seed=42, n_bins=8, days=10)
    second = evaluate_scenario(scenario, cfg, seed=42, n_bins=8, days=10)
    assert first == second


@pytest.mark.sim
def test_different_seed_changes_the_metrics() -> None:
    cfg = load_config()
    scenario = next(s for s in build_scenarios(cfg) if s.name == "noisy")
    first = evaluate_scenario(scenario, cfg, seed=42, n_bins=8, days=10)
    second = evaluate_scenario(scenario, cfg, seed=43, n_bins=8, days=10)
    assert [r.misclass_rate for r in first] != [r.misclass_rate for r in second]


@pytest.mark.sim
def test_noise_actually_degrades_accuracy() -> None:
    """Sanity check that the noise model is wired in at all."""
    cfg = load_config()
    scenarios = {s.name: s for s in build_scenarios(cfg)}
    clean = evaluate_scenario(scenarios["clean"], cfg, seed=42, n_bins=10, days=10)
    noisy = evaluate_scenario(scenarios["noisy"], cfg, seed=42, n_bins=10, days=10)
    assert max(r.misclass_rate for r in noisy) > max(r.misclass_rate for r in clean)


@pytest.mark.sim
def test_center_peak_makes_the_mean_beat_the_single_sensor() -> None:
    """The acceptance criterion in SPEC section 7, as a test."""
    cfg = load_config()
    scenario = next(s for s in build_scenarios(cfg) if s.name == "uneven_center_peak")
    rows = {r.method: r for r in evaluate_scenario(scenario, cfg, seed=42, n_bins=40, days=30)}
    assert rows["b_mean3"].misclass_rate < rows["a_center_only"].misclass_rate


# --- rendering ------------------------------------------------------------------------------


@pytest.mark.unit
def test_markdown_table_has_a_row_per_method_and_no_timestamp() -> None:
    cfg = load_config()
    clean = build_scenarios(cfg)[0]
    rows = evaluate_scenario(clean, cfg, seed=42, n_bins=3, days=4)
    table = render_markdown(rows, "Title")
    assert table.count("\n| clean |") == len(METHODS)
    assert "control" in table
    for label in ("20", "202"):  # a year-like token would mean a timestamp crept in
        assert label not in table.split("|---")[0]


@pytest.mark.unit
def test_verdict_reports_one_line_per_scenario_and_names_the_loser() -> None:
    cfg = load_config()
    scenario = next(s for s in build_scenarios(cfg) if s.name == "noisy")
    lines = verdicts(evaluate_scenario(scenario, cfg, seed=42, n_bins=5, days=6))
    assert len(lines) == 1
    assert lines[0].startswith("noisy [fair test]:")
    assert "(b) mean3" in lines[0]
