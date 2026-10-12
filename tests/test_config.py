"""Tests for the config loader, against the real config/default.yaml."""

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
import yaml

from smartwaste import DEFAULT_CONFIG_PATH, ConfigError, load_config

pytestmark = pytest.mark.unit


def test_default_config_path_points_at_the_shipped_file(default_config_path: Path) -> None:
    assert default_config_path == DEFAULT_CONFIG_PATH
    assert DEFAULT_CONFIG_PATH.is_file()


def test_loads_real_default_yaml_with_expected_values() -> None:
    cfg = load_config()

    assert cfg.seed == 42
    assert 0 < cfg.bin.alpha < 1
    assert cfg.bin.yellow_threshold_pct == 60.0
    assert cfg.bin.red_threshold_pct == 80.0
    assert cfg.sensors.n_ultrasonic == 3
    assert cfg.sim.n_bins == 5000
    assert cfg.sim.days == 30
    assert cfg.paths.results_dir == "results"


def test_loads_the_phase_1_noise_and_eval_parameters() -> None:
    cfg = load_config()

    assert cfg.sensors.min_distance_cm > 0
    assert cfg.sensors.load_cell_sigma_kg >= 0
    assert cfg.sensors.load_cell_bias_max_kg >= 0
    assert cfg.bin.density_kg_per_pct > 0
    assert 0 <= cfg.bin.density_cv < 1
    assert cfg.fusion_eval.n_bins >= 1
    assert cfg.fusion_eval.days >= 1
    assert cfg.fusion_eval.bootstrap_resamples >= 1


def test_density_cv_must_be_a_fraction() -> None:
    with pytest.raises(ConfigError, match="density_cv"):
        load_config(overrides={"bin": {"density_cv": 1.0}})
    with pytest.raises(ConfigError, match="density_cv"):
        load_config(overrides={"bin": {"density_cv": -0.1}})


def test_density_cv_zero_is_allowed() -> None:
    """The sensitivity run needs it: it is the load cell's blind spot, not an error."""
    assert load_config(overrides={"bin": {"density_cv": 0.0}}).bin.density_cv == 0.0


def test_negative_load_cell_noise_is_rejected() -> None:
    with pytest.raises(ConfigError, match="load_cell_sigma_kg"):
        load_config(overrides={"sensors": {"load_cell_sigma_kg": -1.0}})


def test_bootstrap_resamples_must_be_positive() -> None:
    with pytest.raises(ConfigError, match="bootstrap_resamples"):
        load_config(overrides={"fusion_eval": {"bootstrap_resamples": 0}})


def test_weight_threshold_is_alpha_times_w_max() -> None:
    cfg = load_config()
    assert cfg.bin.weight_threshold_kg == pytest.approx(cfg.bin.alpha * cfg.bin.w_max_kg)


def test_yaml_ints_are_accepted_where_floats_are_declared() -> None:
    cfg = load_config()
    assert isinstance(cfg.bin.height_cm, float)
    assert isinstance(cfg.sim.n_bins, int)


def test_override_is_deep_and_does_not_leak_into_other_loads() -> None:
    baseline = load_config()
    overridden = load_config(overrides={"bin": {"alpha": 0.5}})

    assert overridden.bin.alpha == 0.5
    # Siblings in the same section survive the merge.
    assert overridden.bin.w_max_kg == baseline.bin.w_max_kg
    assert overridden.sim.n_bins == baseline.sim.n_bins
    # The earlier object and a fresh load are untouched.
    assert baseline.bin.alpha != 0.5
    assert load_config().bin.alpha == baseline.bin.alpha


def test_override_mapping_is_not_mutated() -> None:
    overrides = {"bin": {"alpha": 0.5}}
    load_config(overrides=overrides)
    assert overrides == {"bin": {"alpha": 0.5}}


def test_invalid_value_raises_config_error() -> None:
    with pytest.raises(ConfigError, match="alpha"):
        load_config(overrides={"bin": {"alpha": 1.5}})


def test_threshold_ordering_is_enforced() -> None:
    with pytest.raises(ConfigError, match="yellow < red"):
        load_config(overrides={"bin": {"yellow_threshold_pct": 90.0}})


def test_unknown_key_in_section_is_rejected_by_name() -> None:
    with pytest.raises(ConfigError, match="alhpa"):
        load_config(overrides={"bin": {"alhpa": 0.5}})


def test_unknown_top_level_section_is_rejected() -> None:
    with pytest.raises(ConfigError, match="blockchain"):
        load_config(overrides={"blockchain": {"enabled": True}})


def test_wrong_type_is_rejected() -> None:
    with pytest.raises(ConfigError, match="sim.n_bins"):
        load_config(overrides={"sim": {"n_bins": "lots"}})


def test_loads_from_an_explicit_path(tmp_path: Path, default_config_path: Path) -> None:
    raw = yaml.safe_load(default_config_path.read_text(encoding="utf-8"))
    raw["seed"] = 7
    raw["sim"]["n_bins"] = 10
    custom = tmp_path / "custom.yaml"
    custom.write_text(yaml.safe_dump(raw), encoding="utf-8")

    cfg = load_config(custom)

    assert cfg.seed == 7
    assert cfg.sim.n_bins == 10


def test_missing_section_is_reported(tmp_path: Path, default_config_path: Path) -> None:
    raw = yaml.safe_load(default_config_path.read_text(encoding="utf-8"))
    del raw["cost"]
    broken = tmp_path / "broken.yaml"
    broken.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ConfigError, match="cost"):
        load_config(broken)


def test_missing_file_is_reported(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")


def test_config_is_frozen() -> None:
    cfg = load_config()
    with pytest.raises(FrozenInstanceError):
        cfg.bin.alpha = 0.1  # type: ignore[misc]


def test_loads_the_phase_2_benchmark_parameters() -> None:
    cfg = load_config()

    assert cfg.bin.max_accumulated_weight_kg > 0
    assert 0 <= cfg.sensors.stuck_bin_fraction <= 1
    assert cfg.benchmark.fixed_schedule_days >= 1
    assert cfg.benchmark.n_seeds >= 1
    assert cfg.benchmark.fault_fractions == (0.0, 0.02, 0.10)


def test_weight_cap_must_leave_the_load_cell_able_to_flag() -> None:
    """Below alpha * W_max, an uncollected bin could never raise the overload flag."""
    with pytest.raises(ConfigError, match="max_accumulated_weight_kg"):
        load_config(overrides={"bin": {"max_accumulated_weight_kg": 1.0}})


def test_stuck_bin_fraction_must_be_a_fraction() -> None:
    with pytest.raises(ConfigError, match="stuck_bin_fraction"):
        load_config(overrides={"sensors": {"stuck_bin_fraction": 1.5}})
    with pytest.raises(ConfigError, match="stuck_bin_fraction"):
        load_config(overrides={"sensors": {"stuck_bin_fraction": -0.1}})


def test_stuck_bin_fraction_endpoints_are_allowed() -> None:
    """The fault sweep runs 0.0, and a 1.0 all-faulted city is a legitimate stress case."""
    assert load_config(overrides={"sensors": {"stuck_bin_fraction": 0.0}}) is not None
    assert load_config(overrides={"sensors": {"stuck_bin_fraction": 1.0}}) is not None


# --- list-of-float coercion (benchmark.fault_fractions) --------------------------------------


def test_fault_fractions_accepts_a_list_of_numbers() -> None:
    cfg = load_config(overrides={"benchmark": {"fault_fractions": [0.0, 0.02, 0.10]}})
    assert cfg.benchmark.fault_fractions == (0.0, 0.02, 0.10)
    assert all(isinstance(value, float) for value in cfg.benchmark.fault_fractions)


def test_fault_fractions_widens_yaml_ints_to_floats() -> None:
    """YAML writes 0 and 1 as ints where we declare floats."""
    cfg = load_config(overrides={"benchmark": {"fault_fractions": [0, 1]}})
    assert cfg.benchmark.fault_fractions == (0.0, 1.0)


def test_fault_fractions_rejects_a_bare_scalar() -> None:
    """A typo'd single value must be an error, not a silent one-element sweep."""
    with pytest.raises(ConfigError, match="fault_fractions"):
        load_config(overrides={"benchmark": {"fault_fractions": 0.02}})


def test_fault_fractions_rejects_a_string() -> None:
    """A string is iterable, so it must be rejected explicitly rather than char by char."""
    with pytest.raises(ConfigError, match="fault_fractions"):
        load_config(overrides={"benchmark": {"fault_fractions": "0.02"}})


def test_fault_fractions_rejects_a_non_number_element() -> None:
    with pytest.raises(ConfigError, match=r"fault_fractions\[1\]"):
        load_config(overrides={"benchmark": {"fault_fractions": [0.0, "x"]}})


def test_fault_fractions_rejects_a_bool_element() -> None:
    """bool is an int subclass, so it needs the same explicit rejection as elsewhere."""
    with pytest.raises(ConfigError, match=r"fault_fractions\[1\]"):
        load_config(overrides={"benchmark": {"fault_fractions": [0.0, True]}})


def test_fault_fractions_rejects_an_empty_list() -> None:
    with pytest.raises(ConfigError, match="at least one value"):
        load_config(overrides={"benchmark": {"fault_fractions": []}})


def test_fault_fractions_values_must_be_fractions() -> None:
    with pytest.raises(ConfigError, match="fault_fractions"):
        load_config(overrides={"benchmark": {"fault_fractions": [0.0, 1.5]}})


def test_paths_resolve_under_the_repo_root(repo_root: Path) -> None:
    cfg = load_config()
    assert cfg.paths.resolve("results_dir") == repo_root / "results"
    with pytest.raises(ConfigError):
        cfg.paths.resolve("nope_dir")
