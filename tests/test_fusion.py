"""Boundary tests for the paper's fusion rule.

The 60 and 80 thresholds and the overload flag are where every misclassification in the Phase 1
results comes from, so each boundary is pinned exactly rather than approximately.
"""

from __future__ import annotations

import pytest

from smartwaste import load_config
from smartwaste.fusion import (
    BinState,
    classify,
    fill_percent,
    fuse,
    is_overloaded,
    mean_distance,
    truth_state,
)
from smartwaste.noise import SensorReading

pytestmark = pytest.mark.unit

YELLOW_AT = 60.0
RED_AT = 80.0


def classify_fv(fv: float, overloaded: bool = False) -> BinState:
    return classify(fv, overloaded, YELLOW_AT, RED_AT)


# --- the two thresholds, exactly -------------------------------------------------------------


def test_fv_exactly_60_is_yellow() -> None:
    assert classify_fv(60.0) is BinState.YELLOW


def test_fv_just_below_60_is_green() -> None:
    assert classify_fv(59.99) is BinState.GREEN


def test_fv_exactly_80_is_red() -> None:
    assert classify_fv(80.0) is BinState.RED


def test_fv_just_below_80_is_yellow() -> None:
    assert classify_fv(79.99) is BinState.YELLOW


def test_fv_zero_and_hundred() -> None:
    assert classify_fv(0.0) is BinState.GREEN
    assert classify_fv(100.0) is BinState.RED


# --- overload overrides the fill ------------------------------------------------------------


def test_overload_at_exactly_alpha_w_max_is_red() -> None:
    cfg = load_config().bin
    assert is_overloaded(cfg.weight_threshold_kg, cfg.w_max_kg, cfg.alpha) is True


def test_overload_just_below_threshold_is_not_overloaded() -> None:
    cfg = load_config().bin
    assert is_overloaded(cfg.weight_threshold_kg - 1e-9, cfg.w_max_kg, cfg.alpha) is False


def test_overload_forces_red_even_at_zero_fill() -> None:
    assert classify_fv(0.0, overloaded=True) is BinState.RED


def test_overload_forces_red_over_a_green_fill() -> None:
    assert classify_fv(10.0, overloaded=True) is BinState.RED


# --- Fv clipping ----------------------------------------------------------------------------


def test_fill_clips_to_zero_when_distance_equals_height() -> None:
    assert fill_percent(100.0, 100.0) == 0.0


def test_fill_clips_to_zero_when_distance_exceeds_height() -> None:
    # Noise can report a distance past the bin floor; Fv must not go negative.
    assert fill_percent(140.0, 100.0) == 0.0


def test_fill_clips_to_hundred_at_zero_distance() -> None:
    assert fill_percent(0.0, 100.0) == 100.0


def test_fill_clips_to_hundred_for_negative_distance() -> None:
    assert fill_percent(-25.0, 100.0) == 100.0


def test_fill_is_linear_between_the_clips() -> None:
    assert fill_percent(40.0, 100.0) == pytest.approx(60.0)


def test_fill_percent_rejects_non_positive_height() -> None:
    with pytest.raises(ValueError, match="height_cm"):
        fill_percent(10.0, 0.0)


# --- d_bar ----------------------------------------------------------------------------------


def test_mean_distance_is_the_arithmetic_mean() -> None:
    assert mean_distance([30.0, 40.0, 50.0]) == pytest.approx(40.0)


def test_mean_distance_of_one_sensor_is_that_sensor() -> None:
    assert mean_distance([37.5]) == pytest.approx(37.5)


def test_mean_distance_rejects_empty() -> None:
    with pytest.raises(ValueError, match="at least one"):
        mean_distance([])


# --- fuse(): channel selection --------------------------------------------------------------


def test_fuse_without_load_cell_ignores_an_overloading_weight() -> None:
    """A heavy but physically empty bin: only the load-cell method may call it RED."""
    cfg = load_config().bin
    reading = SensorReading(distances_cm=(100.0, 100.0, 100.0), weight_kg=cfg.w_max_kg)

    without = fuse(reading, cfg, use_load_cell=False)
    assert without.fill_pct == 0.0
    assert without.overloaded is False
    assert without.state is BinState.GREEN

    with_cell = fuse(reading, cfg, use_load_cell=True)
    assert with_cell.overloaded is True
    assert with_cell.state is BinState.RED


def test_fuse_center_only_reads_just_the_center_sensor() -> None:
    cfg = load_config().bin
    # Left and right say empty, centre says full.
    reading = SensorReading(distances_cm=(100.0, 10.0, 100.0), weight_kg=0.0)
    center = fuse(reading, cfg, use_load_cell=False, sensor_indices=(1,))
    mean3 = fuse(reading, cfg, use_load_cell=False)
    assert center.fill_pct == pytest.approx(90.0)
    assert center.state is BinState.RED
    assert mean3.fill_pct == pytest.approx(30.0)
    assert mean3.state is BinState.GREEN


# --- ground truth uses the same rule --------------------------------------------------------


def test_truth_state_matches_classify_on_the_same_inputs() -> None:
    cfg = load_config().bin
    for fill, weight in ((0.0, 0.0), (60.0, 1.0), (80.0, 1.0), (10.0, cfg.w_max_kg)):
        expected = classify(
            fill,
            is_overloaded(weight, cfg.w_max_kg, cfg.alpha),
            cfg.yellow_threshold_pct,
            cfg.red_threshold_pct,
        )
        assert truth_state(fill, weight, cfg) is expected


def test_truth_state_red_by_weight_alone() -> None:
    cfg = load_config().bin
    assert truth_state(0.0, cfg.weight_threshold_kg, cfg) is BinState.RED
