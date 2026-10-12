"""Tests for the sensor noise model.

The point of these is that each noise effect does what the Phase 1 results claim it does --
especially that ``center_peak`` biases the centre sensor while leaving the 3-sensor mean
unbiased, since that is the mechanism the whole "fusion wins" story rests on.
"""

from __future__ import annotations

import random
import statistics

import pytest

from smartwaste import load_config
from smartwaste.config import Config
from smartwaste.noise import (
    CENTER_INDEX,
    SKEW_CENTER_PEAK,
    SKEW_RANDOM_TILT,
    NoiseParams,
    SensorArray,
)

pytestmark = pytest.mark.unit


def make_array(params: NoiseParams, seed: int = 1) -> tuple[SensorArray, Config]:
    cfg = load_config()
    return SensorArray(params, cfg.bin, cfg.sensors, random.Random(seed)), cfg


# --- no noise ------------------------------------------------------------------------------


def test_zero_noise_returns_the_exact_nominal_distance() -> None:
    array, cfg = make_array(NoiseParams())
    reading = array.read(40.0, 20.0)
    # 40% of a 100 cm bin => 40 cm of waste => 60 cm of air.
    assert reading.distances_cm == (60.0, 60.0, 60.0)
    assert reading.weight_kg == pytest.approx(20.0)


def test_zero_noise_is_exact_at_both_ends() -> None:
    array, cfg = make_array(NoiseParams())
    assert array.read(0.0, 0.0).distances_cm[CENTER_INDEX] == pytest.approx(cfg.bin.height_cm)
    assert array.read(100.0, 0.0).distances_cm[CENTER_INDEX] == pytest.approx(
        cfg.sensors.min_distance_cm
    )


# --- uneven fill ---------------------------------------------------------------------------


def _sample(
    params: NoiseParams, n: int = 4000, fill: float = 50.0
) -> tuple[list[float], list[float]]:
    """Collect centre readings and 3-sensor means over many draws."""
    array, _ = make_array(params, seed=7)
    centers: list[float] = []
    means: list[float] = []
    for _ in range(n):
        d = array.read(fill, 25.0).distances_cm
        centers.append(d[CENTER_INDEX])
        means.append(statistics.fmean(d))
    return centers, means


def test_center_peak_biases_the_center_sensor_short() -> None:
    """Waste heaps at the centre, so the centre sensor reads a shorter distance."""
    skew = 12.0
    centers, _ = _sample(NoiseParams(skew_cm=skew, skew_mode=SKEW_CENTER_PEAK))
    nominal = 50.0  # 50% of a 100 cm bin
    # Offsets are U(0, skew) with the centre heaping up, so the expected bias is -skew/2.
    assert statistics.fmean(centers) == pytest.approx(nominal - skew / 2.0, abs=0.5)


def test_center_peak_leaves_the_three_sensor_mean_unbiased() -> None:
    """This is why fusion can win: the mean absorbs the skew the centre sensor suffers."""
    _, means = _sample(NoiseParams(skew_cm=12.0, skew_mode=SKEW_CENTER_PEAK))
    assert statistics.fmean(means) == pytest.approx(50.0, abs=0.3)


def test_random_tilt_leaves_the_center_sensor_unbiased() -> None:
    """Under a zero-mean tilt the centre sensor is unbiased, so (b) wins only on variance."""
    centers, means = _sample(NoiseParams(skew_cm=12.0, skew_mode=SKEW_RANDOM_TILT))
    assert statistics.fmean(centers) == pytest.approx(50.0, abs=0.3)
    assert statistics.fmean(means) == pytest.approx(50.0, abs=0.3)


def test_random_tilt_mean_has_lower_spread_than_a_single_side_sensor() -> None:
    params = NoiseParams(skew_cm=12.0, skew_mode=SKEW_RANDOM_TILT)
    array, _ = make_array(params, seed=11)
    lefts = []
    means = []
    for _ in range(4000):
        d = array.read(50.0, 25.0).distances_cm
        lefts.append(d[0])
        means.append(statistics.fmean(d))
    assert statistics.pstdev(means) < statistics.pstdev(lefts)


# --- stuck sensors -------------------------------------------------------------------------


def test_hard_stuck_sensor_latches_while_the_others_move() -> None:
    array, _ = make_array(NoiseParams(sigma_cm=1.5, stuck_sensor=0), seed=3)
    first = array.read(10.0, 5.0).distances_cm
    later = [array.read(float(fill), 5.0).distances_cm for fill in (30.0, 50.0, 70.0, 90.0)]

    assert {d[0] for d in later} == {first[0]}, "stuck sensor must repeat its first reading"
    assert len({d[CENTER_INDEX] for d in later}) > 1, "healthy sensors must still vary"


def test_stuck_probability_one_freezes_every_sensor_after_the_first_read() -> None:
    array, _ = make_array(NoiseParams(stuck_probability=1.0), seed=5)
    first = array.read(20.0, 5.0).distances_cm
    assert array.read(90.0, 40.0).distances_cm == first


def test_zero_stuck_probability_never_freezes() -> None:
    array, _ = make_array(NoiseParams(sigma_cm=1.0), seed=5)
    first = array.read(20.0, 5.0).distances_cm
    assert array.read(90.0, 40.0).distances_cm != first


# --- clamping and the load cell ------------------------------------------------------------


def test_distances_are_clamped_to_the_sensor_range() -> None:
    cfg = load_config()
    array, _ = make_array(NoiseParams(sigma_cm=20.0, bias_max_cm=10.0), seed=13)
    for fill in (0.0, 50.0, 100.0):
        for d in array.read(fill, 25.0).distances_cm:
            assert cfg.sensors.min_distance_cm <= d <= cfg.bin.height_cm


def test_weight_is_never_negative() -> None:
    array, _ = make_array(NoiseParams(load_cell_sigma_kg=5.0, load_cell_bias_max_kg=5.0), seed=17)
    for _ in range(200):
        assert array.read(0.0, 0.0).weight_kg >= 0.0


def test_load_cell_noise_perturbs_the_weight() -> None:
    array, _ = make_array(NoiseParams(load_cell_sigma_kg=1.0), seed=19)
    weights = {array.read(50.0, 25.0).weight_kg for _ in range(50)}
    assert len(weights) > 1


# --- reproducibility and validation --------------------------------------------------------


def test_same_seed_gives_identical_readings() -> None:
    params = NoiseParams(sigma_cm=1.5, bias_max_cm=3.0, skew_cm=10.0, skew_mode=SKEW_CENTER_PEAK)
    a, _ = make_array(params, seed=99)
    b, _ = make_array(params, seed=99)
    assert [a.read(50.0, 25.0) for _ in range(20)] == [b.read(50.0, 25.0) for _ in range(20)]


def test_different_seed_gives_different_readings() -> None:
    params = NoiseParams(sigma_cm=1.5)
    a, _ = make_array(params, seed=1)
    b, _ = make_array(params, seed=2)
    assert a.read(50.0, 25.0) != b.read(50.0, 25.0)


def test_reading_carries_no_ground_truth() -> None:
    """Structural guard: a classifier must not be able to reach the truth through a reading."""
    array, _ = make_array(NoiseParams())
    reading = array.read(50.0, 25.0)
    assert set(vars(reading)) == {"distances_cm", "weight_kg"}


def test_unknown_skew_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="skew_mode"):
        make_array(NoiseParams(skew_mode="diagonal"))


def test_out_of_range_stuck_sensor_is_rejected() -> None:
    with pytest.raises(ValueError, match="stuck_sensor"):
        make_array(NoiseParams(stuck_sensor=7))
