"""Strategy selection logic, including the mean-vs-median fault tolerance claim."""

import pytest

from smartwaste import load_config
from smartwaste.config import BinConfig
from smartwaste.fusion import mean_distance, median_distance
from smartwaste.noise import CENTER_INDEX, SensorReading
from smartwaste.strategies import (
    FixedRoute,
    FixedSchedule,
    ThresholdStrategy,
    build_strategies,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def bin_cfg() -> BinConfig:
    return load_config().bin


def _reading(distances: tuple[float, ...], weight: float = 0.0) -> SensorReading:
    return SensorReading(distances_cm=distances, weight_kg=weight)


def _empty(n: int) -> list[SensorReading]:
    """``n`` bins that are empty: 100 cm of air means 0% fill."""
    return [_reading((100.0, 100.0, 100.0)) for _ in range(n)]


def _full(n: int) -> list[SensorReading]:
    """``n`` bins reading 10 cm of air, i.e. 90% full -- above the RED threshold of 80."""
    return [_reading((10.0, 10.0, 10.0)) for _ in range(n)]


# --------------------------------------------------------------------------------------------
# (a) and (b): the sensor-free baselines
# --------------------------------------------------------------------------------------------


def test_fixed_route_selects_every_bin_every_day() -> None:
    strategy = FixedRoute()
    readings = _empty(10)
    for day in range(5):
        assert strategy.select(day, readings) == list(range(10))


def test_fixed_route_ignores_the_readings_entirely() -> None:
    """Even bins that read empty are collected: that is the point of the naive baseline."""
    strategy = FixedRoute()
    assert strategy.select(0, _empty(4)) == strategy.select(0, _full(4))


def test_fixed_schedule_fires_only_every_n_days() -> None:
    strategy = FixedSchedule(3)
    readings = _empty(6)

    assert strategy.select(0, readings) == list(range(6))
    assert strategy.select(1, readings) == []
    assert strategy.select(2, readings) == []
    assert strategy.select(3, readings) == list(range(6))
    assert strategy.select(6, readings) == list(range(6))


def test_fixed_schedule_of_one_is_the_daily_baseline() -> None:
    daily = FixedSchedule(1)
    readings = _empty(5)
    for day in range(4):
        assert daily.select(day, readings) == list(range(5))


def test_fixed_schedule_label_names_its_n() -> None:
    """The label is printed verbatim in the results table, so N must appear in it."""
    assert "every 7 days" in FixedSchedule(7).label


def test_fixed_schedule_rejects_a_non_positive_n() -> None:
    with pytest.raises(ValueError, match="every_n_days"):
        FixedSchedule(0)


# --------------------------------------------------------------------------------------------
# (c), (d), (e): the threshold strategies
# --------------------------------------------------------------------------------------------


def test_threshold_strategies_collect_red_and_only_red(bin_cfg: BinConfig) -> None:
    """A 90%-full bin is RED and collected; an empty one is GREEN and left alone."""
    for strategy in build_strategies(bin_cfg, 3):
        if not strategy.uses_sensors:
            continue
        assert strategy.select(0, _full(3)) == [0, 1, 2]
        assert strategy.select(0, _empty(3)) == []


def test_yellow_bins_are_never_collected(bin_cfg: BinConfig) -> None:
    """The paper's known weakness, reproduced deliberately: 70% full is YELLOW, so skipped."""
    yellow = [_reading((30.0, 30.0, 30.0))]  # 70% fill: between 60 and 80
    for strategy in build_strategies(bin_cfg, 3):
        if strategy.uses_sensors:
            assert strategy.select(0, yellow) == []


def test_load_cell_escalates_a_heavy_but_not_full_bin(bin_cfg: BinConfig) -> None:
    """(c) has no load cell and misses an overweight bin that (d) and (e) catch."""
    # 50% fill by volume, but over the weight threshold (alpha * W_max = 42.5 kg).
    heavy = [_reading((50.0, 50.0, 50.0), weight=45.0)]
    strategies = {s.key: s for s in build_strategies(bin_cfg, 3)}

    assert strategies["c_single_ultrasonic"].select(0, heavy) == []
    assert strategies["d_paper_fusion"].select(0, heavy) == [0]
    assert strategies["e_median_fusion"].select(0, heavy) == [0]


def test_single_sensor_strategy_reads_only_the_centre(bin_cfg: BinConfig) -> None:
    """Side sensors can say anything; (c) must track the centre alone."""
    strategy = ThresholdStrategy(
        "c",
        "centre only",
        bin_cfg,
        use_load_cell=False,
        sensor_indices=(CENTER_INDEX,),
    )
    # Centre says 90% full; the sides say empty. Centre-only must call it RED.
    assert strategy.select(0, [_reading((100.0, 10.0, 100.0))]) == [0]
    # Centre says empty; the sides say full. Centre-only must call it GREEN.
    assert strategy.select(0, [_reading((10.0, 100.0, 10.0))]) == []


def test_median_survives_one_stuck_sensor_where_the_mean_does_not(bin_cfg: BinConfig) -> None:
    """The whole motivation for the (e) extension, as a unit test.

    A genuinely full bin (two sensors read 10 cm) with one sensor latched at 100 cm:
      mean   = (100 + 10 + 10) / 3 = 40 cm -> 60% fill -> YELLOW -> not collected
      median = 10 cm                       -> 90% fill -> RED    -> collected
    """
    stuck_full = [_reading((100.0, 10.0, 10.0))]
    strategies = {s.key: s for s in build_strategies(bin_cfg, 3)}

    assert mean_distance((100.0, 10.0, 10.0)) == pytest.approx(40.0)
    assert median_distance((100.0, 10.0, 10.0)) == pytest.approx(10.0)

    assert strategies["d_paper_fusion"].select(0, stuck_full) == []
    assert strategies["e_median_fusion"].select(0, stuck_full) == [0]


def test_mean_and_median_agree_when_all_sensors_are_healthy(bin_cfg: BinConfig) -> None:
    """With three equal readings there is nothing to disagree about."""
    strategies = {s.key: s for s in build_strategies(bin_cfg, 3)}
    for readings in (_full(4), _empty(4)):
        assert strategies["d_paper_fusion"].select(0, readings) == strategies[
            "e_median_fusion"
        ].select(0, readings)


def test_selection_is_ascending_and_unique(bin_cfg: BinConfig) -> None:
    mixed = [
        _reading((100.0, 100.0, 100.0)),
        _reading((10.0, 10.0, 10.0)),
        _reading((100.0, 100.0, 100.0)),
        _reading((5.0, 5.0, 5.0)),
    ]
    for strategy in build_strategies(bin_cfg, 3):
        selected = strategy.select(0, mixed)
        assert selected == sorted(selected)
        assert len(selected) == len(set(selected))


# --------------------------------------------------------------------------------------------
# The set as a whole
# --------------------------------------------------------------------------------------------


def test_five_strategies_with_stable_unique_keys(bin_cfg: BinConfig) -> None:
    strategies = build_strategies(bin_cfg, 3)
    keys = [s.key for s in strategies]

    assert len(strategies) == 5
    assert len(set(keys)) == 5
    assert keys == [
        "a_fixed_route",
        "b_fixed_schedule",
        "c_single_ultrasonic",
        "d_paper_fusion",
        "e_median_fusion",
    ]


def test_labels_mark_the_paper_and_the_extension(bin_cfg: BinConfig) -> None:
    """Provenance has to be readable straight off the results table."""
    labels = {s.key: s.label for s in build_strategies(bin_cfg, 3)}

    assert "paper's baseline" in labels["a_fixed_route"]
    assert "the paper's method" in labels["d_paper_fusion"]
    assert "extension, not in the paper" in labels["e_median_fusion"]


def test_only_the_sensor_strategies_claim_to_use_sensors(bin_cfg: BinConfig) -> None:
    flags = {s.key: s.uses_sensors for s in build_strategies(bin_cfg, 3)}
    assert flags == {
        "a_fixed_route": False,
        "b_fixed_schedule": False,
        "c_single_ultrasonic": True,
        "d_paper_fusion": True,
        "e_median_fusion": True,
    }
