"""Cost model tests, against hand-computed arithmetic.

Every expected number here is worked out by hand in the test body from the config constants, so a
change to the cost formula fails loudly instead of quietly shifting every published figure.
"""

import pytest

from smartwaste import load_config
from smartwaste.config import CostConfig
from smartwaste.cost import (
    CostTotals,
    DayCost,
    TruckDay,
    aggregate,
    cost_truck_day,
    percent_reduction,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def cost_cfg() -> CostConfig:
    return load_config().cost


def test_hand_computed_truck_day(cost_cfg: CostConfig) -> None:
    """100 km, 20 bins, with every term worked out by hand from config/default.yaml."""
    result = cost_truck_day(TruckDay(distance_km=100.0, bins_visited=20), cost_cfg)

    # Time: 100 km / 25 kmph = 4.0 hr driving; 20 bins x 3 min = 60 min = 1.0 hr service.
    assert result.travel_time_hr == pytest.approx(5.0)
    # Fuel: F = a*D = 0.35 * 100 = 35 L.
    assert result.fuel_l == pytest.approx(35.0)
    # CO2: 2.68 kg/L * 35 L = 93.8 kg.
    assert result.co2_kg == pytest.approx(93.8)
    # Variable cost: fuel 35 * 95 = 3325; maintenance 8 * 100 = 800; labor 180 * 5 = 900.
    assert result.variable_cost == pytest.approx(3325.0 + 800.0 + 900.0)
    assert result.variable_cost == pytest.approx(5025.0)
    assert result.rolled is True


def test_co2_is_linear_in_distance(cost_cfg: CostConfig) -> None:
    """E = b*D: doubling the distance doubles the emissions."""
    one = cost_truck_day(TruckDay(distance_km=10.0, bins_visited=0), cost_cfg)
    two = cost_truck_day(TruckDay(distance_km=20.0, bins_visited=0), cost_cfg)
    assert two.co2_kg == pytest.approx(2.0 * one.co2_kg)
    assert two.fuel_l == pytest.approx(2.0 * one.fuel_l)


def test_service_time_makes_cost_depend_on_bin_count(cost_cfg: CostConfig) -> None:
    """Two routes of equal length but different bin counts must not cost the same."""
    few = cost_truck_day(TruckDay(distance_km=50.0, bins_visited=5), cost_cfg)
    many = cost_truck_day(TruckDay(distance_km=50.0, bins_visited=50), cost_cfg)

    assert few.distance_km == many.distance_km
    assert few.fuel_l == pytest.approx(many.fuel_l)  # fuel is distance-only
    # 45 extra bins x 3 min = 135 min = 2.25 hr of labor at 180/hr = 405.
    assert many.travel_time_hr - few.travel_time_hr == pytest.approx(2.25)
    assert many.variable_cost - few.variable_cost == pytest.approx(405.0)


def test_a_parked_truck_costs_nothing_variable(cost_cfg: CostConfig) -> None:
    result = cost_truck_day(TruckDay(distance_km=0.0, bins_visited=0), cost_cfg)
    assert result.variable_cost == 0.0
    assert result.co2_kg == 0.0
    assert result.rolled is False


def test_a_truck_that_visits_bins_has_rolled_even_at_zero_distance(cost_cfg: CostConfig) -> None:
    """A degenerate zero-length tour still sent a crew out, so it must be charged fixed cost."""
    result = cost_truck_day(TruckDay(distance_km=0.0, bins_visited=3), cost_cfg)
    assert result.rolled is True


def test_negative_inputs_are_rejected(cost_cfg: CostConfig) -> None:
    with pytest.raises(ValueError, match="distance_km"):
        cost_truck_day(TruckDay(distance_km=-1.0, bins_visited=0), cost_cfg)
    with pytest.raises(ValueError, match="bins_visited"):
        cost_truck_day(TruckDay(distance_km=1.0, bins_visited=-1), cost_cfg)


def test_three_cost_bases_are_hand_computed(cost_cfg: CostConfig) -> None:
    """Two rolling truck-days out of a four-truck-day baseline fleet."""
    days = [
        cost_truck_day(TruckDay(distance_km=100.0, bins_visited=20), cost_cfg),
        cost_truck_day(TruckDay(distance_km=100.0, bins_visited=20), cost_cfg),
        cost_truck_day(TruckDay(distance_km=0.0, bins_visited=0), cost_cfg),
        cost_truck_day(TruckDay(distance_km=0.0, bins_visited=0), cost_cfg),
    ]
    totals = aggregate(days, cost_cfg, fleet_truck_days=4)

    variable = 2 * 5025.0
    assert totals.variable_cost == pytest.approx(variable)
    assert totals.rolling_truck_days == 2
    # Rolling basis charges 2000/day for the 2 trucks that rolled.
    assert totals.total_cost_rolling == pytest.approx(variable + 2 * 2000.0)
    # Fixed-fleet basis charges all 4 baseline truck-days.
    assert totals.total_cost_fixed_fleet == pytest.approx(variable + 4 * 2000.0)
    # The fixed-fleet basis is never cheaper than the rolling one.
    assert totals.total_cost_fixed_fleet >= totals.total_cost_rolling


def test_fixed_fleet_basis_uses_the_passed_fleet_not_the_day_count(cost_cfg: CostConfig) -> None:
    """A strategy that leaves clusters idle is still charged for the baseline fleet."""
    days = [cost_truck_day(TruckDay(distance_km=10.0, bins_visited=1), cost_cfg)]
    totals = aggregate(days, cost_cfg, fleet_truck_days=600)
    assert totals.total_cost_fixed_fleet == pytest.approx(totals.variable_cost + 600 * 2000.0)


def test_aggregate_of_nothing_is_zero(cost_cfg: CostConfig) -> None:
    totals = aggregate([], cost_cfg, fleet_truck_days=0)
    assert isinstance(totals, CostTotals)
    assert totals.variable_cost == 0.0
    assert totals.total_cost_rolling == 0.0
    assert totals.rolling_truck_days == 0


def test_percent_reduction_hand_cases() -> None:
    assert percent_reduction(100.0, 75.0) == pytest.approx(25.0)
    assert percent_reduction(100.0, 100.0) == pytest.approx(0.0)
    # A worse result reports negative, not clamped to zero.
    assert percent_reduction(100.0, 120.0) == pytest.approx(-20.0)
    # A zero baseline has no meaningful percentage; 0 rather than a division error.
    assert percent_reduction(0.0, 5.0) == 0.0


def test_day_cost_is_frozen(cost_cfg: CostConfig) -> None:
    result = cost_truck_day(TruckDay(distance_km=1.0, bins_visited=1), cost_cfg)
    assert isinstance(result, DayCost)
    with pytest.raises(Exception):  # noqa: B017 - FrozenInstanceError
        result.distance_km = 2.0  # type: ignore[misc]
