"""Economic and emissions model for a day's collection.

The paper's form (SPEC section 2): fuel ``F = a*D``, CO2 ``E = b*D``, and
``cost = fuel + maintenance + labor + fixed``. The paper states the *form* but none of the
constants, so every constant here comes from ``config/default.yaml`` and every one of them is
**invented and uncited** -- see docs/ASSUMPTIONS.md. None was tuned to reproduce the paper's
reported percentages.

Note on ``E = b*D``: the paper's Eq. 13 prints ``F = bD``, a typo for ``E = bD``. We model CO2 per
litre of diesel rather than per km, so the effective ``b`` is
``co2_kg_per_l * fuel_l_per_km``. That is the same linear-in-distance form, just parameterised so
the two constants can be sourced independently if either is ever given a citation.

### The three cost totals, and why there are three

``fixed_cost_per_day`` is charged per truck per day, and whether an idle truck is charged is a
*modelling choice* that materially changes the headline saving. Rather than pick one quietly, we
report all three bases and label every percentage with the one it used:

- ``variable_cost`` -- fuel + maintenance + labor. No fixed cost, so no fleet assumption at all.
- ``total_cost_rolling`` -- plus fixed cost for each truck-day a truck actually rolled. Flatters
  the sensor strategies, because they park trucks whose cluster has no RED bin.
- ``total_cost_fixed_fleet`` -- plus fixed cost for the whole baseline fleet, every day. You own
  the trucks whether they roll or not.
"""

from __future__ import annotations

from dataclasses import dataclass

from smartwaste.config import CostConfig

MINUTES_PER_HOUR = 60.0


@dataclass(frozen=True)
class TruckDay:
    """One truck's work on one day."""

    distance_km: float
    bins_visited: int

    @property
    def rolled(self) -> bool:
        """Did this truck leave the depot at all?

        Keyed on bins visited, not distance: a truck sent to collect bins has rolled even if its
        cluster is so tight the tour rounds to ~0 km.
        """
        return self.bins_visited > 0


@dataclass(frozen=True)
class DayCost:
    """Costed output for one truck-day."""

    distance_km: float
    bins_visited: int
    travel_time_hr: float
    fuel_l: float
    co2_kg: float
    variable_cost: float
    rolled: bool


def cost_truck_day(day: TruckDay, cfg: CostConfig) -> DayCost:
    """Cost, time, fuel and CO2 for one truck on one day.

    Travel time is driving plus service: a route that visits 60 bins costs real time at each
    one, so time is not a pure function of distance. Labor is charged on that total time.
    """
    if day.distance_km < 0:
        raise ValueError(f"distance_km must be >= 0, got {day.distance_km}")
    if day.bins_visited < 0:
        raise ValueError(f"bins_visited must be >= 0, got {day.bins_visited}")

    driving_hr = day.distance_km / cfg.truck_speed_kmph
    service_hr = day.bins_visited * cfg.service_time_min_per_bin / MINUTES_PER_HOUR
    travel_time_hr = driving_hr + service_hr

    fuel_l = cfg.fuel_l_per_km * day.distance_km
    co2_kg = cfg.co2_kg_per_l * fuel_l
    variable_cost = (
        fuel_l * cfg.fuel_cost_per_l
        + cfg.maintenance_cost_per_km * day.distance_km
        + cfg.labor_cost_per_hr * travel_time_hr
    )

    return DayCost(
        distance_km=day.distance_km,
        bins_visited=day.bins_visited,
        travel_time_hr=travel_time_hr,
        fuel_l=fuel_l,
        co2_kg=co2_kg,
        variable_cost=variable_cost,
        rolled=day.rolled,
    )


@dataclass(frozen=True)
class CostTotals:
    """Aggregated cost over a whole run, on all three fleet bases."""

    distance_km: float
    travel_time_hr: float
    fuel_l: float
    co2_kg: float
    variable_cost: float
    rolling_truck_days: int
    total_cost_rolling: float
    total_cost_fixed_fleet: float


def aggregate(
    days: list[DayCost],
    cfg: CostConfig,
    *,
    fleet_truck_days: int,
) -> CostTotals:
    """Sum per-truck-day costs and apply the fixed cost on both fleet bases.

    Args:
        days: every truck-day of the run, including days a truck stayed parked.
        cfg: supplies ``fixed_cost_per_day``.
        fleet_truck_days: ``n_clusters * days`` -- the baseline fleet's truck-days, used for the
            fixed-fleet basis. Passed in rather than inferred from ``len(days)`` so the basis
            stays the *baseline* fleet even for a strategy that never uses every cluster.
    """
    if fleet_truck_days < 0:
        raise ValueError(f"fleet_truck_days must be >= 0, got {fleet_truck_days}")

    variable = sum(d.variable_cost for d in days)
    rolling_truck_days = sum(1 for d in days if d.rolled)

    return CostTotals(
        distance_km=sum(d.distance_km for d in days),
        travel_time_hr=sum(d.travel_time_hr for d in days),
        fuel_l=sum(d.fuel_l for d in days),
        co2_kg=sum(d.co2_kg for d in days),
        variable_cost=variable,
        rolling_truck_days=rolling_truck_days,
        total_cost_rolling=variable + cfg.fixed_cost_per_day * rolling_truck_days,
        total_cost_fixed_fleet=variable + cfg.fixed_cost_per_day * fleet_truck_days,
    )


def percent_reduction(baseline: float, improved: float) -> float:
    """Percentage drop from ``baseline`` to ``improved``.

    Negative when the "improved" value is worse, which is reported as-is rather than clamped --
    a strategy that costs more than the baseline must look like it costs more.
    """
    if baseline == 0:
        return 0.0
    return 100.0 * (baseline - improved) / baseline
