"""Waste arrival processes: the Phase 2 smooth one, and Phase 3's heterogeneous one.

### Why this module exists

Phase 2 reached a null result and reported it: strategy (b) -- every bin every 3 days, no sensors
at all -- came within 0.75 pp of the paper's sensor fusion on cost with **zero** overflow events.
``docs/PHASE2_RESULTS.md`` names the cause: arrivals were i.i.d. Gaussian with mean 4.0 kg and
sd 1.5 kg, so every bin filled at very nearly the same predictable rate. That is the friendliest
possible world for a fixed calendar, and it leaves a forecaster nothing to learn.

So Phase 3 fixes the *experiment* before adding any model. This module supplies demand that
actually varies, along three axes:

1. **Area type** -- a market bin fills far faster than a residential one.
2. **Day of week** -- weekends are busier than weekdays.
3. **Festival spikes** -- a few days a month with a large multiplier.

### Everything here is INVENTED and UNCITED

The paper models arrivals as ``W_i(t) = W_i(t-1) + dW_i(t)`` with "random daily increments" and
states no distribution, no heterogeneity and no calendar. Every constant in the ``demand`` section
of ``config/default.yaml`` is a modelling choice of ours with **no field data and no citation**
behind it -- see ``docs/ASSUMPTIONS.md``. Results that depend on this process hold **on our
invented demand only**, and the results doc says so.

### Total volume is held constant on purpose

``demand.area_mean_kg`` is scaled so the long-run mean daily arrival per bin matches Phase 2's
``sim.daily_fill_mean_kg`` to within a few percent, *after* the weekday and festival multipliers
are taken into account. Without that, a heterogeneous city would simply make more waste than the
smooth one, and every Phase 3 number would improve for a reason that has nothing to do with
sensors or forecasting. ``tests/test_demand.py`` measures the realised mean and pins it.

### The smooth scenario is bit-for-bit the Phase 2 one

:func:`generate_increments` under ``"smooth"`` reproduces Phase 2's code path exactly -- same seed
label ``"fill:<bin>"``, same per-bin stream, same draw order -- so the Phase 2 tables stay
reproducible. New scenarios use new seed labels only, never the smooth one.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from smartwaste.config import Config, DemandConfig
from smartwaste.seeds import child_seed

SCENARIO_SMOOTH = "smooth"
SCENARIO_HETEROGENEOUS = "heterogeneous"
SCENARIOS = (SCENARIO_SMOOTH, SCENARIO_HETEROGENEOUS)

#: Area type names, in the order their weights and means appear in the config lists.
#: Invented, uncited: three broad categories, not drawn from any survey.
AREA_TYPE_NAMES = ("market", "residential", "apartment")

#: The neutral area type the smooth scenario reports, so Phase 2 code paths see one uniform city.
SMOOTH_AREA_TYPE = 0

DAYS_PER_WEEK = 7

#: Festival rates in the config are "per month"; this is the month length they are quoted against.
#: The paper's horizon is 30 days, so the two line up.
DAYS_PER_MONTH = 30


def day_of_week(day: int) -> int:
    """Day index -> weekday, with day 0 taken to be a Monday.

    Which real weekday day 0 lands on is arbitrary and uncited; fixing it at Monday keeps the
    weekend in the same place for every seed, so a seed-to-seed difference is never just a
    calendar shift.
    """
    return day % DAYS_PER_WEEK


def weekday_profile(days: int) -> tuple[int, ...]:
    """The weekday of each day in the horizon."""
    return tuple(day_of_week(day) for day in range(days))


def pick_festival_days(cfg: DemandConfig, seed: int, days: int) -> tuple[bool, ...]:
    """Choose which days are festival days, from a dedicated random stream.

    Its own seed label, so changing the festival rate cannot reshuffle bin locations, densities
    or the sensor fault draw -- the same controlled-comparison discipline Phase 2 applied to
    ``stuck_bin_fraction``.
    """
    n_festivals = round(days * cfg.festival_days_per_month / DAYS_PER_MONTH)
    n_festivals = max(0, min(days, n_festivals))
    flags = [False] * days
    if n_festivals:
        rng = random.Random(child_seed(seed, "festival"))
        for day in rng.sample(range(days), n_festivals):
            flags[day] = True
    return tuple(flags)


def assign_area_types(cfg: DemandConfig, seed: int, n_bins: int) -> tuple[int, ...]:
    """Draw each bin's area type from ``demand.area_type_weights``.

    Its own seed label for the same reason as the festival draw.
    """
    rng = random.Random(child_seed(seed, "area"))
    indices = list(range(len(cfg.area_type_weights)))
    weights = list(cfg.area_type_weights)
    return tuple(rng.choices(indices, weights=weights, k=n_bins))


def arrival_mean_kg(
    cfg: DemandConfig,
    area_type: int,
    weekday: int,
    is_festival: bool,
) -> float:
    """Mean daily arrival for one bin on one day: area x weekday x festival.

    A plain product of three invented multipliers. Deliberately simple: a reader can work out
    what any bin-day's mean is by hand, which matters more here than realism we cannot source.
    """
    mean = cfg.area_mean_kg[area_type] * cfg.weekday_multipliers[weekday]
    if is_festival:
        mean *= cfg.festival_multiplier
    return mean


@dataclass(frozen=True)
class DemandCalendar:
    """The per-bin and per-day structure a heterogeneous city is generated from.

    The smooth scenario fills this in neutrally (one area type, no festivals) so that downstream
    code -- forecast features in particular -- has the same shape under both scenarios.
    """

    scenario: str
    area_type: tuple[int, ...]
    weekday: tuple[int, ...]
    is_festival: tuple[bool, ...]

    @property
    def n_festival_days(self) -> int:
        return sum(self.is_festival)

    def festival_days_in(self, start_day: int, end_day: int) -> int:
        """Festival days in the half-open window ``[start_day, end_day)``.

        Used to report how much seasonality the forecaster's warm-up actually contained, which is
        a limitation Phase 3 states rather than fixes.
        """
        lo = max(0, start_day)
        hi = min(len(self.is_festival), end_day)
        return sum(1 for day in range(lo, hi) if self.is_festival[day])

    def weekdays_in(self, start_day: int, end_day: int) -> int:
        """Distinct weekdays covered by ``[start_day, end_day)``."""
        lo = max(0, start_day)
        hi = min(len(self.weekday), end_day)
        return len({self.weekday[day] for day in range(lo, hi)})


def build_calendar(
    cfg: Config, seed: int, *, n_bins: int, days: int, scenario: str
) -> DemandCalendar:
    """Area types, weekdays and festival days for one city."""
    if scenario not in SCENARIOS:
        raise ValueError(f"scenario must be one of {SCENARIOS}, got {scenario!r}")
    if scenario == SCENARIO_SMOOTH:
        # One uniform area type and no festivals: the smooth city has no structure to report.
        return DemandCalendar(
            scenario=scenario,
            area_type=(SMOOTH_AREA_TYPE,) * n_bins,
            weekday=weekday_profile(days),
            is_festival=(False,) * days,
        )
    return DemandCalendar(
        scenario=scenario,
        area_type=assign_area_types(cfg.demand, seed, n_bins),
        weekday=weekday_profile(days),
        is_festival=pick_festival_days(cfg.demand, seed, days),
    )


def generate_increments(
    cfg: Config,
    seed: int,
    *,
    n_bins: int,
    days: int,
    scenario: str,
    calendar: DemandCalendar,
) -> tuple[tuple[float, ...], ...]:
    """``increments[bin][day]`` kg of waste arriving. Pre-generated, decision-independent.

    One random stream per bin, keyed on the bin index, so the arrivals of bin 7 never depend on
    how many bins precede it or on what any strategy decided. This is the Phase 2 invariant that
    makes the strategy comparison fair, and it is unchanged here.
    """
    if scenario == SCENARIO_SMOOTH:
        return _smooth_increments(cfg, seed, n_bins=n_bins, days=days)
    if scenario == SCENARIO_HETEROGENEOUS:
        return _heterogeneous_increments(cfg, seed, n_bins=n_bins, days=days, calendar=calendar)
    raise ValueError(f"scenario must be one of {SCENARIOS}, got {scenario!r}")


def _smooth_increments(
    cfg: Config, seed: int, *, n_bins: int, days: int
) -> tuple[tuple[float, ...], ...]:
    """Phase 2's arrival process, character for character.

    **Do not "improve" this function.** The seed label ``"fill:<bin>"``, the ``gauss(mean, sd)``
    call and the ``max(0.0, ...)`` floor together define every number in
    ``docs/PHASE2_RESULTS.md``. Any change to the stream silently invalidates that document.
    """
    mean_kg = cfg.sim.daily_fill_mean_kg
    sd_kg = cfg.sim.daily_fill_sd_kg
    increments: list[tuple[float, ...]] = []
    for bin_index in range(n_bins):
        fill_rng = random.Random(child_seed(seed, f"fill:{bin_index}"))
        increments.append(tuple(max(0.0, fill_rng.gauss(mean_kg, sd_kg)) for _ in range(days)))
    return tuple(increments)


def _heterogeneous_increments(
    cfg: Config,
    seed: int,
    *,
    n_bins: int,
    days: int,
    calendar: DemandCalendar,
) -> tuple[tuple[float, ...], ...]:
    """Area type x weekday x festival, with a per-bin-day coefficient of variation.

    The spread is proportional to the mean (``sd = mean * cv``) rather than absolute, so a market
    bin is noisy in kg and a residential bin is not -- which is the heteroscedasticity a fixed
    calendar cannot exploit and a forecaster might.

    A **new** seed label (``"fill_het:<bin>"``) so the smooth stream is untouched.
    """
    demand = cfg.demand
    increments: list[tuple[float, ...]] = []
    for bin_index in range(n_bins):
        fill_rng = random.Random(child_seed(seed, f"fill_het:{bin_index}"))
        area_type = calendar.area_type[bin_index]
        row: list[float] = []
        for day in range(days):
            mean = arrival_mean_kg(
                demand,
                area_type,
                calendar.weekday[day],
                calendar.is_festival[day],
            )
            row.append(max(0.0, fill_rng.gauss(mean, mean * demand.cv)))
        increments.append(tuple(row))
    return tuple(increments)


def expected_mean_daily_arrival(cfg: DemandConfig, days: int = DAYS_PER_MONTH) -> float:
    """The analytic long-run mean daily arrival per bin under the heterogeneous scenario.

    Only used to document and sanity-check the volume-matching described in the module docstring.
    The test that actually guards it **measures** the realised mean from generated increments
    instead of trusting this formula.
    """
    base = sum(
        weight * mean for weight, mean in zip(cfg.area_type_weights, cfg.area_mean_kg, strict=True)
    )
    weekday_factor = sum(cfg.weekday_multipliers) / DAYS_PER_WEEK
    n_festivals = max(0, min(days, round(days * cfg.festival_days_per_month / DAYS_PER_MONTH)))
    festival_factor = ((days - n_festivals) + n_festivals * cfg.festival_multiplier) / days
    return base * weekday_factor * festival_factor
