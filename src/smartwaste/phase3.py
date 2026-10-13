"""Phase 3 benchmark: ten strategies, two demand scenarios, two scoring windows.

One command (``scripts/run_phase3_benchmark.py``) produces every strategy table in
``docs/PHASE3_RESULTS.md``.

### Why this is a separate module from :mod:`smartwaste.benchmark`

Phase 2's ``run_city`` and ``build_strategies`` are pinned by their own tests to exactly five
strategies, and ``docs/PHASE2_RESULTS.md`` is a published document whose numbers must stay
reproducible. So Phase 3 adds a parallel entry point rather than widening Phase 2's. Running
``scripts/run_benchmark.py`` still yields precisely the Phase 2 tables, which is a guarantee
rather than a test.

### The two scoring windows

The forecast-driven strategies spend days ``0..warmup_days-1`` on their backstop schedule while
the model warms up. A 30-day total would therefore credit (f) with 14 days in which it was
simply a calendar. So:

- the **headline window** is days ``warmup_days..end``, and **every** strategy -- the calendars,
  the paper's fusion, all of them -- is scored on that identical window. Every (f)-vs-anything
  claim is made here and nowhere else;
- the **full horizon** is reported separately and labelled, so Phase 2's tables line up next to
  it.

### Pairing each (f) with the calendar of its own period

``(f, backstop=7)`` collects far less often than a 3-day calendar, so measuring it against (b)
would confuse "forecasting works" with "visiting less often is cheaper". The 7-day sensor-free
calendar ``b7`` exists for that reason and :func:`forecast_verdict` compares like with like.

### Output hygiene

No timestamps, no wall-clock timings and no library versions are printed, so two runs at the same
seed are byte-identical under ``fc``. The measured runtime goes in the results document only.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

from smartwaste.city_sim import OutcomeCounters, build_layout, run_strategy
from smartwaste.config import Config
from smartwaste.cost import CostTotals, aggregate, cost_truck_day, percent_reduction
from smartwaste.demand import SCENARIO_HETEROGENEOUS, SCENARIO_SMOOTH
from smartwaste.routing.cluster import kmeans
from smartwaste.seeds import child_seed
from smartwaste.strategies import ForecastDriven, build_phase3_strategies

#: Scored on both demand processes, always both, so the Phase 2 null result is retested rather
#: than quietly replaced.
SCENARIOS_REPORTED = (SCENARIO_SMOOTH, SCENARIO_HETEROGENEOUS)

BASELINE_KEY = "a_fixed_route"
PAPER_KEY = "d_paper_fusion"

#: Phase 2's five, which also have full-horizon numbers worth comparing against Phase 2.
PHASE2_KEYS = (
    "a_fixed_route",
    "b_fixed_schedule",
    "c_single_ultrasonic",
    "d_paper_fusion",
    "e_median_fusion",
)

WINDOW_LABEL = "post-warm-up window"
FULL_LABEL = "full horizon"

CSV_FIELDS = (
    "scenario",
    "fault_fraction",
    "seed",
    "window",
    "window_start_day",
    "strategy_key",
    "strategy_label",
    "distance_km",
    "travel_time_hr",
    "fuel_l",
    "co2_kg",
    "variable_cost",
    "total_cost_rolling",
    "total_cost_fixed_fleet",
    "bin_visits",
    "missed_pickups",
    "overflow_events",
    "overflow_bin_days",
    "overflowed_bins",
    "true_red_bin_days",
    "overflow_events_per_bin_day",
    "training_rows",
)


@dataclass(frozen=True)
class Phase3Result:
    """One strategy, one city, one scoring window."""

    scenario: str
    seed: int
    fault_fraction: float
    key: str
    label: str
    #: ``WINDOW_LABEL`` or ``FULL_LABEL``.
    window: str
    window_start_day: int
    n_bins: int
    scored_days: int
    totals: CostTotals
    counters: OutcomeCounters
    #: Rows the forecaster actually trained on, or ``None`` for a strategy that does not forecast.
    training_rows: int | None = None

    def cost_on(self, basis: str) -> float:
        if basis == "variable":
            return self.totals.variable_cost
        if basis == "rolling":
            return self.totals.total_cost_rolling
        if basis == "fixed_fleet":
            return self.totals.total_cost_fixed_fleet
        raise ValueError(f"unknown cost basis: {basis!r}")

    @property
    def overflow_events_per_bin_day(self) -> float:
        """Overflow events normalised by bin-days, so the two windows are comparable.

        The headline window is 16 days and the full horizon 30, so raw event counts cannot be
        read across them. This is the rate the prompt asks for.
        """
        bin_days = self.n_bins * self.scored_days
        return self.counters.overflow_events / bin_days if bin_days else 0.0


def run_city_phase3(
    cfg: Config,
    seed: int,
    fault_fraction: float,
    *,
    scenario: str,
    n_bins: int,
    days: int,
    n_clusters: int,
    include_gbm: bool = True,
) -> list[Phase3Result]:
    """Run all Phase 3 strategies over one city, scoring each on both windows.

    The layout and the clustering are built once and shared, so no strategy gets easier
    geography or different arrivals than another -- Phase 2's fairness invariant, unchanged.
    """
    layout = build_layout(
        cfg,
        seed,
        n_bins=n_bins,
        days=days,
        stuck_bin_fraction=fault_fraction,
        scenario=scenario,
    )
    clusters = kmeans(layout.locations, n_clusters, random.Random(child_seed(seed, "clusters")))
    warmup = min(cfg.forecast.warmup_days, days - 1) if days > 1 else 0
    fleet_truck_days_full = clusters.n_clusters * days
    fleet_truck_days_window = clusters.n_clusters * (days - warmup)

    results: list[Phase3Result] = []
    for strategy in build_phase3_strategies(cfg, seed, include_gbm=include_gbm):
        run = run_strategy(cfg, layout, clusters, strategy, seed, window_start_day=warmup)
        training_rows = strategy.training_rows if isinstance(strategy, ForecastDriven) else None

        full_costed = [cost_truck_day(d, cfg.cost) for d in run.truck_days]
        results.append(
            Phase3Result(
                scenario=scenario,
                seed=seed,
                fault_fraction=fault_fraction,
                key=run.key,
                label=run.label,
                window=FULL_LABEL,
                window_start_day=0,
                n_bins=n_bins,
                scored_days=days,
                totals=aggregate(full_costed, cfg.cost, fleet_truck_days=fleet_truck_days_full),
                counters=run.counters,
                training_rows=training_rows,
            )
        )

        window_counters = run.window_counters
        if window_counters is not None:
            window_costed = [cost_truck_day(d, cfg.cost) for d in run.window_truck_days]
            results.append(
                Phase3Result(
                    scenario=scenario,
                    seed=seed,
                    fault_fraction=fault_fraction,
                    key=run.key,
                    label=run.label,
                    window=WINDOW_LABEL,
                    window_start_day=warmup,
                    n_bins=n_bins,
                    scored_days=days - warmup,
                    totals=aggregate(
                        window_costed, cfg.cost, fleet_truck_days=fleet_truck_days_window
                    ),
                    counters=window_counters,
                    training_rows=training_rows,
                )
            )
    return results


def run_phase3_benchmark(
    cfg: Config,
    seeds: Sequence[int],
    fault_fractions: Sequence[float],
    scenarios: Sequence[str],
    *,
    n_bins: int,
    days: int,
    n_clusters: int,
    include_gbm: bool = True,
) -> list[Phase3Result]:
    """Every strategy, on every scenario, seed and fault fraction. Deterministic in this order."""
    out: list[Phase3Result] = []
    for scenario in scenarios:
        for fault_fraction in fault_fractions:
            for seed in seeds:
                out.extend(
                    run_city_phase3(
                        cfg,
                        seed,
                        fault_fraction,
                        scenario=scenario,
                        n_bins=n_bins,
                        days=days,
                        n_clusters=n_clusters,
                        include_gbm=include_gbm,
                    )
                )
    return out


def select(
    results: Sequence[Phase3Result],
    *,
    scenario: str | None = None,
    fault_fraction: float | None = None,
    seed: int | None = None,
    key: str | None = None,
    window: str | None = None,
) -> list[Phase3Result]:
    """Filter results, preserving order."""
    return [
        r
        for r in results
        if (scenario is None or r.scenario == scenario)
        and (fault_fraction is None or r.fault_fraction == fault_fraction)
        and (seed is None or r.seed == seed)
        and (key is None or r.key == key)
        and (window is None or r.window == window)
    ]


# --------------------------------------------------------------------------------------------
# Rendering. No timestamps, no timings: same seed must give byte-identical output.
# --------------------------------------------------------------------------------------------


def _fmt(value: float, places: int = 2) -> str:
    return f"{value:.{places}f}"


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def render_headline(
    results: Sequence[Phase3Result],
    *,
    scenario: str,
    seed: int,
    fault_fraction: float,
    window: str,
    keys: Sequence[str] | None = None,
    quick: bool = False,
) -> str:
    """One row per strategy, for one scenario / seed / fault fraction / window."""
    chosen = select(
        results, scenario=scenario, seed=seed, fault_fraction=fault_fraction, window=window
    )
    if keys is not None:
        allowed = set(keys)
        chosen = [r for r in chosen if r.key in allowed]
    rows = [
        [
            r.label,
            _fmt(r.totals.distance_km),
            _fmt(r.totals.travel_time_hr),
            _fmt(r.totals.co2_kg),
            _fmt(r.cost_on("variable")),
            _fmt(r.cost_on("rolling")),
            _fmt(r.cost_on("fixed_fleet")),
            str(r.counters.bin_visits),
            str(r.counters.missed_pickups),
            str(r.counters.overflow_events),
            str(r.counters.overflow_bin_days),
            str(r.counters.overflowed_bins),
            _fmt(r.overflow_events_per_bin_day, 6),
        ]
        for r in chosen
    ]
    header = (
        "strategy",
        "distance km",
        "time hr",
        "CO2 kg",
        "cost (variable)",
        "cost (rolling fleet)",
        "cost (fixed fleet)",
        "bin-visits",
        "missed pickups",
        "overflow events",
        "overflow bin-days",
        "bins overflowed >=1x",
        "overflow events / bin-day",
    )
    scope = " -- SMOKE RUN, NOT FOR PUBLICATION" if quick else ""
    days_note = ""
    if chosen:
        first_day = chosen[0].window_start_day
        last_day = first_day + chosen[0].scored_days - 1
        days_note = f", days {first_day}..{last_day}"
    title = (
        f"### Headline [{window}{days_note}]: demand `{scenario}`, seed {seed}, "
        f"stuck-sensor fraction {fault_fraction:g}{scope}"
    )
    return f"{title}\n\n{_table(header, rows)}"


def render_reductions(
    results: Sequence[Phase3Result],
    *,
    scenario: str,
    seed: int,
    fault_fraction: float,
    window: str,
    quick: bool = False,
) -> str:
    """Percentage reductions against (a), with the cost basis named on every column."""
    chosen = select(
        results, scenario=scenario, seed=seed, fault_fraction=fault_fraction, window=window
    )
    baseline = next((r for r in chosen if r.key == BASELINE_KEY), None)
    if baseline is None:
        return ""
    rows = []
    for r in chosen:
        if r.key == BASELINE_KEY:
            continue
        rows.append(
            [
                r.label,
                _fmt(percent_reduction(baseline.totals.distance_km, r.totals.distance_km)),
                _fmt(percent_reduction(baseline.totals.travel_time_hr, r.totals.travel_time_hr)),
                _fmt(percent_reduction(baseline.totals.co2_kg, r.totals.co2_kg)),
                _fmt(percent_reduction(baseline.cost_on("variable"), r.cost_on("variable"))),
                _fmt(percent_reduction(baseline.cost_on("rolling"), r.cost_on("rolling"))),
                _fmt(percent_reduction(baseline.cost_on("fixed_fleet"), r.cost_on("fixed_fleet"))),
            ]
        )
    header = (
        "strategy",
        "distance %",
        "travel time %",
        "CO2 %",
        "cost % (variable basis)",
        "cost % (rolling-fleet basis)",
        "cost % (fixed-fleet basis)",
    )
    scope = " -- SMOKE RUN, NOT FOR PUBLICATION" if quick else ""
    title = (
        f"### Reduction vs (a) fixed-route [{window}]: demand `{scenario}`, seed {seed}, "
        f"stuck-sensor fraction {fault_fraction:g}{scope}"
    )
    note = (
        "Positive = better than the baseline. Negative values are printed as-is: a strategy that\n"
        "costs more than the baseline must look like it costs more. Each cost column names the\n"
        "fleet basis it used."
    )
    return f"{title}\n\n{note}\n\n{_table(header, rows)}"


def forecast_verdict(
    results: Sequence[Phase3Result],
    scenarios: Sequence[str],
    fault_fractions: Sequence[float],
) -> list[str]:
    """Does forecasting earn its keep -- against its OWN calendar, and against the MA baseline?

    Two comparisons per backstop, both stated plainly:

    1. the ML (f) against the sensor-free calendar of the same period, and
    2. the ML (f) against the moving-average (f) at the same backstop.

    CLAUDE.md requires that a simple baseline matching or beating the ML model be reported
    plainly, so this is computed and printed rather than left to the reader.
    """
    lines: list[str] = []
    for scenario in scenarios:
        for fault_fraction in fault_fractions:
            for backstop in (3, 7):
                calendar_key = (
                    "b_fixed_schedule" if backstop == 3 else f"b{backstop}_fixed_schedule"
                )
                pairs = (
                    (f"f_gbm_{backstop}", calendar_key, "the sensor-free calendar"),
                    (f"f_gbm_{backstop}", f"f_ma_{backstop}", "the moving-average baseline"),
                )
                for ml_key, other_key, other_name in pairs:
                    ml = _mean_window(results, scenario, fault_fraction, ml_key)
                    other = _mean_window(results, scenario, fault_fraction, other_key)
                    if ml is None or other is None:
                        continue
                    ml_cost, ml_ovf = ml
                    other_cost, other_ovf = other
                    cheaper = ml_cost <= other_cost
                    safer = ml_ovf <= other_ovf
                    if cheaper and safer:
                        note = f"BEATS {other_name} on both cost and overflow"
                    elif cheaper:
                        note = f"is cheaper than {other_name} but overflows more"
                    elif safer:
                        note = f"overflows less than {other_name} but costs more"
                    else:
                        note = (
                            f"is WORSE than {other_name} on both cost and overflow -- "
                            f"the ML model does not earn its keep here"
                        )
                    lines.append(
                        f"  {scenario}, fault {fault_fraction:g}, backstop {backstop}d: "
                        f"ML forecast {note} "
                        f"(cost {ml_cost:.0f} vs {other_cost:.0f} variable basis; "
                        f"overflow events {ml_ovf:.1f} vs {other_ovf:.1f})"
                    )
    return lines


def calendar_verdict(
    results: Sequence[Phase3Result],
    scenarios: Sequence[str],
    fault_fractions: Sequence[float],
) -> list[str]:
    """Does Phase 2's finding survive? Is a dumb calendar still competitive with the sensors?

    Phase 2 concluded that (b), with no sensors at all, matched the paper's fusion on cost with
    zero overflow. Phase 3 asks the same question under demand that actually varies, and prints
    the answer whichever way it falls.
    """
    lines: list[str] = []
    for scenario in scenarios:
        for fault_fraction in fault_fractions:
            paper = _mean_window(results, scenario, fault_fraction, PAPER_KEY)
            if paper is None:
                continue
            paper_cost, paper_ovf = paper
            for key in ("b_fixed_schedule", "b7_fixed_schedule"):
                other = _mean_window(results, scenario, fault_fraction, key)
                if other is None:
                    continue
                cost, ovf = other
                cheaper = cost <= paper_cost
                safer = ovf <= paper_ovf
                if cheaper and safer:
                    note = "MATCHES OR BEATS the paper's method on BOTH cost and overflow"
                elif cheaper:
                    note = "is CHEAPER than the paper's method but overflows more"
                elif safer:
                    note = "overflows no more than the paper's method but costs more"
                else:
                    note = "is worse than the paper's method on both cost and overflow"
                lines.append(
                    f"  {scenario}, fault {fault_fraction:g}: {key} {note} "
                    f"(cost {cost:.0f} vs {paper_cost:.0f} variable basis; "
                    f"overflow events {ovf:.1f} vs {paper_ovf:.1f})"
                )
    return lines


def _mean_window(
    results: Sequence[Phase3Result], scenario: str, fault_fraction: float, key: str
) -> tuple[float, float] | None:
    """Mean variable cost and overflow events over seeds, on the headline window only."""
    runs = select(
        results,
        scenario=scenario,
        fault_fraction=fault_fraction,
        key=key,
        window=WINDOW_LABEL,
    )
    if not runs:
        return None
    cost = sum(r.cost_on("variable") for r in runs) / len(runs)
    overflow = sum(r.counters.overflow_events for r in runs) / len(runs)
    return cost, overflow


def render_training_rows(results: Sequence[Phase3Result]) -> str:
    """How many rows each forecaster actually trained on.

    Reported because the warm-up is deliberately short (14 days) and the honest reading of a
    weak ML result is partly "it had very little to learn from". No constant was changed to
    improve this number.
    """
    seen: dict[tuple[str, int, str], int] = {}
    for r in results:
        if r.training_rows is None or r.window != WINDOW_LABEL:
            continue
        seen[(r.scenario, r.seed, r.key)] = r.training_rows
    rows = [
        [scenario, str(seed), key, str(count)]
        for (scenario, seed, key), count in sorted(seen.items())
    ]
    header = ("scenario", "seed", "strategy", "training rows used")
    title = "### Forecaster training rows (a limitation, not a result)"
    note = (
        "`forecast.warmup_days = 14`, so each model is fitted on a short history and then frozen.\n"
        "Rows whose span contained a collection are dropped, because the fill drop across a\n"
        "collection measures the truck rather than demand."
    )
    return f"{title}\n\n{note}\n\n{_table(header, rows)}"


def csv_rows(results: Sequence[Phase3Result]) -> list[dict[str, object]]:
    """Flatten for ``csv.DictWriter`` using :data:`CSV_FIELDS`."""
    return [
        {
            "scenario": r.scenario,
            "fault_fraction": f"{r.fault_fraction:g}",
            "seed": r.seed,
            "window": r.window,
            "window_start_day": r.window_start_day,
            "strategy_key": r.key,
            "strategy_label": r.label,
            "distance_km": _fmt(r.totals.distance_km),
            "travel_time_hr": _fmt(r.totals.travel_time_hr),
            "fuel_l": _fmt(r.totals.fuel_l),
            "co2_kg": _fmt(r.totals.co2_kg),
            "variable_cost": _fmt(r.totals.variable_cost),
            "total_cost_rolling": _fmt(r.totals.total_cost_rolling),
            "total_cost_fixed_fleet": _fmt(r.totals.total_cost_fixed_fleet),
            "bin_visits": r.counters.bin_visits,
            "missed_pickups": r.counters.missed_pickups,
            "overflow_events": r.counters.overflow_events,
            "overflow_bin_days": r.counters.overflow_bin_days,
            "overflowed_bins": r.counters.overflowed_bins,
            "true_red_bin_days": r.counters.true_red_bin_days,
            "overflow_events_per_bin_day": _fmt(r.overflow_events_per_bin_day, 6),
            "training_rows": "" if r.training_rows is None else r.training_rows,
        }
        for r in results
    ]
