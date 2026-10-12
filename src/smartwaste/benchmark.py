"""Phase 2 benchmark: run every strategy over the city and tabulate the consequences.

One command (``scripts/run_benchmark.py``) produces every table in ``docs/PHASE2_RESULTS.md``.

### What is compared, and on what basis

Five strategies (see :mod:`smartwaste.strategies`) at each of several stuck-sensor fractions, over
several seeds. Each ``(seed, fault_fraction)`` pair is one city, and all five strategies run on
that same city with the same waste arrivals and the same sensor noise -- only their decisions
differ. See :mod:`smartwaste.city_sim` for how that is enforced.

### Reporting rules this module enforces

- Spread across seeds is reported as **mean and min-max**, labelled "range over N seeds". It is
  **not** a confidence interval and is never called one: 5 seeds is a range, not an inference.
- Every percentage reduction carries the **cost basis** it was computed on
  (``variable``, ``rolling`` or ``fixed_fleet``), because the fixed-cost fleet assumption moves
  that number and hiding which one was used would be misleading.
- Output carries no timestamps and no timings, so two runs at the same seed are byte-identical.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

from smartwaste.city_sim import build_layout, run_strategy
from smartwaste.config import Config
from smartwaste.cost import CostTotals, aggregate, cost_truck_day, percent_reduction
from smartwaste.routing.cluster import kmeans
from smartwaste.seeds import child_seed
from smartwaste.strategies import build_strategies

#: The strategy whose numbers every saving is measured against: the paper's naive baseline.
BASELINE_KEY = "a_fixed_route"

#: The paper's own method, and our extension. The docs compare these two head to head.
PAPER_KEY = "d_paper_fusion"
EXTENSION_KEY = "e_median_fusion"

#: The three cost bases. Every reported percentage names one of these.
COST_BASES = ("variable", "rolling", "fixed_fleet")

CSV_FIELDS = (
    "fault_fraction",
    "seed",
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
    "rolling_truck_days",
)


@dataclass(frozen=True)
class StrategyResult:
    """One strategy's outcome on one city (one seed, one fault fraction)."""

    seed: int
    fault_fraction: float
    key: str
    label: str
    totals: CostTotals
    bin_visits: int
    missed_pickups: int
    overflow_events: int
    overflow_bin_days: int
    overflowed_bins: int
    true_red_bin_days: int

    def cost_on(self, basis: str) -> float:
        """This run's cost on one of the three fleet bases."""
        if basis == "variable":
            return self.totals.variable_cost
        if basis == "rolling":
            return self.totals.total_cost_rolling
        if basis == "fixed_fleet":
            return self.totals.total_cost_fixed_fleet
        raise ValueError(f"unknown cost basis: {basis!r}")


def run_city(
    cfg: Config,
    seed: int,
    fault_fraction: float,
    *,
    n_bins: int,
    days: int,
    n_clusters: int,
) -> list[StrategyResult]:
    """Run all five strategies over one city and cost every truck-day.

    The layout and the clustering are built once and shared by all five strategies, so no
    strategy gets easier geography than another.
    """
    layout = build_layout(cfg, seed, n_bins=n_bins, days=days, stuck_bin_fraction=fault_fraction)
    clusters = kmeans(layout.locations, n_clusters, random.Random(child_seed(seed, "clusters")))
    # The fixed-fleet cost basis always uses the BASELINE fleet, so a strategy that leaves a
    # cluster idle all month is still charged as if the city owned every truck.
    fleet_truck_days = clusters.n_clusters * days

    results: list[StrategyResult] = []
    for strategy in build_strategies(cfg.bin, cfg.benchmark.fixed_schedule_days):
        run = run_strategy(cfg, layout, clusters, strategy, seed)
        costed = [cost_truck_day(d, cfg.cost) for d in run.truck_days]
        totals = aggregate(costed, cfg.cost, fleet_truck_days=fleet_truck_days)
        counters = run.counters
        results.append(
            StrategyResult(
                seed=seed,
                fault_fraction=fault_fraction,
                key=run.key,
                label=run.label,
                totals=totals,
                bin_visits=counters.bin_visits,
                missed_pickups=counters.missed_pickups,
                overflow_events=counters.overflow_events,
                overflow_bin_days=counters.overflow_bin_days,
                overflowed_bins=counters.overflowed_bins,
                true_red_bin_days=counters.true_red_bin_days,
            )
        )
    return results


def run_benchmark(
    cfg: Config,
    seeds: Sequence[int],
    fault_fractions: Sequence[float],
    *,
    n_bins: int,
    days: int,
    n_clusters: int,
) -> list[StrategyResult]:
    """Every strategy, on every seed, at every fault fraction. Deterministic in this order."""
    out: list[StrategyResult] = []
    for fault_fraction in fault_fractions:
        for seed in seeds:
            out.extend(
                run_city(
                    cfg,
                    seed,
                    fault_fraction,
                    n_bins=n_bins,
                    days=days,
                    n_clusters=n_clusters,
                )
            )
    return out


def select(
    results: Sequence[StrategyResult],
    *,
    fault_fraction: float | None = None,
    seed: int | None = None,
    key: str | None = None,
) -> list[StrategyResult]:
    """Filter results, preserving order."""
    return [
        r
        for r in results
        if (fault_fraction is None or r.fault_fraction == fault_fraction)
        and (seed is None or r.seed == seed)
        and (key is None or r.key == key)
    ]


# --------------------------------------------------------------------------------------------
# Rendering. No timestamps, no timings: same seed must give byte-identical output.
# --------------------------------------------------------------------------------------------


def _fmt(value: float, places: int = 2) -> str:
    """Format a float with a fixed number of places, so output cannot drift between runs."""
    return f"{value:.{places}f}"


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def render_headline(results: Sequence[StrategyResult], seed: int, fault_fraction: float) -> str:
    """The main table: one row per strategy at one seed and one fault fraction."""
    rows = [
        [
            r.label,
            _fmt(r.totals.distance_km),
            _fmt(r.totals.travel_time_hr),
            _fmt(r.totals.co2_kg),
            _fmt(r.totals.variable_cost),
            _fmt(r.totals.total_cost_rolling),
            _fmt(r.totals.total_cost_fixed_fleet),
            str(r.bin_visits),
            str(r.missed_pickups),
            str(r.overflow_events),
            str(r.overflow_bin_days),
            str(r.overflowed_bins),
        ]
        for r in select(results, seed=seed, fault_fraction=fault_fraction)
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
    )
    title = f"### Headline: all strategies, seed {seed}, stuck-sensor fraction {fault_fraction:g}"
    return f"{title}\n\n{_table(header, rows)}"


def render_reductions(results: Sequence[StrategyResult], seed: int, fault_fraction: float) -> str:
    """Percentage reductions against the baseline, with the cost basis named on every column."""
    chosen = select(results, seed=seed, fault_fraction=fault_fraction)
    baseline = next(r for r in chosen if r.key == BASELINE_KEY)
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
    title = (
        f"### Reduction vs (a) fixed-route, seed {seed}, stuck-sensor fraction {fault_fraction:g}"
    )
    note = (
        "Positive = better than the baseline. Negative values are printed as-is: a strategy that\n"
        "costs more than the baseline must look like it costs more. Each cost column names the\n"
        "fleet basis it used."
    )
    return f"{title}\n\n{note}\n\n{_table(header, rows)}"


def render_seed_range(
    results: Sequence[StrategyResult], seeds: Sequence[int], fault_fraction: float
) -> str:
    """Mean and min-max across seeds. Deliberately NOT called a confidence interval."""
    rows = []
    keys_seen: list[str] = []
    for r in select(results, fault_fraction=fault_fraction, seed=seeds[0]):
        keys_seen.append(r.key)
    for key in keys_seen:
        runs = [r for r in results if r.key == key and r.fault_fraction == fault_fraction]
        label = runs[0].label
        cells = [label]
        for values in (
            [r.totals.distance_km for r in runs],
            [r.cost_on("variable") for r in runs],
            [float(r.overflow_events) for r in runs],
            [float(r.overflowed_bins) for r in runs],
        ):
            mean = sum(values) / len(values)
            cells.append(f"{_fmt(mean)} [{_fmt(min(values))}, {_fmt(max(values))}]")
        rows.append(cells)
    header = (
        "strategy",
        "distance km: mean [min, max]",
        "cost (variable): mean [min, max]",
        "overflow events: mean [min, max]",
        "bins overflowed: mean [min, max]",
    )
    title = (
        f"### Range over {len(seeds)} seeds ({min(seeds)}-{max(seeds)}), "
        f"stuck-sensor fraction {fault_fraction:g}"
    )
    note = (
        f"**This is a range over seeds, not a confidence interval.** {len(seeds)} seeds describe\n"
        "the spread we observed; they do not support an inference about a population."
    )
    return f"{title}\n\n{note}\n\n{_table(header, rows)}"


def render_paired_mean_vs_median(
    results: Sequence[StrategyResult],
    seeds: Sequence[int],
    fault_fractions: Sequence[float],
) -> str:
    """Per-seed paired (d) - (e) differences, so a one-seed fluke cannot look like a win."""
    rows = []
    for fault_fraction in fault_fractions:
        for seed in seeds:
            paper = select(results, seed=seed, fault_fraction=fault_fraction, key=PAPER_KEY)
            ext = select(results, seed=seed, fault_fraction=fault_fraction, key=EXTENSION_KEY)
            if not paper or not ext:
                continue
            d, e = paper[0], ext[0]
            rows.append(
                [
                    f"{fault_fraction:g}",
                    str(seed),
                    str(d.overflow_events),
                    str(e.overflow_events),
                    str(d.overflow_events - e.overflow_events),
                    str(d.missed_pickups),
                    str(e.missed_pickups),
                    str(d.missed_pickups - e.missed_pickups),
                    _fmt(d.cost_on("variable") - e.cost_on("variable")),
                ]
            )
    header = (
        "fault fraction",
        "seed",
        "(d) overflow events",
        "(e) overflow events",
        "(d)-(e) overflow",
        "(d) missed",
        "(e) missed",
        "(d)-(e) missed",
        "(d)-(e) cost (variable)",
    )
    title = "### Paired per-seed comparison: (d) paper's MEAN fusion vs (e) our MEDIAN fusion"
    note = (
        "Paired: both strategies ran on the same city, the same waste arrivals and the same\n"
        "sensor noise, so these differences are not confounded by the draw.\n\n"
        "**A positive `(d)-(e)` value means (e) the median was better on that metric.**\n"
        "Negative means the paper's mean was better and the extension did not pay off there."
    )
    return f"{title}\n\n{note}\n\n{_table(header, rows)}"


def mean_vs_median_verdict(
    results: Sequence[StrategyResult], fault_fractions: Sequence[float]
) -> list[str]:
    """One plain-language verdict line per fault fraction. Says so when (e) does not win."""
    lines: list[str] = []
    for fault_fraction in fault_fractions:
        d_runs = [r for r in results if r.key == PAPER_KEY and r.fault_fraction == fault_fraction]
        e_runs = [
            r for r in results if r.key == EXTENSION_KEY and r.fault_fraction == fault_fraction
        ]
        if not d_runs or not e_runs:
            continue
        d_ovf = sum(r.overflow_events for r in d_runs) / len(d_runs)
        e_ovf = sum(r.overflow_events for r in e_runs) / len(e_runs)
        d_missed = sum(r.missed_pickups for r in d_runs) / len(d_runs)
        e_missed = sum(r.missed_pickups for r in e_runs) / len(e_runs)
        wins = sum(
            1 for d, e in zip(d_runs, e_runs, strict=True) if e.overflow_events < d.overflow_events
        )
        if e_ovf < d_ovf:
            verdict = (
                f"(e) median BEATS (d) mean on overflow events "
                f"({e_ovf:.1f} vs {d_ovf:.1f} mean over seeds)"
            )
        elif e_ovf > d_ovf:
            verdict = (
                f"(e) median is WORSE than (d) mean on overflow events "
                f"({e_ovf:.1f} vs {d_ovf:.1f} mean over seeds) -- the extension does not pay off"
            )
        else:
            verdict = f"(e) median TIES (d) mean on overflow events ({e_ovf:.1f} each)"
        lines.append(
            f"  stuck fraction {fault_fraction:g}: {verdict}; "
            f"missed pickups {e_missed:.1f} vs {d_missed:.1f}; "
            f"(e) won on {wins}/{len(d_runs)} seeds"
        )
    return lines


def baseline_beats_paper_verdict(
    results: Sequence[StrategyResult], fault_fractions: Sequence[float]
) -> list[str]:
    """Where a sensor-free or single-sensor baseline matches or beats the paper's method.

    CLAUDE.md requires this to be reported plainly rather than buried, so it is computed here and
    printed to stdout, not left for a reader to derive from the big table.
    """
    lines: list[str] = []
    for fault_fraction in fault_fractions:
        paper = [r for r in results if r.key == PAPER_KEY and r.fault_fraction == fault_fraction]
        if not paper:
            continue
        paper_cost = sum(r.cost_on("variable") for r in paper) / len(paper)
        paper_ovf = sum(r.overflow_events for r in paper) / len(paper)
        for key in ("b_fixed_schedule", "c_single_ultrasonic"):
            runs = [r for r in results if r.key == key and r.fault_fraction == fault_fraction]
            if not runs:
                continue
            cost = sum(r.cost_on("variable") for r in runs) / len(runs)
            ovf = sum(r.overflow_events for r in runs) / len(runs)
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
                f"  stuck fraction {fault_fraction:g}: {runs[0].label} {note} "
                f"(cost {cost:.0f} vs {paper_cost:.0f} variable basis; "
                f"overflow events {ovf:.1f} vs {paper_ovf:.1f})"
            )
    return lines


def csv_rows(results: Sequence[StrategyResult]) -> list[dict[str, object]]:
    """Flatten results for ``csv.DictWriter`` using :data:`CSV_FIELDS`."""
    return [
        {
            "fault_fraction": f"{r.fault_fraction:g}",
            "seed": r.seed,
            "strategy_key": r.key,
            "strategy_label": r.label,
            "distance_km": _fmt(r.totals.distance_km),
            "travel_time_hr": _fmt(r.totals.travel_time_hr),
            "fuel_l": _fmt(r.totals.fuel_l),
            "co2_kg": _fmt(r.totals.co2_kg),
            "variable_cost": _fmt(r.totals.variable_cost),
            "total_cost_rolling": _fmt(r.totals.total_cost_rolling),
            "total_cost_fixed_fleet": _fmt(r.totals.total_cost_fixed_fleet),
            "bin_visits": r.bin_visits,
            "missed_pickups": r.missed_pickups,
            "overflow_events": r.overflow_events,
            "overflow_bin_days": r.overflow_bin_days,
            "overflowed_bins": r.overflowed_bins,
            "true_red_bin_days": r.true_red_bin_days,
            "rolling_truck_days": r.totals.rolling_truck_days,
        }
        for r in results
    ]
