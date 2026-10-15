"""Generate the committed demo snapshot the Phase 5 dashboard reads.

    python scripts\\make_demo_data.py --seed 42

Runs ONE small seeded city -- 200 bins, 14 days, 5 clusters, strategy (d) the paper's own sensor
fusion -- and writes it to ``dashboard/demo_data/`` as five small files that ARE checked into git.

### Why a committed snapshot and not a live run

The dashboard must never be able to start a long job. The strongest way to guarantee that is for
it to have no simulator to call: ``dashboard/app.py`` imports only
:mod:`smartwaste.dashboard_data`, which reads these files and the committed results docs. So this
script is the only place the simulation and the dashboard meet, and it runs here, once, offline.

### This is NOT the Phase 2 city benchmark

200 bins x 14 days is **4% of the bins and 47% of the days** of ``sim.n_bins`` x ``sim.days``.
Its numbers are a demonstration of the pipeline, not a result, and nothing in
``docs/PHASE5_RESULTS.md`` compares them with Phase 2's. The baseline comparison the dashboard
shows comes from ``docs/PHASE2_RESULTS.md`` instead, parsed at display time.

Demand uses ``cfg.demand.scenario`` (``heterogeneous``), matching the Phase 3 runners.

Output carries no timestamps and no timings, so two same-seed runs are byte-identical:

    python scripts\\make_demo_data.py --seed 42 --out-dir results\\d1
    python scripts\\make_demo_data.py --seed 42 --out-dir results\\d2
    fc results\\d1\\kpis.csv results\\d2\\kpis.csv

ALL DATA HERE IS SIMULATED. No reading in these files came from hardware.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from smartwaste import load_config
from smartwaste.city_sim import build_layout, build_sensor_arrays, step_day
from smartwaste.config import Config
from smartwaste.cost import TruckDay, cost_truck_day
from smartwaste.dashboard_data import (
    BIN_DAYS_FIELDS,
    BIN_DAYS_FILENAME,
    BINS_FIELDS,
    BINS_FILENAME,
    DEMO_DIR,
    DEPOT_BIN_ID,
    FLOAT_FORMAT,
    KPIS_FIELDS,
    KPIS_FILENAME,
    META_FILENAME,
    ROUTES_FIELDS,
    ROUTES_FILENAME,
    SIMULATED_DATA_BANNER,
)
from smartwaste.demand import AREA_TYPE_NAMES
from smartwaste.fusion import fuse, mean_distance, truth_state
from smartwaste.routing.cluster import kmeans
from smartwaste.routing.tsp import Point, solve
from smartwaste.seeds import child_seed
from smartwaste.strategies import RunContext, build_strategies

#: Demo size. Script constants rather than config rows, for the same reason ``QUICK_N_BINS`` in
#: ``scripts\\run_roads.py`` is one: they shape a demonstration, not a result, so promoting them
#: to ``config/default.yaml`` would add three more invented-and-uncited rows for nothing.
DEMO_N_BINS = 200
DEMO_DAYS = 14
DEMO_N_CLUSTERS = 5

#: The paper's own method, so the demo shows what the paper does rather than our extension.
DEMO_STRATEGY_KEY = "d_paper_fusion"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=None, help="config YAML (default: repo)")
    parser.add_argument("--seed", type=int, default=None, help="master seed (default: config)")
    parser.add_argument(
        "--out-dir", type=Path, default=None, help="output dir (default: dashboard/demo_data)"
    )
    parser.add_argument("--n-bins", type=int, default=None, help=f"bins (default: {DEMO_N_BINS})")
    parser.add_argument("--days", type=int, default=None, help=f"days (default: {DEMO_DAYS})")
    parser.add_argument(
        "--n-clusters", type=int, default=None, help=f"clusters (default: {DEMO_N_CLUSTERS})"
    )
    return parser.parse_args(argv)


@dataclass(frozen=True)
class DemoSnapshot:
    """Every row the five output files need, already in write order."""

    bins: list[dict[str, str]]
    bin_days: list[dict[str, str]]
    routes: list[dict[str, str]]
    kpis: list[dict[str, str]]
    meta: dict[str, object]


def _f(value: float) -> str:
    return FLOAT_FORMAT.format(value)


def build_snapshot(
    cfg: Config,
    seed: int,
    *,
    n_bins: int,
    days: int,
    n_clusters: int,
    command: str,
) -> DemoSnapshot:
    """Replay one small city under strategy (d), recording everything the dashboard draws.

    The day loop mirrors :func:`smartwaste.city_sim.run_strategy` exactly -- waste arrives,
    overflow is measured, EVERY bin is read whether or not the strategy looks at it, the strategy
    chooses, the choices are routed per cluster, the chosen bins are emptied. ``run_strategy``
    itself cannot be reused because it returns only aggregates: its ``RouteCache`` yields a
    distance and discards the tour, and the dashboard needs the tour to draw a line. The same
    reason ``smartwaste.phase4.build_compare_instances`` re-implements the loop.
    """
    layout = build_layout(
        cfg,
        seed,
        n_bins=n_bins,
        days=days,
        stuck_bin_fraction=cfg.sensors.stuck_bin_fraction,
        scenario=cfg.demand.scenario,
    )
    clusters = kmeans(layout.locations, n_clusters, random.Random(child_seed(seed, "clusters")))
    arrays = build_sensor_arrays(cfg, layout, seed)

    strategies = {s.key: s for s in build_strategies(cfg.bin, cfg.benchmark.fixed_schedule_days)}
    strategy = strategies[DEMO_STRATEGY_KEY]
    strategy.begin_run(RunContext(calendar=layout.calendar, bin_cfg=cfg.bin, n_bins=n_bins))

    members: list[list[int]] = [[] for _ in range(clusters.n_clusters)]
    for bin_index, label in enumerate(clusters.labels):
        members[label].append(bin_index)

    bins_rows = [
        {
            "bin_id": str(i),
            "x_km": _f(layout.locations[i][0]),
            "y_km": _f(layout.locations[i][1]),
            "cluster": str(clusters.labels[i]),
            "area_type": AREA_TYPE_NAMES[layout.calendar.area_type[i]],
            "kg_per_pct": _f(layout.kg_per_pct[i]),
            "stuck_sensor": str(layout.stuck_sensor[i]),
        }
        for i in range(n_bins)
    ]

    bin_days_rows: list[dict[str, str]] = []
    route_rows: list[dict[str, str]] = []
    kpi_rows: list[dict[str, str]] = []

    weights = [0.0] * n_bins
    was_overflowing = [False] * n_bins
    ever_overflowed = [False] * n_bins

    for day in range(days):
        fills, day_counters = step_day(cfg, layout, weights, was_overflowing, ever_overflowed, day)
        readings = tuple(arrays[i].read(fills[i], weights[i]) for i in range(n_bins))
        selected = strategy.select(day, readings)
        selected_set = set(selected)

        for i in range(n_bins):
            # Recomputed with (d)'s own configuration -- 3-sensor MEAN plus the load cell -- so
            # `sensor_state` is literally what the strategy acted on, not a second opinion.
            fused = fuse(
                readings[i],
                cfg.bin,
                use_load_cell=True,
                sensor_indices=None,
                reducer=mean_distance,
            )
            bin_days_rows.append(
                {
                    "day": str(day),
                    "bin_id": str(i),
                    "true_fill_pct": _f(fills[i]),
                    "true_weight_kg": _f(weights[i]),
                    "est_fill_pct": _f(fused.fill_pct),
                    "est_overloaded": str(fused.overloaded),
                    "true_state": str(truth_state(fills[i], weights[i], cfg.bin)),
                    "sensor_state": str(fused.state),
                    "selected": str(i in selected_set),
                }
            )

        truck_days: list[TruckDay] = []
        for cluster_id in range(clusters.n_clusters):
            cluster_selected = [i for i in members[cluster_id] if i in selected_set]
            points: list[Point] = [layout.depot] + [layout.locations[i] for i in cluster_selected]
            if cluster_selected:
                tour, distance_km = solve(points, cfg.routing.two_opt_max_iters)
            else:
                tour, distance_km = [], 0.0
            truck_days.append(TruckDay(distance_km=distance_km, bins_visited=len(cluster_selected)))

            for seq, point_index in enumerate(tour):
                bin_id = DEPOT_BIN_ID if point_index == 0 else cluster_selected[point_index - 1]
                route_rows.append(
                    {
                        "day": str(day),
                        "cluster": str(cluster_id),
                        "seq": str(seq),
                        "bin_id": str(bin_id),
                        "x_km": _f(points[point_index][0]),
                        "y_km": _f(points[point_index][1]),
                    }
                )
            if tour:
                # Close the polyline: the truck returns to the depot, and `route_distance`
                # already charges that leg, so the drawing has to show it.
                route_rows.append(
                    {
                        "day": str(day),
                        "cluster": str(cluster_id),
                        "seq": str(len(tour)),
                        "bin_id": str(DEPOT_BIN_ID),
                        "x_km": _f(layout.depot[0]),
                        "y_km": _f(layout.depot[1]),
                    }
                )

        costed = [cost_truck_day(td, cfg.cost) for td in truck_days]
        kpi_rows.append(
            {
                "day": str(day),
                "distance_km": _f(sum(c.distance_km for c in costed)),
                "travel_time_hr": _f(sum(c.travel_time_hr for c in costed)),
                "co2_kg": _f(sum(c.co2_kg for c in costed)),
                "variable_cost": _f(sum(c.variable_cost for c in costed)),
                "bins_visited": str(sum(c.bins_visited for c in costed)),
                "overflow_events": str(day_counters.overflow_events),
            }
        )

        strategy.observe_collection(day, selected)
        for i in selected:
            weights[i] = 0.0
            was_overflowing[i] = False

    meta: dict[str, object] = {
        "label": SIMULATED_DATA_BANNER,
        "seed": seed,
        "n_bins": n_bins,
        "days": days,
        "n_clusters": n_clusters,
        "scenario": layout.scenario,
        "strategy_key": strategy.key,
        "strategy_label": strategy.label,
        "grid_km": cfg.sim.grid_km,
        "depot_x_km": layout.depot[0],
        "depot_y_km": layout.depot[1],
        "stuck_bin_fraction": cfg.sensors.stuck_bin_fraction,
        "command": command,
    }

    return DemoSnapshot(
        bins=bins_rows,
        bin_days=bin_days_rows,
        routes=route_rows,
        kpis=kpi_rows,
        meta=meta,
    )


def write_csv(path: Path, fields: Sequence[str], rows: Sequence[dict[str, str]]) -> None:
    """Write one demo CSV.

    ``newline=""`` is what ``csv`` requires to control line endings itself, and
    ``lineterminator="\\n"`` pins them to LF so the committed file is identical on Windows and
    Linux -- otherwise regenerating on the other OS would show every line as changed.
    """
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    seed = cfg.seed if args.seed is None else args.seed
    n_bins = DEMO_N_BINS if args.n_bins is None else args.n_bins
    days = DEMO_DAYS if args.days is None else args.days
    n_clusters = DEMO_N_CLUSTERS if args.n_clusters is None else args.n_clusters
    out_dir = DEMO_DIR if args.out_dir is None else args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    command = f"python scripts\\make_demo_data.py --seed {seed}"
    snapshot = build_snapshot(
        cfg, seed, n_bins=n_bins, days=days, n_clusters=n_clusters, command=command
    )

    write_csv(out_dir / BINS_FILENAME, BINS_FIELDS, snapshot.bins)
    write_csv(out_dir / BIN_DAYS_FILENAME, BIN_DAYS_FIELDS, snapshot.bin_days)
    write_csv(out_dir / ROUTES_FILENAME, ROUTES_FIELDS, snapshot.routes)
    write_csv(out_dir / KPIS_FILENAME, KPIS_FIELDS, snapshot.kpis)
    # newline="\n" rather than write_text(): on Windows write_text translates "\n" to "\r\n",
    # which would commit this one file with different line endings from the four CSVs beside it.
    with (out_dir / META_FILENAME).open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(snapshot.meta, indent=2, sort_keys=True) + "\n")

    print(SIMULATED_DATA_BANNER)
    print(f"demo snapshot: {n_bins} bins x {days} days, {n_clusters} clusters, seed {seed}")
    print(f"strategy: {snapshot.meta['strategy_label']}")
    print(f"demand scenario: {snapshot.meta['scenario']}")
    print("NOT the Phase 2 city benchmark: 200x14 is a demonstration, not a result.")
    print()
    print(f"wrote {out_dir}:")
    for name in (BINS_FILENAME, BIN_DAYS_FILENAME, ROUTES_FILENAME, KPIS_FILENAME, META_FILENAME):
        path = out_dir / name
        print(f"  {name:<14} {path.stat().st_size:>8} bytes")
    total = sum(
        (out_dir / n).stat().st_size
        for n in (
            BINS_FILENAME,
            BIN_DAYS_FILENAME,
            ROUTES_FILENAME,
            KPIS_FILENAME,
            META_FILENAME,
        )
    )
    print(f"  {'TOTAL':<14} {total:>8} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
