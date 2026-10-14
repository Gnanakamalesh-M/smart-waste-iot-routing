"""Phase 4 routing runner: algorithm comparison and capacitated multi-truck VRP.

    python scripts\\run_phase4_vrp.py --seed 42

All logic lives in :mod:`smartwaste.phase4`; this file only parses arguments, writes files and
prints. Output carries no timestamps and no timings, so the stdlib parts are byte-identical
between same-seed runs:

    python scripts\\run_phase4_vrp.py --seed 42 --quick --no-ortools > results\\v1.txt
    python scripts\\run_phase4_vrp.py --seed 42 --quick --no-ortools > results\\v2.txt
    fc results\\v1.txt results\\v2.txt

**The OR-Tools rows are NOT byte-reproducible**, on this or any machine: the solver runs under a
wall-clock time limit (``vrp.ortools_time_limit_s``), so a busier or faster machine searches a
different amount of the space. Every OR-Tools number is labelled same-machine-only in
``docs/PHASE4_RESULTS.md``, and no test asserts one.

``--quick`` is a smoke run: fewer clusters, a shorter solver budget, one capacity. Its tables are
labelled "SMOKE RUN, NOT FOR PUBLICATION".

Phase 2's and Phase 3's runners are untouched and still reproduce their published tables.
"""

from __future__ import annotations

import argparse
import csv
from collections.abc import Sequence
from pathlib import Path

from smartwaste import load_config
from smartwaste.phase4 import (
    CSV_FIELDS,
    Phase4Results,
    build_compare_instances,
    capacity_verdict,
    compare_algorithms,
    compare_capacities,
    compare_plan_bases,
    csv_rows,
    noise_cost_verdict,
    ortools_verdict,
    quick_banner,
    render_algorithms,
    render_capacities,
    render_executions,
)

#: ``--quick`` sizes. A smoke run only: small enough to finish in well under a minute.
QUICK_COMPARE_CLUSTERS = 5
QUICK_TIME_LIMIT_S = 1.0
QUICK_CAPACITIES = (8000.0,)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=None, help="config YAML (default: repo)")
    parser.add_argument("--seed", type=int, default=None, help="master seed (default: config)")
    parser.add_argument("--out-dir", type=Path, default=None, help="output dir (default: results)")
    parser.add_argument("--n-bins", type=int, default=None, help="city size (default: sim.n_bins)")
    parser.add_argument("--days", type=int, default=None, help="horizon (default: sim.days)")
    parser.add_argument(
        "--n-clusters", type=int, default=None, help="KMeans clusters (default: routing)"
    )
    parser.add_argument(
        "--compare-clusters",
        type=int,
        default=None,
        help="clusters fed to the comparison (default: vrp.compare_clusters)",
    )
    parser.add_argument(
        "--compare-day",
        type=int,
        default=None,
        help="simulated day supplying the bin sets (default: vrp.compare_day)",
    )
    parser.add_argument(
        "--capacities",
        type=float,
        nargs="+",
        default=None,
        help="capacity sweep in kg (default: vrp.capacity_sweep_kg)",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=None,
        help="OR-Tools seconds per instance (default: vrp.ortools_time_limit_s)",
    )
    parser.add_argument(
        "--no-ortools",
        action="store_true",
        help="stdlib baselines only: skip every OR-Tools row (Tables B and C are then omitted)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            f"smoke run: {QUICK_COMPARE_CLUSTERS} clusters, {QUICK_TIME_LIMIT_S}s solver budget, "
            f"capacities {list(QUICK_CAPACITIES)}. Tables are labelled not for publication."
        ),
    )
    return parser.parse_args(argv)


def write_outputs(out_dir: Path, results: Phase4Results, markdown: str) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "phase4_vrp.csv"
    md_path = out_dir / "phase4_vrp.md"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_FIELDS))
        writer.writeheader()
        writer.writerows(csv_rows(results))
    md_path.write_text(markdown, encoding="utf-8")
    return csv_path, md_path


def build_report(results: Phase4Results, *, quick: bool) -> str:
    blocks = [
        render_algorithms(results),
        render_capacities(results),
        render_executions(results),
    ]
    body = "\n\n".join(block for block in blocks if block) + "\n"
    return (quick_banner() + "\n" + body) if quick else body


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)

    seed = cfg.seed if args.seed is None else args.seed
    n_bins = cfg.sim.n_bins if args.n_bins is None else args.n_bins
    days = cfg.sim.days if args.days is None else args.days
    n_clusters = cfg.routing.n_clusters if args.n_clusters is None else args.n_clusters
    compare_clusters = (
        cfg.vrp.compare_clusters if args.compare_clusters is None else args.compare_clusters
    )
    compare_day = cfg.vrp.compare_day if args.compare_day is None else args.compare_day
    capacities = (
        list(cfg.vrp.capacity_sweep_kg) if args.capacities is None else list(args.capacities)
    )
    time_limit = cfg.vrp.ortools_time_limit_s if args.time_limit is None else args.time_limit
    out_dir = cfg.paths.resolve("results_dir") if args.out_dir is None else args.out_dir

    if args.quick:
        # Only override what the user did not ask for explicitly.
        if args.compare_clusters is None:
            compare_clusters = QUICK_COMPARE_CLUSTERS
        if args.time_limit is None:
            time_limit = QUICK_TIME_LIMIT_S
        if args.capacities is None:
            capacities = list(QUICK_CAPACITIES)

    use_ortools = not args.no_ortools

    print("Phase 4 routing: algorithm comparison and capacitated VRP")
    if args.quick:
        print("  QUICK SMOKE RUN -- not for publication")
    print(f"  bins={n_bins} days={days} clusters={n_clusters}")
    print(f"  seed={seed} comparison day={compare_day} clusters compared={compare_clusters}")
    print(f"  capacities kg={[f'{c:g}' for c in capacities]}")
    print(f"  OR-Tools={'ON' if use_ortools else 'OFF'} time limit={time_limit:g}s per instance")
    print("  NOTE: OR-Tools runs under a wall-clock limit, so its distances are")
    print("        same-machine-only. The stdlib baselines are byte-reproducible.")
    print()

    instances = build_compare_instances(
        cfg,
        seed,
        n_bins=n_bins,
        days=days,
        n_clusters=n_clusters,
        compare_clusters=compare_clusters,
        compare_day=compare_day,
    )
    n_selected = sum(instance.n_bins for instance in instances)
    print(f"  {len(instances)} non-empty clusters, {n_selected} bins selected on day {compare_day}")
    print()

    algorithms = compare_algorithms(
        cfg, instances, time_limit_s=time_limit, include_ortools=use_ortools
    )
    capacitated = (
        compare_capacities(cfg, instances, capacities, time_limit_s=time_limit)
        if use_ortools
        else ()
    )
    executions = (
        compare_plan_bases(
            cfg, instances, capacity_kg=cfg.cost.truck_capacity_kg, time_limit_s=time_limit
        )
        if use_ortools
        else ()
    )

    results = Phase4Results(
        seed=seed,
        day=compare_day,
        n_clusters=len(instances),
        n_bins_selected=n_selected,
        algorithms=algorithms,
        capacitated=capacitated,
        executions=executions,
    )

    report = build_report(results, quick=args.quick)
    print(report)

    print("Verdict: does OR-Tools beat the hand-written 2-opt by a margin worth the dependency?")
    for line in ortools_verdict(results):
        print(line)
    print()
    print("Verdict: how many trucks does capacity actually force?")
    for line in capacity_verdict(results):
        print(line)
    print()
    print("Verdict: what does planning from noisy sensor weight cost?")
    for line in noise_cost_verdict(results):
        print(line)
    print()

    csv_path, md_path = write_outputs(out_dir, results, report)
    print(f"wrote {csv_path.name} and {md_path.name} to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
