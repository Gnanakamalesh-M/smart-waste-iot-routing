"""Phase 2 benchmark runner: five collection strategies over the simulated city.

    python scripts\\run_benchmark.py --seed 42

All logic lives in :mod:`smartwaste.benchmark`; this file only parses arguments, writes files and
prints. Output carries no timestamps and no timings, so two runs with the same seed are
byte-identical:

    python scripts\\run_benchmark.py --seed 42 > results\\bench1.txt
    python scripts\\run_benchmark.py --seed 42 > results\\bench2.txt
    fc results\\bench1.txt results\\bench2.txt
"""

from __future__ import annotations

import argparse
import csv
from collections.abc import Sequence
from pathlib import Path

from smartwaste import load_config
from smartwaste.benchmark import (
    CSV_FIELDS,
    StrategyResult,
    baseline_beats_paper_verdict,
    csv_rows,
    mean_vs_median_verdict,
    render_headline,
    render_paired_mean_vs_median,
    render_reductions,
    render_seed_range,
    run_benchmark,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=None, help="config YAML (default: repo)")
    parser.add_argument("--seed", type=int, default=None, help="master seed (default: config)")
    parser.add_argument("--out-dir", type=Path, default=None, help="output dir (default: results)")
    parser.add_argument("--n-bins", type=int, default=None, help="city size (default: sim.n_bins)")
    parser.add_argument("--days", type=int, default=None, help="horizon (default: sim.days)")
    parser.add_argument(
        "--n-clusters", type=int, default=None, help="trucks/clusters (default: routing)"
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=None,
        help="how many consecutive seeds to run from --seed (default: benchmark.n_seeds)",
    )
    parser.add_argument(
        "--fault-fractions",
        type=float,
        nargs="+",
        default=None,
        help=(
            "stuck-sensor fractions to sweep. The (d) mean vs (e) median verdict depends on "
            "this, so the default runs several and reports all of them."
        ),
    )
    return parser.parse_args(argv)


def write_outputs(
    out_dir: Path, results: Sequence[StrategyResult], markdown: str
) -> tuple[Path, Path]:
    """Write the CSV and markdown tables. Returns both paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "benchmark.csv"
    md_path = out_dir / "benchmark.md"

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_FIELDS))
        writer.writeheader()
        writer.writerows(csv_rows(results))

    md_path.write_text(markdown, encoding="utf-8")
    return csv_path, md_path


def build_report(
    results: Sequence[StrategyResult],
    seeds: Sequence[int],
    fault_fractions: Sequence[float],
    primary_seed: int,
) -> str:
    """Assemble every table into one markdown document."""
    blocks: list[str] = ["# Phase 2 benchmark output"]

    for fault_fraction in fault_fractions:
        blocks.append(render_headline(results, primary_seed, fault_fraction))
        blocks.append(render_reductions(results, primary_seed, fault_fraction))

    if len(seeds) > 1:
        for fault_fraction in fault_fractions:
            blocks.append(render_seed_range(results, seeds, fault_fraction))

    blocks.append(render_paired_mean_vs_median(results, seeds, fault_fractions))
    return "\n\n".join(blocks) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)

    primary_seed = cfg.seed if args.seed is None else args.seed
    n_seeds = cfg.benchmark.n_seeds if args.n_seeds is None else args.n_seeds
    seeds = [primary_seed + offset for offset in range(n_seeds)]
    fault_fractions = (
        list(cfg.benchmark.fault_fractions)
        if args.fault_fractions is None
        else list(args.fault_fractions)
    )
    n_bins = cfg.sim.n_bins if args.n_bins is None else args.n_bins
    days = cfg.sim.days if args.days is None else args.days
    n_clusters = cfg.routing.n_clusters if args.n_clusters is None else args.n_clusters
    out_dir = cfg.paths.resolve("results_dir") if args.out_dir is None else args.out_dir

    print("Phase 2 city benchmark")
    print(f"  bins={n_bins} days={days} clusters={n_clusters}")
    print(f"  seeds={seeds}")
    print(f"  stuck-sensor fractions={[f'{f:g}' for f in fault_fractions]}")
    print()

    results = run_benchmark(
        cfg,
        seeds,
        fault_fractions,
        n_bins=n_bins,
        days=days,
        n_clusters=n_clusters,
    )

    report = build_report(results, seeds, fault_fractions, primary_seed)
    print(report)

    print("Verdict: (d) the paper's mean fusion vs (e) our median fusion")
    for line in mean_vs_median_verdict(results, fault_fractions):
        print(line)
    print()
    print("Verdict: does a simpler baseline match or beat the paper's method?")
    for line in baseline_beats_paper_verdict(results, fault_fractions):
        print(line)
    print()

    csv_path, md_path = write_outputs(out_dir, results, report)
    print(f"wrote {csv_path.name} and {md_path.name} to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
