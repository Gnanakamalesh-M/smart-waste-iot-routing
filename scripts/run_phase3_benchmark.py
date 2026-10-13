"""Phase 3 benchmark runner: ten strategies, two demand scenarios, two scoring windows.

    python scripts\\run_phase3_benchmark.py --seed 42

All logic lives in :mod:`smartwaste.phase3`; this file only parses arguments, writes files and
prints. Output carries no timestamps and no timings, so two runs with the same seed are
byte-identical:

    python scripts\\run_phase3_benchmark.py --seed 42 --quick > results\\p3a.txt
    python scripts\\run_phase3_benchmark.py --seed 42 --quick > results\\p3b.txt
    fc results\\p3a.txt results\\p3b.txt

``--quick`` is a smoke run on a smaller city with fewer seeds. Its tables are labelled
"SMOKE RUN, NOT FOR PUBLICATION": every number in ``docs/PHASE3_RESULTS.md`` comes from the full
default run.

Phase 2's runner (``scripts/run_benchmark.py``) is untouched and still reproduces exactly the
tables in ``docs/PHASE2_RESULTS.md``.
"""

from __future__ import annotations

import argparse
import csv
from collections.abc import Sequence
from pathlib import Path

from smartwaste import load_config
from smartwaste.phase3 import (
    CSV_FIELDS,
    FULL_LABEL,
    PHASE2_KEYS,
    SCENARIOS_REPORTED,
    WINDOW_LABEL,
    Phase3Result,
    calendar_verdict,
    csv_rows,
    forecast_verdict,
    render_headline,
    render_reductions,
    render_training_rows,
    run_phase3_benchmark,
)

#: ``--quick`` sizes. A smoke run only: small enough to finish in about a minute.
QUICK_N_BINS = 1000
QUICK_N_SEEDS = 2
QUICK_FAULT_FRACTIONS = (0.02,)


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
        "--fault-fractions", type=float, nargs="+", default=None, help="stuck-sensor sweep"
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=None,
        choices=list(SCENARIOS_REPORTED),
        help="demand scenarios to report (default: both)",
    )
    parser.add_argument(
        "--no-gbm",
        action="store_true",
        help="skip the gradient-boosting strategies, which are the only ones needing scikit-learn",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            f"smoke run: {QUICK_N_BINS} bins, {QUICK_N_SEEDS} seeds, one fault fraction. "
            "Tables are labelled NOT FOR PUBLICATION."
        ),
    )
    return parser.parse_args(argv)


def write_outputs(
    out_dir: Path, results: Sequence[Phase3Result], markdown: str
) -> tuple[Path, Path]:
    """Write the CSV and markdown tables. Returns both paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "phase3_benchmark.csv"
    md_path = out_dir / "phase3_benchmark.md"

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_FIELDS))
        writer.writeheader()
        writer.writerows(csv_rows(results))

    md_path.write_text(markdown, encoding="utf-8")
    return csv_path, md_path


def build_report(
    results: Sequence[Phase3Result],
    scenarios: Sequence[str],
    fault_fractions: Sequence[float],
    primary_seed: int,
    *,
    quick: bool,
) -> str:
    """Assemble every table into one markdown document."""
    blocks: list[str] = ["# Phase 3 benchmark output"]
    if quick:
        blocks.append(
            "> **SMOKE RUN, NOT FOR PUBLICATION.** Reduced city, seeds and sweep. Published\n"
            "> numbers come from the full default run."
        )

    blocks.append(
        "The **headline window** excludes the forecasters' warm-up, and **every** strategy is\n"
        "scored on that same window -- the calendars and the paper's fusion included. The\n"
        "**full horizon** tables that follow cover Phase 2's five strategies only, so they line\n"
        "up directly against `docs/PHASE2_RESULTS.md`."
    )

    for scenario in scenarios:
        for fault_fraction in fault_fractions:
            blocks.append(
                render_headline(
                    results,
                    scenario=scenario,
                    seed=primary_seed,
                    fault_fraction=fault_fraction,
                    window=WINDOW_LABEL,
                    quick=quick,
                )
            )
            blocks.append(
                render_reductions(
                    results,
                    scenario=scenario,
                    seed=primary_seed,
                    fault_fraction=fault_fraction,
                    window=WINDOW_LABEL,
                    quick=quick,
                )
            )

    for scenario in scenarios:
        for fault_fraction in fault_fractions:
            blocks.append(
                render_headline(
                    results,
                    scenario=scenario,
                    seed=primary_seed,
                    fault_fraction=fault_fraction,
                    window=FULL_LABEL,
                    keys=PHASE2_KEYS,
                    quick=quick,
                )
            )

    blocks.append(render_training_rows(results))
    return "\n\n".join(block for block in blocks if block) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)

    primary_seed = cfg.seed if args.seed is None else args.seed
    n_seeds = cfg.benchmark.n_seeds if args.n_seeds is None else args.n_seeds
    fault_fractions = (
        list(cfg.benchmark.fault_fractions)
        if args.fault_fractions is None
        else list(args.fault_fractions)
    )
    n_bins = cfg.sim.n_bins if args.n_bins is None else args.n_bins
    if args.quick:
        # Only override what the user did not ask for explicitly.
        if args.n_bins is None:
            n_bins = QUICK_N_BINS
        if args.n_seeds is None:
            n_seeds = QUICK_N_SEEDS
        if args.fault_fractions is None:
            fault_fractions = list(QUICK_FAULT_FRACTIONS)

    seeds = [primary_seed + offset for offset in range(n_seeds)]
    days = cfg.sim.days if args.days is None else args.days
    n_clusters = cfg.routing.n_clusters if args.n_clusters is None else args.n_clusters
    scenarios = list(SCENARIOS_REPORTED) if args.scenarios is None else list(args.scenarios)
    out_dir = cfg.paths.resolve("results_dir") if args.out_dir is None else args.out_dir

    print("Phase 3 city benchmark")
    if args.quick:
        print("  QUICK SMOKE RUN -- not for publication")
    print(f"  bins={n_bins} days={days} clusters={n_clusters}")
    print(f"  seeds={seeds}")
    print(f"  demand scenarios={scenarios}")
    print(f"  stuck-sensor fractions={[f'{f:g}' for f in fault_fractions]}")
    print(f"  warm-up days={cfg.forecast.warmup_days} (headline window starts here)")
    print(f"  gradient boosting={'OFF' if args.no_gbm else 'ON'}")
    print()

    results = run_phase3_benchmark(
        cfg,
        seeds,
        fault_fractions,
        scenarios,
        n_bins=n_bins,
        days=days,
        n_clusters=n_clusters,
        include_gbm=not args.no_gbm,
    )

    report = build_report(results, scenarios, fault_fractions, primary_seed, quick=args.quick)
    print(report)

    print("Verdict: does forecasting earn its keep? (headline window, mean over seeds)")
    for line in forecast_verdict(results, scenarios, fault_fractions):
        print(line)
    print()
    print("Verdict: does Phase 2's finding survive? Is a dumb calendar still competitive?")
    for line in calendar_verdict(results, scenarios, fault_fractions):
        print(line)
    print()

    csv_path, md_path = write_outputs(out_dir, results, report)
    print(f"wrote {csv_path.name} and {md_path.name} to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
