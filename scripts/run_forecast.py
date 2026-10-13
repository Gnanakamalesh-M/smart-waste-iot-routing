"""Phase 3 forecast runner: do the ML model's predictions beat the naive baselines?

    python scripts\\run_forecast.py --seed 42

All logic lives in :mod:`smartwaste.forecast_eval`; this file only parses arguments, writes files
and prints. Output carries no timestamps and no timings, so two runs with the same seed are
byte-identical on this machine.

Note the scope of that promise: the three naive baselines are stdlib and reproduce anywhere, but
the gradient-boosting row comes from scikit-learn, which does not guarantee bit-identical floats
across operating systems, BLAS builds or library versions. ``--versions`` prints the versions
actually used; they are deliberately kept out of the normal output so that ``fc`` still passes.
"""

from __future__ import annotations

import argparse
import csv
from collections.abc import Sequence
from pathlib import Path

from smartwaste import load_config
from smartwaste.demand import SCENARIOS
from smartwaste.forecast_eval import (
    CSV_FIELDS,
    Dataset,
    ForecastScore,
    build_dataset,
    csv_rows,
    render_accuracy,
    render_warmup_diagnostics,
    score_forecasters,
    verdict_lines,
)
from smartwaste.ml.forecast import FORECASTER_KEYS

#: Forecast evaluation is per-bin and does not need the full 5000-bin city to be informative,
#: so the default is smaller than the benchmark's. Invented, uncited.
DEFAULT_N_BINS = 400


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=None, help="config YAML (default: repo)")
    parser.add_argument("--seed", type=int, default=None, help="master seed (default: config)")
    parser.add_argument("--out-dir", type=Path, default=None, help="output dir (default: results)")
    parser.add_argument("--n-bins", type=int, default=DEFAULT_N_BINS, help="bins to simulate")
    parser.add_argument("--days", type=int, default=None, help="horizon (default: sim.days)")
    parser.add_argument(
        "--scenarios", nargs="+", default=None, choices=list(SCENARIOS), help="demand scenarios"
    )
    parser.add_argument(
        "--no-gbm",
        action="store_true",
        help="skip the gradient-boosting model, the only forecaster needing scikit-learn",
    )
    parser.add_argument(
        "--versions",
        action="store_true",
        help="print numpy / scikit-learn versions and exit (kept out of normal output so that "
        "two same-seed runs stay byte-identical)",
    )
    return parser.parse_args(argv)


def print_versions() -> int:
    """Report the library versions the ML rows depend on."""
    for name in ("numpy", "sklearn"):
        try:
            module = __import__(name)
        except ImportError:
            print(f"{name}: NOT INSTALLED")
        else:
            print(f"{name}: {getattr(module, '__version__', 'unknown')}")
    return 0


def write_outputs(
    out_dir: Path, scores: Sequence[ForecastScore], markdown: str
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "forecast_eval.csv"
    md_path = out_dir / "forecast_eval.md"

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_FIELDS))
        writer.writeheader()
        writer.writerows(csv_rows(scores))

    md_path.write_text(markdown, encoding="utf-8")
    return csv_path, md_path


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.versions:
        return print_versions()

    cfg = load_config(args.config)
    seed = cfg.seed if args.seed is None else args.seed
    days = cfg.sim.days if args.days is None else args.days
    scenarios = list(SCENARIOS) if args.scenarios is None else list(args.scenarios)
    out_dir = cfg.paths.resolve("results_dir") if args.out_dir is None else args.out_dir
    keys = [key for key in FORECASTER_KEYS if not (args.no_gbm and key == "gbm")]

    print("Phase 3 forecast evaluation")
    print(f"  bins={args.n_bins} days={days} seed={seed}")
    print(f"  demand scenarios={scenarios}")
    print(f"  warm-up days={cfg.forecast.warmup_days} (train < this, evaluate >= this)")
    print(f"  horizon days={cfg.forecast.horizon_days}")
    print(f"  forecasters={keys}")
    print()

    scores: list[ForecastScore] = []
    datasets: list[tuple[str, int, Dataset]] = []
    blocks: list[str] = ["# Phase 3 forecast evaluation output"]

    for scenario in scenarios:
        dataset = build_dataset(cfg, seed, scenario=scenario, n_bins=args.n_bins, days=days)
        datasets.append((scenario, seed, dataset))

        # The time-based split, asserted here as well as in the tests: if this ever fails, the
        # numbers below would be leakage rather than forecasting.
        latest_train = dataset.latest_train_day()
        earliest_eval = dataset.earliest_eval_day()
        if latest_train is not None and earliest_eval is not None:
            assert latest_train < earliest_eval, (
                f"time-based split violated: a training row from day {latest_train} is not "
                f"strictly before the earliest evaluation day {earliest_eval}"
            )

        scenario_scores = score_forecasters(cfg, dataset, seed, scenario=scenario, keys=keys)
        scores.extend(scenario_scores)
        blocks.append(render_accuracy(scenario_scores, scenario))

    blocks.append(render_warmup_diagnostics(datasets))
    report = "\n\n".join(blocks) + "\n"
    print(report)

    print("Verdict: does the ML model beat the naive baselines?")
    for scenario in scenarios:
        for line in verdict_lines(scores, scenario):
            print(line)
    print()

    csv_path, md_path = write_outputs(out_dir, scores, report)
    print(f"wrote {csv_path.name} and {md_path.name} to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
