"""Phase 1 experiment runner: single sensor vs 3-sensor mean vs 3-sensor mean + load cell.

    python scripts\\run_fusion_eval.py

All logic lives in :mod:`smartwaste.fusion_eval`; this file only parses arguments, writes files
and prints. Output is deliberately free of timestamps and timings so that two runs with the same
seed are byte-identical:

    python scripts\\run_fusion_eval.py --seed 42 > results\\run1.txt
    python scripts\\run_fusion_eval.py --seed 42 > results\\run2.txt
    fc results\\run1.txt results\\run2.txt
"""

from __future__ import annotations

import argparse
import csv
from collections.abc import Sequence
from pathlib import Path

from smartwaste import load_config
from smartwaste.config import Config
from smartwaste.fusion_eval import (
    CSV_FIELDS,
    SWEEP_FIELDS,
    MethodMetrics,
    SweepRow,
    evaluate_all,
    render_markdown,
    sweep,
    verdicts,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=None, help="config YAML (default: repo)")
    parser.add_argument("--seed", type=int, default=None, help="master seed (default: config)")
    parser.add_argument("--out-dir", type=Path, default=None, help="output dir (default: results)")
    parser.add_argument("--n-bins", type=int, default=None, help="bins per scenario")
    parser.add_argument("--days", type=int, default=None, help="days per bin")
    parser.add_argument("--sweep-samples", type=int, default=None, help="diagnostic sweep size")
    parser.add_argument(
        "--density-cv",
        type=float,
        default=None,
        help=(
            "per-bin waste-density spread. The load cell's whole advantage depends on this: "
            "at 0 the weight channel is redundant with the ultrasonic array."
        ),
    )
    parser.add_argument(
        "--no-sensitivity",
        action="store_true",
        help="skip the density_cv=0 sensitivity table",
    )
    parser.add_argument("--no-sweep", action="store_true", help="skip the diagnostic sweep")
    return parser.parse_args(argv)


def write_csv(path: Path, fields: Sequence[str], rows: Sequence[dict[str, str]]) -> None:
    # newline="" per the csv docs, so Windows does not double the line endings.
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def run_table(
    cfg: Config, seed: int, n_bins: int, days: int, out_dir: Path, stem: str, title: str
) -> list[MethodMetrics]:
    """Evaluate every scenario, write the CSV + markdown, and print the table."""
    rows = evaluate_all(cfg, seed=seed, n_bins=n_bins, days=days)
    write_csv(out_dir / f"{stem}.csv", CSV_FIELDS, [r.as_row() for r in rows])
    table = render_markdown(rows, title)
    (out_dir / f"{stem}.md").write_text(table + "\n", encoding="utf-8")
    print(table)
    print()
    print("Verdict (misclassification, vs method (a)):")
    for line in verdicts(rows):
        print(f"  {line}")
    print()
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    overrides: dict[str, dict[str, float]] = {}
    if args.density_cv is not None:
        overrides["bin"] = {"density_cv": args.density_cv}
    cfg = load_config(args.config, overrides=overrides or None)

    seed = cfg.seed if args.seed is None else args.seed
    n_bins = cfg.fusion_eval.n_bins if args.n_bins is None else args.n_bins
    days = cfg.fusion_eval.days if args.days is None else args.days
    samples = cfg.fusion_eval.sweep_samples if args.sweep_samples is None else args.sweep_samples
    out_dir = cfg.paths.resolve("results_dir") if args.out_dir is None else args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Phase 1 sensor-fusion evaluation")
    print(
        f"seed={seed} n_bins={n_bins} days={days} "
        f"density_cv={cfg.bin.density_cv} alpha={cfg.bin.alpha} "
        f"H={cfg.bin.height_cm}cm W_max={cfg.bin.w_max_kg}kg"
    )
    print(
        "Metrics measure agreement with the paper's own RED/YELLOW/GREEN rule applied to the "
        "true fill and true weight. They are NOT collection outcomes."
    )
    print()

    run_table(
        cfg,
        seed,
        n_bins,
        days,
        out_dir,
        "fusion_eval",
        f"Headline: misclassification by noise scenario (density_cv={cfg.bin.density_cv})",
    )

    if not args.no_sensitivity and cfg.bin.density_cv != 0.0:
        # The load cell can only help when weight carries information the distances do not.
        # At density_cv=0 it carries none, by construction. Reporting both stops method (c)
        # looking good purely because of a spread we chose.
        zero_cfg = load_config(args.config, overrides={"bin": {"density_cv": 0.0}})
        run_table(
            zero_cfg,
            seed,
            n_bins,
            days,
            out_dir,
            "fusion_eval_dcv0",
            "Sensitivity: density_cv=0 (weight is a fixed multiple of volume, "
            "so the load cell can add nothing)",
        )

    if not args.no_sweep:
        sweep_rows: list[SweepRow] = sweep(cfg, seed=seed, n_samples=samples)
        write_csv(out_dir / "fusion_sweep.csv", SWEEP_FIELDS, [r.as_row() for r in sweep_rows])
        print(f"Diagnostic sweep ({samples} uniform-fill samples per scenario) -> fusion_sweep.csv")
        print()

    print(f"Wrote CSV + markdown to {out_dir.name}/ (gitignored; regenerate with this command).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
