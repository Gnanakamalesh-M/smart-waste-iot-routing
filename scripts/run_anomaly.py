"""Phase 3 anomaly runner: generate planted sensor faults, then score two detectors.

    python scripts\\run_anomaly.py --seed 42

All logic lives in :mod:`smartwaste.anomaly_eval`; this file only parses arguments, writes files
and prints.

Writes **two separate files** into the data directory (both gitignored):

- ``anomaly_readings.csv`` -- feature columns only;
- ``anomaly_labels.csv``   -- the planted truth.

The detectors never receive the labels path. Thresholds are chosen on a **validation seed** and
applied unchanged to the scoring seed, so no reported number is tuned on its own labels.

As with the forecast runner, the Isolation Forest row depends on scikit-learn, which is not
bit-reproducible across operating systems or library versions; ``--versions`` prints what was
used. The rule baseline is stdlib and reproduces anywhere.
"""

from __future__ import annotations

import argparse
import csv
from collections.abc import Sequence
from pathlib import Path

from smartwaste import load_config
from smartwaste.anomaly_eval import (
    CSV_FIELDS,
    DetectorScore,
    build_dataset,
    csv_rows,
    render_scores,
    render_thresholds,
    score_detectors,
    select_thresholds,
    verdict_lines,
    write_dataset,
)
from smartwaste.ml.anomaly import DETECTOR_KEYS


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=None, help="config YAML (default: repo)")
    parser.add_argument("--seed", type=int, default=None, help="scoring seed (default: config)")
    parser.add_argument("--out-dir", type=Path, default=None, help="output dir (default: results)")
    parser.add_argument(
        "--data-dir", type=Path, default=None, help="where the two CSVs go (default: data)"
    )
    parser.add_argument(
        "--no-ml",
        action="store_true",
        help="skip the Isolation Forest, the only detector needing scikit-learn",
    )
    parser.add_argument(
        "--versions",
        action="store_true",
        help="print numpy / scikit-learn versions and exit",
    )
    return parser.parse_args(argv)


def print_versions() -> int:
    for name in ("numpy", "sklearn"):
        try:
            module = __import__(name)
        except ImportError:
            print(f"{name}: NOT INSTALLED")
        else:
            print(f"{name}: {getattr(module, '__version__', 'unknown')}")
    return 0


def write_outputs(
    out_dir: Path, results: Sequence[DetectorScore], markdown: str
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "anomaly_eval.csv"
    md_path = out_dir / "anomaly_eval.md"

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_FIELDS))
        writer.writeheader()
        writer.writerows(csv_rows(results))

    md_path.write_text(markdown, encoding="utf-8")
    return csv_path, md_path


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.versions:
        return print_versions()

    cfg = load_config(args.config)
    seed = cfg.seed if args.seed is None else args.seed
    out_dir = cfg.paths.resolve("results_dir") if args.out_dir is None else args.out_dir
    data_dir = cfg.paths.resolve("data_dir") if args.data_dir is None else args.data_dir
    keys = [key for key in DETECTOR_KEYS if not (args.no_ml and key == "isolation_forest")]

    print("Phase 3 sensor anomaly detection")
    print(f"  bins={cfg.anomaly.n_bins} days={cfg.anomaly.days} window={cfg.anomaly.window_days}")
    print(f"  scoring seed={seed}")
    print(f"  validation seed={seed + cfg.anomaly.validation_seed_offset} (thresholds only)")
    print(f"  planted fault fraction={cfg.anomaly.fault_bin_fraction}")
    print(f"  detectors={keys}")
    print()

    # Thresholds first, on the validation city, before the scoring dataset is even built.
    thresholds = select_thresholds(cfg, seed, include_ml=not args.no_ml)

    dataset = build_dataset(cfg, seed)
    readings_path, labels_path = write_dataset(dataset, data_dir)
    print(f"wrote {readings_path.name} (features) and {labels_path.name} (planted truth)")
    print("  the detectors are given the readings only; nothing passes them the labels path")
    print()

    results = score_detectors(cfg, dataset, seed, thresholds, keys=keys)

    blocks = [
        "# Phase 3 anomaly detection output",
        render_thresholds(thresholds, seed),
        render_scores(results),
    ]
    report = "\n\n".join(blocks) + "\n"
    print(report)

    print("Verdict: rule baseline vs Isolation Forest, per fault family")
    for line in verdict_lines(results):
        print(line)
    print()

    csv_path, md_path = write_outputs(out_dir, results, report)
    print(f"wrote {csv_path.name} and {md_path.name} to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
