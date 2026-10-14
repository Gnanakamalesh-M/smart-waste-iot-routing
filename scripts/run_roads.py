"""Phase 4 roads runner: how much does a straight line under-state a real road distance?

    python scripts\\run_roads.py --seed 42

Downloads the drivable road network for one bounded Chennai area via OSMnx, caches it under
``cache/osmnx/`` (gitignored), snaps synthetic bins to road nodes, and reports the
road-over-straight-line ratio with its spread -- on the SAME bin set under both metrics.

All logic lives in :mod:`smartwaste.roads_eval`; this file only parses arguments, writes files and
prints. Output carries no timestamps and no timings, so two same-seed runs are byte-identical:

    python scripts\\run_roads.py --seed 42 --quick --no-download > results\\r1.txt
    python scripts\\run_roads.py --seed 42 --quick --no-download > results\\r2.txt
    fc results\\r1.txt results\\r2.txt

``--no-download`` skips the network entirely and exercises the Euclidean fallback, which is how
the loud-fallback path is checked without unplugging anything. ``--quick`` uses fewer bins and
labels its tables "SMOKE RUN, NOT FOR PUBLICATION".

BIN LOCATIONS ARE SYNTHETIC. Road data (c) OpenStreetMap contributors, ODbL 1.0.
"""

from __future__ import annotations

import argparse
import csv
from collections.abc import Sequence
from pathlib import Path

from smartwaste import load_config
from smartwaste.roads_eval import (
    CSV_FIELDS,
    RoadsResult,
    csv_rows,
    detour_verdict,
    quick_banner,
    render_detour,
    render_scope,
    render_tours,
    run_roads_eval,
)
from smartwaste.routing.roads import OSM_CREDIT, SYNTHETIC_BINS_NOTE

#: ``--quick`` size. A smoke run only.
QUICK_N_BINS = 50


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=None, help="config YAML (default: repo)")
    parser.add_argument("--seed", type=int, default=None, help="master seed (default: config)")
    parser.add_argument("--out-dir", type=Path, default=None, help="output dir (default: results)")
    parser.add_argument(
        "--n-bins", type=int, default=None, help="synthetic bins (default: roads.n_bins)"
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="never touch the network: use the cache if present, else the Euclidean fallback",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help=f"smoke run with {QUICK_N_BINS} bins. Tables are labelled not for publication.",
    )
    return parser.parse_args(argv)


def write_outputs(out_dir: Path, result: RoadsResult, markdown: str) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "roads_eval.csv"
    md_path = out_dir / "roads_eval.md"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_FIELDS))
        writer.writeheader()
        writer.writerows(csv_rows(result))
    md_path.write_text(markdown, encoding="utf-8")
    return csv_path, md_path


def build_report(result: RoadsResult, *, quick: bool) -> str:
    blocks = [
        render_scope(result),
        render_detour(result),
        render_tours(result),
    ]
    body = "\n\n".join(block for block in blocks if block) + "\n"
    return (quick_banner() + "\n" + body) if quick else body


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)

    seed = cfg.seed if args.seed is None else args.seed
    n_bins = cfg.roads.n_bins if args.n_bins is None else args.n_bins
    out_dir = cfg.paths.resolve("results_dir") if args.out_dir is None else args.out_dir

    if args.quick and args.n_bins is None:
        # Only override what the user did not ask for explicitly.
        n_bins = QUICK_N_BINS

    print("Phase 4 roads: Euclidean vs real road distance on the same bin set")
    if args.quick:
        print("  QUICK SMOKE RUN -- not for publication")
    print(f"  area={cfg.roads.area_name}")
    print(f"  box={cfg.roads.box_km:g} km  network={cfg.roads.network_type}  bins={n_bins}")
    print(f"  seed={seed}  downloads={'DISABLED' if args.no_download else 'allowed'}")
    print(f"  {SYNTHETIC_BINS_NOTE}")
    print(f"  {OSM_CREDIT}")
    print()

    result = run_roads_eval(cfg, seed, n_bins=n_bins, allow_download=not args.no_download)

    report = build_report(result, quick=args.quick)
    print(report)

    print("Verdict: by how much does a straight line under-state a road distance?")
    for line in detour_verdict(result):
        print(line)
    print()

    csv_path, md_path = write_outputs(out_dir, result, report)
    print(f"wrote {csv_path.name} and {md_path.name} to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
