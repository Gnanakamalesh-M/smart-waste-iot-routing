"""Pull a ThingSpeak channel and classify each reading with the project's own fusion rule.

    python tools\\thingspeak_fetch.py --limit 10

Reads ``THINGSPEAK_CHANNEL_ID`` and ``THINGSPEAK_READ_API_KEY`` from ``.env`` (gitignored; copy
``.env.example`` and fill it in). To run with no network at all, against the committed fixture:

    python tools\\thingspeak_fetch.py --offline tests\\data\\thingspeak_sample.json

All logic lives in :mod:`smartwaste.thingspeak`; this file only parses arguments and prints --
the same split every ``scripts\\run_*.py`` has from its ``smartwaste.*_eval`` module. Tests
import the package, never this file.

**The read key is never printed, and neither is the URL that contains it.** Not in normal output,
not in an error message. See :func:`smartwaste.thingspeak.redact`.

**NOT VERIFIED: no ThingSpeak channel has ever been contacted by this project.** There is no
channel and no real key; the only exercised path is ``--offline``. See ``docs/PHASE5_RESULTS.md``.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from smartwaste import load_config
from smartwaste.config import REPO_ROOT
from smartwaste.thingspeak import (
    DEFAULT_N_RESULTS,
    FetchReport,
    ThingSpeakError,
    classify_rows,
    fetch_body,
    parse_feeds,
    read_env_file,
    settings_from_env,
)

DEFAULT_ENV_PATH = REPO_ROOT / ".env"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=None, help="config YAML (default: repo)")
    parser.add_argument(
        "--env", type=Path, default=None, help="path to .env (default: repo root .env)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_N_RESULTS,
        help=f"rows to request from the channel (default: {DEFAULT_N_RESULTS})",
    )
    parser.add_argument(
        "--offline",
        type=Path,
        default=None,
        help="read a saved JSON response instead of the network (no key needed)",
    )
    return parser.parse_args(argv)


def render(report: FetchReport, body_source: str) -> list[str]:
    """Format a report for stdout. Never receives the key, so it cannot leak it."""
    lines = [f"source: {body_source}"]
    if report.channel_name:
        lines.append(f"channel: {report.channel_name}")
    lines.append(f"rows in response: {report.n_raw}")
    lines.append(f"rows usable:      {len(report.rows)}")
    lines.append(f"rows skipped:     {report.n_skipped}")
    if report.is_empty:
        lines.append("")
        lines.append("The channel is empty. Nothing has ever been posted to it, or the window")
        lines.append("requested contains no entries. This is not an error.")
    for reason in report.skip_reasons:
        lines.append(f"  SKIPPED {reason}")
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)

    if args.offline is not None:
        if not args.offline.is_file():
            print(f"ERROR: --offline file not found: {args.offline}")
            return 2
        body = args.offline.read_text(encoding="utf-8")
        source = f"{args.offline} (offline; no network used)"
    else:
        env_path = DEFAULT_ENV_PATH if args.env is None else args.env
        env = read_env_file(env_path)
        try:
            settings = settings_from_env(env, n_results=args.limit)
            body = fetch_body(settings)
        except ThingSpeakError as exc:
            # Already redacted by the library; printed as-is rather than re-raised so a
            # traceback cannot carry the URL into a terminal scrollback.
            print(f"ERROR: {exc}")
            return 1
        source = f"ThingSpeak channel {settings.channel_id} (read key REDACTED)"

    try:
        report = parse_feeds(body, n_ultrasonic=cfg.sensors.n_ultrasonic)
    except ThingSpeakError as exc:
        print(f"ERROR: {exc}")
        return 1

    print("SIMULATED OR FIXTURE DATA UNLESS A REAL CHANNEL WAS CONFIGURED.")
    print("Firmware is NOT VERIFIED: never compiled, never flashed, never read from hardware.")
    print()
    for line in render(report, source):
        print(line)

    classified = classify_rows(report.rows, cfg.bin)
    if classified:
        print()
        header = f"{'entry':>7}  {'created_at':<22}  {'d_bar cm':>9}  {'W kg':>8}  "
        header += f"{'Fv %':>7}  {'overload':>8}  state"
        print(header)
        print("-" * len(header))
        for row, result in classified:
            d_bar = sum(row.distances_cm) / len(row.distances_cm)
            print(
                f"{row.entry_id:>7}  {row.created_at:<22}  {d_bar:>9.2f}  "
                f"{row.weight_kg:>8.3f}  {result.fill_pct:>7.2f}  "
                f"{'yes' if result.overloaded else 'no':>8}  {result.state}"
            )
        print()
        print("States come from smartwaste.fusion.fuse -- the same rule strategy (d) uses.")
        print("ThingSpeak field5/field6 are IGNORED: the point is to re-derive, not to trust.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
