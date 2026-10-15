"""Everything the Phase 5 dashboard reads -- and nothing it reads comes from a live simulation.

The dashboard has exactly two data sources, and this module is the only way it reaches either:

1. **A committed demo snapshot** under ``dashboard/demo_data/``, generated once by
   ``scripts\\make_demo_data.py`` and checked into git. 200 bins x 14 days, seed 42. It is tiny
   on purpose: the dashboard must never be able to start a long job, so it cannot start a job at
   all. Nothing here imports :mod:`smartwaste.city_sim`, :mod:`smartwaste.benchmark`,
   :mod:`smartwaste.phase3`, :mod:`smartwaste.phase4` or anything in ``smartwaste.ml``, and
   ``tests/test_dashboard_data.py`` asserts that by reading this file's own source.

2. **The committed results docs**, ``docs/PHASE{1,2,3,4}_RESULTS.md``, parsed as markdown at
   display time. No number is ever transcribed into a second place, so no number can drift out of
   agreement with the document it came from. The parser keeps every heading **verbatim**, which
   is what carries the project's honesty labels -- ``[SMOKE RUN, NOT FOR PUBLICATION]``,
   ``## Strategy benchmark: NOT RUN``, ``[UPPER BOUND, NOT REALISTIC]`` -- so the dashboard
   cannot show a table while dropping the warning attached to it.

**Stdlib only.** No streamlit, no plotly, no numpy, no pandas. The ``unit`` pytest chunk imports
this module with the ``[dev]`` extra alone, which is what keeps the dashboard's optional wheels
genuinely optional.

Every CSV is opened with ``newline=""`` on both the read and the write side, so the committed
files round-trip identically whether they were written on Windows or Linux.
"""

from __future__ import annotations

import csv
import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from smartwaste.config import REPO_ROOT

#: Shown at the top of every dashboard page, and in `meta.json`, from this one definition.
SIMULATED_DATA_BANNER = "SIMULATED DATA, NOT HARDWARE READINGS"

#: Default locations, relative to the repo root.
DEMO_DIR = REPO_ROOT / "dashboard" / "demo_data"
DOCS_DIR = REPO_ROOT / "docs"

#: The results docs the dashboard parses, in phase order.
RESULTS_DOCS = (
    "PHASE1_RESULTS.md",
    "PHASE2_RESULTS.md",
    "PHASE3_RESULTS.md",
    "PHASE4_RESULTS.md",
)

#: Demo snapshot filenames. Shared with ``scripts\\make_demo_data.py`` so the writer and the
#: reader can never disagree about a column name.
BINS_FILENAME = "bins.csv"
BIN_DAYS_FILENAME = "bin_days.csv"
ROUTES_FILENAME = "routes.csv"
KPIS_FILENAME = "kpis.csv"
META_FILENAME = "meta.json"

BINS_FIELDS = ("bin_id", "x_km", "y_km", "cluster", "area_type", "kg_per_pct", "stuck_sensor")
BIN_DAYS_FIELDS = (
    "day",
    "bin_id",
    "true_fill_pct",
    "true_weight_kg",
    "est_fill_pct",
    "est_overloaded",
    "true_state",
    "sensor_state",
    "selected",
)
ROUTES_FIELDS = ("day", "cluster", "seq", "bin_id", "x_km", "y_km")
KPIS_FIELDS = (
    "day",
    "distance_km",
    "travel_time_hr",
    "co2_kg",
    "variable_cost",
    "bins_visited",
    "overflow_events",
)

#: ``bin_id`` of a depot stop in ``routes.csv``. A route's first and last row are the depot.
DEPOT_BIN_ID = -1

#: Fixed float format, so two same-seed runs produce byte-identical files.
FLOAT_FORMAT = "{:.4f}"

#: Headings the dashboard asks for by substring. Kept here, not in ``app.py``, so
#: ``tests/test_dashboard_data.py`` can check every one of them still resolves against the real
#: committed docs -- a docs rename then fails a test instead of silently blanking a panel.
#: Deliberately matches all three stuck-sensor fractions (0, 0.02, 0.1), and the dashboard shows
#: all three. ``docs/ASSUMPTIONS.md`` records that the (d)-vs-(e) verdict *reverses* between
#: 0 and 0.02, so a dashboard that displayed one fault fraction would let the reader pick a
#: conclusion the project refused to pick.
BASELINE_HEADING = "Headline: all strategies, seed 42, stuck-sensor fraction"
REDUCTION_HEADING = "Reduction vs (a) fixed-route, seed 42, stuck-sensor fraction"
SIMPLER_BASELINE_HEADING = "Where a simpler baseline matches or beats the paper's method"
PAPER_REFERENCE_HEADING = "REFERENCE ONLY: the paper's reported percentages"
PHASE3_NOT_RUN_HEADING = "Strategy benchmark: NOT RUN"
#: "Table A" rather than the section title: the table's own heading is where
#: ``[SMOKE RUN, NOT FOR PUBLICATION]`` is written, and that label must travel with the table.
PHASE4_ALGORITHMS_HEADING = "Table A"
PHASE4_CAPACITY_HEADING = "Table B"
PHASE4_EXECUTION_HEADING = "Table C"

#: Section headings read as prose, not as tables.
NOT_VERIFIED_HEADING = "NOT VERIFIED"

#: Every heading the dashboard requires to resolve to at least one table. Order is display
#: order, and ``tests/test_dashboard_data.py`` checks each one against the real committed docs.
REQUIRED_TABLE_HEADINGS = (
    BASELINE_HEADING,
    REDUCTION_HEADING,
    SIMPLER_BASELINE_HEADING,
    PAPER_REFERENCE_HEADING,
    PHASE3_NOT_RUN_HEADING,
    PHASE4_ALGORITHMS_HEADING,
    PHASE4_CAPACITY_HEADING,
    PHASE4_EXECUTION_HEADING,
)

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*$")
#: A markdown table separator: only dashes, colons, pipes and spaces, and at least one dash.
_SEPARATOR_RE = re.compile(r"^\|[\s:|-]*-[\s:|-]*\|$")

_TRUE_STRINGS = frozenset({"true", "1", "yes"})
_FALSE_STRINGS = frozenset({"false", "0", "no"})


class DashboardDataError(RuntimeError):
    """A demo file or a required docs heading is missing or malformed.

    Raised loudly rather than returning an empty table: a dashboard panel that silently shows
    nothing is how a missing honesty label stops being visible.
    """


# --- markdown docs ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DocTable:
    """One pipe table lifted out of a results doc, with the heading that introduced it.

    ``heading`` is the nearest preceding markdown heading, **verbatim** minus its leading
    ``#`` characters. That text is where the honesty labels live, so the dashboard renders it
    unchanged next to the table rather than inventing a caption of its own.
    """

    heading: str
    level: int
    columns: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    source_doc: str

    @property
    def n_rows(self) -> int:
        return len(self.rows)

    def as_dicts(self) -> tuple[dict[str, str], ...]:
        """Rows keyed by column name, for a dataframe-shaped renderer.

        A row with the wrong cell count is padded or truncated to the header width rather than
        dropped: a malformed row in a doc should be visible in the dashboard, not disappear.
        """
        width = len(self.columns)
        out: list[dict[str, str]] = []
        for row in self.rows:
            cells = list(row[:width]) + [""] * max(0, width - len(row))
            out.append(dict(zip(self.columns, cells, strict=True)))
        return tuple(out)


def _split_row(line: str) -> tuple[str, ...]:
    """Split a pipe-table line into stripped cells, dropping the empty edges."""
    inner = line.strip()
    if inner.startswith("|"):
        inner = inner[1:]
    if inner.endswith("|"):
        inner = inner[:-1]
    return tuple(cell.strip() for cell in inner.split("|"))


def _is_table_row(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("|") and stripped.endswith("|") and len(stripped) > 1


def parse_markdown_tables(text: str, source_doc: str) -> tuple[DocTable, ...]:
    """Pull every pipe table out of a markdown document, tagged with its heading.

    A table is a header line, then a ``|---|`` separator, then body rows until the first line
    that is not a table row. Headings are tracked as they go past, so each table carries the
    section it sits under.
    """
    lines = text.splitlines()
    tables: list[DocTable] = []
    heading = ""
    level = 0
    index = 0

    while index < len(lines):
        line = lines[index]
        match = _HEADING_RE.match(line)
        if match is not None:
            level = len(match.group(1))
            heading = match.group(2)
            index += 1
            continue

        if not _is_table_row(line) or index + 1 >= len(lines):
            index += 1
            continue
        if not _SEPARATOR_RE.match(lines[index + 1].strip()):
            index += 1
            continue

        columns = _split_row(line)
        body: list[tuple[str, ...]] = []
        cursor = index + 2
        while cursor < len(lines) and _is_table_row(lines[cursor]):
            body.append(_split_row(lines[cursor]))
            cursor += 1

        tables.append(
            DocTable(
                heading=heading,
                level=level,
                columns=columns,
                rows=tuple(body),
                source_doc=source_doc,
            )
        )
        index = cursor

    return tuple(tables)


def load_doc_tables(docs_dir: Path | None = None) -> tuple[DocTable, ...]:
    """Parse every table out of the four committed results docs, in phase order."""
    directory = DOCS_DIR if docs_dir is None else docs_dir
    tables: list[DocTable] = []
    for name in RESULTS_DOCS:
        path = directory / name
        if not path.is_file():
            raise DashboardDataError(f"results doc not found: {path}")
        tables.extend(parse_markdown_tables(path.read_text(encoding="utf-8"), name))
    return tuple(tables)


def find_tables(tables: Iterable[DocTable], heading_contains: str) -> tuple[DocTable, ...]:
    """Every table whose heading contains ``heading_contains``. May be empty."""
    return tuple(t for t in tables if heading_contains in t.heading)


def require_table(tables: Iterable[DocTable], heading_contains: str) -> DocTable:
    """The first table under a heading containing ``heading_contains``.

    Raises :class:`DashboardDataError` when nothing matches, so a renamed or deleted docs section
    fails a test rather than quietly emptying a dashboard panel.
    """
    found = find_tables(tables, heading_contains)
    if not found:
        raise DashboardDataError(
            f"no table found under a heading containing {heading_contains!r}; "
            "docs/PHASE*_RESULTS.md may have been renamed"
        )
    return found[0]


def section_text(doc_text: str, heading_contains: str) -> str:
    """The body text of the section whose heading contains ``heading_contains``.

    Used for the dashboard's limitations panel, which renders the docs' own ``## NOT VERIFIED``
    prose instead of a restatement that could drift away from it. Stops at the next heading of
    the same level or shallower.
    """
    lines = doc_text.splitlines()
    start: int | None = None
    start_level = 0
    for index, line in enumerate(lines):
        match = _HEADING_RE.match(line)
        if match is None:
            continue
        if start is None:
            if heading_contains in match.group(2):
                start = index + 1
                start_level = len(match.group(1))
            continue
        if len(match.group(1)) <= start_level:
            return "\n".join(lines[start:index]).strip()
    if start is None:
        raise DashboardDataError(f"no section under a heading containing {heading_contains!r}")
    return "\n".join(lines[start:]).strip()


# --- the committed demo snapshot -------------------------------------------------------------


@dataclass(frozen=True)
class DemoMeta:
    """Provenance of the demo snapshot. Every field is printed somewhere on the dashboard."""

    label: str
    seed: int
    n_bins: int
    days: int
    n_clusters: int
    scenario: str
    strategy_key: str
    strategy_label: str
    grid_km: float
    depot_x_km: float
    depot_y_km: float
    stuck_bin_fraction: float
    command: str


@dataclass(frozen=True)
class BinRow:
    bin_id: int
    x_km: float
    y_km: float
    cluster: int
    area_type: str
    kg_per_pct: float
    stuck_sensor: int


@dataclass(frozen=True)
class BinDayRow:
    """One bin on one day: what was true, what the sensors said, whether it was collected.

    Both states are kept because the gap between them is the whole point of the project -- a
    dashboard that showed only the sensor's view would hide every missed pickup.
    """

    day: int
    bin_id: int
    true_fill_pct: float
    true_weight_kg: float
    est_fill_pct: float
    est_overloaded: bool
    true_state: str
    sensor_state: str
    selected: bool


@dataclass(frozen=True)
class RoutePoint:
    day: int
    cluster: int
    seq: int
    bin_id: int
    x_km: float
    y_km: float

    @property
    def is_depot(self) -> bool:
        return self.bin_id == DEPOT_BIN_ID


@dataclass(frozen=True)
class KpiRow:
    day: int
    distance_km: float
    travel_time_hr: float
    co2_kg: float
    variable_cost: float
    bins_visited: int
    overflow_events: int


def _parse_bool(raw: str, field: str) -> bool:
    lowered = raw.strip().lower()
    if lowered in _TRUE_STRINGS:
        return True
    if lowered in _FALSE_STRINGS:
        return False
    raise DashboardDataError(f"{field}: expected a boolean, got {raw!r}")


def _read_rows(path: Path, expected: Sequence[str]) -> tuple[dict[str, str], ...]:
    """Read a demo CSV, checking its header matches what this module expects."""
    if not path.is_file():
        raise DashboardDataError(
            f"demo data file not found: {path}\n"
            "Generate it with: python scripts\\make_demo_data.py --seed 42"
        )
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        header = tuple(reader.fieldnames or ())
        if header != tuple(expected):
            raise DashboardDataError(
                f"{path.name}: expected columns {tuple(expected)}, got {header}"
            )
        return tuple(reader)


@dataclass(frozen=True)
class DemoCity:
    """The whole committed demo snapshot, already parsed.

    Small enough to hold entirely in memory -- which is the point: there is nothing to stream,
    nothing to recompute and no way for a page view to trigger work.
    """

    meta: DemoMeta
    bins: tuple[BinRow, ...]
    bin_days: tuple[BinDayRow, ...]
    routes: tuple[RoutePoint, ...]
    kpis: tuple[KpiRow, ...]

    @property
    def days(self) -> tuple[int, ...]:
        return tuple(range(self.meta.days))

    def states_for_day(self, day: int) -> tuple[BinDayRow, ...]:
        """Every bin's row for one day, in bin order."""
        return tuple(row for row in self.bin_days if row.day == day)

    def routes_for_day(self, day: int) -> dict[int, tuple[RoutePoint, ...]]:
        """The day's tours, keyed by cluster and ordered by stop sequence."""
        grouped: dict[int, list[RoutePoint]] = {}
        for point in self.routes:
            if point.day == day:
                grouped.setdefault(point.cluster, []).append(point)
        return {
            cluster: tuple(sorted(points, key=lambda p: p.seq))
            for cluster, points in sorted(grouped.items())
        }

    def kpi_rows(self) -> tuple[KpiRow, ...]:
        return self.kpis


def _load_meta(meta_path: Path) -> DemoMeta:
    if not meta_path.is_file():
        raise DashboardDataError(
            f"demo metadata not found: {meta_path}\n"
            "Generate it with: python scripts\\make_demo_data.py --seed 42"
        )
    raw = json.loads(meta_path.read_text(encoding="utf-8"))
    try:
        return DemoMeta(
            label=str(raw["label"]),
            seed=int(raw["seed"]),
            n_bins=int(raw["n_bins"]),
            days=int(raw["days"]),
            n_clusters=int(raw["n_clusters"]),
            scenario=str(raw["scenario"]),
            strategy_key=str(raw["strategy_key"]),
            strategy_label=str(raw["strategy_label"]),
            grid_km=float(raw["grid_km"]),
            depot_x_km=float(raw["depot_x_km"]),
            depot_y_km=float(raw["depot_y_km"]),
            stuck_bin_fraction=float(raw["stuck_bin_fraction"]),
            command=str(raw["command"]),
        )
    except KeyError as exc:
        raise DashboardDataError(f"{META_FILENAME} is missing key {exc}") from exc


def load_demo(demo_dir: Path | None = None) -> DemoCity:
    """Read the committed demo snapshot. No simulation, no network, no optional wheels."""
    directory = DEMO_DIR if demo_dir is None else demo_dir
    meta = _load_meta(directory / META_FILENAME)

    bins = tuple(
        BinRow(
            bin_id=int(row["bin_id"]),
            x_km=float(row["x_km"]),
            y_km=float(row["y_km"]),
            cluster=int(row["cluster"]),
            area_type=row["area_type"],
            kg_per_pct=float(row["kg_per_pct"]),
            stuck_sensor=int(row["stuck_sensor"]),
        )
        for row in _read_rows(directory / BINS_FILENAME, BINS_FIELDS)
    )
    bin_days = tuple(
        BinDayRow(
            day=int(row["day"]),
            bin_id=int(row["bin_id"]),
            true_fill_pct=float(row["true_fill_pct"]),
            true_weight_kg=float(row["true_weight_kg"]),
            est_fill_pct=float(row["est_fill_pct"]),
            est_overloaded=_parse_bool(row["est_overloaded"], "est_overloaded"),
            true_state=row["true_state"],
            sensor_state=row["sensor_state"],
            selected=_parse_bool(row["selected"], "selected"),
        )
        for row in _read_rows(directory / BIN_DAYS_FILENAME, BIN_DAYS_FIELDS)
    )
    routes = tuple(
        RoutePoint(
            day=int(row["day"]),
            cluster=int(row["cluster"]),
            seq=int(row["seq"]),
            bin_id=int(row["bin_id"]),
            x_km=float(row["x_km"]),
            y_km=float(row["y_km"]),
        )
        for row in _read_rows(directory / ROUTES_FILENAME, ROUTES_FIELDS)
    )
    kpis = tuple(
        KpiRow(
            day=int(row["day"]),
            distance_km=float(row["distance_km"]),
            travel_time_hr=float(row["travel_time_hr"]),
            co2_kg=float(row["co2_kg"]),
            variable_cost=float(row["variable_cost"]),
            bins_visited=int(row["bins_visited"]),
            overflow_events=int(row["overflow_events"]),
        )
        for row in _read_rows(directory / KPIS_FILENAME, KPIS_FIELDS)
    )

    if len(bins) != meta.n_bins:
        raise DashboardDataError(f"{BINS_FILENAME}: expected {meta.n_bins} rows, got {len(bins)}")
    if len(kpis) != meta.days:
        raise DashboardDataError(f"{KPIS_FILENAME}: expected {meta.days} rows, got {len(kpis)}")

    return DemoCity(meta=meta, bins=bins, bin_days=bin_days, routes=routes, kpis=kpis)
