"""Tests for the dashboard's only data path.

Two things are being protected here, and neither is about rendering.

**The honesty labels must survive the trip.** Every warning this project attaches to a table --
``[SMOKE RUN, NOT FOR PUBLICATION]``, ``NOT RUN``, ``fair test``, ``[UPPER BOUND, NOT
REALISTIC]`` -- lives in the markdown heading above it or in a cell inside it. The parser keeps
headings verbatim, so these tests check the labels arrive intact, in the real committed docs and
not just in a hand-built sample.

**A docs rename must fail loudly.** The dashboard asks for tables by heading substring. If a
heading is edited and nothing notices, the panel silently renders nothing -- which is how a
`NOT RUN` warning quietly stops being displayed. So every heading the dashboard requests is
resolved against the real docs here, and :class:`DashboardDataError` is checked to actually fire.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from smartwaste.dashboard_data import (
    BIN_DAYS_FIELDS,
    BIN_DAYS_FILENAME,
    BINS_FIELDS,
    BINS_FILENAME,
    DEMO_DIR,
    DEPOT_BIN_ID,
    KPIS_FIELDS,
    KPIS_FILENAME,
    META_FILENAME,
    NOT_VERIFIED_HEADING,
    REQUIRED_TABLE_HEADINGS,
    RESULTS_DOCS,
    ROUTES_FIELDS,
    ROUTES_FILENAME,
    SIMULATED_DATA_BANNER,
    DashboardDataError,
    find_tables,
    load_demo,
    load_doc_tables,
    parse_markdown_tables,
    require_table,
    section_text,
)

pytestmark = pytest.mark.unit

#: Simulation modules the dashboard path must never reach. A dashboard that can import the
#: simulator is a dashboard that can be made to run it.
FORBIDDEN_IMPORTS = (
    "city_sim",
    "benchmark",
    "phase3",
    "phase4",
    "smartwaste.ml",
    "strategies",
    "demand",
)

SAMPLE_DOC = """# Phase N results

## How these numbers were produced

Some prose that is not a table.

### Table A - capacity-free, the fair comparison [SMOKE RUN, NOT FOR PUBLICATION]

| solver | distance km | trucks |
|---|---|---|
| nearest neighbour | 595.12 | 5 |
| 2-opt [Phase 2 baseline] | 560.19 | 5 |

Prose between the tables.

### Headline: misclassification by noise scenario

| scenario | fairness | rate |
|---|:---:|---|
| clean | control | 0.0000 |
| noisy | fair test | 0.0131 |
| uneven_center_peak | favors fusion by construction | 0.0402 |

## Strategy benchmark: NOT RUN

| what | state |
|---|---|
| the full sweep | **NOT RUN** |

## NOT VERIFIED

- Only 1 physical bin was built.
- Everything at city scale is simulation.

## Something after

Not part of the NOT VERIFIED section.
"""


# --- the parser ------------------------------------------------------------------------------


def test_parses_every_table_in_a_document() -> None:
    tables = parse_markdown_tables(SAMPLE_DOC, "SAMPLE.md")
    assert len(tables) == 3


def test_each_table_carries_the_heading_above_it_verbatim() -> None:
    tables = parse_markdown_tables(SAMPLE_DOC, "SAMPLE.md")
    assert tables[0].heading == (
        "Table A - capacity-free, the fair comparison [SMOKE RUN, NOT FOR PUBLICATION]"
    )
    assert tables[1].heading == "Headline: misclassification by noise scenario"
    assert tables[2].heading == "Strategy benchmark: NOT RUN"


def test_the_smoke_run_label_is_not_stripped_from_the_heading() -> None:
    table = require_table(parse_markdown_tables(SAMPLE_DOC, "SAMPLE.md"), "Table A")
    assert "[SMOKE RUN, NOT FOR PUBLICATION]" in table.heading


def test_the_not_run_label_survives_inside_a_cell() -> None:
    table = require_table(parse_markdown_tables(SAMPLE_DOC, "SAMPLE.md"), "NOT RUN")
    assert table.rows[0] == ("the full sweep", "**NOT RUN**")


def test_the_fairness_column_and_its_three_values_survive() -> None:
    table = require_table(parse_markdown_tables(SAMPLE_DOC, "SAMPLE.md"), "misclassification")
    assert table.columns == ("scenario", "fairness", "rate")
    labels = [row[1] for row in table.rows]
    assert labels == ["control", "fair test", "favors fusion by construction"]


def test_the_alignment_separator_is_not_mistaken_for_a_data_row() -> None:
    # `|---|:---:|---|` is a separator, not a row: a parser that kept it would show a row of
    # dashes in the dashboard.
    table = require_table(parse_markdown_tables(SAMPLE_DOC, "SAMPLE.md"), "misclassification")
    assert table.n_rows == 3
    assert all("---" not in cell for row in table.rows for cell in row)


def test_records_the_source_document() -> None:
    tables = parse_markdown_tables(SAMPLE_DOC, "SAMPLE.md")
    assert {t.source_doc for t in tables} == {"SAMPLE.md"}


def test_records_the_heading_level() -> None:
    tables = parse_markdown_tables(SAMPLE_DOC, "SAMPLE.md")
    assert tables[0].level == 3
    assert tables[2].level == 2


def test_a_document_with_no_tables_yields_nothing() -> None:
    assert parse_markdown_tables("# Title\n\nJust prose.\n", "X.md") == ()


def test_a_header_without_a_separator_is_not_a_table() -> None:
    # A prose line that happens to contain pipes must not become a one-column table.
    assert parse_markdown_tables("| not | a | table |\n\nprose\n", "X.md") == ()


def test_as_dicts_keys_rows_by_column_name() -> None:
    table = require_table(parse_markdown_tables(SAMPLE_DOC, "SAMPLE.md"), "Table A")
    assert table.as_dicts()[0] == {
        "solver": "nearest neighbour",
        "distance km": "595.12",
        "trucks": "5",
    }


def test_as_dicts_pads_a_short_row_rather_than_dropping_it() -> None:
    # A malformed row in a doc should be visible in the dashboard, not silently disappear.
    doc = "## H\n\n| a | b | c |\n|---|---|---|\n| 1 | 2 |\n"
    table = require_table(parse_markdown_tables(doc, "X.md"), "H")
    assert table.as_dicts() == ({"a": "1", "b": "2", "c": ""},)


# --- section text ----------------------------------------------------------------------------


def test_section_text_stops_at_the_next_heading_of_the_same_level() -> None:
    body = section_text(SAMPLE_DOC, NOT_VERIFIED_HEADING)
    assert "Only 1 physical bin was built." in body
    assert "Not part of the NOT VERIFIED section." not in body


def test_section_text_raises_on_a_missing_heading() -> None:
    with pytest.raises(DashboardDataError, match="no section"):
        section_text(SAMPLE_DOC, "a heading that is not there")


# --- against the REAL committed docs ---------------------------------------------------------


def test_the_real_docs_parse_into_tables(repo_root: Path) -> None:
    tables = load_doc_tables(repo_root / "docs")
    assert len(tables) > 20
    assert {t.source_doc for t in tables} == set(RESULTS_DOCS)


@pytest.mark.parametrize("heading", REQUIRED_TABLE_HEADINGS)
def test_every_heading_the_dashboard_asks_for_resolves_in_the_real_docs(
    repo_root: Path, heading: str
) -> None:
    # This is the test that fails when a results doc is renamed, instead of the dashboard
    # quietly rendering an empty panel.
    table = require_table(load_doc_tables(repo_root / "docs"), heading)
    assert table.n_rows >= 1
    assert table.columns


def test_all_three_stuck_sensor_fractions_are_shown_not_just_one(repo_root: Path) -> None:
    # docs/ASSUMPTIONS.md records that the (d)-vs-(e) verdict REVERSES between fault fraction 0
    # and 0.02. Showing one of them would let a reader pick the conclusion, so the dashboard
    # asks for a substring that matches all three.
    tables = find_tables(load_doc_tables(repo_root / "docs"), "Headline: all strategies")
    assert len(tables) == 3
    fractions = {t.heading.rsplit("stuck-sensor fraction ", 1)[1].strip() for t in tables}
    assert fractions == {"0", "0.02", "0.1"}


def test_the_phase4_tables_still_carry_their_smoke_run_labels(repo_root: Path) -> None:
    tables = find_tables(load_doc_tables(repo_root / "docs"), "Table ")
    assert tables
    assert all("SMOKE RUN" in t.heading for t in tables)


def test_the_phase3_benchmark_is_still_labelled_not_run(repo_root: Path) -> None:
    table = require_table(load_doc_tables(repo_root / "docs"), "Strategy benchmark: NOT RUN")
    assert "NOT RUN" in table.heading
    assert any("NOT RUN" in cell for row in table.rows for cell in row)


@pytest.mark.parametrize("name", RESULTS_DOCS)
def test_every_results_doc_has_a_not_verified_section(repo_root: Path, name: str) -> None:
    body = section_text((repo_root / "docs" / name).read_text(encoding="utf-8"), "NOT VERIFIED")
    assert len(body) > 200


def test_a_missing_results_doc_raises(tmp_path: Path) -> None:
    with pytest.raises(DashboardDataError, match="results doc not found"):
        load_doc_tables(tmp_path)


def test_require_table_raises_with_an_actionable_message() -> None:
    with pytest.raises(DashboardDataError, match="may have been renamed"):
        require_table((), "nothing matches this")


# --- the committed demo snapshot -------------------------------------------------------------


def test_the_committed_demo_snapshot_loads() -> None:
    city = load_demo()
    assert city.meta.label == SIMULATED_DATA_BANNER
    assert city.meta.n_bins == len(city.bins)
    assert city.meta.days == len(city.kpis)
    assert len(city.bin_days) == city.meta.n_bins * city.meta.days


def test_the_demo_snapshot_is_committed_not_gitignored(repo_root: Path) -> None:
    # It lives in dashboard/demo_data/, NOT results/ -- results/* is gitignored, so a dashboard
    # reading from there would be empty on a fresh clone.
    for name in (BINS_FILENAME, BIN_DAYS_FILENAME, ROUTES_FILENAME, KPIS_FILENAME, META_FILENAME):
        assert (repo_root / "dashboard" / "demo_data" / name).is_file()


def test_the_demo_snapshot_stays_small(repo_root: Path) -> None:
    # A dashboard demo is not a dataset. If this fails, cut the bin count rather than commit a
    # bigger blob -- see docs/PHASE5_RESULTS.md, which reports the measured size.
    total = sum(p.stat().st_size for p in (repo_root / "dashboard" / "demo_data").iterdir())
    assert total < 1_000_000, f"demo_data is {total} bytes; keep it under 1 MB"


def test_every_bin_day_state_is_one_of_the_three() -> None:
    city = load_demo()
    assert {r.true_state for r in city.bin_days} <= {"GREEN", "YELLOW", "RED"}
    assert {r.sensor_state for r in city.bin_days} <= {"GREEN", "YELLOW", "RED"}


def test_the_demo_contains_all_three_states_so_the_map_is_worth_drawing() -> None:
    city = load_demo()
    assert {r.true_state for r in city.bin_days} == {"GREEN", "YELLOW", "RED"}


def test_states_for_day_returns_every_bin_once() -> None:
    city = load_demo()
    rows = city.states_for_day(7)
    assert len(rows) == city.meta.n_bins
    assert sorted(r.bin_id for r in rows) == list(range(city.meta.n_bins))


def test_routes_start_and_end_at_the_depot() -> None:
    city = load_demo()
    routed = city.routes_for_day(7)
    assert routed, "day 7 should have at least one tour"
    for points in routed.values():
        assert points[0].is_depot
        assert points[-1].is_depot
        assert points[0].bin_id == DEPOT_BIN_ID


def test_routes_visit_only_bins_the_strategy_selected() -> None:
    city = load_demo()
    selected = {r.bin_id for r in city.states_for_day(7) if r.selected}
    visited = {p.bin_id for ps in city.routes_for_day(7).values() for p in ps if not p.is_depot}
    assert visited == selected


def test_routes_visit_each_bin_exactly_once() -> None:
    city = load_demo()
    visited = [p.bin_id for ps in city.routes_for_day(7).values() for p in ps if not p.is_depot]
    assert len(visited) == len(set(visited))


def test_a_route_stays_inside_its_own_cluster() -> None:
    city = load_demo()
    cluster_of = {b.bin_id: b.cluster for b in city.bins}
    for cluster, points in city.routes_for_day(7).items():
        assert all(cluster_of[p.bin_id] == cluster for p in points if not p.is_depot)


def test_route_sequence_numbers_are_contiguous_from_zero() -> None:
    city = load_demo()
    for points in city.routes_for_day(7).values():
        assert [p.seq for p in points] == list(range(len(points)))


def test_selected_bins_are_the_ones_the_sensors_called_red() -> None:
    # The demo runs strategy (d), which collects exactly the bins its own fusion called RED.
    # Pinning that is how a generator change that broke the link would be caught.
    city = load_demo()
    for day in city.days:
        rows = city.states_for_day(day)
        assert {r.bin_id for r in rows if r.selected} == {
            r.bin_id for r in rows if r.sensor_state == "RED"
        }


def test_the_sensors_and_the_truth_do_disagree_somewhere() -> None:
    # If they never disagreed the demo would be hiding the one thing it exists to show.
    city = load_demo()
    assert any(r.true_state != r.sensor_state for r in city.bin_days)


def test_kpi_days_cover_the_horizon_in_order() -> None:
    city = load_demo()
    assert [k.day for k in city.kpi_rows()] == list(city.days)


def test_kpis_are_non_negative() -> None:
    city = load_demo()
    for k in city.kpi_rows():
        assert k.distance_km >= 0
        assert k.travel_time_hr >= 0
        assert k.co2_kg >= 0
        assert k.variable_cost >= 0
        assert k.bins_visited >= 0
        assert k.overflow_events >= 0


def test_bins_visited_matches_the_bins_marked_selected() -> None:
    city = load_demo()
    for k in city.kpi_rows():
        assert k.bins_visited == sum(1 for r in city.states_for_day(k.day) if r.selected)


# --- failure modes ---------------------------------------------------------------------------


def test_a_missing_demo_dir_raises_with_the_generator_command(tmp_path: Path) -> None:
    with pytest.raises(DashboardDataError, match=r"make_demo_data\.py"):
        load_demo(tmp_path)


def test_a_wrong_csv_header_raises_rather_than_silently_misreading(tmp_path: Path) -> None:
    for name in (BINS_FILENAME, BIN_DAYS_FILENAME, ROUTES_FILENAME, KPIS_FILENAME):
        (tmp_path / name).write_text("wrong,header\n1,2\n", encoding="utf-8", newline="")
    (tmp_path / META_FILENAME).write_text(
        (DEMO_DIR / META_FILENAME).read_text(encoding="utf-8"), encoding="utf-8"
    )
    with pytest.raises(DashboardDataError, match="expected columns"):
        load_demo(tmp_path)


def test_a_meta_file_missing_a_key_raises(tmp_path: Path) -> None:
    (tmp_path / META_FILENAME).write_text(json.dumps({"seed": 42}), encoding="utf-8")
    with pytest.raises(DashboardDataError, match="missing key"):
        load_demo(tmp_path)


def test_a_bad_boolean_raises(tmp_path: Path) -> None:
    source = DEMO_DIR
    for name in (BINS_FILENAME, ROUTES_FILENAME, KPIS_FILENAME, META_FILENAME):
        (tmp_path / name).write_bytes((source / name).read_bytes())
    header = ",".join(BIN_DAYS_FIELDS)
    row = "0,0,1.0,1.0,1.0,MAYBE,GREEN,GREEN,False"
    (tmp_path / BIN_DAYS_FILENAME).write_text(f"{header}\n{row}\n", encoding="utf-8", newline="")
    with pytest.raises(DashboardDataError, match="expected a boolean"):
        load_demo(tmp_path)


def test_a_row_count_that_contradicts_meta_raises(tmp_path: Path) -> None:
    source = DEMO_DIR
    for name in (BIN_DAYS_FILENAME, ROUTES_FILENAME, KPIS_FILENAME, META_FILENAME):
        (tmp_path / name).write_bytes((source / name).read_bytes())
    header = ",".join(BINS_FIELDS)
    (tmp_path / BINS_FILENAME).write_text(
        f"{header}\n0,1.0,1.0,0,market,0.5,-1\n", encoding="utf-8", newline=""
    )
    with pytest.raises(DashboardDataError, match="expected 200 rows"):
        load_demo(tmp_path)


# --- the firewall ----------------------------------------------------------------------------


def test_the_loader_imports_no_simulation_module(repo_root: Path) -> None:
    """The dashboard's data path must have nothing to run, not merely choose not to run it."""
    source = (repo_root / "src" / "smartwaste" / "dashboard_data.py").read_text(encoding="utf-8")
    import_lines = [
        line
        for line in source.splitlines()
        if line.startswith(("import ", "from ")) or line.strip().startswith(("import ", "from "))
    ]
    joined = "\n".join(import_lines)
    for forbidden in FORBIDDEN_IMPORTS:
        assert forbidden not in joined, f"dashboard_data.py must not import {forbidden}"


def test_the_loader_needs_no_optional_extra(repo_root: Path) -> None:
    source = (repo_root / "src" / "smartwaste" / "dashboard_data.py").read_text(encoding="utf-8")
    import_lines = "\n".join(
        line for line in source.splitlines() if line.startswith(("import ", "from "))
    )
    for wheel in ("streamlit", "plotly", "numpy", "pandas", "sklearn", "ortools", "osmnx"):
        assert wheel not in import_lines, f"dashboard_data.py must not import {wheel}"


def test_the_fields_the_loader_expects_match_the_committed_files(repo_root: Path) -> None:
    """The writer and the reader share these tuples, so a drift is a code bug, not a data bug."""
    directory = repo_root / "dashboard" / "demo_data"
    pairs = (
        (BINS_FILENAME, BINS_FIELDS),
        (BIN_DAYS_FILENAME, BIN_DAYS_FIELDS),
        (ROUTES_FILENAME, ROUTES_FIELDS),
        (KPIS_FILENAME, KPIS_FIELDS),
    )
    for name, fields in pairs:
        first = Path(directory / name).read_text(encoding="utf-8").splitlines()[0]
        assert first == ",".join(fields)


def test_the_committed_demo_files_use_lf_line_endings(repo_root: Path) -> None:
    """Pinned so regenerating on Linux does not show every line as changed.

    Two things have to hold for this to keep passing: the generator writes LF (``csv`` with
    ``lineterminator="\\n"``, and ``newline="\\n"`` for the JSON), and ``.gitattributes`` pins
    ``eol=lf`` so this repo's ``core.autocrlf=true`` does not undo it on the next clone.
    """
    directory = repo_root / "dashboard" / "demo_data"
    for name in (BINS_FILENAME, BIN_DAYS_FILENAME, ROUTES_FILENAME, KPIS_FILENAME, META_FILENAME):
        assert b"\r\n" not in (directory / name).read_bytes(), f"{name} has CRLF line endings"


def test_gitattributes_pins_the_demo_files_to_lf(repo_root: Path) -> None:
    text = (repo_root / ".gitattributes").read_text(encoding="utf-8")
    assert "dashboard/demo_data/*.csv text eol=lf" in text
    assert "dashboard/demo_data/*.json text eol=lf" in text
