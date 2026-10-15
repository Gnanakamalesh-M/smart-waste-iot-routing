"""Headless smoke test for the Streamlit dashboard, plus the firewall that keeps it harmless.

Uses ``streamlit.testing.v1.AppTest``, which runs ``dashboard/app.py`` **in-process**. No server
is started, no port is bound, no browser is launched and nothing touches the network -- a test
suite that boots a web server is a test suite that hangs in CI.

Marked ``dash`` rather than ``unit``, deliberately. This is the only test in the project that
needs the optional ``app`` extra, and giving it its own chunk is what lets the ``unit`` / ``sim``
/ ``ml`` / ``routing`` / ``slow`` chunks genuinely prove they do not need streamlit, instead of
quietly skipping a test inside the chunk that was supposed to prove it. The ``importorskip``
below still means a machine without the extra skips rather than fails.

Two things are being checked, and the second matters more than "the page renders":

1. The app runs end to end without raising, and the **SIMULATED DATA banner is actually on the
   rendered page** -- not merely present in the source.
2. ``app.py`` imports no simulation module, so no interaction can trigger a long job. There is
   nothing on the page to trigger.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from smartwaste.dashboard_data import SIMULATED_DATA_BANNER

if TYPE_CHECKING:
    # Type-only: a runtime import would pull streamlit in at collection time, which is exactly
    # what this file's `importorskip` exists to avoid on a machine without the `app` extra.
    from streamlit.testing.v1 import AppTest

pytestmark = pytest.mark.dash

#: Simulation modules the dashboard must never import. A page that can import the simulator is a
#: page that can be made to run it -- and CLAUDE.md's runtime rule forbids exactly that.
FORBIDDEN_IMPORTS = (
    "city_sim",
    "benchmark",
    "phase3",
    "phase4",
    "smartwaste.ml",
    "strategies",
    "demand",
    "run_strategy",
    "build_layout",
)

#: Generous, but this is a smoke test for a page that only reads a few small files. If it takes
#: longer than this, something is computing rather than reading, which is the bug we care about.
APP_TIMEOUT_S = 90


#: The reason a skip carries, so a skipped run says what to install.
SKIP_REASON = 'the optional "app" extra is not installed: python -m pip install -e ".[dev,app]"'


def apptest_class() -> type[AppTest]:
    """``AppTest``, or a skip if the ``app`` extra is not genuinely installed.

    ``pytest.importorskip("streamlit.testing.v1")`` alone is NOT enough, and this was measured
    rather than guessed. ``pip uninstall streamlit`` leaves the ``streamlit/`` directory behind
    whenever anything has written a ``__pycache__`` inside it -- which a previous test run has --
    and an empty directory is a perfectly valid **namespace package** in Python 3. So the import
    succeeds, ``importorskip`` does not skip, and every test in this file then dies on an
    ``AttributeError`` instead of skipping cleanly. Half-uninstalled is the state a developer is
    actually in after an uninstall, so the guard checks for the attributes it needs.
    """
    streamlit = pytest.importorskip("streamlit", reason=SKIP_REASON)
    if getattr(streamlit, "__version__", None) is None:
        pytest.skip(f"streamlit is present but not importable as a real package; {SKIP_REASON}")
    testing = pytest.importorskip("streamlit.testing.v1", reason=SKIP_REASON)
    cls = getattr(testing, "AppTest", None)
    if cls is None:
        pytest.skip(f"streamlit.testing.v1 has no AppTest; {SKIP_REASON}")
    return cls  # type: ignore[no-any-return]


def app_test(repo_root: Path) -> AppTest:
    """A fresh, unrun AppTest. Skips rather than fails without the optional extra."""
    loaded: AppTest = apptest_class().from_file(
        str(repo_root / "dashboard" / "app.py"), default_timeout=APP_TIMEOUT_S
    )
    return loaded


@pytest.fixture
def rendered(repo_root: Path) -> AppTest:
    """The app, run once.

    Function-scoped, not module-scoped: `repo_root` in `tests/conftest.py` is function-scoped,
    and widening it would mean editing a pre-existing test file. Re-running is cheap anyway --
    the page only reads a few small files, which is the property the suite exists to protect.
    """
    return app_test(repo_root).run()


# --- it runs -----------------------------------------------------------------------------------


def test_the_app_runs_without_raising(rendered: AppTest) -> None:
    assert not rendered.exception, f"dashboard raised: {rendered.exception}"


def test_the_app_renders_a_title(rendered: AppTest) -> None:
    assert any("Smart waste" in block.value for block in rendered.title)


def test_the_banner_is_on_the_rendered_page_not_just_in_the_source(rendered: AppTest) -> None:
    """The banner is the one element whose absence would make the page dishonest."""
    texts = [block.value for block in rendered.error]
    assert any(SIMULATED_DATA_BANNER in text for text in texts), (
        f"the simulated-data banner is missing from the rendered page; errors were: {texts}"
    )


def test_the_banner_appears_in_the_sidebar_too(rendered: AppTest) -> None:
    """So it is visible on every tab, not only the one that happens to be open."""
    sidebar_errors = [block.value for block in rendered.sidebar.error]
    assert any(SIMULATED_DATA_BANNER in text for text in sidebar_errors)


def test_the_not_run_and_smoke_run_warnings_are_rendered(rendered: AppTest) -> None:
    """Phase 3's NOT RUN and Phase 4's SMOKE RUN must reach the page, not be filtered out."""
    all_text = " ".join(
        [b.value for b in rendered.error]
        + [b.value for b in rendered.warning]
        + [b.value for b in rendered.info]
        + [b.value for b in rendered.markdown]
        + [b.value for b in rendered.caption]
    )
    assert "NOT RUN" in all_text
    assert "SMOKE RUN" in all_text


def test_the_page_states_the_firmware_is_unverified(rendered: AppTest) -> None:
    sidebar_text = " ".join(b.value for b in rendered.sidebar.markdown)
    assert "never been compiled" in sidebar_text
    assert "never been flashed" in sidebar_text


def test_the_page_offers_a_day_slider_over_the_demo_horizon(rendered: AppTest) -> None:
    assert len(rendered.slider) == 1
    slider = rendered.slider[0]
    assert slider.min == 0
    assert slider.max == 13  # 14 days, zero-indexed


def test_the_page_offers_a_truth_versus_sensor_toggle(rendered: AppTest) -> None:
    assert len(rendered.radio) == 1
    assert tuple(rendered.radio[0].options) == ("ground truth", "sensor fusion")


def test_the_page_renders_tables(rendered: AppTest) -> None:
    # The baseline comparison is tables, and they must survive even without plotly.
    assert len(rendered.dataframe) >= 5


def test_changing_the_day_does_not_raise(repo_root: Path) -> None:
    """The interaction most likely to break: every panel re-reads for the new day."""
    at = app_test(repo_root).run()
    at.slider[0].set_value(0).run()
    assert not at.exception
    at.slider[0].set_value(13).run()
    assert not at.exception


def test_switching_to_the_sensor_view_does_not_raise(repo_root: Path) -> None:
    at = app_test(repo_root).run()
    at.radio[0].set_value("sensor fusion").run()
    assert not at.exception


def test_every_day_in_the_horizon_renders(repo_root: Path) -> None:
    """Including any day on which no bin was selected and no truck rolled."""
    at = app_test(repo_root).run()
    for day in range(14):
        at.slider[0].set_value(day).run()
        assert not at.exception, f"day {day} raised: {at.exception}"


# --- the firewall ------------------------------------------------------------------------------


def test_the_app_imports_no_simulation_module(repo_root: Path) -> None:
    """The runtime rule, enforced on the source: there is nothing here to trigger."""
    source = (repo_root / "dashboard" / "app.py").read_text(encoding="utf-8")
    import_lines = "\n".join(
        line
        for line in source.splitlines()
        if line.startswith(("import ", "from ")) or line.strip().startswith(("import ", "from "))
    )
    for forbidden in FORBIDDEN_IMPORTS:
        assert forbidden not in import_lines, f"dashboard/app.py must not import {forbidden}"


def test_the_app_imports_only_dashboard_data_from_the_project(repo_root: Path) -> None:
    source = (repo_root / "dashboard" / "app.py").read_text(encoding="utf-8")
    project_imports = {
        line.split()[1] for line in source.splitlines() if line.startswith("from smartwaste")
    }
    assert project_imports == {"smartwaste.dashboard_data"}


def test_plotly_is_imported_lazily_not_at_module_scope(repo_root: Path) -> None:
    """So a missing extra degrades to tables with a message, not a stack trace."""
    source = (repo_root / "dashboard" / "app.py").read_text(encoding="utf-8")
    top_level = [line for line in source.splitlines() if line.startswith(("import ", "from "))]
    assert not any("plotly" in line for line in top_level)
    assert "import plotly.graph_objects as go" in source


def test_the_app_never_writes_anything(repo_root: Path) -> None:
    """A read-only page. No open-for-write, no mkdir, no to_csv anywhere in it."""
    source = (repo_root / "dashboard" / "app.py").read_text(encoding="utf-8")
    for forbidden in ("open(", ".write_text(", ".mkdir(", ".to_csv(", "subprocess"):
        assert forbidden not in source, f"dashboard/app.py must not use {forbidden}"


def test_the_app_does_not_shell_out_or_reach_the_network(repo_root: Path) -> None:
    source = (repo_root / "dashboard" / "app.py").read_text(encoding="utf-8")
    for forbidden in ("urllib", "requests", "socket", "os.system", "Popen"):
        assert forbidden not in source, f"dashboard/app.py must not use {forbidden}"
