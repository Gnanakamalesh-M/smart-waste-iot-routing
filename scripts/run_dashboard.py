"""Print the command that runs the Phase 5 dashboard. It deliberately does NOT launch it.

    python scripts\\run_dashboard.py

SPEC section 6 lists this path, so it exists -- but as a signpost, not a launcher. Starting a
Streamlit server from inside a script means a long-lived process with a port, a browser and a
signal-handling story, all of which belong to the user's terminal rather than to a script that
might be invoked from a test or a CI job. ``python -m streamlit run`` is one line; this file's
job is to make sure nobody has to remember it.
"""

from __future__ import annotations

from collections.abc import Sequence

from smartwaste.dashboard_data import (
    DEMO_DIR,
    META_FILENAME,
    SIMULATED_DATA_BANNER,
)

COMMAND = "python -m streamlit run dashboard\\app.py"


def main(argv: Sequence[str] | None = None) -> int:
    del argv  # no arguments: this script prints and exits
    print(SIMULATED_DATA_BANNER)
    print()
    print("To run the dashboard, from the repo root:")
    print()
    print(f"    {COMMAND}")
    print()
    print('It needs the optional extra:   python -m pip install -e ".[app]"')
    print()

    if (DEMO_DIR / META_FILENAME).is_file():
        print(f"Demo snapshot found: {DEMO_DIR}")
    else:
        print(f"Demo snapshot MISSING from {DEMO_DIR}.")
        print("Generate it first:   python scripts\\make_demo_data.py --seed 42")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
