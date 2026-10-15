"""Check that every link and every number in README.md still matches its source document.

    python scripts\\check_readme.py

Prints one line per violation and exits non-zero if there is any, so it can be wired into a
pre-commit hook or a CI job. The rules, and why each exists, are documented in
:mod:`smartwaste.readme_check`; ``tests/test_readme.py`` runs the same checks under the ``unit``
marker, so a drifting README fails the test suite whether or not anyone runs this script.
"""

from __future__ import annotations

from collections.abc import Sequence

from smartwaste.readme_check import README_FILENAME, check_readme


def main(argv: Sequence[str] | None = None) -> int:
    del argv  # no arguments: this script checks the committed README and exits
    violations = check_readme()
    if not violations:
        print(f"{README_FILENAME}: every link resolves and every number is sourced.")
        return 0

    print(f"{README_FILENAME}: {len(violations)} violation(s)")
    print()
    for violation in violations:
        print(f"  {violation}")
    print()
    print("Each number must appear VERBATIM in a docs/*.md linked from its own line.")
    print("Paste the figure from the document; do not recompute or round it.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
