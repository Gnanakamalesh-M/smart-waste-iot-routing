"""Tests for the README source checker, and for the committed README itself.

Two different jobs here, and the second is the one that matters.

**The real README must pass.** Every link resolves, every anchor exists, and every number appears
verbatim in a document linked from its own line. This is the check that stops the README drifting
away from `docs/PHASE*_RESULTS.md` -- a renamed doc, a reworded heading or a retyped figure fails
here.

**The checker must be able to fail.** A guard nobody has watched fail is a guard nobody knows
works, so every rule gets a hand-built repo that violates it: a dead link, a dead anchor, an
invented number, a Key findings line with no source, and -- the case that motivates the per-line
rule at all -- a line that cites one doc while its number only appears in another. If any of
those passed, the green result above would mean nothing.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from smartwaste.readme_check import (
    EXEMPT,
    KEY_FINDINGS_HEADING_PREFIX,
    KNOWN_ABSENT_PATHS,
    README_FILENAME,
    ReadmeCheckError,
    Violation,
    check_links,
    check_numbers,
    check_readme,
    docs_cited,
    heading_slugs,
    numeric_tokens,
    slugify_heading,
)

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------------------------
# The committed README
# --------------------------------------------------------------------------------------------


def test_the_committed_readme_passes_every_check(repo_root: Path) -> None:
    violations = check_readme(repo_root)
    assert violations == [], "\n".join(str(v) for v in violations)


def test_the_readme_exists_and_has_a_key_findings_section(repo_root: Path) -> None:
    text = (repo_root / README_FILENAME).read_text(encoding="utf-8")
    assert any(line.startswith(KEY_FINDINGS_HEADING_PREFIX) for line in text.splitlines()), (
        "the per-line source rule is anchored to this heading"
    )


def test_the_readme_really_does_carry_sourced_numbers(repo_root: Path) -> None:
    """Non-vacuity: the green result above must not come from a README with nothing in it."""
    text = (repo_root / README_FILENAME).read_text(encoding="utf-8")
    sourced = [
        line
        for line in text.splitlines()
        if docs_cited(line) and numeric_tokens(line) and not line.startswith("#")
    ]
    assert len(sourced) >= 10, "expected the key findings to be numbers with doc links beside them"


def test_the_readme_never_calls_the_firmware_tested_or_verified(repo_root: Path) -> None:
    """The firmware has never been compiled. CLAUDE.md forbids claiming otherwise."""
    text = (repo_root / README_FILENAME).read_text(encoding="utf-8").lower()
    for sentence in text.replace("\n", " ").split("."):
        if "firmware" not in sentence and "sketch" not in sentence:
            continue
        for banned in ("is tested", "is working", "is verified", "works on hardware"):
            assert banned not in sentence, f"forbidden firmware claim: {sentence.strip()!r}"


def test_every_exemption_and_absent_path_states_a_reason() -> None:
    for token, reason in EXEMPT:
        assert token and reason.strip(), f"EXEMPT entry {token!r} has no reason"
    for path, reason in KNOWN_ABSENT_PATHS:
        assert path and reason.strip(), f"KNOWN_ABSENT_PATHS entry {path!r} has no reason"


# --------------------------------------------------------------------------------------------
# Slugs and tokens
# --------------------------------------------------------------------------------------------


def test_slugify_matches_githubs_rules() -> None:
    assert slugify_heading("Key findings") == "key-findings"
    assert slugify_heading("## `--quick` runs, 42% faster") == "--quick-runs-42-faster"
    assert slugify_heading("**Strategy benchmark: NOT RUN**") == "strategy-benchmark-not-run"


def test_duplicate_headings_get_githubs_numbered_slugs() -> None:
    slugs = heading_slugs("# Scope\n\n# Scope\n\n# Scope\n")
    assert slugs == {"scope", "scope-1", "scope-2"}


def test_filenames_urls_and_shas_are_not_mistaken_for_claims() -> None:
    assert numeric_tokens("see [docs/PHASE1_RESULTS.md](docs/PHASE1_RESULTS.md)") == []
    assert numeric_tokens("commit `3de97a7`, run https://example.com/runs/38689900399") == []
    # `p90` keeps its 90: a percentile label is part of the claim, so it is checked too.
    assert numeric_tokens("median 1.4514x at p90 2.1370") == ["1.4514", "90", "2.1370"]


def test_code_fences_are_not_checked(tmp_path: Path) -> None:
    """Commands are not claims: the digits in `--seed 42` must not need a source."""
    _write_docs(tmp_path)
    _write_readme(tmp_path, "```\npython scripts\\run_x.py --seed 999999\n```\n")
    assert check_numbers(_readme_text(tmp_path), tmp_path) == []


# --------------------------------------------------------------------------------------------
# The checker must be able to fail
# --------------------------------------------------------------------------------------------


def _write_docs(root: Path) -> None:
    """A miniature repo: two docs, each holding a number the other does not."""
    docs = root / "docs"
    docs.mkdir(exist_ok=True)
    (docs / "PHASE1_RESULTS.md").write_text(
        "# Phase 1 results\n\n## Honest summary\n\nmisclassification 26.73% here.\n",
        encoding="utf-8",
    )
    (docs / "PHASE2_RESULTS.md").write_text(
        "# Phase 2 results\n\ncost reduction 66.67% here.\n", encoding="utf-8"
    )


def _readme_text(root: Path) -> str:
    return (root / README_FILENAME).read_text(encoding="utf-8")


def _write_readme(root: Path, body: str) -> None:
    (root / README_FILENAME).write_text(body, encoding="utf-8")


def _kinds(violations: Sequence[Violation]) -> set[str]:
    return {violation.kind for violation in violations}


def test_a_dead_link_fails(tmp_path: Path) -> None:
    _write_docs(tmp_path)
    _write_readme(tmp_path, "See [Phase 9](docs/PHASE9_RESULTS.md) for more.\n")
    assert _kinds(list(check_links(_readme_text(tmp_path), tmp_path))) == {"dead-link"}


def test_a_dead_anchor_fails(tmp_path: Path) -> None:
    _write_docs(tmp_path)
    _write_readme(tmp_path, "See [summary](docs/PHASE1_RESULTS.md#no-such-heading).\n")
    assert _kinds(list(check_links(_readme_text(tmp_path), tmp_path))) == {"dead-anchor"}


def test_a_live_anchor_passes(tmp_path: Path) -> None:
    _write_docs(tmp_path)
    _write_readme(tmp_path, "See [summary](docs/PHASE1_RESULTS.md#honest-summary).\n")
    assert check_links(_readme_text(tmp_path), tmp_path) == []


def test_an_invented_number_fails(tmp_path: Path) -> None:
    _write_docs(tmp_path)
    _write_readme(tmp_path, "Our fusion misclassified only 0.42% of bin-days.\n")
    assert _kinds(list(check_numbers(_readme_text(tmp_path), tmp_path))) == {"unsourced-number"}


def test_a_key_findings_line_with_a_number_and_no_source_fails(tmp_path: Path) -> None:
    _write_docs(tmp_path)
    _write_readme(
        tmp_path,
        f"{KEY_FINDINGS_HEADING_PREFIX}\n\nFusion misclassified 26.73% of bin-days.\n",
    )
    violations = list(check_numbers(_readme_text(tmp_path), tmp_path))
    assert _kinds(violations) == {"unsourced-finding"}


def test_a_key_findings_line_passes_once_it_cites_the_right_doc(tmp_path: Path) -> None:
    _write_docs(tmp_path)
    _write_readme(
        tmp_path,
        f"{KEY_FINDINGS_HEADING_PREFIX}\n\n"
        "Fusion misclassified 26.73% - [Phase 1](docs/PHASE1_RESULTS.md).\n",
    )
    assert check_numbers(_readme_text(tmp_path), tmp_path) == []


def test_a_number_from_another_doc_does_not_count_as_sourced(tmp_path: Path) -> None:
    """The per-line rule: a coincidental match elsewhere must not pass as a citation.

    ``66.67`` is real and is in ``PHASE2_RESULTS.md``, but this line cites Phase 1. Under a
    global "appears in some doc" rule it would pass, which is exactly the hole the per-line rule
    exists to close.
    """
    _write_docs(tmp_path)
    _write_readme(
        tmp_path,
        f"{KEY_FINDINGS_HEADING_PREFIX}\n\n"
        "A calendar cut cost 66.67% - [Phase 1](docs/PHASE1_RESULTS.md).\n",
    )
    violations = list(check_numbers(_readme_text(tmp_path), tmp_path))
    assert _kinds(violations) == {"number-not-in-cited-doc"}
    assert "66.67" in str(violations[0])


def test_a_missing_readme_is_an_error_not_a_pass(tmp_path: Path) -> None:
    with pytest.raises(ReadmeCheckError):
        check_readme(tmp_path)


def test_a_repo_with_no_docs_directory_is_an_error(tmp_path: Path) -> None:
    _write_readme(tmp_path, "nothing to source here.\n")
    with pytest.raises(ReadmeCheckError):
        check_numbers(_readme_text(tmp_path), tmp_path)
