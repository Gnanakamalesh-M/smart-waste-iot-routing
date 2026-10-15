"""Checks that every claim in ``README.md`` still points at the document it came from.

CLAUDE.md requires that every number in the README is pasted from real output and names the
command that produced it. Nothing enforced that: a results doc could be renamed, a heading
reworded or a figure retyped slightly, and the README would go on looking authoritative. This
module is the enforcement, and ``tests/test_readme.py`` runs it on the real committed files.

Three rules, in increasing strictness:

**Links.** Every markdown link target and every inline-code repo path in the README must resolve
on disk, and every ``docs/X.md#anchor`` anchor must match a heading that really exists in that
file (GitHub's own slug rules). A renamed doc or a reworded heading fails here rather than
rotting quietly.

**Numbers, per line.** On any prose line that links to a specific ``docs/*.md``, every numeric
token on that line must appear **verbatim in a document linked from that same line**. A match in
some unrelated doc does not count: that is the whole point, since a figure like ``0.29`` occurs in
more than one place and a coincidental hit elsewhere would let an unsourced number pass as
sourced.

**Numbers in Key findings.** Inside the README's "Key findings" section the per-line rule is not
optional: a line that carries a number but links to no document is itself a violation. Every
finding must name its source.

Everywhere else, a number must at least appear verbatim in some ``docs/*.md``, unless it is listed
in :data:`EXEMPT` with a stated reason -- the Python version, the CI run's own identifiers, and the
test counts this phase measured itself, none of which any results doc contains.

Two deliberate exclusions, both so the checker measures claims rather than syntax:

* **Fenced code blocks are not checked.** They are commands, and the digits in ``--seed 42`` or
  ``--capacities 150 400 8000`` are flags, not findings.
* **Link targets, URLs and file paths are stripped before numbers are extracted**, because
  ``docs/PHASE1_RESULTS.md`` is a filename and its ``1`` is not a measurement.

Matching is a plain verbatim substring search. No rounding, no reformatting, no normalising of
thousands separators -- which is what forces the README to paste a figure instead of recomputing
it.

**Stdlib only**, so the ``unit`` chunk runs it with the ``[dev]`` extra alone.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from smartwaste.config import REPO_ROOT

#: The README this module checks.
README_FILENAME = "README.md"

#: Directory holding every document a README number may be sourced from.
DOCS_DIRNAME = "docs"

#: Heading that opens the section where the per-line source rule is mandatory. Matched as a
#: prefix, so the heading's own wording may grow without breaking the check.
KEY_FINDINGS_HEADING_PREFIX = "## Key findings"

#: File extensions treated as repo paths when they appear in inline code or a link target.
PATH_SUFFIXES = (
    ".md",
    ".py",
    ".ino",
    ".h",
    ".yaml",
    ".yml",
    ".toml",
    ".csv",
    ".json",
)

#: Paths the README names precisely BECAUSE they do not exist: spec items that were never built,
#: and the stale target layout in ``docs/SPEC.md`` section 6. Saying "there is no ``simulate.py``"
#: must not be a dead-path failure, so each is listed here with the reason it is absent.
KNOWN_ABSENT_PATHS: tuple[tuple[str, str], ...] = (
    ("docs/INTERVIEW.md", "listed under Phase 6 in docs/SPEC.md and docs/PHASES.md; not written"),
    ("simulate.py", "docs/SPEC.md section 6 target layout; the real module is bin_sim.py"),
    ("baselines.py", "docs/SPEC.md section 6 target layout; the real module is strategies.py"),
    ("metrics.py", "docs/SPEC.md section 6 target layout; the real module is cost.py"),
    ("scripts/run_sim.py", "docs/SPEC.md section 6 target layout; never created"),
)

_KNOWN_ABSENT = frozenset(path for path, _reason in KNOWN_ABSENT_PATHS)

#: Numeric tokens allowed in the README without appearing in any results doc, each with the
#: reason it cannot be sourced to one. Checked as exact tokens, not substrings.
EXEMPT: tuple[tuple[str, str], ...] = (
    ("3.11", "the Python version; sourced to pyproject.toml's requires-python"),
    ("3", "section, phase and step numbering, and the 3-sensor array"),
    ("6", "phase and section numbering"),
    ("38689900399", "the CI run id; sourced to the run URL quoted beside it"),
    ("1", "the CI run numbering (CI #1) and ordinal list markers"),
    ("2", "the CI run numbering (CI #2)"),
    ("4", "the CI run numbering (CI #4)"),
    ("5", "the CI run numbering (CI #5) and the count of results docs"),
    ("8", "the CI run's duration in seconds; sourced to the run URL"),
    ("2026", "the date of the CI run and of this phase"),
    ("10", "the date of the CI run and of this phase"),
    ("15", "the date of the CI run and of this phase"),
    ("451", "test count measured by this phase's own run, command given beside it"),
    ("125", "test count measured by this phase's own run, command given beside it"),
    ("107", "test count measured by this phase's own run, command given beside it"),
    ("14", "test count measured by this phase's own run, command given beside it"),
    ("17", "test count measured by this phase's own run, command given beside it"),
    ("710", "test count measured by this phase's own run, command given beside it"),
)

_EXEMPT_TOKENS = frozenset(token for token, _reason in EXEMPT)

_LINK_RE = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
_INLINE_CODE_RE = re.compile(r"`([^`]+)`")
_URL_RE = re.compile(r"https?://\S+")
_PATH_LIKE_RE = re.compile(r"[\w.\-]+(?:[/\\][\w.\-]+)+")
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
#: An abbreviated git commit sha. Only matched when it contains a hex letter, so a plain number
#: is never mistaken for one and quietly dropped from the number check.
_SHA_RE = re.compile(r"\b(?=[0-9a-f]*[a-f])[0-9a-f]{7,40}\b")
_FENCE_RE = re.compile(r"^\s*```")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*$")


class ReadmeCheckError(Exception):
    """Raised when the README cannot be checked at all, as opposed to failing a check."""


@dataclass(frozen=True)
class Violation:
    """One failure, reported with the line it is on so the message is actionable."""

    line_no: int
    kind: str
    message: str

    def __str__(self) -> str:
        return f"{README_FILENAME}:{self.line_no}: [{self.kind}] {self.message}"


def slugify_heading(text: str) -> str:
    """Return GitHub's anchor slug for a markdown heading's text.

    GitHub lowercases, drops everything that is not a word character, space or hyphen (markdown
    emphasis and backticks included), then turns spaces into hyphens.
    """
    plain = re.sub(r"[`*_]", "", text)
    plain = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", plain)
    kept = re.sub(r"[^\w\s-]", "", plain.lower())
    return re.sub(r"\s+", "-", kept.strip())


def heading_slugs(text: str) -> set[str]:
    """Every anchor slug a markdown document offers, including GitHub's ``-1`` duplicates."""
    slugs: set[str] = set()
    for line in _iter_prose_lines(text):
        match = _HEADING_RE.match(line.text)
        if match is None:
            continue
        base = slugify_heading(match.group(2))
        if not base:
            continue
        slug = base
        suffix = 1
        while slug in slugs:
            slug = f"{base}-{suffix}"
            suffix += 1
        slugs.add(slug)
    return slugs


@dataclass(frozen=True)
class _Line:
    line_no: int
    text: str
    in_key_findings: bool


def _iter_prose_lines(text: str) -> list[_Line]:
    """The README's lines with fenced code blocks removed.

    Code blocks are commands rather than claims, so their digits and paths are not checked.
    """
    lines: list[_Line] = []
    in_fence = False
    in_key_findings = False
    for line_no, raw in enumerate(text.splitlines(), start=1):
        if _FENCE_RE.match(raw):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if raw.startswith("## "):
            in_key_findings = raw.startswith(KEY_FINDINGS_HEADING_PREFIX)
        lines.append(_Line(line_no=line_no, text=raw, in_key_findings=in_key_findings))
    return lines


def _looks_like_path(candidate: str) -> bool:
    """A bare filename, not a command. A span with whitespace is a command line, not a path."""
    return candidate.endswith(PATH_SUFFIXES) and not any(ch.isspace() for ch in candidate)


def _normalise(candidate: str) -> str:
    return candidate.replace("\\", "/").removeprefix("./")


def link_targets(line: str) -> list[str]:
    """Markdown link targets on one line."""
    return _LINK_RE.findall(line)


def code_paths(line: str) -> list[str]:
    """Inline-code spans on one line that name a repo file."""
    return [span for span in _INLINE_CODE_RE.findall(line) if _looks_like_path(span.strip())]


def docs_cited(line: str) -> list[str]:
    """Documents under ``docs/`` cited on one line, by link or by inline-code path.

    This is what the per-line number rule checks against, so a bare mention in prose does not
    count: the line has to actually point at the file.
    """
    cited: list[str] = []
    for candidate in [*link_targets(line), *code_paths(line)]:
        target = _normalise(candidate.split("#", 1)[0].strip())
        # A row may cite the same document both as a link and as an inline-code path.
        is_doc = target.startswith(f"{DOCS_DIRNAME}/") and target.endswith(".md")
        if is_doc and target not in cited:
            cited.append(target)
    return cited


def numeric_tokens(line: str) -> list[str]:
    """Numbers claimed on one line, with filenames, URLs and link targets stripped first."""
    stripped = _LINK_RE.sub(lambda m: m.group(0).split("](")[0], line)
    stripped = _URL_RE.sub(" ", stripped)
    stripped = _SHA_RE.sub(" ", stripped)
    stripped = _PATH_LIKE_RE.sub(" ", stripped)
    return _NUMBER_RE.findall(stripped)


def check_links(readme_text: str, repo_root: Path) -> list[Violation]:
    """Every link target and inline-code repo path resolves; every anchor exists."""
    violations: list[Violation] = []
    doc_slugs: dict[str, set[str]] = {}
    for line in _iter_prose_lines(readme_text):
        for raw_target in link_targets(line.text):
            if _URL_RE.match(raw_target) or raw_target.startswith("mailto:"):
                continue
            path_part, _, anchor = raw_target.partition("#")
            target = _normalise(path_part.strip())
            if not target:
                continue
            resolved = repo_root / target
            if target in _KNOWN_ABSENT:
                continue
            if not resolved.exists():
                violations.append(
                    Violation(line.line_no, "dead-link", f"{target} does not exist in the repo")
                )
                continue
            if anchor and target.endswith(".md"):
                if target not in doc_slugs:
                    doc_slugs[target] = heading_slugs(resolved.read_text(encoding="utf-8"))
                if anchor not in doc_slugs[target]:
                    violations.append(
                        Violation(
                            line.line_no,
                            "dead-anchor",
                            f"{target} has no heading matching #{anchor}",
                        )
                    )
        for span in code_paths(line.text):
            target = _normalise(span.strip())
            if target in _KNOWN_ABSENT:
                continue
            if not (repo_root / target).exists():
                violations.append(
                    Violation(
                        line.line_no, "dead-path", f"inline-code path {target} does not exist"
                    )
                )
    return violations


def _load_docs(repo_root: Path) -> dict[str, str]:
    docs_dir = repo_root / DOCS_DIRNAME
    if not docs_dir.is_dir():
        raise ReadmeCheckError(f"no {DOCS_DIRNAME}/ directory under {repo_root}")
    return {
        f"{DOCS_DIRNAME}/{path.name}": path.read_text(encoding="utf-8")
        for path in sorted(docs_dir.glob("*.md"))
    }


def check_numbers(readme_text: str, repo_root: Path) -> list[Violation]:
    """Every number is sourced: per-line against the docs that line cites, else anywhere."""
    docs = _load_docs(repo_root)
    violations: list[Violation] = []
    for line in _iter_prose_lines(readme_text):
        tokens = numeric_tokens(line.text)
        if not tokens:
            continue
        cited = docs_cited(line.text)
        if not cited:
            if line.in_key_findings:
                violations.append(
                    Violation(
                        line.line_no,
                        "unsourced-finding",
                        "a Key findings line carries numbers "
                        f"({', '.join(sorted(set(tokens)))}) but links to no docs/*.md",
                    )
                )
                continue
            for token in tokens:
                if token in _EXEMPT_TOKENS:
                    continue
                if not any(token in text for text in docs.values()):
                    violations.append(
                        Violation(
                            line.line_no,
                            "unsourced-number",
                            f"{token} appears in no docs/*.md and is not in EXEMPT",
                        )
                    )
            continue
        cited = [name for name in cited if name not in _KNOWN_ABSENT]
        if not cited:
            continue
        missing_docs = [name for name in cited if name not in docs]
        if missing_docs:
            violations.append(
                Violation(
                    line.line_no,
                    "unknown-doc",
                    f"cited document(s) not found: {', '.join(missing_docs)}",
                )
            )
            continue
        for token in tokens:
            if token in _EXEMPT_TOKENS:
                continue
            if not any(token in docs[name] for name in cited):
                violations.append(
                    Violation(
                        line.line_no,
                        "number-not-in-cited-doc",
                        f"{token} does not appear verbatim in {' or '.join(cited)}",
                    )
                )
    return violations


def check_readme(repo_root: Path | None = None) -> list[Violation]:
    """Run every check against the committed README. An empty list means the README is sourced."""
    root = REPO_ROOT if repo_root is None else repo_root
    readme = root / README_FILENAME
    if not readme.is_file():
        raise ReadmeCheckError(f"{readme} does not exist")
    text = readme.read_text(encoding="utf-8")
    return [*check_links(text, root), *check_numbers(text, root)]
