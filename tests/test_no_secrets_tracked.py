"""Fail the test suite if git is tracking a secret.

``.gitignore`` is the protection; this is the alarm. The two fail differently and that is the
point: a `.gitignore` rule only helps for a file that has never been added, and
``git add -f``, a rule added after the file, or a new secret-shaped file in a directory nobody
thought about all slip straight past it. So this asks **git** what it is actually tracking, every
time the ``unit`` chunk runs, in this phase and every later one.

Deliberately asks git and not the filesystem. A developer's local
``firmware/nodemcu_smartbin/config.h`` and ``.env`` are *supposed* to exist -- that is how you
build the firmware and run the fetch script. Their existence must never fail a test run. Only
their being *tracked* is a problem.

Skips rather than fails when ``git`` is unavailable or the tree is not a repository, so an
exported tarball or a Docker build without git does not report a phantom security problem.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

#: Exact paths that must never be tracked.
FORBIDDEN_PATHS = ("firmware/nodemcu_smartbin/config.h",)

#: The only ``.env*`` file allowed in git.
ALLOWED_ENV_FILE = ".env.example"

#: Tracked text files to scan for secret-shaped content. Extensions, not a whitelist of names,
#: so a new file in one of these directories is covered without anyone remembering to add it.
SCANNED_SUFFIXES = (".h", ".ino", ".cpp", ".c", ".example", ".env", ".py", ".yaml", ".yml")

#: Files whose *assignments* are scanned -- i.e. the ones where a credential is actually
#: written. ``.py`` is excluded from this scan and this scan only: Python source legitimately
#: contains lines like ``READ_API_KEY_VAR = "THINGSPEAK_READ_API_KEY"``, which assign a variable
#: NAME, not a secret. Including it produced exactly that false positive, and a security check
#: that cries wolf is a security check someone switches off. Python is still covered by the
#: key-shape scan below, which is what would catch a real pasted key.
ASSIGNMENT_SUFFIXES = (".h", ".ino", ".cpp", ".c", ".example", ".env", ".yaml", ".yml")

#: Directories whose tracked files are scanned for secret-shaped content.
SCANNED_PREFIXES = ("firmware/", "tools/", "dashboard/", "config/", "src/")

#: Placeholders this project ships on purpose. A match that equals one of these is fine.
PLACEHOLDERS = frozenset(
    {
        "your_wifi_ssid",
        "your_wifi_password",
        "0000000000000000",
        "replace_me",
        "000000",
        "",
        "abcd1234efgh5678",  # the obviously-fake key used in tests/test_thingspeak.py
    }
)

#: ``#define WIFI_SSID "value"`` / ``KEY=value`` / ``api_key=value``, capturing the value.
_ASSIGNMENT_RES = (
    re.compile(r'#define\s+\w*(?:SSID|PASSWORD|API_KEY)\w*\s+"([^"]*)"', re.IGNORECASE),
    re.compile(
        r"^\s*\w*(?:SSID|PASSWORD|API_KEY)\w*\s*=\s*\"?([^\"\s#]*)\"?",
        re.IGNORECASE | re.MULTILINE,
    ),
)

#: A ThingSpeak key is 16 upper-case alphanumerics. Matching the SHAPE catches a real key that
#: was pasted somewhere no assignment pattern covers.
_KEYLIKE_RE = re.compile(r"\b[A-Z0-9]{16}\b")


def tracked_files() -> list[str]:
    """Every path git is tracking, or a skip when git cannot answer."""
    try:
        completed = subprocess.run(
            ["git", "ls-files"],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        pytest.skip(f"git is unavailable, so tracked files cannot be checked: {exc}")
    if completed.returncode != 0:
        pytest.skip("not a git repository (or git failed), so tracked files cannot be checked")
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


@pytest.fixture(scope="module")
def tracked() -> list[str]:
    return tracked_files()


def test_git_answered_at_all(tracked: list[str]) -> None:
    """Guards the guard: an empty listing would make every test below pass vacuously."""
    assert len(tracked) > 20
    assert "pyproject.toml" in tracked


@pytest.mark.parametrize("path", FORBIDDEN_PATHS)
def test_a_forbidden_path_is_not_tracked(tracked: list[str], path: str) -> None:
    assert path not in tracked, (
        f"{path} is tracked by git and must not be: it holds the Wi-Fi password and the "
        "ThingSpeak WRITE key. Run: git rm --cached "
        f"{path}  -- and if the key was ever pushed, REVOKE it in ThingSpeak first."
    )


def test_only_the_env_example_is_tracked(tracked: list[str]) -> None:
    env_files = [p for p in tracked if p.split("/")[-1].startswith(".env")]
    assert env_files == [ALLOWED_ENV_FILE], (
        f"only {ALLOWED_ENV_FILE} may be tracked, but git is tracking {env_files}"
    )


def test_the_env_example_is_tracked_so_the_project_is_still_usable(tracked: list[str]) -> None:
    # The mirror image: over-zealously gitignoring the example would leave a fresh clone with no
    # way to know which variables to set.
    assert ALLOWED_ENV_FILE in tracked


def test_the_firmware_example_header_is_tracked(tracked: list[str]) -> None:
    assert "firmware/nodemcu_smartbin/config.example.h" in tracked


def test_gitignore_covers_the_firmware_config_header(repo_root: Path) -> None:
    # Belt and braces: the alarm above catches a file already added, this catches the rule going
    # missing, which is what would let the next one in.
    text = (repo_root / ".gitignore").read_text(encoding="utf-8")
    assert "firmware/nodemcu_smartbin/config.h" in text


def scannable(tracked: list[str], suffixes: tuple[str, ...] = SCANNED_SUFFIXES) -> list[str]:
    return [
        path
        for path in tracked
        if path == ALLOWED_ENV_FILE
        or (path.startswith(SCANNED_PREFIXES) and path.endswith(suffixes))
    ]


def test_there_is_something_to_scan(tracked: list[str]) -> None:
    """Guards the guard again: a prefix typo would silently scan nothing."""
    paths = scannable(tracked)
    assert "firmware/nodemcu_smartbin/config.example.h" in paths
    assert ALLOWED_ENV_FILE in paths
    assert len(paths) > 10


def test_the_firmware_and_env_files_are_in_the_assignment_scan(tracked: list[str]) -> None:
    """The assignment scan excludes .py, so check it still covers where credentials really go."""
    paths = scannable(tracked, ASSIGNMENT_SUFFIXES)
    assert "firmware/nodemcu_smartbin/config.example.h" in paths
    assert ALLOWED_ENV_FILE in paths
    assert not any(p.endswith(".py") for p in paths)


def test_no_tracked_file_assigns_a_real_looking_credential(
    repo_root: Path, tracked: list[str]
) -> None:
    """Every SSID / password / api_key assignment in a tracked file must be a placeholder."""
    offenders: list[str] = []
    for path in scannable(tracked, ASSIGNMENT_SUFFIXES):
        full = repo_root / path
        if not full.is_file():
            continue
        text = full.read_text(encoding="utf-8", errors="replace")
        for pattern in _ASSIGNMENT_RES:
            for match in pattern.finditer(text):
                value = match.group(1).strip()
                if value.lower() not in PLACEHOLDERS:
                    offenders.append(f"{path}: {match.group(0).strip()[:80]}")
    assert not offenders, "tracked files assign non-placeholder credentials:\n" + "\n".join(
        offenders
    )


def test_no_tracked_file_contains_a_thingspeak_key_shaped_string(
    repo_root: Path, tracked: list[str]
) -> None:
    """Catches a real 16-character key pasted somewhere no assignment pattern would see."""
    offenders: list[str] = []
    for path in scannable(tracked):
        full = repo_root / path
        if not full.is_file():
            continue
        for number, line in enumerate(
            full.read_text(encoding="utf-8", errors="replace").splitlines(), start=1
        ):
            for match in _KEYLIKE_RE.finditer(line):
                if match.group(0).lower() in PLACEHOLDERS:
                    continue
                offenders.append(f"{path}:{number}: {match.group(0)}")
    assert not offenders, (
        "tracked files contain 16-character key-shaped strings. If any is a real ThingSpeak "
        "key, REVOKE it first, then remove it:\n" + "\n".join(offenders)
    )


def test_the_detector_would_actually_catch_a_real_credential(tmp_path: Path) -> None:
    """A security check nobody has seen fail is a check nobody knows works."""
    planted = '#define WIFI_PASSWORD "hunter2-correct-horse"\n'
    assert any(pattern.search(planted) for pattern in _ASSIGNMENT_RES)
    values = [m.group(1) for p in _ASSIGNMENT_RES for m in p.finditer(planted)]
    assert values and all(v.lower() not in PLACEHOLDERS for v in values)


def test_the_detector_would_catch_a_real_looking_key() -> None:
    assert _KEYLIKE_RE.search("THINGSPEAK_WRITE_API_KEY=QZ7X2M9PLK4RT1WB")
    assert "qz7x2m9plk4rt1wb" not in PLACEHOLDERS


def test_the_detector_accepts_the_shipped_placeholders() -> None:
    """The other failure mode: a check so noisy it gets disabled."""
    shipped = (
        '#define WIFI_SSID "YOUR_WIFI_SSID"\n'
        '#define WIFI_PASSWORD "YOUR_WIFI_PASSWORD"\n'
        '#define THINGSPEAK_WRITE_API_KEY "0000000000000000"\n'
        "THINGSPEAK_READ_API_KEY=replace_me\n"
    )
    values = [m.group(1).strip() for p in _ASSIGNMENT_RES for m in p.finditer(shipped)]
    assert values
    assert all(v.lower() in PLACEHOLDERS for v in values)
