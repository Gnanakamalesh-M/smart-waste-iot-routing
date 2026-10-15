"""Pull one bin's telemetry out of a ThingSpeak channel and hand it to the Python fusion code.

This is the other half of ``firmware/nodemcu_smartbin/nodemcu_smartbin.ino``: the sketch posts
three ultrasonic distances and a load-cell weight to a ThingSpeak channel, and this module reads
them back, converts each row into the project's existing :class:`~smartwaste.noise.SensorReading`
and lets :func:`smartwaste.fusion.fuse` classify it **unchanged**. No second copy of the rule
exists on this path.

### NOT VERIFIED

**No ThingSpeak channel has ever been contacted.** There is no channel, no real read key and no
reading from the physical bin anywhere in this project. Every test here uses a fake fetcher and
``tests/data/thingspeak_sample.json``. See ``docs/PHASE5_RESULTS.md``.

### The field mapping is INVENTED

The paper says the NodeMCU sends "fill % and weight" to ThingSpeak and **never states which field
carries what**. So this mapping is ours, uncited, and it is the one thing here that a real channel
could silently disagree with:

====================  ==========================================
``field1``/2/3        ultrasonic distances, left / centre / right, **cm**
``field4``            load-cell weight, **kg**
``field5``            ``Fv``, the fused fill percentage (posted by the sketch, re-derived here)
``field6``            state code: 0 GREEN, 1 YELLOW, 2 RED
====================  ==========================================

Fields 5 and 6 are deliberately **not trusted**: this module recomputes ``Fv`` and the state from
fields 1-4, because the point of the exercise is to check that the firmware's arithmetic and the
Python rule agree, not to take the firmware's word for it.

### Two design choices worth the paragraph

**The network is injected, not imported.** :data:`Fetcher` is a plain
``Callable[[str], str]``, and the default :func:`urlopen_fetcher` does its ``urllib.request``
import *inside the function body*. Importing this module therefore cannot touch the network, and
no test needs a socket block or a monkeypatch. The closest existing precedent in the project is
``allow_download: bool`` in :mod:`smartwaste.routing.roads`; a callable is used here because a
fetcher has a return value worth faking.

**The read key is never printed, logged or raised.** Every string this module emits -- stdout,
``repr``, exception text -- passes through :func:`redact` first, including the failure paths.
``tests/test_thingspeak.py`` asserts the key appears in none of them. A secret that reaches a log
file is a leaked secret, and an exception message is a log line.

Nothing here is lenient about data: an empty channel yields zero rows, and a malformed row is
skipped with a stated reason and counted. Neither raises.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from smartwaste.config import BinConfig
from smartwaste.fusion import FusionResult, fuse, mean_distance
from smartwaste.noise import SensorReading

#: What a redacted secret looks like everywhere it would otherwise appear.
REDACTED = "***REDACTED***"

#: Environment variable names. Already declared in the tracked ``.env.example``.
CHANNEL_ID_VAR = "THINGSPEAK_CHANNEL_ID"
READ_API_KEY_VAR = "THINGSPEAK_READ_API_KEY"

#: Placeholder values shipped in ``.env.example``. Treated as "not configured" so a user who
#: copied the example without editing it gets a clear message instead of a 400 from ThingSpeak.
PLACEHOLDER_VALUES = frozenset({"replace_me", "000000", ""})

#: The invented field mapping. See the module docstring.
DISTANCE_FIELDS = ("field1", "field2", "field3")
WEIGHT_FIELD = "field4"

THINGSPEAK_HOST = "https://api.thingspeak.com"

#: Rows requested per call. Not a config row: it is a CLI convenience with no effect on any
#: result, and ThingSpeak's own documented maximum is 8000.
DEFAULT_N_RESULTS = 100
MAX_N_RESULTS = 8000


class ThingSpeakError(RuntimeError):
    """A fetch or a configuration problem. Its message is always redacted."""


def redact(text: str, *secrets: str) -> str:
    """Replace every secret with :data:`REDACTED`, plus any ``api_key=`` query value.

    Both halves matter. The explicit ``secrets`` cover the key we know about; the ``api_key=``
    rewrite covers a URL that arrived from somewhere else -- an exception message raised by
    ``urllib``, say, which quotes the URL it failed on.
    """
    out = text
    for secret in secrets:
        if secret and secret not in PLACEHOLDER_VALUES:
            out = out.replace(secret, REDACTED)
    # Rewrite every surviving api_key query parameter, whatever its value. The cursor only ever
    # moves forward: a message quoting two different URLs must have BOTH keys rewritten, so an
    # already-redacted parameter is stepped over rather than treated as the end of the work.
    cursor = 0
    marker = "api_key="
    while True:
        found = out.find(marker, cursor)
        if found < 0:
            return out
        start = found + len(marker)
        end = start
        while end < len(out) and out[end] not in "&? \t\n\"'":
            end += 1
        if out[start:end] == REDACTED:
            cursor = end
            continue
        out = out[:start] + REDACTED + out[end:]
        cursor = start + len(REDACTED)


@dataclass(frozen=True, repr=False)
class ThingSpeakSettings:
    """Which channel to read, and with what key.

    ``repr`` is overridden rather than inherited: a frozen dataclass's generated ``repr`` would
    print the key, and a dataclass ends up inside tracebacks, log lines and pytest output.
    """

    channel_id: str
    read_api_key: str
    n_results: int = DEFAULT_N_RESULTS

    def __repr__(self) -> str:
        return (
            f"ThingSpeakSettings(channel_id={self.channel_id!r}, "
            f"read_api_key='{REDACTED}', n_results={self.n_results})"
        )

    def validate(self) -> None:
        if not self.channel_id or self.channel_id in PLACEHOLDER_VALUES:
            raise ThingSpeakError(
                f"{CHANNEL_ID_VAR} is not set (or is still the .env.example placeholder). "
                "Copy .env.example to .env and fill it in; .env is gitignored."
            )
        if not self.read_api_key or self.read_api_key in PLACEHOLDER_VALUES:
            raise ThingSpeakError(
                f"{READ_API_KEY_VAR} is not set (or is still the .env.example placeholder). "
                "Copy .env.example to .env and fill it in; .env is gitignored."
            )
        if not 1 <= self.n_results <= MAX_N_RESULTS:
            raise ThingSpeakError(
                f"n_results must be in [1, {MAX_N_RESULTS}], got {self.n_results}"
            )

    def redact(self, text: str) -> str:
        """Redact this settings object's key out of an arbitrary string."""
        return redact(text, self.read_api_key)


# --- configuration ---------------------------------------------------------------------------


def parse_env_text(text: str) -> dict[str, str]:
    """Parse ``.env`` text into a mapping. Deliberately minimal, deliberately stdlib.

    Handles ``#`` comment lines, blank lines, ``KEY=VALUE``, an optional ``export`` prefix and
    optional surrounding single or double quotes. Anything else is ignored rather than raising:
    a stray line in a developer's ``.env`` should not stop the tool.

    A ``python-dotenv`` dependency for fifteen lines would mean another wheel in the install
    path of a project whose whole point is that its core needs only ``pyyaml``.
    """
    out: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key:
            out[key] = value
    return out


def read_env_file(path: Path) -> dict[str, str]:
    """Read a ``.env`` file. A missing file is an empty mapping, not an error."""
    if not path.is_file():
        return {}
    return parse_env_text(path.read_text(encoding="utf-8"))


def settings_from_env(
    env: Mapping[str, str], *, n_results: int = DEFAULT_N_RESULTS
) -> ThingSpeakSettings:
    """Build validated settings from an environment mapping."""
    settings = ThingSpeakSettings(
        channel_id=env.get(CHANNEL_ID_VAR, ""),
        read_api_key=env.get(READ_API_KEY_VAR, ""),
        n_results=n_results,
    )
    settings.validate()
    return settings


# --- fetching --------------------------------------------------------------------------------

#: Takes a URL, returns the response body. Injected so tests never need a socket.
Fetcher = Callable[[str], str]


def build_url(settings: ThingSpeakSettings) -> str:
    """The channel feed URL. **Contains the read key, so never print this.**

    Use :func:`ThingSpeakSettings.redact` on anything derived from it that a human will see.
    """
    return (
        f"{THINGSPEAK_HOST}/channels/{settings.channel_id}/feeds.json"
        f"?api_key={settings.read_api_key}&results={settings.n_results}"
    )


def urlopen_fetcher(url: str) -> str:
    """Default fetcher. Imports ``urllib`` inside the body, so importing this module is inert."""
    import urllib.request  # noqa: PLC0415 - deliberately lazy; see the module docstring

    with urllib.request.urlopen(url, timeout=30) as response:  # noqa: S310 - https, built above
        body: bytes = response.read()
    return body.decode("utf-8")


def fetch_body(settings: ThingSpeakSettings, fetcher: Fetcher | None = None) -> str:
    """Fetch the raw response body, converting any failure into a redacted error.

    ``urllib`` puts the URL it failed on into its exception message, and that URL carries the
    read key -- so the message is rebuilt rather than re-raised.
    """
    settings.validate()
    call = urlopen_fetcher if fetcher is None else fetcher
    url = build_url(settings)
    try:
        return call(url)
    except Exception as exc:  # noqa: BLE001 - any transport failure, reported without the key
        raise ThingSpeakError(
            f"fetch failed for channel {settings.channel_id}: "
            f"{type(exc).__name__}: {settings.redact(str(exc))}"
        ) from None


# --- parsing ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class ThingSpeakRow:
    """One usable feed entry: three distances and a weight, both already numeric."""

    entry_id: int
    created_at: str
    distances_cm: tuple[float, ...]
    weight_kg: float


@dataclass(frozen=True)
class FetchReport:
    """What came back, and what was thrown away and why.

    Skipped rows are counted and explained rather than raised: a bin that was offline for an hour
    or a sensor that reported ``null`` is an ordinary event on real hardware, and a tool that
    crashes on it is useless. But they are never silently dropped either -- ``n_skipped`` and
    ``skip_reasons`` are printed by the CLI.
    """

    rows: tuple[ThingSpeakRow, ...]
    n_raw: int
    skip_reasons: tuple[str, ...]
    channel_name: str = ""

    @property
    def n_skipped(self) -> int:
        return len(self.skip_reasons)

    @property
    def is_empty(self) -> bool:
        return self.n_raw == 0


def _coerce_float(entry: Mapping[str, object], field: str) -> float:
    raw = entry.get(field)
    if raw is None:
        raise ValueError(f"{field} is null or missing")
    if isinstance(raw, bool):
        raise ValueError(f"{field} is a boolean, not a number")
    try:
        return float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field}={raw!r} is not a number") from exc


def parse_feeds(json_text: str, *, n_ultrasonic: int = len(DISTANCE_FIELDS)) -> FetchReport:
    """Turn a ThingSpeak feeds response into rows, skipping what cannot be used.

    An empty body, a body with no ``feeds`` key and ``{"feeds": []}`` all yield zero rows and no
    exception -- a channel that has never received a post is a normal state, not a failure.
    Malformed JSON is the one thing that does raise, because it means the response was not a
    ThingSpeak response at all.
    """
    if n_ultrasonic < 1 or n_ultrasonic > len(DISTANCE_FIELDS):
        raise ThingSpeakError(
            f"n_ultrasonic must be in [1, {len(DISTANCE_FIELDS)}], got {n_ultrasonic}"
        )

    stripped = json_text.strip()
    if not stripped:
        return FetchReport(rows=(), n_raw=0, skip_reasons=())
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ThingSpeakError(f"response is not valid JSON: {exc}") from None
    if not isinstance(payload, dict):
        raise ThingSpeakError(f"response is not a JSON object, got {type(payload).__name__}")

    channel = payload.get("channel")
    channel_name = str(channel.get("name", "")) if isinstance(channel, dict) else ""

    raw_feeds = payload.get("feeds", [])
    if not isinstance(raw_feeds, list):
        raise ThingSpeakError(f"'feeds' is not a list, got {type(raw_feeds).__name__}")

    rows: list[ThingSpeakRow] = []
    reasons: list[str] = []
    for position, entry in enumerate(raw_feeds):
        if not isinstance(entry, dict):
            reasons.append(f"row {position}: not a JSON object")
            continue
        try:
            distances = tuple(_coerce_float(entry, DISTANCE_FIELDS[i]) for i in range(n_ultrasonic))
            weight = _coerce_float(entry, WEIGHT_FIELD)
        except ValueError as exc:
            reasons.append(f"row {position}: {exc}")
            continue

        entry_id_raw = entry.get("entry_id", position)
        try:
            entry_id = int(entry_id_raw)
        except (TypeError, ValueError):
            entry_id = position
        rows.append(
            ThingSpeakRow(
                entry_id=entry_id,
                created_at=str(entry.get("created_at", "")),
                distances_cm=distances,
                weight_kg=weight,
            )
        )

    return FetchReport(
        rows=tuple(rows),
        n_raw=len(raw_feeds),
        skip_reasons=tuple(reasons),
        channel_name=channel_name,
    )


# --- into the existing fusion code -----------------------------------------------------------


def to_sensor_reading(row: ThingSpeakRow) -> SensorReading:
    """Convert a feed row into the project's own reading type.

    This is the whole point of the module: once a hardware reading is a
    :class:`~smartwaste.noise.SensorReading`, every classifier Phases 1-4 built already accepts
    it, and no rule is reimplemented for the hardware path.
    """
    return SensorReading(distances_cm=row.distances_cm, weight_kg=row.weight_kg)


def classify_rows(
    rows: Sequence[ThingSpeakRow], bin_cfg: BinConfig
) -> tuple[tuple[ThingSpeakRow, FusionResult], ...]:
    """Classify each row with the paper's own fusion rule -- 3-sensor mean plus the load cell.

    Calls :func:`smartwaste.fusion.fuse` directly, with the same arguments strategy (d) uses, so
    a hardware reading is judged by exactly the rule the simulation was judged by.
    """
    return tuple(
        (
            row,
            fuse(
                to_sensor_reading(row),
                bin_cfg,
                use_load_cell=True,
                sensor_indices=None,
                reducer=mean_distance,
            ),
        )
        for row in rows
    )
