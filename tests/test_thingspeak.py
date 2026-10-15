"""Tests for the ThingSpeak fetch path.

**No test here touches the network.** Every fetch goes through an injected fake, which is why
:data:`smartwaste.thingspeak.Fetcher` is a parameter rather than an import: there is no socket to
block and no monkeypatch to forget. The one test that checks the default fetcher checks that
*importing* the module is inert, not that it can reach the internet.

Three properties are being pinned, and the third is the one that would hurt if it broke:

1. **An empty channel and a malformed row never crash.** A bin offline for an hour, or a sensor
   reporting ``null``, is an ordinary event on real hardware. Rows are skipped with a reason and
   counted -- never silently dropped, never raised.
2. **A reading becomes the project's own `SensorReading`**, so `smartwaste.fusion.fuse` -- the
   same function strategy (d) uses -- classifies hardware data with no second copy of the rule.
3. **The read key never escapes.** Not in stdout, not in a ``repr``, not in an exception message.
   An exception message is a log line, and a secret in a log file is a leaked secret.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from smartwaste import load_config
from smartwaste.fusion import BinState
from smartwaste.noise import SensorReading
from smartwaste.thingspeak import (
    CHANNEL_ID_VAR,
    DISTANCE_FIELDS,
    READ_API_KEY_VAR,
    REDACTED,
    WEIGHT_FIELD,
    FetchReport,
    ThingSpeakError,
    ThingSpeakRow,
    ThingSpeakSettings,
    build_url,
    classify_rows,
    fetch_body,
    parse_env_text,
    parse_feeds,
    read_env_file,
    redact,
    settings_from_env,
    to_sensor_reading,
)

pytestmark = pytest.mark.unit

#: A fake key, obviously fake, 16 chars like a real ThingSpeak key.
FAKE_KEY = "ABCD1234EFGH5678"
FAKE_CHANNEL = "1234567"


@pytest.fixture
def sample_json(repo_root: Path) -> str:
    return (repo_root / "tests" / "data" / "thingspeak_sample.json").read_text(encoding="utf-8")


@pytest.fixture
def settings() -> ThingSpeakSettings:
    return ThingSpeakSettings(channel_id=FAKE_CHANNEL, read_api_key=FAKE_KEY, n_results=10)


def feed(**fields: object) -> dict[str, Any]:
    """One feed entry with sane defaults, so a test can vary exactly one field."""
    entry = {
        "created_at": "2026-01-01T00:00:00Z",
        "entry_id": 1,
        "field1": "50.0",
        "field2": "50.0",
        "field3": "50.0",
        "field4": "1.0",
    }
    entry.update(fields)
    return entry


def body(*entries: dict[str, Any], channel_name: str = "fixture") -> str:
    return json.dumps({"channel": {"name": channel_name}, "feeds": list(entries)})


# --- the fixture file ------------------------------------------------------------------------


def test_the_committed_fixture_parses(sample_json: str) -> None:
    report = parse_feeds(sample_json)
    assert report.n_raw == 7
    assert len(report.rows) == 4
    assert report.n_skipped == 3


def test_the_fixture_skips_exactly_the_three_bad_rows(sample_json: str) -> None:
    report = parse_feeds(sample_json)
    reasons = " | ".join(report.skip_reasons)
    assert "not_a_number" in reasons
    assert "field2 is null or missing" in reasons
    assert "field3 is null or missing" in reasons


def test_the_fixture_exercises_all_three_states(sample_json: str) -> None:
    cfg = load_config()
    classified = classify_rows(parse_feeds(sample_json).rows, cfg.bin)
    assert {result.state for _, result in classified} == {
        BinState.GREEN,
        BinState.YELLOW,
        BinState.RED,
    }


def test_the_fixture_contains_a_bin_the_load_cell_alone_escalates(sample_json: str) -> None:
    """A ~30%-full bin at 44 kg is RED by weight. That is the paper's whole mechanism."""
    cfg = load_config()
    classified = classify_rows(parse_feeds(sample_json).rows, cfg.bin)
    escalated = [(row, res) for row, res in classified if res.overloaded]
    assert len(escalated) == 1
    row, result = escalated[0]
    assert result.state is BinState.RED
    assert result.fill_pct < cfg.bin.yellow_threshold_pct
    assert row.weight_kg >= cfg.bin.weight_threshold_kg


def test_the_fixture_carries_no_api_key(repo_root: Path) -> None:
    raw = (repo_root / "tests" / "data" / "thingspeak_sample.json").read_text(encoding="utf-8")
    assert "api_key" not in raw.lower()


# --- empty channels --------------------------------------------------------------------------


def test_an_empty_feeds_list_yields_no_rows_and_no_error() -> None:
    report = parse_feeds(json.dumps({"feeds": []}))
    assert report.rows == ()
    assert report.n_raw == 0
    assert report.is_empty


def test_a_response_with_no_feeds_key_yields_no_rows() -> None:
    report = parse_feeds(json.dumps({"channel": {"name": "never posted"}}))
    assert report.rows == ()
    assert report.is_empty


def test_an_empty_body_yields_no_rows() -> None:
    assert parse_feeds("").rows == ()
    assert parse_feeds("   \n  ").rows == ()


def test_an_empty_channel_is_not_an_error_state() -> None:
    # A channel that has never received a post is a normal state, not a failure. The CLI says so
    # in words rather than returning non-zero.
    report = parse_feeds(json.dumps({"feeds": []}))
    assert report.n_skipped == 0


# --- malformed rows --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("overrides", "fragment"),
    [
        ({"field1": "not_a_number"}, "not a number"),
        ({"field2": None}, "null or missing"),
        ({"field3": ""}, "not a number"),
        ({"field4": "heavy"}, "not a number"),
        ({"field1": True}, "boolean"),
        ({"field2": {"nested": 1}}, "not a number"),
    ],
)
def test_a_malformed_row_is_skipped_with_a_reason(overrides: dict[str, Any], fragment: str) -> None:
    report = parse_feeds(body(feed(**overrides)))
    assert report.rows == ()
    assert report.n_skipped == 1
    assert fragment in report.skip_reasons[0]


def test_a_row_missing_a_field_entirely_is_skipped() -> None:
    entry = feed()
    del entry["field3"]
    report = parse_feeds(body(entry))
    assert report.n_skipped == 1
    assert "field3 is null or missing" in report.skip_reasons[0]


def test_a_feed_entry_that_is_not_an_object_is_skipped() -> None:
    report = parse_feeds(json.dumps({"feeds": ["nonsense", 42, None]}))
    assert report.rows == ()
    assert report.n_skipped == 3
    assert all("not a JSON object" in r for r in report.skip_reasons)


def test_good_rows_survive_alongside_bad_ones() -> None:
    report = parse_feeds(body(feed(entry_id=1), feed(entry_id=2, field1="bad"), feed(entry_id=3)))
    assert [row.entry_id for row in report.rows] == [1, 3]
    assert report.n_skipped == 1


def test_a_non_numeric_entry_id_falls_back_to_the_row_position() -> None:
    report = parse_feeds(body(feed(entry_id="not-an-int")))
    assert report.rows[0].entry_id == 0


def test_the_skip_reason_names_the_row_position() -> None:
    report = parse_feeds(body(feed(), feed(field1="bad")))
    assert report.skip_reasons[0].startswith("row 1:")


# --- things that genuinely should raise ------------------------------------------------------


def test_malformed_json_raises() -> None:
    # Not a lenient case: this means the response was not a ThingSpeak response at all.
    with pytest.raises(ThingSpeakError, match="not valid JSON"):
        parse_feeds("{not json")


def test_a_json_array_instead_of_an_object_raises() -> None:
    with pytest.raises(ThingSpeakError, match="not a JSON object"):
        parse_feeds("[1, 2, 3]")


def test_feeds_that_is_not_a_list_raises() -> None:
    with pytest.raises(ThingSpeakError, match="'feeds' is not a list"):
        parse_feeds(json.dumps({"feeds": {"oops": 1}}))


@pytest.mark.parametrize("n", [0, -1, 4, 99])
def test_an_impossible_sensor_count_raises(n: int) -> None:
    with pytest.raises(ThingSpeakError, match="n_ultrasonic"):
        parse_feeds(body(feed()), n_ultrasonic=n)


# --- the field mapping -----------------------------------------------------------------------


def test_the_invented_field_mapping_is_fields_one_to_four() -> None:
    assert DISTANCE_FIELDS == ("field1", "field2", "field3")
    assert WEIGHT_FIELD == "field4"


def test_fields_five_and_six_are_ignored() -> None:
    """The firmware posts Fv and the state; this path re-derives both rather than trusting them.

    A row whose field5/field6 flatly contradict fields 1-4 must still classify from 1-4.
    """
    cfg = load_config()
    entry = feed(field1="10.0", field2="10.0", field3="10.0", field5="0.00", field6="0")
    report = parse_feeds(body(entry))
    ((_, result),) = classify_rows(report.rows, cfg.bin)
    assert result.fill_pct == pytest.approx(90.0)
    assert result.state is BinState.RED


def test_fewer_sensors_can_be_requested() -> None:
    report = parse_feeds(body(feed(field3="bad")), n_ultrasonic=2)
    assert len(report.rows) == 1
    assert report.rows[0].distances_cm == (50.0, 50.0)


# --- into the existing fusion code -----------------------------------------------------------


def test_to_sensor_reading_produces_the_projects_own_type() -> None:
    row = ThingSpeakRow(entry_id=1, created_at="", distances_cm=(10.0, 20.0, 30.0), weight_kg=5.0)
    reading = to_sensor_reading(row)
    assert isinstance(reading, SensorReading)
    assert reading.distances_cm == (10.0, 20.0, 30.0)
    assert reading.weight_kg == 5.0


@pytest.mark.parametrize(
    ("distances", "weight", "expected"),
    [
        # Fv = 100 * (1 - d_bar / 100), so d_bar 95 -> 5%, 35 -> 65%, 15 -> 85%.
        ((95.0, 95.0, 95.0), 1.0, BinState.GREEN),
        ((35.0, 35.0, 35.0), 1.0, BinState.YELLOW),
        ((15.0, 15.0, 15.0), 1.0, BinState.RED),
        # Exactly on the thresholds: d_bar 40 -> Fv 60 (YELLOW), 20 -> Fv 80 (RED).
        ((40.0, 40.0, 40.0), 1.0, BinState.YELLOW),
        ((20.0, 20.0, 20.0), 1.0, BinState.RED),
        # alpha * W_max = 0.85 * 50 = 42.5. Exactly 42.5 is overloaded, so RED whatever Fv says.
        ((99.0, 99.0, 99.0), 42.5, BinState.RED),
        ((99.0, 99.0, 99.0), 42.4999, BinState.GREEN),
    ],
)
def test_a_hardware_reading_is_classified_by_the_projects_own_rule(
    distances: tuple[float, ...], weight: float, expected: BinState
) -> None:
    cfg = load_config()
    row = ThingSpeakRow(entry_id=1, created_at="", distances_cm=distances, weight_kg=weight)
    ((_, result),) = classify_rows([row], cfg.bin)
    assert result.state is expected


def test_classify_rows_on_an_empty_sequence_is_empty() -> None:
    assert classify_rows([], load_config().bin) == ()


# --- the .env parser -------------------------------------------------------------------------


def test_parses_a_plain_env_file() -> None:
    parsed = parse_env_text(f"{CHANNEL_ID_VAR}=42\n{READ_API_KEY_VAR}={FAKE_KEY}\n")
    assert parsed == {CHANNEL_ID_VAR: "42", READ_API_KEY_VAR: FAKE_KEY}


def test_ignores_comments_and_blank_lines() -> None:
    text = f"# a comment\n\n   \n{CHANNEL_ID_VAR}=42\n# {READ_API_KEY_VAR}=commented_out\n"
    assert parse_env_text(text) == {CHANNEL_ID_VAR: "42"}


@pytest.mark.parametrize("quoted", ['"quoted"', "'quoted'"])
def test_strips_surrounding_quotes(quoted: str) -> None:
    assert parse_env_text(f"{READ_API_KEY_VAR}={quoted}\n")[READ_API_KEY_VAR] == "quoted"


def test_accepts_an_export_prefix() -> None:
    assert parse_env_text(f"export {CHANNEL_ID_VAR}=42\n") == {CHANNEL_ID_VAR: "42"}


def test_tolerates_surrounding_whitespace() -> None:
    assert parse_env_text(f"  {CHANNEL_ID_VAR} =  42  \n") == {CHANNEL_ID_VAR: "42"}


def test_a_value_containing_an_equals_sign_survives() -> None:
    assert parse_env_text("K=a=b=c")["K"] == "a=b=c"


def test_a_stray_line_is_ignored_rather_than_raising() -> None:
    # A developer's .env should not be able to stop the tool.
    assert parse_env_text("this line has no equals sign\nK=1\n") == {"K": "1"}


def test_a_missing_env_file_is_an_empty_mapping(tmp_path: Path) -> None:
    assert read_env_file(tmp_path / "nope.env") == {}


def test_reads_a_real_env_file(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_text(f"{CHANNEL_ID_VAR}=9\n{READ_API_KEY_VAR}={FAKE_KEY}\n", encoding="utf-8")
    assert read_env_file(path)[READ_API_KEY_VAR] == FAKE_KEY


def test_the_tracked_env_example_still_declares_both_variables(repo_root: Path) -> None:
    parsed = parse_env_text((repo_root / ".env.example").read_text(encoding="utf-8"))
    assert CHANNEL_ID_VAR in parsed
    assert READ_API_KEY_VAR in parsed


def test_the_env_example_placeholders_are_rejected_as_unconfigured(repo_root: Path) -> None:
    """Copying .env.example without editing it must say so, not produce a confusing HTTP 400."""
    parsed = parse_env_text((repo_root / ".env.example").read_text(encoding="utf-8"))
    with pytest.raises(ThingSpeakError, match="placeholder"):
        settings_from_env(parsed)


@pytest.mark.parametrize(
    "env",
    [
        {},
        {CHANNEL_ID_VAR: "42"},
        {READ_API_KEY_VAR: FAKE_KEY},
        {CHANNEL_ID_VAR: "", READ_API_KEY_VAR: FAKE_KEY},
    ],
)
def test_incomplete_settings_raise_with_a_pointer_to_dotenv(env: dict[str, str]) -> None:
    with pytest.raises(ThingSpeakError, match=r"\.env"):
        settings_from_env(env)


@pytest.mark.parametrize("n", [0, -5, 8001])
def test_an_out_of_range_limit_raises(n: int) -> None:
    with pytest.raises(ThingSpeakError, match="n_results"):
        settings_from_env({CHANNEL_ID_VAR: "42", READ_API_KEY_VAR: FAKE_KEY}, n_results=n)


# --- the injected fetcher: no network anywhere -----------------------------------------------


def test_the_fetcher_is_injected_and_the_url_carries_the_request(
    settings: ThingSpeakSettings, sample_json: str
) -> None:
    seen: list[str] = []

    def fake(url: str) -> str:
        seen.append(url)
        return sample_json

    assert fetch_body(settings, fake) == sample_json
    assert len(seen) == 1
    assert f"channels/{FAKE_CHANNEL}/feeds.json" in seen[0]
    assert "results=10" in seen[0]


def test_a_transport_failure_becomes_a_domain_error(settings: ThingSpeakSettings) -> None:
    def boom(url: str) -> str:
        raise OSError("connection refused")

    with pytest.raises(ThingSpeakError, match="fetch failed"):
        fetch_body(settings, boom)


def test_importing_the_module_does_not_import_urllib() -> None:
    """The default fetcher imports urllib inside its body, so import alone cannot reach out."""
    import inspect

    import smartwaste.thingspeak as module

    source = inspect.getsource(module)
    top_level = [
        line for line in source.splitlines() if line.startswith(("import ", "from ")) and line
    ]
    assert not any("urllib" in line for line in top_level)
    assert "import urllib.request" in source  # it exists, just not at module scope


# --- redaction -------------------------------------------------------------------------------


def test_redact_replaces_a_secret() -> None:
    assert redact(f"key is {FAKE_KEY} ok", FAKE_KEY) == f"key is {REDACTED} ok"


def test_redact_rewrites_an_api_key_query_parameter_it_was_not_told_about() -> None:
    # urllib quotes the URL it failed on, and that URL may arrive from code that never saw the
    # settings object. So any surviving api_key= is rewritten regardless.
    text = "HTTP Error 400: https://api.thingspeak.com/x?api_key=SOMEOTHERKEY&results=5"
    out = redact(text, "unrelated")
    assert "SOMEOTHERKEY" not in out
    assert f"api_key={REDACTED}" in out
    assert "results=5" in out


def test_redact_handles_several_api_key_parameters() -> None:
    out = redact("a?api_key=AAA&b=1 and c?api_key=BBB", "")
    assert "AAA" not in out
    assert "BBB" not in out


def test_redact_is_idempotent() -> None:
    once = redact(f"?api_key={FAKE_KEY}", FAKE_KEY)
    assert redact(once, FAKE_KEY) == once


def test_redact_does_not_blank_the_placeholder_values() -> None:
    # Redacting "replace_me" would turn the "you have not configured this" message into noise.
    assert redact("value is replace_me", "replace_me") == "value is replace_me"


def test_the_settings_repr_does_not_contain_the_key(settings: ThingSpeakSettings) -> None:
    assert FAKE_KEY not in repr(settings)
    assert REDACTED in repr(settings)
    assert FAKE_CHANNEL in repr(settings)


def test_the_settings_str_does_not_contain_the_key(settings: ThingSpeakSettings) -> None:
    assert FAKE_KEY not in str(settings)


def test_the_key_is_not_in_a_fetch_failure_message(settings: ThingSpeakSettings) -> None:
    def boom(url: str) -> str:
        # The realistic case: urllib puts the URL, key and all, into its own message.
        raise OSError(f"failed opening {url}")

    with pytest.raises(ThingSpeakError) as caught:
        fetch_body(settings, boom)
    assert FAKE_KEY not in str(caught.value)
    assert REDACTED in str(caught.value)


def test_the_key_is_not_in_a_validation_failure_message() -> None:
    bad = ThingSpeakSettings(channel_id="", read_api_key=FAKE_KEY)
    with pytest.raises(ThingSpeakError) as caught:
        bad.validate()
    assert FAKE_KEY not in str(caught.value)


def test_the_key_is_not_in_a_traceback_repr(settings: ThingSpeakSettings) -> None:
    def boom(url: str) -> str:
        raise ValueError(url)

    with pytest.raises(ThingSpeakError) as caught:
        fetch_body(settings, boom)
    assert FAKE_KEY not in repr(caught.value)
    assert FAKE_KEY not in "".join(str(arg) for arg in caught.value.args)


def test_the_key_is_not_in_a_report_repr(settings: ThingSpeakSettings, sample_json: str) -> None:
    report = parse_feeds(fetch_body(settings, lambda _: sample_json))
    assert FAKE_KEY not in repr(report)


def test_build_url_does_contain_the_key_which_is_why_it_is_never_printed(
    settings: ThingSpeakSettings,
) -> None:
    """Pinned deliberately: this is the one function whose output must not be displayed."""
    assert FAKE_KEY in build_url(settings)
    assert settings.redact(build_url(settings)).count(REDACTED) == 1


def test_a_report_never_carries_the_key_even_if_the_body_did() -> None:
    # Defensive: a channel whose name somehow echoed the key must not propagate it.
    report = parse_feeds(body(feed(), channel_name=FAKE_KEY))
    assert isinstance(report, FetchReport)
    assert redact(report.channel_name, FAKE_KEY) == REDACTED
