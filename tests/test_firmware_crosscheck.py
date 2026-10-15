"""Cross-check the NodeMCU sketch's rule against the Python fusion code.

=============================================================================================
**THIS DOES NOT VERIFY THE C++.**

Nothing in this file compiles ``firmware/nodemcu_smartbin/nodemcu_smartbin.ino``, links it, or
runs it. No ``arduino-cli`` was available, no ESP8266 toolchain was installed, no board was
connected. What these tests establish is narrower, and worth stating exactly:

1. :mod:`smartwaste.firmware_reference` -- a hand transcription of the sketch's arithmetic --
   agrees with :mod:`smartwaste.fusion` on the RED/YELLOW/GREEN state.
2. The six ``#define``d constants *written in the sketch text* equal the values in
   ``config/default.yaml``.

The first tests the rule we wrote down. The second parses text with a regular expression. Neither
can catch a C++ type error, a ``float`` vs ``double`` difference, an overflow the transcription
did not reproduce, a pin mis-assignment, or a library quirk. The firmware remains **NOT
VERIFIED** until someone compiles it and flashes it. See ``docs/PHASE5_RESULTS.md``.
=============================================================================================

The point of having it: the sketch necessarily reimplements the fusion rule, because an ESP8266
cannot import Python. Two copies of a rule drift. These tests make the drift *detectable* -- edit
a threshold in one place and two tests fail.
"""

from __future__ import annotations

import random
import re

import pytest

from smartwaste import load_config
from smartwaste.config import REPO_ROOT, BinConfig
from smartwaste.firmware_reference import (
    ALPHA,
    HEIGHT_CM,
    INVALID_DISTANCE_CM,
    MIN_DISTANCE_CM,
    N_ULTRASONIC,
    RED_THRESHOLD_PCT,
    STATE_GREEN,
    STATE_RED,
    STATE_YELLOW,
    W_MAX_KG,
    YELLOW_THRESHOLD_PCT,
    FirmwareNoValidSensors,
    classify_reading,
    classify_state,
    distance_from_echo_us,
    fill_percent,
    is_overloaded,
    mean_valid_distance_cm,
)
from smartwaste.fusion import BinState, fuse, mean_distance
from smartwaste.noise import SensorReading

pytestmark = pytest.mark.unit

SKETCH_PATH = REPO_ROOT / "firmware" / "nodemcu_smartbin" / "nodemcu_smartbin.ino"
EXAMPLE_HEADER_PATH = REPO_ROOT / "firmware" / "nodemcu_smartbin" / "config.example.h"

#: ``#define NAME 100.0f`` -> ("NAME", 100.0). ``f``/``F`` suffix and ``L`` are optional, and a
#: trailing block comment is allowed because the sketch documents every constant inline.
_DEFINE_RE = r"^\s*#define\s+{name}\s+\(?\s*(-?[0-9]*\.?[0-9]+)\s*[fFlL]?\s*\)?\s*(?:/\*.*)?$"

#: Sketch constant -> the dotted config path it must equal.
SKETCH_TO_CONFIG = {
    "HEIGHT_CM": ("bin", "height_cm"),
    "W_MAX_KG": ("bin", "w_max_kg"),
    "ALPHA": ("bin", "alpha"),
    "YELLOW_THRESHOLD_PCT": ("bin", "yellow_threshold_pct"),
    "RED_THRESHOLD_PCT": ("bin", "red_threshold_pct"),
    "MIN_DISTANCE_CM": ("sensors", "min_distance_cm"),
}

#: Sketch constant -> the transcription's module-level constant.
SKETCH_TO_REFERENCE = {
    "HEIGHT_CM": HEIGHT_CM,
    "W_MAX_KG": W_MAX_KG,
    "ALPHA": ALPHA,
    "YELLOW_THRESHOLD_PCT": YELLOW_THRESHOLD_PCT,
    "RED_THRESHOLD_PCT": RED_THRESHOLD_PCT,
    "MIN_DISTANCE_CM": MIN_DISTANCE_CM,
}


@pytest.fixture(scope="module")
def sketch_text() -> str:
    assert SKETCH_PATH.is_file(), f"sketch not found: {SKETCH_PATH}"
    return SKETCH_PATH.read_text(encoding="utf-8")


def parse_define(text: str, name: str) -> float:
    """Pull one numeric ``#define`` out of the sketch.

    Raises rather than returning a default when the constant is absent or unparseable: a renamed
    constant must fail the check, not quietly skip it.
    """
    match = re.search(_DEFINE_RE.format(name=re.escape(name)), text, flags=re.MULTILINE)
    if match is None:
        raise AssertionError(
            f"#define {name} not found in the sketch, or its value is not a plain number. "
            "If it was renamed, update SKETCH_TO_CONFIG in this test."
        )
    return float(match.group(1))


def as_reading(distances: tuple[float, ...], weight: float) -> SensorReading:
    return SensorReading(distances_cm=distances, weight_kg=weight)


def python_state(distances: tuple[float, ...], weight: float, bin_cfg: BinConfig) -> BinState:
    """The Python side, configured exactly as strategy (d) configures it."""
    return fuse(
        as_reading(distances, weight),
        bin_cfg,
        use_load_cell=True,
        sensor_indices=None,
        reducer=mean_distance,
    ).state


# --- the constants: sketch vs config vs transcription ----------------------------------------
#
# Three sources must agree. The transcription duplicates the sketch's values ON PURPOSE rather
# than reading the config, so that this comparison can actually fail.


@pytest.mark.parametrize(("name", "path"), sorted(SKETCH_TO_CONFIG.items()))
def test_a_sketch_constant_equals_the_config_value(
    sketch_text: str, name: str, path: tuple[str, str]
) -> None:
    """Parses TEXT out of the .ino. Does NOT verify the C++ -- only the numbers written in it."""
    cfg = load_config()
    section, field = path
    expected = getattr(getattr(cfg, section), field)
    assert parse_define(sketch_text, name) == pytest.approx(expected), (
        f"#define {name} in the sketch disagrees with {section}.{field} in config/default.yaml"
    )


@pytest.mark.parametrize(("name", "value"), sorted(SKETCH_TO_REFERENCE.items()))
def test_the_transcription_matches_the_sketch_text(
    sketch_text: str, name: str, value: float
) -> None:
    assert parse_define(sketch_text, name) == pytest.approx(value)


def test_the_sensor_count_agrees_everywhere(sketch_text: str) -> None:
    cfg = load_config()
    assert parse_define(sketch_text, "N_ULTRASONIC") == cfg.sensors.n_ultrasonic
    assert cfg.sensors.n_ultrasonic == N_ULTRASONIC


def test_the_overload_threshold_is_the_configs_derived_value() -> None:
    cfg = load_config()
    assert pytest.approx(cfg.bin.weight_threshold_kg) == ALPHA * W_MAX_KG


def test_a_renamed_constant_fails_loudly_rather_than_being_skipped() -> None:
    with pytest.raises(AssertionError, match="not found in the sketch"):
        parse_define("#define SOMETHING_ELSE 1.0f", "HEIGHT_CM")


# --- the state codes -------------------------------------------------------------------------


def test_the_state_codes_in_the_sketch_match_the_transcription(sketch_text: str) -> None:
    assert parse_define(sketch_text, "STATE_GREEN") == STATE_GREEN
    assert parse_define(sketch_text, "STATE_YELLOW") == STATE_YELLOW
    assert parse_define(sketch_text, "STATE_RED") == STATE_RED


def test_the_state_codes_are_distinct() -> None:
    assert len({STATE_GREEN, STATE_YELLOW, STATE_RED}) == 3


# --- agreement on the rule: the boundaries ---------------------------------------------------


@pytest.mark.parametrize(
    ("d_bar", "expected"),
    [
        # Fv = 100 * (1 - d_bar / 100), so these distances land exactly on the thresholds.
        (40.0, BinState.YELLOW),  # Fv exactly 60.0 -> YELLOW, not GREEN
        (40.0001, BinState.GREEN),  # a hair under 60
        (20.0, BinState.RED),  # Fv exactly 80.0 -> RED, not YELLOW
        (20.0001, BinState.YELLOW),  # a hair under 80
        (100.0, BinState.GREEN),  # Fv 0
        (2.0, BinState.RED),  # Fv 98
    ],
)
def test_both_implementations_agree_on_the_fill_boundaries(
    d_bar: float, expected: BinState
) -> None:
    cfg = load_config()
    distances = (d_bar, d_bar, d_bar)
    assert classify_reading(distances, 0.0).state is expected
    assert python_state(distances, 0.0, cfg.bin) is expected


@pytest.mark.parametrize(
    ("weight", "overloaded"),
    [
        (42.5, True),  # exactly alpha * W_max -- the boundary IS overloaded
        (42.4999, False),
        (42.5001, True),
        (0.0, False),
        (1000.0, True),
    ],
)
def test_both_implementations_agree_on_the_overload_boundary(
    weight: float, overloaded: bool
) -> None:
    cfg = load_config()
    # An almost-empty bin, so only the load cell can make it RED.
    distances = (99.0, 99.0, 99.0)
    expected = BinState.RED if overloaded else BinState.GREEN
    assert is_overloaded(weight) is overloaded
    assert classify_reading(distances, weight).state is expected
    assert python_state(distances, weight, cfg.bin) is expected


def test_weight_wins_over_a_low_fill_in_both() -> None:
    """The paper's rule: an overloaded bin is RED whatever the ultrasonic sensors say."""
    cfg = load_config()
    distances = (95.0, 95.0, 95.0)  # Fv = 5%, visually nearly empty
    assert classify_reading(distances, 45.0).state is BinState.RED
    assert python_state(distances, 45.0, cfg.bin) is BinState.RED


# --- agreement on the rule: a seeded sweep ---------------------------------------------------


def test_both_implementations_agree_on_2000_seeded_readings() -> None:
    """The bulk check. Same fixed seed every run, so a disagreement is reproducible."""
    cfg = load_config()
    rng = random.Random(42)
    disagreements: list[tuple[tuple[float, ...], float, BinState, BinState]] = []

    for _ in range(2000):
        distances = tuple(
            round(rng.uniform(MIN_DISTANCE_CM, HEIGHT_CM), 4) for _ in range(N_ULTRASONIC)
        )
        weight = round(rng.uniform(0.0, W_MAX_KG * 1.5), 4)
        firmware = classify_reading(distances, weight).state
        python = python_state(distances, weight, cfg.bin)
        if firmware is not python:
            disagreements.append((distances, weight, firmware, python))

    assert not disagreements, f"{len(disagreements)} disagreements, first: {disagreements[0]}"


def test_the_sweep_actually_covers_all_three_states() -> None:
    """A sweep that only ever produced GREEN would pass the test above while proving nothing."""
    cfg = load_config()
    rng = random.Random(42)
    seen: set[BinState] = set()
    for _ in range(2000):
        distances = tuple(
            round(rng.uniform(MIN_DISTANCE_CM, HEIGHT_CM), 4) for _ in range(N_ULTRASONIC)
        )
        weight = round(rng.uniform(0.0, W_MAX_KG * 1.5), 4)
        seen.add(python_state(distances, weight, cfg.bin))
    assert seen == {BinState.GREEN, BinState.YELLOW, BinState.RED}


def test_both_agree_on_the_fused_fill_percentage_too() -> None:
    """Not just the state: the intermediate Fv must match, or the agreement is a coincidence."""
    cfg = load_config()
    rng = random.Random(7)
    for _ in range(500):
        distances = tuple(
            round(rng.uniform(MIN_DISTANCE_CM, HEIGHT_CM), 4) for _ in range(N_ULTRASONIC)
        )
        weight = round(rng.uniform(0.0, W_MAX_KG), 4)
        firmware = classify_reading(distances, weight)
        python = fuse(
            as_reading(distances, weight),
            cfg.bin,
            use_load_cell=True,
            sensor_indices=None,
            reducer=mean_distance,
        )
        assert firmware.fill_pct == pytest.approx(python.fill_pct)
        assert firmware.overloaded is python.overloaded


# --- agreement on the rule: degenerate cases -------------------------------------------------


@pytest.mark.parametrize(
    "distances",
    [
        (HEIGHT_CM, HEIGHT_CM, HEIGHT_CM),  # empty bin
        (MIN_DISTANCE_CM, MIN_DISTANCE_CM, MIN_DISTANCE_CM),  # sensor floor, full bin
        (MIN_DISTANCE_CM, HEIGHT_CM, MIN_DISTANCE_CM),  # violently uneven
        (HEIGHT_CM, MIN_DISTANCE_CM, HEIGHT_CM),  # centre peak
    ],
)
def test_both_implementations_agree_on_degenerate_distances(distances: tuple[float, ...]) -> None:
    cfg = load_config()
    for weight in (0.0, 20.0, 42.5, 60.0):
        assert classify_reading(distances, weight).state is python_state(distances, weight, cfg.bin)


def test_a_timed_out_sensor_is_dropped_from_the_mean_not_averaged_as_zero() -> None:
    """The single most dangerous bug this transcription exists to pin.

    Averaging a dead sensor in as 0 cm would read as "waste is touching the sensor" -- the exact
    opposite of "no reading" -- and would post a silent false RED on an empty bin.
    """
    distances = (90.0, INVALID_DISTANCE_CM, 90.0)
    result = classify_reading(distances, 0.0)
    assert result.n_valid == 2
    assert result.d_bar_cm == pytest.approx(90.0)
    assert result.state is BinState.GREEN

    # What the bug would have produced: mean(90, 0, 90) = 60 -> Fv 40%, and worse shapes go RED.
    naive_mean = (90.0 + 0.0 + 90.0) / 3
    assert naive_mean != pytest.approx(result.d_bar_cm)


def test_a_dropped_sensor_matches_python_restricted_to_the_same_sensors() -> None:
    """With one sensor out, the firmware's mean must equal Python's mean over the SAME subset."""
    cfg = load_config()
    rng = random.Random(1234)
    for _ in range(300):
        distances = [round(rng.uniform(MIN_DISTANCE_CM, HEIGHT_CM), 4) for _ in range(2)]
        dead_index = rng.randrange(N_ULTRASONIC)
        distances.insert(dead_index, INVALID_DISTANCE_CM)
        weight = round(rng.uniform(0.0, W_MAX_KG), 4)

        firmware = classify_reading(tuple(distances), weight)
        alive_indices = tuple(i for i in range(N_ULTRASONIC) if i != dead_index)
        python = fuse(
            as_reading(tuple(distances), weight),
            cfg.bin,
            use_load_cell=True,
            sensor_indices=alive_indices,
            reducer=mean_distance,
        )
        assert firmware.n_valid == 2
        assert firmware.fill_pct == pytest.approx(python.fill_pct)
        assert firmware.state is python.state


def test_all_three_sensors_dead_posts_nothing() -> None:
    """The sketch bails out instead of fabricating an Fv, so the transcription raises."""
    with pytest.raises(FirmwareNoValidSensors):
        classify_reading((INVALID_DISTANCE_CM,) * N_ULTRASONIC, 10.0)


def test_a_negative_weight_is_floored_at_zero() -> None:
    """A negative mass is a tare error, not a measurement."""
    assert classify_reading((99.0, 99.0, 99.0), -5.0).overloaded is False
    assert classify_reading((99.0, 99.0, 99.0), -5.0).state is BinState.GREEN


# --- the transcribed helpers in isolation ----------------------------------------------------


def test_the_clamp_happens_before_the_average_not_after() -> None:
    """Clamp order is part of the rule. Swapping it changes the answer near a full bin.

    Raw echoes of 1 cm and 99 cm clamp to 2 cm and 99 cm -> mean 50.5. Averaging first gives
    mean 50.0, then nothing to clamp -- a different Fv.
    """
    clamped_then_meaned = (MIN_DISTANCE_CM + 99.0) / 2
    meaned_then_clamped = (1.0 + 99.0) / 2
    assert clamped_then_meaned != pytest.approx(meaned_then_clamped)

    result = classify_reading([1.0, 99.0], 0.0, clamp_inputs=True)
    assert result.d_bar_cm == pytest.approx(clamped_then_meaned)


@pytest.mark.parametrize(
    ("echo_us", "expected"),
    [
        (0, INVALID_DISTANCE_CM),  # timeout
        (58.0, 2.0),  # 1 cm raw, clamped up to the 2 cm floor
        (58.0 * 50, 50.0),
        (58.0 * 200, HEIGHT_CM),  # 200 cm raw, clamped down to the bin height
    ],
)
def test_echo_microseconds_convert_and_clamp(echo_us: float, expected: float) -> None:
    assert distance_from_echo_us(echo_us) == pytest.approx(expected)


def test_mean_valid_distance_reports_how_many_sensors_contributed() -> None:
    assert mean_valid_distance_cm([10.0, 20.0, 30.0]) == (20.0, 3)
    assert mean_valid_distance_cm([10.0, INVALID_DISTANCE_CM, 30.0]) == (20.0, 2)
    assert mean_valid_distance_cm([INVALID_DISTANCE_CM] * 3) == (INVALID_DISTANCE_CM, 0)


@pytest.mark.parametrize(
    ("d_bar", "expected"),
    [(0.0, 100.0), (50.0, 50.0), (HEIGHT_CM, 0.0), (150.0, 0.0), (-10.0, 100.0)],
)
def test_fill_percent_is_clipped_to_the_percentage_range(d_bar: float, expected: float) -> None:
    assert fill_percent(d_bar) == pytest.approx(expected)


def test_classify_state_returns_the_field6_codes() -> None:
    assert classify_state(10.0, False) == STATE_GREEN
    assert classify_state(YELLOW_THRESHOLD_PCT, False) == STATE_YELLOW
    assert classify_state(RED_THRESHOLD_PCT, False) == STATE_RED
    assert classify_state(0.0, True) == STATE_RED


# --- the sketch says it is unverified, in the file itself -------------------------------------


def test_the_sketch_says_it_is_not_verified(sketch_text: str) -> None:
    """The warning must live in the artefact, not only in a doc nobody opens next to it."""
    header = sketch_text[:2000]
    assert "NOT VERIFIED" in header
    assert "never been compiled" in header.lower() or "never compiled" in header.lower()
    assert "never been flashed" in header.lower() or "never flashed" in header.lower()


def test_the_sketch_labels_the_thingspeak_field_mapping_as_invented(sketch_text: str) -> None:
    assert "INVENTED" in sketch_text


def test_the_sketch_includes_the_gitignored_config_header(sketch_text: str) -> None:
    assert '#include "config.h"' in sketch_text


def test_the_example_header_ships_placeholders_only() -> None:
    """Whether `config.h` is TRACKED is tests/test_no_secrets_tracked.py's job, not this one.

    That check asks git, not the disk, deliberately: a developer's local `config.h` is supposed
    to exist and must not fail their test run. Here we only check the tracked example.
    """
    text = EXAMPLE_HEADER_PATH.read_text(encoding="utf-8")
    assert "YOUR_WIFI_SSID" in text
    assert "YOUR_WIFI_PASSWORD" in text
    assert "0000000000000000" in text
    assert "NEVER be committed" in text


def test_the_sketch_never_prints_the_write_key(sketch_text: str) -> None:
    """The URL carries the write key, so the sketch must log the HTTP status instead."""
    printed_url_lines = [
        line
        for line in sketch_text.splitlines()
        if "println(url" in line.replace(" ", "") or "print(url" in line.replace(" ", "")
    ]
    assert not printed_url_lines, f"the sketch prints the URL: {printed_url_lines}"
    assert "THINGSPEAK_WRITE_API_KEY" in sketch_text  # it is used, just never printed
