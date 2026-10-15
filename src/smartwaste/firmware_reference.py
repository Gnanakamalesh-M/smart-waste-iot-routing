"""A Python transcription of the arithmetic in ``firmware/nodemcu_smartbin/nodemcu_smartbin.ino``.

=============================================================================================
**THIS IS A TRANSCRIPTION, NOT A COMPILER.**

Agreement between this module and :mod:`smartwaste.fusion` does **NOT** verify the C++ in
``nodemcu_smartbin.ino``. Nothing in this project compiles that file, links it, or runs it. What
the cross-check establishes is narrower and worth stating precisely:

    the rule we *wrote down* in the sketch is the same rule
    :mod:`smartwaste.fusion` implements.

It cannot catch a C++ type error, an integer overflow the transcription did not reproduce, a
``float`` vs ``double`` difference, a pin mis-assignment, a library quirk, or anything about how
the ESP8266 toolchain treats the code. A real verification needs ``arduino-cli compile`` and then
a bin with waste in it. See ``docs/PHASE5_RESULTS.md``.
=============================================================================================

### Why have it at all

The sketch reimplements the fusion rule, because an ESP8266 cannot import Python. A
reimplementation is a second copy, and two copies of a rule drift. This module makes the drift
*detectable*: it is a mechanical, line-by-line translation of the sketch, so
``tests/test_firmware_crosscheck.py`` can feed thousands of readings to it and to
:func:`smartwaste.fusion.fuse` and demand the identical state. If someone edits the sketch's
thresholds and not the config, two tests fail.

### What was transcribed, and the two things that matter

**The clamp order.** Each distance is clamped to ``[MIN_DISTANCE_CM, HEIGHT_CM]`` *first*, then
averaged, then ``Fv`` is clipped to ``[0, 100]``. The Python pipeline does the same two stages in
the same order (:class:`smartwaste.noise.SensorArray` clamps the reading,
:func:`smartwaste.fusion.fill_percent` clips ``Fv``). Averaging before clamping gives different
answers near a full or empty bin, so the order is part of the rule, not an implementation detail.

**The dropped sensor.** A timed-out HC-SR04 returns :data:`INVALID_DISTANCE_CM` and is left out
of the mean, rather than averaged in as zero. Zero would read as "waste is touching the sensor",
the exact opposite of "no reading", and would post a silent false RED.

Float arithmetic note: the sketch uses C ``float`` (32-bit) and this module uses Python ``float``
(64-bit double). The transcription therefore agrees on the *rule* but not necessarily bit-for-bit
on a value sitting within ~1e-7 of a threshold. The cross-check tests exact boundaries with
exactly representable values (60.0, 80.0, 42.5) where the two agree, and the limitation is
recorded rather than engineered around.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from smartwaste.fusion import BinState

# --- the six constants, transcribed from the sketch's #define block --------------------------
#
# These are duplicated here ON PURPOSE rather than read from config/default.yaml: this module
# must mirror the SKETCH, so that a test comparing it against the config can actually fail.
# Reading the config here would make that test compare the config with itself.
#
# `tests/test_firmware_crosscheck.py` parses the .ino and asserts all three sources agree:
# the sketch, this module, and config/default.yaml.

#: ``bin.height_cm`` -- inner height H, cm.
HEIGHT_CM = 100.0
#: ``bin.w_max_kg`` -- capacity W_max, kg.
W_MAX_KG = 50.0
#: ``bin.alpha`` -- W_th = ALPHA * W_MAX_KG. Not from the paper; see docs/ASSUMPTIONS.md.
ALPHA = 0.85
#: ``bin.yellow_threshold_pct`` -- the paper's 60.
YELLOW_THRESHOLD_PCT = 60.0
#: ``bin.red_threshold_pct`` -- the paper's 80.
RED_THRESHOLD_PCT = 80.0
#: ``sensors.min_distance_cm`` -- HC-SR04 minimum range, cm.
MIN_DISTANCE_CM = 2.0

#: ``sensors.n_ultrasonic``.
N_ULTRASONIC = 3

#: Microseconds of round-trip echo per centimetre. Standard HC-SR04 arithmetic.
US_PER_CM_ROUND_TRIP = 58.0

#: What ``readDistanceCm()`` returns when the echo never came back. Negative by design.
INVALID_DISTANCE_CM = -1.0

#: ThingSpeak ``field6`` state codes, as the sketch posts them.
STATE_GREEN = 0
STATE_YELLOW = 1
STATE_RED = 2

#: Sketch state code -> the project's :class:`~smartwaste.fusion.BinState`.
STATE_CODE_TO_BIN_STATE = {
    STATE_GREEN: BinState.GREEN,
    STATE_YELLOW: BinState.YELLOW,
    STATE_RED: BinState.RED,
}


class FirmwareNoValidSensors(ValueError):
    """Every ultrasonic sensor timed out.

    The sketch's ``loop()`` posts nothing in this case, so there is no state to compare and the
    cross-check has nothing to assert. Raising is the honest translation of "returned
    INVALID_DISTANCE_CM and bailed out".
    """


@dataclass(frozen=True)
class FirmwareResult:
    """What the sketch would have computed and posted.

    Mirrors :class:`smartwaste.fusion.FusionResult`, plus the two things only the firmware has:
    the state code it puts in ``field6`` and how many sensors actually contributed.
    """

    fill_pct: float
    overloaded: bool
    state: BinState
    state_code: int
    n_valid: int
    d_bar_cm: float


def distance_from_echo_us(echo_us: float) -> float:
    """``readDistanceCm()``: echo microseconds to a clamped distance in cm.

    A zero echo is a timeout and yields :data:`INVALID_DISTANCE_CM`. Otherwise the raw
    ``us / 58`` is clamped to ``[MIN_DISTANCE_CM, HEIGHT_CM]`` before it goes anywhere.
    """
    if echo_us == 0:
        return INVALID_DISTANCE_CM
    return clamp_distance_cm(echo_us / US_PER_CM_ROUND_TRIP)


def clamp_distance_cm(cm: float) -> float:
    """The sketch's two-sided clamp on a single reading, before any averaging."""
    if cm < MIN_DISTANCE_CM:
        return MIN_DISTANCE_CM
    if cm > HEIGHT_CM:
        return HEIGHT_CM
    return cm


def mean_valid_distance_cm(distances_cm: Sequence[float]) -> tuple[float, int]:
    """``meanValidDistanceCm()``: mean over the valid sensors, and how many there were.

    "Valid" is ``> 0.0``, exactly as the C does, which is what excludes
    :data:`INVALID_DISTANCE_CM`.
    """
    valid = [d for d in distances_cm if d > 0.0]
    if not valid:
        return INVALID_DISTANCE_CM, 0
    return sum(valid) / len(valid), len(valid)


def fill_percent(d_bar_cm: float) -> float:
    """``fillPercent()``: Fv = 100 * (1 - d_bar / H), clipped to [0, 100]."""
    fv = 100.0 * (1.0 - d_bar_cm / HEIGHT_CM)
    if fv < 0.0:
        return 0.0
    if fv > 100.0:
        return 100.0
    return fv


def is_overloaded(weight_kg: float) -> bool:
    """``isOverloaded()``: W >= ALPHA * W_MAX_KG. Note ``>=``, so the boundary is overloaded."""
    return weight_kg >= (ALPHA * W_MAX_KG)


def classify_state(fill_pct: float, overloaded: bool) -> int:
    """``classifyState()``: the paper's three-state rule, returning the ``field6`` code."""
    if overloaded or fill_pct >= RED_THRESHOLD_PCT:
        return STATE_RED
    if fill_pct >= YELLOW_THRESHOLD_PCT:
        return STATE_YELLOW
    return STATE_GREEN


def classify_reading(
    distances_cm: Sequence[float], weight_kg: float, *, clamp_inputs: bool = False
) -> FirmwareResult:
    """One pass of the sketch's ``loop()``, from raw distances to the state it would post.

    Args:
        distances_cm: per-sensor distances. A value ``<= 0`` is treated as a timed-out sensor and
            dropped from the mean, which is what :data:`INVALID_DISTANCE_CM` means.
        weight_kg: load-cell reading. Negatives are floored at 0, as the sketch does -- a
            negative mass is a tare error, not a measurement.
        clamp_inputs: when ``True``, apply the sketch's per-reading clamp to each distance first.
            ``readDistanceCm()`` has already clamped anything coming off real hardware, so the
            default is ``False`` and the caller is taken at its word. The cross-check passes
            pre-clamped values, matching what :class:`smartwaste.noise.SensorArray` produces.

    Raises:
        FirmwareNoValidSensors: every sensor timed out, which is the case the sketch refuses to
            post at all.
    """
    readings = list(distances_cm)
    if clamp_inputs:
        readings = [clamp_distance_cm(d) if d > 0.0 else d for d in readings]

    d_bar, n_valid = mean_valid_distance_cm(readings)
    if n_valid == 0:
        raise FirmwareNoValidSensors(
            "every ultrasonic sensor timed out; the sketch posts nothing in this case"
        )

    weight = max(0.0, weight_kg)
    fv = fill_percent(d_bar)
    overloaded = is_overloaded(weight)
    code = classify_state(fv, overloaded)

    return FirmwareResult(
        fill_pct=fv,
        overloaded=overloaded,
        state=STATE_CODE_TO_BIN_STATE[code],
        state_code=code,
        n_valid=n_valid,
        d_bar_cm=d_bar,
    )
