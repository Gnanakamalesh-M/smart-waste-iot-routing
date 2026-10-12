"""Sensor noise model for the 3 ultrasonic sensors and the load cell.

The paper reports no sensor error model at all, so its claim that three sensors plus a load cell
beat one sensor cannot be checked against it. This module supplies the error model that makes
that claim measurable. Every effect here is a *modelling choice of ours*, not a measurement from
the physical bin -- see docs/ASSUMPTIONS.md.

Four effects, applied in this order to each reading:

1. **Uneven / lateral fill.** The waste surface is not flat, so the left, centre and right
   sensors look at different heights. Two shapes are modelled:

   - ``center_peak``: the paper angles the inner walls to push waste towards the centre, so the
     heap peaks in the middle. Height offsets are ``(-s/2, +s, -s/2)`` for ``s ~ U(0, skew_cm)``,
     which leaves the mean of the three *distances* unbiased while the centre sensor reads
     systematically short. This is the mechanism that lets fusion win.
   - ``random_tilt``: the surface tilts in a random direction, offsets ``(+t, 0, -t)`` for
     ``t ~ U(-skew_cm, +skew_cm)``. Here the centre sensor is unbiased too, so the 3-sensor mean
     can only win by variance reduction -- a much weaker effect.

2. **Gaussian measurement noise**, independent per sensor per reading.
3. **Constant per-sensor bias** (miscalibration), drawn once when the array is constructed and
   fixed for the array's life. Averaging three independent biases shrinks their spread; it does
   not remove a bias common to all three.
4. **Stuck fault.** Either one nominated sensor latches its first reading for the whole episode
   (the realistic hard failure), or every sensor independently repeats its previous value with
   ``stuck_probability``.

Distances are finally clamped to ``[min_distance_cm, height_cm]``, which is what a real HC-SR04
does at the ends of its range. Note that this clamp is itself a small bias near a full or empty
bin, in a direction that happens to *help* every method equally.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from smartwaste.config import BinConfig, SensorsConfig

SKEW_NONE = "none"
SKEW_CENTER_PEAK = "center_peak"
SKEW_RANDOM_TILT = "random_tilt"
SKEW_MODES = (SKEW_NONE, SKEW_CENTER_PEAK, SKEW_RANDOM_TILT)

#: Index of the centre sensor in the (left, centre, right) array.
CENTER_INDEX = 1

NO_STUCK_SENSOR = -1


@dataclass(frozen=True)
class NoiseParams:
    """One noise scenario's settings.

    Not loaded from YAML (the config loader only takes scalars), so scenarios are built in
    :mod:`smartwaste.fusion_eval` from config values.
    """

    sigma_cm: float = 0.0
    bias_max_cm: float = 0.0
    skew_cm: float = 0.0
    skew_mode: str = SKEW_NONE
    stuck_sensor: int = NO_STUCK_SENSOR
    stuck_probability: float = 0.0
    load_cell_sigma_kg: float = 0.0
    load_cell_bias_max_kg: float = 0.0

    def validate(self) -> None:
        if self.skew_mode not in SKEW_MODES:
            raise ValueError(f"skew_mode must be one of {SKEW_MODES}, got {self.skew_mode!r}")
        if self.sigma_cm < 0:
            raise ValueError(f"sigma_cm must be >= 0, got {self.sigma_cm}")
        if not 0 <= self.stuck_probability <= 1:
            raise ValueError(f"stuck_probability must be in [0, 1], got {self.stuck_probability}")


@dataclass(frozen=True)
class SensorReading:
    """What a classifier is allowed to see: noisy distances and one noisy weight.

    Ground truth is deliberately **not** a field. That is the structural reason a classifier in
    this project cannot read the truth even by mistake.
    """

    distances_cm: tuple[float, ...]
    weight_kg: float


class SensorArray:
    """One bin's sensor hardware: ``n`` ultrasonic sensors plus a load cell.

    One instance per simulated bin, because the constant biases and the stuck fault belong to
    that bin's hardware and must persist across its readings.
    """

    def __init__(
        self,
        params: NoiseParams,
        bin_cfg: BinConfig,
        sensors_cfg: SensorsConfig,
        rng: random.Random,
    ) -> None:
        params.validate()
        if params.stuck_sensor != NO_STUCK_SENSOR and not (
            0 <= params.stuck_sensor < sensors_cfg.n_ultrasonic
        ):
            raise ValueError(f"stuck_sensor out of range: {params.stuck_sensor}")
        self._params = params
        self._bin = bin_cfg
        self._sensors = sensors_cfg
        self._rng = rng
        self._n = sensors_cfg.n_ultrasonic
        # Miscalibration is a property of the hardware: drawn once, then fixed.
        self._biases = tuple(
            rng.uniform(-params.bias_max_cm, params.bias_max_cm) for _ in range(self._n)
        )
        self._weight_bias = rng.uniform(-params.load_cell_bias_max_kg, params.load_cell_bias_max_kg)
        self._last: list[float | None] = [None] * self._n

    def _height_offsets(self) -> tuple[float, ...]:
        """Per-sensor waste-height offsets from the uneven-fill model (cm, + = higher waste)."""
        mode = self._params.skew_mode
        if mode == SKEW_NONE or self._params.skew_cm == 0.0 or self._n != 3:
            return (0.0,) * self._n
        if mode == SKEW_CENTER_PEAK:
            s = self._rng.uniform(0.0, self._params.skew_cm)
            # Centre heaps up by s; the sides drop by s/2 each, so the mean height is unchanged.
            return (-s / 2.0, s, -s / 2.0)
        t = self._rng.uniform(-self._params.skew_cm, self._params.skew_cm)
        return (t, 0.0, -t)

    def read(self, true_fill_pct: float, true_weight_kg: float) -> SensorReading:
        """Take one noisy reading of a bin whose true state is given.

        The true values are consumed here and never put into the returned
        :class:`SensorReading`.
        """
        height_cm = self._bin.height_cm
        waste_height = height_cm * true_fill_pct / 100.0
        offsets = self._height_offsets()

        distances: list[float] = []
        for i in range(self._n):
            # Higher waste at this sensor means a shorter distance to it.
            nominal = height_cm - (waste_height + offsets[i])
            value = nominal + self._biases[i]
            if self._params.sigma_cm > 0:
                value += self._rng.gauss(0.0, self._params.sigma_cm)

            previous = self._last[i]
            if previous is not None and self._is_stuck(i):
                value = previous

            value = max(self._sensors.min_distance_cm, min(height_cm, value))
            self._last[i] = value
            distances.append(value)

        weight = true_weight_kg + self._weight_bias
        if self._params.load_cell_sigma_kg > 0:
            weight += self._rng.gauss(0.0, self._params.load_cell_sigma_kg)
        return SensorReading(distances_cm=tuple(distances), weight_kg=max(0.0, weight))

    def _is_stuck(self, index: int) -> bool:
        """A nominated sensor is stuck for good; otherwise it is a per-reading coin flip.

        The coin flip is drawn for every sensor on every reading even when the probability is 0,
        so that adding a stuck fault does not shift the rest of the random stream and silently
        change every other number in a scenario.
        """
        draw = self._rng.random()
        if index == self._params.stuck_sensor:
            return True
        return draw < self._params.stuck_probability
