"""The paper's sensor-fusion rule, implemented exactly as published.

From SPEC section 2 (the paper's equations):

    d_bar = (d1 + d2 + d3) / 3
    phi   = 1 - d_bar / H
    Fv    = 100 * phi, clipped to [0, 100]
    W_th  = alpha * W_max,  0 < alpha < 1
    Iw    = 1 if W >= W_th else 0
    state = RED    if Iw == 1 or Fv >= 80
            YELLOW if Iw == 0 and 60 <= Fv < 80
            GREEN  otherwise

The same :func:`classify` is used for predictions and for ground truth. Ground truth just feeds
it the true fill and true weight instead of sensor estimates, so any difference between a
method's output and the truth can only come from the sensors -- never from two rule
implementations that drifted apart.

Nothing here can read ground truth by accident: a classifier is handed a
:class:`~smartwaste.noise.SensorReading`, which carries only noisy distances and a noisy weight.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum

from smartwaste.config import BinConfig
from smartwaste.noise import SensorReading


class BinState(StrEnum):
    """The paper's three bin states."""

    GREEN = "GREEN"
    YELLOW = "YELLOW"
    RED = "RED"


@dataclass(frozen=True)
class FusionResult:
    """What a classifier concluded, and the intermediate values that got it there."""

    fill_pct: float
    overloaded: bool
    state: BinState


def mean_distance(distances: Sequence[float]) -> float:
    """d_bar: the plain arithmetic mean of the ultrasonic distances.

    With one distance this is that distance, which is how the single-center-sensor method
    reuses the same code path as the 3-sensor mean.
    """
    if not distances:
        raise ValueError("mean_distance needs at least one distance")
    return sum(distances) / len(distances)


def median_distance(distances: Sequence[float]) -> float:
    """The median of the ultrasonic distances -- our extension, NOT the paper's rule.

    The paper specifies a plain arithmetic mean. Phase 1 measured why that is fragile: with one
    sensor hard-latched, the mean was 20.85 pp *worse* than the single centre sensor, because
    averaging cannot discard a dead reading. A median of three survives one bad sensor outright.

    The trade-off is real and runs the other way when the hardware is healthy: with three good
    sensors the median of 3 is a noisier estimator of the surface height than the mean (it throws
    away two readings' worth of averaging). So this is expected to lose slightly at a zero fault
    rate and win as faults rise; Phase 2 reports both rather than picking the flattering one.
    """
    if not distances:
        raise ValueError("median_distance needs at least one distance")
    ordered = sorted(distances)
    n = len(ordered)
    middle = n // 2
    if n % 2 == 1:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def fill_percent(d_bar: float, height_cm: float) -> float:
    """Fv = 100 * (1 - d_bar / H), clipped to [0, 100].

    Clipping matters: noise can push d_bar above H (reads as a negative fill) or below 0
    (over 100%), and the paper's state rule assumes a percentage.
    """
    if height_cm <= 0:
        raise ValueError(f"height_cm must be > 0, got {height_cm}")
    fv = 100.0 * (1.0 - d_bar / height_cm)
    return max(0.0, min(100.0, fv))


def is_overloaded(weight_kg: float, w_max_kg: float, alpha: float) -> bool:
    """Iw: True when W >= alpha * W_max (the load cell's overload flag)."""
    return weight_kg >= alpha * w_max_kg


def classify(
    fill_pct: float,
    overloaded: bool,
    yellow_threshold_pct: float,
    red_threshold_pct: float,
) -> BinState:
    """The paper's state rule. Overload forces RED regardless of the measured fill."""
    if overloaded or fill_pct >= red_threshold_pct:
        return BinState.RED
    if fill_pct >= yellow_threshold_pct:
        return BinState.YELLOW
    return BinState.GREEN


def fuse(
    reading: SensorReading,
    bin_cfg: BinConfig,
    *,
    use_load_cell: bool,
    sensor_indices: Sequence[int] | None = None,
    reducer: Callable[[Sequence[float]], float] = mean_distance,
) -> FusionResult:
    """Run the paper's rule over one sensor reading.

    Args:
        reading: noisy distances and weight. The only input a classifier ever gets.
        bin_cfg: geometry (H, W_max) and thresholds (alpha, 60, 80).
        use_load_cell: when False the overload flag is forced to 0, so the method decides on
            distance alone. This is what separates the ultrasonic-only methods from the
            paper's full fusion.
        sensor_indices: which sensors this method reads, or all of them when None. The
            single-center-sensor method passes ``(CENTER_INDEX,)``.
        reducer: how the chosen distances collapse to ``d_bar``. Defaults to
            :func:`mean_distance`, which is the paper's published rule, so every existing caller
            keeps its exact behaviour. Phase 2's median-fusion extension passes
            :func:`median_distance` here and changes nothing else.
    """
    if sensor_indices is None:
        distances: Sequence[float] = reading.distances_cm
    else:
        distances = [reading.distances_cm[i] for i in sensor_indices]
    fv = fill_percent(reducer(distances), bin_cfg.height_cm)
    overloaded = use_load_cell and is_overloaded(reading.weight_kg, bin_cfg.w_max_kg, bin_cfg.alpha)
    state = classify(fv, overloaded, bin_cfg.yellow_threshold_pct, bin_cfg.red_threshold_pct)
    return FusionResult(fill_pct=fv, overloaded=overloaded, state=state)


def truth_state(true_fill_pct: float, true_weight_kg: float, bin_cfg: BinConfig) -> BinState:
    """Ground truth: the same rule applied to the TRUE fill and TRUE weight.

    Takes plain floats rather than a ``BinDay`` so that this module stays below ``bin_sim`` in
    the import order (``config -> noise -> fusion -> bin_sim``) with no import cycle.
    """
    overloaded = is_overloaded(true_weight_kg, bin_cfg.w_max_kg, bin_cfg.alpha)
    return classify(
        true_fill_pct,
        overloaded,
        bin_cfg.yellow_threshold_pct,
        bin_cfg.red_threshold_pct,
    )
