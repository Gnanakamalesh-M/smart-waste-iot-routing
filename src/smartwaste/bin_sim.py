"""Single-bin simulation: the ground truth everything else is scored against.

Paper-faithful in its state variable: the paper tracks weight, ``W(t) = W(t-1) + dW(t)`` with
random daily increments (SPEC section 2). Volume fill follows from weight via a density.

Why a *per-bin* density rather than one global constant: if weight were a fixed multiple of
volume, the load cell would carry exactly the same information as the ultrasonic array and could
add nothing by construction. Real bins differ -- a bin of wet kitchen waste is heavy at 60% full,
a bin of bulky packaging is light at 100%. ``bin.density_cv`` is the spread we assume, and it is
the entire reason the paper's load cell can help. With ``density_cv = 0`` the weight channel is
redundant; the Phase 1 evaluation reports that case explicitly as a sensitivity check.

Everything in a :class:`BinDay` is ground truth. Only :mod:`smartwaste.noise` (to corrupt it)
and :func:`smartwaste.fusion.truth_state` (to label it) are allowed to read these values.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from smartwaste.config import Config
from smartwaste.fusion import BinState, truth_state


@dataclass(frozen=True)
class BinDay:
    """One day of one bin, as ground truth."""

    day: int
    true_weight_kg: float
    true_fill_pct: float


def simulate_bin(cfg: Config, rng: random.Random, days: int | None = None) -> list[BinDay]:
    """Simulate one bin for ``days`` days.

    The bin is emptied on the day after its **ground-truth** state is RED. Using ground truth
    rather than a method's estimate is deliberate: every method then sees the identical fill
    trajectory, so the comparison between them is paired and the difference is purely
    classification quality. A real fleet would of course act on the noisy estimate and so drift
    onto a different trajectory -- that feedback loop is out of scope for Phase 1.

    Args:
        cfg: supplies the arrival process (``sim``) and the bin's geometry/thresholds (``bin``).
        rng: seeded generator; the same seed gives the same list.
        days: horizon, defaulting to ``fusion_eval.days``.

    Returns:
        One :class:`BinDay` per day, in day order.
    """
    horizon = cfg.fusion_eval.days if days is None else days
    if horizon < 1:
        raise ValueError(f"days must be >= 1, got {horizon}")

    bin_cfg = cfg.bin
    # This bin's waste density, fixed for its lifetime (see module docstring).
    density_factor = rng.uniform(1.0 - bin_cfg.density_cv, 1.0 + bin_cfg.density_cv)
    kg_per_pct = bin_cfg.density_kg_per_pct * density_factor

    out: list[BinDay] = []
    weight = 0.0
    for day in range(horizon):
        weight += max(0.0, rng.gauss(cfg.sim.daily_fill_mean_kg, cfg.sim.daily_fill_sd_kg))
        fill_pct = min(100.0, weight / kg_per_pct)
        out.append(BinDay(day=day, true_weight_kg=weight, true_fill_pct=fill_pct))

        if truth_state(fill_pct, weight, bin_cfg) is BinState.RED:
            weight = 0.0  # collected overnight
    return out
