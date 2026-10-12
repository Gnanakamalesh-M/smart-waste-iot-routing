"""Tests for the single-bin ground-truth simulation."""

from __future__ import annotations

import random
from typing import Any

import pytest

from smartwaste import load_config
from smartwaste.bin_sim import BinDay, simulate_bin
from smartwaste.config import Config
from smartwaste.fusion import BinState, truth_state

pytestmark = pytest.mark.sim


def run(
    seed: int = 42,
    days: int = 30,
    overrides: dict[str, Any] | None = None,
) -> tuple[list[BinDay], Config]:
    cfg = load_config(overrides=overrides)
    return simulate_bin(cfg, random.Random(seed), days=days), cfg


def test_returns_one_entry_per_day_in_order() -> None:
    days_data, _ = run(days=25)
    assert len(days_data) == 25
    assert [d.day for d in days_data] == list(range(25))


def test_fill_stays_within_bounds() -> None:
    for seed in range(20):
        days_data, _ = run(seed=seed, days=60)
        for d in days_data:
            assert 0.0 <= d.true_fill_pct <= 100.0
            assert d.true_weight_kg >= 0.0


def test_weight_only_drops_after_a_red_day() -> None:
    """Weight accumulates; the only decrease allowed is the collection after a RED day."""
    days_data, cfg = run(days=90)
    for previous, current in zip(days_data[:-1], days_data[1:], strict=True):
        if current.true_weight_kg < previous.true_weight_kg:
            assert (
                truth_state(previous.true_fill_pct, previous.true_weight_kg, cfg.bin)
                is BinState.RED
            ), "weight dropped on a day that was not collected"


def test_a_red_day_is_followed_by_a_near_empty_bin() -> None:
    days_data, cfg = run(days=90)
    reds = [
        i
        for i, d in enumerate(days_data[:-1])
        if truth_state(d.true_fill_pct, d.true_weight_kg, cfg.bin) is BinState.RED
    ]
    assert reds, "the horizon should contain at least one RED day"
    for i in reds:
        # Emptied overnight, then one day's arrivals land on an empty bin.
        assert days_data[i + 1].true_weight_kg < days_data[i].true_weight_kg


def test_the_bin_actually_reaches_red() -> None:
    """If it never filled up, the experiment would have no RED class to score against."""
    days_data, cfg = run(days=30)
    states = {truth_state(d.true_fill_pct, d.true_weight_kg, cfg.bin) for d in days_data}
    assert BinState.RED in states


def test_same_seed_gives_identical_days() -> None:
    assert run(seed=7)[0] == run(seed=7)[0]


def test_different_seed_gives_different_days() -> None:
    assert run(seed=7)[0] != run(seed=8)[0]


def test_density_cv_zero_ties_weight_to_fill_exactly() -> None:
    """No density spread means weight is a fixed multiple of volume: the load cell's blind spot."""
    days_data, cfg = run(overrides={"bin": {"density_cv": 0.0}})
    for d in days_data:
        if d.true_fill_pct < 100.0:
            assert d.true_weight_kg == pytest.approx(d.true_fill_pct * cfg.bin.density_kg_per_pct)


def test_density_cv_spreads_the_weight_at_a_given_fill() -> None:
    """With spread, two bins at the same fill can differ in weight -- why the load cell helps."""
    cfg = load_config()
    ratios = set()
    for seed in range(30):
        days_data = simulate_bin(cfg, random.Random(seed), days=5)
        d = days_data[0]
        if d.true_fill_pct > 0:
            ratios.add(round(d.true_weight_kg / d.true_fill_pct, 6))
    assert len(ratios) > 1


def test_rejects_a_non_positive_horizon() -> None:
    cfg = load_config()
    with pytest.raises(ValueError, match="days"):
        simulate_bin(cfg, random.Random(0), days=0)
