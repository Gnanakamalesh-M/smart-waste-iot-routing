"""Phase 1 experiment: does the paper's sensor fusion actually beat one sensor?

Three classifiers, scored against ground truth from the single-bin simulation:

- **(a) single centre sensor** -- one ultrasonic reading, no load cell.
- **(b) 3-sensor mean** -- the mean of three distances, no load cell.
- **(c) 3-sensor mean + load cell** -- the paper's full method.

(a) and (b) get no load cell, so they can never raise the overload flag. That is the honest
reading of "single sensor" and "mean only", and the weight-triggered REDs they miss are exactly
what the load cell contributes.

All three are scored on the **same** :class:`~smartwaste.noise.SensorReading`, so the comparison
is paired: the only difference between methods is which channels they read.

Honesty notes that belong with the numbers, not just in the docs:

- Noise levels come from ``config/default.yaml`` as written. None was tuned to make fusion win.
- Two of the six scenarios **favour fusion by construction** and are labelled as such in every
  output. ``uneven_center_peak`` defines a skew that biases the centre sensor, and
  ``stuck_center`` aims the fault at the one sensor method (a) depends on. They demonstrate a
  mechanism; they are not evidence about how often it occurs in a real bin.
- Confidence intervals resample **whole bins**, not readings. Readings within a bin share a
  density factor and a stuck fault, so resampling readings would understate the interval.
- These metrics measure how faithfully each sensor set reproduces the paper's own RED/YELLOW/GREEN
  rule. They are not collection outcomes -- nothing here routes a truck or counts an overflow.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

from smartwaste.bin_sim import simulate_bin
from smartwaste.config import Config
from smartwaste.fusion import BinState, fuse, truth_state
from smartwaste.noise import (
    CENTER_INDEX,
    NO_STUCK_SENSOR,
    SKEW_CENTER_PEAK,
    SKEW_NONE,
    SKEW_RANDOM_TILT,
    NoiseParams,
    SensorArray,
)

# Imported with an explicit `as` re-export: Phase 2's city benchmark shares this helper, and
# the Phase 1 import path plus the values pinned in tests/test_fusion_eval.py must not move.
from smartwaste.seeds import child_seed as child_seed

#: Scenario fairness labels, so a reader can tell a measurement from a demonstration.
FAIRNESS_CONTROL = "control"
FAIRNESS_FAIR = "fair test"
FAIRNESS_FAVORS_FUSION = "favors fusion by construction"
FAIRNESS_LABELS = (FAIRNESS_CONTROL, FAIRNESS_FAIR, FAIRNESS_FAVORS_FUSION)

CI_LEVEL = 0.95


@dataclass(frozen=True)
class MethodSpec:
    """One classifier under test."""

    key: str
    label: str
    use_load_cell: bool
    sensor_indices: tuple[int, ...] | None  # None = every sensor


METHODS: tuple[MethodSpec, ...] = (
    MethodSpec("a_center_only", "(a) single center sensor", False, (CENTER_INDEX,)),
    MethodSpec("b_mean3", "(b) 3-sensor mean", False, None),
    MethodSpec("c_mean3_loadcell", "(c) 3-sensor mean + load cell", True, None),
)


@dataclass(frozen=True)
class Scenario:
    """A named noise condition plus its fairness label."""

    name: str
    params: NoiseParams
    fairness: str


def build_scenarios(cfg: Config) -> tuple[Scenario, ...]:
    """The six noise scenarios, built from config values with no tuning."""
    s = cfg.sensors

    def noisy(
        *,
        skew_cm: float = 0.0,
        skew_mode: str = SKEW_NONE,
        stuck_sensor: int = NO_STUCK_SENSOR,
    ) -> NoiseParams:
        """The baseline noise every non-clean scenario carries, plus one extra effect."""
        return NoiseParams(
            sigma_cm=s.noise_sigma_cm,
            bias_max_cm=s.bias_max_cm,
            stuck_probability=s.stuck_probability,
            load_cell_sigma_kg=s.load_cell_sigma_kg,
            load_cell_bias_max_kg=s.load_cell_bias_max_kg,
            skew_cm=skew_cm,
            skew_mode=skew_mode,
            stuck_sensor=stuck_sensor,
        )

    return (
        # No noise at all: every method should agree with the truth exactly.
        Scenario("clean", NoiseParams(), FAIRNESS_CONTROL),
        # Symmetric noise + miscalibration. No mechanism here favours the array.
        Scenario("noisy", noisy(), FAIRNESS_FAIR),
        # Zero-mean tilt: the centre sensor is unbiased, so the mean can only cut variance.
        Scenario(
            "uneven_random_tilt",
            noisy(skew_cm=s.uneven_fill_skew_cm, skew_mode=SKEW_RANDOM_TILT),
            FAIRNESS_FAIR,
        ),
        # The fault hits a sensor method (a) never reads, so it handicaps the array instead.
        Scenario("stuck_left", noisy(stuck_sensor=0), FAIRNESS_FAIR),
        # The skew is defined to bias the centre sensor: fusion wins by construction.
        Scenario(
            "uneven_center_peak",
            noisy(skew_cm=s.uneven_fill_skew_cm, skew_mode=SKEW_CENTER_PEAK),
            FAIRNESS_FAVORS_FUSION,
        ),
        # The fault is aimed at the only sensor method (a) has: fusion wins by construction.
        Scenario("stuck_center", noisy(stuck_sensor=CENTER_INDEX), FAIRNESS_FAVORS_FUSION),
    )


@dataclass
class _BinCounts:
    """Per-bin tallies, kept separately so the bootstrap can resample whole bins."""

    total: int = 0
    wrong: int = 0
    not_red: int = 0
    false_red: int = 0
    red: int = 0
    false_green: int = 0


@dataclass(frozen=True)
class MethodMetrics:
    """One row of the results table: a method's scores under one scenario."""

    scenario: str
    fairness: str
    method: str
    method_label: str
    n_readings: int
    misclass_rate: float
    misclass_ci: tuple[float, float] | None
    n_not_red: int
    false_red_rate: float | None
    n_red: int
    false_green_rate: float | None
    false_green_ci: tuple[float, float] | None

    def as_row(self) -> dict[str, str]:
        """Flatten to CSV-friendly strings (fixed precision, so reruns are byte-identical)."""
        return {
            "scenario": self.scenario,
            "fairness": self.fairness,
            "method": self.method,
            "method_label": self.method_label,
            "n_readings": str(self.n_readings),
            "misclass_rate": _fmt(self.misclass_rate),
            "misclass_ci_low": _fmt_ci_part(self.misclass_ci, 0),
            "misclass_ci_high": _fmt_ci_part(self.misclass_ci, 1),
            "n_not_red": str(self.n_not_red),
            "false_red_rate": _fmt_opt(self.false_red_rate),
            "n_red": str(self.n_red),
            "false_green_rate": _fmt_opt(self.false_green_rate),
            "false_green_ci_low": _fmt_ci_part(self.false_green_ci, 0),
            "false_green_ci_high": _fmt_ci_part(self.false_green_ci, 1),
        }


CSV_FIELDS: tuple[str, ...] = (
    "scenario",
    "fairness",
    "method",
    "method_label",
    "n_readings",
    "misclass_rate",
    "misclass_ci_low",
    "misclass_ci_high",
    "n_not_red",
    "false_red_rate",
    "n_red",
    "false_green_rate",
    "false_green_ci_low",
    "false_green_ci_high",
)


def _fmt(value: float) -> str:
    return f"{value:.6f}"


def _fmt_opt(value: float | None) -> str:
    return "n/a" if value is None else _fmt(value)


def _fmt_ci_part(ci: tuple[float, float] | None, index: int) -> str:
    return "n/a" if ci is None else _fmt(ci[index])


def ratio(numerator: float, denominator: float) -> float | None:
    """A rate, or None when the denominator is empty.

    Returning None instead of 0.0 keeps an unmeasurable rate visibly unmeasured, rather than
    letting it read as a perfect score.
    """
    if denominator <= 0:
        return None
    return numerator / denominator


def bootstrap_ratio_ci(
    per_bin: Sequence[tuple[float, float]],
    rng: random.Random,
    n_resamples: int,
    level: float = CI_LEVEL,
) -> tuple[float, float] | None:
    """Percentile bootstrap CI for a ratio, resampling **whole bins** with replacement.

    Args:
        per_bin: one ``(numerator, denominator)`` pair per bin.
        rng: seeded generator, so the interval is reproducible.
        n_resamples: bootstrap replicates.
        level: coverage, e.g. 0.95.

    Returns:
        ``(low, high)``, or None when no bin contributes any denominator (the rate itself is
        then undefined and reported as ``n/a``).
    """
    if not per_bin or sum(d for _, d in per_bin) <= 0:
        return None

    n = len(per_bin)
    estimates: list[float] = []
    for _ in range(n_resamples):
        num = 0.0
        den = 0.0
        for _ in range(n):
            pair = per_bin[rng.randrange(n)]
            num += pair[0]
            den += pair[1]
        if den > 0:
            estimates.append(num / den)
    if not estimates:
        return None

    estimates.sort()
    tail = (1.0 - level) / 2.0
    low = estimates[min(len(estimates) - 1, int(tail * len(estimates)))]
    high = estimates[min(len(estimates) - 1, int((1.0 - tail) * len(estimates)))]
    return (low, high)


def evaluate_scenario(
    scenario: Scenario,
    cfg: Config,
    seed: int,
    n_bins: int,
    days: int,
) -> list[MethodMetrics]:
    """Simulate ``n_bins`` bins under one scenario and score all three methods on them."""
    counts: dict[str, list[_BinCounts]] = {m.key: [] for m in METHODS}

    for bin_index in range(n_bins):
        # Seeded per (scenario, bin) so a bin's trajectory does not depend on how many bins
        # ran before it -- and so --n-bins can change without reshuffling earlier bins.
        bin_rng = random.Random(child_seed(seed, f"{scenario.name}:bin:{bin_index}"))
        sensor_rng = random.Random(child_seed(seed, f"{scenario.name}:sensors:{bin_index}"))
        days_data = simulate_bin(cfg, bin_rng, days=days)
        array = SensorArray(scenario.params, cfg.bin, cfg.sensors, sensor_rng)

        per_method = {m.key: _BinCounts() for m in METHODS}
        for day in days_data:
            truth = truth_state(day.true_fill_pct, day.true_weight_kg, cfg.bin)
            reading = array.read(day.true_fill_pct, day.true_weight_kg)
            for method in METHODS:
                predicted = fuse(
                    reading,
                    cfg.bin,
                    use_load_cell=method.use_load_cell,
                    sensor_indices=method.sensor_indices,
                ).state
                tally = per_method[method.key]
                tally.total += 1
                if predicted is not truth:
                    tally.wrong += 1
                if truth is BinState.RED:
                    tally.red += 1
                    if predicted is BinState.GREEN:
                        tally.false_green += 1
                else:
                    tally.not_red += 1
                    if predicted is BinState.RED:
                        tally.false_red += 1
        for key, tally in per_method.items():
            counts[key].append(tally)

    out: list[MethodMetrics] = []
    for method in METHODS:
        bins = counts[method.key]
        total = sum(b.total for b in bins)
        wrong = sum(b.wrong for b in bins)
        n_red = sum(b.red for b in bins)
        n_not_red = sum(b.not_red for b in bins)

        misclass_rng = random.Random(
            child_seed(seed, f"bootstrap:misclass:{scenario.name}:{method.key}")
        )
        green_rng = random.Random(
            child_seed(seed, f"bootstrap:false_green:{scenario.name}:{method.key}")
        )
        resamples = cfg.fusion_eval.bootstrap_resamples

        out.append(
            MethodMetrics(
                scenario=scenario.name,
                fairness=scenario.fairness,
                method=method.key,
                method_label=method.label,
                n_readings=total,
                misclass_rate=wrong / total if total else 0.0,
                misclass_ci=bootstrap_ratio_ci(
                    [(b.wrong, b.total) for b in bins], misclass_rng, resamples
                ),
                n_not_red=n_not_red,
                false_red_rate=ratio(sum(b.false_red for b in bins), n_not_red),
                n_red=n_red,
                false_green_rate=ratio(sum(b.false_green for b in bins), n_red),
                false_green_ci=bootstrap_ratio_ci(
                    [(b.false_green, b.red) for b in bins], green_rng, resamples
                ),
            )
        )
    return out


def evaluate_all(cfg: Config, seed: int, n_bins: int, days: int) -> list[MethodMetrics]:
    """Every scenario x every method, in a stable order."""
    rows: list[MethodMetrics] = []
    for scenario in build_scenarios(cfg):
        rows.extend(evaluate_scenario(scenario, cfg, seed, n_bins, days))
    return rows


# --------------------------------------------------------------------------------------------
# Diagnostic sweep: where in the fill range does each method go wrong?
# --------------------------------------------------------------------------------------------

SWEEP_BUCKETS = 10


@dataclass(frozen=True)
class SweepRow:
    """Mean absolute Fv error and misclassification, by true-fill decile."""

    scenario: str
    method: str
    fill_bucket: str
    n: int
    mean_abs_fv_error: float
    misclass_rate: float

    def as_row(self) -> dict[str, str]:
        return {
            "scenario": self.scenario,
            "method": self.method,
            "fill_bucket": self.fill_bucket,
            "n": str(self.n),
            "mean_abs_fv_error": _fmt(self.mean_abs_fv_error),
            "misclass_rate": _fmt(self.misclass_rate),
        }


SWEEP_FIELDS: tuple[str, ...] = (
    "scenario",
    "method",
    "fill_bucket",
    "n",
    "mean_abs_fv_error",
    "misclass_rate",
)


@dataclass
class _SweepBucket:
    n: int = 0
    abs_error: float = 0.0
    wrong: int = 0


def sweep(cfg: Config, seed: int, n_samples: int) -> list[SweepRow]:
    """Draw true fill uniformly over [0, 100] and report error against the true fill.

    The episode-based headline table inherits whatever fill distribution the arrival process
    produces, which is uneven near the 60 and 80 boundaries where the errors actually live.
    This sweep covers the range evenly instead. It is a diagnostic, not the headline: a uniform
    fill distribution is not what a real bin does.
    """
    rows: list[SweepRow] = []
    for scenario in build_scenarios(cfg):
        buckets: dict[str, dict[int, _SweepBucket]] = {
            m.key: {b: _SweepBucket() for b in range(SWEEP_BUCKETS)} for m in METHODS
        }
        rng = random.Random(child_seed(seed, f"sweep:{scenario.name}"))
        for i in range(n_samples):
            true_fill = rng.uniform(0.0, 100.0)
            factor = rng.uniform(1.0 - cfg.bin.density_cv, 1.0 + cfg.bin.density_cv)
            true_weight = true_fill * cfg.bin.density_kg_per_pct * factor
            # A fresh array per sample keeps samples independent. A stuck sensor only shows up
            # once it has a previous value to latch, so prime the array with one throwaway
            # reading at an unrelated fill -- otherwise the stuck scenarios would silently
            # degrade into plain noise scenarios in this diagnostic.
            array = SensorArray(
                scenario.params,
                cfg.bin,
                cfg.sensors,
                random.Random(child_seed(seed, f"sweep:{scenario.name}:{i}")),
            )
            prime_fill = rng.uniform(0.0, 100.0)
            array.read(prime_fill, prime_fill * cfg.bin.density_kg_per_pct)
            reading = array.read(true_fill, true_weight)
            truth = truth_state(true_fill, true_weight, cfg.bin)
            index = min(SWEEP_BUCKETS - 1, int(true_fill / (100.0 / SWEEP_BUCKETS)))
            for method in METHODS:
                result = fuse(
                    reading,
                    cfg.bin,
                    use_load_cell=method.use_load_cell,
                    sensor_indices=method.sensor_indices,
                )
                bucket = buckets[method.key][index]
                bucket.n += 1
                bucket.abs_error += abs(result.fill_pct - true_fill)
                if result.state is not truth:
                    bucket.wrong += 1
        for method in METHODS:
            for index in range(SWEEP_BUCKETS):
                bucket = buckets[method.key][index]
                if bucket.n == 0:
                    continue
                low = index * (100 // SWEEP_BUCKETS)
                rows.append(
                    SweepRow(
                        scenario=scenario.name,
                        method=method.key,
                        fill_bucket=f"{low}-{low + 100 // SWEEP_BUCKETS}",
                        n=bucket.n,
                        mean_abs_fv_error=bucket.abs_error / bucket.n,
                        misclass_rate=bucket.wrong / bucket.n,
                    )
                )
    return rows


# --------------------------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------------------------


def _ci_text(ci: tuple[float, float] | None) -> str:
    if ci is None:
        return "n/a"
    return f"[{ci[0] * 100:.2f}, {ci[1] * 100:.2f}]"


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.2f}"


def render_markdown(rows: Sequence[MethodMetrics], title: str) -> str:
    """Render the results as a markdown table. No timestamps: reruns must be identical."""
    lines = [
        f"### {title}",
        "",
        "| scenario | fairness | method | n | misclass % | misclass 95% CI | "
        "false-RED % | n(not RED) | false-GREEN % | false-GREEN 95% CI | n(RED) |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r.scenario} | {r.fairness} | {r.method_label} | {r.n_readings} | "
            f"{_pct(r.misclass_rate)} | {_ci_text(r.misclass_ci)} | "
            f"{_pct(r.false_red_rate)} | {r.n_not_red} | "
            f"{_pct(r.false_green_rate)} | {_ci_text(r.false_green_ci)} | {r.n_red} |"
        )
    return "\n".join(lines)


def verdicts(rows: Sequence[MethodMetrics]) -> list[str]:
    """One plain-language verdict per scenario, including when fusion loses."""
    by_scenario: dict[str, dict[str, MethodMetrics]] = {}
    order: list[str] = []
    for row in rows:
        if row.scenario not in by_scenario:
            by_scenario[row.scenario] = {}
            order.append(row.scenario)
        by_scenario[row.scenario][row.method] = row

    out: list[str] = []
    for name in order:
        group = by_scenario[name]
        a = group["a_center_only"]
        b = group["b_mean3"]
        c = group["c_mean3_loadcell"]
        parts = [f"{name} [{a.fairness}]:"]
        for label, challenger in (("(b) mean3", b), ("(c) mean3+load cell", c)):
            delta = challenger.misclass_rate - a.misclass_rate
            if abs(delta) < 1e-12:
                # An exact tie needs no CI caveat.
                parts.append(f"{label} ties (a);")
                continue
            if delta < 0:
                parts.append(f"{label} beats (a) by {abs(delta) * 100:.2f} pp")
            else:
                parts.append(f"{label} is WORSE than (a) by {delta * 100:.2f} pp")
            parts[-1] += (
                " (CIs overlap: unresolved);"
                if _overlaps(challenger.misclass_ci, a.misclass_ci)
                else ";"
            )
        out.append(" ".join(parts))
    return out


def _overlaps(x: tuple[float, float] | None, y: tuple[float, float] | None) -> bool:
    if x is None or y is None:
        return False
    return x[0] <= y[1] and y[0] <= x[1]
