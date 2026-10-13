"""Generate sensor streams with PLANTED faults, then score two detectors against the truth.

One command (``scripts/run_anomaly.py``) produces every anomaly table in
``docs/PHASE3_RESULTS.md``.

### Two files, and only the scorer opens the second one

:func:`write_dataset` writes

- ``data/anomaly_readings.csv`` -- feature columns only: bin, day, the three distances, weight;
- ``data/anomaly_labels.csv``  -- the planted truth: bin, day, fault family.

The detectors in :mod:`smartwaste.ml.anomaly` are handed :class:`~smartwaste.ml.anomaly.
SensorWindow` objects built from the readings alone. They have no parameter that could take a
label and never name the labels file. Only :func:`score_detectors` here reads the labels, and it
reads them *after* every threshold has been frozen.

### Thresholds are selected on a different city

:func:`select_thresholds` runs on the **validation seed** (``seed +
anomaly.validation_seed_offset``) and picks, by F1 on *that* city's planted labels:

- the rule baseline's stuck tolerance and disagreement limit, from
  :data:`RULE_TOLERANCE_GRID` / :data:`RULE_DISAGREEMENT_GRID`;
- the Isolation Forest's contamination, from :data:`CONTAMINATION_GRID`.

Those values are then applied **unchanged** to the scoring seed. The config values are the
starting points of the grids and are themselves invented and uncited.

### Scoring

Precision, recall and PR-AUC, overall and **per fault family**, each printed beside the
positive-class prevalence (PR-AUC is only meaningful against it -- a useless ranker scores about
the prevalence, not 0.5). **Accuracy is never reported**: faulty *windows* are a small minority
(``fault_bin_fraction`` is a share of bins, and each episode covers only part of the horizon), so
"nothing is ever faulty" would score very high accuracy while catching nothing. The exact trivial
accuracy is computed from the real rows and printed in the table's note rather than asserted
here.

Each family carries a fairness label from :data:`~smartwaste.ml.anomaly.FAMILY_FAIRNESS`, because
the rule baseline's zero-variance test *is* a stuck/dropout detector and a win there is a
demonstration rather than a measurement.

### Window labelling

A window is labelled faulty when **its last day falls inside the planted fault episode**. That is
a choice, not a fact: labelling by "any day overlaps" would mark early windows faulty before the
fault had produced enough evidence to be visible, and labelling by "every day overlaps" would
discard the onset entirely. It is stated here and in the results doc.
"""

from __future__ import annotations

import csv
import random
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from smartwaste.config import AnomalyConfig, Config
from smartwaste.ml.anomaly import (
    DETECTOR_KEYS,
    FAMILY_FAIRNESS,
    FAULT_DRIFT,
    FAULT_DROPOUT,
    FAULT_FAMILIES,
    FAULT_NOISE_BURST,
    FAULT_STUCK,
    Detector,
    RuleDetector,
    SensorWindow,
    build_detector,
)
from smartwaste.ml.metrics import ClassificationScore, score_detector
from smartwaste.noise import NO_STUCK_SENSOR, NoiseParams, SensorArray
from smartwaste.seeds import child_seed

READINGS_FILENAME = "anomaly_readings.csv"
LABELS_FILENAME = "anomaly_labels.csv"

READINGS_FIELDS = ("bin", "day", "d1_cm", "d2_cm", "d3_cm", "weight_kg")
LABELS_FIELDS = ("bin", "day", "fault_family")

#: Threshold grids searched on the VALIDATION seed only. Invented and uncited; the config values
#: sit inside each grid so "what the config says" is always among the candidates.
RULE_TOLERANCE_GRID = (0.0, 0.05, 0.5, 1.0, 2.0)
RULE_DISAGREEMENT_GRID = (10.0, 15.0, 20.0, 25.0, 35.0, 50.0)
CONTAMINATION_GRID = (0.05, 0.10, 0.15, 0.20, 0.30)


@dataclass(frozen=True)
class PlantedFault:
    """One bin's planted fault: which sensor, which family, which days."""

    bin_index: int
    sensor: int
    family: str
    start_day: int
    end_day: int

    def covers(self, day: int) -> bool:
        return self.start_day <= day <= self.end_day


@dataclass(frozen=True)
class AnomalyDataset:
    """Readings, the planted faults, and the windows derived from the readings.

    ``windows`` and ``labels`` are parallel: ``labels[i]`` is the truth for ``windows[i]``. They
    are kept as separate sequences rather than one object so a detector can be handed the windows
    with no possibility of touching the labels.
    """

    #: ``distances[bin][day][sensor]``
    distances: tuple[tuple[tuple[float, ...], ...], ...]
    #: ``weights[bin][day]``
    weights: tuple[tuple[float, ...], ...]
    faults: tuple[PlantedFault, ...]
    windows: tuple[SensorWindow, ...]
    labels: tuple[bool, ...]
    families: tuple[str, ...]

    @property
    def n_bins(self) -> int:
        return len(self.distances)

    @property
    def prevalence(self) -> float:
        return sum(self.labels) / len(self.labels) if self.labels else 0.0


def plant_faults(cfg: Config, seed: int) -> tuple[PlantedFault, ...]:
    """Choose which bins are faulty, with which family, on which days.

    Every draw happens for every bin whether or not it ends up faulty, so changing
    ``fault_bin_fraction`` cannot reshuffle the stream for the bins that follow -- the same
    controlled-comparison discipline :func:`smartwaste.city_sim.build_layout` uses.
    """
    acfg = cfg.anomaly
    rng = random.Random(child_seed(seed, "anomaly_faults"))
    faults: list[PlantedFault] = []
    for bin_index in range(acfg.n_bins):
        faulty = rng.random() < acfg.fault_bin_fraction
        sensor = rng.randrange(cfg.sensors.n_ultrasonic)
        family = FAULT_FAMILIES[rng.randrange(len(FAULT_FAMILIES))]
        duration = rng.randint(acfg.min_fault_days, acfg.max_fault_days)
        start = rng.randint(0, max(0, acfg.days - duration))
        if faulty:
            faults.append(
                PlantedFault(
                    bin_index=bin_index,
                    sensor=sensor,
                    family=family,
                    start_day=start,
                    end_day=start + duration - 1,
                )
            )
    return tuple(faults)


def generate_readings(
    cfg: Config, seed: int, faults: Sequence[PlantedFault]
) -> tuple[tuple[tuple[tuple[float, ...], ...], ...], tuple[tuple[float, ...], ...]]:
    """Simulate each bin's sensor stream, then inject its planted fault.

    The healthy stream comes from the existing :class:`~smartwaste.noise.SensorArray`, so the
    baseline noise here is the same model Phases 1 and 2 used -- the faults are an addition to
    it, not a different sensor model.

    The four families, as transforms of the faulty sensor's reading:

    - ``stuck``: latches the value it held when the fault began.
    - ``drift``: accumulates ``drift_cm_per_day`` for each day of the episode.
    - ``dropout``: reads full range (``height_cm``), i.e. "I see nothing" -- a bin that looks
      empty however full it is.
    - ``noise_burst``: gains an extra Gaussian with ``noise_burst_sigma_cm``.
    """
    acfg = cfg.anomaly
    fault_by_bin = {fault.bin_index: fault for fault in faults}
    height = cfg.bin.height_cm
    min_distance = cfg.sensors.min_distance_cm

    all_distances: list[tuple[tuple[float, ...], ...]] = []
    all_weights: list[tuple[float, ...]] = []

    for bin_index in range(acfg.n_bins):
        rng = random.Random(child_seed(seed, f"anomaly:{bin_index}"))
        array = SensorArray(
            NoiseParams(
                sigma_cm=cfg.sensors.noise_sigma_cm,
                bias_max_cm=cfg.sensors.bias_max_cm,
                skew_cm=cfg.sensors.uneven_fill_skew_cm,
                skew_mode="center_peak",
                stuck_sensor=NO_STUCK_SENSOR,
                stuck_probability=0.0,
                load_cell_sigma_kg=cfg.sensors.load_cell_sigma_kg,
                load_cell_bias_max_kg=cfg.sensors.load_cell_bias_max_kg,
            ),
            cfg.bin,
            cfg.sensors,
            rng,
        )
        fault = fault_by_bin.get(bin_index)
        fault_rng = random.Random(child_seed(seed, f"anomaly_fault:{bin_index}"))

        # A simple saw-tooth fill: rise at a per-bin rate, emptied on a fixed calendar. The
        # anomaly dataset is about sensor behaviour, so the fill process is kept deliberately
        # plain rather than reusing the full city demand model.
        rate = rng.uniform(4.0, 12.0)
        period = rng.randint(5, 12)
        latched: float | None = None

        bin_distances: list[tuple[float, ...]] = []
        bin_weights: list[float] = []
        for day in range(acfg.days):
            fill = min(100.0, rate * (day % period))
            weight = fill * cfg.bin.density_kg_per_pct
            reading = array.read(fill, weight)
            distances = list(reading.distances_cm)

            if fault is not None and fault.covers(day):
                sensor = fault.sensor
                if fault.family == FAULT_STUCK:
                    if latched is None:
                        latched = distances[sensor]
                    distances[sensor] = latched
                elif fault.family == FAULT_DRIFT:
                    elapsed = day - fault.start_day + 1
                    distances[sensor] += acfg.drift_cm_per_day * elapsed
                elif fault.family == FAULT_DROPOUT:
                    distances[sensor] = height
                elif fault.family == FAULT_NOISE_BURST:
                    distances[sensor] += fault_rng.gauss(0.0, acfg.noise_burst_sigma_cm)
                distances[sensor] = max(min_distance, min(height, distances[sensor]))

            bin_distances.append(tuple(distances))
            bin_weights.append(reading.weight_kg)

        all_distances.append(tuple(bin_distances))
        all_weights.append(tuple(bin_weights))

    return tuple(all_distances), tuple(all_weights)


def build_windows(
    cfg: Config,
    distances: Sequence[Sequence[Sequence[float]]],
    weights: Sequence[Sequence[float]],
    faults: Sequence[PlantedFault],
) -> tuple[tuple[SensorWindow, ...], tuple[bool, ...], tuple[str, ...]]:
    """Roll a window over each bin's stream and label it by its last day.

    Returns windows, their labels, and the fault family of each window (``""`` when healthy) so
    the scorer can break results out per family.
    """
    width = cfg.anomaly.window_days
    fault_by_bin = {fault.bin_index: fault for fault in faults}

    windows: list[SensorWindow] = []
    labels: list[bool] = []
    families: list[str] = []
    for bin_index, bin_distances in enumerate(distances):
        fault = fault_by_bin.get(bin_index)
        for end in range(width - 1, len(bin_distances)):
            start = end - width + 1
            windows.append(
                SensorWindow(
                    bin_index=bin_index,
                    end_day=end,
                    distances=tuple(tuple(row) for row in bin_distances[start : end + 1]),
                    weights=tuple(weights[bin_index][start : end + 1]),
                )
            )
            faulty = fault is not None and fault.covers(end)
            labels.append(faulty)
            families.append(fault.family if (faulty and fault is not None) else "")
    return tuple(windows), tuple(labels), tuple(families)


def build_dataset(cfg: Config, seed: int) -> AnomalyDataset:
    """Plant faults, generate the readings, and window them."""
    faults = plant_faults(cfg, seed)
    distances, weights = generate_readings(cfg, seed, faults)
    windows, labels, families = build_windows(cfg, distances, weights, faults)
    return AnomalyDataset(
        distances=distances,
        weights=weights,
        faults=faults,
        windows=windows,
        labels=labels,
        families=families,
    )


def write_dataset(dataset: AnomalyDataset, out_dir: Path) -> tuple[Path, Path]:
    """Write readings and labels to **separate** files. Returns both paths.

    The separation is the point: a detector is given the readings path and could not find the
    labels even if it tried to read a file, because it is never told where they are.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    readings_path = out_dir / READINGS_FILENAME
    labels_path = out_dir / LABELS_FILENAME

    with readings_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(READINGS_FIELDS)
        for bin_index, bin_distances in enumerate(dataset.distances):
            for day, row in enumerate(bin_distances):
                writer.writerow(
                    [bin_index, day, *(f"{value:.4f}" for value in row)]
                    + [f"{dataset.weights[bin_index][day]:.4f}"]
                )

    with labels_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(LABELS_FIELDS)
        for fault in dataset.faults:
            for day in range(fault.start_day, fault.end_day + 1):
                writer.writerow([fault.bin_index, day, fault.family])

    return readings_path, labels_path


# --------------------------------------------------------------------------------------------
# Threshold selection -- on the VALIDATION seed, never the scoring seed
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Thresholds:
    """Detector knobs, chosen on the validation city and then frozen."""

    validation_seed: int
    stuck_tolerance_cm: float
    disagreement_cm: float
    contamination: float
    rule_f1: float
    forest_f1: float
    #: False when the Isolation Forest was skipped, so its row reads "not evaluated" rather
    #: than a misleading 0.0000.
    forest_evaluated: bool


def _f1(precision: float, recall: float) -> float:
    return 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)


def select_thresholds(cfg: Config, scoring_seed: int, *, include_ml: bool = True) -> Thresholds:
    """Pick thresholds by F1 on the VALIDATION seed's planted labels.

    This is the one place planted labels legitimately steer a knob, and it happens on a
    **different city** from the one any reported number comes from. Reading the scoring seed's
    labels here would make every headline figure optimistic.
    """
    validation_seed = scoring_seed + cfg.anomaly.validation_seed_offset
    dataset = build_dataset(cfg, validation_seed)

    best_rule = (0.0, cfg.anomaly.stuck_tolerance_cm, cfg.anomaly.disagreement_cm)
    for tolerance in RULE_TOLERANCE_GRID:
        for disagreement in RULE_DISAGREEMENT_GRID:
            detector = RuleDetector(tolerance, disagreement)
            flags = detector.flag_all(dataset.windows)
            scores = detector.score_all(dataset.windows)
            result = score_detector(dataset.labels, scores, flags)
            f1 = _f1(result.precision, result.recall)
            if f1 > best_rule[0]:
                best_rule = (f1, tolerance, disagreement)

    best_forest = (0.0, cfg.anomaly.contamination)
    if include_ml:
        for contamination in CONTAMINATION_GRID:
            forest = build_detector(
                "isolation_forest", _with_contamination(cfg, contamination), validation_seed
            )
            forest.fit(dataset.windows)
            flags = forest.flag_all(dataset.windows)
            scores = forest.score_all(dataset.windows)
            result = score_detector(dataset.labels, scores, flags)
            f1 = _f1(result.precision, result.recall)
            if f1 > best_forest[0]:
                best_forest = (f1, contamination)

    return Thresholds(
        validation_seed=validation_seed,
        stuck_tolerance_cm=best_rule[1],
        disagreement_cm=best_rule[2],
        contamination=best_forest[1],
        rule_f1=best_rule[0],
        forest_f1=best_forest[0],
        forest_evaluated=include_ml,
    )


def _with_contamination(cfg: Config, contamination: float) -> AnomalyConfig:
    """A copy of the anomaly config with one knob replaced.

    The config is frozen, so :func:`dataclasses.replace` makes a new one rather than mutating --
    mutating would leak a tuning value into every later call.
    """
    return replace(cfg.anomaly, contamination=contamination)


# --------------------------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class DetectorScore:
    """One detector's result on one family (or overall)."""

    seed: int
    detector_key: str
    detector_label: str
    family: str
    fairness: str
    score: ClassificationScore


OVERALL = "overall"

CSV_FIELDS = (
    "seed",
    "detector_key",
    "detector_label",
    "family",
    "fairness",
    "n",
    "n_positive",
    "prevalence",
    "precision",
    "recall",
    "pr_auc",
    "true_positives",
    "false_positives",
    "false_negatives",
)


def score_detectors(
    cfg: Config,
    dataset: AnomalyDataset,
    seed: int,
    thresholds: Thresholds,
    *,
    keys: Sequence[str] = DETECTOR_KEYS,
) -> list[DetectorScore]:
    """Score each detector overall and per fault family, using the frozen thresholds.

    This is the **only** function that reads ``dataset.labels``, and it does so after every
    threshold is fixed.
    """
    acfg = _with_contamination(cfg, thresholds.contamination)
    results: list[DetectorScore] = []

    for key in keys:
        detector: Detector
        if key == "rule":
            # Built directly rather than via build_detector, because the thresholds here come
            # from the validation city rather than from the config.
            detector = RuleDetector(thresholds.stuck_tolerance_cm, thresholds.disagreement_cm)
        else:
            detector = build_detector(key, acfg, seed)
        detector.fit(dataset.windows)

        # Batched: a per-window call into scikit-learn would cost minutes on this dataset.
        scores = detector.score_all(dataset.windows)
        flags = detector.flag_all(dataset.windows)

        results.append(
            DetectorScore(
                seed=seed,
                detector_key=detector.key,
                detector_label=detector.label,
                family=OVERALL,
                fairness="all families pooled",
                score=score_detector(dataset.labels, scores, flags),
            )
        )

        for family in FAULT_FAMILIES:
            # Per family: that family's windows as positives, plus every healthy window as the
            # negatives. Including the healthy windows is what keeps precision meaningful --
            # scoring a family against only its own windows would make precision always 1.0.
            keep = [
                index
                for index, window_family in enumerate(dataset.families)
                if window_family in (family, "")
            ]
            results.append(
                DetectorScore(
                    seed=seed,
                    detector_key=detector.key,
                    detector_label=detector.label,
                    family=family,
                    fairness=FAMILY_FAIRNESS[family],
                    score=score_detector(
                        [dataset.labels[index] for index in keep],
                        [scores[index] for index in keep],
                        [flags[index] for index in keep],
                    ),
                )
            )
    return results


# --------------------------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------------------------


def _fmt(value: float, places: int = 4) -> str:
    return f"{value:.{places}f}"


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def render_scores(results: Sequence[DetectorScore]) -> str:
    """Precision / recall / PR-AUC per detector per family, with prevalence beside each."""
    rows = [
        [
            r.detector_label,
            r.family,
            r.fairness,
            str(r.score.n),
            str(r.score.n_positive),
            _fmt(r.score.prevalence),
            _fmt(r.score.precision),
            _fmt(r.score.recall),
            _fmt(r.score.average_precision),
        ]
        for r in results
    ]
    header = (
        "detector",
        "fault family",
        "how to read it",
        "n windows",
        "n faulty",
        "prevalence",
        "precision",
        "recall",
        "PR-AUC",
    )
    # Computed from the real pooled rows rather than asserted, so the sentence cannot drift
    # away from the dataset it is describing.
    pooled = next((r for r in results if r.family == OVERALL), None)
    trivial_accuracy = (1.0 - pooled.score.prevalence) * 100.0 if pooled else 0.0

    title = "### Anomaly detection: rule baseline vs Isolation Forest, on planted faults"
    note = (
        "**Accuracy is deliberately absent.** At this prevalence a detector that never flags\n"
        f"anything would score {trivial_accuracy:.1f}% accuracy while catching nothing, so "
        "precision, recall and\n"
        "PR-AUC are reported instead -- each beside the prevalence, which is the number PR-AUC\n"
        "must be judged against (a useless ranker scores about the prevalence, not 0.5).\n\n"
        "`stuck` and `dropout` are marked **favors the rule baseline by construction**: the\n"
        "rule's zero-variance test is very nearly a definition of those two faults, so a win\n"
        "there is a demonstration, not a measurement. `drift` and `noise_burst` are fair tests."
    )
    return f"{title}\n\n{note}\n\n{_table(header, rows)}"


def render_thresholds(thresholds: Thresholds, scoring_seed: int) -> str:
    """Show the threshold/seed separation in the output itself, not just in prose."""
    rows = [
        ["rule: stuck tolerance (cm)", _fmt(thresholds.stuck_tolerance_cm, 3)],
        ["rule: disagreement limit (cm)", _fmt(thresholds.disagreement_cm, 3)],
        [
            "Isolation Forest: contamination",
            _fmt(thresholds.contamination, 3)
            if thresholds.forest_evaluated
            else f"{_fmt(thresholds.contamination, 3)} (config default; forest not evaluated)",
        ],
        ["best rule F1 on the validation city", _fmt(thresholds.rule_f1)],
        [
            "best forest F1 on the validation city",
            _fmt(thresholds.forest_f1) if thresholds.forest_evaluated else "not evaluated",
        ],
        ["validation seed (thresholds chosen here)", str(thresholds.validation_seed)],
        ["scoring seed (every reported number)", str(scoring_seed)],
    ]
    title = "### Thresholds, and the seed separation that chose them"
    note = (
        "Every knob was chosen by F1 on the **validation** city's planted labels and then applied\n"
        "unchanged to the **scoring** city. The scoring city's labels are read only to compute\n"
        "the table above, after these values were frozen. Choosing thresholds on the scoring\n"
        "labels would make every number in this document optimistic."
    )
    return f"{title}\n\n{note}\n\n{_table(('knob', 'value'), rows)}"


def verdict_lines(results: Sequence[DetectorScore]) -> list[str]:
    """State plainly where the simple rule matches or beats the ML detector."""
    by_key: dict[tuple[str, str], DetectorScore] = {(r.detector_key, r.family): r for r in results}
    lines: list[str] = []
    for family in (OVERALL, *FAULT_FAMILIES):
        rule = by_key.get(("rule", family))
        forest = by_key.get(("isolation_forest", family))
        if rule is None or forest is None:
            continue
        fairness = rule.fairness
        rule_auc = rule.score.average_precision
        forest_auc = forest.score.average_precision
        if rule_auc > forest_auc:
            verdict = (
                f"the RULE BASELINE beats the Isolation Forest on PR-AUC "
                f"({_fmt(rule_auc)} vs {_fmt(forest_auc)})"
            )
        elif rule_auc < forest_auc:
            verdict = (
                f"the Isolation Forest beats the rule baseline on PR-AUC "
                f"({_fmt(forest_auc)} vs {_fmt(rule_auc)})"
            )
        else:
            verdict = f"the two detectors TIE on PR-AUC ({_fmt(rule_auc)} each)"
        lines.append(
            f"  {family} [{fairness}]: {verdict}; prevalence {_fmt(rule.score.prevalence)}"
        )
    return lines


def csv_rows(results: Sequence[DetectorScore]) -> list[dict[str, object]]:
    """Flatten for ``csv.DictWriter`` using :data:`CSV_FIELDS`."""
    return [
        {
            "seed": r.seed,
            "detector_key": r.detector_key,
            "detector_label": r.detector_label,
            "family": r.family,
            "fairness": r.fairness,
            "n": r.score.n,
            "n_positive": r.score.n_positive,
            "prevalence": _fmt(r.score.prevalence),
            "precision": _fmt(r.score.precision),
            "recall": _fmt(r.score.recall),
            "pr_auc": _fmt(r.score.average_precision),
            "true_positives": r.score.true_positives,
            "false_positives": r.score.false_positives,
            "false_negatives": r.score.false_negatives,
        }
        for r in results
    ]
