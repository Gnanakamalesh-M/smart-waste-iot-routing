"""Sensor anomaly detection: the leakage firewall, the rule baseline, and the seed separation.

The tests that matter most here are the leakage ones. A detector that peeked at the planted
labels would post excellent numbers, so the firewall is asserted structurally -- against the
module's own source and its function signatures -- rather than trusted to review.

Marker discipline: everything using the Isolation Forest carries **ml**; the rule baseline, the
generator, the window labelling and the metrics are stdlib and run in **unit** / **sim**.
"""

import inspect
from pathlib import Path

import pytest

from smartwaste import load_config
from smartwaste.anomaly_eval import (
    CSV_FIELDS,
    LABELS_FILENAME,
    OVERALL,
    READINGS_FILENAME,
    build_dataset,
    csv_rows,
    plant_faults,
    render_scores,
    render_thresholds,
    score_detectors,
    select_thresholds,
    verdict_lines,
    write_dataset,
)
from smartwaste.config import Config, ConfigError
from smartwaste.ml import anomaly as anomaly_module
from smartwaste.ml.anomaly import (
    DETECTOR_KEYS,
    FAMILY_FAIRNESS,
    FAULT_DROPOUT,
    FAULT_FAMILIES,
    FAULT_STUCK,
    Detector,
    RuleDetector,
    SensorWindow,
    build_detector,
    max_disagreement,
    min_sensor_variation,
    window_features,
)

SEED = 42

#: Small enough to be fast, big enough to contain every fault family.
TINY = {"n_bins": 60, "days": 30, "window_days": 5, "max_fault_days": 12}


@pytest.fixture
def cfg() -> Config:
    return load_config(overrides={"anomaly": TINY})


def _window(series: list[list[float]], weights: list[float] | None = None) -> SensorWindow:
    return SensorWindow(
        bin_index=0,
        end_day=len(series) - 1,
        distances=tuple(tuple(row) for row in series),
        weights=tuple(weights or [1.0] * len(series)),
    )


# --------------------------------------------------------------------------------------------
# The leakage firewall. These are the most important tests in this file.
# --------------------------------------------------------------------------------------------


@pytest.mark.unit
def test_the_detector_module_never_names_the_labels_file() -> None:
    """A detector must not be able to open the planted truth, even by accident."""
    source = inspect.getsource(anomaly_module)
    assert LABELS_FILENAME not in source
    assert "anomaly_labels" not in source
    assert "fault_family" not in source


@pytest.mark.unit
def test_the_detector_module_never_references_ground_truth() -> None:
    source = inspect.getsource(anomaly_module)
    assert "truth_state" not in source
    assert "true_fill" not in source
    assert "PlantedFault" not in source


@pytest.mark.unit
def test_no_detector_method_can_accept_a_label() -> None:
    """Structural, not conventional: there is no parameter a label could arrive through."""
    for method_name in ("fit", "score", "flag"):
        signature = inspect.signature(getattr(Detector, method_name))
        names = set(signature.parameters) - {"self"}
        for forbidden in ("label", "labels", "y", "truth", "target", "families"):
            assert forbidden not in names, (
                f"Detector.{method_name} accepts {forbidden!r}, which would let the planted "
                f"truth reach a detector"
            )


@pytest.mark.unit
def test_the_sensor_window_carries_no_label() -> None:
    fields = set(SensorWindow.__dataclass_fields__)
    assert fields == {"bin_index", "end_day", "distances", "weights"}


@pytest.mark.unit
def test_window_features_depend_only_on_the_readings() -> None:
    """Two windows with identical readings must produce identical features.

    The windows differ in ``bin_index`` and ``end_day``, which are bookkeeping. If either leaked
    into the feature vector, a detector could learn *which bins* are faulty from the dataset's
    layout instead of from their readings.
    """
    series = [[10.0, 11.0, 12.0], [13.0, 14.0, 15.0], [16.0, 17.0, 18.0]]
    first = SensorWindow(0, 2, tuple(tuple(r) for r in series), (1.0, 2.0, 3.0))
    second = SensorWindow(99, 77, tuple(tuple(r) for r in series), (1.0, 2.0, 3.0))
    assert window_features(first) == window_features(second)


@pytest.mark.sim
def test_readings_and_labels_go_to_separate_files(cfg: Config, tmp_path: Path) -> None:
    dataset = build_dataset(cfg, SEED)
    readings_path, labels_path = write_dataset(dataset, tmp_path)

    assert readings_path.name == READINGS_FILENAME
    assert labels_path.name == LABELS_FILENAME
    assert readings_path != labels_path

    header = readings_path.read_text(encoding="utf-8").splitlines()[0]
    # No column in the readings file may hint at the truth.
    assert "fault" not in header
    assert "label" not in header
    assert header == "bin,day,d1_cm,d2_cm,d3_cm,weight_kg"

    labels_header = labels_path.read_text(encoding="utf-8").splitlines()[0]
    assert labels_header == "bin,day,fault_family"


@pytest.mark.sim
def test_the_labels_file_lists_exactly_the_planted_fault_days(cfg: Config, tmp_path: Path) -> None:
    dataset = build_dataset(cfg, SEED)
    _, labels_path = write_dataset(dataset, tmp_path)
    rows = labels_path.read_text(encoding="utf-8").splitlines()[1:]
    expected = sum(fault.end_day - fault.start_day + 1 for fault in dataset.faults)
    assert len(rows) == expected


# --------------------------------------------------------------------------------------------
# The planted faults exist and are what they claim to be
# --------------------------------------------------------------------------------------------


@pytest.mark.sim
def test_every_fault_family_is_planted(cfg: Config) -> None:
    """A family with no instances would make its results row meaningless."""
    families = {fault.family for fault in plant_faults(cfg, SEED)}
    assert families == set(FAULT_FAMILIES)


@pytest.mark.sim
def test_the_planted_fault_rate_follows_the_config(cfg: Config) -> None:
    faults = plant_faults(cfg, SEED)
    share = len(faults) / cfg.anomaly.n_bins
    assert share == pytest.approx(cfg.anomaly.fault_bin_fraction, abs=0.1)


@pytest.mark.sim
def test_changing_the_fault_rate_does_not_reshuffle_the_other_draws(cfg: Config) -> None:
    """The same controlled-comparison discipline the city simulation uses."""
    low = load_config(overrides={"anomaly": {**TINY, "fault_bin_fraction": 0.1}})
    high = load_config(overrides={"anomaly": {**TINY, "fault_bin_fraction": 0.9}})
    low_faults = {f.bin_index: (f.family, f.sensor, f.start_day) for f in plant_faults(low, SEED)}
    high_faults = {f.bin_index: (f.family, f.sensor, f.start_day) for f in plant_faults(high, SEED)}
    shared = set(low_faults) & set(high_faults)
    assert shared
    for bin_index in shared:
        assert low_faults[bin_index] == high_faults[bin_index]


@pytest.mark.sim
def test_faults_stay_inside_the_horizon(cfg: Config) -> None:
    for fault in plant_faults(cfg, SEED):
        assert 0 <= fault.start_day <= fault.end_day < cfg.anomaly.days
        assert 0 <= fault.sensor < cfg.sensors.n_ultrasonic


@pytest.mark.sim
def test_a_stuck_sensor_really_stops_moving(cfg: Config) -> None:
    """The generator must actually produce the fault it labelled."""
    dataset = build_dataset(cfg, SEED)
    stuck = next(f for f in dataset.faults if f.family == FAULT_STUCK)
    series = [
        dataset.distances[stuck.bin_index][day][stuck.sensor]
        for day in range(stuck.start_day, stuck.end_day + 1)
    ]
    assert len(set(series)) == 1, "a stuck sensor's readings changed"


@pytest.mark.sim
def test_a_dropout_sensor_reads_full_range(cfg: Config) -> None:
    dataset = build_dataset(cfg, SEED)
    dropout = next(f for f in dataset.faults if f.family == FAULT_DROPOUT)
    for day in range(dropout.start_day, dropout.end_day + 1):
        assert dataset.distances[dropout.bin_index][day][dropout.sensor] == pytest.approx(
            cfg.bin.height_cm
        )


@pytest.mark.sim
def test_readings_stay_within_sensor_range(cfg: Config) -> None:
    dataset = build_dataset(cfg, SEED)
    for bin_distances in dataset.distances:
        for row in bin_distances:
            for value in row:
                assert cfg.sensors.min_distance_cm <= value <= cfg.bin.height_cm


@pytest.mark.sim
def test_windows_are_labelled_by_their_last_day(cfg: Config) -> None:
    """The documented labelling convention, pinned."""
    dataset = build_dataset(cfg, SEED)
    fault_by_bin = {f.bin_index: f for f in dataset.faults}
    for window, label in zip(dataset.windows, dataset.labels, strict=True):
        fault = fault_by_bin.get(window.bin_index)
        expected = fault is not None and fault.covers(window.end_day)
        assert label is expected


@pytest.mark.sim
def test_the_dataset_is_imbalanced_which_is_why_accuracy_is_banned(cfg: Config) -> None:
    dataset = build_dataset(cfg, SEED)
    assert 0.0 < dataset.prevalence < 0.5


@pytest.mark.sim
def test_the_dataset_is_deterministic_for_one_seed(cfg: Config) -> None:
    first = build_dataset(cfg, SEED)
    second = build_dataset(cfg, SEED)
    assert first.distances == second.distances
    assert first.labels == second.labels
    assert first.faults == second.faults


# --------------------------------------------------------------------------------------------
# The rule baseline, as a definition
# --------------------------------------------------------------------------------------------


@pytest.mark.unit
def test_min_sensor_variation_is_zero_for_a_frozen_sensor() -> None:
    frozen = _window([[50.0, 10.0, 20.0], [50.0, 15.0, 25.0], [50.0, 20.0, 30.0]])
    assert min_sensor_variation(frozen) == pytest.approx(0.0)


@pytest.mark.unit
def test_min_sensor_variation_is_positive_when_everything_moves() -> None:
    lively = _window([[10.0, 11.0, 12.0], [20.0, 21.0, 22.0], [30.0, 31.0, 32.0]])
    assert min_sensor_variation(lively) > 0.0


@pytest.mark.unit
def test_max_disagreement_measures_the_worst_day() -> None:
    window = _window([[10.0, 10.0, 10.0], [10.0, 10.0, 60.0]])
    assert max_disagreement(window) == pytest.approx(50.0)


@pytest.mark.unit
def test_the_rule_flags_a_frozen_sensor() -> None:
    detector = RuleDetector(stuck_tolerance_cm=0.05, disagreement_cm=25.0)
    frozen = _window([[50.0, 10.0, 20.0], [50.0, 15.0, 25.0], [50.0, 20.0, 30.0]])
    assert detector.flag(frozen)


@pytest.mark.unit
def test_the_rule_flags_wild_disagreement() -> None:
    detector = RuleDetector(stuck_tolerance_cm=0.05, disagreement_cm=25.0)
    arguing = _window([[10.0, 11.0, 90.0], [12.0, 13.0, 95.0], [14.0, 15.0, 99.0]])
    assert detector.flag(arguing)


@pytest.mark.unit
def test_the_rule_leaves_a_healthy_window_alone() -> None:
    detector = RuleDetector(stuck_tolerance_cm=0.05, disagreement_cm=25.0)
    healthy = _window([[60.0, 61.0, 62.0], [55.0, 56.0, 57.0], [50.0, 51.0, 52.0]])
    assert not detector.flag(healthy)


@pytest.mark.unit
def test_the_rule_score_is_higher_for_a_worse_window() -> None:
    """PR-AUC needs a ranking, so the rule emits a continuous score, not just a flag."""
    detector = RuleDetector(stuck_tolerance_cm=0.05, disagreement_cm=25.0)
    frozen = _window([[50.0, 10.0, 20.0], [50.0, 15.0, 25.0], [50.0, 20.0, 30.0]])
    healthy = _window([[60.0, 61.0, 62.0], [55.0, 56.0, 57.0], [50.0, 51.0, 52.0]])
    assert detector.score(frozen) > detector.score(healthy)


@pytest.mark.unit
def test_the_rule_needs_no_ml_extra() -> None:
    cfg = load_config()
    assert build_detector("rule", cfg.anomaly, SEED).needs_ml_extra is False
    assert build_detector("isolation_forest", cfg.anomaly, SEED).needs_ml_extra is True


@pytest.mark.unit
def test_unknown_detector_key_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown detector key"):
        build_detector("autoencoder", load_config().anomaly, SEED)


@pytest.mark.unit
def test_every_family_carries_a_fairness_label() -> None:
    """A built-in advantage must never be reported as if it were evidence."""
    assert set(FAMILY_FAIRNESS) == set(FAULT_FAMILIES)
    assert "construction" in FAMILY_FAIRNESS[FAULT_STUCK]
    assert "construction" in FAMILY_FAIRNESS[FAULT_DROPOUT]
    assert FAMILY_FAIRNESS["drift"] == "fair test"
    assert FAMILY_FAIRNESS["noise_burst"] == "fair test"


# --------------------------------------------------------------------------------------------
# Threshold selection happens on a different seed
# --------------------------------------------------------------------------------------------


@pytest.mark.sim
def test_thresholds_are_chosen_on_a_different_seed(cfg: Config) -> None:
    """The scoring seed's labels must never pick a knob."""
    thresholds = select_thresholds(cfg, SEED, include_ml=False)
    assert thresholds.validation_seed == SEED + cfg.anomaly.validation_seed_offset
    assert thresholds.validation_seed != SEED


@pytest.mark.sim
def test_a_zero_validation_offset_is_rejected() -> None:
    """Tuning on the scoring city is a config error, not a judgement call."""
    with pytest.raises(ConfigError, match="validation_seed_offset must be non-zero"):
        load_config(overrides={"anomaly": {**TINY, "validation_seed_offset": 0}})


@pytest.mark.sim
def test_threshold_selection_is_deterministic(cfg: Config) -> None:
    first = select_thresholds(cfg, SEED, include_ml=False)
    second = select_thresholds(cfg, SEED, include_ml=False)
    assert first == second


@pytest.mark.sim
def test_the_threshold_table_shows_the_seed_separation(cfg: Config) -> None:
    """The separation must be visible in the output, not only in prose."""
    thresholds = select_thresholds(cfg, SEED, include_ml=False)
    rendered = render_thresholds(thresholds, SEED)
    assert str(thresholds.validation_seed) in rendered
    assert "validation seed" in rendered
    assert "scoring seed" in rendered
    assert "not evaluated" in rendered  # the forest was skipped here


# --------------------------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------------------------


@pytest.mark.sim
def test_the_rule_is_scored_per_family_and_overall(cfg: Config) -> None:
    thresholds = select_thresholds(cfg, SEED, include_ml=False)
    results = score_detectors(cfg, build_dataset(cfg, SEED), SEED, thresholds, keys=("rule",))
    families = [r.family for r in results]
    assert families == [OVERALL, *FAULT_FAMILIES]
    for r in results:
        assert 0.0 <= r.score.precision <= 1.0
        assert 0.0 <= r.score.recall <= 1.0
        assert 0.0 <= r.score.average_precision <= 1.0
        assert r.score.n_positive > 0


@pytest.mark.sim
def test_a_per_family_row_keeps_the_healthy_windows_as_negatives(cfg: Config) -> None:
    """Scoring a family against only its own windows would make precision always 1.0."""
    dataset = build_dataset(cfg, SEED)
    thresholds = select_thresholds(cfg, SEED, include_ml=False)
    results = score_detectors(cfg, dataset, SEED, thresholds, keys=("rule",))
    for r in results:
        if r.family == OVERALL:
            continue
        # Far more windows than just the faulty ones for that family.
        assert r.score.n > r.score.n_positive * 10


@pytest.mark.sim
def test_the_rule_beats_chance_on_the_families_it_targets(cfg: Config) -> None:
    """A floor, and labelled as favoured by construction rather than claimed as a win.

    PR-AUC must beat the prevalence, which is what a useless ranker scores.
    """
    thresholds = select_thresholds(cfg, SEED, include_ml=False)
    results = score_detectors(cfg, build_dataset(cfg, SEED), SEED, thresholds, keys=("rule",))
    by_family = {r.family: r for r in results}
    for family in (FAULT_STUCK, FAULT_DROPOUT):
        row = by_family[family]
        assert row.score.average_precision > row.score.prevalence


@pytest.mark.sim
def test_accuracy_is_never_reported(cfg: Config) -> None:
    """CLAUDE.md forbids it on imbalanced problems."""
    thresholds = select_thresholds(cfg, SEED, include_ml=False)
    results = score_detectors(cfg, build_dataset(cfg, SEED), SEED, thresholds, keys=("rule",))
    rendered = render_scores(results)

    # The note explains *why* accuracy is omitted, so the word appears in the prose. What must
    # not exist is an accuracy COLUMN or an accuracy field -- i.e. a reported accuracy number.
    header = next(line for line in rendered.splitlines() if line.startswith("| detector"))
    columns = {cell.strip().lower() for cell in header.strip("|").split("|")}
    assert "accuracy" not in columns
    assert {"precision", "recall", "pr-auc", "prevalence"} <= columns

    assert "accuracy" not in CSV_FIELDS
    assert set(csv_rows(results)[0]) == set(CSV_FIELDS)


@pytest.mark.sim
def test_the_trivial_accuracy_note_is_computed_from_the_real_rows(cfg: Config) -> None:
    """The sentence about accuracy must match the dataset, not a hardcoded guess."""
    thresholds = select_thresholds(cfg, SEED, include_ml=False)
    dataset = build_dataset(cfg, SEED)
    results = score_detectors(cfg, dataset, SEED, thresholds, keys=("rule",))
    expected = f"{(1.0 - dataset.prevalence) * 100.0:.1f}%"
    assert expected in render_scores(results)


@pytest.mark.sim
def test_the_results_table_labels_the_favoured_families(cfg: Config) -> None:
    thresholds = select_thresholds(cfg, SEED, include_ml=False)
    results = score_detectors(cfg, build_dataset(cfg, SEED), SEED, thresholds, keys=("rule",))
    rendered = render_scores(results)
    assert "favors the rule baseline by construction" in rendered
    assert "fair test" in rendered


# --------------------------------------------------------------------------------------------
# ml chunk: the Isolation Forest
# --------------------------------------------------------------------------------------------


@pytest.mark.ml
def test_both_detectors_are_scored_on_the_same_windows(cfg: Config) -> None:
    thresholds = select_thresholds(cfg, SEED)
    results = score_detectors(cfg, build_dataset(cfg, SEED), SEED, thresholds)
    keys = {r.detector_key for r in results}
    assert keys == set(DETECTOR_KEYS)
    overall = [r for r in results if r.family == OVERALL]
    assert len({r.score.n for r in overall}) == 1
    assert len({r.score.n_positive for r in overall}) == 1


@pytest.mark.ml
def test_the_forest_produces_a_usable_ranking(cfg: Config) -> None:
    """A property, not an exact score: scikit-learn is not bit-reproducible across machines."""
    thresholds = select_thresholds(cfg, SEED)
    results = score_detectors(
        cfg, build_dataset(cfg, SEED), SEED, thresholds, keys=("isolation_forest",)
    )
    overall = next(r for r in results if r.family == OVERALL)
    assert 0.0 <= overall.score.average_precision <= 1.0
    assert overall.score.true_positives + overall.score.false_positives > 0


@pytest.mark.ml
def test_the_forest_is_reproducible_on_this_machine(cfg: Config) -> None:
    dataset = build_dataset(cfg, SEED)
    thresholds = select_thresholds(cfg, SEED)
    first = score_detectors(cfg, dataset, SEED, thresholds, keys=("isolation_forest",))
    second = score_detectors(cfg, dataset, SEED, thresholds, keys=("isolation_forest",))
    for a, b in zip(first, second, strict=True):
        assert a.score == b.score


@pytest.mark.ml
def test_the_forest_refuses_to_score_before_being_fitted(cfg: Config) -> None:
    detector = build_detector("isolation_forest", cfg.anomaly, SEED)
    window = _window([[10.0, 11.0, 12.0], [13.0, 14.0, 15.0]])
    with pytest.raises(RuntimeError, match="before fit"):
        detector.score(window)


@pytest.mark.ml
def test_the_verdict_names_both_detectors_and_the_fairness_label(cfg: Config) -> None:
    thresholds = select_thresholds(cfg, SEED)
    results = score_detectors(cfg, build_dataset(cfg, SEED), SEED, thresholds)
    lines = verdict_lines(results)
    assert len(lines) == 1 + len(FAULT_FAMILIES)
    assert any("construction" in line for line in lines)
    assert all("prevalence" in line for line in lines)


# --------------------------------------------------------------------------------------------
# Config validation
# --------------------------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"n_bins": 0}, "anomaly.n_bins must be >= 1"),
        ({"days": 0}, "anomaly.days must be >= 1"),
        ({"fault_bin_fraction": 1.5}, "fault_bin_fraction must be in"),
        ({"window_days": 1}, "window_days must be >= 2"),
        ({"window_days": 999}, "window_days must be <= anomaly.days"),
        ({"validation_seed_offset": 0}, "validation_seed_offset must be non-zero"),
        ({"min_fault_days": 0}, "min_fault_days <= max_fault_days"),
        ({"max_fault_days": 999}, "max_fault_days must be <= anomaly.days"),
        ({"drift_cm_per_day": -1.0}, "drift_cm_per_day must be >= 0"),
        ({"noise_burst_sigma_cm": -1.0}, "noise_burst_sigma_cm must be >= 0"),
        ({"contamination": 0.0}, "contamination must be in"),
        ({"contamination": 0.6}, "contamination must be in"),
        ({"n_estimators": 0}, "n_estimators must be >= 1"),
        ({"stuck_tolerance_cm": -1.0}, "stuck_tolerance_cm must be >= 0"),
        ({"disagreement_cm": -1.0}, "disagreement_cm must be >= 0"),
    ],
)
def test_invalid_anomaly_config_is_rejected(override: dict[str, object], message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        load_config(overrides={"anomaly": override})


@pytest.mark.unit
def test_anomaly_config_defaults_are_the_documented_ones() -> None:
    """Tripwire against silently retuning a published constant."""
    anomaly = load_config().anomaly
    assert anomaly.n_bins == 400
    assert anomaly.days == 60
    assert anomaly.fault_bin_fraction == 0.15
    assert anomaly.window_days == 5
    assert anomaly.validation_seed_offset == 1
