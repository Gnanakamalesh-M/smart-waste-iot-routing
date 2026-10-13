"""Forecasting: the naive baselines, the leakage firewall, and the time-based split.

Split across markers on purpose:

- **unit / sim**: features, the three naive baselines, the no-leakage assertions and the split.
  All stdlib, so these run with the optional ``ml`` extra uninstalled.
- **ml**: anything that fits the gradient-boosting model, i.e. anything importing scikit-learn.

Nothing here pins an exact float produced by a model. scikit-learn is not bit-reproducible
across operating systems, BLAS builds or library versions, so the ML tests assert **properties**
instead -- see ``docs/PHASE3_RESULTS.md``.
"""

import inspect

import pytest

from smartwaste import load_config
from smartwaste.config import Config, ConfigError
from smartwaste.demand import SCENARIO_HETEROGENEOUS, SCENARIO_SMOOTH
from smartwaste.forecast_eval import build_dataset, score_forecasters
from smartwaste.ml import forecast as forecast_module
from smartwaste.ml.forecast import (
    FORECASTER_KEYS,
    BinHistory,
    LastValueForecaster,
    MovingAverageForecaster,
    SeasonalNaiveForecaster,
    build_features,
    build_forecaster,
    estimate_fill,
)
from smartwaste.noise import SensorReading

SEED = 42

#: Tiny on purpose: these tests check behaviour, not statistics.
TINY = {"n_bins": 25, "days": 22}


@pytest.fixture
def cfg() -> Config:
    return load_config()


def _history(values: list[float], start_day: int = 0) -> BinHistory:
    history = BinHistory()
    for offset, value in enumerate(values):
        history.observe(start_day + offset, value, value * 0.5)
    return history


# --------------------------------------------------------------------------------------------
# No leakage. These are the tests that matter most in this file.
# --------------------------------------------------------------------------------------------


@pytest.mark.unit
def test_forecast_module_never_references_ground_truth() -> None:
    """Structural guarantee, asserted against the module's own source.

    ``truth_state`` is the only function in the project that reads the true fill and true
    weight. A forecaster that called it would be cheating, so its absence is pinned here rather
    than trusted to review.
    """
    # The module's prose explains *why* it must not read ground truth, so the docstrings do
    # mention these names. What must be absent is any import or call of them.
    assert not hasattr(forecast_module, "truth_state")
    source = inspect.getsource(forecast_module)
    assert "truth_state(" not in source
    assert "import truth_state" not in source
    assert "true_fill" not in source
    assert "true_weight" not in source


@pytest.mark.unit
def test_sensor_reading_carries_no_ground_truth() -> None:
    """Why the guarantee above holds by construction and not by convention."""
    fields = set(SensorReading.__dataclass_fields__)
    assert fields == {"distances_cm", "weight_kg"}


@pytest.mark.unit
def test_features_are_finite_and_have_the_documented_length(cfg: Config) -> None:
    from smartwaste.ml.forecast import feature_names

    history = _history([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    row = build_features(
        history,
        day=5,
        area_type=0,
        weekday=3,
        is_festival=False,
        cfg=cfg.forecast,
    )
    assert row is not None
    assert len(row) == len(feature_names(cfg.forecast))
    assert all(isinstance(value, float) for value in row)


@pytest.mark.unit
def test_features_are_none_until_enough_history_exists(cfg: Config) -> None:
    short = _history([10.0, 20.0])
    assert (
        build_features(short, day=1, area_type=0, weekday=0, is_festival=False, cfg=cfg.forecast)
        is None
    )


@pytest.mark.unit
def test_estimate_fill_uses_only_the_reading(cfg: Config) -> None:
    """A full bin reads short distances; the estimate must rise accordingly."""
    empty = estimate_fill(SensorReading((100.0, 100.0, 100.0), 0.0), cfg.bin)
    full = estimate_fill(SensorReading((10.0, 10.0, 10.0), 0.0), cfg.bin)
    assert empty < full
    assert 0.0 <= empty <= 100.0
    assert 0.0 <= full <= 100.0


# --------------------------------------------------------------------------------------------
# The naive baselines, as definitions
# --------------------------------------------------------------------------------------------


@pytest.mark.unit
def test_last_value_returns_the_most_recent_rise() -> None:
    history = _history([10.0, 20.0, 26.0])
    assert LastValueForecaster().predict_rate(history, None) == pytest.approx(6.0)


@pytest.mark.unit
def test_moving_average_averages_the_window() -> None:
    history = _history([0.0, 10.0, 20.0, 23.0])
    # rises are 10, 10, 3 -> a window of 3 averages to 23/3
    assert MovingAverageForecaster(3).predict_rate(history, None) == pytest.approx(23 / 3)
    # a window of 1 is the last value
    assert MovingAverageForecaster(1).predict_rate(history, None) == pytest.approx(3.0)


@pytest.mark.unit
def test_seasonal_naive_reaches_back_one_week() -> None:
    # Eight observations give seven rises; the 7-back rise is the first one.
    history = _history([0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 99.0])
    assert SeasonalNaiveForecaster().predict_rate(history, None) == pytest.approx(5.0)


@pytest.mark.unit
def test_seasonal_naive_falls_back_to_the_mean_rise_when_short() -> None:
    history = _history([0.0, 4.0, 8.0])
    assert SeasonalNaiveForecaster().predict_rate(history, None) == pytest.approx(4.0)


@pytest.mark.unit
def test_baselines_handle_an_empty_history() -> None:
    """Day 0 has no rise to average, and must not raise."""
    empty = BinHistory()
    for key in ("last_value", "moving_average", "seasonal_naive"):
        forecaster = build_forecaster(key, load_config().forecast, SEED)
        assert forecaster.predict_rate(empty, None) == 0.0


@pytest.mark.unit
def test_collection_resets_are_excluded_from_the_rises() -> None:
    """A bin emptied from 90% to 5% must not read as demand of -85 points."""
    history = _history([10.0, 50.0, 90.0, 5.0, 45.0])
    assert all(rise >= 0.0 for rise in history.positive_rises())
    assert -85.0 not in history.rises() or min(history.rises()) < 0  # the raw series does dip


@pytest.mark.unit
def test_baselines_need_no_ml_extra() -> None:
    """The three naive forecasters must not require scikit-learn."""
    for key in ("last_value", "moving_average", "seasonal_naive"):
        assert build_forecaster(key, load_config().forecast, SEED).needs_ml_extra is False
    assert build_forecaster("gbm", load_config().forecast, SEED).needs_ml_extra is True


@pytest.mark.unit
def test_unknown_forecaster_key_is_rejected(cfg: Config) -> None:
    with pytest.raises(ValueError, match="unknown forecaster key"):
        build_forecaster("deep_learning", cfg.forecast, SEED)


@pytest.mark.unit
def test_baselines_are_listed_before_the_ml_model() -> None:
    """CLAUDE.md wants baselines reported first; the ordering is the mechanism."""
    assert FORECASTER_KEYS.index("gbm") == len(FORECASTER_KEYS) - 1


# --------------------------------------------------------------------------------------------
# The time-based split
# --------------------------------------------------------------------------------------------


@pytest.mark.sim
def test_the_split_is_strictly_by_time(cfg: Config) -> None:
    """Every training row must precede every evaluation row. Never a random split."""
    dataset = build_dataset(cfg, SEED, scenario=SCENARIO_HETEROGENEOUS, **TINY)
    assert dataset.train, "no training rows were built, so the split proves nothing"
    assert dataset.eval_rows, "no evaluation rows were built"

    latest_train = max(row.day for row in dataset.train)
    earliest_eval = min(row.day for row in dataset.eval_rows)
    assert latest_train < earliest_eval
    assert latest_train < cfg.forecast.warmup_days
    assert earliest_eval >= cfg.forecast.warmup_days


@pytest.mark.sim
def test_no_training_row_spans_a_collection(cfg: Config) -> None:
    """A target rate measured across a collection would teach the model that bins self-empty.

    The threshold separates two different things rather than demanding non-negativity:

    - **sensor noise** makes the fused estimate dip by a point or two day to day, so small
      negative observed rates are legitimate and must not fail this test;
    - **a collection reset** drops a near-full bin by tens of percentage points, which is what
      would actually poison the training target.

    -15 pp/day sits far above the noise floor (about 1 pp/day sd from the configured sensor
    noise) and far below any real reset, so it distinguishes the two.
    """
    dataset = build_dataset(cfg, SEED, scenario=SCENARIO_HETEROGENEOUS, **TINY)
    assert dataset.train, "no training rows were built, so this test proved nothing"
    worst = min(row.target_rate for row in dataset.train)
    assert worst > -15.0, f"a collection reset appears to have leaked in: {worst:.2f} pp/day"


@pytest.mark.sim
def test_a_three_day_backstop_can_still_be_trained(cfg: Config) -> None:
    """The bug that motivated the variable span, pinned so it cannot come back.

    With a fixed 3-day horizon and a 3-day collection calendar, every span contains a collection
    (any three consecutive days include a multiple of three), so every row is dropped and the
    model silently never fits. :func:`clean_span` takes the longest clean span instead.
    """
    from smartwaste.ml.forecast import clean_span

    collections = [0, 3, 6, 9, 12]
    available = set(range(30))
    spans = [clean_span(day, 3, collections, available) for day in range(1, 12)]
    assert any(span > 0 for span in spans), "a 3-day backstop could never produce a training row"
    # A collection day itself yields nothing: its own reading precedes the reset.
    assert clean_span(3, 3, collections, available) == 0
    # Day 1 is clean up to but not including the collection on day 3.
    assert clean_span(1, 3, collections, available) == 2


@pytest.mark.sim
def test_dataset_is_deterministic_for_one_seed(cfg: Config) -> None:
    first = build_dataset(cfg, SEED, scenario=SCENARIO_HETEROGENEOUS, **TINY)
    second = build_dataset(cfg, SEED, scenario=SCENARIO_HETEROGENEOUS, **TINY)
    assert [r.features for r in first.train] == [r.features for r in second.train]
    assert [r.fv_future for r in first.eval_rows] == [r.fv_future for r in second.eval_rows]


@pytest.mark.sim
def test_warmup_diagnostics_are_reported(cfg: Config) -> None:
    """The thin-warm-up limitation is measured, not asserted in prose only."""
    dataset = build_dataset(cfg, SEED, scenario=SCENARIO_HETEROGENEOUS, n_bins=25, days=30)
    assert dataset.warmup_days == cfg.forecast.warmup_days
    assert 0 <= dataset.festival_days_in_warmup <= dataset.total_festival_days
    assert 1 <= dataset.weekdays_in_warmup <= 7


@pytest.mark.sim
def test_baselines_score_without_the_ml_extra(cfg: Config) -> None:
    """The stdlib forecasters must be scoreable with scikit-learn absent."""
    dataset = build_dataset(cfg, SEED, scenario=SCENARIO_SMOOTH, **TINY)
    scores = score_forecasters(
        cfg,
        dataset,
        SEED,
        scenario=SCENARIO_SMOOTH,
        keys=("last_value", "moving_average", "seasonal_naive"),
    )
    assert len(scores) == 3
    for score in scores:
        assert score.mae_pp >= 0.0
        assert score.rmse_pp >= score.mae_pp - 1e-9
        assert score.n_eval == len(dataset.eval_rows)


@pytest.mark.sim
def test_prediction_cannot_see_beyond_the_row_being_predicted(cfg: Config) -> None:
    """A moving average over the full history would be leakage through the back door.

    The harness replays a truncated prefix per row; this checks the prefix really is truncated,
    by confirming a forecaster scored on prefixes differs from one handed whole histories.
    """
    dataset = build_dataset(cfg, SEED, scenario=SCENARIO_HETEROGENEOUS, **TINY)
    scores = score_forecasters(
        cfg, dataset, SEED, scenario=SCENARIO_HETEROGENEOUS, keys=("moving_average",)
    )
    leaky = MovingAverageForecaster(cfg.forecast.moving_average_window)
    horizon = dataset.horizon_days
    actual = [row.fv_future for row in dataset.eval_rows]
    predicted = [
        row.fv_now + leaky.predict_rate(dataset.histories[row.bin_index], row.features) * horizon
        for row in dataset.eval_rows
    ]
    from smartwaste.ml.metrics import mae

    assert scores[0].mae_pp != pytest.approx(mae(actual, predicted))


# --------------------------------------------------------------------------------------------
# ml chunk: anything that fits the gradient-boosting model
# --------------------------------------------------------------------------------------------


@pytest.mark.ml
def test_gbm_fits_and_predicts_a_finite_rate(cfg: Config) -> None:
    import math

    dataset = build_dataset(cfg, SEED, scenario=SCENARIO_HETEROGENEOUS, **TINY)
    gbm = build_forecaster("gbm", cfg.forecast, SEED)
    gbm.fit(dataset.train)
    assert gbm.is_fitted

    row = dataset.eval_rows[0]
    rate = gbm.predict_rate(dataset.histories[row.bin_index], row.features)
    assert math.isfinite(rate)


@pytest.mark.ml
def test_gbm_without_training_rows_falls_back_rather_than_crashing(cfg: Config) -> None:
    gbm = build_forecaster("gbm", cfg.forecast, SEED)
    gbm.fit([])
    assert not gbm.is_fitted
    history = _history([0.0, 10.0, 20.0])
    assert gbm.predict_rate(history, None) == pytest.approx(10.0)


@pytest.mark.ml
def test_gbm_is_reproducible_on_this_machine(cfg: Config) -> None:
    """Same seed, same machine, same numbers.

    Deliberately NOT a cross-machine claim: scikit-learn promises no bit-identical floats across
    operating systems, BLAS builds or versions, so no exact float is pinned as a literal here.
    """
    dataset = build_dataset(cfg, SEED, scenario=SCENARIO_HETEROGENEOUS, **TINY)
    first = score_forecasters(cfg, dataset, SEED, scenario="x", keys=("gbm",))[0]
    second = score_forecasters(cfg, dataset, SEED, scenario="x", keys=("gbm",))[0]
    assert first.mae_pp == second.mae_pp
    assert first.rmse_pp == second.rmse_pp


@pytest.mark.ml
def test_all_four_forecasters_are_scored_on_the_same_rows(cfg: Config) -> None:
    """The comparison must be paired, or a difference could just be a different sample."""
    dataset = build_dataset(cfg, SEED, scenario=SCENARIO_HETEROGENEOUS, **TINY)
    scores = score_forecasters(cfg, dataset, SEED, scenario=SCENARIO_HETEROGENEOUS)
    assert [s.key for s in scores] == list(FORECASTER_KEYS)
    assert len({s.n_eval for s in scores}) == 1


@pytest.mark.ml
def test_every_forecaster_beats_a_do_nothing_prediction(cfg: Config) -> None:
    """A floor, not a victory condition.

    Predicting "the fill will not change" is the weakest possible forecast. Any method that
    cannot beat it is broken rather than merely unimpressive. This is asserted as a property
    with a margin, never as an exact score.
    """
    from smartwaste.ml.metrics import mae

    dataset = build_dataset(cfg, SEED, scenario=SCENARIO_HETEROGENEOUS, n_bins=60, days=30)
    actual = [row.fv_future for row in dataset.eval_rows]
    do_nothing = mae(actual, [row.fv_now for row in dataset.eval_rows])

    for score in score_forecasters(cfg, dataset, SEED, scenario=SCENARIO_HETEROGENEOUS):
        assert score.mae_pp < do_nothing, (
            f"{score.key} ({score.mae_pp:.3f} pp) is no better than predicting no change "
            f"({do_nothing:.3f} pp)"
        )


# --------------------------------------------------------------------------------------------
# Config validation
# --------------------------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"warmup_days": 0}, "warmup_days must be >= 1"),
        ({"horizon_days": 0}, "horizon_days must be >= 1"),
        ({"history_days": 99}, "history_days must be <= forecast.warmup_days"),
        ({"moving_average_window": 0}, "moving_average_window must be >= 1"),
        ({"backstop_days": [0.0]}, "whole numbers >= 1"),
        ({"backstop_days": [3.5]}, "whole numbers >= 1"),
        ({"gbm_max_iter": 0}, "gbm_max_iter must be >= 1"),
        ({"gbm_learning_rate": 0.0}, "gbm_learning_rate must be > 0"),
        ({"gbm_max_depth": 0}, "gbm_max_depth must be >= 1"),
    ],
)
def test_invalid_forecast_config_is_rejected(override: dict[str, object], message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        load_config(overrides={"forecast": override})


@pytest.mark.unit
def test_forecast_config_defaults_are_the_documented_ones(cfg: Config) -> None:
    """Tripwire against silently retuning a published constant to help the model."""
    assert cfg.forecast.warmup_days == 14
    assert cfg.forecast.horizon_days == 3
    assert cfg.forecast.history_days == 5
    assert cfg.forecast.moving_average_window == 3
    assert cfg.forecast.backstops == (3, 7)
