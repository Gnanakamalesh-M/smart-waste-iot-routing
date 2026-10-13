"""Typed loader for ``config/default.yaml``.

Every assumption in this project lives in one YAML file. This module turns that file into
frozen dataclasses so a typo becomes an error at load time instead of a silent wrong number
deep inside a simulation.

    from smartwaste import load_config

    cfg = load_config()                                   # config/default.yaml
    cfg = load_config(overrides={"bin": {"alpha": 0.9}})  # one value changed, rest untouched
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, TypeVar, cast

import yaml

# src/smartwaste/config.py -> src/smartwaste -> src -> repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "default.yaml"


class ConfigError(ValueError):
    """Raised when the config file is missing a key, has an extra key, or an invalid value."""


@dataclass(frozen=True)
class BinConfig:
    """Bin geometry and the paper's state thresholds."""

    height_cm: float
    w_max_kg: float
    alpha: float
    yellow_threshold_pct: float
    red_threshold_pct: float
    density_kg_per_pct: float
    density_cv: float
    max_accumulated_weight_kg: float

    def validate(self) -> None:
        if self.height_cm <= 0:
            raise ConfigError(f"bin.height_cm must be > 0, got {self.height_cm}")
        if self.w_max_kg <= 0:
            raise ConfigError(f"bin.w_max_kg must be > 0, got {self.w_max_kg}")
        if not 0 < self.alpha < 1:
            raise ConfigError(f"bin.alpha must be strictly between 0 and 1, got {self.alpha}")
        if not 0 <= self.yellow_threshold_pct < self.red_threshold_pct <= 100:
            raise ConfigError(
                "bin thresholds must satisfy 0 <= yellow < red <= 100, got "
                f"yellow={self.yellow_threshold_pct}, red={self.red_threshold_pct}"
            )
        if self.density_kg_per_pct <= 0:
            raise ConfigError(f"bin.density_kg_per_pct must be > 0, got {self.density_kg_per_pct}")
        if not 0 <= self.density_cv < 1:
            raise ConfigError(f"bin.density_cv must be in [0, 1), got {self.density_cv}")
        # A missed bin keeps accruing weight up to this cap. It must sit above the overload
        # threshold alpha * W_max, or an uncollected bin could never raise the load-cell flag
        # and the fusion strategies would never recover it.
        if self.max_accumulated_weight_kg < self.weight_threshold_kg:
            raise ConfigError(
                "bin.max_accumulated_weight_kg must be >= alpha * w_max_kg "
                f"({self.weight_threshold_kg}), got {self.max_accumulated_weight_kg}"
            )

    @property
    def weight_threshold_kg(self) -> float:
        """W_th = alpha * W_max (paper's load-cell threshold)."""
        return self.alpha * self.w_max_kg


@dataclass(frozen=True)
class SensorsConfig:
    """Ultrasonic array size and the noise model applied to its readings."""

    n_ultrasonic: int
    noise_sigma_cm: float
    bias_max_cm: float
    stuck_probability: float
    uneven_fill_skew_cm: float
    min_distance_cm: float
    load_cell_sigma_kg: float
    load_cell_bias_max_kg: float
    stuck_bin_fraction: float

    def validate(self) -> None:
        if not 0 <= self.stuck_bin_fraction <= 1:
            raise ConfigError(
                f"sensors.stuck_bin_fraction must be in [0, 1], got {self.stuck_bin_fraction}"
            )
        if self.n_ultrasonic < 1:
            raise ConfigError(f"sensors.n_ultrasonic must be >= 1, got {self.n_ultrasonic}")
        if self.noise_sigma_cm < 0:
            raise ConfigError(f"sensors.noise_sigma_cm must be >= 0, got {self.noise_sigma_cm}")
        if self.bias_max_cm < 0:
            raise ConfigError(f"sensors.bias_max_cm must be >= 0, got {self.bias_max_cm}")
        if not 0 <= self.stuck_probability <= 1:
            raise ConfigError(
                f"sensors.stuck_probability must be in [0, 1], got {self.stuck_probability}"
            )
        if self.uneven_fill_skew_cm < 0:
            raise ConfigError(
                f"sensors.uneven_fill_skew_cm must be >= 0, got {self.uneven_fill_skew_cm}"
            )
        if self.min_distance_cm < 0:
            raise ConfigError(f"sensors.min_distance_cm must be >= 0, got {self.min_distance_cm}")
        if self.load_cell_sigma_kg < 0:
            raise ConfigError(
                f"sensors.load_cell_sigma_kg must be >= 0, got {self.load_cell_sigma_kg}"
            )
        if self.load_cell_bias_max_kg < 0:
            raise ConfigError(
                f"sensors.load_cell_bias_max_kg must be >= 0, got {self.load_cell_bias_max_kg}"
            )


@dataclass(frozen=True)
class SimConfig:
    """Size of the city simulation and the daily waste arrival process."""

    n_bins: int
    days: int
    grid_km: float
    daily_fill_mean_kg: float
    daily_fill_sd_kg: float

    def validate(self) -> None:
        if self.n_bins < 1:
            raise ConfigError(f"sim.n_bins must be >= 1, got {self.n_bins}")
        if self.days < 1:
            raise ConfigError(f"sim.days must be >= 1, got {self.days}")
        if self.grid_km <= 0:
            raise ConfigError(f"sim.grid_km must be > 0, got {self.grid_km}")
        if self.daily_fill_mean_kg < 0:
            raise ConfigError(f"sim.daily_fill_mean_kg must be >= 0, got {self.daily_fill_mean_kg}")
        if self.daily_fill_sd_kg < 0:
            raise ConfigError(f"sim.daily_fill_sd_kg must be >= 0, got {self.daily_fill_sd_kg}")


@dataclass(frozen=True)
class DemandConfig:
    """Phase 3's waste arrival process. **Every value here is invented and uncited.**

    The paper states only that daily increments are "random" -- no distribution, no
    heterogeneity, no calendar. So area types, the weekday cycle and festival spikes are all
    modelling choices of ours with no field data behind them. See
    :mod:`smartwaste.demand` and docs/ASSUMPTIONS.md.
    """

    scenario: str
    area_type_weights: tuple[float, ...]
    area_mean_kg: tuple[float, ...]
    cv: float
    weekday_multipliers: tuple[float, ...]
    festival_days_per_month: float
    festival_multiplier: float

    def validate(self) -> None:
        # Imported lazily: smartwaste.demand imports this module, so a top-level import here
        # would be a cycle.
        from smartwaste.demand import DAYS_PER_WEEK, SCENARIOS

        if self.scenario not in SCENARIOS:
            raise ConfigError(f"demand.scenario must be one of {SCENARIOS}, got {self.scenario!r}")
        if not self.area_type_weights:
            raise ConfigError("demand.area_type_weights must list at least one value")
        if len(self.area_type_weights) != len(self.area_mean_kg):
            raise ConfigError(
                "demand.area_type_weights and demand.area_mean_kg must have the same length, got "
                f"{len(self.area_type_weights)} and {len(self.area_mean_kg)}"
            )
        if any(weight < 0 for weight in self.area_type_weights):
            raise ConfigError(
                f"demand.area_type_weights must all be >= 0, got {list(self.area_type_weights)}"
            )
        total = sum(self.area_type_weights)
        if abs(total - 1.0) > 1e-6:
            raise ConfigError(f"demand.area_type_weights must sum to 1.0, got {total}")
        if any(mean < 0 for mean in self.area_mean_kg):
            raise ConfigError(
                f"demand.area_mean_kg must all be >= 0, got {list(self.area_mean_kg)}"
            )
        if self.cv < 0:
            raise ConfigError(f"demand.cv must be >= 0, got {self.cv}")
        if len(self.weekday_multipliers) != DAYS_PER_WEEK:
            raise ConfigError(
                f"demand.weekday_multipliers must have exactly {DAYS_PER_WEEK} entries "
                f"(Mon..Sun), got {len(self.weekday_multipliers)}"
            )
        if any(multiplier < 0 for multiplier in self.weekday_multipliers):
            raise ConfigError(
                f"demand.weekday_multipliers must all be >= 0, got {list(self.weekday_multipliers)}"
            )
        if self.festival_days_per_month < 0:
            raise ConfigError(
                f"demand.festival_days_per_month must be >= 0, got {self.festival_days_per_month}"
            )
        if self.festival_multiplier < 0:
            raise ConfigError(
                f"demand.festival_multiplier must be >= 0, got {self.festival_multiplier}"
            )


@dataclass(frozen=True)
class ForecastConfig:
    """Phase 3 fill-level forecasting. **Every value here is invented and uncited.**"""

    warmup_days: int
    horizon_days: int
    history_days: int
    moving_average_window: int
    backstop_days: tuple[float, ...]
    gbm_max_iter: int
    gbm_learning_rate: float
    gbm_max_depth: int

    def validate(self) -> None:
        if self.warmup_days < 1:
            raise ConfigError(f"forecast.warmup_days must be >= 1, got {self.warmup_days}")
        if self.horizon_days < 1:
            raise ConfigError(f"forecast.horizon_days must be >= 1, got {self.horizon_days}")
        if self.history_days < 1:
            raise ConfigError(f"forecast.history_days must be >= 1, got {self.history_days}")
        if self.history_days > self.warmup_days:
            raise ConfigError(
                "forecast.history_days must be <= forecast.warmup_days, or no training row "
                f"could ever be built: got {self.history_days} > {self.warmup_days}"
            )
        if self.moving_average_window < 1:
            raise ConfigError(
                f"forecast.moving_average_window must be >= 1, got {self.moving_average_window}"
            )
        if not self.backstop_days:
            raise ConfigError("forecast.backstop_days must list at least one value")
        for value in self.backstop_days:
            if value < 1 or value != int(value):
                raise ConfigError(
                    f"forecast.backstop_days values must be whole numbers >= 1, got {value}"
                )
        if self.gbm_max_iter < 1:
            raise ConfigError(f"forecast.gbm_max_iter must be >= 1, got {self.gbm_max_iter}")
        if self.gbm_learning_rate <= 0:
            raise ConfigError(
                f"forecast.gbm_learning_rate must be > 0, got {self.gbm_learning_rate}"
            )
        if self.gbm_max_depth < 1:
            raise ConfigError(f"forecast.gbm_max_depth must be >= 1, got {self.gbm_max_depth}")

    @property
    def backstops(self) -> tuple[int, ...]:
        """``backstop_days`` as ints, which is how the schedules actually use them."""
        return tuple(int(value) for value in self.backstop_days)


@dataclass(frozen=True)
class AnomalyConfig:
    """Phase 3 sensor anomaly detection. **Every value here is invented and uncited.**

    ``validation_seed_offset`` exists so detector thresholds are chosen on a *different* city
    from the one they are scored on. Tuning a threshold on the scoring seed's planted labels
    would be the same leakage this project forbids everywhere else.
    """

    n_bins: int
    days: int
    fault_bin_fraction: float
    window_days: int
    validation_seed_offset: int
    min_fault_days: int
    max_fault_days: int
    drift_cm_per_day: float
    noise_burst_sigma_cm: float
    stuck_tolerance_cm: float
    disagreement_cm: float
    contamination: float
    n_estimators: int

    def validate(self) -> None:
        # Dataset size first: the duration checks below are stated relative to `days`, so a bad
        # `days` must be reported as such rather than surfacing as a confusing duration error.
        if self.n_bins < 1:
            raise ConfigError(f"anomaly.n_bins must be >= 1, got {self.n_bins}")
        if self.days < 1:
            raise ConfigError(f"anomaly.days must be >= 1, got {self.days}")
        if not 1 <= self.min_fault_days <= self.max_fault_days:
            raise ConfigError(
                "anomaly fault durations must satisfy 1 <= min_fault_days <= max_fault_days, got "
                f"min={self.min_fault_days}, max={self.max_fault_days}"
            )
        if self.max_fault_days > self.days:
            raise ConfigError(
                f"anomaly.max_fault_days must be <= anomaly.days, got "
                f"{self.max_fault_days} > {self.days}"
            )
        if self.drift_cm_per_day < 0:
            raise ConfigError(f"anomaly.drift_cm_per_day must be >= 0, got {self.drift_cm_per_day}")
        if self.noise_burst_sigma_cm < 0:
            raise ConfigError(
                f"anomaly.noise_burst_sigma_cm must be >= 0, got {self.noise_burst_sigma_cm}"
            )
        if not 0 <= self.fault_bin_fraction <= 1:
            raise ConfigError(
                f"anomaly.fault_bin_fraction must be in [0, 1], got {self.fault_bin_fraction}"
            )
        if self.window_days < 2:
            raise ConfigError(
                f"anomaly.window_days must be >= 2 (a variance needs two points), got "
                f"{self.window_days}"
            )
        if self.window_days > self.days:
            raise ConfigError(
                f"anomaly.window_days must be <= anomaly.days, got {self.window_days} > {self.days}"
            )
        if self.validation_seed_offset == 0:
            raise ConfigError(
                "anomaly.validation_seed_offset must be non-zero: thresholds must be chosen on a "
                "different city from the one they are scored on"
            )
        if self.stuck_tolerance_cm < 0:
            raise ConfigError(
                f"anomaly.stuck_tolerance_cm must be >= 0, got {self.stuck_tolerance_cm}"
            )
        if self.disagreement_cm < 0:
            raise ConfigError(f"anomaly.disagreement_cm must be >= 0, got {self.disagreement_cm}")
        if not 0 < self.contamination < 0.5:
            raise ConfigError(
                f"anomaly.contamination must be in (0, 0.5), got {self.contamination}"
            )
        if self.n_estimators < 1:
            raise ConfigError(f"anomaly.n_estimators must be >= 1, got {self.n_estimators}")


@dataclass(frozen=True)
class CostConfig:
    """Fuel, CO2, labor and fixed-cost rates for the economic model."""

    fuel_l_per_km: float
    fuel_cost_per_l: float
    co2_kg_per_l: float
    maintenance_cost_per_km: float
    labor_cost_per_hr: float
    fixed_cost_per_day: float
    truck_speed_kmph: float
    service_time_min_per_bin: float
    truck_capacity_kg: float

    def validate(self) -> None:
        non_negative = (
            "fuel_l_per_km",
            "fuel_cost_per_l",
            "co2_kg_per_l",
            "maintenance_cost_per_km",
            "labor_cost_per_hr",
            "fixed_cost_per_day",
            "service_time_min_per_bin",
        )
        for name in non_negative:
            value = getattr(self, name)
            if value < 0:
                raise ConfigError(f"cost.{name} must be >= 0, got {value}")
        if self.truck_speed_kmph <= 0:
            raise ConfigError(f"cost.truck_speed_kmph must be > 0, got {self.truck_speed_kmph}")
        if self.truck_capacity_kg <= 0:
            raise ConfigError(f"cost.truck_capacity_kg must be > 0, got {self.truck_capacity_kg}")


@dataclass(frozen=True)
class RoutingConfig:
    """Clustering and TSP/VRP knobs."""

    n_clusters: int
    n_trucks_per_cluster: int
    two_opt_max_iters: int

    def validate(self) -> None:
        if self.n_clusters < 1:
            raise ConfigError(f"routing.n_clusters must be >= 1, got {self.n_clusters}")
        if self.n_trucks_per_cluster < 1:
            raise ConfigError(
                f"routing.n_trucks_per_cluster must be >= 1, got {self.n_trucks_per_cluster}"
            )
        if self.two_opt_max_iters < 0:
            raise ConfigError(
                f"routing.two_opt_max_iters must be >= 0, got {self.two_opt_max_iters}"
            )


@dataclass(frozen=True)
class FusionEvalConfig:
    """Size of the Phase 1 sensor-fusion evaluation."""

    n_bins: int
    days: int
    sweep_samples: int
    bootstrap_resamples: int

    def validate(self) -> None:
        if self.n_bins < 1:
            raise ConfigError(f"fusion_eval.n_bins must be >= 1, got {self.n_bins}")
        if self.days < 1:
            raise ConfigError(f"fusion_eval.days must be >= 1, got {self.days}")
        if self.sweep_samples < 1:
            raise ConfigError(f"fusion_eval.sweep_samples must be >= 1, got {self.sweep_samples}")
        if self.bootstrap_resamples < 1:
            raise ConfigError(
                f"fusion_eval.bootstrap_resamples must be >= 1, got {self.bootstrap_resamples}"
            )


@dataclass(frozen=True)
class BenchmarkConfig:
    """Size and shape of the Phase 2 city benchmark."""

    fixed_schedule_days: int
    n_seeds: int
    fault_fractions: tuple[float, ...]

    def validate(self) -> None:
        if self.fixed_schedule_days < 1:
            raise ConfigError(
                f"benchmark.fixed_schedule_days must be >= 1, got {self.fixed_schedule_days}"
            )
        if self.n_seeds < 1:
            raise ConfigError(f"benchmark.n_seeds must be >= 1, got {self.n_seeds}")
        if not self.fault_fractions:
            raise ConfigError("benchmark.fault_fractions must list at least one value")
        for value in self.fault_fractions:
            if not 0 <= value <= 1:
                raise ConfigError(
                    f"benchmark.fault_fractions values must be in [0, 1], got {value}"
                )


@dataclass(frozen=True)
class PathsConfig:
    """Output and cache directories, relative to the repo root unless absolute."""

    data_dir: str
    cache_dir: str
    results_dir: str

    def validate(self) -> None:
        for name in ("data_dir", "cache_dir", "results_dir"):
            if not str(getattr(self, name)).strip():
                raise ConfigError(f"paths.{name} must not be empty")

    def resolve(self, name: str, root: Path | None = None) -> Path:
        """Return one directory as an absolute path (does not create it)."""
        if name not in {"data_dir", "cache_dir", "results_dir"}:
            raise ConfigError(f"unknown path name: {name!r}")
        return (root or REPO_ROOT) / str(getattr(self, name))


@dataclass(frozen=True)
class Config:
    """The whole config file."""

    seed: int
    bin: BinConfig
    sensors: SensorsConfig
    sim: SimConfig
    demand: DemandConfig
    forecast: ForecastConfig
    anomaly: AnomalyConfig
    cost: CostConfig
    routing: RoutingConfig
    fusion_eval: FusionEvalConfig
    benchmark: BenchmarkConfig
    paths: PathsConfig

    def validate(self) -> None:
        if self.seed < 0:
            raise ConfigError(f"seed must be >= 0, got {self.seed}")
        self.bin.validate()
        self.sensors.validate()
        self.sim.validate()
        self.demand.validate()
        self.forecast.validate()
        self.anomaly.validate()
        self.cost.validate()
        self.routing.validate()
        self.fusion_eval.validate()
        self.benchmark.validate()
        self.paths.validate()


def load_config(
    path: Path | str | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> Config:
    """Load, override and validate the config.

    Args:
        path: YAML file to read. Defaults to ``config/default.yaml`` in the repo.
        overrides: Nested mapping deep-merged over the file, e.g. ``{"bin": {"alpha": 0.9}}``.

    Raises:
        ConfigError: unknown key, missing key, or an out-of-range value.
    """
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.is_file():
        raise ConfigError(f"config file not found: {config_path}")

    with config_path.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ConfigError(f"{config_path} must contain a YAML mapping at the top level")

    raw: dict[str, Any] = _deep_merge(loaded, overrides or {})
    _reject_unknown_top_level(raw)

    cfg = Config(
        seed=_require_int(raw, "seed"),
        bin=_section(raw, "bin", BinConfig),
        sensors=_section(raw, "sensors", SensorsConfig),
        sim=_section(raw, "sim", SimConfig),
        demand=_section(raw, "demand", DemandConfig),
        forecast=_section(raw, "forecast", ForecastConfig),
        anomaly=_section(raw, "anomaly", AnomalyConfig),
        cost=_section(raw, "cost", CostConfig),
        routing=_section(raw, "routing", RoutingConfig),
        fusion_eval=_section(raw, "fusion_eval", FusionEvalConfig),
        benchmark=_section(raw, "benchmark", BenchmarkConfig),
        paths=_section(raw, "paths", PathsConfig),
    )
    cfg.validate()
    return cfg


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Merge ``override`` onto ``base`` without mutating either.

    Nested mappings are merged key by key; every other value is replaced outright.
    """
    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge(existing, value)
        else:
            merged[key] = value
    return merged


def _reject_unknown_top_level(raw: Mapping[str, Any]) -> None:
    known = {field.name for field in fields(Config)}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ConfigError(f"unknown top-level config key(s): {', '.join(unknown)}")


S = TypeVar("S")


def _require_int(raw: Mapping[str, Any], key: str) -> int:
    if key not in raw:
        raise ConfigError(f"missing config key: {key}")
    value = raw[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise _wrong_type(value, "int", key)
    return value


def _section(raw: Mapping[str, Any], name: str, cls: type[S]) -> S:
    """Build one dataclass section, rejecting missing and unknown keys by name."""
    if name not in raw:
        raise ConfigError(f"missing config section: {name}")
    block = raw[name]
    if not isinstance(block, Mapping):
        raise ConfigError(f"config section {name} must be a mapping, got {type(block).__name__}")

    # Annotations arrive as strings because of ``from __future__ import annotations``.
    expected = {field.name: str(field.type) for field in fields(cast(Any, cls))}
    unknown = sorted(set(block) - set(expected))
    if unknown:
        raise ConfigError(f"unknown config key(s) in {name}: {', '.join(unknown)}")
    missing = sorted(set(expected) - set(block))
    if missing:
        raise ConfigError(f"missing config key(s) in {name}: {', '.join(missing)}")

    values = {
        key: _coerce_annotated(block[key], annotation, f"{name}.{key}")
        for key, annotation in expected.items()
    }
    return cls(**values)


def _coerce_annotated(
    value: object, annotation: str, where: str
) -> int | float | str | tuple[float, ...]:
    """Coerce one YAML value according to its dataclass field annotation.

    YAML writes ``30`` where we declare a float, so ints are widened; bools are never accepted
    as numbers, and anything else wrong-typed is an error naming the key.
    """
    if annotation == "tuple[float, ...]":
        # A YAML sequence of numbers, e.g. benchmark.fault_fractions: [0.0, 0.02, 0.10].
        # A bare scalar is rejected rather than wrapped, so a typo'd `0.02` is an error
        # instead of a silently one-element sweep.
        if isinstance(value, str) or not isinstance(value, list | tuple):
            raise _wrong_type(value, "a list of numbers", where)
        items: list[float] = []
        for position, item in enumerate(value):
            if isinstance(item, bool) or not isinstance(item, int | float):
                raise _wrong_type(item, "float", f"{where}[{position}]")
            items.append(float(item))
        return tuple(items)
    if annotation == "int":
        if isinstance(value, bool) or not isinstance(value, int):
            raise _wrong_type(value, "int", where)
        return value
    if annotation == "float":
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise _wrong_type(value, "float", where)
        return float(value)
    if annotation == "str":
        if not isinstance(value, str):
            raise _wrong_type(value, "str", where)
        return value
    raise ConfigError(f"unsupported field type {annotation!r} for {where}")


def _wrong_type(value: object, expected: str, where: str) -> ConfigError:
    return ConfigError(f"{where} must be {expected}, got {type(value).__name__} ({value!r})")
