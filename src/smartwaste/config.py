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

    def validate(self) -> None:
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
    cost: CostConfig
    routing: RoutingConfig
    fusion_eval: FusionEvalConfig
    paths: PathsConfig

    def validate(self) -> None:
        if self.seed < 0:
            raise ConfigError(f"seed must be >= 0, got {self.seed}")
        self.bin.validate()
        self.sensors.validate()
        self.sim.validate()
        self.cost.validate()
        self.routing.validate()
        self.fusion_eval.validate()
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
        cost=_section(raw, "cost", CostConfig),
        routing=_section(raw, "routing", RoutingConfig),
        fusion_eval=_section(raw, "fusion_eval", FusionEvalConfig),
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


def _coerce_annotated(value: object, annotation: str, where: str) -> int | float | str:
    """Coerce one YAML value according to its dataclass field annotation.

    YAML writes ``30`` where we declare a float, so ints are widened; bools are never accepted
    as numbers, and anything else wrong-typed is an error naming the key.
    """
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
