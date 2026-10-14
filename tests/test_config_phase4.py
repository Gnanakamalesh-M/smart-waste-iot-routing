"""Config loading for Phase 4's two new sections, `vrp` and `roads`.

A separate file from `tests/test_config.py` on purpose: Phase 4 must not edit a single
pre-existing test, so that `git diff --name-only <phase 3 sha> HEAD -- tests/` lists only new
files and "every existing test passes unchanged" is checkable rather than claimed.
"""

from pathlib import Path

import pytest

from smartwaste.config import ConfigError, load_config
from smartwaste.routing.roads import NETWORK_TYPES
from smartwaste.routing.vrp import FIRST_SOLUTION_STRATEGIES, LOCAL_SEARCH_METAHEURISTICS

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------------------------
# The committed values. These are the numbers docs/PHASE4_RESULTS.md is produced with.
# --------------------------------------------------------------------------------------------


def test_loads_the_phase_4_vrp_parameters(default_config_path: Path) -> None:
    cfg = load_config(default_config_path)
    assert cfg.vrp.capacity_sweep_kg == (4000.0, 8000.0, 16000.0)
    assert cfg.vrp.max_trucks_per_cluster == 20
    assert cfg.vrp.ortools_time_limit_s == 5.0
    assert cfg.vrp.first_solution_strategy == "PATH_CHEAPEST_ARC"
    assert cfg.vrp.local_search_metaheuristic == "GUIDED_LOCAL_SEARCH"
    assert cfg.vrp.compare_clusters == 20
    assert cfg.vrp.compare_day == 14


def test_loads_the_phase_4_roads_parameters(default_config_path: Path) -> None:
    cfg = load_config(default_config_path)
    assert cfg.roads.area_name == "T. Nagar, Chennai, India"
    assert cfg.roads.center_lat == 13.0418
    assert cfg.roads.center_lon == 80.2341
    assert cfg.roads.box_km == 3.0
    assert cfg.roads.network_type == "drive"
    assert cfg.roads.n_bins == 300
    assert cfg.roads.cache_subdir == "osmnx"


def test_the_sweep_contains_the_committed_truck_capacity(default_config_path: Path) -> None:
    """The sweep has to bracket the real config value, or it is not a sensitivity test of it.

    `cost.truck_capacity_kg` was deliberately NOT changed in Phase 4 -- raising it to tidy the
    truck counts would be tuning to the output -- so the sweep must still contain it.
    """
    cfg = load_config(default_config_path)
    assert cfg.cost.truck_capacity_kg in cfg.vrp.capacity_sweep_kg
    assert min(cfg.vrp.capacity_sweep_kg) < cfg.cost.truck_capacity_kg
    assert max(cfg.vrp.capacity_sweep_kg) > cfg.cost.truck_capacity_kg


def test_the_comparison_day_falls_inside_the_horizon(default_config_path: Path) -> None:
    cfg = load_config(default_config_path)
    assert cfg.vrp.compare_day < cfg.sim.days


def test_the_solver_knobs_are_inside_their_whitelists(default_config_path: Path) -> None:
    cfg = load_config(default_config_path)
    assert cfg.vrp.first_solution_strategy in FIRST_SOLUTION_STRATEGIES
    assert cfg.vrp.local_search_metaheuristic in LOCAL_SEARCH_METAHEURISTICS
    assert cfg.roads.network_type in NETWORK_TYPES


# --------------------------------------------------------------------------------------------
# Rejections. A typo in the config must be an error at load time, not a wrong number later.
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"capacity_sweep_kg": []}, "must list at least one value"),
        ({"capacity_sweep_kg": [8000.0, 0.0]}, "values must be > 0"),
        ({"capacity_sweep_kg": [-1.0]}, "values must be > 0"),
        ({"max_trucks_per_cluster": 0}, "must be >= 1"),
        ({"ortools_time_limit_s": 0.0}, "must be > 0"),
        ({"ortools_time_limit_s": -2.0}, "must be > 0"),
        ({"first_solution_strategy": "NOPE"}, "first_solution_strategy must be one of"),
        ({"local_search_metaheuristic": "NOPE"}, "local_search_metaheuristic must be one of"),
        ({"compare_clusters": 0}, "must be >= 1"),
        ({"compare_day": -1}, "must be >= 0"),
    ],
)
def test_rejects_a_bad_vrp_value(
    default_config_path: Path, override: dict[str, object], message: str
) -> None:
    with pytest.raises(ConfigError, match=message):
        load_config(default_config_path, overrides={"vrp": override})


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"area_name": "   "}, "area_name must not be empty"),
        ({"center_lat": 91.0}, r"center_lat must be in \[-90, 90\]"),
        ({"center_lat": -91.0}, r"center_lat must be in \[-90, 90\]"),
        ({"center_lon": 181.0}, r"center_lon must be in \[-180, 180\]"),
        ({"box_km": 0.0}, "box_km must be > 0"),
        ({"box_km": -3.0}, "box_km must be > 0"),
        ({"network_type": "hovercraft"}, "network_type must be one of"),
        ({"n_bins": 1}, "n_bins must be >= 2"),
        ({"cache_subdir": ""}, "cache_subdir must not be empty"),
    ],
)
def test_rejects_a_bad_roads_value(
    default_config_path: Path, override: dict[str, object], message: str
) -> None:
    with pytest.raises(ConfigError, match=message):
        load_config(default_config_path, overrides={"roads": override})


def test_rejects_an_unknown_key_in_either_new_section(default_config_path: Path) -> None:
    with pytest.raises(ConfigError, match="unknown config key"):
        load_config(default_config_path, overrides={"vrp": {"made_up": 1}})
    with pytest.raises(ConfigError, match="unknown config key"):
        load_config(default_config_path, overrides={"roads": {"made_up": 1}})


def test_rejects_a_wrong_type(default_config_path: Path) -> None:
    with pytest.raises(ConfigError, match="vrp.max_trucks_per_cluster must be int"):
        load_config(default_config_path, overrides={"vrp": {"max_trucks_per_cluster": "two"}})
    with pytest.raises(ConfigError, match="roads.box_km must be float"):
        load_config(default_config_path, overrides={"roads": {"box_km": "three"}})
    with pytest.raises(ConfigError, match="roads.area_name must be str"):
        load_config(default_config_path, overrides={"roads": {"area_name": 42}})


def test_rejects_a_scalar_where_the_sweep_needs_a_list(default_config_path: Path) -> None:
    """A typo'd `8000.0` must not become a silently one-element sweep."""
    with pytest.raises(ConfigError, match="capacity_sweep_kg must be a list of numbers"):
        load_config(default_config_path, overrides={"vrp": {"capacity_sweep_kg": 8000.0}})


def test_a_good_override_still_loads(default_config_path: Path) -> None:
    """The rejection tests above must not be passing because every override fails."""
    cfg = load_config(
        default_config_path,
        overrides={
            "vrp": {"compare_clusters": 3, "ortools_time_limit_s": 1.0},
            "roads": {"n_bins": 50, "box_km": 1.5},
        },
    )
    assert cfg.vrp.compare_clusters == 3
    assert cfg.vrp.ortools_time_limit_s == 1.0
    assert cfg.roads.n_bins == 50
    assert cfg.roads.box_km == 1.5
    # Everything not overridden is untouched.
    assert cfg.vrp.capacity_sweep_kg == (4000.0, 8000.0, 16000.0)
    assert cfg.roads.area_name == "T. Nagar, Chennai, India"
