"""Euclidean vs real road distance on the SAME bin set, and the tables for it.

The one question: **by how much does a straight line under-state a road distance?**
``docs/ASSUMPTIONS.md`` has said since Phase 0 that this was "an amount we have not yet
measured". This module measures it, two ways:

1. **pairwise** -- ``road / euclidean`` for every pair of bins, with spread, not just a mean;
2. **tour-length** -- the same bin set routed by the same nearest-neighbour-plus-2-opt code under
   each matrix, so the ratio also appears as something a truck would actually drive.

### What is being held constant, and what is not

Held constant: the bin set, the depot, the routing algorithm, the iteration cap. The *only*
difference between the two numbers is the distance metric.

**Not** held constant with anything in Phases 0-3. This is a ~3 km area with a few hundred bins;
Phase 2 is a 100 km square with 5000. There is no "% vs Phase 2" column anywhere, deliberately.

Road data (c) OpenStreetMap contributors, ODbL 1.0. **Bin locations are synthetic** -- see
:mod:`smartwaste.routing.roads`.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass

from smartwaste.config import Config
from smartwaste.routing.roads import (
    FALLBACK_LABEL,
    OSM_CREDIT,
    SOURCE_UNAVAILABLE,
    SYNTHETIC_BINS_NOTE,
    BoundingBox,
    DetourStats,
    LatLon,
    bbox_for,
    detour_ratio_stats,
    euclidean_matrix,
    graph_cache_path,
    load_road_graph,
    road_distance_matrix,
    sample_bins_on_graph,
)
from smartwaste.routing.vrp import matrix_nearest_neighbour, matrix_tour_length, matrix_two_opt
from smartwaste.seeds import child_seed

CSV_FIELDS = (
    "table",
    "row_key",
    "label",
    "value",
)


@dataclass(frozen=True)
class RoadsResult:
    """Everything one roads run produced."""

    area_name: str
    bbox: BoundingBox
    network_type: str
    graph_source: str
    graph_note: str
    n_graph_nodes: int
    n_graph_edges: int
    n_bins: int
    #: Distinct road nodes the bins snapped to. Fewer than ``n_bins`` means collisions.
    n_distinct_nodes: int
    detour: DetourStats
    euclidean_tour_km: float
    road_tour_km: float

    @property
    def road_available(self) -> bool:
        return self.graph_source != SOURCE_UNAVAILABLE

    @property
    def tour_ratio(self) -> float:
        if self.euclidean_tour_km <= 0:
            return math.nan
        return self.road_tour_km / self.euclidean_tour_km


def run_roads_eval(
    cfg: Config,
    seed: int,
    *,
    n_bins: int,
    allow_download: bool = True,
) -> RoadsResult:
    """Fetch the graph, sample synthetic bins, build both matrices, route under each.

    On a graph failure this returns a result with ``graph_source == "unavailable"``, zero graph
    size and a **road matrix equal to the Euclidean one**, so every ratio is exactly 1.0 and the
    caller's tables carry :data:`~smartwaste.routing.roads.FALLBACK_LABEL`. A ratio of 1.0 is not
    a finding about roads; it is the visible signature of the fallback, and it is labelled as one.
    """
    bbox = bbox_for(cfg.roads.center_lat, cfg.roads.center_lon, cfg.roads.box_km)
    cache_path = graph_cache_path(
        cfg.paths.resolve("cache_dir"), cfg.roads.cache_subdir, bbox, cfg.roads.network_type
    )
    result = load_road_graph(
        bbox, cfg.roads.network_type, cache_path, allow_download=allow_download
    )

    rng = random.Random(child_seed(seed, "roads:bins"))

    if not result.available:
        # No graph: sample in the bbox anyway so the Euclidean half of the comparison still has a
        # bin set, and report the fallback loudly in every table.
        points = _sample_without_graph(bbox, n_bins, rng)
        euclid = euclidean_matrix(points)
        tour = _tour_km(euclid, cfg.routing.two_opt_max_iters)
        return RoadsResult(
            area_name=cfg.roads.area_name,
            bbox=bbox,
            network_type=cfg.roads.network_type,
            graph_source=result.source,
            graph_note=result.note,
            n_graph_nodes=0,
            n_graph_edges=0,
            n_bins=n_bins,
            n_distinct_nodes=0,
            detour=detour_ratio_stats(euclid, euclid),
            euclidean_tour_km=tour,
            road_tour_km=tour,
        )

    graph = result.graph
    nodes, snapped = sample_bins_on_graph(graph, bbox, n_bins, rng)

    # The depot is the bbox centre, snapped like everything else, and sits at index 0 so the
    # matrix tours start and end there exactly as Phase 2's depot-constrained TSP does.
    depot_nodes, depot_points = _snap_depot(graph, cfg)
    all_nodes = depot_nodes + nodes
    all_points: list[LatLon] = depot_points + snapped

    euclid = euclidean_matrix(all_points)
    road = road_distance_matrix(graph, all_nodes)

    return RoadsResult(
        area_name=cfg.roads.area_name,
        bbox=bbox,
        network_type=cfg.roads.network_type,
        graph_source=result.source,
        graph_note=result.note,
        n_graph_nodes=int(graph.number_of_nodes()),  # type: ignore[union-attr]
        n_graph_edges=int(graph.number_of_edges()),  # type: ignore[union-attr]
        n_bins=n_bins,
        n_distinct_nodes=len(set(nodes)),
        detour=detour_ratio_stats(euclid, road),
        euclidean_tour_km=_tour_km(euclid, cfg.routing.two_opt_max_iters),
        road_tour_km=_tour_km(road, cfg.routing.two_opt_max_iters),
    )


def _snap_depot(graph: object, cfg: Config) -> tuple[list[int], list[LatLon]]:
    """Snap the area centre to a road node: the single central depot, as in Phase 2."""
    import osmnx as ox

    node = int(ox.distance.nearest_nodes(graph, X=cfg.roads.center_lon, Y=cfg.roads.center_lat))
    graph_nodes = graph.nodes  # type: ignore[attr-defined]
    return [node], [(float(graph_nodes[node]["y"]), float(graph_nodes[node]["x"]))]


def _sample_without_graph(bbox: BoundingBox, n_bins: int, rng: random.Random) -> list[LatLon]:
    """The fallback bin sample: the same draws :func:`sample_bins_on_graph` makes, unsnapped.

    Same number of draws in the same order, so the fallback and the road run see the same
    underlying random stream and a comparison between the two is not also a comparison of samples.
    A depot at the bbox centre is prepended, matching the road path's index 0.
    """
    points: list[LatLon] = [((bbox.north + bbox.south) / 2.0, (bbox.east + bbox.west) / 2.0)]
    for _ in range(n_bins):
        lat = rng.uniform(bbox.south, bbox.north)
        lon = rng.uniform(bbox.west, bbox.east)
        points.append((lat, lon))
    return points


def _tour_km(matrix: Sequence[Sequence[float]], max_iters: int) -> float:
    """Nearest neighbour then 2-opt through a matrix, from the depot at index 0.

    Unreachable legs are replaced by a large finite stand-in so 2-opt has something to compare:
    an ``inf`` in the matrix makes every candidate swap incomparable and the tour meaningless.
    The substitution is reported through ``DetourStats.n_unreachable_pairs`` rather than hidden,
    and it is why the pairwise ratio -- which skips unreachable pairs outright -- is the primary
    number and the tour ratio the secondary one.
    """
    finite = _finite_matrix(matrix)
    tour = matrix_two_opt(finite, matrix_nearest_neighbour(finite), max_iters)
    return matrix_tour_length(finite, tour)


def _finite_matrix(matrix: Sequence[Sequence[float]]) -> list[list[float]]:
    """Replace every non-finite entry with twice the largest finite distance present."""
    finite_values = [v for row in matrix for v in row if math.isfinite(v)]
    stand_in = 2.0 * max(finite_values) if finite_values else 0.0
    return [[v if math.isfinite(v) else stand_in for v in row] for row in matrix]


# ------------------------------------------------------------------------------------------------
# Rendering
# ------------------------------------------------------------------------------------------------


def _fmt(value: float, places: int = 4) -> str:
    """Fixed decimal places, so output cannot drift between runs."""
    return f"{value:.{places}f}"


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def render_scope(result: RoadsResult) -> str:
    """The header every table has to be read against."""
    lines = [
        "### Scope of the road-distance measurement",
        "",
        f"- **Area:** {result.area_name}",
        f"- **Bounding box:** south {_fmt(result.bbox.south, 5)}, "
        f"west {_fmt(result.bbox.west, 5)}, "
        f"north {_fmt(result.bbox.north, 5)}, east {_fmt(result.bbox.east, 5)}",
        f"- **Network filter:** `{result.network_type}`",
        f"- **Graph:** {result.n_graph_nodes} nodes, {result.n_graph_edges} edges "
        f"({result.graph_source}: {result.graph_note})",
        f"- **Bins:** {result.n_bins}, snapping to {result.n_distinct_nodes} distinct road nodes",
        "",
        f"> {SYNTHETIC_BINS_NOTE}",
        "",
        f"> {OSM_CREDIT}",
        "",
        "> **NOT comparable with Phase 2.** Phase 2 is 5000 bins on a 100 x 100 km square with",
        "> Euclidean distance. This is a ~3 km area with a few hundred bins on real roads.",
        "> The Euclidean-vs-road ratio below is computed on **this** bin set under two metrics,",
        "> which is the only comparison that isolates the distance metric.",
    ]
    if not result.road_available:
        lines.insert(
            2,
            f"\n> **{FALLBACK_LABEL}.** The road graph could not be obtained "
            f"({result.graph_note}), so every ratio below is exactly 1.0 by construction. That is "
            f"the signature of the fallback, not a finding about roads.\n",
        )
    return "\n".join(lines)


def render_detour(result: RoadsResult) -> str:
    """The pairwise detour-ratio table. Spread, not just a mean."""
    label = f" [{FALLBACK_LABEL}]" if not result.road_available else ""
    title = f"### Detour ratio: road distance / straight-line distance, per bin pair{label}"
    note = (
        "A ratio of 1.0 means the road is as short as the crow flies; 1.4 means the truck drives\n"
        "40% further than the straight line. **The spread matters as much as the mean** -- one\n"
        "average would hide that some pairs are nearly straight and others double back.\n\n"
        "`aggregate` is `sum(road) / sum(straight)`, which is what a total-distance figure would\n"
        "show. It is not the same as the mean of the per-pair ratios, so both are reported.\n\n"
        "Pairs are **skipped and counted**, never silently dropped: `zero pairs` are two bins\n"
        "that snapped to the same road node, `unreachable pairs` have no road path inside the\n"
        "extract."
    )
    d = result.detour
    header = [
        "pairs",
        "zero pairs",
        "unreachable pairs",
        "mean",
        "median",
        "p10",
        "p90",
        "min",
        "max",
        "aggregate",
    ]
    rows = [
        [
            str(d.n_pairs),
            str(d.n_zero_pairs),
            str(d.n_unreachable_pairs),
            _fmt(d.mean),
            _fmt(d.median),
            _fmt(d.p10),
            _fmt(d.p90),
            _fmt(d.minimum),
            _fmt(d.maximum),
            _fmt(d.aggregate),
        ]
    ]
    return f"{title}\n\n{note}\n\n{_table(header, rows)}"


def render_tours(result: RoadsResult) -> str:
    """The same bin set, the same algorithm, two metrics."""
    label = f" [{FALLBACK_LABEL}]" if not result.road_available else ""
    title = f"### Tour length under each metric: same bins, same algorithm{label}"
    note = (
        "Both tours are built by the **identical** nearest-neighbour-then-2-opt code reading a\n"
        "distance matrix, from the same depot (the area centre, snapped to a road node). The only\n"
        "difference between the two numbers is the matrix.\n\n"
        "Unreachable legs are replaced by a large finite stand-in so 2-opt can compare swaps at\n"
        "all; that is why the **pairwise** ratio above is the primary number and this one the\n"
        "secondary one."
    )
    header = ["metric", "tour km", "ratio vs straight-line"]
    rows = [
        ["straight-line (haversine)", _fmt(result.euclidean_tour_km, 3), _fmt(1.0)],
        ["real roads (OSM shortest path)", _fmt(result.road_tour_km, 3), _fmt(result.tour_ratio)],
    ]
    return f"{title}\n\n{note}\n\n{_table(header, rows)}"


def detour_verdict(result: RoadsResult) -> list[str]:
    """How much Euclidean under-states road distance, stated plainly."""
    if not result.road_available:
        return [
            f"  {FALLBACK_LABEL}: the road graph could not be obtained ({result.graph_note}).",
            "  NO road-distance measurement was made. The ratios are 1.0 by construction.",
        ]
    d = result.detour
    lines = [
        f"  over {d.n_pairs} bin pairs on real {result.network_type} roads in {result.area_name}:",
        f"  straight-line distance under-states road distance by a factor of "
        f"{_fmt(d.median)} at the median ({_fmt(d.p10)} at p10, {_fmt(d.p90)} at p90, "
        f"{_fmt(d.minimum)} to {_fmt(d.maximum)} overall)",
        f"  on total distance the factor is {_fmt(d.aggregate)}; on a routed tour it is "
        f"{_fmt(result.tour_ratio)}",
    ]
    if d.n_unreachable_pairs:
        lines.append(
            f"  {d.n_unreachable_pairs} pairs had no road path inside the extract and were "
            f"excluded from the pairwise ratios"
        )
    if d.n_zero_pairs:
        lines.append(
            f"  {d.n_zero_pairs} pairs snapped to the same road node (ratio undefined) and were "
            f"excluded"
        )
    lines.append(
        "  This is ONE area at ONE bbox size with SYNTHETIC bin locations. It is not a "
        "city-wide figure and it does not transfer to Phase 2's 100 km grid."
    )
    return lines


def csv_rows(result: RoadsResult) -> list[dict[str, object]]:
    """One flat CSV: every reported number as a named row."""
    d = result.detour
    scope: list[tuple[str, str, object]] = [
        ("area_name", "the bounded area", result.area_name),
        ("network_type", "OSMnx network filter", result.network_type),
        ("graph_source", "cache / download / unavailable", result.graph_source),
        ("n_graph_nodes", "road graph nodes", result.n_graph_nodes),
        ("n_graph_edges", "road graph edges", result.n_graph_edges),
        ("n_bins", "synthetic bins", result.n_bins),
        ("n_distinct_nodes", "distinct snapped road nodes", result.n_distinct_nodes),
    ]
    detour: list[tuple[str, str, object]] = [
        ("n_pairs", "bin pairs scored", d.n_pairs),
        ("n_zero_pairs", "pairs skipped: same road node", d.n_zero_pairs),
        ("n_unreachable_pairs", "pairs skipped: no road path", d.n_unreachable_pairs),
        ("mean", "mean road/straight ratio", _fmt(d.mean)),
        ("median", "median road/straight ratio", _fmt(d.median)),
        ("p10", "p10 road/straight ratio", _fmt(d.p10)),
        ("p90", "p90 road/straight ratio", _fmt(d.p90)),
        ("min", "min road/straight ratio", _fmt(d.minimum)),
        ("max", "max road/straight ratio", _fmt(d.maximum)),
        ("aggregate", "sum(road)/sum(straight)", _fmt(d.aggregate)),
    ]
    tours: list[tuple[str, str, object]] = [
        (
            "euclidean_tour_km",
            "2-opt tour, straight-line matrix",
            _fmt(result.euclidean_tour_km, 3),
        ),
        ("road_tour_km", "2-opt tour, road matrix", _fmt(result.road_tour_km, 3)),
        ("tour_ratio", "road tour / straight-line tour", _fmt(result.tour_ratio)),
    ]
    rows: list[dict[str, object]] = []
    for table, items in (("scope", scope), ("detour", detour), ("tours", tours)):
        for key, label, value in items:
            rows.append({"table": table, "row_key": key, "label": label, "value": value})
    return rows


def quick_banner() -> str:
    """The smoke-run warning, prepended ONCE to a report rather than to every table."""
    return (
        "> **SMOKE RUN, NOT FOR PUBLICATION.** Smaller than the default run. Every number"
        " in `docs/PHASE4_RESULTS.md` comes from the full run only.\n"
    )
