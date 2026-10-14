"""Real road distances for one bounded Chennai area, via OSMnx.

Every distance in Phases 0-3 is a straight line between two random points in a 100 x 100 km
square. ``docs/ASSUMPTIONS.md`` admits outright that this "under-states road distance by an amount
we have not yet measured". This module measures it, on a real road network.

### Road data attribution

Road geometry comes from **OpenStreetMap**. (c) OpenStreetMap contributors, licensed **ODbL 1.0**:
https://www.openstreetmap.org/copyright

### BIN LOCATIONS ARE SYNTHETIC

The *roads* are real. The *bins* are not. :func:`sample_bins_on_graph` draws uniform-random points
inside the bounding box and snaps each to the nearest drivable road node. Nobody surveyed a bin
site in T. Nagar. Every table built from this module repeats that sentence, because a real road
network makes synthetic bins look more authoritative than they are.

### What is comparable with what

**Nothing here is comparable with Phase 2's 5000-bin, 100 km Euclidean city.** Different area,
different bin count, different distance metric. The Euclidean-vs-road ratio below is computed on
**the same bin set** under two metrics, which is the only comparison that isolates the metric; the
``% vs Phase 2`` column that would be the obvious thing to want does not exist, deliberately.

### Failure is loud

If the download fails for any reason -- no network, no osmnx, an HTTP error, an empty graph --
:func:`load_road_graph` returns ``graph=None`` after printing a banner to **stderr**, and the
caller falls back to Euclidean and labels every table ``EUCLIDEAN FALLBACK, NOT ROAD DISTANCES``.
A silent degrade here would publish straight-line numbers under a heading that says "road".

``osmnx`` and ``networkx`` are imported **lazily**, inside the functions that need them.
"""

from __future__ import annotations

import math
import random
import statistics
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: OSMnx network filters we allow in the config. ``drive`` is the one that matches a truck.
NETWORK_TYPES = frozenset({"drive", "drive_service", "all", "all_private", "bike", "walk"})

#: Mean Earth radius in km, for the haversine comparator.
EARTH_RADIUS_KM = 6371.0088

#: :attr:`RoadGraphResult.source` values.
SOURCE_CACHE = "cache"
SOURCE_DOWNLOAD = "download"
SOURCE_UNAVAILABLE = "unavailable"

#: Printed verbatim wherever road numbers appear, and in the results doc.
OSM_CREDIT = (
    "Road data (c) OpenStreetMap contributors, ODbL 1.0 -- https://www.openstreetmap.org/copyright"
)

#: Printed verbatim wherever road numbers appear, and in the results doc.
SYNTHETIC_BINS_NOTE = (
    "BIN LOCATIONS ARE SYNTHETIC: uniform-random points in the bbox, snapped to the nearest "
    "drivable road node. The roads are real; the bins are not."
)

#: The label every table carries when the road graph could not be obtained.
FALLBACK_LABEL = "EUCLIDEAN FALLBACK, NOT ROAD DISTANCES"

#: A point as (latitude, longitude) in degrees. Deliberately NOT ``tsp.Point``, which is km on a
#: flat grid -- mixing the two up is how a 3 km area turns into a 300 km one.
LatLon = tuple[float, float]


@dataclass(frozen=True)
class BoundingBox:
    """A lat/lon bounding box in degrees."""

    north: float
    south: float
    east: float
    west: float


@dataclass(frozen=True)
class RoadGraphResult:
    """The outcome of trying to obtain a road graph.

    ``graph`` is ``None`` exactly when ``source == SOURCE_UNAVAILABLE``. It is typed ``object``
    because ``networkx`` is an optional import: this module must type-check and import with the
    ``roads`` extra absent.
    """

    graph: object | None
    source: str
    note: str

    @property
    def available(self) -> bool:
        return self.graph is not None


@dataclass(frozen=True)
class DetourStats:
    """How much a straight line under-states a road distance, over many pairs.

    The **spread** matters as much as the mean: one average ratio would hide that some pairs are
    nearly straight and others double back a long way.
    """

    n_pairs: int
    #: Pairs skipped because the two bins snapped to the same node (Euclidean distance 0).
    n_zero_pairs: int
    #: Pairs skipped because no road path exists between them in the downloaded extract.
    n_unreachable_pairs: int
    mean: float
    median: float
    p10: float
    p90: float
    minimum: float
    maximum: float
    #: ``sum(road) / sum(euclidean)`` -- the ratio a total-distance figure would show, which is
    #: not the same as the mean of the per-pair ratios and is reported alongside it.
    aggregate: float


def bbox_for(center_lat: float, center_lon: float, box_km: float) -> BoundingBox:
    """A square-ish bounding box of ``box_km`` on a side, centred on a point.

    A flat-earth conversion: 1 degree of latitude is about 111.32 km everywhere, and 1 degree of
    longitude is that shrunk by ``cos(latitude)``. Over 3 km at 13 degrees north the error is far
    below the size of a city block, and it keeps this function pure stdlib and testable with no
    network. Pure stdlib is the point -- the bbox is the one part of this module that must work
    without the extra.
    """
    if box_km <= 0:
        raise ValueError(f"box_km must be > 0, got {box_km}")
    if not -90.0 <= center_lat <= 90.0:
        raise ValueError(f"center_lat must be in [-90, 90], got {center_lat}")

    km_per_degree_lat = 111.32
    half = box_km / 2.0
    d_lat = half / km_per_degree_lat
    cos_lat = math.cos(math.radians(center_lat))
    # At a pole the longitude span is undefined; clamp rather than divide by ~0.
    d_lon = half / (km_per_degree_lat * cos_lat) if abs(cos_lat) > 1e-9 else 180.0

    return BoundingBox(
        north=min(90.0, center_lat + d_lat),
        south=max(-90.0, center_lat - d_lat),
        east=min(180.0, center_lon + d_lon),
        west=max(-180.0, center_lon - d_lon),
    )


def haversine_km(a: LatLon, b: LatLon) -> float:
    """Great-circle distance in km between two (lat, lon) points.

    This is the Euclidean comparator: the straight line **on the same sphere** the road network
    lives on. Comparing road metres against a distance computed in degrees would measure the
    projection as well as the detour.
    """
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    d_lat = lat2 - lat1
    d_lon = lon2 - lon1
    h = math.sin(d_lat / 2.0) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(d_lon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * math.asin(math.sqrt(min(1.0, h)))


def euclidean_matrix(points: Sequence[LatLon]) -> list[list[float]]:
    """Full haversine distance matrix in km. Symmetric by construction."""
    n = len(points)
    matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = haversine_km(points[i], points[j])
            matrix[i][j] = d
            matrix[j][i] = d
    return matrix


def graph_cache_path(
    cache_dir: Path,
    subdir: str,
    bbox: BoundingBox,
    network_type: str,
) -> Path:
    """Where the downloaded graph is cached. Deterministic for a given bbox and filter.

    The bbox is rounded to 5 decimal places (about a metre) in the filename, so a re-run with the
    same config hits the cache rather than re-downloading.
    """
    name = (
        f"roads_{bbox.south:.5f}_{bbox.west:.5f}_{bbox.north:.5f}_{bbox.east:.5f}"
        f"_{network_type}.graphml"
    )
    return cache_dir / subdir / name


def load_road_graph(
    bbox: BoundingBox,
    network_type: str,
    cache_path: Path,
    *,
    allow_download: bool = True,
) -> RoadGraphResult:
    """Load the road graph from cache, else download it, else fail **loudly**.

    Args:
        bbox: the area to fetch.
        network_type: an OSMnx network filter, e.g. ``"drive"``.
        cache_path: graphml file to read from and write to.
        allow_download: ``False`` skips the network entirely. Used to exercise the fallback path
            and to keep tests off the network.

    Returns a :class:`RoadGraphResult` whose ``graph`` is ``None`` on failure. Never raises for a
    missing network or a missing package -- the caller is expected to fall back to Euclidean and
    say so in its output, which is only possible if this returns rather than explodes.
    """
    try:
        import osmnx as ox
    except ImportError as exc:
        return _unavailable(f"osmnx is not installed ({exc})")

    if cache_path.is_file():
        try:
            graph = ox.load_graphml(cache_path)
        except Exception as exc:  # noqa: BLE001 - a corrupt cache must not be fatal
            _loud(f"cached graph at {cache_path} could not be read ({exc}); re-downloading")
        else:
            return RoadGraphResult(
                graph=graph,
                source=SOURCE_CACHE,
                note=f"loaded {cache_path.name} from cache (no network access)",
            )

    if not allow_download:
        return _unavailable(f"no cached graph at {cache_path} and downloads are disabled")

    try:
        graph = _download_graph(ox, bbox, network_type)
    except Exception as exc:  # noqa: BLE001 - any download failure falls back, loudly
        return _unavailable(f"download failed ({type(exc).__name__}: {exc})")

    if graph is None or graph.number_of_nodes() == 0:
        return _unavailable("the downloaded graph has no nodes")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        ox.save_graphml(graph, cache_path)
    except Exception as exc:  # noqa: BLE001 - a failed cache write is not a failed run
        _loud(f"could not cache the graph to {cache_path} ({exc}); continuing without a cache")

    return RoadGraphResult(
        graph=graph,
        source=SOURCE_DOWNLOAD,
        note=f"downloaded from OpenStreetMap and cached to {cache_path.name}",
    )


def sample_bins_on_graph(
    graph: object,
    bbox: BoundingBox,
    n_bins: int,
    rng: random.Random,
) -> tuple[list[int], list[LatLon]]:
    """Draw ``n_bins`` **synthetic** bins in the bbox and snap each to the nearest road node.

    Returns the snapped node ids and the snapped nodes' own (lat, lon) -- not the random draw's.
    Both matrices are then built over the *same* snapped coordinates, so the Euclidean comparator
    measures straight lines between the places the trucks actually stop.
    """
    import osmnx as ox

    lats: list[float] = []
    lons: list[float] = []
    for _ in range(n_bins):
        # Both draws happen for every bin, in this order, so the sample is a pure function of
        # (seed, n_bins) and a --quick run is a prefix of the full one.
        lats.append(rng.uniform(bbox.south, bbox.north))
        lons.append(rng.uniform(bbox.west, bbox.east))

    nodes = ox.distance.nearest_nodes(graph, X=lons, Y=lats)
    node_list = [int(node) for node in nodes]

    graph_nodes = graph.nodes  # type: ignore[attr-defined]
    snapped: list[LatLon] = [
        (float(graph_nodes[node]["y"]), float(graph_nodes[node]["x"])) for node in node_list
    ]
    return node_list, snapped


def road_distance_matrix(graph: object, nodes: Sequence[int]) -> list[list[float]]:
    """Shortest-path road distance matrix in km, by one Dijkstra per source node.

    Edge weight is OSM's ``length`` attribute, in metres. Pairs with no path through the
    downloaded extract come back as ``math.inf`` -- recorded, counted and reported, never quietly
    replaced with a straight line. An extract is a cut-out of a larger network, so a few
    unreachable pairs near its edge are expected.
    """
    import networkx as nx

    n = len(nodes)
    matrix = [[math.inf] * n for _ in range(n)]
    index_of: dict[int, list[int]] = {}
    for position, node in enumerate(nodes):
        # Two bins can snap to the same node, so one node may map to several matrix rows.
        index_of.setdefault(node, []).append(position)

    for node, positions in index_of.items():
        lengths = nx.single_source_dijkstra_path_length(graph, node, weight="length")
        row = [float(lengths[other]) / 1000.0 if other in lengths else math.inf for other in nodes]
        for position in positions:
            matrix[position] = list(row)
        for position in positions:
            matrix[position][position] = 0.0

    return matrix


def detour_ratio_stats(
    euclidean: Sequence[Sequence[float]],
    road: Sequence[Sequence[float]],
) -> DetourStats:
    """Per-pair ``road / euclidean`` ratios, with spread, over the upper triangle.

    Skipped and **counted**, not dropped silently:

    - pairs whose Euclidean distance is 0 (two bins snapped to the same road node), because the
      ratio is undefined;
    - pairs with no road path, because ``inf / d`` is not a detour.
    """
    n = len(euclidean)
    if len(road) != n:
        raise ValueError(f"matrices must be the same size, got {n} and {len(road)}")

    ratios: list[float] = []
    sum_euclid = 0.0
    sum_road = 0.0
    zero_pairs = 0
    unreachable = 0

    for i in range(n):
        for j in range(i + 1, n):
            straight = euclidean[i][j]
            driven = road[i][j]
            if not math.isfinite(driven):
                unreachable += 1
                continue
            if straight <= 0.0:
                zero_pairs += 1
                continue
            ratios.append(driven / straight)
            sum_euclid += straight
            sum_road += driven

    if not ratios:
        return DetourStats(
            n_pairs=0,
            n_zero_pairs=zero_pairs,
            n_unreachable_pairs=unreachable,
            mean=math.nan,
            median=math.nan,
            p10=math.nan,
            p90=math.nan,
            minimum=math.nan,
            maximum=math.nan,
            aggregate=math.nan,
        )

    ordered = sorted(ratios)
    return DetourStats(
        n_pairs=len(ratios),
        n_zero_pairs=zero_pairs,
        n_unreachable_pairs=unreachable,
        mean=statistics.fmean(ordered),
        median=statistics.median(ordered),
        p10=_percentile(ordered, 0.10),
        p90=_percentile(ordered, 0.90),
        minimum=ordered[0],
        maximum=ordered[-1],
        aggregate=sum_road / sum_euclid if sum_euclid > 0 else math.nan,
    )


def _percentile(ordered: Sequence[float], fraction: float) -> float:
    """Nearest-rank percentile of an already-sorted sequence.

    Nearest-rank rather than interpolated, so the reported value is always a ratio that really
    occurred in the data.
    """
    if not ordered:
        return math.nan
    position = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[position]


def _download_graph(
    ox: Any,  # noqa: ANN401 - osmnx is untyped, so Any is the honest annotation here
    bbox: BoundingBox,
    network_type: str,
) -> Any:  # noqa: ANN401 - likewise for the graph it returns
    """Call OSMnx's bbox graph builder across its v1 and v2 signatures.

    OSMnx 2.x takes a single ``bbox=(west, south, east, north)`` tuple; 1.x took four keyword
    arguments. The config pins ``>=1.9``, which spans the change, so both are tried rather than
    pinning the project to one minor version.
    """
    graph_from_bbox = ox.graph_from_bbox
    try:
        return graph_from_bbox(
            bbox=(bbox.west, bbox.south, bbox.east, bbox.north),
            network_type=network_type,
        )
    except TypeError:
        return graph_from_bbox(
            north=bbox.north,
            south=bbox.south,
            east=bbox.east,
            west=bbox.west,
            network_type=network_type,
        )


def _unavailable(reason: str) -> RoadGraphResult:
    """Announce the fallback loudly and return an unavailable result."""
    _loud(f"ROAD GRAPH UNAVAILABLE -- FALLING BACK TO EUCLIDEAN ({reason})")
    return RoadGraphResult(graph=None, source=SOURCE_UNAVAILABLE, note=reason)


def _loud(message: str) -> None:
    """Print a banner to stderr.

    stderr, not stdout: the runners' stdout must stay byte-identical between same-seed runs, and
    a message that depends on whether a download happened to succeed is exactly the kind of thing
    that would break that.
    """
    print("=" * 96, file=sys.stderr)
    print(f"ROADS: {message}", file=sys.stderr)
    print("=" * 96, file=sys.stderr)
