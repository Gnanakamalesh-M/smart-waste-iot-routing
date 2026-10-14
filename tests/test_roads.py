"""Road-distance plumbing. **No test in this file touches the network.**

That is the hard rule: a test suite that downloads a map is a test suite that fails on a train.
Everything here runs against either pure stdlib maths or a **tiny hand-built graph** -- six nodes
in a square with one edge deliberately missing, so the detour it forces is known in advance and
can be asserted exactly.

The `networkx` tests are guarded with `pytest.importorskip`, so the `routing` chunk stays green
whether or not the optional `roads` extra is installed.
"""

import math
from pathlib import Path
from typing import Any

import pytest

from smartwaste.routing.roads import (
    FALLBACK_LABEL,
    SOURCE_UNAVAILABLE,
    BoundingBox,
    bbox_for,
    detour_ratio_stats,
    euclidean_matrix,
    graph_cache_path,
    haversine_km,
    load_road_graph,
    road_distance_matrix,
)

pytestmark = pytest.mark.routing

#: Roughly T. Nagar, Chennai -- the config's area.
TNAGAR_LAT = 13.0418
TNAGAR_LON = 80.2341


# --------------------------------------------------------------------------------------------
# bbox_for and haversine: pure stdlib, no extra needed
# --------------------------------------------------------------------------------------------


def test_bbox_is_centred_on_the_point() -> None:
    bbox = bbox_for(TNAGAR_LAT, TNAGAR_LON, 3.0)
    assert (bbox.north + bbox.south) / 2.0 == pytest.approx(TNAGAR_LAT)
    assert (bbox.east + bbox.west) / 2.0 == pytest.approx(TNAGAR_LON)


def test_a_3km_box_really_spans_about_3km() -> None:
    """The reason the flat-earth conversion is acceptable at this scale, measured."""
    bbox = bbox_for(TNAGAR_LAT, TNAGAR_LON, 3.0)
    height = haversine_km((bbox.south, TNAGAR_LON), (bbox.north, TNAGAR_LON))
    width = haversine_km((TNAGAR_LAT, bbox.west), (TNAGAR_LAT, bbox.east))
    assert height == pytest.approx(3.0, rel=0.01)
    assert width == pytest.approx(3.0, rel=0.01)


def test_a_bigger_box_is_bigger() -> None:
    small = bbox_for(TNAGAR_LAT, TNAGAR_LON, 1.0)
    large = bbox_for(TNAGAR_LAT, TNAGAR_LON, 10.0)
    assert large.north - large.south > small.north - small.south
    assert large.east - large.west > small.east - small.west


def test_longitude_span_widens_towards_the_equator() -> None:
    """A degree of longitude is longer near the equator, so the same km needs fewer degrees."""
    at_equator = bbox_for(0.0, 0.0, 10.0)
    far_north = bbox_for(60.0, 0.0, 10.0)
    assert (far_north.east - far_north.west) > (at_equator.east - at_equator.west)


def test_bbox_rejects_nonsense() -> None:
    with pytest.raises(ValueError, match="box_km must be > 0"):
        bbox_for(TNAGAR_LAT, TNAGAR_LON, 0.0)
    with pytest.raises(ValueError, match=r"center_lat must be in \[-90, 90\]"):
        bbox_for(91.0, TNAGAR_LON, 3.0)


def test_haversine_is_zero_for_a_point_against_itself() -> None:
    assert haversine_km((TNAGAR_LAT, TNAGAR_LON), (TNAGAR_LAT, TNAGAR_LON)) == pytest.approx(0.0)


def test_haversine_is_symmetric() -> None:
    a = (13.0, 80.2)
    b = (13.05, 80.25)
    assert haversine_km(a, b) == pytest.approx(haversine_km(b, a))


def test_one_degree_of_latitude_is_about_111km() -> None:
    assert haversine_km((0.0, 0.0), (1.0, 0.0)) == pytest.approx(111.19, rel=0.001)


def test_euclidean_matrix_is_symmetric_with_a_zero_diagonal() -> None:
    points = [(13.0, 80.2), (13.01, 80.21), (13.02, 80.19)]
    matrix = euclidean_matrix(points)
    for i in range(len(points)):
        assert matrix[i][i] == pytest.approx(0.0)
        for j in range(len(points)):
            assert matrix[i][j] == pytest.approx(matrix[j][i])


# --------------------------------------------------------------------------------------------
# detour_ratio_stats: fed hand-written matrices, so every number is checkable by hand
# --------------------------------------------------------------------------------------------


def test_ratios_are_road_over_straight_line() -> None:
    euclid = [[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]]
    road = [[0.0, 2.0, 2.0], [2.0, 0.0, 3.0], [2.0, 3.0, 0.0]]
    stats = detour_ratio_stats(euclid, road)
    # Upper triangle only: (0,1) = 2/1, (0,2) = 2/2, (1,2) = 3/1.
    assert stats.n_pairs == 3
    assert stats.minimum == pytest.approx(1.0)
    assert stats.maximum == pytest.approx(3.0)
    assert stats.median == pytest.approx(2.0)
    assert stats.mean == pytest.approx((2.0 + 1.0 + 3.0) / 3.0)


def test_the_aggregate_is_not_the_mean_of_the_ratios() -> None:
    """Both are reported precisely because they differ, so the test pins that they do."""
    euclid = [[0.0, 1.0], [1.0, 0.0]]
    road = [[0.0, 4.0], [4.0, 0.0]]
    stats = detour_ratio_stats(euclid, road)
    assert stats.aggregate == pytest.approx(4.0)

    euclid2 = [[0.0, 1.0, 10.0], [1.0, 0.0, 10.0], [10.0, 10.0, 0.0]]
    road2 = [[0.0, 4.0, 10.0], [4.0, 0.0, 10.0], [10.0, 10.0, 0.0]]
    stats2 = detour_ratio_stats(euclid2, road2)
    # mean of (4.0, 1.0, 1.0) = 2.0, but sum(road)/sum(euclid) = 24/21 = 1.142...
    assert stats2.mean == pytest.approx(2.0)
    assert stats2.aggregate == pytest.approx(24.0 / 21.0)
    assert stats2.mean != pytest.approx(stats2.aggregate)


def test_unreachable_pairs_are_counted_not_dropped_silently() -> None:
    euclid = [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]
    road = [[0.0, 2.0, math.inf], [2.0, 0.0, 1.5], [math.inf, 1.5, 0.0]]
    stats = detour_ratio_stats(euclid, road)
    assert stats.n_unreachable_pairs == 1
    assert stats.n_pairs == 2


def test_zero_distance_pairs_are_counted_not_dropped_silently() -> None:
    """Two bins can snap to the same road node, which makes the ratio undefined."""
    euclid = [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0]]
    road = [[0.0, 0.0, 2.0], [0.0, 0.0, 2.0], [2.0, 2.0, 0.0]]
    stats = detour_ratio_stats(euclid, road)
    assert stats.n_zero_pairs == 1
    assert stats.n_pairs == 2


def test_no_usable_pairs_gives_nan_rather_than_a_made_up_number() -> None:
    stats = detour_ratio_stats([[0.0]], [[0.0]])
    assert stats.n_pairs == 0
    assert math.isnan(stats.mean)
    assert math.isnan(stats.aggregate)


def test_identical_matrices_give_a_ratio_of_exactly_one() -> None:
    """This is the signature of the Euclidean fallback, and the docs label it as such."""
    euclid = [[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 3.0, 0.0]]
    stats = detour_ratio_stats(euclid, euclid)
    assert stats.mean == pytest.approx(1.0)
    assert stats.aggregate == pytest.approx(1.0)
    assert stats.minimum == pytest.approx(1.0)
    assert stats.maximum == pytest.approx(1.0)


def test_mismatched_matrix_sizes_are_rejected() -> None:
    with pytest.raises(ValueError, match="must be the same size"):
        detour_ratio_stats([[0.0, 1.0], [1.0, 0.0]], [[0.0]])


def test_percentiles_are_real_observed_ratios() -> None:
    """Nearest-rank, so p10 and p90 are always values that actually occurred."""
    euclid = [[0.0] * 5 for _ in range(5)]
    road = [[0.0] * 5 for _ in range(5)]
    wanted = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 3.0]
    pairs = [(i, j) for i in range(5) for j in range(i + 1, 5)]
    for (i, j), ratio in zip(pairs, wanted, strict=True):
        euclid[i][j] = euclid[j][i] = 1.0
        road[i][j] = road[j][i] = ratio
    stats = detour_ratio_stats(euclid, road)
    assert stats.n_pairs == 10
    assert stats.p10 in wanted
    assert stats.p90 in wanted
    assert stats.p10 <= stats.median <= stats.p90


# --------------------------------------------------------------------------------------------
# graph_cache_path: deterministic, and inside the gitignored cache dir
# --------------------------------------------------------------------------------------------


def test_the_cache_path_is_deterministic() -> None:
    bbox = bbox_for(TNAGAR_LAT, TNAGAR_LON, 3.0)
    first = graph_cache_path(Path("cache"), "osmnx", bbox, "drive")
    second = graph_cache_path(Path("cache"), "osmnx", bbox, "drive")
    assert first == second
    assert first.suffix == ".graphml"
    assert first.parent == Path("cache") / "osmnx"


def test_a_different_area_or_filter_is_a_different_cache_file() -> None:
    small = bbox_for(TNAGAR_LAT, TNAGAR_LON, 3.0)
    large = bbox_for(TNAGAR_LAT, TNAGAR_LON, 5.0)
    assert graph_cache_path(Path("cache"), "osmnx", small, "drive") != graph_cache_path(
        Path("cache"), "osmnx", large, "drive"
    )
    assert graph_cache_path(Path("cache"), "osmnx", small, "drive") != graph_cache_path(
        Path("cache"), "osmnx", small, "walk"
    )


# --------------------------------------------------------------------------------------------
# The loud fallback. No network: downloads disabled and no cache file.
# --------------------------------------------------------------------------------------------


def test_the_fallback_is_loud_and_returns_no_graph(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bbox = bbox_for(TNAGAR_LAT, TNAGAR_LON, 3.0)
    missing = tmp_path / "osmnx" / "not_downloaded.graphml"
    result = load_road_graph(bbox, "drive", missing, allow_download=False)

    assert result.graph is None
    assert result.available is False
    assert result.source == SOURCE_UNAVAILABLE
    assert result.note

    captured = capsys.readouterr()
    # The banner goes to stderr, so stdout stays byte-identical between same-seed runs.
    assert "ROADS:" in captured.err
    assert "FALLING BACK TO EUCLIDEAN" in captured.err
    assert captured.out == ""


def test_the_fallback_label_is_shouted_not_whispered() -> None:
    """A guard on the constant itself: the label must stay unmissable in a table heading."""
    assert FALLBACK_LABEL.upper() == FALLBACK_LABEL
    assert "EUCLIDEAN" in FALLBACK_LABEL
    assert "NOT ROAD" in FALLBACK_LABEL


# --------------------------------------------------------------------------------------------
# road_distance_matrix against a tiny hand-built graph. Still no network.
# --------------------------------------------------------------------------------------------


@pytest.fixture
def square_with_a_missing_edge() -> Any:
    """Four corners of a 1 km square, with the 1-2 edge removed.

    Nodes 1 and 2 are 1 km apart in a straight line, but with that edge gone the only road
    between them goes the long way round, 1 -> 0 -> 3 -> 2, which is 3 km. So the detour ratio
    for that one pair is exactly 3.0 and can be asserted rather than approximated.

        0 ---- 1
        |      x   (x = the missing 1-2 edge)
        3 ---- 2
    """
    nx = pytest.importorskip("networkx", reason="the optional 'roads' extra is not installed")

    #: One km in degrees of latitude, so the hand-built geometry matches its edge lengths.
    km = 1.0 / 111.32
    graph = nx.MultiDiGraph()
    coords = {
        0: (13.00 + km, 80.00),
        1: (13.00 + km, 80.00 + km),
        2: (13.00, 80.00 + km),
        3: (13.00, 80.00),
    }
    for node, (lat, lon) in coords.items():
        graph.add_node(node, y=lat, x=lon)
    # length is in metres, as OSM stores it. Edges both ways, so the graph is strongly connected.
    for a, b in ((0, 1), (0, 3), (3, 2)):
        graph.add_edge(a, b, length=1000.0)
        graph.add_edge(b, a, length=1000.0)
    return graph


def test_the_missing_edge_forces_a_known_three_fold_detour(square_with_a_missing_edge: Any) -> None:
    graph = square_with_a_missing_edge
    nodes = [0, 1, 2, 3]
    road = road_distance_matrix(graph, nodes)

    # The long way round: 1 -> 0 -> 3 -> 2.
    assert road[1][2] == pytest.approx(3.0)
    assert road[2][1] == pytest.approx(3.0)
    # The edges that do exist are 1 km.
    assert road[0][1] == pytest.approx(1.0)
    assert road[0][3] == pytest.approx(1.0)
    assert road[3][2] == pytest.approx(1.0)


def test_the_detour_ratio_of_the_missing_edge_pair_is_three(
    square_with_a_missing_edge: Any,
) -> None:
    """The whole measurement, end to end, on a graph whose answer is known by hand."""
    graph = square_with_a_missing_edge
    nodes = [0, 1, 2, 3]
    points = [(graph.nodes[n]["y"], graph.nodes[n]["x"]) for n in nodes]

    euclid = euclidean_matrix(points)
    road = road_distance_matrix(graph, nodes)
    stats = detour_ratio_stats(euclid, road)

    assert stats.n_pairs == 6
    assert stats.n_unreachable_pairs == 0
    assert stats.maximum == pytest.approx(3.0, rel=0.01)
    # Straight-line never beats the road, so no ratio can be below 1.
    assert stats.minimum >= 1.0 - 1e-9


def test_the_road_matrix_diagonal_is_zero(square_with_a_missing_edge: Any) -> None:
    road = road_distance_matrix(square_with_a_missing_edge, [0, 1, 2, 3])
    for i in range(4):
        assert road[i][i] == pytest.approx(0.0)


def test_two_bins_on_the_same_node_get_a_zero_road_distance(
    square_with_a_missing_edge: Any,
) -> None:
    """Snapping collisions are real, so the matrix has to handle a repeated node."""
    road = road_distance_matrix(square_with_a_missing_edge, [0, 1, 1, 2])
    assert road[1][2] == pytest.approx(0.0)
    assert road[2][1] == pytest.approx(0.0)
    assert road[1][3] == pytest.approx(road[2][3])


def test_an_unreachable_node_comes_back_as_infinity(square_with_a_missing_edge: Any) -> None:
    """An extract is a cut-out, so some pairs genuinely have no path. That must not be hidden."""
    nx = pytest.importorskip("networkx")
    graph = nx.MultiDiGraph(square_with_a_missing_edge)
    graph.add_node(99, y=13.5, x=80.5)  # an island with no edges at all

    road = road_distance_matrix(graph, [0, 99])
    assert math.isinf(road[0][1])
    assert math.isinf(road[1][0])


def test_a_hand_built_bounding_box_round_trips_through_the_cache_name() -> None:
    bbox = BoundingBox(north=13.1, south=13.0, east=80.3, west=80.2)
    path = graph_cache_path(Path("cache"), "osmnx", bbox, "drive")
    assert "13.00000" in path.name
    assert "80.20000" in path.name
    assert "drive" in path.name
