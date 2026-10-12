"""Routing correctness: the properties a route must satisfy, not just the numbers it produces."""

import math
import random

import pytest

from smartwaste.routing.tsp import (
    DEPOT_INDEX,
    RouteCache,
    distance,
    nearest_neighbour,
    route_distance,
    solve,
    two_opt,
)

pytestmark = pytest.mark.routing

#: (depot, locations, cache) handed to the RouteCache tests.
CacheFixture = tuple[tuple[float, float], list[tuple[float, float]], RouteCache]

#: A unit square with the depot at the origin. Optimal closed tour is the perimeter, length 4.
SQUARE = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]


def _random_points(n: int, seed: int, span: float = 25.0) -> list[tuple[float, float]]:
    rng = random.Random(seed)
    return [(rng.uniform(0, span), rng.uniform(0, span)) for _ in range(n)]


def test_distance_is_euclidean() -> None:
    assert distance((0.0, 0.0), (3.0, 4.0)) == pytest.approx(5.0)


def test_route_distance_closes_the_loop() -> None:
    """The perimeter of the unit square is 4, which requires counting the leg back to start."""
    assert route_distance(SQUARE, [0, 1, 2, 3]) == pytest.approx(4.0)


def test_route_distance_of_a_non_route_is_zero() -> None:
    assert route_distance(SQUARE, []) == 0.0
    assert route_distance(SQUARE, [0]) == 0.0


@pytest.mark.parametrize("n", [2, 3, 5, 20, 60])
def test_every_point_is_visited_exactly_once(n: int) -> None:
    points = _random_points(n, seed=n)
    tour = nearest_neighbour(points)
    assert sorted(tour) == list(range(n))
    assert len(tour) == len(set(tour)) == n


@pytest.mark.parametrize("n", [4, 5, 20, 60])
def test_two_opt_keeps_every_point_exactly_once(n: int) -> None:
    points = _random_points(n, seed=100 + n)
    improved = two_opt(points, nearest_neighbour(points), max_iters=1000)
    assert sorted(improved) == list(range(n))


@pytest.mark.parametrize("n", [2, 3, 5, 20, 60])
def test_route_starts_and_ends_at_the_depot(n: int) -> None:
    """A closed tour beginning at the depot ends there too, by returning on the final leg."""
    points = _random_points(n, seed=200 + n)
    tour = two_opt(points, nearest_neighbour(points), max_iters=1000)

    assert tour[0] == DEPOT_INDEX
    # The tour is closed, so the step after the last node is the depot again.
    assert tour[(len(tour) - 1 + 1) % len(tour)] == DEPOT_INDEX


@pytest.mark.parametrize("n", [4, 5, 8, 20, 40, 80])
def test_two_opt_is_never_longer_than_nearest_neighbour(n: int) -> None:
    """The core guarantee: 2-opt only accepts strict improvements, so it cannot regress."""
    points = _random_points(n, seed=300 + n)
    nn_tour = nearest_neighbour(points)
    nn_length = route_distance(points, nn_tour)

    improved_length = route_distance(points, two_opt(points, nn_tour, max_iters=1000))

    assert improved_length <= nn_length + 1e-9


def test_two_opt_actually_improves_a_known_bad_tour() -> None:
    """The crossed diagonal tour of a square is longer than the perimeter; 2-opt must fix it."""
    crossed = [0, 2, 1, 3]
    crossed_length = route_distance(SQUARE, crossed)
    assert crossed_length > 4.0  # 2 + 2*sqrt(2)

    improved = two_opt(SQUARE, crossed, max_iters=1000)

    assert route_distance(SQUARE, improved) == pytest.approx(4.0)
    assert improved[0] == DEPOT_INDEX


def test_hand_checked_collinear_case() -> None:
    """Depot at 0, bins at 1, 2, 3 on a line: the only sensible tour is out and back, length 6."""
    points = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)]
    _, length = solve(points, max_iters=1000)
    assert length == pytest.approx(6.0)


def test_two_opt_below_four_points_is_a_no_op() -> None:
    """With fewer than 4 nodes there is no pair of non-adjacent edges to exchange."""
    points = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
    tour = [0, 1, 2]
    assert two_opt(points, tour, max_iters=1000) == tour


def test_two_opt_respects_the_iteration_cap() -> None:
    """A zero cap must return the input tour unchanged rather than loop or error."""
    points = _random_points(30, seed=7)
    nn_tour = nearest_neighbour(points)
    assert two_opt(points, nn_tour, max_iters=0) == nn_tour


def test_nearest_neighbour_of_empty_is_empty() -> None:
    assert nearest_neighbour([]) == []


def test_solve_is_deterministic() -> None:
    points = _random_points(40, seed=11)
    first = solve(points, max_iters=1000)
    second = solve(points, max_iters=1000)
    assert first == second


# --------------------------------------------------------------------------------------------
# RouteCache
# --------------------------------------------------------------------------------------------


@pytest.fixture
def cache_fixture() -> CacheFixture:
    locations = _random_points(40, seed=99)
    depot = (12.5, 12.5)
    return depot, locations, RouteCache(depot, locations, max_iters=1000)


def test_memoised_distance_equals_a_fresh_recompute(cache_fixture: CacheFixture) -> None:
    """The cache must be a pure memo: a hit has to equal solving the same selection again.

    This is what makes the (a)/(b) route reuse safe. If the key were wrong -- missing the cluster
    id, say -- a hit could return another cluster's tour and every published distance would be
    quietly wrong.
    """
    _depot, _locations, cache = cache_fixture
    selected = [3, 7, 11, 19, 25, 31]

    first = cache.distance_for(cluster_id=0, selected=selected)
    assert cache.misses == 1 and cache.hits == 0

    again = cache.distance_for(cluster_id=0, selected=selected)
    assert cache.hits == 1

    assert again == first
    assert again == pytest.approx(cache.solve_fresh(selected))


def test_cache_is_insensitive_to_selection_order(cache_fixture: CacheFixture) -> None:
    """The same set in a different order is the same route, from cache or fresh."""
    _depot, _locations, cache = cache_fixture
    ascending = [2, 5, 9, 14]
    shuffled = [14, 2, 9, 5]

    first = cache.distance_for(cluster_id=1, selected=ascending)
    second = cache.distance_for(cluster_id=1, selected=shuffled)

    assert second == first
    assert cache.hits == 1  # the shuffled call hit, so the key is order-independent
    assert cache.solve_fresh(shuffled) == pytest.approx(cache.solve_fresh(ascending))


def test_cache_separates_clusters(cache_fixture: CacheFixture) -> None:
    """Two clusters holding different bins must not share a cache entry."""
    _depot, _locations, cache = cache_fixture

    a = cache.distance_for(cluster_id=0, selected=[1, 2, 3])
    b = cache.distance_for(cluster_id=1, selected=[30, 31, 32])

    assert cache.misses == 2
    assert a != b
    assert b == pytest.approx(cache.solve_fresh([30, 31, 32]))


def test_same_bins_in_different_clusters_do_not_collide(cache_fixture: CacheFixture) -> None:
    """Same bin set, different cluster id: still two entries, and both correct.

    Cannot happen in the simulation (clusters are disjoint), but the key must be sound anyway.
    """
    _depot, _locations, cache = cache_fixture
    selected = [4, 8, 12]

    first = cache.distance_for(cluster_id=0, selected=selected)
    second = cache.distance_for(cluster_id=5, selected=selected)

    assert cache.misses == 2
    assert first == second == pytest.approx(cache.solve_fresh(selected))


def test_empty_selection_costs_nothing(cache_fixture: CacheFixture) -> None:
    """A cluster with no selected bin parks its truck: zero distance, no tour."""
    _depot, _locations, cache = cache_fixture
    assert cache.distance_for(cluster_id=0, selected=[]) == 0.0
    assert cache.solve_fresh([]) == 0.0


def test_single_bin_route_is_there_and_back(cache_fixture: CacheFixture) -> None:
    depot, locations, cache = cache_fixture
    got = cache.distance_for(cluster_id=0, selected=[6])
    assert got == pytest.approx(2.0 * math.dist(depot, locations[6]))
