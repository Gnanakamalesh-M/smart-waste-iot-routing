"""Depot-constrained TSP: nearest-neighbour construction, then 2-opt improvement.

The paper solves "each truck starts and ends at a depot and visits only the RED bins in its
cluster" as a TSP with Euclidean distance (SPEC section 2). That is what this module does, with
no road network and no capacity limit -- both are Phase 4.

A tour is a list of point indices that starts at the depot, visits every selected bin exactly
once and returns to the depot. The depot is index 0 by convention inside this module, so callers
pass ``[depot, bin1, bin2, ...]``.

2-opt accepts a swap only when it strictly shortens the tour, so the improved tour can never be
longer than the nearest-neighbour tour it started from. ``tests/test_tsp.py`` pins that.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

#: A point on the 2D grid, in km.
Point = tuple[float, float]

DEPOT_INDEX = 0

#: Only a strictly larger gain than this counts as an improvement. Without it, float noise lets
#: 2-opt "improve" a tour by ~1e-16 forever and the iteration cap becomes the only stop.
_MIN_GAIN = 1e-12


def distance(a: Point, b: Point) -> float:
    """Euclidean distance between two grid points, in km."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def route_distance(points: Sequence[Point], tour: Sequence[int]) -> float:
    """Total length of a closed tour, including the leg back to the start.

    An empty tour, or a tour of just the depot, has length 0: a truck that does not roll
    travels nothing.
    """
    if len(tour) < 2:
        return 0.0
    total = 0.0
    for position in range(len(tour)):
        here = points[tour[position]]
        following = points[tour[(position + 1) % len(tour)]]
        total += distance(here, following)
    return total


def nearest_neighbour(points: Sequence[Point], start: int = DEPOT_INDEX) -> list[int]:
    """Greedy nearest-neighbour tour over every point, beginning at ``start``.

    Ties break to the lowest index, so the tour is deterministic for a given input order
    rather than dependent on set iteration order.
    """
    n = len(points)
    if n == 0:
        return []
    unvisited = [i for i in range(n) if i != start]
    tour = [start]
    current = start
    while unvisited:
        best_position = 0
        best_cost = distance(points[current], points[unvisited[0]])
        for position in range(1, len(unvisited)):
            cost = distance(points[current], points[unvisited[position]])
            if cost < best_cost:
                best_cost = cost
                best_position = position
        current = unvisited.pop(best_position)
        tour.append(current)
    return tour


def two_opt(points: Sequence[Point], tour: Sequence[int], max_iters: int) -> list[int]:
    """Improve a closed tour by reversing segments, keeping the depot pinned at position 0.

    The classic 2-opt move: for edges ``(t[i-1], t[i])`` and ``(t[k], t[k+1])``, reversing the
    segment ``t[i..k]`` replaces them with ``(t[i-1], t[k])`` and ``(t[i], t[k+1])``. Only the
    four endpoints change, so the gain is computed from four distances rather than by re-measuring
    the whole tour.

    ``i`` starts at 1, never 0, which is what keeps the depot at the start of the tour.

    Args:
        points: coordinates indexed by the values in ``tour``.
        tour: starting tour, typically from :func:`nearest_neighbour`.
        max_iters: cap on improvement sweeps, a runtime guard. Reaching it returns the best tour
            found so far, which is still no longer than the input.

    Returns:
        A new tour list. Never longer than ``tour`` in total distance.
    """
    improved = list(tour)
    n = len(improved)
    if n < 4:
        # Fewer than 4 points has no pair of non-adjacent edges to exchange.
        return improved

    for _ in range(max_iters):
        found = False
        for i in range(1, n - 1):
            previous = points[improved[i - 1]]
            here = points[improved[i]]
            for k in range(i + 1, n):
                there = points[improved[k]]
                following = points[improved[(k + 1) % n]]
                gain = (distance(previous, here) + distance(there, following)) - (
                    distance(previous, there) + distance(here, following)
                )
                if gain > _MIN_GAIN:
                    improved[i : k + 1] = reversed(improved[i : k + 1])
                    found = True
                    here = points[improved[i]]
        if not found:
            break
    return improved


def solve(points: Sequence[Point], max_iters: int) -> tuple[list[int], float]:
    """Build a nearest-neighbour tour over ``points`` and 2-opt it.

    ``points[0]`` must be the depot. Returns the tour and its closed length.
    """
    tour = nearest_neighbour(points)
    tour = two_opt(points, tour, max_iters)
    return tour, route_distance(points, tour)


class RouteCache:
    """Memoises solved tours so a repeated selection is not re-solved.

    Strategy (a) visits every bin every day and strategy (b) every N days, so their per-cluster
    selection is byte-identical on every firing day. Without this the full-city 2-opt (123 ms per
    250-bin cluster, measured) would run 30 times per cluster instead of once.

    The key is ``(cluster_id, selected bins in sorted order)``. Cluster id is part of the key
    because two clusters can hold different bins, and the sorted tuple makes the key independent
    of the order the caller happened to collect the selection in. Distances are a pure function of
    that key for a fixed layout, so a hit is exactly equal to a recompute --
    ``tests/test_tsp.py`` pins that too.
    """

    def __init__(self, depot: Point, locations: Sequence[Point], max_iters: int) -> None:
        self._depot = depot
        self._locations = locations
        self._max_iters = max_iters
        self._cache: dict[tuple[int, tuple[int, ...]], float] = {}
        self.hits = 0
        self.misses = 0

    def distance_for(self, cluster_id: int, selected: Sequence[int]) -> float:
        """Closed-tour length for one cluster's selected bins, from cache or freshly solved."""
        key = (cluster_id, tuple(sorted(selected)))
        cached = self._cache.get(key)
        if cached is not None:
            self.hits += 1
            return cached
        self.misses += 1
        value = self.solve_fresh(selected)
        self._cache[key] = value
        return value

    def solve_fresh(self, selected: Sequence[int]) -> float:
        """Solve a selection from scratch, bypassing the cache (used by the cache's own test).

        Bins are sorted before solving, exactly as the cache key is. Nearest-neighbour depends on
        the input order, so without this a caller that collected the same set in a different order
        would get a different distance from a miss than from a hit.
        """
        if not selected:
            return 0.0
        points: list[Point] = [self._depot]
        points.extend(self._locations[i] for i in sorted(selected))
        _, length = solve(points, self._max_iters)
        return length
