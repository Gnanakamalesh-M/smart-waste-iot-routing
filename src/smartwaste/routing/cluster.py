"""Geographic clustering: one truck's territory per cluster.

The paper "splits the city into geographic clusters" and gives each truck one cluster, without
stating a cluster count or an algorithm. We use KMeans with k-means++ seeding, written in the
standard library so Phase 2 adds no dependency.

Clusters are computed **once per city layout**, not once per day: the bins do not move, so every
strategy and every day share the same territories. That also means the clustering cannot
accidentally favour a strategy -- all five are routed over identical cluster boundaries.

Determinism matters here, because a different labelling changes every downstream distance. Three
things secure it: the caller passes a seeded ``random.Random``, assignment ties break to the
lowest cluster index, and an emptied cluster is re-seeded from a deterministically chosen point
rather than a fresh random draw.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass

from smartwaste.routing.tsp import Point

#: Stop when no centroid moves more than this (km) in a full iteration.
_TOLERANCE = 1e-9

#: Hard cap on Lloyd iterations, so a pathological layout cannot spin forever.
_MAX_ITERATIONS = 100


@dataclass(frozen=True)
class ClusterResult:
    """The outcome of clustering one city layout."""

    #: ``labels[i]`` is the cluster index of bin ``i``.
    labels: tuple[int, ...]
    #: Centroid of each cluster, in km.
    centroids: tuple[Point, ...]

    @property
    def n_clusters(self) -> int:
        return len(self.centroids)

    def members(self, cluster_id: int) -> list[int]:
        """Indices of the bins assigned to one cluster, ascending."""
        return [i for i, label in enumerate(self.labels) if label == cluster_id]


def _squared(a: Point, b: Point) -> float:
    """Squared Euclidean distance. KMeans only ever compares distances, so the sqrt is waste."""
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx * dx + dy * dy


def _kmeans_plus_plus(points: Sequence[Point], k: int, rng: random.Random) -> list[Point]:
    """k-means++ seeding: spread the initial centroids out instead of picking k at random.

    Plain random seeding regularly puts two centroids in the same neighbourhood, which costs
    iterations and can settle in a visibly worse local optimum. k-means++ picks each new centroid
    with probability proportional to its squared distance from the nearest centroid chosen so far.
    """
    first = rng.randrange(len(points))
    centroids = [points[first]]
    # Distance from each point to the nearest centroid chosen so far.
    best = [_squared(p, points[first]) for p in points]

    while len(centroids) < k:
        total = sum(best)
        if total <= 0.0:
            # Every point coincides with a centroid already (heavy duplicates). Any further
            # centroid is arbitrary, so take a deterministic one rather than looping.
            centroids.append(points[rng.randrange(len(points))])
        else:
            target = rng.random() * total
            cumulative = 0.0
            chosen = len(points) - 1
            for i, weight in enumerate(best):
                cumulative += weight
                if cumulative >= target:
                    chosen = i
                    break
            centroids.append(points[chosen])
        newest = centroids[-1]
        for i, p in enumerate(points):
            candidate = _squared(p, newest)
            if candidate < best[i]:
                best[i] = candidate
    return centroids


def _assign(points: Sequence[Point], centroids: Sequence[Point]) -> list[int]:
    """Label each point with its nearest centroid, ties going to the lowest index."""
    labels: list[int] = []
    for p in points:
        best_label = 0
        best_cost = _squared(p, centroids[0])
        for label in range(1, len(centroids)):
            cost = _squared(p, centroids[label])
            if cost < best_cost:  # strict: a tie keeps the lower label
                best_cost = cost
                best_label = label
        labels.append(best_label)
    return labels


def kmeans(points: Sequence[Point], k: int, rng: random.Random) -> ClusterResult:
    """Cluster ``points`` into ``k`` groups with seeded k-means++ and Lloyd iterations.

    Args:
        points: bin locations in km.
        k: number of clusters, i.e. number of trucks in Phase 2. Clamped down to ``len(points)``
            when there are fewer bins than clusters, since an empty territory has no meaning.
        rng: seeded generator. The same points, k and seed give identical labels.

    Returns:
        A :class:`ClusterResult`. Every point is assigned exactly one label in ``range(k)``.
    """
    if not points:
        raise ValueError("kmeans needs at least one point")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    k = min(k, len(points))

    centroids = _kmeans_plus_plus(points, k, rng)
    labels = _assign(points, centroids)

    for _ in range(_MAX_ITERATIONS):
        sums: list[list[float]] = [[0.0, 0.0] for _ in range(k)]
        counts = [0] * k
        for i, label in enumerate(labels):
            sums[label][0] += points[i][0]
            sums[label][1] += points[i][1]
            counts[label] += 1

        moved = 0.0
        updated: list[Point] = []
        for label in range(k):
            if counts[label] == 0:
                # An emptied cluster would otherwise stay empty forever. Re-seed it on the point
                # furthest from its own centroid -- deterministic, and it splits the worst cluster.
                updated.append(_furthest_point(points, centroids, labels))
            else:
                updated.append((sums[label][0] / counts[label], sums[label][1] / counts[label]))
            moved = max(moved, math.dist(centroids[label], updated[label]))

        centroids = updated
        labels = _assign(points, centroids)
        if moved <= _TOLERANCE:
            break

    return ClusterResult(labels=tuple(labels), centroids=tuple(centroids))


def _furthest_point(
    points: Sequence[Point], centroids: Sequence[Point], labels: Sequence[int]
) -> Point:
    """The point worst served by its current centroid; ties go to the lowest index."""
    best_index = 0
    best_cost = -1.0
    for i, p in enumerate(points):
        cost = _squared(p, centroids[labels[i]])
        if cost > best_cost:
            best_cost = cost
            best_index = i
    return points[best_index]
