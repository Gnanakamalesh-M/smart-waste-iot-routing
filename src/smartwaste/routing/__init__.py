"""Route optimization: geographic clustering and depot-constrained TSP.

Phase 2 scope: Euclidean distance on a random grid, one truck per cluster, no capacity limit.
Real road distances (OSMnx) and capacitated multi-truck VRP are Phase 4.
"""

from smartwaste.routing.cluster import ClusterResult, kmeans
from smartwaste.routing.tsp import Point, RouteCache, nearest_neighbour, route_distance, two_opt

__all__ = [
    "ClusterResult",
    "Point",
    "RouteCache",
    "kmeans",
    "nearest_neighbour",
    "route_distance",
    "two_opt",
]
