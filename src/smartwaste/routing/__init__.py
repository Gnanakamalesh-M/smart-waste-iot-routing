"""Route optimization: clustering, depot-constrained TSP, capacitated VRP and real road distances.

Phase 2 scope: Euclidean distance on a random grid, one truck per cluster, no capacity limit --
all still here, unchanged, in :mod:`smartwaste.routing.tsp` and :mod:`smartwaste.routing.cluster`.

Phase 4 adds :mod:`smartwaste.routing.vrp` (truck capacity, multiple trucks, OR-Tools) and
:mod:`smartwaste.routing.roads` (OSMnx road distances for one bounded Chennai area). Both import
``ortools`` / ``osmnx`` / ``networkx`` **lazily**, inside the functions that need them, so this
package imports cleanly without the optional ``roads`` extra installed.
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
