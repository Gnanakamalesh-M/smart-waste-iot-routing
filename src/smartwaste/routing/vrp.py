"""Capacitated multi-truck routing (VRP), and the honest version of "the plan was wrong".

Phase 2 routed one depot-constrained TSP per cluster with **no capacity limit at all**, so a tour
could collect more waste than a real truck can carry. That flattered every strategy that visits
many bins, and ``docs/PHASE2_RESULTS.md`` said so. This module adds the limit.

### The two weights, and why they are different

A dispatcher planning tomorrow's routes has **noisy sensor weight** -- the load cell's reading,
bias and all. The truck's own scale, at the landfill, has the **true weight**. Those are not the
same number, so a route planned to fit can turn out not to fit. This module keeps them strictly
apart:

- :func:`solve_cvrp` is handed the **planning** demand only. Nothing in the solve path can see
  true weight, which is the same no-leakage rule the forecasting and anomaly code follows.
- :func:`execute_with_capacity` is handed the **true** loads only, and measures what happens.

### What happens when a planned route does not fit

The policy here is **return to depot, unload, resume**: the truck drives to the depot, tips, comes
back to the next bin on its list and carries on. The extra kilometres are charged and the event is
counted. Nothing is abandoned and nothing is silently carried over capacity.

### OR-Tools is not reproducible across machines, and that shapes the tests

``vrp.ortools_time_limit_s`` is a **wall-clock** budget: a faster machine searches more of the
space in the same five seconds and can return a different (usually shorter) route. So no test in
this project asserts an OR-Tools *distance*. They assert **properties** -- every bin visited
exactly once, every route closed at the depot, no route over capacity -- and the results doc
labels every OR-Tools number same-machine-only.

Distances reported here are always recomputed in float with :func:`~smartwaste.routing.tsp.
route_distance`, the same function the nearest-neighbour and 2-opt baselines use. OR-Tools
*searches* on a rounded integer matrix (it requires integer arc costs), but it is not also scored
on one -- otherwise the comparison would measure the rounding as well as the algorithms.

``ortools`` is imported **lazily**, inside the two functions that need it, so this module imports
cleanly without the ``roads`` extra installed.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from smartwaste.routing.tsp import DEPOT_INDEX, Point, distance, nearest_neighbour, route_distance
from smartwaste.routing.tsp import two_opt as two_opt_tour

#: OR-Tools first-solution heuristics we allow in the config. Not the solver's full list -- just
#: the three that make sense for a depot-constrained collection problem.
FIRST_SOLUTION_STRATEGIES = frozenset({"PATH_CHEAPEST_ARC", "SAVINGS", "CHRISTOFIDES"})

#: OR-Tools local-search metaheuristics we allow in the config.
LOCAL_SEARCH_METAHEURISTICS = frozenset({"GUIDED_LOCAL_SEARCH", "TABU_SEARCH", "AUTOMATIC"})

#: Solver labels, used as table row keys and in :attr:`VrpSolution.solver`.
SOLVER_NEAREST_NEIGHBOUR = "nearest_neighbour"
SOLVER_TWO_OPT = "two_opt"
SOLVER_ORTOOLS_TSP = "ortools_tsp"
SOLVER_ORTOOLS_CVRP = "ortools_cvrp"

#: OR-Tools needs integer arc costs, so km are scaled to metres and rounded. Reported distances
#: are recomputed in float from the returned tour, so this rounding affects only the *search*.
_DISTANCE_SCALE = 1000


class VrpError(RuntimeError):
    """Base class for routing failures that must not be papered over."""


class VrpUnavailable(VrpError):
    """``ortools`` is not installed. Raised instead of silently falling back to a TSP."""


class VrpInfeasible(VrpError):
    """The solver returned no solution. Never a partial answer dressed up as a complete one."""


@dataclass(frozen=True)
class VrpRoute:
    """One truck's closed tour. ``tour[0]`` is the depot and the tour returns to it."""

    tour: tuple[int, ...]
    distance_km: float
    #: Load as *planned*, i.e. summed from whatever demand vector the solver was given.
    planned_load_kg: float

    @property
    def n_bins(self) -> int:
        """Stops excluding the depot."""
        return max(0, len(self.tour) - 1)


@dataclass(frozen=True)
class VrpSolution:
    """Every route for one instance, plus the solver that produced it."""

    routes: tuple[VrpRoute, ...]
    total_distance_km: float
    #: Routes that actually carry at least one bin. An empty vehicle is not a truck you sent out.
    n_trucks: int
    solver: str

    @property
    def total_planned_load_kg(self) -> float:
        return sum(route.planned_load_kg for route in self.routes)


@dataclass(frozen=True)
class ExecutedRoute:
    """What actually happened when a planned route met the truck's real scale."""

    #: Planned distance plus every mid-route depot return.
    distance_km: float
    #: Planned distance, for the detour arithmetic.
    planned_distance_km: float
    depot_returns: int
    peak_true_load_kg: float
    capacity_breached: bool

    @property
    def detour_km(self) -> float:
        """Extra distance the depot returns cost. Zero when the plan held."""
        return self.distance_km - self.planned_distance_km


def scaled_matrix(points: Sequence[Point]) -> list[list[int]]:
    """Integer distance matrix in metres, for OR-Tools' arc costs.

    Rounded, because OR-Tools requires integers. The rounding is a known, small source of
    divergence between what the solver optimises and what we report -- stated in the results doc
    rather than hidden.
    """
    n = len(points)
    return [
        [round(distance(points[i], points[j]) * _DISTANCE_SCALE) for j in range(n)]
        for i in range(n)
    ]


def solve_nearest_neighbour(points: Sequence[Point], demands_kg: Sequence[float]) -> VrpSolution:
    """Phase 2's nearest-neighbour tour, wrapped as a one-truck :class:`VrpSolution`.

    Reuses :func:`smartwaste.routing.tsp.nearest_neighbour` unchanged -- this is the Phase 2
    baseline, not a reimplementation of it.
    """
    tour = nearest_neighbour(points, DEPOT_INDEX)
    return _single_route_solution(points, tour, demands_kg, SOLVER_NEAREST_NEIGHBOUR)


def solve_two_opt(
    points: Sequence[Point], demands_kg: Sequence[float], *, max_iters: int
) -> VrpSolution:
    """Phase 2's nearest-neighbour-then-2-opt tour, wrapped as a one-truck solution.

    This is the baseline OR-Tools has to beat, and it is the *same code* Phase 2 published with.
    """
    tour = two_opt_tour(points, nearest_neighbour(points, DEPOT_INDEX), max_iters)
    return _single_route_solution(points, tour, demands_kg, SOLVER_TWO_OPT)


def solve_tsp_ortools(
    points: Sequence[Point],
    demands_kg: Sequence[float],
    *,
    time_limit_s: float,
    first_solution: str,
    local_search: str,
) -> VrpSolution:
    """One vehicle, **no capacity limit**: the capacity-free OR-Tools column.

    This is the fair comparison against 2-opt -- same problem, different solver. The capacitated
    solve below answers a strictly harder question and is labelled accordingly.
    """
    solution = _solve_with_ortools(
        points,
        demands_kg,
        capacity_kg=None,
        max_vehicles=1,
        time_limit_s=time_limit_s,
        first_solution=first_solution,
        local_search=local_search,
    )
    return VrpSolution(
        routes=solution.routes,
        total_distance_km=solution.total_distance_km,
        n_trucks=solution.n_trucks,
        solver=SOLVER_ORTOOLS_TSP,
    )


def solve_cvrp(
    points: Sequence[Point],
    demands_kg: Sequence[float],
    *,
    capacity_kg: float,
    max_vehicles: int,
    time_limit_s: float,
    first_solution: str,
    local_search: str,
) -> VrpSolution:
    """Capacitated multi-truck VRP: as few trucks as the solver can manage under ``capacity_kg``.

    ``demands_kg`` is the **planning** demand -- in this project the fused *noisy* sensor weight.
    ``demands_kg[DEPOT_INDEX]`` is ignored and treated as zero.

    Demands are rounded **up** to integer kg, so the rounding can only ever overstate a load. An
    understating rounding would let the solver plan a route that is infeasible by a hair and call
    it feasible, which is the one error a capacity model must not make.

    Raises:
        VrpUnavailable: ``ortools`` is not installed.
        VrpInfeasible: the solver found no solution within the time limit.
    """
    if capacity_kg <= 0:
        raise ValueError(f"capacity_kg must be > 0, got {capacity_kg}")
    if max_vehicles < 1:
        raise ValueError(f"max_vehicles must be >= 1, got {max_vehicles}")
    return _solve_with_ortools(
        points,
        demands_kg,
        capacity_kg=capacity_kg,
        max_vehicles=max_vehicles,
        time_limit_s=time_limit_s,
        first_solution=first_solution,
        local_search=local_search,
    )


def execute_with_capacity(
    route: VrpRoute,
    points: Sequence[Point],
    true_loads_kg: Sequence[float],
    capacity_kg: float,
) -> ExecutedRoute:
    """Drive one planned route against the truck's real scale.

    Walks the tour accumulating **true** weight. When the next bin would put the truck over
    capacity, the truck drives depot-and-back instead of straight on: the leg
    ``current -> next`` is replaced by ``current -> depot -> next``, the load resets, and
    ``depot_returns`` goes up by one.

    A single bin heavier than the whole truck cannot be made to fit by any number of depot trips.
    That is flagged (``capacity_breached``) and the bin is still collected -- the alternative,
    leaving it on the street forever, would be a different and less honest model.

    ``capacity_breached`` is True whenever the plan did not hold: either a depot return was
    needed, or one bin alone exceeded capacity.
    """
    if capacity_kg <= 0:
        raise ValueError(f"capacity_kg must be > 0, got {capacity_kg}")

    planned = route_distance(points, route.tour)
    stops = list(route.tour)
    if len(stops) < 2:
        return ExecutedRoute(
            distance_km=planned,
            planned_distance_km=planned,
            depot_returns=0,
            peak_true_load_kg=0.0,
            capacity_breached=False,
        )

    depot = points[stops[0]]
    travelled = 0.0
    load = 0.0
    peak = 0.0
    depot_returns = 0
    oversized_bins = 0

    for position in range(1, len(stops)):
        here = points[stops[position - 1]]
        nxt_index = stops[position]
        nxt = points[nxt_index]
        demand = true_loads_kg[nxt_index]

        # Checked per stop, not latched: one bin heavier than the whole truck must not stop the
        # rest of the route from returning to the depot when it legitimately needs to.
        too_big_for_any_truck = demand > capacity_kg
        if too_big_for_any_truck:
            oversized_bins += 1

        if load + demand > capacity_kg and not too_big_for_any_truck:
            travelled += distance(here, depot) + distance(depot, nxt)
            depot_returns += 1
            load = 0.0
        else:
            travelled += distance(here, nxt)

        load += demand
        peak = max(peak, load)

    # Close the loop back to the depot, as route_distance does for the plan.
    travelled += distance(points[stops[-1]], depot)

    return ExecutedRoute(
        distance_km=travelled,
        planned_distance_km=planned,
        depot_returns=depot_returns,
        peak_true_load_kg=peak,
        capacity_breached=depot_returns > 0 or oversized_bins > 0,
    )


def _single_route_solution(
    points: Sequence[Point],
    tour: Sequence[int],
    demands_kg: Sequence[float],
    solver: str,
) -> VrpSolution:
    """Wrap one stdlib TSP tour as a :class:`VrpSolution` so every solver returns one shape."""
    length = route_distance(points, tour)
    load = sum(demands_kg[i] for i in tour if i != DEPOT_INDEX)
    route = VrpRoute(tour=tuple(tour), distance_km=length, planned_load_kg=load)
    carries_bins = 1 if route.n_bins > 0 else 0
    return VrpSolution(
        routes=(route,),
        total_distance_km=length,
        n_trucks=carries_bins,
        solver=solver,
    )


def _solve_with_ortools(
    points: Sequence[Point],
    demands_kg: Sequence[float],
    *,
    capacity_kg: float | None,
    max_vehicles: int,
    time_limit_s: float,
    first_solution: str,
    local_search: str,
) -> VrpSolution:
    """Shared OR-Tools solve. ``capacity_kg=None`` means no capacity dimension at all."""
    if first_solution not in FIRST_SOLUTION_STRATEGIES:
        raise ValueError(f"unknown first_solution strategy: {first_solution!r}")
    if local_search not in LOCAL_SEARCH_METAHEURISTICS:
        raise ValueError(f"unknown local_search metaheuristic: {local_search!r}")
    if time_limit_s <= 0:
        raise ValueError(f"time_limit_s must be > 0, got {time_limit_s}")
    if len(demands_kg) != len(points):
        raise ValueError(
            "demands_kg and points must be the same length, got "
            f"{len(demands_kg)} and {len(points)}"
        )

    # Imported here, not at module scope, so this module works without the `roads` extra.
    try:
        from ortools.constraint_solver import pywrapcp, routing_enums_pb2
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise VrpUnavailable(
            "ortools is not installed; install the optional extra with: "
            'python -m pip install -e ".[roads]"'
        ) from exc

    n = len(points)
    if n <= 1:
        # Nothing but a depot. An empty solution, not an error.
        return VrpSolution(routes=(), total_distance_km=0.0, n_trucks=0, solver=SOLVER_ORTOOLS_CVRP)

    matrix = scaled_matrix(points)
    manager = pywrapcp.RoutingIndexManager(n, max_vehicles, DEPOT_INDEX)
    model = pywrapcp.RoutingModel(manager)

    def arc_cost(from_index: int, to_index: int) -> int:
        return int(matrix[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)])

    transit = model.RegisterTransitCallback(arc_cost)
    model.SetArcCostEvaluatorOfAllVehicles(transit)

    if capacity_kg is not None:
        # Rounded UP, so rounding never understates a load (see solve_cvrp's docstring).
        demands_int = [0] + [math.ceil(demands_kg[i]) for i in range(1, n)]
        capacity_int = int(math.floor(capacity_kg))

        def demand_cb(from_index: int) -> int:
            return int(demands_int[manager.IndexToNode(from_index)])

        demand_callback = model.RegisterUnaryTransitCallback(demand_cb)
        model.AddDimensionWithVehicleCapacity(
            demand_callback,
            0,  # no slack: a truck cannot un-fill itself between stops
            [capacity_int] * max_vehicles,
            True,  # every truck starts empty
            "Capacity",
        )

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = getattr(
        routing_enums_pb2.FirstSolutionStrategy, first_solution
    )
    params.local_search_metaheuristic = getattr(
        routing_enums_pb2.LocalSearchMetaheuristic, local_search
    )
    # Milliseconds, so a fractional --quick budget like 1.0 s is honoured exactly.
    params.time_limit.FromMilliseconds(int(time_limit_s * 1000))

    assignment = model.SolveWithParameters(params)
    if assignment is None:
        raise VrpInfeasible(
            f"OR-Tools found no solution for {n - 1} bins with "
            f"capacity={capacity_kg} and max_vehicles={max_vehicles} "
            f"within {time_limit_s} s"
        )

    routes: list[VrpRoute] = []
    total = 0.0
    for vehicle in range(max_vehicles):
        index = model.Start(vehicle)
        tour: list[int] = []
        while not model.IsEnd(index):
            tour.append(manager.IndexToNode(index))
            index = assignment.Value(model.NextVar(index))
        if len(tour) <= 1:
            # An unused vehicle: it never left the depot. Not a truck you sent out.
            continue
        # Distance in float, from the same function the stdlib baselines use, so the comparison
        # does not also measure the integer rounding OR-Tools searched on.
        length = route_distance(points, tour)
        load = sum(demands_kg[i] for i in tour if i != DEPOT_INDEX)
        routes.append(VrpRoute(tour=tuple(tour), distance_km=length, planned_load_kg=load))
        total += length

    return VrpSolution(
        routes=tuple(routes),
        total_distance_km=total,
        n_trucks=len(routes),
        solver=SOLVER_ORTOOLS_CVRP,
    )


# ------------------------------------------------------------------------------------------------
# Matrix-based tours
#
# Phase 2's TSP takes *coordinates* and computes distance with a formula. A road distance is not a
# formula -- it is a shortest path through a graph, so it can only arrive as a precomputed matrix,
# and it is not even guaranteed symmetric (one-way streets). These three functions are the same
# algorithms as tsp.py's, reading a matrix instead of calling `distance`.
#
# They exist so that the Euclidean and road tours in `run_roads.py` are built by *identical* code
# and the only difference between the two numbers is the distance metric itself. Using tsp.py for
# the Euclidean tour and something else for the road tour would have put a second difference in a
# comparison that is supposed to isolate one.
# ------------------------------------------------------------------------------------------------

#: A square matrix of distances in km. ``matrix[i][j]`` may differ from ``matrix[j][i]``.
Matrix = Sequence[Sequence[float]]


def matrix_tour_length(matrix: Matrix, tour: Sequence[int]) -> float:
    """Closed-tour length through a distance matrix. ``0.0`` for a tour of fewer than 2 stops."""
    if len(tour) < 2:
        return 0.0
    total = 0.0
    for position in range(len(tour)):
        total += matrix[tour[position]][tour[(position + 1) % len(tour)]]
    return total


def matrix_nearest_neighbour(matrix: Matrix, start: int = DEPOT_INDEX) -> list[int]:
    """Nearest-neighbour tour through a matrix. Ties keep the lowest index, as in tsp.py."""
    n = len(matrix)
    if n == 0:
        return []
    unvisited = [i for i in range(n) if i != start]
    tour = [start]
    current = start
    while unvisited:
        best_position = 0
        best_cost = matrix[current][unvisited[0]]
        for position in range(1, len(unvisited)):
            cost = matrix[current][unvisited[position]]
            if cost < best_cost:
                best_cost = cost
                best_position = position
        current = unvisited.pop(best_position)
        tour.append(current)
    return tour


def matrix_two_opt(matrix: Matrix, tour: Sequence[int], max_iters: int) -> list[int]:
    """2-opt through a matrix, accepting strict improvements only.

    As in tsp.py, ``i`` starts at 1 so the depot stays pinned at position 0, and only a gain
    larger than a float-noise floor counts -- otherwise the iteration cap becomes the only stop.
    """
    improved = list(tour)
    n = len(improved)
    if n < 4:
        return improved
    for _ in range(max_iters):
        found = False
        for i in range(1, n - 1):
            for k in range(i + 1, n):
                previous = improved[i - 1]
                here = improved[i]
                there = improved[k]
                following = improved[(k + 1) % n]
                gain = (matrix[previous][here] + matrix[there][following]) - (
                    matrix[previous][there] + matrix[here][following]
                )
                if gain > _MIN_MATRIX_GAIN:
                    improved[i : k + 1] = reversed(improved[i : k + 1])
                    found = True
        if not found:
            break
    return improved


#: Mirrors tsp.py's ``_MIN_GAIN``: only a strictly larger gain than this counts as an improvement.
_MIN_MATRIX_GAIN = 1e-12
