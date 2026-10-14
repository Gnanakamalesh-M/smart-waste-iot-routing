"""Capacitated VRP correctness: the properties a route must satisfy.

**No test here asserts an OR-Tools distance**, and that is deliberate rather than lazy.
``vrp.ortools_time_limit_s`` is a wall-clock budget, so a faster or less busy machine searches
more of the space in the same five seconds and can return a different route. Pinning a distance
would make this suite fail on someone else's laptop for a reason that is not a bug. What *is*
asserted is every property the solution has to have however long the search ran: every bin visited
exactly once, every route closed at the depot, no route over capacity.

The stdlib half of the module -- the integer scaling, the capacity executor, the matrix tours --
is tested without ortools at all, which is what keeps the `routing` chunk green when the optional
`roads` extra is not installed.
"""

import math
import random

import pytest

from smartwaste.routing.tsp import DEPOT_INDEX, route_distance
from smartwaste.routing.vrp import (
    FIRST_SOLUTION_STRATEGIES,
    LOCAL_SEARCH_METAHEURISTICS,
    SOLVER_NEAREST_NEIGHBOUR,
    SOLVER_ORTOOLS_CVRP,
    SOLVER_ORTOOLS_TSP,
    SOLVER_TWO_OPT,
    VrpRoute,
    execute_with_capacity,
    matrix_nearest_neighbour,
    matrix_tour_length,
    matrix_two_opt,
    scaled_matrix,
    solve_cvrp,
    solve_nearest_neighbour,
    solve_tsp_ortools,
    solve_two_opt,
)

pytestmark = pytest.mark.routing

#: Depot at the origin, three bins on the positive x axis at 1, 2 and 3 km.
LINE = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)]

#: A unit square with the depot at the origin. Optimal closed tour is the perimeter, length 4.
SQUARE = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]

#: Default solver knobs for the OR-Tools tests, matching config/default.yaml.
FIRST_SOLUTION = "PATH_CHEAPEST_ARC"
LOCAL_SEARCH = "GUIDED_LOCAL_SEARCH"

#: A short budget: these are correctness tests, not a benchmark.
TEST_TIME_LIMIT_S = 2.0


def _random_points(n: int, seed: int, span: float = 25.0) -> list[tuple[float, float]]:
    rng = random.Random(seed)
    return [(rng.uniform(0, span), rng.uniform(0, span)) for _ in range(n)]


def _ortools_or_skip() -> None:
    pytest.importorskip("ortools", reason="the optional 'roads' extra is not installed")


# --------------------------------------------------------------------------------------------
# The stdlib half: no ortools, no osmnx
# --------------------------------------------------------------------------------------------


def test_scaled_matrix_is_metres_and_integral() -> None:
    matrix = scaled_matrix(LINE)
    assert matrix[0][1] == 1000
    assert matrix[0][3] == 3000
    assert matrix[2][0] == 2000
    assert all(isinstance(value, int) for row in matrix for value in row)


def test_scaled_matrix_diagonal_is_zero() -> None:
    matrix = scaled_matrix(SQUARE)
    assert all(matrix[i][i] == 0 for i in range(len(SQUARE)))


def test_baselines_reuse_the_phase_2_tsp() -> None:
    """The baseline distances must equal Phase 2's own route_distance of the same tour."""
    demands = [0.0, 1.0, 1.0, 1.0]
    nn = solve_nearest_neighbour(LINE, demands)
    opt = solve_two_opt(LINE, demands, max_iters=1000)
    assert nn.solver == SOLVER_NEAREST_NEIGHBOUR
    assert opt.solver == SOLVER_TWO_OPT
    assert nn.total_distance_km == pytest.approx(route_distance(LINE, nn.routes[0].tour))
    assert opt.total_distance_km == pytest.approx(route_distance(LINE, opt.routes[0].tour))
    # Out and back along a line: 3 km each way.
    assert opt.total_distance_km == pytest.approx(6.0)


def test_baseline_load_sums_the_demand_and_ignores_the_depot() -> None:
    solution = solve_two_opt(LINE, [999.0, 1.0, 2.0, 3.0], max_iters=1000)
    assert solution.total_planned_load_kg == pytest.approx(6.0)


def test_two_opt_is_never_longer_than_nearest_neighbour() -> None:
    """Phase 2's invariant, re-checked through the Phase 4 wrappers."""
    for n in (4, 8, 20, 40):
        points = _random_points(n, seed=400 + n)
        demands = [0.0] * n
        nn = solve_nearest_neighbour(points, demands)
        opt = solve_two_opt(points, demands, max_iters=1000)
        assert opt.total_distance_km <= nn.total_distance_km + 1e-9


# --------------------------------------------------------------------------------------------
# execute_with_capacity: the "return to depot, resume" policy
# --------------------------------------------------------------------------------------------


def test_a_plan_that_fits_pays_nothing_extra() -> None:
    route = VrpRoute(tour=(0, 1, 2, 3), distance_km=6.0, planned_load_kg=30.0)
    executed = execute_with_capacity(route, LINE, [0.0, 10.0, 10.0, 10.0], capacity_kg=100.0)
    assert executed.depot_returns == 0
    assert executed.capacity_breached is False
    assert executed.detour_km == pytest.approx(0.0)
    assert executed.distance_km == pytest.approx(6.0)
    assert executed.peak_true_load_kg == pytest.approx(30.0)


def test_one_breach_costs_one_depot_round_trip() -> None:
    """Depot, then bins at 1 and 2 km. Capacity 15 kg, loads 10 kg each.

    Planned: 0 -> 1 -> 2 -> 0 = 1 + 1 + 2 = 4 km.
    Executed: 0 -> 1 (1 km), then 1 -> depot -> 2 (1 + 2 km) because 20 kg > 15 kg,
    then 2 -> 0 (2 km) = 6 km. So the detour is exactly 2 km.
    """
    points = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)]
    route = VrpRoute(tour=(0, 1, 2), distance_km=4.0, planned_load_kg=20.0)
    executed = execute_with_capacity(route, points, [0.0, 10.0, 10.0], capacity_kg=15.0)
    assert executed.depot_returns == 1
    assert executed.capacity_breached is True
    assert executed.planned_distance_km == pytest.approx(4.0)
    assert executed.distance_km == pytest.approx(6.0)
    assert executed.detour_km == pytest.approx(2.0)
    # The truck unloaded, so it never actually carried more than one bin's worth.
    assert executed.peak_true_load_kg == pytest.approx(10.0)


def test_a_bin_heavier_than_the_truck_is_flagged_not_looped() -> None:
    """No number of depot trips makes a 50 kg bin fit in a 5 kg truck.

    It is flagged and still collected. The alternative -- leaving it on the street forever --
    would be a different and less honest model, so the breach is reported instead.
    """
    points = [(0.0, 0.0), (1.0, 0.0)]
    route = VrpRoute(tour=(0, 1), distance_km=2.0, planned_load_kg=50.0)
    executed = execute_with_capacity(route, points, [0.0, 50.0], capacity_kg=5.0)
    assert executed.capacity_breached is True
    assert executed.depot_returns == 0
    assert executed.distance_km == pytest.approx(2.0)


def test_an_oversized_bin_does_not_latch_off_later_depot_returns() -> None:
    """One impossible bin must not stop the rest of the route behaving correctly.

    Bins at 1, 2 and 3 km with loads 50, 10, 10 kg and a 15 kg truck: bin 1 is impossible, but
    the 10 + 10 kg pair that follows still needs one depot return between them.
    """
    executed = execute_with_capacity(
        VrpRoute(tour=(0, 1, 2, 3), distance_km=6.0, planned_load_kg=70.0),
        LINE,
        [0.0, 50.0, 10.0, 10.0],
        capacity_kg=15.0,
    )
    assert executed.capacity_breached is True
    assert executed.depot_returns >= 1


def test_an_empty_route_is_free() -> None:
    executed = execute_with_capacity(
        VrpRoute(tour=(0,), distance_km=0.0, planned_load_kg=0.0),
        LINE,
        [0.0, 0.0, 0.0, 0.0],
        capacity_kg=100.0,
    )
    assert executed.distance_km == pytest.approx(0.0)
    assert executed.depot_returns == 0
    assert executed.capacity_breached is False


def test_a_non_positive_capacity_is_rejected() -> None:
    route = VrpRoute(tour=(0, 1), distance_km=2.0, planned_load_kg=1.0)
    with pytest.raises(ValueError, match="capacity_kg must be > 0"):
        execute_with_capacity(route, LINE, [0.0, 1.0], capacity_kg=0.0)


def test_the_oracle_invariant_zero_breaches_by_construction() -> None:
    """Table C's oracle row claims zero breaches *by construction*. This pins that claim.

    When the capacity check is applied to exactly the weights the plan respected, no depot return
    can ever be needed. The test builds the situation directly rather than trusting the solver:
    a route whose true loads sum to within capacity breaches nothing, for any route.
    """
    for n in (2, 5, 12):
        points = _random_points(n, seed=700 + n)
        loads = [0.0] + [10.0] * (n - 1)
        capacity = sum(loads)  # exactly what the plan was allowed to carry
        route = VrpRoute(
            tour=tuple(range(n)),
            distance_km=route_distance(points, range(n)),
            planned_load_kg=sum(loads),
        )
        executed = execute_with_capacity(route, points, loads, capacity_kg=capacity)
        assert executed.depot_returns == 0
        assert executed.capacity_breached is False
        assert executed.detour_km == pytest.approx(0.0)


# --------------------------------------------------------------------------------------------
# Matrix tours: road distance is a lookup, not a formula
# --------------------------------------------------------------------------------------------


def test_matrix_tour_length_closes_the_loop() -> None:
    matrix = [[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]]
    assert matrix_tour_length(matrix, [0, 1, 2]) == pytest.approx(1.0 + 1.0 + 2.0)


def test_matrix_tour_length_of_a_non_route_is_zero() -> None:
    assert matrix_tour_length([[0.0]], [0]) == pytest.approx(0.0)
    assert matrix_tour_length([[0.0]], []) == pytest.approx(0.0)


def test_matrix_tours_match_the_euclidean_tsp_on_a_euclidean_matrix() -> None:
    """Fed a Euclidean matrix, the matrix tours must agree with Phase 2's coordinate TSP.

    That equivalence is what makes the roads comparison fair: the only difference between the
    straight-line and road tour is the matrix, not the algorithm.
    """
    points = _random_points(12, seed=808)
    matrix = [[math.dist(a, b) for b in points] for a in points]
    tour = matrix_two_opt(matrix, matrix_nearest_neighbour(matrix), 1000)
    assert matrix_tour_length(matrix, tour) == pytest.approx(route_distance(points, tour))


def test_matrix_two_opt_never_lengthens_the_tour() -> None:
    points = _random_points(25, seed=909)
    matrix = [[math.dist(a, b) for b in points] for a in points]
    start = matrix_nearest_neighbour(matrix)
    improved = matrix_two_opt(matrix, start, 1000)
    assert matrix_tour_length(matrix, improved) <= matrix_tour_length(matrix, start) + 1e-9
    assert sorted(improved) == sorted(start)


def test_matrix_tours_pin_the_depot_at_position_zero() -> None:
    points = _random_points(15, seed=1010)
    matrix = [[math.dist(a, b) for b in points] for a in points]
    tour = matrix_two_opt(matrix, matrix_nearest_neighbour(matrix), 1000)
    assert tour[0] == DEPOT_INDEX


def test_matrix_nearest_neighbour_of_empty_is_empty() -> None:
    assert matrix_nearest_neighbour([]) == []


def test_matrix_two_opt_respects_the_iteration_cap() -> None:
    points = _random_points(10, seed=1111)
    matrix = [[math.dist(a, b) for b in points] for a in points]
    start = matrix_nearest_neighbour(matrix)
    assert matrix_two_opt(matrix, start, 0) == start


# --------------------------------------------------------------------------------------------
# OR-Tools. Properties only -- never a distance (see this module's docstring).
# --------------------------------------------------------------------------------------------


def test_the_config_whitelists_are_real_ortools_enum_members() -> None:
    """Every name config.py accepts must actually exist in the solver's enums."""
    _ortools_or_skip()
    from ortools.constraint_solver import routing_enums_pb2

    for name in FIRST_SOLUTION_STRATEGIES:
        assert hasattr(routing_enums_pb2.FirstSolutionStrategy, name)
    for name in LOCAL_SEARCH_METAHEURISTICS:
        assert hasattr(routing_enums_pb2.LocalSearchMetaheuristic, name)


def test_ortools_tsp_visits_every_bin_exactly_once() -> None:
    _ortools_or_skip()
    points = _random_points(15, seed=1212)
    solution = solve_tsp_ortools(
        points,
        [0.0] * len(points),
        time_limit_s=TEST_TIME_LIMIT_S,
        first_solution=FIRST_SOLUTION,
        local_search=LOCAL_SEARCH,
    )
    assert solution.solver == SOLVER_ORTOOLS_TSP
    visited = [stop for route in solution.routes for stop in route.tour]
    assert sorted(visited) == sorted(range(len(points)))


def test_ortools_tsp_distance_is_finite_and_in_a_loose_sanity_band() -> None:
    """A LOOSE band, not an equality: the wall-clock limit makes the exact value machine-specific.

    All this asserts is that OR-Tools produced a real tour of a sane length -- not longer than
    twice the 2-opt tour, and not shorter than the straight out-and-back to the furthest bin.
    """
    _ortools_or_skip()
    points = _random_points(15, seed=1313)
    demands = [0.0] * len(points)
    baseline = solve_two_opt(points, demands, max_iters=1000).total_distance_km
    solution = solve_tsp_ortools(
        points,
        demands,
        time_limit_s=TEST_TIME_LIMIT_S,
        first_solution=FIRST_SOLUTION,
        local_search=LOCAL_SEARCH,
    )
    furthest = max(math.dist(points[0], p) for p in points[1:])
    assert math.isfinite(solution.total_distance_km)
    assert 2.0 * furthest - 1e-9 <= solution.total_distance_km <= 2.0 * baseline


def test_cvrp_visits_every_bin_exactly_once() -> None:
    _ortools_or_skip()
    points = _random_points(20, seed=1414)
    demands = [0.0] + [100.0] * (len(points) - 1)
    solution = solve_cvrp(
        points,
        demands,
        capacity_kg=600.0,
        max_vehicles=10,
        time_limit_s=TEST_TIME_LIMIT_S,
        first_solution=FIRST_SOLUTION,
        local_search=LOCAL_SEARCH,
    )
    bins_visited = [stop for route in solution.routes for stop in route.tour if stop != DEPOT_INDEX]
    assert sorted(bins_visited) == sorted(range(1, len(points)))


def test_every_cvrp_route_starts_and_ends_at_the_depot() -> None:
    _ortools_or_skip()
    points = _random_points(18, seed=1515)
    demands = [0.0] + [100.0] * (len(points) - 1)
    solution = solve_cvrp(
        points,
        demands,
        capacity_kg=500.0,
        max_vehicles=10,
        time_limit_s=TEST_TIME_LIMIT_S,
        first_solution=FIRST_SOLUTION,
        local_search=LOCAL_SEARCH,
    )
    assert solution.routes, "the solver returned no routes at all"
    for route in solution.routes:
        assert route.tour[0] == DEPOT_INDEX
        # The tour is stored open and closed by route_distance, as in Phase 2's TSP.
        assert DEPOT_INDEX not in route.tour[1:]


def test_no_cvrp_route_exceeds_capacity() -> None:
    """The constraint the whole phase exists for."""
    _ortools_or_skip()
    points = _random_points(20, seed=1616)
    demands = [0.0] + [100.0] * (len(points) - 1)
    capacity = 450.0
    solution = solve_cvrp(
        points,
        demands,
        capacity_kg=capacity,
        max_vehicles=10,
        time_limit_s=TEST_TIME_LIMIT_S,
        first_solution=FIRST_SOLUTION,
        local_search=LOCAL_SEARCH,
    )
    for route in solution.routes:
        assert route.planned_load_kg <= capacity + 1e-9


def test_a_capacity_too_small_for_one_truck_forces_several() -> None:
    _ortools_or_skip()
    points = _random_points(12, seed=1717)
    demands = [0.0] + [100.0] * (len(points) - 1)
    # 11 bins x 100 kg = 1100 kg, so a 300 kg truck cannot possibly do it alone.
    solution = solve_cvrp(
        points,
        demands,
        capacity_kg=300.0,
        max_vehicles=10,
        time_limit_s=TEST_TIME_LIMIT_S,
        first_solution=FIRST_SOLUTION,
        local_search=LOCAL_SEARCH,
    )
    assert solution.n_trucks > 1
    assert solution.n_trucks >= math.ceil(1100.0 / 300.0)


def test_a_generous_capacity_needs_only_one_truck() -> None:
    _ortools_or_skip()
    points = _random_points(12, seed=1818)
    demands = [0.0] + [10.0] * (len(points) - 1)
    solution = solve_cvrp(
        points,
        demands,
        capacity_kg=100000.0,
        max_vehicles=10,
        time_limit_s=TEST_TIME_LIMIT_S,
        first_solution=FIRST_SOLUTION,
        local_search=LOCAL_SEARCH,
    )
    assert solution.n_trucks == 1


def test_an_executed_cvrp_route_never_breaches_its_own_planning_weights() -> None:
    """The oracle row's invariant, end to end through the real solver.

    Plan on a weight vector and then check against that same vector: because the solver enforced
    capacity on exactly those numbers, no depot return can be needed. This is what makes Table
    C's "zero by construction" a tested claim rather than an assertion.
    """
    _ortools_or_skip()
    points = _random_points(15, seed=1919)
    demands = [0.0] + [100.0] * (len(points) - 1)
    capacity = 500.0
    solution = solve_cvrp(
        points,
        demands,
        capacity_kg=capacity,
        max_vehicles=10,
        time_limit_s=TEST_TIME_LIMIT_S,
        first_solution=FIRST_SOLUTION,
        local_search=LOCAL_SEARCH,
    )
    for route in solution.routes:
        executed = execute_with_capacity(route, points, demands, capacity)
        assert executed.depot_returns == 0
        assert executed.capacity_breached is False


def test_cvrp_rejects_a_bad_capacity_and_vehicle_count() -> None:
    _ortools_or_skip()
    with pytest.raises(ValueError, match="capacity_kg must be > 0"):
        solve_cvrp(
            LINE,
            [0.0, 1.0, 1.0, 1.0],
            capacity_kg=0.0,
            max_vehicles=1,
            time_limit_s=TEST_TIME_LIMIT_S,
            first_solution=FIRST_SOLUTION,
            local_search=LOCAL_SEARCH,
        )
    with pytest.raises(ValueError, match="max_vehicles must be >= 1"):
        solve_cvrp(
            LINE,
            [0.0, 1.0, 1.0, 1.0],
            capacity_kg=10.0,
            max_vehicles=0,
            time_limit_s=TEST_TIME_LIMIT_S,
            first_solution=FIRST_SOLUTION,
            local_search=LOCAL_SEARCH,
        )


def test_cvrp_rejects_mismatched_demands_and_unknown_knobs() -> None:
    _ortools_or_skip()
    with pytest.raises(ValueError, match="same length"):
        solve_cvrp(
            LINE,
            [0.0, 1.0],
            capacity_kg=10.0,
            max_vehicles=1,
            time_limit_s=TEST_TIME_LIMIT_S,
            first_solution=FIRST_SOLUTION,
            local_search=LOCAL_SEARCH,
        )
    with pytest.raises(ValueError, match="unknown first_solution strategy"):
        solve_cvrp(
            LINE,
            [0.0, 1.0, 1.0, 1.0],
            capacity_kg=10.0,
            max_vehicles=1,
            time_limit_s=TEST_TIME_LIMIT_S,
            first_solution="NOT_A_STRATEGY",
            local_search=LOCAL_SEARCH,
        )
    with pytest.raises(ValueError, match="unknown local_search metaheuristic"):
        solve_cvrp(
            LINE,
            [0.0, 1.0, 1.0, 1.0],
            capacity_kg=10.0,
            max_vehicles=1,
            time_limit_s=TEST_TIME_LIMIT_S,
            first_solution=FIRST_SOLUTION,
            local_search="NOT_A_METAHEURISTIC",
        )


def test_the_cvrp_solver_label_is_stable() -> None:
    _ortools_or_skip()
    solution = solve_cvrp(
        LINE,
        [0.0, 1.0, 1.0, 1.0],
        capacity_kg=1000.0,
        max_vehicles=2,
        time_limit_s=TEST_TIME_LIMIT_S,
        first_solution=FIRST_SOLUTION,
        local_search=LOCAL_SEARCH,
    )
    assert solution.solver == SOLVER_ORTOOLS_CVRP
