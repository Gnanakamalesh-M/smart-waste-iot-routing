"""Phase 4: the capacitated-VRP and routing-algorithm comparison, and its tables.

Three questions, and nothing else:

1. **How many trucks does capacity actually force?** Phase 2 imposed no limit, so a tour could
   collect more than a truck can carry.
2. **Does OR-Tools beat our hand-written 2-opt by a margin worth the dependency?** Reported
   plainly either way -- if the margin is small or negative, that is the finding.
3. **What does planning from noisy sensor weight cost?** Measured against an oracle that plans on
   the true weight, which is labelled *upper bound, not realistic* everywhere it appears.

### Every algorithm sees the identical bin sets

The instances come from **one** simulated day of **one** seed: the city is replayed to
``vrp.compare_day`` under the paper's own fusion strategy (d), and the bins it selects that day,
split by cluster, are the instances. Nearest neighbour, 2-opt, OR-Tools TSP and OR-Tools CVRP all
solve those same sets, so the comparison is paired and the only difference is the solver.

### Phase 2 and Phase 3 are untouched

Nothing here imports from or modifies ``benchmark.py`` or ``phase3.py``, and ``city_sim.py`` is
used only through its existing public functions. Phase 4 is a parallel path, exactly as Phase 3
was.

### Runtime lives in the results doc, never in stdout

No timestamp and no timing is printed, so two same-seed runs are byte-identical wherever the code
is deterministic. OR-Tools is the exception and says so: a wall-clock time limit makes its
distances machine-dependent, which is why no test and no determinism check covers them.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

from smartwaste.city_sim import build_layout, build_sensor_arrays, step_day
from smartwaste.config import Config
from smartwaste.demand import SCENARIO_SMOOTH
from smartwaste.routing.cluster import kmeans
from smartwaste.routing.tsp import Point
from smartwaste.routing.vrp import (
    SOLVER_NEAREST_NEIGHBOUR,
    SOLVER_ORTOOLS_CVRP,
    SOLVER_ORTOOLS_TSP,
    SOLVER_TWO_OPT,
    VrpSolution,
    execute_with_capacity,
    solve_cvrp,
    solve_nearest_neighbour,
    solve_tsp_ortools,
    solve_two_opt,
)
from smartwaste.seeds import child_seed
from smartwaste.strategies import RunContext, build_strategies

#: Index of the paper's own fusion strategy (d) in :func:`build_strategies`' fixed order. That is
#: the strategy whose selection supplies the bin sets, because it is the paper's method and so the
#: least arbitrary choice available.
PAPER_FUSION_INDEX = 3

#: A margin smaller than this is reported as "not meaningful". **Declared here, before any run**,
#: so the threshold cannot be chosen after seeing the number.
MEANINGFUL_MARGIN_PCT = 2.0

#: The two Table C rows.
PLAN_NOISY = "noisy"
PLAN_ORACLE = "oracle"

PLAN_LABELS = {
    PLAN_NOISY: "noisy plan (fused sensor weight) -> true-weight check [the realistic case]",
    PLAN_ORACLE: "oracle plan (TRUE weight) -> true-weight check [upper bound, not realistic]",
}

SOLVER_LABELS = {
    SOLVER_NEAREST_NEIGHBOUR: "nearest neighbour [Phase 2 baseline]",
    SOLVER_TWO_OPT: "nearest neighbour + 2-opt [Phase 2 baseline, the one to beat]",
    SOLVER_ORTOOLS_TSP: "OR-Tools, capacity-free [same problem as 2-opt]",
    SOLVER_ORTOOLS_CVRP: "OR-Tools CVRP, capacitated [a HARDER problem than the rows above]",
}

CSV_FIELDS = (
    "table",
    "row_key",
    "label",
    "seed",
    "day",
    "n_clusters",
    "n_bins_selected",
    "capacity_kg",
    "distance_km",
    "executed_distance_km",
    "detour_km",
    "n_trucks",
    "n_routes",
    "depot_returns",
    "capacity_breaches",
    "mean_load_kg",
)


@dataclass(frozen=True)
class ClusterInstance:
    """One cluster's bin set on the comparison day: identical input for every algorithm.

    ``points[0]`` is the depot; ``points[i]`` for ``i >= 1`` are the selected bins.
    ``noisy_kg`` and ``true_kg`` are indexed the same way, with ``[0] == 0.0`` for the depot.
    """

    cluster_id: int
    points: tuple[Point, ...]
    #: What a dispatcher has: the load cell's noisy weight reading.
    noisy_kg: tuple[float, ...]
    #: What the truck's own scale has. Never handed to a planner except in the labelled oracle row.
    true_kg: tuple[float, ...]

    @property
    def n_bins(self) -> int:
        return max(0, len(self.points) - 1)


@dataclass(frozen=True)
class AlgorithmResult:
    """One solver's result over all the instances, summed."""

    solver: str
    n_bins_selected: int
    total_distance_km: float
    n_trucks: int
    n_routes: int
    #: ``None`` for the capacity-free rows.
    capacity_kg: float | None = None
    mean_load_kg: float = 0.0


@dataclass(frozen=True)
class ExecutionResult:
    """One Table C row: a plan basis, executed against the true weight."""

    plan_basis: str
    capacity_kg: float
    planned_distance_km: float
    executed_distance_km: float
    depot_returns: int
    capacity_breaches: int
    n_trucks: int
    n_bins_selected: int

    @property
    def detour_km(self) -> float:
        return self.executed_distance_km - self.planned_distance_km


@dataclass(frozen=True)
class Phase4Results:
    """Everything one run produced."""

    seed: int
    day: int
    n_clusters: int
    n_bins_selected: int
    algorithms: tuple[AlgorithmResult, ...]
    capacitated: tuple[AlgorithmResult, ...]
    executions: tuple[ExecutionResult, ...]


def build_compare_instances(
    cfg: Config,
    seed: int,
    *,
    n_bins: int,
    days: int,
    n_clusters: int,
    compare_clusters: int,
    compare_day: int,
    scenario: str = SCENARIO_SMOOTH,
) -> tuple[ClusterInstance, ...]:
    """Replay the city to ``compare_day`` under strategy (d) and return its per-cluster bin sets.

    The replay deliberately mirrors ``city_sim.run_strategy``'s day loop -- waste arrives, every
    bin is read, the strategy chooses, the chosen bins are emptied -- using the *same* public
    functions, so the bins captured here are the bins Phase 2's own fusion strategy would have
    routed on that day. ``run_strategy`` itself is not reused because it returns aggregate
    distances rather than the selection, and widening it would change a Phase 2 entry point.

    Both weight channels are captured for the chosen bins:

    - ``noisy_kg`` from ``SensorReading.weight_kg`` -- the load cell, bias and noise included;
    - ``true_kg`` from the simulation's own accumulated weight.

    Only the first ``compare_clusters`` clusters are returned, so a ``--quick`` run is a strict
    prefix of the full one rather than a different sample.
    """
    if compare_day >= days:
        raise ValueError(f"compare_day must be < days, got {compare_day} and {days}")
    if compare_clusters < 1:
        raise ValueError(f"compare_clusters must be >= 1, got {compare_clusters}")

    layout = build_layout(
        cfg,
        seed,
        n_bins=n_bins,
        days=days,
        stuck_bin_fraction=cfg.sensors.stuck_bin_fraction,
        scenario=scenario,
    )
    clusters = kmeans(layout.locations, n_clusters, random.Random(child_seed(seed, "clusters")))
    arrays = build_sensor_arrays(cfg, layout, seed)
    strategy = build_strategies(cfg.bin, cfg.benchmark.fixed_schedule_days)[PAPER_FUSION_INDEX]
    strategy.begin_run(RunContext(calendar=layout.calendar, bin_cfg=cfg.bin, n_bins=layout.n_bins))

    weights = [0.0] * layout.n_bins
    was_overflowing = [False] * layout.n_bins
    ever_overflowed = [False] * layout.n_bins

    selected: list[int] = []
    readings_weight: list[float] = [0.0] * layout.n_bins
    true_weight: list[float] = [0.0] * layout.n_bins

    for day in range(compare_day + 1):
        fills, _ = step_day(cfg, layout, weights, was_overflowing, ever_overflowed, day)
        readings = tuple(arrays[i].read(fills[i], weights[i]) for i in range(layout.n_bins))
        selected = list(strategy.select(day, readings))
        if day == compare_day:
            readings_weight = [reading.weight_kg for reading in readings]
            true_weight = list(weights)
            break
        strategy.observe_collection(day, selected)
        for i in selected:
            weights[i] = 0.0
            was_overflowing[i] = False

    selected_set = set(selected)
    members: list[list[int]] = [[] for _ in range(clusters.n_clusters)]
    for bin_index, label in enumerate(clusters.labels):
        members[label].append(bin_index)

    instances: list[ClusterInstance] = []
    for cluster_id in range(min(compare_clusters, clusters.n_clusters)):
        chosen = [i for i in members[cluster_id] if i in selected_set]
        if not chosen:
            # A cluster with nothing selected is not an instance. Reported as a count, not padded
            # with an empty row that every solver would "tie" on.
            continue
        points: list[Point] = [layout.depot]
        points.extend(layout.locations[i] for i in chosen)
        instances.append(
            ClusterInstance(
                cluster_id=cluster_id,
                points=tuple(points),
                noisy_kg=(0.0, *(max(0.0, readings_weight[i]) for i in chosen)),
                true_kg=(0.0, *(true_weight[i] for i in chosen)),
            )
        )
    return tuple(instances)


def compare_algorithms(
    cfg: Config,
    instances: Sequence[ClusterInstance],
    *,
    time_limit_s: float,
    include_ortools: bool = True,
) -> tuple[AlgorithmResult, ...]:
    """Table A: the capacity-free comparison, on the identical instances.

    Nearest neighbour and 2-opt are Phase 2's own code. OR-Tools solves the same single-vehicle,
    no-capacity problem, so this row is the fair one; the capacitated rows answer a harder
    question and are labelled so.
    """
    solutions: dict[str, list[VrpSolution]] = {
        SOLVER_NEAREST_NEIGHBOUR: [],
        SOLVER_TWO_OPT: [],
    }
    if include_ortools:
        solutions[SOLVER_ORTOOLS_TSP] = []

    for instance in instances:
        solutions[SOLVER_NEAREST_NEIGHBOUR].append(
            solve_nearest_neighbour(instance.points, instance.noisy_kg)
        )
        solutions[SOLVER_TWO_OPT].append(
            solve_two_opt(
                instance.points, instance.noisy_kg, max_iters=cfg.routing.two_opt_max_iters
            )
        )
        if include_ortools:
            solutions[SOLVER_ORTOOLS_TSP].append(
                solve_tsp_ortools(
                    instance.points,
                    instance.noisy_kg,
                    time_limit_s=time_limit_s,
                    first_solution=cfg.vrp.first_solution_strategy,
                    local_search=cfg.vrp.local_search_metaheuristic,
                )
            )

    n_selected = sum(instance.n_bins for instance in instances)
    return tuple(
        _summarise(solver, found, n_selected, capacity_kg=None)
        for solver, found in solutions.items()
    )


def compare_capacities(
    cfg: Config,
    instances: Sequence[ClusterInstance],
    capacities_kg: Sequence[float],
    *,
    time_limit_s: float,
) -> tuple[AlgorithmResult, ...]:
    """Table B: the capacitated solve at each capacity in the sweep.

    The sweep exists because ``cost.truck_capacity_kg`` is an invented number. One capacity would
    let the truck-count result rest on that one guess -- the same reason Phase 2 sweeps
    ``benchmark.fault_fractions``.
    """
    results: list[AlgorithmResult] = []
    for capacity in capacities_kg:
        solutions = [
            solve_cvrp(
                instance.points,
                instance.noisy_kg,
                capacity_kg=capacity,
                max_vehicles=cfg.vrp.max_trucks_per_cluster,
                time_limit_s=time_limit_s,
                first_solution=cfg.vrp.first_solution_strategy,
                local_search=cfg.vrp.local_search_metaheuristic,
            )
            for instance in instances
        ]
        n_selected = sum(instance.n_bins for instance in instances)
        results.append(_summarise(SOLVER_ORTOOLS_CVRP, solutions, n_selected, capacity_kg=capacity))
    return tuple(results)


def compare_plan_bases(
    cfg: Config,
    instances: Sequence[ClusterInstance],
    *,
    capacity_kg: float,
    time_limit_s: float,
) -> tuple[ExecutionResult, ...]:
    """Table C: the noisy plan and the oracle plan, both executed against the TRUE weight.

    The two rows differ in **one** thing -- which weight vector the solver was given:

    - ``noisy``: the fused sensor weight. The realistic case. Breaches are possible and are the
      measurement.
    - ``oracle``: the true weight. **Upper bound, not realistic.** Because the solver's capacity
      constraint is enforced on exactly the weights the scale then checks, this row's breach count
      and depot returns are zero *by construction*. It is a reference line, not a result, and
      ``tests/test_vrp.py`` pins that invariant so the claim is tested rather than asserted.

    The gap between the rows is what planning from noisy sensors costs.
    """
    results: list[ExecutionResult] = []
    for basis in (PLAN_NOISY, PLAN_ORACLE):
        planned_total = 0.0
        executed_total = 0.0
        depot_returns = 0
        breaches = 0
        trucks = 0
        for instance in instances:
            demands = instance.noisy_kg if basis == PLAN_NOISY else instance.true_kg
            solution = solve_cvrp(
                instance.points,
                demands,
                capacity_kg=capacity_kg,
                max_vehicles=cfg.vrp.max_trucks_per_cluster,
                time_limit_s=time_limit_s,
                first_solution=cfg.vrp.first_solution_strategy,
                local_search=cfg.vrp.local_search_metaheuristic,
            )
            trucks += solution.n_trucks
            for route in solution.routes:
                executed = execute_with_capacity(
                    route, instance.points, instance.true_kg, capacity_kg
                )
                planned_total += executed.planned_distance_km
                executed_total += executed.distance_km
                depot_returns += executed.depot_returns
                if executed.capacity_breached:
                    breaches += 1
        results.append(
            ExecutionResult(
                plan_basis=basis,
                capacity_kg=capacity_kg,
                planned_distance_km=planned_total,
                executed_distance_km=executed_total,
                depot_returns=depot_returns,
                capacity_breaches=breaches,
                n_trucks=trucks,
                n_bins_selected=sum(instance.n_bins for instance in instances),
            )
        )
    return tuple(results)


def _summarise(
    solver: str,
    solutions: Sequence[VrpSolution],
    n_bins_selected: int,
    *,
    capacity_kg: float | None,
) -> AlgorithmResult:
    """Sum one solver's per-instance solutions into a single table row."""
    routes = [route for solution in solutions for route in solution.routes if route.n_bins > 0]
    loads = [route.planned_load_kg for route in routes]
    return AlgorithmResult(
        solver=solver,
        n_bins_selected=n_bins_selected,
        total_distance_km=sum(solution.total_distance_km for solution in solutions),
        n_trucks=sum(solution.n_trucks for solution in solutions),
        n_routes=len(routes),
        capacity_kg=capacity_kg,
        mean_load_kg=sum(loads) / len(loads) if loads else 0.0,
    )


# ------------------------------------------------------------------------------------------------
# Rendering. Fixed decimal places everywhere, so output cannot drift between runs.
# ------------------------------------------------------------------------------------------------


def _fmt(value: float, places: int = 2) -> str:
    """Format a float with a fixed number of places, so output cannot drift between runs."""
    return f"{value:.{places}f}"


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def render_algorithms(results: Phase4Results) -> str:
    """Table A: capacity-free, same problem for every solver. The fair comparison."""
    title = (
        f"### Table A - routing algorithm comparison, capacity-free "
        f"(seed {results.seed}, day {results.day}, {results.n_clusters} clusters, "
        f"{results.n_bins_selected} bins)"
    )
    note = (
        "Every solver sees the **identical** bin sets, so this is paired. All three solve the\n"
        "same single-vehicle, no-capacity problem. `% vs 2-opt` is positive when the solver is\n"
        "**shorter** than 2-opt."
    )
    baseline = next(
        (r for r in results.algorithms if r.solver == SOLVER_TWO_OPT),
        None,
    )
    header = ["solver", "distance km", "routes", "trucks", "% vs 2-opt"]
    rows = []
    for r in results.algorithms:
        if baseline is None or baseline.total_distance_km <= 0:
            margin = "n/a"
        else:
            delta = baseline.total_distance_km - r.total_distance_km
            margin = _fmt(100.0 * delta / baseline.total_distance_km)
        rows.append(
            [
                SOLVER_LABELS.get(r.solver, r.solver),
                _fmt(r.total_distance_km),
                str(r.n_routes),
                str(r.n_trucks),
                margin,
            ]
        )
    return f"{title}\n\n{note}\n\n{_table(header, rows)}"


def render_capacities(results: Phase4Results) -> str:
    """Table B: the capacitated sweep.

    The `% vs capacity-free 2-opt` column is **not a fair comparison** and the note says so: a
    capacitated solution satisfies a constraint the 2-opt tour ignores, so it *should* be longer.
    The column is there to size that cost, not to declare a winner.
    """
    if not results.capacitated:
        return ""
    title = (
        f"### Table B - capacitated VRP, capacity sweep "
        f"(seed {results.seed}, day {results.day}, {results.n_bins_selected} bins)"
    )
    note = (
        "**`% vs capacity-free 2-opt` is NOT a fair comparison.** The capacitated solver obeys a\n"
        "truck-payload limit the 2-opt tour ignores entirely, so a longer distance here is the\n"
        "*correct* answer to a harder question, not a worse answer to the same one. Negative\n"
        "values are printed as-is.\n\n"
        "Capacity is swept because `cost.truck_capacity_kg` is **invented and uncited**; 8000 kg\n"
        "is the config value and was **not** changed to tidy these truck counts."
    )
    baseline = next((r for r in results.algorithms if r.solver == SOLVER_TWO_OPT), None)
    header = [
        "capacity kg",
        "distance km",
        "trucks used",
        "routes",
        "mean load kg",
        "% vs capacity-free 2-opt [NOT a fair comparison]",
    ]
    rows = []
    for r in results.capacitated:
        if baseline is None or baseline.total_distance_km <= 0:
            margin = "n/a"
        else:
            delta = baseline.total_distance_km - r.total_distance_km
            margin = _fmt(100.0 * delta / baseline.total_distance_km)
        rows.append(
            [
                _fmt(r.capacity_kg or 0.0),
                _fmt(r.total_distance_km),
                str(r.n_trucks),
                str(r.n_routes),
                _fmt(r.mean_load_kg),
                margin,
            ]
        )
    return f"{title}\n\n{note}\n\n{_table(header, rows)}"


def render_executions(results: Phase4Results) -> str:
    """Table C: noisy plan vs oracle plan, both checked against the true weight."""
    if not results.executions:
        return ""
    capacity = results.executions[0].capacity_kg
    title = (
        f"### Table C - planned vs executed: what planning from noisy sensor weight costs "
        f"(capacity {_fmt(capacity)} kg, seed {results.seed}, day {results.day})"
    )
    note = (
        "Both rows solve the **same bin sets** at the **same capacity** and are checked against\n"
        "the **same true weights**. They differ in one thing: which weight vector the solver was\n"
        "given.\n\n"
        "- **noisy plan** - the fused sensor weight. This is the realistic case: a dispatcher has\n"
        "  the load cell's reading, not the landfill scale's.\n"
        "- **oracle plan** - the true weight. **UPPER BOUND, NOT REALISTIC.** Its breach count\n"
        "  and depot returns are zero *by construction*, because capacity is enforced on exactly\n"
        "  the weights that are then checked. It is a reference line, not a Phase 4 result.\n\n"
        "`detour km` is executed minus planned: the distance the depot returns cost. A breach is\n"
        "counted per route whose plan did not hold."
    )
    header = [
        "plan basis",
        "planned km",
        "executed km",
        "detour km",
        "depot returns",
        "capacity breaches",
        "trucks used",
    ]
    rows = [
        [
            PLAN_LABELS.get(r.plan_basis, r.plan_basis),
            _fmt(r.planned_distance_km),
            _fmt(r.executed_distance_km),
            _fmt(r.detour_km),
            str(r.depot_returns),
            str(r.capacity_breaches),
            str(r.n_trucks),
        ]
        for r in results.executions
    ]
    return f"{title}\n\n{note}\n\n{_table(header, rows)}"


def ortools_verdict(results: Phase4Results) -> list[str]:
    """Does OR-Tools beat 2-opt by a margin worth the dependency?

    The threshold is :data:`MEANINGFUL_MARGIN_PCT`, declared as a module constant before any run
    so it cannot be picked after seeing the margin. A small or negative margin is stated plainly.
    """
    baseline = next((r for r in results.algorithms if r.solver == SOLVER_TWO_OPT), None)
    ortools = next((r for r in results.algorithms if r.solver == SOLVER_ORTOOLS_TSP), None)
    if baseline is None or ortools is None:
        return ["  OR-Tools was not run, so there is no comparison to report."]
    if baseline.total_distance_km <= 0:
        return ["  the 2-opt baseline covered no distance, so no margin can be computed."]

    delta = baseline.total_distance_km - ortools.total_distance_km
    margin = 100.0 * delta / baseline.total_distance_km
    lines = [
        f"  threshold for 'meaningful', declared before the run: {_fmt(MEANINGFUL_MARGIN_PCT)}%",
        f"  2-opt {_fmt(baseline.total_distance_km)} km vs OR-Tools "
        f"{_fmt(ortools.total_distance_km)} km on the same bin sets",
    ]
    if margin > MEANINGFUL_MARGIN_PCT:
        lines.append(
            f"  OR-Tools BEATS 2-opt by {_fmt(margin)}% -- above the threshold, so the "
            f"dependency buys something on this instance set"
        )
    elif margin >= 0:
        lines.append(
            f"  OR-Tools beats 2-opt by only {_fmt(margin)}%, BELOW the "
            f"{_fmt(MEANINGFUL_MARGIN_PCT)}% threshold -- stated plainly: on these instances the "
            f"hand-written 2-opt is essentially as good, and OR-Tools does not earn its place on "
            f"distance alone"
        )
    else:
        lines.append(
            f"  OR-Tools is WORSE than 2-opt by {_fmt(-margin)}% -- printed as-is. On these "
            f"instances the hand-written 2-opt wins"
        )
    lines.append(
        "  NOTE: OR-Tools ran under a WALL-CLOCK time limit, so this margin is "
        "same-machine-only and will differ on other hardware."
    )
    return lines


def capacity_verdict(results: Phase4Results) -> list[str]:
    """How many trucks capacity forces, against Phase 2's one-truck-per-cluster assumption."""
    if not results.capacitated:
        return ["  the capacitated solve was not run."]
    lines = [
        f"  Phase 2 assumed ONE truck per cluster and no capacity limit "
        f"({results.n_clusters} clusters routed here).",
    ]
    for r in results.capacitated:
        lines.append(
            f"  capacity {_fmt(r.capacity_kg or 0.0)} kg: {r.n_trucks} trucks over "
            f"{r.n_routes} routes for {r.n_bins_selected} bins "
            f"(mean load {_fmt(r.mean_load_kg)} kg)"
        )
    lines.append(
        "  Every capacity here is invented and uncited, so this is a sensitivity table, "
        "not a fleet-size recommendation."
    )
    return lines


def noise_cost_verdict(results: Phase4Results) -> list[str]:
    """What the noisy plan costs against the oracle. The oracle is a bound, not a result."""
    noisy = next((r for r in results.executions if r.plan_basis == PLAN_NOISY), None)
    oracle = next((r for r in results.executions if r.plan_basis == PLAN_ORACLE), None)
    if noisy is None or oracle is None:
        return ["  the plan-basis comparison was not run."]

    lines = [
        f"  noisy plan:  executed {_fmt(noisy.executed_distance_km)} km, "
        f"{noisy.depot_returns} depot returns, {noisy.capacity_breaches} breached routes, "
        f"{noisy.n_trucks} trucks",
        f"  oracle plan: executed {_fmt(oracle.executed_distance_km)} km, "
        f"{oracle.depot_returns} depot returns, {oracle.capacity_breaches} breached routes, "
        f"{oracle.n_trucks} trucks  [UPPER BOUND, NOT REALISTIC]",
    ]
    if oracle.executed_distance_km > 0:
        gap = 100.0 * (
            (noisy.executed_distance_km - oracle.executed_distance_km) / oracle.executed_distance_km
        )
        lines.append(
            f"  planning from noisy sensor weight cost {_fmt(gap)}% more executed distance "
            f"than planning from the truth"
        )
    if noisy.capacity_breaches == 0:
        lines.append(
            "  the noisy plan breached capacity on ZERO routes, so on this instance set sensor "
            "noise did not change the outcome -- reported as it came out, not as a win"
        )
    lines.append(
        "  The oracle row exists only to bound the noisy one. No real deployment has the true "
        "weight before the truck arrives."
    )
    return lines


def csv_rows(results: Phase4Results) -> list[dict[str, object]]:
    """Flatten every table into one CSV, one row per table row."""
    rows: list[dict[str, object]] = []
    for r in results.algorithms:
        rows.append(_algo_row("A_algorithms", r, results))
    for r in results.capacitated:
        rows.append(_algo_row("B_capacities", r, results))
    for e in results.executions:
        rows.append(
            {
                "table": "C_plan_basis",
                "row_key": e.plan_basis,
                "label": PLAN_LABELS.get(e.plan_basis, e.plan_basis),
                "seed": results.seed,
                "day": results.day,
                "n_clusters": results.n_clusters,
                "n_bins_selected": e.n_bins_selected,
                "capacity_kg": _fmt(e.capacity_kg),
                "distance_km": _fmt(e.planned_distance_km),
                "executed_distance_km": _fmt(e.executed_distance_km),
                "detour_km": _fmt(e.detour_km),
                "n_trucks": e.n_trucks,
                "n_routes": "",
                "depot_returns": e.depot_returns,
                "capacity_breaches": e.capacity_breaches,
                "mean_load_kg": "",
            }
        )
    return rows


def _algo_row(table: str, r: AlgorithmResult, results: Phase4Results) -> dict[str, object]:
    return {
        "table": table,
        "row_key": r.solver,
        "label": SOLVER_LABELS.get(r.solver, r.solver),
        "seed": results.seed,
        "day": results.day,
        "n_clusters": results.n_clusters,
        "n_bins_selected": r.n_bins_selected,
        "capacity_kg": "" if r.capacity_kg is None else _fmt(r.capacity_kg),
        "distance_km": _fmt(r.total_distance_km),
        "executed_distance_km": "",
        "detour_km": "",
        "n_trucks": r.n_trucks,
        "n_routes": r.n_routes,
        "depot_returns": "",
        "capacity_breaches": "",
        "mean_load_kg": _fmt(r.mean_load_kg),
    }


def quick_banner() -> str:
    """The smoke-run warning, prepended ONCE to a report rather than to every table."""
    return (
        "> **SMOKE RUN, NOT FOR PUBLICATION.** Smaller than the default run. Every number"
        " in `docs/PHASE4_RESULTS.md` comes from the full run only.\n"
    )
