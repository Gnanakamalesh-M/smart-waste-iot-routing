# Phase 4 results — capacitated VRP, routing-algorithm comparison, real Chennai roads

## How these numbers were produced

```
python scripts\run_phase4_vrp.py --seed 42 --quick
python scripts\run_roads.py --seed 42 --quick
python scripts\run_phase4_vrp.py --seed 42 --quick --capacities 150 400 8000
```

> ### The publication-size runs have NOT been run
>
> **Every table below is a `--quick` SMOKE RUN** and is labelled as such in its own heading.
> The full-size commands are given under [To produce the full run](#to-produce-the-full-run) and
> are yours to run; CLAUDE.md's runtime rule forbids a long job inside the session that built
> this, after Phase 3 lost a 2.3-hour run that way.
>
> **This is a deliberate departure from how Phase 3 handled the same situation.** Phase 3 omitted
> its un-run tables entirely rather than publish smoke numbers. Here the smoke tables are
> included, because what they establish is **mechanism and direction** — does capacity bind, does
> OR-Tools beat 2-opt, how far does a road detour go — and all three of those are visible at
> smoke size. What they do **not** establish is any city-scale **magnitude**. Every heading says
> `SMOKE RUN`, and no number below should be quoted as a Phase 4 city-scale result.

- **Seed:** 42. Every sub-seed is derived from the master seed with sha256.
- **Code commit:** `d7b8734` ("Phase 4: make the --quick banner say what it means"), on top of
  `20031b0` ("Phase 4: capacitated VRP, routing-algorithm comparison, real Chennai roads"). The
  working tree was clean when these ran (`git status --porcelain` printed nothing), so every
  number below comes from committed code.
- **Environment:** Python 3.11, Windows. **ortools 9.15.6755, osmnx 2.1.1, networkx 3.6.1**.
- **Config:** every constant from `config/default.yaml` as committed. **Nothing was tuned.**
  `cost.truck_capacity_kg` is still the 8000.0 it has been since Phase 0.
- **Runtimes** (in this document only — never printed to stdout, so same-seed runs stay
  byte-identical): the VRP smoke run **24.7 s**, the roads smoke run **22.3 s** including the
  first OpenStreetMap download.

## Road data attribution

Road geometry is from **OpenStreetMap**. © OpenStreetMap contributors, licensed under the
**Open Database License (ODbL) v1.0** — https://www.openstreetmap.org/copyright

## Scope of Phase 4

**In:** truck capacity as a real constraint; multi-truck capacitated VRP via Google OR-Tools;
nearest neighbour and 2-opt kept as baselines and reused unchanged from Phase 2; an
algorithm comparison on identical bin sets; noisy-sensor-weight planning checked against true
weight, with a return-to-depot-and-resume policy when a plan does not fit; a labelled oracle
reference row; real road distances for one bounded Chennai area via OSMnx with a disk cache and a
loud Euclidean fallback.

**Out, deliberately:** **no Yellow-bin logic** (still unbuilt, now carried into Phase 5 — SPEC §7
lists it under Phase 3); no ML changes; no dashboard; no change to any Phase 2 or Phase 3 entry
point, constant or output; no time windows, no shift limits, no traffic, no driver-hour caps.

## Phase 2 and Phase 3 are untouched — checked, not assumed

Phase 4 is a **parallel path**, exactly as Phase 3 was. Verified three ways:

- **No Phase 2/3 source file changed.** `git status --porcelain` over `city_sim.py`,
  `strategies.py`, `benchmark.py`, `phase3.py`, `routing/tsp.py`, `routing/cluster.py`,
  `cost.py` and `demand.py` printed **nothing**. Phase 4's capacity logic lives in new modules
  (`routing/vrp.py`, `routing/roads.py`, `phase4.py`, `roads_eval.py`) and two new runners.
- **No pre-existing test file was edited.** `git diff --name-only 6443be0 HEAD -- tests/` lists
  exactly three paths, all **new**: `tests/test_config_phase4.py`, `tests/test_roads.py`,
  `tests/test_vrp.py`. The `vrp`/`roads` config tests went into a new file specifically so this
  check stays clean.
- **`tests/test_demand.py::test_smooth_scenario_reproduces_the_phase2_numbers_exactly`**, which
  pins Phase 2's per-strategy totals as literals captured from the pre-Phase-3 tree, **passes
  unchanged**.

The full Phase 2 benchmark was **not** re-run this phase (it takes 7.8 min and the runtime rule
applies); the literal-pinning test plus the untouched-source check are what stand behind the claim
instead. The re-run command is in [To produce the full run](#to-produce-the-full-run).

## 1. Routing algorithm comparison — does OR-Tools beat our 2-opt?

Every solver sees the **identical** bin sets: the city is replayed to `vrp.compare_day = 14` under
the paper's own fusion strategy (d), and the bins it selects that day, split by cluster, are the
instances. So the comparison is paired and the only difference is the solver.

### Table A — capacity-free, the fair comparison [SMOKE RUN, NOT FOR PUBLICATION]

Seed 42, day 14, **5 clusters, 46 bins**, 1 s solver budget per instance.
`% vs 2-opt` is positive when the solver is **shorter** than 2-opt.

| solver | distance km | routes | trucks | % vs 2-opt |
|---|---|---|---|---|
| nearest neighbour [Phase 2 baseline] | 594.86 | 5 | 5 | -5.88 |
| nearest neighbour + 2-opt [Phase 2 baseline, the one to beat] | 561.80 | 5 | 5 | 0.00 |
| OR-Tools, capacity-free [same problem as 2-opt] | 560.19 | 5 | 5 | 0.29 |

Verdict lines, verbatim from the same run:

```
Verdict: does OR-Tools beat the hand-written 2-opt by a margin worth the dependency?
  threshold for 'meaningful', declared before the run: 2.00%
  2-opt 561.80 km vs OR-Tools 560.19 km on the same bin sets
  OR-Tools beats 2-opt by only 0.29%, BELOW the 2.00% threshold -- stated plainly: on these instances the hand-written 2-opt is essentially as good, and OR-Tools does not earn its place on distance alone
  NOTE: OR-Tools ran under a WALL-CLOCK time limit, so this margin is same-machine-only and will differ on other hardware.
```

### Reading this honestly

**OR-Tools does not meaningfully beat our hand-written 2-opt here — 0.29%, against a 2% threshold
that was declared in the code before any run.** `MEANINGFUL_MARGIN_PCT = 2.0` is a module constant
in `src/smartwaste/phase4.py`, so the bar could not be moved after seeing the number.

- **2-opt does earn its keep over plain nearest neighbour**: 5.88%, which is twenty times the
  OR-Tools margin. The cheap local search is where the gain is; the industrial solver adds almost
  nothing on top.
- **Why that is unsurprising at this size.** These are 5 instances averaging ~9 bins each. A
  9-stop tour is close to exhaustively solvable and 2-opt very nearly finds the optimum, so there
  is almost no headroom left for OR-Tools to take. The margin would plausibly widen on larger
  per-cluster bin sets — but that is a **prediction from the problem's shape, not a measurement**,
  and the full run is what would settle it.
- **So Phase 4 does not demonstrate that the `ortools` dependency pays for itself on distance.**
  It is kept because it is what makes the *capacitated multi-truck* problem solvable at all —
  which 2-opt cannot express — not because of the 0.29%.

## 2. Truck capacity — and the finding that it never binds

### Table B — capacity sweep [SMOKE RUN, NOT FOR PUBLICATION]

Seed 42, day 14, 46 bins. `cost.truck_capacity_kg = 8000.0` is the committed config value and was
**not** changed.

| capacity kg | distance km | trucks used | routes | mean load kg | % vs capacity-free 2-opt **[NOT a fair comparison]** |
|---|---|---|---|---|---|
| 8000.00 | 560.19 | 5 | 5 | 296.24 | 0.29 |

**`% vs capacity-free 2-opt` is NOT a fair comparison** and is labelled so in the runner's own
output: the capacitated solver obeys a payload limit the 2-opt tour ignores entirely, so a longer
distance there would be the *correct* answer to a harder question, not a worse answer to the same
one.

```
Verdict: how many trucks does capacity actually force?
  Phase 2 assumed ONE truck per cluster and no capacity limit (5 clusters routed here).
  capacity 8000.00 kg: 5 trucks over 5 routes for 46 bins (mean load 296.24 kg)
  Every capacity here is invented and uncited, so this is a sensitivity table, not a fleet-size recommendation.
```

### Reading this honestly — this is the main result of Phase 4

**At the project's own numbers, truck capacity is not a binding constraint, and Phase 2's
"no capacity limit" limitation turns out to have flattered nothing.**

The arithmetic is simple and it is worth stating plainly rather than burying:

- `bin.w_max_kg = 50.0`, so a full bin holds about 50 kg;
- strategy (d) selects roughly 9 bins per cluster per day at these settings;
- so a cluster's whole day is about **296 kg** against an **8000 kg** truck — **3.7% of capacity.**

A truck would have to visit **~160 full bins in one tour** before capacity mattered. Phase 2's
`docs/PHASE2_RESULTS.md` listed "no truck capacity limit, so a tour may collect more waste than a
real truck could carry, which flatters every strategy that visits many bins" as a limitation. On
these constants **that limitation was not actually biting**, and Phase 4's honest contribution is
to have measured that rather than to have fixed a problem.

**Why the sweep found nothing.** `vrp.capacity_sweep_kg = [4000, 8000, 16000]` was chosen before
any run, to bracket the config value. All three sit far above 296 kg, so all three are
non-binding and the sweep is flat. That is a real weakness of the sweep's design, and the
constants were **not** retuned afterwards to make it interesting.

### Table B(ii) — a binding capacity, to show the mechanism works [SMOKE RUN; 150 kg is NOT a realistic truck]

Because the config sweep is flat, this extra run drops capacity below the per-cluster load so the
constraint actually engages. **150 kg is three bins' worth — it is not a refuse truck.** This row
demonstrates that the capacity dimension and the multi-truck solve function; it is **not** a Phase
4 result about fleets.

```
python scripts\run_phase4_vrp.py --seed 42 --quick --capacities 150 400 8000
```

| capacity kg | distance km | trucks used | routes | mean load kg | % vs capacity-free 2-opt **[NOT a fair comparison]** |
|---|---|---|---|---|---|
| 150.00 **[binding; not a realistic truck]** | 1192.97 | 14 | 14 | 105.80 | -112.35 |
| 400.00 | 560.19 | 5 | 5 | 296.24 | 0.29 |
| 8000.00 **[the config value]** | 560.19 | 5 | 5 | 296.24 | 0.29 |

At 150 kg the solver is forced from 5 trucks to **14**, and total distance **more than doubles**
(-112.35%, printed as-is). That negative number is the capacity constraint being satisfied, not a
solver failure: more trucks means more depot trips. It is also the shape of the cost Phase 2 never
had to pay.

## 3. Planned vs executed — what planning from noisy sensor weight costs

### Table C — noisy plan against an oracle plan [SMOKE RUN, NOT FOR PUBLICATION]

Both rows solve the **same bin sets** at the **same capacity** (8000 kg, the config value) and are
checked against the **same true weights**. They differ in exactly one thing: which weight vector
the solver was given.

| plan basis | planned km | executed km | detour km | depot returns | capacity breaches | trucks used |
|---|---|---|---|---|---|---|
| **noisy plan** (fused sensor weight) → true-weight check **[the realistic case]** | 560.19 | 560.19 | 0.00 | 0 | 0 | 5 |
| **oracle plan** (TRUE weight) → true-weight check **[UPPER BOUND, NOT REALISTIC]** | 560.19 | 560.19 | 0.00 | 0 | 0 | 5 |

- **noisy plan** — the fused load-cell reading, noise and bias included. This is what a real
  dispatcher has, and it is the only row that is a Phase 4 result.
- **oracle plan** — the true accumulated weight. **Upper bound, not realistic.** Its breach count
  and depot returns are zero *by construction*: capacity is enforced on exactly the weights that
  are then checked. It is a reference line, not a result. No deployment knows a bin's true weight
  before the truck arrives.

```
Verdict: what does planning from noisy sensor weight cost?
  noisy plan:  executed 560.19 km, 0 depot returns, 0 breached routes, 5 trucks
  oracle plan: executed 560.19 km, 0 depot returns, 0 breached routes, 5 trucks  [UPPER BOUND, NOT REALISTIC]
  planning from noisy sensor weight cost 0.00% more executed distance than planning from the truth
  the noisy plan breached capacity on ZERO routes, so on this instance set sensor noise did not change the outcome -- reported as it came out, not as a win
  The oracle row exists only to bound the noisy one. No real deployment has the true weight before the truck arrives.
```

### Reading this honestly

**The two rows are identical, and that is a null result, not a win for the sensors.**

It would be easy to present "noisy planning costs 0% extra" as sensor fusion being good enough for
dispatch. It is not that. It follows directly from the finding above: **capacity never binds, so
the planner's weight estimate never changes a routing decision.** A load-cell error of a few kg
cannot push a 296 kg tour past an 8000 kg limit. Sensor noise is simply not being *tested* here.

The honest statement is: **Phase 4 did not measure what planning from noisy sensor weight costs,
because at these constants nothing was riding on the weight estimate.** The machinery to measure it
exists and is tested — `tests/test_vrp.py` pins the depot-return policy on hand-built cases where
the plan genuinely fails, and pins the oracle's zero-breach invariant both directly and end to end
through the real solver — but the city's own numbers never exercise it. A capacity near the
per-cluster load, or much larger bins, would.

## 4. Real road distance vs Euclidean, on the same bin set

```
python scripts\run_roads.py --seed 42 --quick
```

### Scope [SMOKE RUN, NOT FOR PUBLICATION]

| | |
|---|---|
| Area | T. Nagar, Chennai, India |
| Bounding box | south 13.02833, west 80.22027, north 13.05527, east 80.24793 (~3 × 3 km) |
| Network filter | `drive` |
| Road graph | **1551 nodes, 3657 edges**, downloaded from OpenStreetMap and cached |
| Bins | **50** synthetic, snapping to **49** distinct road nodes |
| Cache | `cache/osmnx/roads_13.02833_80.22027_13.05527_80.24793_drive.graphml` (1.52 MB, gitignored) |

> **BIN LOCATIONS ARE SYNTHETIC.** Uniform-random points inside the bounding box, snapped to the
> nearest drivable road node. **The roads are real; the bins are not.** Nobody surveyed a bin site
> in T. Nagar.

> **NOT comparable with Phase 2.** Phase 2 is 5000 bins on a 100 × 100 km square under Euclidean
> distance. This is a ~3 km area with 50 bins on real roads. There is deliberately **no
> "% vs Phase 2" column anywhere** in this section. The ratio below is computed on **this** bin set
> under **two** metrics, which is the only comparison that isolates the distance metric.

### Detour ratio: road distance / straight-line distance, per bin pair [SMOKE RUN]

| pairs | zero pairs | unreachable pairs | mean | median | p10 | p90 | min | max | aggregate |
|---|---|---|---|---|---|---|---|---|---|
| 1274 | 1 | 0 | 1.6438 | 1.4514 | 1.2340 | 2.1370 | 1.0000 | 12.0161 | 1.5364 |

`aggregate` is `sum(road) / sum(straight)` — what a total-distance figure would show. It is not
the same as the mean of the per-pair ratios, which is why both are reported. Pairs are **skipped
and counted, never silently dropped**: the 1 `zero pair` is two bins that snapped to the same road
node (ratio undefined); `unreachable pairs` would be pairs with no road path inside the extract,
and there were none.

### Tour length under each metric: same bins, same algorithm, same depot [SMOKE RUN]

| metric | tour km | ratio vs straight-line |
|---|---|---|
| straight-line (haversine) | 17.321 | 1.0000 |
| real roads (OSM shortest path) | 31.742 | **1.8326** |

Both tours are built by the **identical** nearest-neighbour-then-2-opt code reading a distance
matrix, from the same depot (the area centre, snapped to a road node). The only difference between
the two numbers is the matrix. `tests/test_vrp.py` pins that the matrix tours agree with Phase 2's
coordinate TSP when fed a Euclidean matrix, so the two halves really are the same algorithm.

```
Verdict: by how much does a straight line under-state a road distance?
  over 1274 bin pairs on real drive roads in T. Nagar, Chennai, India:
  straight-line distance under-states road distance by a factor of 1.4514 at the median (1.2340 at p10, 2.1370 at p90, 1.0000 to 12.0161 overall)
  on total distance the factor is 1.5364; on a routed tour it is 1.8326
  1 pairs snapped to the same road node (ratio undefined) and were excluded
  This is ONE area at ONE bbox size with SYNTHETIC bin locations. It is not a city-wide figure and it does not transfer to Phase 2's 100 km grid.
```

### Reading this honestly

**This is the one Phase 4 measurement that answers a question `docs/ASSUMPTIONS.md` had been
carrying open since Phase 0** — "a straight line under-states road distance by an amount we have
not yet measured". On this area, with synthetic bins:

- **Median detour factor 1.45**, i.e. the drive is about 45% longer than the crow flies.
- **The spread is wide and matters more than the median.** p10 is 1.23, p90 is 2.14, and the
  maximum is **12.02**. A single average would have hidden that some pairs are nearly straight and
  at least one doubles back enormously — plausibly a pair separated by a railway line, a one-way
  system, or the edge of the extract. Reporting one number here would have been the dishonest
  option.
- **Three different "the detour factor" numbers come out of the same data**: 1.45 (median pair),
  1.54 (total distance), 1.83 (routed tour). They are not interchangeable. The tour ratio is
  highest because a tour chains many short legs, and short legs are where the road network's
  detours are proportionally worst. The **pairwise** figure is the primary one because it skips
  unreachable pairs outright; the tour figure substitutes a large finite stand-in for any
  unreachable leg so 2-opt can compare swaps at all.
- **What this does NOT license.** It does not mean Phase 2's distances should be multiplied by
  1.45. Phase 2's grid is 100 km of uniform-random points — a completely different geometry, at a
  scale where long legs dominate and the proportional detour would be *smaller*. Applying this
  factor to Phase 2's numbers would be exactly the kind of false comparability this phase is meant
  to avoid, and no table here does it.

### The Euclidean fallback was tested, not assumed

`--no-download` with no cache present drives the failure path. It prints a banner to **stderr**
(not stdout, so same-seed stdout stays byte-identical), returns no graph, and every table is then
labelled `EUCLIDEAN FALLBACK, NOT ROAD DISTANCES` with every ratio exactly 1.0 — the visible
signature of the fallback rather than a finding about roads:

```
================================================================================================
ROADS: ROAD GRAPH UNAVAILABLE -- FALLING BACK TO EUCLIDEAN (osmnx is not installed (No module named 'osmnx'))
================================================================================================
```

The cache was also verified to serve the graph with **no network access at all**: a second
`--quick --no-download` run reported `1551 nodes, 3657 edges (cache: ...)` and reproduced the
median 1.4514 exactly.

## Verification (Windows cmd, from the repo root)

```
.venv\Scripts\activate
python -m pip install -e ".[dev,ml,roads]"

python -m ruff check .
python -m ruff format --check .
python -m mypy

python -m pytest -m unit
python -m pytest -m sim
python -m pytest -m ml
python -m pytest -m routing
python -m pytest -m slow
```

**473 tests pass** (231 `unit`, 125 `sim`, 14 `ml`, 107 `routing`, 5 `slow`; a few carry two
markers, so the chunk counts sum to more than 473). `ruff check`, `ruff format --check` and
`mypy --strict` are all clean.

### The unit / sim / routing / slow chunks must not need ortools or osmnx

All three are imported **lazily**, inside the functions that need them. Verified the real way —
by removing them and running all four chunks:

```
python -m pip uninstall -y ortools osmnx networkx
python -m pytest -m unit
python -m pytest -m sim
python -m pytest -m routing
python -m pytest -m slow
python -m pip install -e ".[dev,ml,roads]"
```

**All four were green with ortools, osmnx and networkx absent.** The `routing` chunk ran 90 tests
and skipped 17 (`pytest.importorskip`); with the extra installed all **107** run. The bbox maths,
the haversine comparator, the detour-ratio statistics, the integer scaling, the capacity executor
and the matrix tours are all pure stdlib.

### No test touches the network

`tests/test_roads.py` never downloads anything. The road-distance tests run against a **hand-built
4-node graph**: the corners of a 1 km square with the 1→2 edge deliberately removed, so the only
road between those two nodes goes the long way round (1→0→3→2, 3 km) and the detour ratio for that
pair is **exactly 3.0** — asserted, not approximated. An isolated node with no edges is used to
check that an unreachable pair really comes back as `inf` rather than being hidden.

### Determinism

```
python scripts\run_roads.py --seed 42 --quick --no-download > results\r1.txt
python scripts\run_roads.py --seed 42 --quick --no-download > results\r2.txt
fc results\r1.txt results\r2.txt

python scripts\run_phase4_vrp.py --seed 42 --quick --no-ortools > results\v1.txt
python scripts\run_phase4_vrp.py --seed 42 --quick --no-ortools > results\v2.txt
fc results\v1.txt results\v2.txt
```

**Verified:** the roads run produced **byte-identical** output at **3,801 bytes**, and the
stdlib-only VRP run at **1,560 bytes**, compared with `cmp`. No timestamp, wall-clock timing or
library version is printed to stdout, which is what makes that possible; runtimes live in this
document only.

**The OR-Tools rows are deliberately excluded from that guarantee.** `vrp.ortools_time_limit_s` is
a wall-clock budget, so a faster or busier machine searches a different amount of the space and can
return a different route. Every OR-Tools number in this document is **same-machine-only**, and no
test asserts one.

## To produce the full run

For a separate plain cmd window — **not** inside a Claude Code session, per CLAUDE.md's runtime
rule:

```
python scripts\run_phase4_vrp.py --seed 42
python scripts\run_roads.py --seed 42
```

| run | what changes from the smoke run | estimated runtime |
|---|---|---|
| `run_phase4_vrp.py --seed 42` | 20 clusters instead of 5, 5 s solver budget instead of 1 s, all three capacities | **~10–12 min** — 20 instances × 5 s × (1 TSP + 3 capacities + 2 plan bases) ≈ 600 s of solver, plus the city replay |
| `run_roads.py --seed 42` | 300 bins instead of 50 | **~2–5 min** — 301 Dijkstras and a 301 × 301 haversine matrix; the graph is already cached so no download |

Each writes `results/phase4_vrp.{csv,md}` and `results/roads_eval.{csv,md}` (both gitignored). The
estimates are scaled from the measured smoke runtimes (24.7 s and 22.3 s) and are **estimates, not
measurements** — Phase 3's 35-minute estimate turned into 2.3 hours, so treat them as a floor.

Also worth re-running, to re-confirm Phase 2 end to end rather than relying on the pinning test:

```
python scripts\run_benchmark.py --seed 42
```

Expect ~7.8 min, and expect all 63 data rows to match `docs/PHASE2_RESULTS.md`.

## Honest summary

- **Truck capacity does not bind at this project's own constants, and that is the headline.** A
  cluster's whole day is ~296 kg against an 8000 kg truck — 3.7% of capacity. Phase 2 listed its
  missing capacity limit as something that "flatters every strategy that visits many bins"; on
  these numbers it flattered nothing. Phase 4's real contribution on this point is the
  measurement, not the fix.
- **OR-Tools does not meaningfully beat our hand-written 2-opt**: 0.29% against a 2% threshold
  declared in code before the run. Plain 2-opt over nearest neighbour is worth twenty times more
  (5.88%). `ortools` is justified by being able to *express* the capacitated multi-truck problem,
  not by distance.
- **The noisy-vs-oracle planning comparison came out as a tie, and that is a null result.** It
  follows from capacity not binding: nothing was riding on the weight estimate, so sensor noise
  was never tested. Said plainly rather than presented as the sensors being good enough.
- **A straight line under-states road distance by about 1.45× at the median in T. Nagar**, with a
  wide spread (p10 1.23, p90 2.14, max 12.02) and three non-interchangeable aggregate figures
  (1.45 pairwise / 1.54 total / 1.83 routed). This closes a question open since Phase 0 — for
  **one** 3 km area with **synthetic** bins, and it does **not** transfer to Phase 2's grid.
- **Nothing was tuned.** `cost.truck_capacity_kg` stayed at 8000.0 even though that made Table B
  flat and uninteresting. The capacity sweep stayed at 4000/8000/16000 even after it was clear all
  three were non-binding. The 2% OR-Tools threshold was a module constant before the first run.
- **Phase 2 and Phase 3 behaviour is unchanged**, and no pre-existing test file was edited.

## NOT VERIFIED

- **The publication-size runs were NOT RUN.** Every table in this document is a `--quick` smoke
  run at 5 clusters / 46 bins / 1 s solver budget, or 50 road bins. **No number here is a
  city-scale Phase 4 result.** The full commands and runtime estimates are above.
- **The full Phase 2 benchmark was NOT re-run this phase.** "Phase 2 still reproduces" rests on
  the literal-pinning test in `tests/test_demand.py` plus a check that no Phase 2/3 source file
  changed — not on a fresh 63-row comparison.
- **Every OR-Tools number is same-machine-only.** The solver runs under a wall-clock time limit,
  so distances are not reproducible across machines, operating systems or load conditions. These
  rows come from **ortools 9.15.6755 on Windows / Python 3.11**. No test asserts an OR-Tools
  distance; the tests assert route properties instead.
- **OR-Tools searches on a rounded integer matrix.** Arc costs are km × 1000 rounded to metres,
  because the solver requires integers. Reported distances are recomputed in float from the
  returned tour with Phase 2's own `route_distance`, so the comparison does not also measure the
  rounding — but the *search* saw rounded values. Capacity demands are rounded **up**, so that
  rounding can only overstate a load, never understate one.
- **Every new constant is invented and uncited.** `vrp.capacity_sweep_kg`,
  `max_trucks_per_cluster`, `ortools_time_limit_s`, `compare_clusters`, `compare_day`, and the
  whole `roads` section (area, centre, bbox size, bin count) have no citation or measurement
  behind them. See the third banner in `docs/ASSUMPTIONS.md`.
- **BIN LOCATIONS ON THE ROAD NETWORK ARE SYNTHETIC.** Uniform-random points in the bbox, snapped
  to the nearest drivable road node. The roads are real OpenStreetMap data; the bins are invented.
  A real road network makes synthetic bins look more authoritative than they are.
- **One area, one bbox size, one network filter, one seed.** The detour ratio is not a Chennai
  figure, let alone a general one. A planned-grid neighbourhood (Anna Nagar) would plausibly give
  a lower ratio and a river- or coast-bounded one (Adyar) a higher one; neither was run.
- **The road comparison is NOT comparable with any Phase 2 or Phase 3 number.** Different area
  (3 km vs 100 km), different bin count (50 vs 5000), different distance metric. No table here
  draws that comparison and none should be derived from it.
- **The capacity sweep is flat and therefore uninformative.** All three swept capacities sit far
  above the per-cluster load, so the sweep distinguishes nothing. The 150 kg binding row exists to
  show the mechanism works and is labelled **not a realistic truck**.
- **The oracle plan row is a reference line, not a result.** It is given the true weight, which no
  deployment has before the truck arrives. Its zero breaches are **by construction**, and the
  tests pin that rather than measure it.
- **The capacity model is crude.** Return-to-depot-and-resume charges distance but no extra
  service time, no tipping time and no driver-hour limit. A bin heavier than the whole truck is
  flagged and still collected, because the alternative — abandoning it forever — would be a
  different and less honest model.
- **One instance set, one day, one seed.** The algorithm comparison uses the bins strategy (d)
  selected on day 14 of seed 42 only. No seed range, no range over days, so there is no spread on
  any Phase 4 number. Phase 2 at least reported a five-seed range; Phase 4 does not.
- **Capacity is applied per cluster, so the fleet is still territory-bound.** KMeans clusters are
  unchanged from Phase 2 and a truck cannot cross into another cluster, which means the truck
  counts here are an upper bound on what a city-wide VRP would need.
- **Yellow-bin opportunistic collection is STILL not built**, though SPEC §7 lists it under Phase
  3. Out of scope for Phase 3 and for Phase 4; the paper's known weakness now carries into Phase
  5.
- **All data is synthetic, and only 1 physical bin was built** (in the paper's hardware work).
  Zero readings from it appear anywhere in this document. The 5000-bin city is simulation.
- **No time windows, no shift limits, no traffic, no one-way-street penalties in the cost model.**
  The road graph is direction-aware for *distance*, but nothing here models congestion or turn
  restrictions.
- **GitHub Actions CI has never executed.** `git push` is forbidden by CLAUDE.md, so the workflow
  — including the new `roads` extra in the `types` job and the `routing` chunk — is committed but
  **unproven**. Every check above was run locally on Python 3.11 / Windows.
