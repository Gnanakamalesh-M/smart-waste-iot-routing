# ASSUMPTIONS

The IEEE paper we extend leaves most of its simulation constants unstated (alpha, fuel rate,
wages, speeds, truck capacity). Every such value lives in exactly one place —
`config/default.yaml` — and every one is recorded below.

**As of Phase 0, every value in this file is a `PLACEHOLDER`.** No simulation has been run, no
number has been measured, and nothing here is claimed to be realistic — not even values taken
from the paper, since we have not yet exercised them. Phase 0 built the skeleton only.

## Status legend

| Status | Meaning |
|---|---|
| `PLACEHOLDER` | In the config, but nothing has run that depends on it. No claim made about it. |
| `USED` | A real run reads it and its value changes the output. Still not independently justified. |
| `VALIDATED` | Checked against real output or a citable external source, with the command/source recorded here. |

Only the phase named in **Validated in** may promote a row's status, and only by pasting the
real output or citation that justifies it.

## Source column

- `paper` — stated in the paper.
- `chosen` — we picked it because the paper does not state it.
- `derived` — computed from other values.

## Parameters

All rows are set in `config/default.yaml`.

### Global

| Parameter | Value | Source | Status | Validated in | Why |
|---|---|---|---|---|---|
| `seed` | 42 | chosen | PLACEHOLDER | Phase 1 | One global seed so the same command and seed reproduce the same numbers. |

### `bin` — geometry and state thresholds

| Parameter | Value | Source | Status | Validated in | Why |
|---|---|---|---|---|---|
| `bin.height_cm` | 100.0 | chosen | PLACEHOLDER | Phase 1 | Inner height `H` in `phi = 1 - d_bar / H`. Paper gives no dimensions. |
| `bin.w_max_kg` | 50.0 | chosen | PLACEHOLDER | Phase 1 | Bin weight capacity `W_max`. Paper gives no value. |
| `bin.alpha` | 0.85 | chosen | PLACEHOLDER | Phase 1 | Paper defines `W_th = alpha * W_max` with `0 < alpha < 1` but never states alpha. |
| `bin.yellow_threshold_pct` | 60.0 | paper | PLACEHOLDER | Phase 1 | YELLOW when the load-cell flag is 0 and `60 <= Fv < 80`. |
| `bin.red_threshold_pct` | 80.0 | paper | PLACEHOLDER | Phase 1 | RED when the load-cell flag is 1 or `Fv >= 80`. |

### `sensors` — array size and noise model

| Parameter | Value | Source | Status | Validated in | Why |
|---|---|---|---|---|---|
| `sensors.n_ultrasonic` | 3 | paper | PLACEHOLDER | Phase 1 | Three ultrasonic sensors (left / center / right) on the upper inner surface. |
| `sensors.noise_sigma_cm` | 1.5 | chosen | PLACEHOLDER | Phase 1 | Gaussian measurement noise per reading. Paper reports no sensor error model. |
| `sensors.bias_max_cm` | 3.0 | chosen | PLACEHOLDER | Phase 1 | Max absolute constant per-sensor bias (miscalibration). |
| `sensors.stuck_probability` | 0.01 | chosen | PLACEHOLDER | Phase 1 | Chance a sensor repeats its previous reading (stuck fault). |
| `sensors.uneven_fill_skew_cm` | 10.0 | chosen | PLACEHOLDER | Phase 1 | Lateral height difference across the three sensors; this is the effect fusion is supposed to absorb. |

### `sim` — city simulation size and waste arrivals

| Parameter | Value | Source | Status | Validated in | Why |
|---|---|---|---|---|---|
| `sim.n_bins` | 5000 | paper | PLACEHOLDER | Phase 2 | Paper's city-scale simulation size. |
| `sim.days` | 30 | paper | PLACEHOLDER | Phase 2 | Paper says 30 days in one place and "three-month" in another; we use 30 and note the inconsistency. |
| `sim.grid_km` | 100.0 | paper | PLACEHOLDER | Phase 2 | 100 x 100 km grid, per the paper's figure. Not a real city layout. |
| `sim.daily_fill_mean_kg` | 4.0 | chosen | PLACEHOLDER | Phase 2 | Mean of the daily increment `dW_i(t)`. Paper says "random increments" without a distribution. |
| `sim.daily_fill_sd_kg` | 1.5 | chosen | PLACEHOLDER | Phase 2 | Spread of the daily increment. |

### `cost` — fuel, CO2, labor, time

| Parameter | Value | Source | Status | Validated in | Why |
|---|---|---|---|---|---|
| `cost.fuel_l_per_km` | 0.35 | chosen | PLACEHOLDER | Phase 2 | The `a` in `F = a * D`. Paper uses the form but not the constant. |
| `cost.fuel_cost_per_l` | 95.0 | chosen | PLACEHOLDER | Phase 2 | Diesel price (INR/L). Needs a citation before it can be `VALIDATED`. |
| `cost.co2_kg_per_l` | 2.68 | chosen | PLACEHOLDER | Phase 2 | The `b` in `E = b * D` (paper's Eq. 13 prints `F = bD`, a typo). Needs a citation. |
| `cost.maintenance_cost_per_km` | 8.0 | chosen | PLACEHOLDER | Phase 2 | Maintenance term of the cost model (INR/km). |
| `cost.labor_cost_per_hr` | 180.0 | chosen | PLACEHOLDER | Phase 2 | Crew cost per truck-hour (INR). |
| `cost.fixed_cost_per_day` | 2000.0 | chosen | PLACEHOLDER | Phase 2 | Fixed cost per truck per day (INR). |
| `cost.truck_speed_kmph` | 25.0 | chosen | PLACEHOLDER | Phase 2 | Average urban collection speed; converts distance to travel time. |
| `cost.service_time_min_per_bin` | 3.0 | chosen | PLACEHOLDER | Phase 2 | Time to empty one bin; makes travel time depend on bin count, not just distance. |
| `cost.truck_capacity_kg` | 8000.0 | chosen | PLACEHOLDER | Phase 4 | Payload capacity. Unused until the VRP phase. |

### `routing`

| Parameter | Value | Source | Status | Validated in | Why |
|---|---|---|---|---|---|
| `routing.n_clusters` | 20 | chosen | PLACEHOLDER | Phase 2 | KMeans geographic clusters; paper splits the city into clusters without giving a count. |
| `routing.n_trucks_per_cluster` | 1 | chosen | PLACEHOLDER | Phase 4 | Phase 2 runs one depot-constrained tour per cluster; VRP relaxes this. |
| `routing.two_opt_max_iters` | 1000 | chosen | PLACEHOLDER | Phase 2 | Iteration cap for the 2-opt improvement pass (runtime guard). |

### `paths`

| Parameter | Value | Source | Status | Validated in | Why |
|---|---|---|---|---|---|
| `paths.data_dir` | `data` | chosen | PLACEHOLDER | Phase 2 | Generated datasets. Gitignored. |
| `paths.cache_dir` | `cache` | chosen | PLACEHOLDER | Phase 4 | Downloaded artifacts, e.g. OSMnx road graphs. Gitignored. |
| `paths.results_dir` | `results` | chosen | PLACEHOLDER | Phase 2 | Benchmark tables and figures. Gitignored. |

## Not verified / Limitations

As of Phase 0:

- **Every config value above is a `PLACEHOLDER`.** None has been exercised by a run.
- **Only 1 physical bin was built** (in the paper's hardware work). Everything at city scale is
  simulation.
- **The 5000-bin, 30-day results are simulation**, not measurements from deployed bins.
- **All data is synthetic.** Fill levels, sensor readings and faults are generated, not collected.
- **Distances will be Euclidean** on a random grid until Phase 4 adds real road distances; a
  straight line under-states road distance by an amount we have not yet measured.
- **The paper's baseline is naive** (visit every bin daily), so part of any reported saving is
  simply skipping bins rather than better routing. Fairer baselines land in Phase 2.
- **Yellow bins are never collected** under the paper's rule, which risks overflow. Phase 3
  addresses this.
- **The paper's own reported figures** (34.04% travel time, 22.22% cost, 31.85% CO2) are
  reference only. We do not promise to reproduce them and will not tune toward them.
- **Paper inconsistencies** carried forward knowingly: 30 days vs "three-month"; "34.04%
  distance" vs the travel-time figure in Table I; Eq. 13 printing `F = bD` instead of `E = bD`;
  "1926 percent" for 19-26%.
- **No firmware has been written or flashed**; no Docker image exists. Both are later phases and
  will be marked NOT VERIFIED unless actually run.
- **GitHub Actions CI has never executed.** The workflow is committed but nothing has been
  pushed (CLAUDE.md forbids `git push`), so CI is NOT VERIFIED.
