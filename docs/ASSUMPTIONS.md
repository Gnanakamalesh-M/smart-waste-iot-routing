# ASSUMPTIONS

The IEEE paper we extend leaves most of its simulation constants unstated (alpha, fuel rate,
wages, speeds, truck capacity). Every such value lives in exactly one place —
`config/default.yaml` — and every one is recorded below.

**As of Phase 1**, the parameters the Phase 1 experiment actually reads are `USED`; everything
else is still a `PLACEHOLDER`. **Nothing is `VALIDATED`** — that requires an external citation or
a real measurement, and Phase 1 has neither. `USED` means only "a real run reads this and its
value changes the output", not "this value is realistic".

The command that promoted every `USED` row below, run from the repo root:

```
python scripts\run_fusion_eval.py
```

Its output is in `docs/PHASE1_RESULTS.md` (and, regenerated, in `results/`).

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
| `seed` | 42 | chosen | USED | Phase 1 | One global seed so the same command and seed reproduce the same numbers. Phase 1 derives every sub-seed from it with sha256 (not `hash()`, which is salted per process); two runs at the same seed produce byte-identical output, checked with `fc`. |

### `bin` — geometry and state thresholds

| Parameter | Value | Source | Status | Validated in | Why |
|---|---|---|---|---|---|
| `bin.height_cm` | 100.0 | chosen | USED | Phase 1 | Inner height `H` in `phi = 1 - d_bar / H`. Paper gives no dimensions. Converts every fill % to a sensor distance, so it sets the scale the noise in cm is judged against. |
| `bin.w_max_kg` | 50.0 | chosen | USED | Phase 1 | Bin weight capacity `W_max`. Paper gives no value. Sets the overload threshold with `alpha`. |
| `bin.alpha` | 0.85 | chosen | USED | Phase 1 | Paper defines `W_th = alpha * W_max` with `0 < alpha < 1` but never states alpha. Decides which bin-days are RED by weight, so it decides how much the load cell can contribute. |
| `bin.yellow_threshold_pct` | 60.0 | paper | USED | Phase 1 | YELLOW when the load-cell flag is 0 and `60 <= Fv < 80`. Pinned exactly by boundary tests in `tests/test_fusion.py`. |
| `bin.red_threshold_pct` | 80.0 | paper | USED | Phase 1 | RED when the load-cell flag is 1 or `Fv >= 80`. Pinned exactly by boundary tests in `tests/test_fusion.py`. |
| `bin.density_kg_per_pct` | 0.5 | chosen | USED | Phase 1 | kg of waste per 1% of bin volume (`w_max_kg / 100`). Converts the paper's weight-based arrival process into a volume fill for the ultrasonic sensors. |
| `bin.density_cv` | 0.3 | chosen | USED | Phase 1 | Per-bin spread of that density, so a dense bin is heavy before it looks full. **The load cell's advantage is entirely conditional on this number being non-zero** — at 0 the weight channel is a deterministic function of the volume channel and carries no extra information. Phase 1 therefore reports a `density_cv = 0` sensitivity table next to the headline one. We have no field data on real density spread; 0.3 is a guess. |

### `sensors` — array size and noise model

| Parameter | Value | Source | Status | Validated in | Why |
|---|---|---|---|---|---|
| `sensors.n_ultrasonic` | 3 | paper | USED | Phase 1 | Three ultrasonic sensors (left / center / right) on the upper inner surface. |
| `sensors.noise_sigma_cm` | 1.5 | chosen | USED | Phase 1 | Gaussian measurement noise per reading. Paper reports no sensor error model, so this is our invention; it drives the `noisy` scenario and is **not** tuned to make fusion win (`tests/test_fusion_eval.py` asserts the scenario reads it unchanged). |
| `sensors.bias_max_cm` | 3.0 | chosen | USED | Phase 1 | Max absolute constant per-sensor bias (miscalibration), drawn once per bin and fixed. Averaging three independent biases shrinks their spread, which is part of why the 3-sensor mean helps. |
| `sensors.stuck_probability` | 0.01 | chosen | USED | Phase 1 | Per-reading chance a sensor repeats its previous value. Small, so the dedicated `stuck_left` / `stuck_center` scenarios (one sensor hard-latched) carry the stuck-fault result instead. |
| `sensors.uneven_fill_skew_cm` | 10.0 | chosen | USED | Phase 1 | Max lateral height difference across the three sensors — the effect fusion is supposed to absorb. Used by both uneven scenarios: `center_peak` (biases the centre sensor) and `random_tilt` (zero-mean, so only variance differs). |
| `sensors.min_distance_cm` | 2.0 | chosen | USED | Phase 1 | HC-SR04 minimum range. Readings clamp to `[min_distance_cm, height_cm]`, which is a small bias near a full or empty bin that affects all three methods alike. |
| `sensors.load_cell_sigma_kg` | 0.3 | chosen | USED | Phase 1 | Gaussian load-cell noise. Paper reports none. Too large a value would make method (c) worse than (b) near the overload threshold. |
| `sensors.load_cell_bias_max_kg` | 0.5 | chosen | USED | Phase 1 | Constant load-cell miscalibration, drawn once per bin. A single load cell has no second opinion to average against, so this bias is never cancelled — one reason method (c) raises some false REDs. |

### `sim` — city simulation size and waste arrivals

| Parameter | Value | Source | Status | Validated in | Why |
|---|---|---|---|---|---|
| `sim.n_bins` | 5000 | paper | PLACEHOLDER | Phase 2 | Paper's city-scale simulation size. |
| `sim.days` | 30 | paper | PLACEHOLDER | Phase 2 | Paper says 30 days in one place and "three-month" in another; we use 30 and note the inconsistency. |
| `sim.grid_km` | 100.0 | paper | PLACEHOLDER | Phase 2 | 100 x 100 km grid, per the paper's figure. Not a real city layout. |
| `sim.daily_fill_mean_kg` | 4.0 | chosen | USED | Phase 1 | Mean of the daily increment `dW_i(t)`. Paper says "random increments" without a distribution; we use a Gaussian floored at 0. **Row retargeted from Phase 2 to Phase 1**: the single-bin simulation already drives the arrival process, so Phase 1 genuinely exercises it. It sets how often a bin reaches RED and therefore the size of the `n(RED)` denominator behind every false-GREEN rate. |
| `sim.daily_fill_sd_kg` | 1.5 | chosen | USED | Phase 1 | Spread of the daily increment. Retargeted from Phase 2 for the same reason. |

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

### `fusion_eval` — size of the Phase 1 experiment

| Parameter | Value | Source | Status | Validated in | Why |
|---|---|---|---|---|---|
| `fusion_eval.n_bins` | 200 | chosen | USED | Phase 1 | Independent bins simulated per noise scenario. Also the bootstrap's resampling unit, so it sets how wide the confidence intervals are. |
| `fusion_eval.days` | 30 | chosen | USED | Phase 1 | Days per bin, matching the paper's 30-day horizon. |
| `fusion_eval.sweep_samples` | 20000 | chosen | USED | Phase 1 | Samples in the diagnostic uniform-fill sweep (`results/fusion_sweep.csv`), which shows *where* in the fill range each method errs. Diagnostic only: a uniform fill distribution is not what a real bin does. |
| `fusion_eval.bootstrap_resamples` | 2000 | chosen | USED | Phase 1 | Bin-level bootstrap replicates for the 95% CIs. Resampling whole bins, not readings, because readings in one bin share a density factor and a stuck fault. |

### `paths`

| Parameter | Value | Source | Status | Validated in | Why |
|---|---|---|---|---|---|
| `paths.data_dir` | `data` | chosen | PLACEHOLDER | Phase 2 | Generated datasets. Gitignored. |
| `paths.cache_dir` | `cache` | chosen | PLACEHOLDER | Phase 4 | Downloaded artifacts, e.g. OSMnx road graphs. Gitignored. |
| `paths.results_dir` | `results` | chosen | PLACEHOLDER | Phase 2 | Benchmark tables and figures. Gitignored. |

## Not verified / Limitations

New in Phase 1:

- **The entire sensor error model is invented.** The paper publishes no sensor error model, so
  there is nothing to calibrate against. Every noise number (sigma, bias, skew, stuck rates,
  load-cell noise) is a guess by us. The Phase 1 result is therefore "fusion helps *under this
  assumed noise model*", never "fusion helps".
- **Two of the six scenarios favour fusion by construction** and are labelled as such in the
  results table. `uneven_center_peak` defines a skew that biases the centre sensor;
  `stuck_center` aims the fault at the only sensor method (a) has. They demonstrate a mechanism;
  they are **not** evidence about how often that mechanism occurs in a real bin.
- **Fusion is not uniformly better.** Under `stuck_left` the 3-sensor mean is clearly *worse*
  than the single centre sensor, because the mean averages in a dead sensor while the centre
  sensor is unaffected. See `docs/PHASE1_RESULTS.md`. Nothing was retuned to hide this.
- **The load cell's advantage is conditional on `bin.density_cv`**, a number we chose with no
  field data. The `density_cv = 0` sensitivity table shows the case where it can add nothing.
- **Phase 1 measures rule-reproduction, not collection outcomes.** The metrics say how
  faithfully each sensor configuration reproduces the paper's own RED/YELLOW/GREEN rule. No truck
  is routed, no overflow is counted, no cost is computed. A method that reproduces the rule
  perfectly also inherits the rule's flaws — notably that YELLOW bins are never collected.
- **Collection is driven by ground truth, not by the estimate.** Every method sees the same fill
  trajectory, which makes the comparison paired and clean, but it removes the feedback loop a
  real deployment would have (a missed pickup changes the next day's fill).
- **Uneven fill is modelled as a height offset per sensor**, not as real geometry. There is no
  cone, no wall angle, no acoustic beam width, no soft-surface absorption and no temperature
  drift in the speed of sound.
- **The bin is a straight-sided cylinder in the model.** The paper's angled inner walls appear
  only indirectly, as the reason the `center_peak` skew has the sign it does.
- **One physical bin's worth of hardware, zero readings from it.** No value here was measured on
  the physical bin; nothing in Phase 1 touches hardware or ThingSpeak.
- **CI has still never run** (see below), so the Phase 1 tests are verified locally only.

Carried over from Phase 0:

- **Every config value not marked `USED` above is a `PLACEHOLDER`.** None has been exercised by a
  run. No value anywhere in this file is `VALIDATED`.
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
