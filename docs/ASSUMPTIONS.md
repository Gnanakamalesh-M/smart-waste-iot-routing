# ASSUMPTIONS

The IEEE paper we extend leaves most of its simulation constants unstated (alpha, fuel rate,
wages, speeds, truck capacity). Every such value lives in exactly one place —
`config/default.yaml` — and every one is recorded below.

**As of Phase 2**, the parameters the Phase 1 and Phase 2 experiments actually read are `USED`;
everything else is still a `PLACEHOLDER`. **Nothing anywhere in this file is `VALIDATED`** - that
requires an external citation or a real measurement, and neither phase has either. `USED` means
only "a real run reads this and its value changes the output", not "this value is realistic".

> ### Banner: the entire `cost` section is invented and uncited
>
> Phase 2 promotes every `cost` row to `USED`, because `scripts\run_benchmark.py` genuinely reads
> each one and each one changes the output. **That is all `USED` means here.** Not one cost
> constant - fuel rate, diesel price, CO2 per litre, maintenance, wages, fixed cost, speed,
> service time - has a citation or a measurement behind it. The paper states the *form* of its
> cost model (`F = a*D`, `E = b*D`, `cost = fuel + maintenance + labor + fixed`) and **none of
> the constants**, so there was nothing to calibrate against and we picked plausible figures.
>
> **No constant was tuned toward the paper's reported percentages** (34.04% / 22.22% / 31.85%).
> They were chosen once, before any benchmark ran, and never adjusted afterwards. Phase 2's
> reductions come out substantially *larger* than the paper's; see `docs/PHASE2_RESULTS.md`,
> where that gap is stated plainly rather than closed by retuning.
>
> Because the absolute constants are invented, **the absolute INR and kg figures in the results
> carry no external meaning.** Only the *relative* comparison between strategies is informative,
> and even that shifts if the relative sizes of the fuel, labor and fixed terms are wrong.

The commands that promoted the `USED` rows below, run from the repo root:

```
python scripts\run_fusion_eval.py
python scripts\run_benchmark.py --seed 42
```

Their output is in `docs/PHASE1_RESULTS.md` and `docs/PHASE2_RESULTS.md` (and, regenerated, in
`results/`).

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
| `bin.max_accumulated_weight_kg` | 100.0 | **invented, uncited** | USED | Phase 2 | Hard cap on an uncollected bin's weight. Phase 2 lets a strategy miss a full bin, so without a cap an abandoned bin would accumulate weight without bound for 30 days and the load cell's overload flag would be reached by runaway arithmetic rather than by anything physical. **No field data, no citation.** `config.py` constrains it to sit at or above `alpha * w_max_kg` (42.5 kg), because below that a missed bin could never raise the flag at all and could never be rescued. At 100.0 it is twice the bin's nominal capacity, which is a guess at how much a bin holds once it is overflowing. |
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
| `sensors.stuck_bin_fraction` | 0.02 | **invented, uncited** | USED | Phase 2 | Fraction of city bins carrying one hard-latched ultrasonic sensor. **This single invented number decides the headline (d)-vs-(e) verdict**: at 0 the paper's mean fusion wins on overflow, at 0.02 and above our median fusion wins. We have no field data on real sensor failure rates, so Phase 2 does not stake the conclusion on one value - it sweeps `benchmark.fault_fractions` and reports all three outcomes side by side. Which of the 3 sensors sticks is drawn **uniformly**, because Phase 1 showed the mean collapses when a side sensor sticks but merely ties when the centre one does, so hand-picking either would pre-decide the comparison. |
| `sensors.load_cell_bias_max_kg` | 0.5 | chosen | USED | Phase 1 | Constant load-cell miscalibration, drawn once per bin. A single load cell has no second opinion to average against, so this bias is never cancelled — one reason method (c) raises some false REDs. |

### `sim` — city simulation size and waste arrivals

| Parameter | Value | Source | Status | Validated in | Why |
|---|---|---|---|---|---|
| `sim.n_bins` | 5000 | paper | USED | Phase 2 | Paper's city-scale simulation size. Read by `run_benchmark.py`; sets the whole scale of every distance and cost reported. |
| `sim.days` | 30 | paper | USED | Phase 2 | Paper says 30 days in one place and "three-month" in another; we use 30 and carry the inconsistency forward knowingly. The horizon drives how many overflow episodes a weak strategy can accumulate, so a longer run would widen the gaps Phase 2 reports. |
| `sim.grid_km` | 100.0 | paper | USED | Phase 2 | 100 x 100 km grid, per the paper's figure. **Not a real city layout**: bins are uniform-random on a square, which is why every absolute distance here is a Euclidean fiction. A 100 km grid with 5000 bins makes tours very long, which inflates the fuel and CO2 terms relative to the labor term. |
| `sim.daily_fill_mean_kg` | 4.0 | chosen | USED | Phase 1 | Mean of the daily increment `dW_i(t)`. Paper says "random increments" without a distribution; we use a Gaussian floored at 0. **Row retargeted from Phase 2 to Phase 1**: the single-bin simulation already drives the arrival process, so Phase 1 genuinely exercises it. It sets how often a bin reaches RED and therefore the size of the `n(RED)` denominator behind every false-GREEN rate. |
| `sim.daily_fill_sd_kg` | 1.5 | chosen | USED | Phase 1 | Spread of the daily increment. Retargeted from Phase 2 for the same reason. |

### `cost` — fuel, CO2, labor, time

| Parameter | Value | Source | Status | Validated in | Why |
|---|---|---|---|---|---|
| `cost.fuel_l_per_km` | 0.35 | **invented, uncited** | USED | Phase 2 | The `a` in `F = a * D`. Paper uses the form and never gives the constant. No citation. |
| `cost.fuel_cost_per_l` | 95.0 | **invented, uncited** | USED | Phase 2 | Diesel price (INR/L). Plausible for India around the time of writing, but **we have not cited a source**, so it cannot be `VALIDATED`. |
| `cost.co2_kg_per_l` | 2.68 | **invented, uncited** | USED | Phase 2 | The CO2 half of `E = b * D` (paper's Eq. 13 prints `F = bD`, a typo). Close to commonly quoted diesel figures, but **uncited here**, so every CO2 number inherits that. |
| `cost.maintenance_cost_per_km` | 8.0 | **invented, uncited** | USED | Phase 2 | Maintenance term (INR/km). Pure guess. |
| `cost.labor_cost_per_hr` | 180.0 | **invented, uncited** | USED | Phase 2 | Crew cost per truck-hour (INR). Pure guess, and an influential one: labor is charged on driving *plus* service time, so it is the term that rewards visiting fewer bins. |
| `cost.fixed_cost_per_day` | 2000.0 | **invented, uncited** | USED | Phase 2 | Fixed cost per truck per day (INR). Pure guess. **Whether an idle truck is charged this is a modelling choice that moves the headline saving**, so Phase 2 reports three cost bases (variable-only, rolling fleet, fleet fixed at baseline size) instead of picking one. |
| `cost.truck_speed_kmph` | 25.0 | **invented, uncited** | USED | Phase 2 | Average urban collection speed; converts distance to driving time. Pure guess. |
| `cost.service_time_min_per_bin` | 3.0 | **invented, uncited** | USED | Phase 2 | Time to empty one bin. Pure guess. It is what makes travel time depend on bin count and not only on distance, and it is why the travel-time reduction is much larger than the distance reduction. |
| `cost.truck_capacity_kg` | 8000.0 | chosen | PLACEHOLDER | Phase 4 | Payload capacity. Still genuinely unread: Phase 2 imposes **no capacity limit**, so a tour may collect more than a real truck could carry. The VRP phase fixes that. |

### `routing`

| Parameter | Value | Source | Status | Validated in | Why |
|---|---|---|---|---|---|
| `routing.n_clusters` | 20 | chosen | USED | Phase 2 | KMeans geographic clusters, one truck each; the paper splits the city into clusters without giving a count. 20 clusters over 5000 bins is 250 bins per territory. This value sets the baseline fleet size, so it also sets the fixed-cost denominator in two of the three cost bases. |
| `routing.n_trucks_per_cluster` | 1 | chosen | PLACEHOLDER | Phase 4 | Phase 2 runs one depot-constrained tour per cluster; VRP relaxes this. |
| `routing.two_opt_max_iters` | 1000 | chosen | USED | Phase 2 | Iteration cap for the 2-opt improvement pass, a runtime guard. 2-opt only accepts strict improvements, so hitting the cap returns a tour that is still no longer than the nearest-neighbour tour it started from (pinned in `tests/test_tsp.py`). |

### `fusion_eval` — size of the Phase 1 experiment

| Parameter | Value | Source | Status | Validated in | Why |
|---|---|---|---|---|---|
| `fusion_eval.n_bins` | 200 | chosen | USED | Phase 1 | Independent bins simulated per noise scenario. Also the bootstrap's resampling unit, so it sets how wide the confidence intervals are. |
| `fusion_eval.days` | 30 | chosen | USED | Phase 1 | Days per bin, matching the paper's 30-day horizon. |
| `fusion_eval.sweep_samples` | 20000 | chosen | USED | Phase 1 | Samples in the diagnostic uniform-fill sweep (`results/fusion_sweep.csv`), which shows *where* in the fill range each method errs. Diagnostic only: a uniform fill distribution is not what a real bin does. |
| `fusion_eval.bootstrap_resamples` | 2000 | chosen | USED | Phase 1 | Bin-level bootstrap replicates for the 95% CIs. Resampling whole bins, not readings, because readings in one bin share a density factor and a stuck fault. |

### `benchmark` - shape of the Phase 2 city benchmark

| Parameter | Value | Source | Status | Validated in | Why |
|---|---|---|---|---|---|
| `benchmark.fixed_schedule_days` | 3 | chosen | USED | Phase 2 | The N in strategy (b), "collect every bin every N days". Chosen as a plausible municipal schedule, **not** tuned: it is the sensor-free baseline the sensor strategies must beat, so tuning it would be tuning the thing we are trying to measure against. Note that N strongly determines (b)'s overflow count, and at N=3 it overflows **zero** times, which is a large part of the Phase 2 story. |
| `benchmark.n_seeds` | 5 | chosen | USED | Phase 2 | Seeds 42-46. Each seed regenerates the layout, the densities, the fault draw and every waste arrival, so the spread covers the whole pipeline. Reported as a **range over seeds**; 5 seeds is not enough for a confidence interval and the results doc never calls it one. |
| `benchmark.fault_fractions` | [0.0, 0.02, 0.10] | chosen | USED | Phase 2 | The stuck-sensor sweep. Exists precisely because `sensors.stuck_bin_fraction` is invented: 0.0 is the no-fault control, 0.02 matches the config value, 0.10 is the stress row. The (d)-vs-(e) verdict **reverses** between 0.0 and 0.02, so reporting a single fault rate would have let us choose the conclusion. |

### `paths`

| Parameter | Value | Source | Status | Validated in | Why |
|---|---|---|---|---|---|
| `paths.data_dir` | `data` | chosen | PLACEHOLDER | Phase 2 | Generated datasets. Gitignored. |
| `paths.cache_dir` | `cache` | chosen | PLACEHOLDER | Phase 4 | Downloaded artifacts, e.g. OSMnx road graphs. Gitignored. |
| `paths.results_dir` | `results` | chosen | USED | Phase 2 | Benchmark tables and figures. `run_benchmark.py` writes `benchmark.csv` and `benchmark.md` here. Gitignored, so the committed numbers live in `docs/PHASE2_RESULTS.md`. |

## Not verified / Limitations

New in Phase 2:

- **Phase 1's ground-truth-collection shortcut is GONE, and that is the main change.** In Phase 1
  every method saw the identical fill trajectory because collection was triggered by ground truth,
  so missing a full bin cost nothing. Phase 2 drives collection from **each strategy's own noisy
  classification**, so a missed bin keeps filling and overflows, and the trajectories diverge. The
  Phase 1 results remain correct for what they measured (rule reproduction); they simply did not
  measure consequences.
- **The whole `cost` section is invented and uncited** (see the banner at the top of this file).
  The absolute INR and kg figures therefore carry no external meaning. Only the comparison between
  strategies is informative.
- **Nothing was tuned toward the paper's percentages,** and our reductions came out much larger
  than the paper's (roughly 64% distance / 77% travel time / 67% cost against its 34.04% / 22.22%
  / 31.85%). That gap is reported, not closed. It is mostly an artefact of our naive baseline on a
  uniform-random 100 km grid, where skipping 92% of bins shortens tours enormously.
- **The (d) mean vs (e) median verdict reverses with an invented number.** At
  `stuck_bin_fraction = 0` the paper's mean fusion has roughly 3x fewer overflow events than our
  median extension; at 0.02 and 0.10 the median wins by a wide margin. Since we have no field data
  on sensor failure rates, Phase 2 reports all three fault fractions rather than picking one. **The
  extension is not a clear win.**
- **(e) median fusion misses more RED bin-days than (d) at every fault fraction**, even where it
  overflows far less. A median of 3 is a noisier estimator than a mean, so it is less accurate day
  to day; it wins on overflow only because it cannot be dragged below the threshold by one dead
  sensor. Both facts are in the results table.
- **A sensor-free schedule is very hard to beat here.** Strategy (b), collecting every bin every 3
  days with no sensors at all, had **zero** overflow events at every fault fraction and came within
  about 2-3% of the paper's method on variable cost. On this synthetic arrival process the sensors
  buy a small cost improvement and a worse overflow record. That is a finding about our
  assumptions, not proof that telemetry is worthless.
- **Overflow is modelled, not simulated physically.** True fill caps at 100% and weight accrues to
  an invented cap; nothing spills, nothing is lost, no street-level consequence is represented, and
  there is no penalty cost attached to an overflow event. Overflow is reported as a count beside
  cost, never folded into it.
- **No truck capacity limit.** A Phase 2 tour may collect more waste than a real truck could
  carry, which flatters every strategy that visits many bins. Capacity arrives with the VRP in
  Phase 4.
- **Euclidean distance on a uniform-random grid.** Distances are straight lines between random
  points in a 100 x 100 km square, so they are not road distances and the layout is not a city.
  Phase 4 adds a real road network.
- **One truck per cluster, single central depot, no time windows, no shift limits.** Travel times
  of thousands of hours per 30 days are spread across 20 trucks with no cap on a driver's day.
- **KMeans clusters are fixed for the whole month** and computed on bin locations only, ignoring
  how much waste each bin produces, so territories are balanced by area rather than by workload.
- **Yellow bins are still never collected** under every sensor strategy, faithfully reproducing the
  paper's rule and its known weakness. Phase 3 addresses it.
- **Five seeds is a range, not a confidence interval.** The results doc reports mean and min-max
  across seeds 42-46 and deliberately never calls it a CI.
- **The benchmark took 7.8 minutes** on one Windows machine; that is a measured wall-clock figure
  for this hardware, not a benchmark of the algorithms.

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
- **Collection is driven by ground truth, not by the estimate.** **[SUPERSEDED IN PHASE 2 -
  still true of the Phase 1 experiment only.]** Every method sees the same fill trajectory, which
  makes the comparison paired and clean, but it removes the feedback loop a real deployment would
  have (a missed pickup changes the next day's fill). Phase 2's city benchmark drives collection
  from each strategy's own noisy classification instead; see the Phase 2 block above.
- **Uneven fill is modelled as a height offset per sensor**, not as real geometry. There is no
  cone, no wall angle, no acoustic beam width, no soft-surface absorption and no temperature
  drift in the speed of sound.
- **The bin is a straight-sided cylinder in the model.** The paper's angled inner walls appear
  only indirectly, as the reason the `center_peak` skew has the sign it does.
- **One physical bin's worth of hardware, zero readings from it.** No value here was measured on
  the physical bin; nothing in Phase 1 touches hardware or ThingSpeak.
- **CI has still never run** (see below), so the Phase 1 tests are verified locally only.

Carried over from Phase 0:

- **Every config value not marked `USED` above is a `PLACEHOLDER`.** As of Phase 2 only
  `cost.truck_capacity_kg`, `routing.n_trucks_per_cluster`, `paths.data_dir` and
  `paths.cache_dir` remain unread by any run. **No value anywhere in this file is
  `VALIDATED`.**
- **Only 1 physical bin was built** (in the paper's hardware work). Everything at city scale is
  simulation.
- **The 5000-bin, 30-day results are simulation**, not measurements from deployed bins.
- **All data is synthetic.** Fill levels, sensor readings and faults are generated, not collected.
- **Distances are Euclidean** on a random grid until Phase 4 adds real road distances; a
  straight line under-states road distance by an amount we have not yet measured. Phase 2
  reports Euclidean distances throughout.
- **The paper's baseline is naive** (visit every bin daily), so most of any reported saving
  is simply skipping bins rather than better routing. Phase 2 added the fairer baselines
  (b) and (c), and they confirm it: a sensor-free 3-day schedule captures essentially the
  same saving as the paper's sensor fusion.
- **Yellow bins are never collected** under the paper's rule, which risks overflow. Still
  true in Phase 2, by design. Phase 3 addresses it.
- **The paper's own reported figures** (34.04% travel time, 22.22% cost, 31.85% CO2) are
  reference only. We do not promise to reproduce them and will not tune toward them.
- **Paper inconsistencies** carried forward knowingly: 30 days vs "three-month"; "34.04%
  distance" vs the travel-time figure in Table I; Eq. 13 printing `F = bD` instead of `E = bD`;
  "1926 percent" for 19-26%.
- **No firmware has been written or flashed**; no Docker image exists. Both are later phases and
  will be marked NOT VERIFIED unless actually run.
- **GitHub Actions CI has never executed.** The workflow is committed but nothing has been
  pushed (CLAUDE.md forbids `git push`), so CI is NOT VERIFIED.
