# ASSUMPTIONS

The IEEE paper we extend leaves most of its simulation constants unstated (alpha, fuel rate,
wages, speeds, truck capacity). Every such value lives in exactly one place —
`config/default.yaml` — and every one is recorded below.

**As of Phase 4 every row is `USED`** - the last three `PLACEHOLDER`s
(`cost.truck_capacity_kg`, `routing.n_trucks_per_cluster`, `paths.cache_dir`) were promoted by the
Phase 4 VRP and roads runs. **Phase 5 added no row at all**: it introduced no config value and no
new constant, so the parameter table is unchanged from Phase 4. Its one invented quantity - the
ThingSpeak field mapping - is not a config value, and gets a banner below instead of a row. **Nothing anywhere in this file is `VALIDATED`** - that requires an
external citation or a real measurement, and no phase has either. `USED` means only "a real run
reads this and its value changes the output", not "this value is realistic".

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

> ### Banner: the entire `demand`, `forecast` and `anomaly` sections are invented and uncited
>
> Phase 3 adds three whole sections, and **not one value in any of them has a citation or a
> measurement behind it.** The paper models arrivals as `W_i(t) = W_i(t-1) + dW_i(t)` with
> "random daily increments" and states **no distribution, no heterogeneity and no calendar**; it
> describes no forecasting and no anomaly detection at all. So area types, the weekday cycle,
> festival spikes, every forecasting knob and every planted-fault parameter are modelling
> choices of ours with no field data.
>
> **The consequence is specific and it matters:** Phase 3's finding that a forecaster earns its
> keep where Phase 2's calendar did not **holds on our invented demand only.** It is not
> evidence about a real city. `docs/PHASE3_RESULTS.md` states this beside the tables.
>
> **Nothing in these sections was tuned to make a model win.** In particular
> `forecast.warmup_days` stays at 14 even though that left the gradient-boosting model with
> **zero festival days in training** (measured, and reported). Lengthening the warm-up until the
> ML model looked better would have been tuning to the result. Likewise `demand.area_mean_kg`
> was rescaled before any Phase 3 benchmark ran, and rescaled to *remove* an advantage rather
> than to create one - see its row below.

> ### Banner: the entire `vrp` and `roads` sections are invented and uncited
>
> Phase 4 adds two more sections, and **not one value in either has a citation or a measurement
> behind it.** The paper mentions truck capacity as an unstated constant and describes no road
> network at all, so the capacity sweep, the solver budget, the comparison day, the Chennai area,
> the bounding-box size and the road bin count are all modelling choices of ours.
>
> **Nothing was tuned, and one case is worth spelling out.** `cost.truck_capacity_kg` stayed at
> the **8000.0** it has had since Phase 0 even after the first run showed that a cluster's whole
> day is about **296 kg** — 3.7% of it — so the capacity constraint **never binds** and Phase 4's
> capacity table is flat and uninteresting. Lowering the capacity until the table looked
> interesting would have been tuning to the output. `vrp.capacity_sweep_kg` likewise stayed at
> [4000, 8000, 16000] after it was clear all three were non-binding. A separate 150 kg run exists
> to show the mechanism works and is labelled **not a realistic truck** everywhere it appears.
>
> **The consequence is specific:** Phase 4's noisy-plan-vs-oracle comparison came out a **tie**,
> and that is a *null result caused by the above*, not evidence that sensor fusion is accurate
> enough for dispatch. With capacity non-binding, nothing rides on the weight estimate, so sensor
> noise is never tested. `docs/PHASE4_RESULTS.md` states that beside the table.
>
> **The road detour ratio is a measurement, and the only one Phase 4 really produced** — but it
> covers **one** ~3 km area with **synthetic** bin locations, and it does **not** transfer to
> Phase 2's 100 km grid. Road data is © OpenStreetMap contributors, **ODbL 1.0**.

> ### Banner: the ThingSpeak field mapping is invented and uncited
>
> Phase 5 adds **no config row** - no `dashboard` section, no `thingspeak` section, nothing. It does
> add one invented quantity, and it lives in code rather than in `config/default.yaml`, so it gets a
> banner instead of a table row.
>
> The paper says the NodeMCU sends "fill % and weight" to ThingSpeak and **never states which
> channel field carries what.** So this mapping is ours, with no citation: `field1`/`field2`/`field3`
> = ultrasonic distances left/centre/right in cm, `field4` = load-cell weight in kg, `field5` = `Fv`,
> `field6` = state code (0 GREEN, 1 YELLOW, 2 RED). It is declared in
> `src/smartwaste/thingspeak.py`, in the sketch, in `firmware/nodemcu_smartbin/README.md` and in
> `docs/PHASE5_RESULTS.md`, labelled **INVENTED** in each.
>
> **The consequence is specific: a real channel could silently disagree and nothing would detect
> it.** No ThingSpeak channel has ever been contacted by this project, so the mapping has never been
> checked against a live feed. `field5` and `field6` are deliberately **ignored** on the read path -
> the fetch script re-derives `Fv` and the state from fields 1-4 - because the point is to check that
> the firmware's arithmetic and the Python rule agree, not to trust the firmware's own answer.
>
> The demo snapshot's size (200 bins, 14 days, 5 clusters) is likewise **not** a config row. It is a
> set of script constants in the generator, for the same reason `QUICK_N_BINS` in the roads runner is
> one: it shapes a demonstration, not a result. Promoting it would have added three more
> invented-and-uncited rows for nothing.

The commands that promoted the `USED` rows below, run from the repo root:

```
python scripts\run_fusion_eval.py
python scripts\run_benchmark.py --seed 42
python scripts\run_phase3_benchmark.py --seed 42
python scripts\run_forecast.py --seed 42
python scripts\run_anomaly.py --seed 42
python scripts\run_phase4_vrp.py --seed 42 --quick
python scripts\run_roads.py --seed 42 --quick
python scripts\make_demo_data.py --seed 42
```

**The Phase 5 command promotes nothing.** The demo-snapshot generator reads existing `bin`,
`sensors`, `sim`, `demand`, `cost` and `routing` rows that Phases 1-4 already promoted, and writes
the committed snapshot the dashboard displays. It introduces no row of its own, and its output is a
demonstration rather than a result.

**The two Phase 4 commands were run at `--quick` size only.** Their rows below are `USED` because
a real run genuinely reads each one and its value changes the output — which is all `USED` has ever
meant here — not because a publication-size run exists. It does not; see
`docs/PHASE4_RESULTS.md`.

Their output is in `docs/PHASE1_RESULTS.md`, `docs/PHASE2_RESULTS.md`,
`docs/PHASE3_RESULTS.md`, `docs/PHASE4_RESULTS.md` and `docs/PHASE5_RESULTS.md` (and, regenerated,
in `results/`). The Phase 5 demo snapshot is the one exception to the `results/` pattern: it is
**committed**, under `dashboard/demo_data/`, because `results/*` is gitignored and a dashboard
reading from there would be blank on a fresh clone.

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

### `demand` - Phase 3 waste arrival process (**every row invented and uncited**)

See the second banner at the top of this file. `smartwaste.demand` implements two scenarios:
`smooth` is Phase 2's process reproduced bit for bit (so `docs/PHASE2_RESULTS.md` stays
reproducible), and `heterogeneous` is the Phase 3 process these rows configure. Both are
reported.

| Parameter | Value | Source | Status | Validated in | Why |
|---|---|---|---|---|---|
| `demand.scenario` | `heterogeneous` | **invented, uncited** | USED | Phase 3 | Which arrival process the Phase 3 runners default to. `smooth` stays selectable and is `build_layout`'s default, because every Phase 2 caller omits the argument and must keep getting Phase 2's exact numbers - `tests/test_demand.py` pins those numbers as literals. |
| `demand.area_type_weights` | [0.2, 0.5, 0.3] | **invented, uncited** | USED | Phase 3 | Share of bins that are market / residential / apartment. No survey behind the split; a guess at a plausible mix. Must sum to 1.0 (enforced in `config.py`). |
| `demand.area_mean_kg` | [6.0, 2.0, 3.4] | **invented, uncited** | USED | Phase 3 | Mean daily arrival per area type. **Deliberately rescaled to remove an advantage, not to create one.** The first draft, [9.0, 3.0, 5.0], gave an effective mean of **5.883 kg/bin/day - 47.1% more waste than Phase 2's 4.0** - once the weekday mean (**1.1143**) and the festival factor (**1.1000**) were applied. A heterogeneous city that simply makes half again as much waste would improve every Phase 3 number for a reason unrelated to sensors or forecasting. These values give **3.947 kg/bin/day, -1.3% from 4.0**, so the two scenarios differ in *variance*, not volume. `tests/test_demand.py` **measures** the realised mean over 400 bins x 700 days and fails if it drifts more than 5% from `sim.daily_fill_mean_kg`. |
| `demand.cv` | 0.35 | **invented, uncited** | USED | Phase 3 | Per-bin-day coefficient of variation; `sd = mean * cv`, so spread scales with the mean. A market bin is then noisy in kg and a residential bin quiet - the heteroscedasticity a fixed calendar cannot exploit and a forecaster might. Pure guess. |
| `demand.weekday_multipliers` | [1.0, 1.0, 1.0, 1.0, 1.15, 1.4, 1.25] | **invented, uncited** | USED | Phase 3 | Mon..Sun. Day 0 of every horizon is taken to be a Monday, fixed so a seed-to-seed difference is never just a calendar shift. Their mean is **1.1143**, part of the volume-matching arithmetic above. No citation. |
| `demand.festival_days_per_month` | 2 | **invented, uncited** | USED | Phase 3 | Quoted per 30 days, so a 30-day horizon gets 2 festival days. **Measured in the real run: both landed outside the 14-day warm-up**, so the forecaster trained on zero of them. Reported as a limitation, not fixed. |
| `demand.festival_multiplier` | 2.5 | **invented, uncited** | USED | Phase 3 | Arrival multiplier on a festival day. Contributes the **1.1000** factor to the 30-day mean. Pure guess. |

### `forecast` - Phase 3 fill-level forecasting (**every row invented and uncited**)

| Parameter | Value | Source | Status | Validated in | Why |
|---|---|---|---|---|---|
| `forecast.warmup_days` | 14 | **invented, uncited** | USED | Phase 3 | Days 0..13 run the backstop calendar only and log noisy estimates; the model is fitted once at day 14 and frozen. This **is** the time-based split CLAUDE.md requires - no random split exists anywhere in the forecasting code. It also defines the headline scoring window, since before day 14 strategy (f) is literally its own calendar. **Not tuned:** at 14 days the model saw 7 distinct weekdays but **0 festival days** (measured), which plausibly costs it accuracy on festival days. Lengthening the warm-up until the ML model won would be tuning to the result, so it stays. |
| `forecast.horizon_days` | 3 | **invented, uncited** | USED | Phase 3 | How far ahead a forecast looks. Capped per row by the longest *clean* span - one containing no collection - because a fill drop across a collection measures the truck, not demand. Without that cap a 3-day horizon under a 3-day calendar yields **zero** training rows, since any three consecutive days contain a multiple of three. |
| `forecast.history_days` | 5 | **invented, uncited** | USED | Phase 3 | Lags of `Fv_hat` and noisy weight used as features. Constrained in `config.py` to be `<= warmup_days`, or no training row could ever be built. |
| `forecast.moving_average_window` | 3 | **invented, uncited** | USED | Phase 3 | Window for the moving-average **baseline** - the one the gradient-boosting model has to beat. Not tuned, for the same reason `benchmark.fixed_schedule_days` is not: tuning the baseline would be tuning the thing being measured against. |
| `forecast.backstop_days` | [3.0, 7.0] | **invented, uncited** | USED | Phase 3 | The scheduled-visit floor each forecast-driven strategy falls back on. **Both are run, and each is compared against the sensor-free calendar of the same period** (3 vs `b`, 7 vs `b7`); otherwise "forecasting wins" could just mean "collecting less often is cheaper". |
| `forecast.gbm_max_iter` | 200 | **invented, uncited** | USED | Phase 3 | `HistGradientBoostingRegressor` iterations. Never swept; one value, chosen before any run. |
| `forecast.gbm_learning_rate` | 0.1 | **invented, uncited** | USED | Phase 3 | As above - the library default, not tuned. |
| `forecast.gbm_max_depth` | 4 | **invented, uncited** | USED | Phase 3 | As above. **No hyperparameter search was performed at all**, deliberately: a tuned ML model against an untuned baseline would be an unfair comparison. |

### `anomaly` - Phase 3 sensor anomaly detection (**every row invented and uncited**)

| Parameter | Value | Source | Status | Validated in | Why |
|---|---|---|---|---|---|
| `anomaly.n_bins` | 400 | **invented, uncited** | USED | Phase 3 | Bins in the planted-fault dataset. With `days = 60` and `window_days = 5` this yields 22,400 scored windows. |
| `anomaly.days` | 60 | **invented, uncited** | USED | Phase 3 | Days per bin. Longer than the 30-day city horizon so a drift fault has room to develop. |
| `anomaly.fault_bin_fraction` | 0.15 | **invented, uncited** | USED | Phase 3 | Share of bins given a planted fault. Note this is a share of **bins**, and each episode covers only part of the horizon, so the share of faulty **windows** is far lower - **measured at 0.0312**. That is precisely why accuracy is banned here: "never faulty" would score **96.9%** while catching nothing. |
| `anomaly.window_days` | 5 | **invented, uncited** | USED | Phase 3 | Rolling window the detectors score over. A window is labelled faulty when **its last day** falls inside the episode - a choice, stated in the results doc, not a fact. |
| `anomaly.validation_seed_offset` | 1 | **invented** | USED | Phase 3 | Thresholds are chosen on seed+1 and applied **unchanged** to the scoring seed. `config.py` **rejects 0**, because choosing a threshold on the scoring seed's planted labels is exactly the leakage this project forbids everywhere else. |
| `anomaly.min_fault_days` | 6 | **invented, uncited** | USED | Phase 3 | Shortest planted episode. |
| `anomaly.max_fault_days` | 20 | **invented, uncited** | USED | Phase 3 | Longest planted episode. Constrained to `<= anomaly.days`. |
| `anomaly.drift_cm_per_day` | 1.2 | **invented, uncited** | USED | Phase 3 | How fast a drifting sensor wanders. **An influential guess:** drift turned out to be the family both detectors are near-blind to (PR-AUC 0.0311 rule / 0.0152 forest against a 0.0071 prevalence), and a larger drift rate would be easier to catch. |
| `anomaly.noise_burst_sigma_cm` | 12.0 | **invented, uncited** | USED | Phase 3 | Extra Gaussian sigma during a noise burst. Same caveat as drift. |
| `anomaly.stuck_tolerance_cm` | 0.05 | **invented, uncited** | USED | Phase 3 | Starting point for the rule baseline's "no variance" tolerance, and one candidate in `RULE_TOLERANCE_GRID`. The *reported* run used **2.000**, selected on the validation seed. |
| `anomaly.disagreement_cm` | 25.0 | **invented, uncited** | USED | Phase 3 | Starting point for the rule's 3-sensor disagreement limit. The reported run selected **25.000** on the validation seed. |
| `anomaly.contamination` | 0.15 | **invented, uncited** | USED | Phase 3 | Starting point for the Isolation Forest's expected outlier share. The reported run selected **0.100** on the validation seed. |
| `anomaly.n_estimators` | 200 | **invented, uncited** | USED | Phase 3 | Isolation Forest trees. Not swept. |

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
| `cost.truck_capacity_kg` | 8000.0 | **invented, uncited** | USED | Phase 4 | Payload capacity, now genuinely read: `scripts\run_phase4_vrp.py` enforces it as an OR-Tools capacity dimension and Table C plans against it. **Measured consequence: it never binds.** A bin holds ~50 kg (`bin.w_max_kg`) and strategy (d) selects ~9 bins per cluster per day, so a cluster's whole day is **296.24 kg** against 8000 — **3.7%**. A truck would need ~160 full bins in one tour before capacity mattered. So Phase 2's "no capacity limit flatters strategies that visit many bins" limitation **was not actually biting**. **The value was NOT changed** once that was known; lowering it until the table looked interesting would be tuning to the output. |

### `routing`

| Parameter | Value | Source | Status | Validated in | Why |
|---|---|---|---|---|---|
| `routing.n_clusters` | 20 | chosen | USED | Phase 2 | KMeans geographic clusters, one truck each; the paper splits the city into clusters without giving a count. 20 clusters over 5000 bins is 250 bins per territory. This value sets the baseline fleet size, so it also sets the fixed-cost denominator in two of the three cost bases. |
| `routing.n_trucks_per_cluster` | 1 | chosen | USED | Phase 4 | Phase 2 runs one depot-constrained tour per cluster. Phase 4's VRP relaxes it: `vrp.max_trucks_per_cluster` is the upper bound handed to the solver, and this value remains what Phase 2's tables were built on. At the committed capacity the solver chose **1 truck per cluster anyway** (5 trucks over 5 clusters), so Phase 2's assumption turned out to be what capacity would have forced in any case. At a deliberately binding 150 kg it chose 14. |
| `routing.two_opt_max_iters` | 1000 | chosen | USED | Phase 2 | Iteration cap for the 2-opt improvement pass, a runtime guard. 2-opt only accepts strict improvements, so hitting the cap returns a tour that is still no longer than the nearest-neighbour tour it started from (pinned in `tests/test_tsp.py`). |

### `vrp` - Phase 4 capacitated VRP and algorithm comparison (**every row invented and uncited**)

See the third banner at the top of this file.

| Parameter | Value | Source | Status | Validated in | Why |
|---|---|---|---|---|---|
| `vrp.capacity_sweep_kg` | [4000.0, 8000.0, 16000.0] | **invented, uncited** | USED | Phase 4 | The capacity sweep, which exists for the same reason `benchmark.fault_fractions` does: `cost.truck_capacity_kg` is invented, so the truck-count result must not rest on one guess. 8000 is exactly that config value; 4000 and 16000 bracket it. **Measured outcome: the sweep is flat.** All three sit far above the 296 kg per-cluster load, so all three are non-binding and distinguish nothing. That is a genuine weakness of the sweep's design and **the values were not retuned afterwards** - a separate `--capacities 150 400 8000` run shows the mechanism engaging, with 150 kg labelled *not a realistic truck*. |
| `vrp.max_trucks_per_cluster` | 20 | **invented, uncited** | USED | Phase 4 | Upper bound on vehicles offered to the solver per cluster. Not a fleet assumption: the solver uses as few as it can and the tables report how many it actually needed (1 per cluster at the committed capacity, 14 across 5 clusters at a binding 150 kg). Too low a value would make a feasible instance infeasible, which is why it is generous. |
| `vrp.ortools_time_limit_s` | 5.0 | **invented, uncited** | USED | Phase 4 | Per-instance OR-Tools search budget. **This is a WALL-CLOCK limit, and it is the reason no Phase 4 OR-Tools number is reproducible across machines**: a faster or less loaded machine searches more of the space in the same 5 s and can return a different route. Consequently **no test asserts an OR-Tools distance** - the tests assert route validity, every-bin-visited-once and capacity-respected instead - and every OR-Tools row in `docs/PHASE4_RESULTS.md` is labelled same-machine-only. `--quick` uses 1.0 s. |
| `vrp.first_solution_strategy` | `PATH_CHEAPEST_ARC` | **invented, uncited** | USED | Phase 4 | OR-Tools first-solution heuristic. Chosen once, never swept. `config.py` restricts it to a whitelist in `smartwaste.routing.vrp`, and `tests/test_vrp.py` checks every whitelisted name really exists in the solver's enum, so a typo is a load-time error rather than an `AttributeError` mid-run. |
| `vrp.local_search_metaheuristic` | `GUIDED_LOCAL_SEARCH` | **invented, uncited** | USED | Phase 4 | OR-Tools improvement phase. As above: one value, no sweep. **No solver tuning was performed at all**, deliberately - a tuned OR-Tools against an untuned 2-opt would be an unfair comparison, and the honest reading of the 0.29% margin is that it is a floor rather than the best OR-Tools could do. |
| `vrp.compare_clusters` | 20 | **invented, uncited** | USED | Phase 4 | How many KMeans clusters feed the comparison tables. Only the first N are taken, so a `--quick` run (5) is a strict **prefix** of the full one rather than a different sample. |
| `vrp.compare_day` | 14 | **invented, uncited** | USED | Phase 4 | Which simulated day supplies the bin sets. Every algorithm is compared on the **identical** bins from this one day, so the comparison is paired. Day 14 was picked to match `forecast.warmup_days`, purely so the figure is one the project already uses, **not** because any run was inspected first. **An influential limitation:** one day of one seed means every Phase 4 number has **no spread at all** - Phase 2 at least reported a five-seed range. |

### `roads` - Phase 4 real road network (**area and bin count invented and uncited**)

Road data (c) OpenStreetMap contributors, **ODbL 1.0** - https://www.openstreetmap.org/copyright

**Bin locations on the network are synthetic**: uniform-random points in the bbox snapped to the
nearest drivable road node. The roads are real; the bins are not.

| Parameter | Value | Source | Status | Validated in | Why |
|---|---|---|---|---|---|
| `roads.area_name` | `T. Nagar, Chennai, India` | chosen | USED | Phase 4 | The bounded area, printed in every table header. A **label only** - it is never geocoded, because a name lookup would make the extract depend on a remote search index that could silently return a different place. The bbox below is what is actually downloaded. |
| `roads.center_lat` | 13.0418 | **invented, uncited** | USED | Phase 4 | Approximate T. Nagar centre. Not surveyed; read off a map. Also the **depot**: the centre is snapped to a road node and used as the single central depot, matching Phase 2's model. |
| `roads.center_lon` | 80.2341 | **invented, uncited** | USED | Phase 4 | As above. |
| `roads.box_km` | 3.0 | **invented, uncited** | USED | Phase 4 | Bounding-box side length, so roughly 3 x 3 km. Chosen small on purpose: a bounded download, a 1.52 MB cache and a few thousand nodes, which keeps the run short enough to be honest about. **It also bounds the result**: the measured detour ratio is a 3 km-neighbourhood figure and nothing larger. The bbox is computed with a flat-earth degree conversion (111.32 km/degree, longitude scaled by cos(lat)); `tests/test_roads.py` **measures** that the resulting box really spans 3.0 km within 1% at this latitude. |
| `roads.network_type` | `drive` | chosen | USED | Phase 4 | OSMnx network filter. Collection trucks use drivable roads, so `drive` rather than `all`. Direction-aware for distance, so one-way streets are respected; **no** turn restrictions, traffic or congestion are modelled. |
| `roads.n_bins` | 300 | **invented, uncited** | USED | Phase 4 | Synthetic bins snapped to road nodes in the full run; `--quick` uses 50, which is what every published Phase 4 road number actually came from. **NOT comparable with `sim.n_bins` = 5000 on the 100 km grid** - different area, different count, different distance metric - and `config.py` requires `>= 2` because two bins is the smallest set with a distance between them. |
| `roads.cache_subdir` | `osmnx` | chosen | USED | Phase 4 | Subdirectory of `paths.cache_dir` holding the graphml cache. Gitignored. |

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
| `paths.data_dir` | `data` | chosen | USED | Phase 3 | Generated datasets. Gitignored. `run_anomaly.py` writes `anomaly_readings.csv` and `anomaly_labels.csv` here - **two separate files**, so the detectors can be handed the readings with no path to the planted truth. |
| `paths.cache_dir` | `cache` | chosen | USED | Phase 4 | Downloaded artifacts. Gitignored. `scripts\run_roads.py` writes the OSMnx road graph to `cache/osmnx/roads_<south>_<west>_<north>_<east>_<network>.graphml` (1.52 MB for the T. Nagar extract). The filename is derived from the bbox and the network filter, so a re-run with the same config hits the cache and needs **no network access** - verified. |
| `paths.results_dir` | `results` | chosen | USED | Phase 2 | Benchmark tables and figures. `run_benchmark.py` writes `benchmark.csv`/`benchmark.md`; Phase 3 adds `phase3_benchmark.*`, `forecast_eval.*` and `anomaly_eval.*`. Gitignored, so the committed numbers live in `docs/PHASE{1,2,3}_RESULTS.md`. |

## Not verified / Limitations

New in Phase 5:

- **The NodeMCU firmware has NEVER been compiled and NEVER been flashed.** `arduino-cli` was
  unavailable in the session that wrote it and no compilation was attempted; there is no ESP8266
  toolchain, no board, no waste and no reading. This closes nothing that Phase 0's "no firmware has
  been written or flashed" line left open - **the firmware is now written, and still not verified.**
  Every claim about what the sketch does is a claim about what its text says.
- **No ThingSpeak channel has ever been contacted.** No channel, no real read key, no real write
  key, and not one reading from the physical bin has ever been ingested. The only executed path is
  `--offline` against the committed fixture `tests/data/thingspeak_sample.json`; the `urllib` branch
  has never run. Phase 0's "zero readings from the physical bin" remains exactly true.
- **The firmware cross-check tests a Python transcription, not the C++.** `smartwaste.firmware_reference`
  is a hand translation of the sketch's arithmetic, and it agrees with `smartwaste.fusion` on 2000
  seeded readings plus every threshold boundary. That verifies the rule we *wrote down*. It cannot
  catch a C++ type error, a `float`-vs-`double` difference, a pin mis-assignment or a library quirk.
  The companion constants test applies a regular expression to a text file.
- **The ThingSpeak field mapping is invented and uncited** - see the banner above.
- **The wiring table in the firmware README is untested.** Pin choices, the HC-SR04 ECHO
  level-shifter advice and the HX711 calibration procedure are reasoned from datasheets, not from a
  built circuit.
- **Every figure the dashboard shows is synthetic or copied.** The demo snapshot is 200 bins x 14
  days of generated data; the baseline tables are text parsed out of the committed results docs at
  display time. **Not one figure on the dashboard is a hardware reading**, and a banner says so on
  every tab.
- **The demo snapshot is comparable with nothing.** 200 bins is 4% of `sim.n_bins`, 14 days is 47%
  of `sim.days`, its demand scenario is `heterogeneous` where Phase 2's tables are `smooth`, and its
  cost constants are the invented ones. No table anywhere places its figures beside a Phase 2
  number, deliberately.
- **The dashboard has only ever run headlessly**, via streamlit's in-process `AppTest`.
  `streamlit run` was never started, so no browser has rendered the page: nothing is known about its
  layout at a real viewport, on a slow connection, or in a browser's dark mode.
- **The map deliberately does NOT project onto real coordinates.** The city simulation has no
  latitude or longitude; bins are uniform-random x/y on a 100 x 100 km square. Painting them onto a
  real Chennai basemap would have looked better and would have repeated Phase 4's own stated problem
  - that a real road network makes synthetic bins look more authoritative than they are. No
  fabricated coordinate exists.
- **Yellow-bin opportunistic collection is STILL not built.** SPEC section 7 lists it under Phase 3;
  it was out of scope there, in Phase 4, and here. The demo snapshot makes the consequence visible
  without fixing it: the YELLOW count climbs to **57 of 200 bins by day 13**, with **11 overflow
  events** over the 14 days. **Carried to Phase 6.**
- **Phase 3's strategy benchmark and Phase 4's publication-size runs remain NOT RUN.** Phase 5 did
  not attempt either and changed nothing they depend on. The dashboard displays both with their own
  `NOT RUN` and `SMOKE RUN` labels rather than smoothing them over.
- **GitHub Actions CI has still never executed**, and Phase 5 enlarges what is unproven there: a
  sixth `dash` chunk and a fourth extra on the `types` job. Both verified locally only.
- **`.gitattributes` is new and its cross-platform effect is untested.** The `eol=lf` rule for the
  committed demo files is verified on this Windows machine with `core.autocrlf=true`; no clone on
  Linux or macOS has confirmed the round trip there.
- **Two bugs were found by this phase's own new tests**, which is recorded here because the tests
  are the only reason either was caught: `redact()` stopped at the first already-redacted `api_key=`
  and would have leaked a second key from a message quoting two URLs; and
  `importorskip("streamlit.testing.v1")` succeeded on the empty namespace-package shell `pip
  uninstall` leaves behind, so the `dash` chunk errored instead of skipping. Both are fixed with
  committed tests. Neither was in Phase 1-4 code.
- **Phase 5 ran on one machine only** - streamlit 1.66.0 / plotly 7.1.0 / Python 3.11.9 on Windows.
  The stdlib parts keep the project's byte-identical guarantee (the demo snapshot reproduces
  byte-for-byte, checked with `cmp`); the `AppTest` results are streamlit-version-dependent, and no
  test pins a rendered pixel or a layout.

New in Phase 4:

- **Phase 4's publication-size runs were NOT RUN.** Every table in `docs/PHASE4_RESULTS.md` is a
  `--quick` smoke run (5 clusters / 46 bins / 1 s solver budget; 50 road bins). The mechanism and
  the direction of each finding are established; **no city-scale magnitude is.** This is a
  deliberate departure from Phase 3, which omitted its un-run tables entirely - the reasoning is
  stated in that document rather than left implicit.
- **Truck capacity does not bind, so Phase 2's missing capacity limit flattered nothing.**
  Measured: 296.24 kg per cluster-day against an 8000 kg truck, 3.7% of capacity. Phase 2 and
  Phase 3 both listed "no truck capacity, so a tour may collect more than a real truck could
  carry, which flatters every strategy that visits many bins" as a limitation. **That turns out
  to have been wrong about this project's own constants** - with ~50 kg bins and ~9 bins per
  cluster per day there is no plausible truck it would strain. The limitation is therefore
  *withdrawn as stated* and replaced by this measurement. **No constant was changed to rescue the
  table.**
- **OR-Tools does not meaningfully beat the hand-written 2-opt**: 0.29% shorter, against a 2.0%
  "meaningful" threshold that is a module constant in `smartwaste.phase4` and was declared before
  any run. Plain 2-opt over nearest neighbour is worth twenty times more (5.88%). CLAUDE.md
  requires a simple baseline matching the sophisticated one be said plainly, so it is the headline
  of that section. `ortools` is justified by being able to *express* the capacitated multi-truck
  problem at all, not by distance.
- **The noisy-plan-vs-oracle comparison is a TIE, and that is a null result, not a win.** Both
  rows executed 560.19 km with zero breaches. It follows directly from capacity not binding: a
  load-cell error of a few kg cannot push a 296 kg tour past 8000 kg, so **sensor noise is never
  tested at the planning stage.** The machinery is built and tested on hand-built cases where a
  plan genuinely fails; the city's own numbers simply never exercise it.
- **The oracle row is a reference line, not a result**, and is labelled *upper bound, not
  realistic* in the table, the verdict and the code. It is handed the true weight, which no
  deployment has before the truck arrives. Its zero breaches are **by construction** and
  `tests/test_vrp.py` pins that invariant rather than measuring it.
- **A straight line under-states road distance by ~1.45x at the median in T. Nagar**, p10 1.23,
  p90 2.14, **max 12.02**. This closes a question `docs/ASSUMPTIONS.md` has carried open since
  Phase 0. Three non-interchangeable aggregates come out of the same data - 1.45 pairwise, 1.54 on
  total distance, 1.83 on a routed tour - and the pairwise one is primary because it excludes
  unreachable pairs outright instead of substituting a stand-in.
- **That ratio must NOT be applied to Phase 2's distances.** One ~3 km area, synthetic bins, one
  seed. Phase 2's geometry is 100 km of uniform-random points, where long legs dominate and the
  proportional detour would be *smaller*. No table anywhere draws that comparison, deliberately.
- **Bin locations on the road network are synthetic.** The roads are real OpenStreetMap data; the
  bins are uniform-random points snapped to road nodes. A real road network makes synthetic bins
  look more authoritative than they are, which is why the sentence appears in the module
  docstring, the stdout header and every table.
- **One instance set, one day, one seed - so no Phase 4 number has any spread.** The comparison
  uses the bins strategy (d) selected on day 14 of seed 42. Phase 2 reported a five-seed range;
  Phase 4 reports a point estimate, which is weaker.
- **Every OR-Tools number is same-machine-only.** A wall-clock time limit means a different
  machine can return a different route. Rows come from ortools 9.15.6755 / osmnx 2.1.1 /
  networkx 3.6.1 on Windows, Python 3.11. The stdlib paths keep the project's byte-identical
  guarantee and were checked with `cmp` (3,801 and 1,560 bytes).
- **OR-Tools searches on a rounded integer matrix** (km x 1000, metres), because it requires
  integer arc costs. Reported distances are recomputed in float from the returned tour with Phase
  2's own `route_distance`, so the comparison does not also measure the rounding - but the search
  did see rounded values. Capacity demands are rounded **up**, so that rounding can only overstate
  a load, never understate one.
- **The capacity model is crude.** Return-to-depot-and-resume charges distance only: no extra
  service time, no tipping time, no driver-hour limit. A bin heavier than the whole truck is
  flagged and still collected, because abandoning it forever would be a different and less honest
  model.
- **Capacity is applied per cluster**, and KMeans territories are unchanged from Phase 2, so a
  truck cannot cross into another cluster. The truck counts are therefore an **upper bound** on
  what a city-wide VRP would need.
- **The full Phase 2 benchmark was NOT re-run this phase.** "Phase 2 still reproduces" rests on
  `tests/test_demand.py::test_smooth_scenario_reproduces_the_phase2_numbers_exactly` (which pins
  its totals as literals and passes) plus a `git status` check that no Phase 2/3 source file
  changed - not on a fresh 63-row comparison as Phase 3 did.
- **Yellow-bin opportunistic collection is STILL not built.** SPEC section 7 lists it under Phase
  3; it was out of scope there and out of scope here. The paper's known weakness now carries into
  Phase 5.
- **No tests touch the network**, which is a strength but also a limit: the road code paths are
  tested against a hand-built 4-node graph with one edge removed, so OSMnx's own download, snapping
  and graphml round-trip are exercised only by the smoke runs, never by a test.
- **GitHub Actions CI has still never executed.** The new `roads` extra in the `types` job and the
  `routing` chunk are committed but unproven.

New in Phase 3:

- **Phase 2 was re-checked, not assumed.** Phase 3 changed `city_sim.py` and `strategies.py`, so
  the full Phase 2 benchmark was re-run from the Phase 3 tree: all **63 data rows** it produced
  appear verbatim in `docs/PHASE2_RESULTS.md`. Phase 2's entry points were deliberately not
  widened (Phase 3 added a parallel runner instead), `build_layout`'s new `scenario` argument is
  keyword-only and defaults to `"smooth"`, and a test pins Phase 2's totals as literals. All five
  changed test files are **new** - no pre-existing test was edited.

- **The entire Phase 3 demand model is invented** (see the second banner above), and it is the
  foundation everything else in Phase 3 stands on. Area types, the weekday cycle and festival
  spikes have no field data behind them. **Where a Phase 3 result depends on demand varying, it
  holds on our invented demand only** and is not evidence about a real city.
- **The city-scale strategy benchmark was NOT RUN, so Phase 3 settles nothing about collection
  outcomes.** The full sweep measures about 2.3 hours (the phase was planned on a 35-50 minute
  estimate, which did not account for 2-opt cost growing with the larger per-day bin sets the
  forecast strategies select). `docs/PHASE3_RESULTS.md` therefore contains **no** overflow,
  distance or cost figure for strategy (f), and **no verdict** on whether Phase 2's "a
  sensor-free 3-day calendar matches the paper's fusion" finding survives heterogeneous demand -
  in either direction. The code, the ten strategies and the two scoring windows are built and
  tested; only the publication-size run is missing. A lower forecast MAE is **not** evidence of a
  better collection outcome, which was Phase 2's whole lesson.
- **The simple baseline beat the ML detector on every anomaly family, including the two fair
  tests.** The stdlib rule baseline beat the Isolation Forest on PR-AUC overall (0.5664 vs
  0.1523) and on all four families. CLAUDE.md requires that be said plainly rather than buried,
  so it is the headline of the anomaly section.
- **`stuck` and `dropout` favour the rule baseline by construction** and are labelled as such in
  every table. The rule's zero-variance test is very nearly a *definition* of those two faults,
  so a win there demonstrates a mechanism; it is not a measurement. `drift` and `noise_burst`
  are the fair tests, and the rule won those too - by much smaller margins.
- **Both detectors are nearly blind to drift.** PR-AUC 0.0311 (rule) and 0.0152 (forest) against
  a prevalence of 0.0071. A slow drift looks like a slowly filling bin, which is exactly what the
  sensor is supposed to report. `anomaly.drift_cm_per_day` is invented, and a larger value would
  be easier to catch, so this is partly a statement about our fault model.
- **The gradient-boosting model trained on ZERO festival days.** Measured: `forecast.warmup_days
  = 14` and both of the horizon's 2 festival days fell outside it. The model is therefore scored
  on days containing a phenomenon it never saw in training. **No constant was changed to fix
  this**, because lengthening the warm-up until the model looked better would be tuning to the
  result.
- **The forecaster predicts its own sensors, not the truth.** Targets are the future *fused
  noisy estimate*, because a forecaster on real hardware has no ground truth to regress on. The
  honest consequence is that a persistent sensor bias is **learned rather than corrected**, and
  the reported MAE/RMSE measure agreement with the sensors, not with reality.
- **ML numbers are reproducible on this machine only.** The stdlib parts of this project keep
  their cross-machine byte-identical guarantee, but scikit-learn promises no bit-identical floats
  across operating systems, BLAS builds or library versions. The reported ML rows come from
  **numpy 2.4.6 and scikit-learn 1.9.1 on Windows / Python 3.11**. No test pins an exact float
  from a model; the ML tests assert properties instead.
- **scikit-learn was substituted for xgboost.** SPEC section 5 says "XGBoost first"; Phase 3 uses
  `HistGradientBoostingRegressor` - the same model family, one fewer heavy dependency. That is a
  deviation from the spec, recorded here rather than quietly made.
- **No hyperparameter search was performed, for either model.** `gbm_max_iter`,
  `gbm_learning_rate`, `gbm_max_depth` and `n_estimators` were each chosen once before any run.
  This is deliberate: a tuned ML model against an untuned baseline would be an unfair comparison.
  It also means the ML results are a floor, not the best the model could do.
- **The anomaly window label is a modelling choice.** A window counts as faulty when **its last
  day** falls inside the planted episode. Labelling by "any day overlaps" would mark windows
  faulty before the fault had produced visible evidence; "every day overlaps" would discard the
  onset. Neither is more correct; the choice is stated, not justified.
- **Forecast-driven collection spreads visits across days, which is expected to cost more per
  bin.** The calendars collect every bin on one day and get one efficient tour; strategy (f)
  collects smaller sets on more days, so each tour pays the depot trip again. This is a
  structural consequence of the routing model (one tour per cluster per day, single central
  depot) rather than a forecasting failure - but note it is a **prediction from the model's
  shape, not a measurement**, since the benchmark was not run.
- **The headline window is 16 of 30 days** (days 14-29), because the forecasters' warm-up is
  excluded. Every strategy is scored on that identical window, and overflow is reported as events
  per bin-day so the 16-day and 30-day tables could be read against each other. This is how the
  benchmark is built and tested; it has not yet produced published numbers.
- **Yellow-bin opportunistic collection was NOT built.** SPEC section 7 lists it under Phase 3;
  this phase's scope excluded it, so YELLOW bins are still never collected except when a forecast
  or the backstop happens to pick them up. The paper's known weakness is therefore only partly
  addressed.
- **The anomaly dataset uses a plain saw-tooth fill process**, not the Phase 3 demand model. It
  is a dataset about *sensor behaviour*, so the fill process is deliberately simple; that also
  means the detectors have never been tested against festival spikes or area heterogeneity.
- **No VRP, no truck capacity, no road network** in Phase 3 itself - every Phase 3 distance is
  Euclidean on the same uniform-random 100 km grid. **Phase 4 built all three**, as a parallel
  path that left Phase 3's numbers untouched; it did **not** re-run Phase 3's benchmark, which
  remains NOT RUN.

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
- **No truck capacity limit** in Phase 2. **Phase 4 added one and found it does not bind:** a
  cluster's whole day is ~296 kg against an 8000 kg truck, so this limitation, as originally
  stated ("flatters every strategy that visits many bins"), **was wrong about this project's own
  constants.** The Phase 2 numbers are unaffected either way.
- **Euclidean distance on a uniform-random grid.** Distances are straight lines between random
  points in a 100 x 100 km square, so they are not road distances and the layout is not a city.
  **Phase 4 added a real road network for one 3 km Chennai area** and measured a ~1.45x median
  detour there - but on a different geometry at a different scale, so the Phase 2 distances stand
  uncorrected and are not comparable with it.
- **One truck per cluster, single central depot, no time windows, no shift limits.** Travel times
  of thousands of hours per 30 days are spread across 20 trucks with no cap on a driver's day.
- **KMeans clusters are fixed for the whole month** and computed on bin locations only, ignoring
  how much waste each bin produces, so territories are balanced by area rather than by workload.
- **Yellow bins are still never collected** under every sensor strategy, faithfully reproducing the
  paper's rule and its known weakness. **Phase 3 did NOT fix this** - opportunistic Yellow-bin
  collection was outside its scope, so the weakness survives into Phase 4. Phase 3's
  forecast-driven strategies do pick up some YELLOW bins, but only incidentally, when a forecast
  or the backstop schedule happens to select them.
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

- **Every config value not marked `USED` above is a `PLACEHOLDER`.** **As of Phase 4 there are
  none left:** `cost.truck_capacity_kg`, `routing.n_trucks_per_cluster` and `paths.cache_dir` -
  the last three - were promoted by the Phase 4 VRP and roads runs. **No value anywhere in this
  file is `VALIDATED`**, and that has not changed: every row is still either `paper`, or a guess
  of ours with no citation behind it.
- **Only 1 physical bin was built** (in the paper's hardware work). Everything at city scale is
  simulation.
- **The 5000-bin, 30-day results are simulation**, not measurements from deployed bins.
- **All data is synthetic.** Fill levels, sensor readings and faults are generated, not collected.
- **Distances are Euclidean** on a random grid everywhere in Phases 1-3. **Phase 4 measured the
  understatement** this line used to call unmeasured: ~1.45x at the median on real T. Nagar roads
  (p10 1.23, p90 2.14, max 12.02). **That factor does not apply to the Phase 1-3 numbers** - it
  was measured on a ~3 km area with synthetic bins, not on the 100 km grid - so every Phase 1-3
  distance remains Euclidean and uncorrected.
- **The paper's baseline is naive** (visit every bin daily), so most of any reported saving
  is simply skipping bins rather than better routing. Phase 2 added the fairer baselines
  (b) and (c), and they confirm it: a sensor-free 3-day schedule captures essentially the
  same saving as the paper's sensor fusion.
- **Yellow bins are never collected** under the paper's rule, which risks overflow. Still
  true after Phase 3: the dedicated Yellow-bin logic SPEC section 7 lists under Phase 3 was
  **not built**, being outside this phase's scope. **Still true after Phase 5**, which was out of
  scope again - but Phase 5 at least made it *visible*: the dashboard's demo snapshot shows the
  YELLOW count climbing to 57 of 200 bins by day 13 with 11 overflow events. Carried to Phase 6.
- **The paper's own reported figures** (34.04% travel time, 22.22% cost, 31.85% CO2) are
  reference only. We do not promise to reproduce them and will not tune toward them.
- **Paper inconsistencies** carried forward knowingly: 30 days vs "three-month"; "34.04%
  distance" vs the travel-time figure in Table I; Eq. 13 printing `F = bD` instead of `E = bD`;
  "1926 percent" for 19-26%.
- **No firmware has been written or flashed**; no Docker image exists. Both are later phases and
  will be marked NOT VERIFIED unless actually run. **Updated in Phase 5: the firmware is now
  written, and it is still NOT VERIFIED** - never compiled, never flashed, no `arduino-cli`
  available. Only the second half of this line was discharged. Docker remains untouched.
- **GitHub Actions CI has never executed.** The workflow is committed but nothing has been
  pushed (CLAUDE.md forbids `git push`), so CI is NOT VERIFIED.
