# Phase 3 results — heterogeneous demand, fill forecasting, sensor anomaly detection

## How these numbers were produced

```
python scripts\run_forecast.py --seed 42
python scripts\run_anomaly.py --seed 42
```

> **The strategy benchmark (`run_phase3_benchmark.py`) has NOT been run at publication size.**
> Its tables are absent from this document rather than filled from a smaller run — see
> [Strategy benchmark: NOT RUN](#strategy-benchmark-not-run) below. Everything else in this
> document comes from the two commands above.

- **Seed:** 42. Every sub-seed is derived from the master seed with sha256.
- **Code commit:** `694ceb4086e9fc5b2b793cf3c32679b2938a3aaf` ("Phase 3: heterogeneous demand,
  fill forecasting, sensor anomaly detection"). The working tree was clean when these ran, so
  every number below comes from committed code.
- **Environment:** Python 3.11, Windows. **numpy 2.4.6, scikit-learn 1.9.1** (printed by
  `python scripts\run_forecast.py --versions`).
- **Config:** every constant from `config/default.yaml` as committed. **Nothing was tuned.**

Every table below is pasted verbatim from those commands' real output, and a checker confirmed
each pasted numeric line appears verbatim in the captured run logs. Re-running them regenerates
`results/forecast_eval.{csv,md}` and `results/anomaly_eval.{csv,md}` (both gitignored).

## Why Phase 3 exists, and what it had to fix first

Phase 2 produced a null result and reported it: strategy (b) — every bin every 3 days, with **no
sensors at all** — came within 0.75 pp of the paper's sensor fusion on cost and overflowed
**zero** times against (d)'s 90. `docs/PHASE2_RESULTS.md` named the cause rather than explaining
it away: arrivals were i.i.d. Gaussian with mean 4.0 kg and sd 1.5 kg, so every bin filled at
very nearly the same predictable rate. That is the friendliest possible world for a fixed
calendar, and it leaves a forecaster **nothing to learn**.

So Phase 3 fixed the *experiment* before adding any model:

1. a heterogeneous demand model (area types, a weekday cycle, festival spikes) — **invented and
   uncited**, labelled as such everywhere;
2. fill-level forecasting, naive baselines first, then one gradient-boosting model;
3. sensor anomaly detection on planted faults, a rule baseline against an Isolation Forest.

## Scope of Phase 3

**In:** the heterogeneous demand model with the Phase 2 "smooth" process kept selectable and
reproducible; fill forecasting with three naive baselines and a gradient-boosting model on a
time-based split; forecast-driven collection (strategy f) at two backstop schedules, each paired
with the sensor-free calendar of its own period; sensor anomaly detection with four planted fault
families, scored by precision / recall / PR-AUC on labels held in a separate file.

**Out, deliberately:** **no Yellow-bin opportunistic collection.** Both `docs/SPEC.md` §7 and
`docs/PHASES.md` list it under Phase 3 ("predict time-to-Red, collect Yellow bins only if near an
existing route"); **this phase's instructions explicitly excluded it**, so it was not built. The
paper's "YELLOW bins are never collected" weakness is therefore only *incidentally* addressed — a
forecast or a backstop visit sometimes picks a YELLOW bin up, but no rule targets them. It
carries into Phase 4. Also out: no VRP, no truck capacity, no multi-truck-per-cluster
and no real road network (all Phase 4); no hardware, no firmware, no ThingSpeak.

## Phase 2 stays reproducible — this is a guarantee, not a hope

Phase 3 did **not** widen Phase 2's entry points. `scripts\run_benchmark.py`,
`smartwaste.benchmark.run_city` and `build_strategies` are untouched and still produce exactly the
five-strategy Phase 2 tables; Phase 3 added a parallel `scripts\run_phase3_benchmark.py` instead.
`build_layout`'s new `scenario` argument is keyword-only and **defaults to `"smooth"`**, which is
Phase 2's arrival process reproduced bit for bit from the same seed labels.

Three things back it up:

- **The full Phase 2 benchmark was re-run from the Phase 3 tree and matched**, row for row — see
  [Phase 2 reproducibility gate](#phase-2-reproducibility-gate) for the check.
- `tests/test_demand.py::test_smooth_scenario_reproduces_the_phase2_numbers_exactly` holds Phase
  2's per-strategy totals as **literals captured from the pre-Phase-3 tree**, and fails if any
  drift.
- **Every pre-existing test passes unchanged.** Verified with
  `git diff --name-only 5a3758a HEAD -- tests/`: all five changed test files are **new**; not one
  existing test file was edited, weakened or deleted to accommodate Phase 3.

## The ten strategies

| id | label | sensors | rule |
|---|---|---|---|
| a | fixed-route, all bins daily | none | the paper's baseline |
| b | fixed schedule, every 3 days | none | sensor-free baseline for the 3-day backstop |
| **b7** | **fixed schedule, every 7 days** | none | **new in Phase 3** — sensor-free baseline for the 7-day backstop |
| c | single-ultrasonic threshold | centre only, no load cell | collect RED |
| d | **the paper's sensor fusion** | 3-sensor MEAN + load cell | collect RED |
| e | median fusion — extension, not in the paper | 3-sensor MEDIAN + load cell | collect RED |
| f_ma_3 | forecast-driven, **moving-average baseline**, 3-day backstop | fusion + forecast | collect when forecast reaches RED first |
| f_ma_7 | forecast-driven, **moving-average baseline**, 7-day backstop | fusion + forecast | as above |
| f_gbm_3 | forecast-driven, **gradient boosting [ML]**, 3-day backstop | fusion + forecast | as above |
| f_gbm_7 | forecast-driven, **gradient boosting [ML]**, 7-day backstop | fusion + forecast | as above |

### Why b7 exists, and why it matters

A 7-day-backstop forecast strategy collects far less often than a 3-day calendar. Measuring it
against (b) would confuse **"forecasting works"** with **"visiting less often is cheaper"**. So
each (f) is compared against the sensor-free calendar of *its own* period: `f_*_3` against (b),
`f_*_7` against (b7). Without b7 the 7-day rows would be unreadable.

### The backstop is the same absolute calendar, not a per-bin clock

Strategy (f) collects a bin when it is already RED, when the forecast says it reaches RED before
the next scheduled visit, **or** when its backstop visit is due — and "due" is the same
`day % N == 0` calendar `FixedSchedule` uses. Two consequences, both deliberate:

- during the warm-up (f) is **exactly** its calendar partner, which is what makes the post-warm-up
  window a clean comparison (pinned by a test);
- (f)'s visits are a strict **superset** of its calendar's, so a wrong forecast can only ever add
  a visit, never strand a bin by pushing its next visit later. A relative "days since collection"
  clock broke both properties, and an earlier draft of this phase had that bug.

## The two scoring windows

*(How the benchmark is built. It has not been run at publication size, so no table below uses
these windows — see [Strategy benchmark: NOT RUN](#strategy-benchmark-not-run).)*

The forecasters spend days 0–13 warming up, so a 30-day total would credit (f) with 14 days in
which it was simply a calendar. Every table the benchmark emits therefore names its window:

- **Post-warm-up window, days 14–29 (16 days) — the headline.** **Every** strategy is scored on
  this identical window, calendars and the paper's fusion included. Any (f)-vs-anything claim
  would be made here and nowhere else.
- **Full horizon, 30 days — reported separately**, for Phase 2's five strategies, so they line up
  directly against `docs/PHASE2_RESULTS.md`.

Overflow is emitted as **events per bin-day** as well as a raw count, because raw counts cannot
be read across a 16-day and a 30-day window.

## Strategy benchmark: NOT RUN

**There are no strategy-comparison tables in this document, because the benchmark that produces
them has not been run at publication size.**

What exists and what does not:

| item | state |
|---|---|
| the ten strategies, the two scoring windows, the b7 calendar | **built and tested** (387 tests, see Verification) |
| `scripts\run_phase3_benchmark.py` | **works**, and its output is deterministic (verified below) |
| a `--quick` smoke run (1000 bins, 2 seeds, 1 fault fraction) | **ran successfully**, but its tables are self-labelled `SMOKE RUN, NOT FOR PUBLICATION` and are deliberately **not** reproduced here |
| the full default sweep (5000 bins × 30 days × 5 seeds × 3 fault fractions × 2 scenarios) | **NOT RUN** |

### Why it was not run

The full sweep costs **about 2.3 hours** on this machine (a lower bound — see the measured
runtimes under [Determinism](#determinism)). That is far more than the 35–50 minutes estimated
when this phase was planned; the estimate was wrong because it did not account for the 2-opt TSP
pass growing superlinearly with the larger per-day bin sets the forecast-driven strategies
select. A first attempt was started and was killed at roughly one hour when the session ended.

Rather than fill these tables from the 1000-bin smoke run and hope the reader notices the
caveat, the tables are omitted. **The honest state is "not measured", and that is what this
section says.**

### What this means for Phase 3's claims

Phase 3's *forecast accuracy* and *anomaly detection* results below are complete and verified.
What is **not** established is the **downstream** question — whether forecast-driven collection
actually reduces overflow, distance and cost against the calendars at city scale.

So **this document makes no claim about that**, in either direction. In particular:

- it does **not** claim that Phase 2's null result ("a sensor-free 3-day calendar matches the
  paper's fusion") fails under heterogeneous demand;
- it does **not** claim that it survives;
- it does **not** report any overflow, distance or cost figure for strategy (f).

A lower forecast MAE is **not** evidence of a better collection outcome. The whole point of
Phase 2 was that a method can be more accurate and still not pay for itself, so the downstream
metrics are the ones that would settle it — and they are missing.

### To produce them

```
python scripts\run_phase3_benchmark.py --seed 42
```

writes `results/phase3_benchmark.{csv,md}` (gitignored) and prints the headline window table, the
full-horizon table for Phase 2's five strategies, the training-row diagnostics and two verdict
blocks — "does forecasting earn its keep?" and "does Phase 2's finding survive?". Expect ~2.3 h.
`--quick` gives a ~4 min smoke run whose tables are labelled not for publication.

## Forecasting: does the ML model beat the naive baselines?

```
python scripts\run_forecast.py --seed 42
```

400 bins × 30 days, observed under a fixed calendar so all four forecasters see the **identical
rows** — same city, same arrivals, same sensor noise. The comparison is paired. Error is in
**percentage points of fill**, so it reads directly against the RED threshold of 80 pp.

### Forecast accuracy, demand scenario `smooth`

| forecaster | seed | train rows | eval rows | MAE (fill pp) | RMSE (fill pp) |
|---|---|---|---|---|---|
| last value [naive baseline] | 42 | 3600 | 4800 | 8.027 | 10.378 |
| moving average [naive baseline] | 42 | 3600 | 4800 | 5.608 | 7.194 |
| same-weekday seasonal naive [naive baseline] | 42 | 3600 | 4800 | 7.757 | 10.126 |
| gradient boosting (HistGradientBoostingRegressor) [ML] | 42 | 3600 | 4800 | 4.357 | 5.575 |

### Forecast accuracy, demand scenario `heterogeneous`

| forecaster | seed | train rows | eval rows | MAE (fill pp) | RMSE (fill pp) |
|---|---|---|---|---|---|
| last value [naive baseline] | 42 | 3600 | 4800 | 16.071 | 25.784 |
| moving average [naive baseline] | 42 | 3600 | 4800 | 11.936 | 16.919 |
| same-weekday seasonal naive [naive baseline] | 42 | 3600 | 4800 | 13.312 | 19.480 |
| gradient boosting (HistGradientBoostingRegressor) [ML] | 42 | 3600 | 4800 | 6.568 | 9.520 |

Verdict lines, verbatim from the same run:

```
Verdict: does the ML model beat the naive baselines?
  smooth: the ML model beats last value [naive baseline] on MAE (4.357 vs 8.027 pp)
  smooth: the ML model beats moving average [naive baseline] on MAE (4.357 vs 5.608 pp)
  smooth: the ML model beats same-weekday seasonal naive [naive baseline] on MAE (4.357 vs 7.757 pp)
  heterogeneous: the ML model beats last value [naive baseline] on MAE (6.568 vs 16.071 pp)
  heterogeneous: the ML model beats moving average [naive baseline] on MAE (6.568 vs 11.936 pp)
  heterogeneous: the ML model beats same-weekday seasonal naive [naive baseline] on MAE (6.568 vs 13.312 pp)
```

### Reading this honestly

**The gradient-boosting model wins on forecast accuracy, on both scenarios, against all three
baselines.** This is the one place in Phase 3 where the ML model is clearly ahead, and the margin
is not marginal: on heterogeneous demand it halves the moving average's MAE (6.568 vs 11.936 pp).

Four caveats that belong next to that result, not below it:

1. **The moving average is by far the strongest baseline**, and the one worth comparing against.
   `last_value` and `seasonal_naive` are both *worse* than a 3-day moving average on both
   scenarios. Reporting only "the ML model beat the naive baselines" would overstate the gap;
   against the best baseline it is 5.608 → 4.357 pp on smooth demand, a far smaller win than
   against `last_value`.
2. **Every error roughly doubles under heterogeneous demand** (the GBM goes 4.357 → 6.568 pp).
   Varying demand is harder to predict for everything, which is the point: it is also what gives
   a forecaster something to do.
3. **`seasonal_naive` is barely seasonal over a 30-day horizon.** It falls back to the mean rise
   whenever a week of history is unavailable, which early in the run is most of the time. It is
   therefore a weaker test of seasonality than its name suggests.
4. **The target is the sensors' own future estimate, not the truth** — see the limitation below.

### Warm-up diagnostics: how much the model had to learn from

| scenario | seed | warm-up days | distinct weekdays in warm-up | festival days in warm-up | festival days in horizon | train rows | eval rows |
|---|---|---|---|---|---|---|---|
| smooth | 42 | 14 | 7 | 0 | 0 | 3600 | 4800 |
| heterogeneous | 42 | 14 | 7 | 0 | 2 | 3600 | 4800 |

**The gradient-boosting model trained on ZERO festival days**, then was scored on a horizon
containing two of them. That is measured, not estimated. It saw all 7 weekdays, so the weekday
cycle was learnable; the festival spike was not.

**No constant was changed to fix this.** `forecast.warmup_days` stays at 14. Lengthening the
warm-up until the ML model looked better would be tuning to the result, which this project
forbids. The honest reading is that the GBM's win above was achieved **despite** never seeing the
largest single driver of variance in the demand model — and that a longer warm-up would probably
widen its lead rather than narrow it.

## Sensor anomaly detection: the simple rule beats the ML detector

```
python scripts\run_anomaly.py --seed 42
```

400 bins × 60 days × a 5-day rolling window = **22,400 scored windows**, of which 699 are faulty
(**prevalence 0.0312**).

### Thresholds, and the seed separation that chose them

| knob | value |
|---|---|
| rule: stuck tolerance (cm) | 2.000 |
| rule: disagreement limit (cm) | 25.000 |
| Isolation Forest: contamination | 0.100 |
| best rule F1 on the validation city | 0.6235 |
| best forest F1 on the validation city | 0.2009 |
| validation seed (thresholds chosen here) | 43 |
| scoring seed (every reported number) | 42 |

Every knob was chosen by F1 on the **validation** city's planted labels and then applied
unchanged to the **scoring** city. The scoring city's labels are read only to compute the table
below, after these values were frozen. Choosing thresholds on the scoring labels would have made
every number here optimistic. `config.py` **rejects** a zero validation offset, so the separation
cannot be switched off by a config edit.

### Precision, recall and PR-AUC on the planted labels

**Accuracy is deliberately absent, and is not implemented anywhere in the codebase.** At this
prevalence a detector that never flags anything would score **96.9%** accuracy while catching
nothing. PR-AUC must be read against the prevalence column: a useless ranker scores about the
prevalence, not 0.5.

| detector | fault family | how to read it | n windows | n faulty | prevalence | precision | recall | PR-AUC |
|---|---|---|---|---|---|---|---|---|
| rule baseline: zero variance OR 3-sensor disagreement [stdlib] | overall | all families pooled | 22400 | 699 | 0.0312 | 0.8300 | 0.5308 | 0.5664 |
| rule baseline: zero variance OR 3-sensor disagreement [stdlib] | stuck | favors the rule baseline by construction | 21881 | 180 | 0.0082 | 0.6592 | 0.8167 | 0.6333 |
| rule baseline: zero variance OR 3-sensor disagreement [stdlib] | drift | fair test | 21856 | 155 | 0.0071 | 0.1828 | 0.1097 | 0.0311 |
| rule baseline: zero variance OR 3-sensor disagreement [stdlib] | dropout | favors the rule baseline by construction | 21861 | 160 | 0.0073 | 0.6652 | 0.9437 | 0.7337 |
| rule baseline: zero variance OR 3-sensor disagreement [stdlib] | noise_burst | fair test | 21905 | 204 | 0.0093 | 0.4242 | 0.2745 | 0.1955 |
| Isolation Forest [ML] | overall | all families pooled | 22400 | 699 | 0.0312 | 0.1330 | 0.4263 | 0.1523 |
| Isolation Forest [ML] | stuck | favors the rule baseline by construction | 21881 | 180 | 0.0082 | 0.0614 | 0.7056 | 0.1742 |
| Isolation Forest [ML] | drift | fair test | 21856 | 155 | 0.0071 | 0.0152 | 0.1935 | 0.0152 |
| Isolation Forest [ML] | dropout | favors the rule baseline by construction | 21861 | 160 | 0.0073 | 0.0587 | 0.7562 | 0.0780 |
| Isolation Forest [ML] | noise_burst | fair test | 21905 | 204 | 0.0093 | 0.0102 | 0.0980 | 0.0116 |

Verdict lines, verbatim from the same run:

```
Verdict: rule baseline vs Isolation Forest, per fault family
  overall [all families pooled]: the RULE BASELINE beats the Isolation Forest on PR-AUC (0.5664 vs 0.1523); prevalence 0.0312
  stuck [favors the rule baseline by construction]: the RULE BASELINE beats the Isolation Forest on PR-AUC (0.6333 vs 0.1742); prevalence 0.0082
  drift [fair test]: the RULE BASELINE beats the Isolation Forest on PR-AUC (0.0311 vs 0.0152); prevalence 0.0071
  dropout [favors the rule baseline by construction]: the RULE BASELINE beats the Isolation Forest on PR-AUC (0.7337 vs 0.0780); prevalence 0.0093
  noise_burst [fair test]: the RULE BASELINE beats the Isolation Forest on PR-AUC (0.1955 vs 0.0116); prevalence 0.0093
```

### Reading this honestly

**The stdlib rule baseline beats the Isolation Forest on every single family, including both
fair tests.** Overall PR-AUC 0.5664 vs 0.1523 — nearly 4×. CLAUDE.md requires this be stated
plainly rather than buried, so it is the headline of this section rather than a footnote.

- **Two families favour the rule by construction, and are labelled so in the table.** The rule's
  zero-variance test is very nearly a *definition* of `stuck` and `dropout`, so its wins there
  (PR-AUC 0.6333 and 0.7337) demonstrate a mechanism; they are **not** evidence that rules beat
  ML in general.
- **The rule also wins the two fair tests**, by smaller margins: `noise_burst` 0.1955 vs 0.0116
  and `drift` 0.0311 vs 0.0152. The 3-sensor disagreement test catches a noisy sensor reasonably
  well, and that is a fair catch — nothing about the noise-burst fault was designed around it.
- **Both detectors are nearly blind to drift.** 0.0311 and 0.0152 against a prevalence of 0.0071:
  the rule is roughly 4× better than chance and the forest roughly 2×, which in absolute terms is
  "almost nothing" for both. A slow drift looks exactly like a slowly filling bin, which is what
  the sensor is supposed to report. `anomaly.drift_cm_per_day = 1.2` is invented, and a larger
  value would be easier to catch, so this is partly a statement about our fault model.
- **The forest has high recall and terrible precision** (overall 0.4263 recall at 0.1330
  precision). It flags roughly as many windows as its contamination parameter tells it to, and
  most of them are healthy. An unsupervised outlier detector on 14 hand-built features is simply
  a weaker tool here than two targeted physical tests.
- **The forest was not tuned beyond contamination.** `n_estimators` was fixed at 200 and no
  feature selection or hyperparameter search was done — deliberately, since a tuned ML model
  against an untuned baseline is not a fair comparison. A better-engineered ML detector may well
  beat the rule; this result says the *obvious* ML detector does not.

## Honest summary

**What Phase 3 established:**

- **The ML forecaster earns its keep on forecast accuracy.** It beats all three naive baselines
  on both demand scenarios, halving the best baseline's error on heterogeneous demand (6.568 vs
  11.936 pp MAE) — and it did so **despite training on zero festival days.**
- **The ML anomaly detector does not earn its keep.** A two-rule stdlib baseline beats Isolation
  Forest on every fault family, both fair tests included (overall PR-AUC 0.5664 vs 0.1523).
- **Heterogeneous demand is genuinely harder to predict**: every forecaster's error roughly
  doubles against the smooth scenario. That is the sign the Phase 2 experiment really was too
  easy, which was the premise of this phase.
- **The demand model does not cheat on volume.** `demand.area_mean_kg` was rescaled, before any
  run, to hold the mean daily arrival within 1.3% of Phase 2's 4.0 kg/bin/day, so the two
  scenarios differ in *variance* rather than in how much waste the city makes. A test measures it.
- **Phase 2 remains reproducible.** Its entry points are untouched and its per-strategy totals
  are pinned as literals; every pre-existing test passes unchanged, with no edits.

**What Phase 3 did NOT establish:**

- **Whether forecast-driven collection is actually worth it.** The city-scale benchmark was not
  run (see [Strategy benchmark: NOT RUN](#strategy-benchmark-not-run)), so there is **no**
  overflow, distance or cost figure for strategy (f) in this document, and **no** verdict on
  whether Phase 2's "a dumb calendar matches the sensors" finding survives heterogeneous demand.
  Better forecast accuracy is not evidence of a better collection outcome — Phase 2's whole
  lesson was that those two can come apart.
- **Whether the Yellow-bin weakness is fixed.** It was out of scope and was not built.

**Caveats that apply to everything above:**

- **Nothing was tuned.** Every constant came from `config/default.yaml` as committed, before any
  Phase 3 run, including the 14-day warm-up that cost the ML model its festival exposure.
- **Every result that depends on demand varying holds on our invented demand only.**
- **The ML figures are same-machine reproducible only** (numpy 2.4.6 / scikit-learn 1.9.1).

## Verification (Windows cmd, from the repo root)

```
.venv\Scripts\activate
python -m pip install -e ".[dev,ml]"

python -m ruff check .
python -m ruff format --check .
python -m mypy

python -m pytest -m unit
python -m pytest -m sim
python -m pytest -m ml
python -m pytest -m routing
python -m pytest -m slow
```

**387 tests pass** (203 `unit`, 125 `sim`, 14 `ml`, 49 `routing`, 5 `slow`; a few carry two
markers, so the chunk counts sum to more than 387). `ruff check`, `ruff format --check` and
`mypy --strict` are all clean.

### The unit / sim / routing / slow chunks must not need scikit-learn

scikit-learn is imported **lazily**, inside the functions that fit a model. Verified the real
way — by removing it and running all four chunks:

```
python -m pip uninstall -y scikit-learn numpy
python -m pytest -m unit
python -m pytest -m sim
python -m pytest -m routing
python -m pytest -m slow
python -m pip install -e ".[dev,ml]"
```

**All four were green with scikit-learn absent.** The demand model, the three naive forecast
baselines, the rule-based anomaly detector and the moving-average forecast-driven strategies are
all pure stdlib; only `f_gbm_*` and the Isolation Forest need the extra, and every test touching
them carries the `ml` marker.

### Determinism

```
python scripts\run_phase3_benchmark.py --seed 42 --quick > results\p3a.txt
python scripts\run_phase3_benchmark.py --seed 42 --quick > results\p3b.txt
fc results\p3a.txt results\p3b.txt
```

**Verified:** two `--quick` runs at seed 42 produced **byte-identical** output — **14,697 bytes**,
compared with `cmp`. That includes the gradient-boosting rows, so the whole Phase 3 pipeline is
reproducible on this machine. No timestamp, wall-clock timing or library version is printed to
stdout, which is what makes byte-identical output possible; runtimes are recorded in this
document only.

**Measured runtimes, and why the full sweep was not run.** A clean `--quick` run (1000 bins, 2
seeds, 1 fault fraction, both scenarios) took **218 s (3 min 38 s)**. The two determinism runs
above took 543 s and 671 s, but those are **not** clean measurements — they ran alongside other
work on this machine and scikit-learn is multi-threaded, so they reflect CPU contention rather
than the algorithms.

Scaling the clean 218 s by the full default sweep (5× the bins, 2.5× the seeds, 3× the fault
fractions = 37.5×) gives **about 2.3 hours**, and that is a **lower bound**: the 2-opt TSP pass
is superlinear in the number of bins selected per cluster per day, so a 5× larger city costs more
than 5× per day. This is the figure that should have been in the plan instead of 35–50 minutes.

### Phase 2 reproducibility gate

Phase 3 changed `city_sim.py` and `strategies.py`, so "Phase 2 still reproduces" needed checking
rather than asserting. The full Phase 2 benchmark was **re-run from the Phase 3 tree**:

```
python scripts\run_benchmark.py --seed 42
```

**Verified:** every one of the **63 data rows** the fresh run produced appears **verbatim** in
`docs/PHASE2_RESULTS.md`. (The published document contains 65 distinct data rows; the two extras
are from its hand-assembled comparison tables, not from the benchmark's own output.) So the
Phase 2 results document is still true after Phase 3.

Two mechanisms keep it that way, and they are belt and braces:

- **Phase 2's entry points were not widened.** `scripts\run_benchmark.py`,
  `smartwaste.benchmark.run_city` and `build_strategies` are untouched and still return exactly
  five strategies; Phase 3 added a parallel `run_phase3_benchmark.py` rather than extending them.
  `build_layout`'s new `scenario` argument is keyword-only and defaults to `"smooth"`.
- **A test pins the numbers.**
  `tests/test_demand.py::test_smooth_scenario_reproduces_the_phase2_numbers_exactly` holds Phase
  2's per-strategy totals as literals captured from the pre-Phase-3 tree, so a regression fails a
  test in seconds rather than silently invalidating a published document.

## NOT VERIFIED

- **The entire Phase 3 demand model is invented and uncited.** Area types, the weekday cycle and
  festival spikes have no field data behind them. The paper states only that daily increments are
  "random". **Every Phase 3 result that depends on demand varying holds on this invented demand
  only** and is not evidence about a real city.
- **Every `demand`, `forecast` and `anomaly` constant is invented and uncited** — see the second
  banner in `docs/ASSUMPTIONS.md`. The absolute numbers carry no external meaning; only the
  comparisons between methods are informative.
- **The gradient-boosting model trained on zero festival days** (measured). It is scored on days
  containing a phenomenon it never saw. No constant was changed to fix this.
- **The forecaster predicts its own sensors, not the truth.** Targets are the future *fused noisy
  estimate*, because a forecaster on real hardware has no ground truth to regress on. A persistent
  sensor bias is therefore **learned rather than corrected**, and the reported MAE/RMSE measure
  agreement with the sensors, not with reality.
- **ML numbers are reproducible on this machine only.** scikit-learn promises no bit-identical
  floats across operating systems, BLAS builds or library versions. These rows come from **numpy
  2.4.6 / scikit-learn 1.9.1 on Windows, Python 3.11**. No test pins an exact float from a model;
  the ML tests assert properties instead. The stdlib parts keep the project's cross-machine
  byte-identical guarantee.
- **scikit-learn was substituted for xgboost.** SPEC §5 says "XGBoost first"; Phase 3 uses
  `HistGradientBoostingRegressor` — same model family, one fewer heavy dependency. A deviation
  from the spec, recorded rather than quietly made.
- **No hyperparameter search was performed for either model.** Deliberate, so that a tuned ML
  model is not compared against an untuned baseline — but it means the ML results are a floor,
  not the best these models could do.
- **Yellow-bin opportunistic collection was NOT built**, though SPEC §7 lists it under Phase 3.
  Out of scope for this phase; the paper's known weakness survives into Phase 4.
- **The anomaly window label is a modelling choice**, not a fact: a window counts as faulty when
  its **last day** falls inside the planted episode. "Any day overlaps" and "every day overlaps"
  are equally defensible and would move the numbers.
- **The anomaly dataset uses a plain saw-tooth fill process**, not the Phase 3 demand model. It is
  a dataset about sensor behaviour, so the fill process is deliberately simple — which also means
  the detectors have never been tested against festival spikes or area heterogeneity.
- **All data is synthetic. Only 1 physical bin was built** (in the paper's hardware work), and
  **zero readings from it** appear anywhere in this document. The 5000-bin results are simulation.
- **Euclidean distance on a uniform-random 100 × 100 km grid**, unchanged from Phase 2. Not road
  distances, not a real city. No truck capacity, no VRP, no time windows, no shift limits.
- **Overflow has no cost attached.** It is reported as a count and a rate beside cost, never
  folded into it, because we have no defensible price for an overflowing bin.
- **Five seeds is a range, not a confidence interval**, and is never described as one.
- **GitHub Actions CI has never executed.** `git push` is forbidden by CLAUDE.md, so the workflow
  — including the new conditional `ml`-extra install — is committed but **unproven**. Every check
  above was run locally on Python 3.11 / Windows.
