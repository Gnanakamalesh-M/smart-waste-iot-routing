# Phase 1 results — does the paper's sensor fusion beat a single sensor?

## How these numbers were produced

```
python scripts\run_fusion_eval.py
```

- **Seed:** 42 (from `config/default.yaml`; every sub-seed is derived from it with sha256).
- **Code commit:** `edf1ec09678b918acef28c78a696e6c997ef13a1` ("Phase 1: single-bin simulation,
  sensor noise model, paper's sensor fusion"). The working tree was clean when this ran, so these
  numbers come from committed code.
- **Size:** 6 scenarios x 200 bins x 30 days = 6000 readings per method per scenario. All three
  methods are scored on the *same* readings (paired comparison).
- **Environment:** Python 3.11, Windows, stdlib only.

Every table below is pasted from that command's real output. Re-running it regenerates
`results/fusion_eval.{csv,md}`, `results/fusion_eval_dcv0.{csv,md}` and `results/fusion_sweep.csv`
(all gitignored).

## What these numbers mean — and do not mean

**They measure how faithfully each sensor configuration reproduces the paper's own
RED/YELLOW/GREEN rule**, where ground truth is that same rule applied to the true fill and the
true weight. They are **not** collection outcomes. Phase 1 routes no truck, counts no overflow and
computes no cost. A method that reproduces the rule perfectly also inherits every flaw of the rule
— notably that YELLOW bins are never collected at all. Routing and overflow consequences are
Phase 2.

The three methods:

| method | distances used | load cell |
|---|---|---|
| (a) single center sensor | centre only | no |
| (b) 3-sensor mean | mean of 3 | no |
| (c) 3-sensor mean + load cell | mean of 3 | yes — the paper's method |

(a) and (b) have no load cell, so they can never raise the overload flag `Iw`. Every scenario is
labelled **control**, **fair test**, or **favors fusion by construction** so a measurement can be
told apart from a demonstration. No noise level was tuned; all come straight from
`config/default.yaml`.

## Headline table (density_cv = 0.3)

### Headline: misclassification by noise scenario (density_cv=0.3)

| scenario | fairness | method | n | misclass % | misclass 95% CI | false-RED % | n(not RED) | false-GREEN % | false-GREEN 95% CI | n(RED) |
|---|---|---|---|---|---|---|---|---|---|---|
| clean | control | (a) single center sensor | 6000 | 2.32 | [1.88, 2.78] | 0.00 | 5496 | 0.00 | [0.00, 0.00] | 504 |
| clean | control | (b) 3-sensor mean | 6000 | 2.32 | [1.88, 2.78] | 0.00 | 5496 | 0.00 | [0.00, 0.00] | 504 |
| clean | control | (c) 3-sensor mean + load cell | 6000 | 0.00 | [0.00, 0.00] | 0.00 | 5496 | 0.00 | [0.00, 0.00] | 504 |
| noisy | fair test | (a) single center sensor | 6000 | 5.02 | [4.48, 5.55] | 0.54 | 5511 | 0.20 | [0.00, 0.62] | 489 |
| noisy | fair test | (b) 3-sensor mean | 6000 | 3.65 | [3.18, 4.13] | 0.27 | 5511 | 0.00 | [0.00, 0.00] | 489 |
| noisy | fair test | (c) 3-sensor mean + load cell | 6000 | 2.03 | [1.63, 2.43] | 0.45 | 5511 | 0.00 | [0.00, 0.00] | 489 |
| uneven_random_tilt | fair test | (a) single center sensor | 6000 | 5.82 | [5.28, 6.35] | 0.78 | 5499 | 0.20 | [0.00, 0.63] | 501 |
| uneven_random_tilt | fair test | (b) 3-sensor mean | 6000 | 4.25 | [3.75, 4.80] | 0.44 | 5499 | 0.00 | [0.00, 0.00] | 501 |
| uneven_random_tilt | fair test | (c) 3-sensor mean + load cell | 6000 | 2.47 | [2.10, 2.85] | 0.64 | 5499 | 0.00 | [0.00, 0.00] | 501 |
| stuck_left | fair test | (a) single center sensor | 6000 | 5.88 | [5.33, 6.45] | 0.93 | 5501 | 0.00 | [0.00, 0.00] | 499 |
| stuck_left | fair test | (b) 3-sensor mean | 6000 | 26.73 | [25.70, 27.73] | 0.00 | 5501 | 69.34 | [64.04, 74.39] | 499 |
| stuck_left | fair test | (c) 3-sensor mean + load cell | 6000 | 23.18 | [21.88, 24.52] | 0.20 | 5501 | 34.07 | [28.69, 39.24] | 499 |
| uneven_center_peak | favors fusion by construction | (a) single center sensor | 6000 | 10.87 | [10.08, 11.62] | 4.05 | 5507 | 0.00 | [0.00, 0.00] | 493 |
| uneven_center_peak | favors fusion by construction | (b) 3-sensor mean | 6000 | 4.12 | [3.63, 4.62] | 0.36 | 5507 | 0.00 | [0.00, 0.00] | 493 |
| uneven_center_peak | favors fusion by construction | (c) 3-sensor mean + load cell | 6000 | 2.23 | [1.90, 2.62] | 0.45 | 5507 | 0.00 | [0.00, 0.00] | 493 |
| stuck_center | favors fusion by construction | (a) single center sensor | 6000 | 26.27 | [25.30, 27.27] | 0.00 | 5496 | 100.00 | [100.00, 100.00] | 504 |
| stuck_center | favors fusion by construction | (b) 3-sensor mean | 6000 | 26.27 | [25.30, 27.23] | 0.00 | 5496 | 70.04 | [65.26, 74.75] | 504 |
| stuck_center | favors fusion by construction | (c) 3-sensor mean + load cell | 6000 | 22.63 | [21.25, 23.90] | 0.11 | 5496 | 32.74 | [28.12, 37.76] | 504 |

Verdict lines, verbatim from the same run:

```
  clean [control]: (b) mean3 ties (a); (c) mean3+load cell beats (a) by 2.32 pp;
  noisy [fair test]: (b) mean3 beats (a) by 1.37 pp; (c) mean3+load cell beats (a) by 2.98 pp;
  uneven_random_tilt [fair test]: (b) mean3 beats (a) by 1.57 pp; (c) mean3+load cell beats (a) by 3.35 pp;
  stuck_left [fair test]: (b) mean3 is WORSE than (a) by 20.85 pp; (c) mean3+load cell is WORSE than (a) by 17.30 pp;
  uneven_center_peak [favors fusion by construction]: (b) mean3 beats (a) by 6.75 pp; (c) mean3+load cell beats (a) by 8.63 pp;
  stuck_center [favors fusion by construction]: (b) mean3 ties (a); (c) mean3+load cell beats (a) by 3.63 pp;
```

No CI-overlap caveats appear above, so every difference listed is resolved at this sample size
(200 bins). The CIs resample **whole bins**, not readings, because readings within one bin share a
density factor and a stuck fault.

## Reading of each scenario

**clean (control) — the one surprise, and it is a real one.** With literally zero sensor noise,
(a) and (b) still get 2.32% wrong while (c) gets 0.00%. That 2.32% is not sensor error; it is the
cost of *having no load cell at all*. Those are bin-days that are RED by weight
(`W >= 0.85 * W_max`) while `Fv` is still below 80, because a dense bin is heavy before it looks
full. A method without a load cell cannot see them in principle, however perfect its distances.
This is the cleanest statement of what the load cell buys.

**noisy (fair test) — fusion wins, modestly.** The 3-sensor mean cuts misclassification from 5.02%
to 3.65% (−1.37 pp). Plain variance reduction plus the averaging-away of three independent biases;
no mechanism here is rigged in the array's favour.

**uneven_random_tilt (fair test) — fusion wins, modestly.** 5.82% → 4.25% (−1.57 pp). This is the
honest uneven-fill number: under a zero-mean tilt the centre sensor is *unbiased*, so the mean can
only win on variance. Compare it with `uneven_center_peak` below.

**stuck_left (fair test) — fusion LOSES, and badly.** The 3-sensor mean is 26.73% vs the single
sensor's 5.88% — **20.85 pp worse** — and it misses 69.34% of genuinely full bins against the
single sensor's 0.00%. Averaging in a dead sensor poisons the mean, while the centre sensor simply
does not care that the left sensor died. The load cell recovers some of it (23.18%, false-GREEN
down to 34.07%) but nowhere near enough to catch up. This is the most important result in the
table and it contradicts the paper's implicit claim that more sensors are simply better. A plain
mean has no fault tolerance; the fix is a median or an outlier-rejecting fusion rule, which the
paper does not specify and Phase 1 does not implement.

**uneven_center_peak (favors fusion by construction) — fusion wins, largest margin.** 10.87% →
4.12% (−6.75 pp). Treat this as a demonstration, not a measurement: the skew is *defined* to bias
the centre sensor (waste heaps in the middle, so the centre reads short), so the mean was always
going to win. It does show the mechanism works — and it is the SPEC §7 acceptance criterion
("fusion beats single-sensor on misclassification under uneven fill"), which is met here and also
met, more modestly, in the fair `uneven_random_tilt` scenario.

**stuck_center (favors fusion by construction) — fusion wins on safety, ties on accuracy.** Both
(a) and (b) sit at 26.27% misclassification, but the headline rate hides the thing that matters:
(a) misses **100%** of full bins, (b) misses 70.04%, and (c) misses 32.74%. Again by construction
— the fault is aimed at the only sensor (a) has.

## Sensitivity: density_cv = 0

The load cell can only help when weight carries information the distances do not. The only thing
providing that in our model is `bin.density_cv`, a per-bin waste-density spread **we chose with no
field data**. At `density_cv = 0`, weight is a fixed multiple of volume and the load cell is
redundant by construction. That run:

```
python scripts\run_fusion_eval.py --density-cv 0.0
```

### Sensitivity: density_cv=0 (weight is a fixed multiple of volume, so the load cell can add nothing)

| scenario | fairness | method | n | misclass % | misclass 95% CI | false-RED % | n(not RED) | false-GREEN % | false-GREEN 95% CI | n(RED) |
|---|---|---|---|---|---|---|---|---|---|---|
| clean | control | (a) single center sensor | 6000 | 0.00 | [0.00, 0.00] | 0.00 | 5538 | 0.00 | [0.00, 0.00] | 462 |
| clean | control | (b) 3-sensor mean | 6000 | 0.00 | [0.00, 0.00] | 0.00 | 5538 | 0.00 | [0.00, 0.00] | 462 |
| clean | control | (c) 3-sensor mean + load cell | 6000 | 0.00 | [0.00, 0.00] | 0.00 | 5538 | 0.00 | [0.00, 0.00] | 462 |
| noisy | fair test | (a) single center sensor | 6000 | 4.07 | [3.58, 4.55] | 1.17 | 5551 | 0.00 | [0.00, 0.00] | 449 |
| noisy | fair test | (b) 3-sensor mean | 6000 | 2.50 | [2.10, 2.92] | 0.63 | 5551 | 0.00 | [0.00, 0.00] | 449 |
| noisy | fair test | (c) 3-sensor mean + load cell | 6000 | 2.50 | [2.07, 2.92] | 0.63 | 5551 | 0.00 | [0.00, 0.00] | 449 |
| uneven_random_tilt | fair test | (a) single center sensor | 6000 | 4.28 | [3.77, 4.80] | 0.94 | 5539 | 0.00 | [0.00, 0.00] | 461 |
| uneven_random_tilt | fair test | (b) 3-sensor mean | 6000 | 2.30 | [1.93, 2.68] | 0.52 | 5539 | 0.00 | [0.00, 0.00] | 461 |
| uneven_random_tilt | fair test | (c) 3-sensor mean + load cell | 6000 | 2.30 | [1.92, 2.70] | 0.52 | 5539 | 0.00 | [0.00, 0.00] | 461 |
| stuck_left | fair test | (a) single center sensor | 6000 | 4.45 | [3.97, 4.95] | 1.01 | 5544 | 0.00 | [0.00, 0.00] | 456 |
| stuck_left | fair test | (b) 3-sensor mean | 6000 | 29.40 | [28.77, 30.03] | 0.00 | 5544 | 67.54 | [62.64, 72.10] | 456 |
| stuck_left | fair test | (c) 3-sensor mean + load cell | 6000 | 26.50 | [25.83, 27.12] | 0.00 | 5544 | 55.26 | [50.22, 59.73] | 456 |
| uneven_center_peak | favors fusion by construction | (a) single center sensor | 6000 | 11.12 | [10.33, 11.92] | 5.67 | 5541 | 0.00 | [0.00, 0.00] | 459 |
| uneven_center_peak | favors fusion by construction | (b) 3-sensor mean | 6000 | 2.33 | [1.97, 2.72] | 0.54 | 5541 | 0.00 | [0.00, 0.00] | 459 |
| uneven_center_peak | favors fusion by construction | (c) 3-sensor mean + load cell | 6000 | 2.33 | [1.97, 2.70] | 0.54 | 5541 | 0.00 | [0.00, 0.00] | 459 |
| stuck_center | favors fusion by construction | (a) single center sensor | 6000 | 29.60 | [28.93, 30.28] | 0.00 | 5536 | 100.00 | [100.00, 100.00] | 464 |
| stuck_center | favors fusion by construction | (b) 3-sensor mean | 6000 | 29.60 | [28.90, 30.27] | 0.00 | 5536 | 66.59 | [62.07, 71.31] | 464 |
| stuck_center | favors fusion by construction | (c) 3-sensor mean + load cell | 6000 | 26.72 | [26.00, 27.42] | 0.00 | 5536 | 55.17 | [50.43, 59.91] | 464 |

What this shows, and it is worth being precise about:

- **In `clean`, all three methods now score 0.00%.** The 2.32% penalty that (a) and (b) paid in the
  headline table was entirely an artefact of our chosen `density_cv = 0.3`. **If real bins have a
  tight density spread, the load cell's headline advantage largely disappears.** We have no data
  either way.
- **In `noisy`, `uneven_random_tilt` and `uneven_center_peak`, (b) and (c) are now identical**
  (2.50/2.50, 2.30/2.30, 2.33/2.33). Exactly as predicted: with weight a deterministic function of
  volume, the load cell contributes nothing.
- **But in the stuck-sensor scenarios (c) still beats (b)** even at `density_cv = 0` (26.50 vs
  29.40; 26.72 vs 29.60, with false-GREEN cut from ~67% to ~55%). A redundant channel is still
  worth having when the primary channel is *broken*, even when it is redundant while the primary
  channel works. That is a genuine argument for the load cell that does not depend on our density
  assumption.
- The (b)-beats-(a) margins are *larger* here (e.g. 8.78 pp in `uneven_center_peak` vs 6.75 pp),
  because the headline table's own ultrasonic-only error floor is removed.

## Diagnostic: where in the fill range do the errors live?

From `results/fusion_sweep.csv` (20000 uniform-fill samples per scenario — a diagnostic, since a
uniform fill distribution is not what a real bin does). Mean absolute `Fv` error, averaged over the
ten deciles:

| scenario | (a) center | (b) mean3 |
|---|---|---|
| clean | 0.02 | 0.02 |
| noisy | 2.14 | 1.37 |
| uneven_random_tilt | 2.16 | 1.43 |
| stuck_left | 2.11 | 11.42 |
| uneven_center_peak | 5.22 | 1.42 |
| stuck_center | 33.42 | 11.44 |

This confirms the mechanism behind each headline row: under `center_peak` the centre sensor carries
a systematic offset of about 5 percentage points that the mean removes, while under `stuck_left`
the mean's error is over 5x the single sensor's. Misclassification concentrates in the deciles
straddling the 60 and 80 thresholds — e.g. for `uneven_center_peak` in the 70-80 bucket, (a) is
wrong 50.2% of the time, (b) 30.2%, (c) 5.8%.

## Honest summary

- **Fusion is not uniformly better, and we will not claim it is.** It wins under symmetric noise
  and both uneven-fill mechanisms; it loses heavily — 20.85 pp worse — when one of the three
  sensors is stuck. A plain arithmetic mean, which is what the paper specifies, has no fault
  tolerance at all.
- **The SPEC §7 acceptance criterion is met**: under uneven fill, the 3-sensor mean beats the
  single sensor on misclassification (−6.75 pp with a centre-peaked heap, −1.57 pp under a fair
  zero-mean tilt).
- **The load cell's value splits in two.** Its headline contribution depends entirely on
  `bin.density_cv`, which we invented. Its fault-tolerance contribution (recovering a RED when the
  ultrasonic array is broken) survives even at `density_cv = 0` and is the more defensible claim.
- **Nothing was tuned.** All noise comes from `config/default.yaml` as committed;
  `tests/test_fusion_eval.py` asserts the scenarios read those values unchanged.

## NOT VERIFIED

- **The entire sensor error model is invented.** The paper publishes none, so there was nothing to
  calibrate against. Every result above is conditional on noise numbers we chose. No reading in
  this document came from physical hardware.
- **No hardware, no ThingSpeak, no firmware** is involved in Phase 1.
- **GitHub Actions CI has never executed** — `git push` is forbidden by CLAUDE.md, so the workflow
  is committed but unproven. All checks above were run locally on Python 3.11 / Windows.
  **Correction 2026-10-15: CI has since run green, see README.**
- **The 2.32% control-row figure and every density-dependent number** would change with a
  different `density_cv`; we have no field measurement of real waste-density spread.
- **`uneven_center_peak` and `stuck_center` are constructions**, not estimates of real-world
  frequency. We do not know how often a real heap is centre-peaked or how often a centre sensor
  fails.
