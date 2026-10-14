# Phase 2 results - city simulation, cost model, baselines and clustered TSP routing

## How these numbers were produced

```
python scripts\run_benchmark.py --seed 42
```

- **Seed:** 42, plus 43-46 for the across-seed range (`benchmark.n_seeds = 5`). Every sub-seed is
  derived from the master seed with sha256.
- **Code commit:** `9de1094a475c903b3258a626bf7763f5aa82aad9` ("Phase 2: city simulation, cost
  model, baselines, clustered TSP routing"). The working tree was clean when this ran
  (`git status --porcelain` printed nothing), so every number below comes from committed code.
- **Size:** 5000 bins x 30 days x 5 strategies x 3 stuck-sensor fractions x 5 seeds.
- **Environment:** Python 3.11, Windows, **stdlib only** - no numpy, no external solver.
- **Wall clock:** 468.6 s (7.8 min) for the full default run on one Windows machine.
- **Config:** every constant from `config/default.yaml` as committed. **Nothing was tuned.**

Every table below is pasted verbatim from that command's real output. Re-running it regenerates
`results/benchmark.csv` and `results/benchmark.md` (both gitignored).

## Scope of Phase 2

**In:** city simulation, overflow accounting, the cost/fuel/CO2 model, five labelled strategies,
KMeans clustering, depot-constrained TSP (nearest-neighbour then 2-opt), Euclidean distance, a
single central depot.

**Out, deliberately:** no Yellow-bin handling, no ML forecasting, no anomaly detection (Phase 3);
no truck capacity, no VRP, no multi-truck-per-cluster, no real road network (Phase 4); no
hardware, no firmware, no ThingSpeak.

## What changed since Phase 1: the shortcut is gone

Phase 1 measured **rule reproduction** and said so. Collection was triggered by **ground truth**,
so all methods saw the identical fill trajectory and a method that missed a full bin paid no
price. That made the comparison clean but removed every consequence.

**Phase 2 drives collection from each strategy's own noisy classification.** A strategy that
fails to notice a full bin does not collect it; the bin keeps filling and overflows. The
trajectories genuinely diverge, which is the only reason the overflow and missed-pickup columns
below mean anything.

Fairness is preserved where it should be: all *exogenous* randomness is a pure function of
`(seed, bin, day)` rather than of call order. Waste arrivals are pre-generated per bin per day;
each bin's sensor array is rebuilt per strategy from the same per-bin seed and read once per bin
per day **even when the strategy ignores the reading**; the fault draw has its own stream. So all
five strategies face the same city, the same arrivals and the same sensor errors, and differ only
in their decisions. `tests/test_city_sim.py` pins that invariant.

## The five strategies

| id | label | sensors | rule |
|---|---|---|---|
| a | fixed-route, all bins daily | none | the paper's baseline |
| b | fixed schedule, every 3 days | none | sensor-free fair baseline |
| c | single-ultrasonic threshold | centre only, no load cell | collect RED |
| d | **the paper's sensor fusion** | 3-sensor MEAN + load cell | collect RED |
| e | median fusion - **extension, not in the paper** | 3-sensor MEDIAN + load cell | collect RED |

(e) is ours. Phase 1 found the paper's plain mean was 20.85 pp *worse* than a single sensor when
one sensor was stuck, because a mean cannot discard a dead reading. A median of three survives one
bad sensor. Whether that helps in collection outcomes is the question the fault sweep answers -
and the answer is **not a clean win**.

## Headline tables, seed 42

Three fault fractions, because `sensors.stuck_bin_fraction` is a number we invented and the
(d)-vs-(e) verdict **reverses** across it.

### No stuck sensors (`stuck_bin_fraction = 0`)

#### Headline: all strategies, seed 42, stuck-sensor fraction 0

| strategy | distance km | time hr | CO2 kg | cost (variable) | cost (rolling fleet) | cost (fixed fleet) | bin-visits | missed pickups | overflow events | overflow bin-days | bins overflowed >=1x |
|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) fixed-route: all bins daily [paper's baseline] | 199756.77 | 15490.27 | 187371.85 | 11028215.67 | 12228215.67 | 12228215.67 | 150000 | 0 | 0 | 0 | 0 |
| (b) fixed schedule: all bins every 3 days | 66585.59 | 5163.42 | 62457.28 | 3676071.89 | 4076071.89 | 4876071.89 | 50000 | 0 | 0 | 0 | 0 |
| (c) single-ultrasonic threshold: center sensor only, no load cell | 74746.44 | 3658.46 | 70112.16 | 3741813.02 | 4751813.02 | 4941813.02 | 13372 | 2478 | 8 | 9 | 8 |
| (d) paper's sensor fusion: 3-sensor MEAN + load cell [the paper's method] | 71842.09 | 3500.53 | 67387.88 | 3593582.24 | 4593582.24 | 4793582.24 | 12537 | 681 | 13 | 13 | 13 |
| (e) median fusion: 3-sensor MEDIAN + load cell [extension, not in the paper] | 71452.17 | 3476.89 | 67022.14 | 3573241.85 | 4575241.85 | 4773241.85 | 12376 | 1693 | 29 | 29 | 28 |

#### Reduction vs (a) fixed-route, seed 42, stuck-sensor fraction 0

Positive = better than the baseline. Negative values are printed as-is: a strategy that
costs more than the baseline must look like it costs more. Each cost column names the
fleet basis it used.

| strategy | distance % | travel time % | CO2 % | cost % (variable basis) | cost % (rolling-fleet basis) | cost % (fixed-fleet basis) |
|---|---|---|---|---|---|---|
| (b) fixed schedule: all bins every 3 days | 66.67 | 66.67 | 66.67 | 66.67 | 66.67 | 60.12 |
| (c) single-ultrasonic threshold: center sensor only, no load cell | 62.58 | 76.38 | 62.58 | 66.07 | 61.14 | 59.59 |
| (d) paper's sensor fusion: 3-sensor MEAN + load cell [the paper's method] | 64.04 | 77.40 | 64.04 | 67.41 | 62.43 | 60.80 |
| (e) median fusion: 3-sensor MEDIAN + load cell [extension, not in the paper] | 64.23 | 77.55 | 64.23 | 67.60 | 62.58 | 60.97 |

### The config value (`stuck_bin_fraction = 0.02`)

#### Headline: all strategies, seed 42, stuck-sensor fraction 0.02

| strategy | distance km | time hr | CO2 kg | cost (variable) | cost (rolling fleet) | cost (fixed fleet) | bin-visits | missed pickups | overflow events | overflow bin-days | bins overflowed >=1x |
|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) fixed-route: all bins daily [paper's baseline] | 199756.77 | 15490.27 | 187371.85 | 11028215.67 | 12228215.67 | 12228215.67 | 150000 | 0 | 0 | 0 | 0 |
| (b) fixed schedule: all bins every 3 days | 66585.59 | 5163.42 | 62457.28 | 3676071.89 | 4076071.89 | 4876071.89 | 50000 | 0 | 0 | 0 | 0 |
| (c) single-ultrasonic threshold: center sensor only, no load cell | 74593.03 | 3648.42 | 69968.26 | 3733678.14 | 4743678.14 | 4933678.14 | 13294 | 3077 | 38 | 538 | 38 |
| (d) paper's sensor fusion: 3-sensor MEAN + load cell [the paper's method] | 71834.52 | 3498.03 | 67380.78 | 3592819.29 | 4592819.29 | 4792819.29 | 12493 | 942 | 90 | 146 | 51 |
| (e) median fusion: 3-sensor MEDIAN + load cell [extension, not in the paper] | 71422.03 | 3475.58 | 66993.87 | 3571763.57 | 4573763.57 | 4771763.57 | 12374 | 1710 | 31 | 31 | 30 |

#### Reduction vs (a) fixed-route, seed 42, stuck-sensor fraction 0.02

Positive = better than the baseline. Negative values are printed as-is: a strategy that
costs more than the baseline must look like it costs more. Each cost column names the
fleet basis it used.

| strategy | distance % | travel time % | CO2 % | cost % (variable basis) | cost % (rolling-fleet basis) | cost % (fixed-fleet basis) |
|---|---|---|---|---|---|---|
| (b) fixed schedule: all bins every 3 days | 66.67 | 66.67 | 66.67 | 66.67 | 66.67 | 60.12 |
| (c) single-ultrasonic threshold: center sensor only, no load cell | 62.66 | 76.45 | 62.66 | 66.14 | 61.21 | 59.65 |
| (d) paper's sensor fusion: 3-sensor MEAN + load cell [the paper's method] | 64.04 | 77.42 | 64.04 | 67.42 | 62.44 | 60.81 |
| (e) median fusion: 3-sensor MEDIAN + load cell [extension, not in the paper] | 64.25 | 77.56 | 64.25 | 67.61 | 62.60 | 60.98 |

### Stress (`stuck_bin_fraction = 0.10`)

#### Headline: all strategies, seed 42, stuck-sensor fraction 0.1

| strategy | distance km | time hr | CO2 kg | cost (variable) | cost (rolling fleet) | cost (fixed fleet) | bin-visits | missed pickups | overflow events | overflow bin-days | bins overflowed >=1x |
|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) fixed-route: all bins daily [paper's baseline] | 199756.77 | 15490.27 | 187371.85 | 11028215.67 | 12228215.67 | 12228215.67 | 150000 | 0 | 0 | 0 | 0 |
| (b) fixed schedule: all bins every 3 days | 66585.59 | 5163.42 | 62457.28 | 3676071.89 | 4076071.89 | 4876071.89 | 50000 | 0 | 0 | 0 | 0 |
| (c) single-ultrasonic threshold: center sensor only, no load cell | 74032.89 | 3607.97 | 69442.85 | 3703290.64 | 4711290.64 | 4903290.64 | 12933 | 5808 | 171 | 2931 | 171 |
| (d) paper's sensor fusion: 3-sensor MEAN + load cell [the paper's method] | 71420.46 | 3473.12 | 66992.39 | 3571255.30 | 4571255.30 | 4771255.30 | 12326 | 1883 | 367 | 635 | 189 |
| (e) median fusion: 3-sensor MEDIAN + load cell [extension, not in the paper] | 71412.21 | 3474.49 | 66984.65 | 3571161.50 | 4573161.50 | 4771161.50 | 12360 | 1813 | 37 | 37 | 34 |

#### Reduction vs (a) fixed-route, seed 42, stuck-sensor fraction 0.1

Positive = better than the baseline. Negative values are printed as-is: a strategy that
costs more than the baseline must look like it costs more. Each cost column names the
fleet basis it used.

| strategy | distance % | travel time % | CO2 % | cost % (variable basis) | cost % (rolling-fleet basis) | cost % (fixed-fleet basis) |
|---|---|---|---|---|---|---|
| (b) fixed schedule: all bins every 3 days | 66.67 | 66.67 | 66.67 | 66.67 | 66.67 | 60.12 |
| (c) single-ultrasonic threshold: center sensor only, no load cell | 62.94 | 76.71 | 62.94 | 66.42 | 61.47 | 59.90 |
| (d) paper's sensor fusion: 3-sensor MEAN + load cell [the paper's method] | 64.25 | 77.58 | 64.25 | 67.62 | 62.62 | 60.98 |
| (e) median fusion: 3-sensor MEDIAN + load cell [extension, not in the paper] | 64.25 | 77.57 | 64.25 | 67.62 | 62.60 | 60.98 |

## Range over 5 seeds

**These are ranges over seeds 42-46, not confidence intervals.** Five seeds describe the spread we
observed; they do not support an inference about a population. Each seed regenerates the layout,
the densities, the fault draw and every waste arrival.

#### Range over 5 seeds (42-46), stuck-sensor fraction 0

**This is a range over seeds, not a confidence interval.** 5 seeds describe
the spread we observed; they do not support an inference about a population.

| strategy | distance km: mean [min, max] | cost (variable): mean [min, max] | overflow events: mean [min, max] | bins overflowed: mean [min, max] |
|---|---|---|---|---|
| (a) fixed-route: all bins daily [paper's baseline] | 199960.98 [199078.57, 201599.77] | 11038109.64 [10995356.92, 11117508.64] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| (b) fixed schedule: all bins every 3 days | 66653.66 [66359.52, 67199.92] | 3679369.88 [3665118.97, 3705836.21] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| (c) single-ultrasonic threshold: center sensor only, no load cell | 74866.40 [73954.69, 75748.27] | 3747565.75 [3702894.60, 3790324.58] | 6.40 [4.00, 8.00] | 6.40 [4.00, 8.00] |
| (d) paper's sensor fusion: 3-sensor MEAN + load cell [the paper's method] | 71906.64 [71593.71, 72536.66] | 3596634.07 [3581710.42, 3626829.36] | 8.40 [6.00, 13.00] | 8.40 [6.00, 13.00] |
| (e) median fusion: 3-sensor MEDIAN + load cell [extension, not in the paper] | 71432.46 [70884.01, 72064.70] | 3572186.06 [3545669.07, 3602522.79] | 29.00 [24.00, 35.00] | 28.00 [23.00, 34.00] |

#### Range over 5 seeds (42-46), stuck-sensor fraction 0.02

**This is a range over seeds, not a confidence interval.** 5 seeds describe
the spread we observed; they do not support an inference about a population.

| strategy | distance km: mean [min, max] | cost (variable): mean [min, max] | overflow events: mean [min, max] | bins overflowed: mean [min, max] |
|---|---|---|---|---|
| (a) fixed-route: all bins daily [paper's baseline] | 199960.98 [199078.57, 201599.77] | 11038109.64 [10995356.92, 11117508.64] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| (b) fixed schedule: all bins every 3 days | 66653.66 [66359.52, 67199.92] | 3679369.88 [3665118.97, 3705836.21] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| (c) single-ultrasonic threshold: center sensor only, no load cell | 74707.12 [73783.23, 75610.04] | 3739022.40 [3693606.26, 3782889.49] | 41.00 [35.00, 49.00] | 41.00 [35.00, 49.00] |
| (d) paper's sensor fusion: 3-sensor MEAN + load cell [the paper's method] | 71810.76 [71448.39, 72452.04] | 3591590.83 [3574246.50, 3622297.43] | 79.20 [62.00, 90.00] | 45.00 [35.00, 51.00] |
| (e) median fusion: 3-sensor MEDIAN + load cell [extension, not in the paper] | 71425.62 [70906.84, 72044.83] | 3571818.54 [3546748.54, 3601514.79] | 30.80 [24.00, 38.00] | 29.80 [23.00, 37.00] |

#### Range over 5 seeds (42-46), stuck-sensor fraction 0.1

**This is a range over seeds, not a confidence interval.** 5 seeds describe
the spread we observed; they do not support an inference about a population.

| strategy | distance km: mean [min, max] | cost (variable): mean [min, max] | overflow events: mean [min, max] | bins overflowed: mean [min, max] |
|---|---|---|---|---|
| (a) fixed-route: all bins daily [paper's baseline] | 199960.98 [199078.57, 201599.77] | 11038109.64 [10995356.92, 11117508.64] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| (b) fixed schedule: all bins every 3 days | 66653.66 [66359.52, 67199.92] | 3679369.88 [3665118.97, 3705836.21] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| (c) single-ultrasonic threshold: center sensor only, no load cell | 74140.09 [73321.81, 75002.56] | 3708450.14 [3667803.60, 3750613.03] | 170.40 [156.00, 198.00] | 170.40 [156.00, 198.00] |
| (d) paper's sensor fusion: 3-sensor MEAN + load cell [the paper's method] | 71379.89 [70937.59, 71967.70] | 3569197.65 [3547824.21, 3597391.08] | 355.40 [330.00, 374.00] | 186.60 [172.00, 198.00] |
| (e) median fusion: 3-sensor MEDIAN + load cell [extension, not in the paper] | 71368.27 [70903.90, 71927.67] | 3568906.88 [3546506.91, 3595730.85] | 38.00 [26.00, 49.00] | 36.60 [25.00, 48.00] |

## (d) the paper's MEAN fusion vs (e) our MEDIAN fusion

#### Paired per-seed comparison: (d) paper's MEAN fusion vs (e) our MEDIAN fusion

Paired: both strategies ran on the same city, the same waste arrivals and the same
sensor noise, so these differences are not confounded by the draw.

**A positive `(d)-(e)` value means (e) the median was better on that metric.**
Negative means the paper's mean was better and the extension did not pay off there.

| fault fraction | seed | (d) overflow events | (e) overflow events | (d)-(e) overflow | (d) missed | (e) missed | (d)-(e) missed | (d)-(e) cost (variable) |
|---|---|---|---|---|---|---|---|---|
| 0 | 42 | 13 | 29 | -16 | 681 | 1693 | -1012 | 20340.39 |
| 0 | 43 | 8 | 35 | -27 | 698 | 1633 | -935 | 24306.57 |
| 0 | 44 | 6 | 24 | -18 | 714 | 1603 | -889 | 17833.94 |
| 0 | 45 | 9 | 33 | -24 | 661 | 1566 | -905 | 23717.81 |
| 0 | 46 | 6 | 24 | -18 | 723 | 1645 | -922 | 36041.35 |
| 0.02 | 42 | 90 | 31 | 59 | 942 | 1710 | -768 | 21055.72 |
| 0.02 | 43 | 77 | 38 | 39 | 947 | 1670 | -723 | 20782.64 |
| 0.02 | 44 | 62 | 26 | 36 | 925 | 1627 | -702 | 14052.34 |
| 0.02 | 45 | 89 | 35 | 54 | 921 | 1589 | -668 | 15472.80 |
| 0.02 | 46 | 78 | 24 | 54 | 978 | 1664 | -686 | 27497.96 |
| 0.1 | 42 | 367 | 37 | 330 | 1883 | 1813 | 70 | 93.80 |
| 0.1 | 43 | 338 | 49 | 289 | 1853 | 1770 | 83 | 1660.23 |
| 0.1 | 44 | 330 | 36 | 294 | 1843 | 1724 | 119 | 1242.81 |
| 0.1 | 45 | 374 | 42 | 332 | 1872 | 1678 | 194 | -2860.27 |
| 0.1 | 46 | 368 | 26 | 342 | 1972 | 1762 | 210 | 1317.29 |

Verdict lines, verbatim from the same run:

```
Verdict: (d) the paper's mean fusion vs (e) our median fusion
  stuck fraction 0: (e) median is WORSE than (d) mean on overflow events (29.0 vs 8.4 mean over seeds) -- the extension does not pay off; missed pickups 1628.0 vs 695.4; (e) won on 0/5 seeds
  stuck fraction 0.02: (e) median BEATS (d) mean on overflow events (30.8 vs 79.2 mean over seeds); missed pickups 1652.0 vs 942.6; (e) won on 5/5 seeds
  stuck fraction 0.1: (e) median BEATS (d) mean on overflow events (38.0 vs 355.4 mean over seeds); missed pickups 1749.4 vs 1884.6; (e) won on 5/5 seeds
```

### Reading this honestly

**The extension is not a clean win, and the direction depends on a number we invented.**

- **At zero faults, (e) LOSES, and clearly.** 29.0 overflow events against (d)'s 8.4 - about 3.5x
  worse - and it won on **0 of 5 seeds**. This is exactly what should happen: with three healthy
  sensors a median of 3 is a *noisier* estimator of the surface height than a mean, because it
  throws away two readings' worth of averaging. We predicted this before running it and it is why
  (e) is labelled an extension rather than an improvement.
- **At 0.02 and 0.10, (e) WINS, and by a lot.** 2.6x fewer overflow events at 0.02 and 9.4x fewer
  at 0.10, winning on **5 of 5 seeds** at both. One hard-latched sensor can drag the mean
  permanently below the RED threshold, so (d) never collects that bin at all; the median ignores
  the dead sensor.
- **(e) misses MORE RED bin-days than (d) at every fault fraction**, including the ones where it
  wins (1652 vs 943 at 0.02). These two facts are consistent and both matter: the median is less
  accurate day to day, but it cannot be held below the threshold indefinitely by one dead sensor.
  Day-to-day misses get caught the next day; a stuck mean does not.
- **Which failure you care about decides the winner.** If overflow is the thing you are trying to
  avoid, the median is better as soon as any meaningful fraction of sensors fail. If you are
  optimising day-to-day pickup accuracy, the mean is better. We have **no field data on real
  sensor failure rates**, so we are not claiming either rate is realistic.

## Where a simpler baseline matches or beats the paper's method

This is the most important result in Phase 2.

```
Verdict: does a simpler baseline match or beat the paper's method?
  stuck fraction 0: (b) fixed schedule: all bins every 3 days overflows no more than the paper's method but costs more (cost 3679370 vs 3596634 variable basis; overflow events 0.0 vs 8.4)
  stuck fraction 0: (c) single-ultrasonic threshold: center sensor only, no load cell overflows no more than the paper's method but costs more (cost 3747566 vs 3596634 variable basis; overflow events 6.4 vs 8.4)
  stuck fraction 0.02: (b) fixed schedule: all bins every 3 days overflows no more than the paper's method but costs more (cost 3679370 vs 3591591 variable basis; overflow events 0.0 vs 79.2)
  stuck fraction 0.02: (c) single-ultrasonic threshold: center sensor only, no load cell overflows no more than the paper's method but costs more (cost 3739022 vs 3591591 variable basis; overflow events 41.0 vs 79.2)
  stuck fraction 0.1: (b) fixed schedule: all bins every 3 days overflows no more than the paper's method but costs more (cost 3679370 vs 3569198 variable basis; overflow events 0.0 vs 355.4)
  stuck fraction 0.1: (c) single-ultrasonic threshold: center sensor only, no load cell overflows no more than the paper's method but costs more (cost 3708450 vs 3569198 variable basis; overflow events 170.4 vs 355.4)
```

**Strategy (b) - every bin every 3 days, with no sensors at all - gets within 0.75 percentage
points of the paper's sensor fusion on cost, with ZERO overflow events at every fault fraction.**

At `stuck_bin_fraction = 0.02`, on the variable-cost basis:

| | (b) fixed 3-day schedule | (d) paper's fusion |
|---|---|---|
| cost reduction vs (a) | **66.67%** | **67.42%** |
| variable cost | 3,676,071.89 | 3,592,819.29 |
| overflow events | **0** | **90** |
| bins overflowed at least once | **0** | **51** |
| missed pickups | **0** | 942 |
| bin-visits | 50,000 | 12,493 |

(b) costs 2.32% more than (d) and overflows **zero** times against (d)'s 90 events. (c), the
single ultrasonic sensor with no load cell, costs 3.92% more than (d) and also overflows less (38
events vs 90).

**What that means, stated plainly:** on this synthetic arrival process, the entire sensor
apparatus buys a ~0.75 pp cost improvement over a dumb calendar, and buys it at the price of a
*worse* overflow record. The honest conclusion is that **Phase 2 does not demonstrate a business
case for the sensors.**

**Why this happens, and what it does not prove.** Our waste arrival process is a Gaussian with
mean 4.0 kg/day and sd 1.5 kg - bins fill at very similar, very predictable rates, which is the
best possible world for a fixed calendar. Real cities have heterogeneous, bursty, seasonal and
event-driven bins, and that heterogeneity is exactly what sensors exist to exploit. So this
result is **a finding about our assumptions**, not evidence that telemetry is worthless. Making
the arrival process heterogeneous is the obvious next experiment; it is not in Phase 2's scope and
we have not run it.

## REFERENCE ONLY: the paper's reported percentages

**This section is for reference. We do not promise to reproduce these numbers and nothing was
tuned toward them.**

The paper reports travel time 16,192 -> 10,680 hr (**34.04%**), cost 21,144,450 -> 16,446,330
(**22.22%**), CO2 571,259 -> 389,294 kg (**31.85%**).

Ours, for (d) the paper's own method against (a) its own baseline, seed 42,
`stuck_bin_fraction = 0.02`:

| metric | paper | ours (d vs a) | our basis |
|---|---|---|---|
| travel time | 34.04% | **77.42%** | - |
| cost | 22.22% | **67.42%** | variable-only |
| cost | 22.22% | **62.44%** | rolling fleet |
| cost | 22.22% | **60.81%** | fleet fixed at baseline size |
| CO2 | 31.85% | **64.04%** | - |
| distance | (34.04% is attributed to distance in one place) | **64.04%** | - |

**Our numbers differ substantially, and in the same direction on every metric: we report far
larger savings than the paper does.** We cannot reconcile the gap, and we are not going to adjust
constants until it closes. What we can say about the likely causes:

1. **The baseline dominates everything.** Both we and the paper measure against "visit all 5000
   bins every day". Strategy (d) visits 12,493 bin-days out of 150,000 - **8.3%** of the
   baseline's visits. Most of the saving is simply not driving to 92% of the bins, and that
   proportion is set by how often our arrival process pushes a bin to RED, which depends on our
   invented `daily_fill_mean_kg`, `density_kg_per_pct` and thresholds.
2. **Our grid is 100 x 100 km with uniform-random bins.** Tours on such a grid are enormously
   long, so dropping most stops shortens them dramatically. A denser, more realistic layout would
   shrink the proportional gain.
3. **Travel time falls faster than distance (77.42% vs 64.04%)** because our time model charges 3
   minutes of service per bin. Visiting 8.3% of the bins removes almost all the service time,
   while the distance saving is limited by the remaining tour geometry. The paper does not state
   whether its travel time includes per-bin service.
4. **Our cost constants are invented** (see the banner in `docs/ASSUMPTIONS.md`), so the *mix* of
   fuel, maintenance, labor and fixed cost is almost certainly different from the paper's, and
   the cost percentage is sensitive to that mix - visible in our own three bases spanning
   60.81% to 67.42% from the fleet assumption alone.
5. **CO2 % equals distance % exactly in our model** (64.04% both) because we model emissions as
   strictly linear in distance (`E = b*D` via litres per km). The paper reports different
   percentages for CO2 and distance, which it could only do with a non-linear or load-dependent
   emissions model it does not describe.

## Cost bases: why there are three numbers

`fixed_cost_per_day` is charged per truck per day, and whether an **idle** truck is charged is a
modelling choice that moves the headline. Rather than pick one quietly, every table reports all
three and every percentage names its basis:

- **variable** - fuel + maintenance + labor only. No fleet assumption at all. Used for headline %.
- **rolling fleet** - plus fixed cost for each truck-day a truck actually rolled. Flatters the
  sensor strategies, which park trucks whose cluster has no RED bin.
- **fleet fixed at baseline size** - plus fixed cost for all 20 trucks every day. You own the
  trucks either way.

For (d) at seed 42, fault 0.02, the cost reduction is **67.42% / 62.44% / 60.81%** on those three
bases. The spread of 6.6 pp is caused by nothing but that one assumption.

## Routing

- Clusters: KMeans (k-means++ seeded, stdlib) over the 5000 bin locations, computed **once per
  seed** because bins do not move - so all five strategies are routed over identical territories.
- `routing.n_clusters = 20`, one truck per cluster, single central depot at the grid centre.
- Per cluster per day: nearest-neighbour tour from the depot, then 2-opt (`two_opt_max_iters =
  1000`). 2-opt accepts only strict improvements, so it can never return a longer tour than
  nearest-neighbour; `tests/test_tsp.py` pins that, plus "every selected bin visited exactly once"
  and "route starts and ends at the depot".
- Routes are memoised on `(cluster_id, sorted selected bins)`, so (a) and (b), whose selection
  repeats, solve the full-city tour once per seed rather than 30 times. A test asserts a cached
  distance equals a fresh recompute.

## Verification (Windows cmd, from the repo root)

```
.venv\Scripts\activate

python -m ruff check .
python -m ruff format --check .
python -m mypy

python -m pytest -m unit
python -m pytest -m sim
python -m pytest -m routing
python -m pytest -m slow

python scripts\run_benchmark.py --seed 42
```

Byte-identical output at the same seed. **Verified:** two full `--seed 42` runs were
captured and compared; the 13,723 bytes of stdout were identical.

```
python scripts\run_benchmark.py --seed 42 > results\bench1.txt
python scripts\run_benchmark.py --seed 42 > results\bench2.txt
fc results\bench1.txt results\bench2.txt
```

## Honest summary

- **The paper's method works, and so does a calendar.** (d) cuts cost 67.42% against the naive
  baseline. (b), with no sensors whatsoever, cuts it 66.67% and never overflows. On our
  assumptions the sensors are not clearly worth their complexity.
- **Our savings are roughly double the paper's on every metric.** Reported, not reconciled,
  not tuned away. Most of it is the naive baseline plus a 100 km uniform-random grid.
- **Our median-fusion extension is not a clean win.** It loses to the paper's mean at zero faults
  (3.5x more overflow, 0/5 seeds) and wins at 0.02 and 0.10 (5/5 seeds). The crossover sits on an
  invented number, so all three fault fractions are reported.
- **(e) misses more RED bin-days than (d) everywhere**, even where it overflows far less. Both
  numbers are in the tables.
- **Phase 1's ground-truth shortcut is gone.** Collection now follows each strategy's own noisy
  decision, so missing a full bin really overflows.
- **Nothing was tuned.** Every constant came from `config/default.yaml` as committed, before any
  benchmark ran.

## NOT VERIFIED

- **Every cost and emissions constant is invented and uncited.** Fuel rate, diesel price, CO2 per
  litre, maintenance, wages, fixed cost, speed and service time are all guesses; the paper gives
  the form of the cost model and none of the constants. **The absolute INR and kg figures above
  carry no external meaning** - only the comparison between strategies does.
- **`bin.max_accumulated_weight_kg = 100.0` and `sensors.stuck_bin_fraction = 0.02` are invented**
  with no field data. The second one *reverses* the (d)-vs-(e) verdict, which is why the sweep
  exists.
- **5000 bins over 30 days is simulation, not measurement.** All data is synthetic. No reading in
  this document came from physical hardware. The paper built **one** physical bin; we have zero
  readings from it.
- **Euclidean distance on a uniform-random 100 x 100 km grid.** Not road distances, not a real
  city. Phase 4 adds OSMnx road graphs.
- **No truck capacity limit**, so a tour may collect more than a real truck could carry. This
  flatters strategies that visit many bins. Capacity arrives in Phase 4.
- **No time windows, no shift limits, no traffic.** Travel times in the thousands of hours per 30
  days are spread over 20 trucks with no cap on a driver's day.
- **Overflow has no cost attached.** It is reported as a count beside cost, never folded into it,
  because we have no defensible price for an overflowing bin.
- **Yellow bins are never collected**, faithfully reproducing the paper's rule and its known
  weakness. *(Correction added in Phase 3: Phase 3 did **not** address this - opportunistic
  Yellow-bin collection was outside its scope. No Phase 2 measurement above is affected.)*
- **Five seeds is a range, not a confidence interval**, and is never described as one.
- **The 7.8 minute runtime** is wall clock on one Windows machine, not a benchmark of the
  algorithms.
- **GitHub Actions CI has never executed.** `git push` is forbidden by CLAUDE.md, so the workflow
  is committed but unproven. Every check above was run locally on Python 3.11 / Windows.
- **The "sensors are not worth it" finding is conditional on our arrival process** being a
  low-variance Gaussian, which is the friendliest possible case for a fixed calendar. We have
  **not** run a heterogeneous or bursty arrival process.
