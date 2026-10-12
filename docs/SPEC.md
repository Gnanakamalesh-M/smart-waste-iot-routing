# SPEC — Smart Waste Collection: IoT + AI + Route Optimization

Extension of the IEEE conference paper:
"Sensor-Fusion Driven Smart Waste Collection Optimization for Metropolitan Cities:
A Hardware-Integrated and Economic Impact Analysis Framework"
(Dharun Kumar V, Gnanakamalesh M, Abinash SJ, et al., Sri Sairam Institute of Technology).

Goal: a GitHub portfolio project + interview material. Honest, reproducible, explainable.

---

## 1. Problem
Fixed-route trucks visit every bin daily, regardless of fill level. This wastes fuel, money,
labor, and CO2. The system collects only bins that actually need it, using optimized routes.

## 2. What the paper did (the baseline we extend)
**Hardware (1 physical bin only):** 3 ultrasonic sensors (left/center/right) on the upper inner
surface + 1 load cell at the bottom, on a NodeMCU, sending fill % and weight over Wi-Fi to
ThingSpeak. Inner side walls angled to push waste to the center.

**Sensor fusion (paper equations):**
- d_bar = (d1 + d2 + d3) / 3
- phi = 1 - d_bar / H ; Fv = 100 * phi, clipped to [0, 100]
- W_th = alpha * W_max, 0 < alpha < 1 (alpha value NOT given in paper; we choose and document it)
- Iw = 1 if W >= W_th else 0
- State: RED if Iw == 1 or Fv >= 80 ; YELLOW if Iw == 0 and 60 <= Fv < 80 ; GREEN otherwise

**Routing:** city split into geographic clusters; each truck starts/ends at a depot and visits
only RED bins in its cluster, solved as depot-constrained TSP (Euclidean distance).

**Simulation:** 5000 bins on a 2D grid (100 x 100 km in the figure), 30 days,
W_i(t) = W_i(t-1) + dW_i(t) with random daily increments. Baseline = fixed route over all
5000 bins daily. Fuel F = a*D, CO2 E = b*D, cost = fuel + maintenance + labor + fixed.

**Paper's reported results (reference only, we do NOT promise to match them):**
travel time 16,192 -> 10,680 hr (34.04%), cost 21,144,450 -> 16,446,330 (22.22%),
CO2 571,259 -> 389,294 kg (31.85%). Red bins about 19-26% per day.

**Known weaknesses of the paper (we address these, and say so honestly):**
- Only 1 physical bin; 5000-bin results are simulation.
- Euclidean distance, random grid, not real roads.
- Baseline is naive (visits everything), so part of the gain is just skipping bins.
- Yellow bins are never collected (overflow risk).
- Paper inconsistencies: 30 days vs "three-month"; "34.04% distance" vs Table I travel time;
  Eq.13 typo (F=bD should be E=bD); "1926 percent" typo for 19-26%.
- Many simulation constants (alpha, fuel rate, wages, speeds, truck capacity) not stated.
  We choose realistic values, put them in ONE config file, and document each as an assumption.

---

## 3. What we build (extensions)
1. Reproducible simulation of the paper (5000 bins, 30 days) with a single config file.
2. Fair baselines: (a) fixed-route all bins, (b) fixed schedule every N days,
   (c) single-ultrasonic threshold only, (d) paper's sensor-fusion rule, (e) fusion + ML forecast.
3. Sensor noise model (noisy/biased/stuck ultrasonic sensors, uneven fill) so fusion's value is measurable.
4. ML fill-level forecasting per bin (XGBoost first, LSTM optional). Time-based split. Compare against
   naive baselines (last value, moving average).
5. Sensor anomaly detection (Isolation Forest or similar) on SYNTHETIC data with PLANTED faults;
   the detector never reads the truth file; score with precision/recall.
6. Yellow-bin handling: predict time-to-Red; pick up Yellow bins opportunistically if on the route.
7. Routing: clustering (KMeans) -> TSP (nearest neighbor, 2-opt, OR-Tools) -> VRP with truck
   capacity and multiple trucks (OR-Tools). Compare algorithms by distance and runtime.
8. Real road network for a Chennai area via OSMnx (graph distances instead of Euclidean).
   Cache the downloaded graph to disk. If network is unavailable, fall back to Euclidean and say so.
9. Streamlit dashboard: bin map (G/Y/R), daily routes, KPI charts, baseline comparison.
10. NodeMCU firmware (Arduino C++) + Python script to pull data from ThingSpeak.
    Firmware is NOT verified on hardware by Claude Code. Mark it NOT VERIFIED unless the user tests it.
11. Docs: README, architecture diagram, results tables, limitations, interview prep.

## 4. Non-goals (do not build)
Blockchain, mobile app, waste-type image classification, anything needing paid APIs.

## 5. Tech stack
Python 3.11; numpy, pandas, scikit-learn, xgboost, ortools, networkx, osmnx, matplotlib/plotly,
streamlit, pytest, ruff, mypy. Optional later: torch (LSTM), Docker.

## 6. Repo layout (target)
```
smart-waste-iot-routing/
  CLAUDE.md
  README.md
  pyproject.toml
  config/default.yaml          # ALL assumptions live here
  docs/SPEC.md  docs/PHASES.md  docs/ASSUMPTIONS.md  docs/INTERVIEW.md
  src/smartwaste/
    config.py  fusion.py  simulate.py  noise.py  baselines.py
    routing/ (cluster.py tsp.py vrp.py roads.py)
    ml/ (forecast.py anomaly.py)
    metrics.py  cost.py
  scripts/  run_sim.py  run_benchmark.py  run_forecast.py  run_anomaly.py  run_dashboard.py
  firmware/nodemcu_smartbin/nodemcu_smartbin.ino
  tools/thingspeak_fetch.py
  dashboard/app.py
  tests/
  .github/workflows/ci.yml
```

## 7. Phases (one per session, commit after each)
- **Phase 0:** skeleton, pyproject, config loader, lint/type/test setup, CI (Python 3.11),
  .gitignore, ASSUMPTIONS.md stub. No features yet.
- **Phase 1:** bin simulation + sensor noise + sensor fusion classifier + tests.
  Acceptance: fusion beats single-sensor on misclassification under uneven fill (measured, reported).
- **Phase 2:** city simulation (5000 bins x 30 days), cost/fuel/CO2 model, baselines (a)-(d),
  Euclidean TSP routing with clustering. Acceptance: benchmark table produced by one command.
- **Phase 3:** ML forecasting + anomaly detection + Yellow-bin logic. Time-based split.
  Acceptance: forecast vs naive baselines reported honestly; anomaly precision/recall on planted faults.
- **Phase 4:** VRP (capacity, multi-truck), algorithm comparison, real Chennai roads (OSMnx).
- **Phase 5:** Streamlit dashboard + NodeMCU firmware + ThingSpeak fetch script.
- **Phase 6:** README, architecture diagram, tests cleanup, Docker (build and RUN it, or mark NOT VERIFIED),
  interview doc (`docs/INTERVIEW.md`: pitch, architecture, formulas, results, limitations, Q&A, quiz).

## 8. Definition of done (every phase)
- Tests pass locally on Python 3.11, in chunks.
- One documented command reproduces the phase output.
- Numbers in docs are copied from real output with the command that made them.
- Short summary: what was built, how it works, shortcuts taken, NOT VERIFIED items.
