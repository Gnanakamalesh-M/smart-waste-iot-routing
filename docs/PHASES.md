# PHASES — copy-paste prompts for Claude Code

How to use: open Claude Code in the repo folder, switch to PLAN MODE (Shift+Tab), paste ONE prompt.
Read the plan. Answer its questions. Approve. When the phase is done, review, test, commit, then
start a NEW session for the next phase.

---

## Phase 0 — skeleton
```
Read CLAUDE.md and docs/SPEC.md fully. We are doing Phase 0 ONLY: project skeleton.
Create: pyproject.toml, src/smartwaste package, config/default.yaml with a config loader,
.gitignore (include .venv, .env, data caches), tests/ with one real test of the config loader,
ruff + mypy + pytest setup with markers (unit, sim, ml, routing, slow),
.github/workflows/ci.yml on Python 3.11 that runs lint, mypy, and tests in chunks,
and docs/ASSUMPTIONS.md stub. No simulation or ML yet.
Do not git push. No co-author lines. Plan first, ask me design questions, then build.
At the end: tell me exactly which commands to run on Windows cmd to verify, and what is NOT VERIFIED.
```

## Phase 1 — bin, noise, sensor fusion
```
Phase 1 ONLY (read CLAUDE.md and docs/SPEC.md sections 2, 3, 7).
Build: single-bin simulation (daily waste added, bin height H, W_max), sensor noise model
(Gaussian noise, bias, stuck sensor, uneven/lateral fill affecting each of 3 ultrasonic sensors differently),
sensor fusion exactly as the paper equations (mean of 3 distances, Fv, weight threshold alpha*Wmax,
RED/YELLOW/GREEN). Put alpha, H, W_max, noise levels in config/default.yaml and document each in ASSUMPTIONS.md.
Experiment: compare single-center-sensor classifier vs 3-sensor mean vs 3-sensor + load cell
using ground-truth state computed from the true fill. Report misclassification and false-red/false-green rates.
Script: scripts/run_fusion_eval.py (plain python). Tests for fusion edge cases (boundaries 60/80, overload, clipping).
If fusion does NOT help under our noise model, say so plainly.
```

## Phase 2 — city simulation, baselines, TSP
```
Phase 2 ONLY. Build the 5000-bin x 30-day city simulation (seeded, config-driven),
cost/fuel/CO2/time model (document every constant in ASSUMPTIONS.md), baselines:
(a) fixed-route all bins daily, (b) fixed schedule every N days, (c) single-ultrasonic threshold,
(d) paper's sensor-fusion rule. Routing: KMeans clusters, depot-constrained TSP per cluster
(nearest neighbor + 2-opt), Euclidean distance for now.
One command: python scripts/run_benchmark.py -> prints and saves a results table (CSV + markdown).
Compare honestly to the paper's reported percentages as REFERENCE only; do not tune to match them.
Also report missed pickups / overflow events per strategy (bins that exceed 100%).
```

## Phase 3 — ML forecasting + anomaly detection
```
Phase 3 ONLY. (1) Per-bin fill forecasting (XGBoost first). Features: recent fill history, day of week,
area type (market/residential/apartment), festival/weekend flag. TIME-BASED split only.
Baselines: last value, moving average. Metrics: MAE/RMSE for forecast, and the downstream metric:
overflow events and trips saved when forecast drives scheduling.
(2) Anomaly detection for faulty sensors: generate synthetic faults with PLANTED ground truth in a
separate file; the detector must never read it; score precision/recall/PR-AUC.
(3) Yellow-bin logic: predict time-to-Red, collect Yellow bins only if near an existing route.
Report plainly if simple baselines match the ML model.
```

## Phase 4 — VRP + real roads
```
Phase 4 ONLY. Add OR-Tools VRP (truck capacity, multiple trucks per cluster) and compare
nearest neighbor vs 2-opt vs OR-Tools on distance and runtime.
Add real road distances for one Chennai area using OSMnx; cache the graph to disk; if the download fails,
fall back to Euclidean and log it clearly. Report how much Euclidean underestimates road distance.
Keep the simulation size configurable (road-network runs may use fewer bins; say so in the results).
```

## Phase 5 — dashboard + firmware
```
Phase 5 ONLY. Streamlit dashboard (dashboard/app.py): map of bins colored G/Y/R, selected-day truck routes,
KPI charts, baseline comparison table. Runs with plain: python -m streamlit run dashboard/app.py.
Firmware: firmware/nodemcu_smartbin/nodemcu_smartbin.ino for NodeMCU (3 HC-SR04 + HX711 load cell, Wi-Fi,
ThingSpeak HTTP), with pins and credentials in a separate config header that is gitignored (provide an example).
tools/thingspeak_fetch.py to pull channel data. Mark firmware NOT VERIFIED unless I test it on hardware.
```

## Phase 6 — docs, polish, interview prep
```
Phase 6 ONLY. Write README (problem, architecture diagram, how to run each phase's command,
results tables with the exact command that produced each number, limitations, NOT VERIFIED list).
Clean tests, make CI green for real reasons (do not weaken tests). Dockerfile: actually build and run it,
or mark NOT VERIFIED. Write docs/INTERVIEW.md: 30-second pitch, architecture, formulas, results,
every shortcut/limitation, 20 likely interview questions with answers, and a quiz for me.
Then check every claim in the README against real output and list anything you could not verify.
```

---

## After each phase (you do this)
1. `git status` and `git log --oneline`: check what changed.
2. Run the verify commands yourself. Check the numbers are real.
3. Ask Claude Code: "Explain what you built, quiz me on it, list every shortcut."
4. Commit. Start a new session for the next phase.
