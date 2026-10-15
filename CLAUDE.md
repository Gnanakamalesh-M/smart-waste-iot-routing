# CLAUDE.md — standing rules for this repo

Project: smart-waste-iot-routing. Full spec: `docs/SPEC.md`. Phase prompts are given to you one at a time.

## Workflow rules
1. Work on ONE phase at a time. Stop at the end of the phase. Do not start the next one.
2. Start every phase in plan mode. Ask design questions with options instead of guessing.
3. Never run `git push`. Never add `Co-Authored-By` lines or any Claude attribution to commits.
4. Commit after each phase with a clear message. Do not squash or rewrite history.
5. Do not skip, delete, or weaken tests to get green. If a test fails, fix the code or tell me why.
6. Do not claim something works unless you ran it. If you could not run it, say "NOT VERIFIED".

## Environment rules (Windows, cmd)
- Every script must run with plain `python` (no `make`, no bash-only commands).
- Use cmd-friendly commands in docs (`findstr`, `.venv\Scripts\activate`).
- Target Python 3.11 (same as CI). Do not use features newer than 3.11.
- Tests must be splittable into chunks (pytest markers: `unit`, `sim`, `ml`, `routing`, `slow`,
  `dash`). `dash` was added in Phase 5 for the streamlit dashboard smoke test; it needs the
  optional `app` extra, and the other five must stay green without it.
  Never rely on one giant test process.
- Real secrets never go in git. `.env` is gitignored; only `.env.example` is tracked.

## Data / ML rules
- No leakage. Detectors and models NEVER read ground-truth files or columns.
- Use time-based splits for forecasting. Never random-split time series.
- Imbalanced metrics: report PR-AUC / precision / recall, not accuracy.
- Fixed random seeds everywhere. Same command + same seed = same numbers.
- Always compare against fair baselines. If a simple baseline matches or beats the ML model,
  report that plainly. Do not hide it.

## Honesty rules (README + docs)
- Every number in README/docs must be pasted from real output AND list the command that produced it.
- Keep a "Not verified / Limitations" section. Include: only 1 physical bin was built,
  5000-bin results are simulation, synthetic data, anything not actually run (Docker, hardware).
- Do not cite counts (tests, files, families) you have not seen in real output.

## Explainability
- Keep code readable. Prefer simple over clever.
- At the end of each phase, give a short "what I built / how it works / shortcuts taken" summary
  so I can explain it in interviews.
