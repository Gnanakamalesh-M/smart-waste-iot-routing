# Phase 5 results — Streamlit dashboard, NodeMCU firmware, ThingSpeak fetch

> ### Phase 5 produces NO new simulation result
>
> This is a presentation and hardware-interface phase. It adds no finding about sensors,
> forecasting, routing or cost, and it was careful not to appear to. Every figure the dashboard
> shows is either **synthetic demo data** generated here, or a number **parsed out of
> `docs/PHASE{1,2,3,4}_RESULTS.md`** at display time. Phase 3's strategy benchmark is still
> **NOT RUN** and Phase 4's tables are still **SMOKE RUN**s; the dashboard displays them with
> those labels rather than smoothing them over.

> ### The firmware is NOT VERIFIED
>
> `firmware/nodemcu_smartbin/nodemcu_smartbin.ino` has **never been compiled and never been
> flashed.** `arduino-cli` was not available in this session and was not attempted; no ESP8266
> toolchain was installed, no board was connected, and **no ThingSpeak channel has ever been
> contacted by this project** — there is no channel and no real key. Nothing below claims the
> sketch builds or runs.

## How these numbers were produced

Code and tests: commit `e9edd77` (`e9edd77506032efbbb17dee255bf91fd01e40c20`).
This document and the `docs/ASSUMPTIONS.md` update are the commit that follows it.

```
python scripts\make_demo_data.py --seed 42
python -m pip install -e ".[dev,app]"
python -m pytest -m unit
python -m pytest -m dash
python -m pytest -m sim
python -m pytest -m routing
python -m pytest -m ml
python -m pytest -m slow
python -m ruff check .
python -m ruff format --check .
python -m mypy
python tools\thingspeak_fetch.py --offline tests\data\thingspeak_sample.json
python scripts\run_dashboard.py
```

Config `config/default.yaml` unchanged — **Phase 5 added no config row and no new constant**.
Demo sizes are script constants in `scripts\make_demo_data.py` (`DEMO_N_BINS = 200`,
`DEMO_DAYS = 14`, `DEMO_N_CLUSTERS = 5`), the same choice `QUICK_N_BINS` in
`scripts\run_roads.py` represents: they shape a demonstration, not a result.

Versions: Python 3.11.9 on Windows, streamlit 1.66.0, plotly 7.1.0.

## Scope of Phase 5

**Built:** the Streamlit dashboard and its stdlib data loader; the committed demo snapshot and
its generator; the NodeMCU sketch, its gitignored-config pattern and its wiring README; the
ThingSpeak library and CLI; a firmware cross-check; a tracked-secrets guard; the `dash` marker and
CI chunk.

**Deliberately NOT built, and carried forward:**

| Item | Where it was supposed to be | State |
|---|---|---|
| Yellow-bin opportunistic collection | SPEC section 7, Phase 3 | **STILL NOT BUILT.** Out of scope in Phase 3, Phase 4 and here. Carried to Phase 6. |
| Phase 3 city-scale strategy benchmark | Phase 3 | **STILL NOT RUN** (~2.3 h). Not re-attempted; Phase 5 changed nothing it depends on. |
| Phase 4 publication-size runs | Phase 4 | **STILL NOT RUN.** All Phase 4 tables remain `--quick` smoke runs. |
| Any ML or routing change | — | **None made.** No file under `src/smartwaste/ml/` or `src/smartwaste/routing/` was touched. |

Phases 1–4 are untouched: `git show --stat e9edd77` lists no pre-existing source file except
`pyproject.toml`, `.github/workflows/ci.yml`, `.gitignore` and `CLAUDE.md`, none of which carries
a number. **No pre-existing test file was edited** — all five test files are new.

---

## 1. The dashboard

### What it reads, and why it cannot start a job

Two sources, both committed, reached only through `src/smartwaste/dashboard_data.py`:

1. **`dashboard/demo_data/`** — a seeded snapshot generated once by
   `scripts\make_demo_data.py`, checked into git.
2. **`docs/PHASE{1,2,3,4}_RESULTS.md`** — parsed as markdown when the page loads.

`dashboard/app.py` imports **exactly one** project module, `smartwaste.dashboard_data`, which is
stdlib only. There is no import of `city_sim`, `benchmark`, `phase3`, `phase4`, `strategies`,
`demand` or anything under `smartwaste.ml` on this path. That is the strongest form of CLAUDE.md's
runtime rule: the page cannot trigger a long job because it has no simulator to call.
`tests/test_dashboard_app.py` and `tests/test_dashboard_data.py` assert it on the source, the way
`tests/test_anomaly.py` asserts the leakage firewall.

The same tests pin that `app.py` contains no `open(`, `.write_text(`, `.mkdir(`, `subprocess`,
`urllib`, `requests` or `socket`. It is a read-only page with no network access.

### Why the baseline tables are parsed and not transcribed

`results/benchmark.csv` is gitignored (`results/*`), so a dashboard reading from there is blank on
a fresh clone. The alternative — copying Phase 2's table into a CSV beside the dashboard — creates
a second copy of every number that can drift from the document. So the tables are parsed out of
the docs at display time. **There is exactly one copy of every published number, and it is the one
in the results doc.**

The parser keeps each table's markdown heading **verbatim**, and that is deliberate: the heading is
where this project's honesty labels live. `### Table A — capacity-free, the fair comparison [SMOKE
RUN, NOT FOR PUBLICATION]` reaches the page intact, so a Phase 4 table cannot be displayed with its
warning stripped off. Measured: 32 tables parse out of the four docs, and
`tests/test_dashboard_data.py` resolves all eight headings the dashboard asks for against the real
committed files — a docs rename fails a test instead of silently blanking a panel.

### All three fault fractions are shown

`docs/ASSUMPTIONS.md` records that the (d)-mean-versus-(e)-median verdict **reverses** between
`stuck_bin_fraction` 0 and 0.02, and that the fault rate is an invented number. A dashboard showing
one fault fraction would let the reader pick the conclusion the project refused to pick, so the
baseline panel asks for a heading substring that matches all three (0, 0.02, 0.1) and renders every
match. A test pins that it finds exactly three.

### The map is kilometres, not geography

`CityLayout.locations` are x/y in km on a uniform-random 100 × 100 km square; the city simulation
has no latitude or longitude anywhere. The map is therefore a plain kilometre scatter with equal
axes, captioned *"Synthetic 100 x 100 km grid. NOT a real city and NOT geographic coordinates."*

**Projecting the grid onto real Chennai coordinates was considered and rejected.** It would look
better in a screenshot and it would be the Phase 4 problem again — `docs/ASSUMPTIONS.md` already
says *"a real road network makes synthetic bins look more authoritative than they are"*. Painting
uniform-random bins onto real streets does the same thing, harder. No fabricated coordinate exists
in this phase.

### The screens, described (no screenshots are committed)

No image files are committed, so this is a text description of what renders. It was produced by a
headless `AppTest` run, not from a browser.

**Everywhere, above the tabs:** a red `st.error` box reading **SIMULATED DATA, NOT HARDWARE
READINGS**, then two captions naming the generating command, the seed, the demo size, the demand
scenario, the stuck-sensor fraction, and the strategy driving collection. The banner text comes
from `meta.json`'s own `label` field, so the file and the page cannot disagree about it. The
sidebar repeats the banner plus *"The NodeMCU firmware in this repo has never been compiled and
never been flashed"*, so both warnings are visible on every tab.

**Sidebar controls:** a day slider (0–13) and a radio for *ground truth* vs *sensor fusion*, with
the help text *"Ground truth exists only because the city is simulated. A real deployment sees the
sensor column only."*

**Tab "Bin map".** 200 dots on a square kilometre plot, coloured green / orange / dark red, with
the depot a blue square at (50, 50). Hovering a bin shows its id, true fill %, sensor fill % and
true weight. Below: a two-column table of ground-truth versus sensor state counts, and two metrics
— *bins where the two disagree* and *RED bins not collected (missed pickups)*. The caption says
the gap between the columns is the point of the project, and that the ground-truth column exists
only because the city is simulated.

**Tab "Routes".** One polyline per cluster for the chosen day, over the same axes, each starting
and ending at the depot, plus a per-cluster table of stops and bins visited. Captioned straight
lines, **not** roads, with the Phase 4 ~1.45× median detour mentioned *and* the note that
`docs/ASSUMPTIONS.md` says that factor does not transfer to this geometry. On days 0–2 the panel
instead reads *"No bins were selected on day N, so no truck rolled"* — under the paper's rule a day
with no RED bin is a day with no collection.

**Tab "KPIs".** Six total metrics, then four per-day bar charts (distance, travel time, CO2,
variable cost) and a combined bins-visited line over an overflow-events bar. Captioned with the
demo size and *"NOT the Phase 2 city benchmark"*, and with the reminder that every cost and CO2
constant behind the figures is invented and uncited, so the absolute INR and kg carry no external
meaning.

**Tab "Baseline comparison".** The parsed Phase 2 headline and reduction tables at all three fault
fractions, the *"Where a simpler baseline matches or beats the paper's method"* table under an
amber warning restating that the sensor-free 3-day calendar had **zero** overflow events, the
paper's reference percentages, Phase 3's `NOT RUN` state table under a red error box, and Phase 4's
Tables A, B, B(ii) and C under a red error box naming the 0.29% OR-Tools margin against the 2.0%
threshold. Every table carries its own verbatim heading and a `source: docs/PHASEn_RESULTS.md`
caption.

**Tab "Not verified".** Four expanders rendering each results doc's own `## NOT VERIFIED` section
text, straight from the file rather than restated.

### The demo snapshot

```
python scripts\make_demo_data.py --seed 42
```

```
SIMULATED DATA, NOT HARDWARE READINGS
demo snapshot: 200 bins x 14 days, 5 clusters, seed 42
strategy: (d) paper's sensor fusion: 3-sensor MEAN + load cell [the paper's method]
demand scenario: heterogeneous
NOT the Phase 2 city benchmark: 200x14 is a demonstration, not a result.

wrote D:\smart-waste-iot-routing\dashboard\demo_data:
  bins.csv           8432 bytes
  bin_days.csv     148852 bytes
  routes.csv         6519 bytes
  kpis.csv            660 bytes
  meta.json           437 bytes
  TOTAL            164900 bytes
```

**Total committed size: 164,900 bytes (161.0 KiB).** A test fails if it exceeds 1 MB, with the
instruction to cut the bin count rather than commit a bigger blob.

It replays the day loop of `smartwaste.city_sim.run_strategy` exactly — waste arrives, overflow is
measured, **every** bin is read whether or not the strategy looks at it, the strategy chooses, the
choices are routed per cluster, the chosen bins are emptied. `run_strategy` itself could not be
reused because it returns only aggregates: its `RouteCache` yields a distance and discards the
tour, and a map needs the tour. `smartwaste.phase4.build_compare_instances` re-implements the loop
for the same reason.

#### Demo snapshot: what 200 bins x 14 days at seed 42 actually did [DEMONSTRATION, NOT A RESULT]

| day | true G/Y/R | sensor G/Y/R | disagree | selected | missed RED |
|---|---|---|---|---|---|
| 0 | 200/0/0 | 200/0/0 | 0 | 0 | 0 |
| 1 | 200/0/0 | 200/0/0 | 0 | 0 | 0 |
| 2 | 199/1/0 | 199/1/0 | 0 | 0 | 0 |
| 3 | 193/6/1 | 194/5/1 | 1 | 1 | 0 |
| 4 | 180/19/1 | 181/18/1 | 3 | 1 | 0 |
| 5 | 162/27/11 | 163/26/11 | 1 | 11 | 0 |
| 6 | 153/27/20 | 153/26/21 | 5 | 21 | 1 |
| 7 | 166/26/8 | 166/26/8 | 2 | 8 | 0 |
| 8 | 163/31/6 | 162/28/10 | 5 | 10 | 0 |
| 9 | 127/46/27 | 127/45/28 | 3 | 28 | 0 |
| 10 | 130/47/23 | 130/47/23 | 2 | 23 | 0 |
| 11 | 133/51/16 | 130/55/15 | 4 | 15 | 1 |
| 12 | 125/53/22 | 125/52/23 | 3 | 23 | 0 |
| 13 | 123/57/20 | 125/54/21 | 7 | 21 | 0 |

Totals over the 14 days: **4954.28 km**, **206.27 hr**, **4647.12 kg CO2**, **241,493.02 INR**
variable cost, **162 bin visits**, **11 overflow events**.

**These numbers are a demonstration and are compared with nothing.** 200 bins is 4% of
`sim.n_bins` and 14 days is 47% of `sim.days`, the demand scenario is `heterogeneous` where Phase
2's tables are `smooth`, and the cost constants are invented and uncited. No table anywhere in this
project places them beside a Phase 2 figure, deliberately.

**Two things in it are worth reading anyway**, because they are the project's known weaknesses
showing up in a page a human will actually look at:

- **11 overflow events in 14 days**, and the YELLOW column growing monotonically to 57 bins by day
  13. YELLOW bins are never collected under the paper's rule, and this is what that looks like.
  Phase 5 did not fix it — see the scope table.
- **2 missed RED pickups** (days 6 and 11), out of 164 genuinely-RED bin-days. The sensors and the
  truth disagree on 0–7 bins a day. That gap is the entire subject of Phases 1–2, and the map's
  truth/sensor toggle exists to show it rather than hide it.

### Determinism

```
python scripts\make_demo_data.py --seed 42 --out-dir results\d1
python scripts\make_demo_data.py --seed 42 --out-dir results\d2
fc results\d1\kpis.csv results\d2\kpis.csv
```

All five files byte-identical (checked with `cmp`): `bins.csv`, `bin_days.csv`, `routes.csv`,
`kpis.csv`, `meta.json`. Regenerating also reproduced the committed snapshot exactly —
`git diff --quiet dashboard/demo_data/` passed after a fresh run. No timestamps, no timings, fixed
`{:.4f}` float formatting.

Line endings are pinned to LF on both sides: the generator writes `lineterminator="\n"` and
`newline="\n"`, and a new `.gitattributes` sets `eol=lf` for these files. Without the
`.gitattributes` rule this repo's `core.autocrlf=true` would rewrite them on the next clone, so the
LF test would have passed for whoever generated them and failed for the next person — caught during
this phase, not after.

---

## 2. The firmware — NOT VERIFIED

**Never compiled. Never flashed. No hardware.** `arduino-cli` was confirmed unavailable before
anything was written, and no compilation was attempted, so there is no compile output to show and
none is implied. The warning is in the sketch's own header block, in
`firmware/nodemcu_smartbin/README.md`, in `src/smartwaste/firmware_reference.py`, in
`tests/test_firmware_crosscheck.py`, in the dashboard sidebar, and here. A test asserts the sketch
text itself still says it.

### What the cross-check does and does not establish

`tests/test_firmware_crosscheck.py` — 57 tests, `unit` chunk — establishes two things:

| # | What is checked | What it proves |
|---|---|---|
| 1 | `src/smartwaste/firmware_reference.py`, a hand transcription of the sketch's arithmetic, agrees with `smartwaste.fusion.fuse` on the state | The rule we **wrote down** matches the Python rule |
| 2 | The six `#define`d constants parsed out of the `.ino` text equal `config/default.yaml` | The numbers we **wrote down** match the config |

**Neither runs a C++ compiler.** The first compares two Python functions; the second applies a
regular expression to a text file. They cannot catch a C++ type error, a `float`-versus-`double`
difference, an overflow the transcription did not reproduce, a pin mis-assignment, or a library
quirk. The transcription exists because the sketch necessarily reimplements the rule — an ESP8266
cannot import Python — and two copies of a rule drift. This makes the drift *detectable*: edit a
threshold in one place and two tests fail.

Inputs fed to both implementations: every threshold boundary (`Fv` exactly 60.0 and exactly 80.0,
`W` exactly `alpha * W_max` = 42.5 kg, and values a hair either side), 2000 seeded pseudo-random
`(d1, d2, d3, W)` tuples from `random.Random(42)`, and degenerate cases (all distances at `H`, all
at the 2.0 cm floor, violently uneven, centre peak, one timed-out sensor). **Zero disagreements.**
A companion test confirms the sweep actually produces all three states, so it is not passing
vacuously.

Two details of the rule are load-bearing and are pinned rather than assumed:

- **Clamp order.** Each distance is clamped to `[MIN_DISTANCE_CM, HEIGHT_CM]` *first*, then
  averaged, then `Fv` is clipped to `[0, 100]`. The Python pipeline does the same two stages in the
  same order (`noise.SensorArray` then `fusion.fill_percent`). A test shows that averaging first
  gives a different `d_bar` (50.0 versus 50.5 for raw echoes of 1 cm and 99 cm), so the order is
  part of the rule.
- **A timed-out sensor is dropped from the mean, not averaged in as zero.** Zero reads as "waste is
  touching the sensor" — the opposite of "no reading" — and would post a silent false RED. A test
  pins that a `(90, dead, 90)` reading yields `d_bar = 90.0` over 2 sensors and stays GREEN, and
  that the naive `mean(90, 0, 90) = 60` is *not* what happens. With all three dead the sketch posts
  nothing, and the transcription raises.

A float caveat, stated rather than engineered around: the sketch uses C `float` (32-bit) and the
transcription uses Python `float` (64-bit). They agree on the rule, not necessarily bit-for-bit on
a value within ~1e-7 of a threshold. The boundary tests use exactly representable values (60.0,
80.0, 42.5) where both agree.

### Secrets

`config.h` holds the Wi-Fi password and the ThingSpeak **write** key. It is gitignored by a
dedicated `.gitignore` rule — the existing `.env` / `.env.*` patterns do not cover a `.h` file —
and only `config.example.h`, with `YOUR_WIFI_SSID` / `YOUR_WIFI_PASSWORD` / `0000000000000000`, is
tracked. The sketch logs the HTTP status and never the URL, because the URL carries the write key;
a test greps the sketch to confirm it never prints `url`.

`tests/test_no_secrets_tracked.py` (12 tests, `unit`) is the alarm behind that protection, and it
runs on **every** `unit` invocation in this and every later phase. It asks `git ls-files`, not the
filesystem — a developer's local `config.h` and `.env` are *supposed* to exist and must never fail
a test run; only their being **tracked** is a problem. It fails if git tracks
`firmware/nodemcu_smartbin/config.h`, any `.env*` other than `.env.example`, a non-placeholder
SSID/password/`api_key` assignment, or any 16-character key-shaped string. It skips, rather than
failing, when `git` is unavailable.

Two guards-on-the-guard are included, because a security check nobody has seen fail is a check
nobody knows works: one test plants `#define WIFI_PASSWORD "hunter2-correct-horse"` and asserts the
detector catches it; another asserts the shipped placeholders are **not** flagged, since a check
that cries wolf is a check someone disables. A third asserts the `git ls-files` listing is
non-empty, so the whole file cannot pass vacuously.

The assignment scan excludes `.py` deliberately, and this was a measured false positive, not a
guess: Python source legitimately contains `READ_API_KEY_VAR = "THINGSPEAK_READ_API_KEY"`, which
assigns a variable *name*. Python is still covered by the key-shape scan, which is what would catch
a real pasted key.

---

## 3. The ThingSpeak fetch path

```
python tools\thingspeak_fetch.py --offline tests\data\thingspeak_sample.json
```

```
SIMULATED OR FIXTURE DATA UNLESS A REAL CHANNEL WAS CONFIGURED.
Firmware is NOT VERIFIED: never compiled, never flashed, never read from hardware.

source: tests\data\thingspeak_sample.json (offline; no network used)
channel: smartbin-fixture (NOT a real channel)
rows in response: 7
rows usable:      4
rows skipped:     3
  SKIPPED row 3: field1='not_a_number' is not a number
  SKIPPED row 4: field2 is null or missing
  SKIPPED row 6: field3 is null or missing

  entry  created_at               d_bar cm      W kg     Fv %  overload  state
------------------------------------------------------------------------------
      1  2026-01-01T06:00:00Z        92.10     1.250     7.90        no  GREEN
      2  2026-01-01T12:00:00Z        37.67    18.900    62.33        no  YELLOW
      3  2026-01-01T18:00:00Z        13.57    31.400    86.43        no  RED
      6  2026-01-02T18:00:00Z        70.00    44.000    30.00       yes  RED

States come from smartwaste.fusion.fuse -- the same rule strategy (d) uses.
ThingSpeak field5/field6 are IGNORED: the point is to re-derive, not to trust.
```

**This is a committed fixture, not a channel.** `--offline` is the only path ever exercised.

Entry 6 is in the fixture on purpose: a bin only 30% full by ultrasonic but at 44.0 kg against the
42.5 kg threshold, so the **load cell alone** escalates it to RED. That is the paper's whole reason
for having a load cell, and a test asserts the fixture still contains exactly one such row.

### Design notes

**Logic lives in `src/smartwaste/thingspeak.py`; `tools/thingspeak_fetch.py` only parses arguments
and prints** — the same split every `scripts\run_*.py` has from its `smartwaste.*_eval` module.
Tests import the package, never `tools/`.

**The network is injected, not imported.** `Fetcher` is a plain `Callable[[str], str]`, and the
default `urlopen_fetcher` does its `urllib.request` import *inside the function body*, so importing
the module cannot touch the network. No test needs a socket block or a monkeypatch, and a test
asserts no top-level import mentions `urllib`. The closest existing precedent is
`allow_download: bool` in `smartwaste/routing/roads.py`; a callable is used here because a fetcher
has a return value worth faking. **This is the first injected callable in the codebase.**

**A reading becomes the project's own type.** `to_sensor_reading` produces
`smartwaste.noise.SensorReading`, so `smartwaste.fusion.fuse` classifies hardware data unchanged
with exactly strategy (d)'s arguments. No rule is reimplemented on the hardware path.

**Fields 5 and 6 are ignored on purpose.** The sketch posts `Fv` and the state; this module
recomputes both from fields 1–4, because the point is to check that the firmware's arithmetic and
the Python rule agree, not to take the firmware's word for it. A test feeds a row whose `field5`
flatly contradicts fields 1–3 and asserts the recomputed value wins.

**Nothing is lenient about data, and nothing crashes on it.** An empty body, a missing `feeds` key
and `{"feeds": []}` all yield zero rows and no exception — a channel that has never received a post
is a normal state. A malformed row (non-numeric, `null`, missing field, short row, wrong JSON type)
is skipped with a stated reason and counted, never silently dropped and never raised. Malformed
JSON *does* raise, because it means the response was not a ThingSpeak response at all.

### The read key never escapes

Not in stdout, not in a `repr`, not in an exception message — an exception message is a log line,
and a secret in a log file is a leaked secret. `ThingSpeakSettings` overrides `__repr__` rather
than inheriting the dataclass default, because a dataclass ends up inside tracebacks and pytest
output. `fetch_body` rebuilds any transport failure's message instead of re-raising it, since
`urllib` puts the URL it failed on — key and all — into its own text. Tests assert the key is absent
from the settings `repr` and `str`, from a fetch failure message, from `repr(exc)`, from
`exc.args`, from a validation failure, and from a report `repr`.

**A bug was found here and fixed in this phase.** `redact()` scanned for `api_key=` in a `while`
loop that *broke* on encountering an already-redacted parameter — so a message quoting two
different URLs would have had the first key redacted and **the second one left intact**. The cursor
now only ever moves forward, stepping over an already-redacted parameter instead of treating it as
the end of the work. The test that caught it (`test_redact_handles_several_api_key_parameters`) is
committed.

### The field mapping is INVENTED

The paper says the NodeMCU sends "fill % and weight" to ThingSpeak and **never states which field
carries what.** This mapping is ours and uncited, and it is the one thing here a real channel could
silently disagree with:

| Field | Meaning | Source |
|---|---|---|
| `field1` / `field2` / `field3` | ultrasonic distances, left / centre / right, cm | **invented, uncited** |
| `field4` | load-cell weight, kg | **invented, uncited** |
| `field5` | `Fv`, fused fill percentage (posted, not trusted) | **invented, uncited** |
| `field6` | state code: 0 GREEN, 1 YELLOW, 2 RED | **invented, uncited** |

---

## 4. Dependencies and CI

The existing **`app` extra** (`streamlit>=1.37`, `plotly>=5.23`) was declared in Phase 0 and never
used. Phase 5 documents and uses it rather than adding a second extra installing the same two
wheels, and `docs/PHASE5_RESULTS.md` records that choice so it does not look like an oversight.

A new **`dash` pytest marker** and CI chunk carry the one test that needs those wheels. It has its
own chunk on purpose: putting it in `unit` would mean the chunk that is supposed to *prove*
independence from streamlit would instead quietly skip a test inside itself. `CLAUDE.md`'s marker
list was updated in the same commit so the two cannot disagree.

CI changes, both lessons from Phases 3 and 4 applied in the same phase as the code:

- `types` job installs `.[dev,ml,roads,app]`. mypy now also checks `tools/` and `dashboard/`, so
  the `import streamlit` in `dashboard/app.py` and the lazy `import plotly.graph_objects` must
  resolve. Without the extra, mypy fails on the import lines themselves — exactly how the Phase 3
  types job broke, and then the Phase 4 one.
- `tests` matrix gains `dash`, installed as `.[dev,app]`. Every other chunk keeps `.[dev]`. The
  existing marker-registered guard and exit-code-5 tolerance handle the new chunk unchanged.
- The `slow` job's comment now states it deliberately omits `app` as well.

### The unit / sim / routing / ml / slow chunks must not need streamlit or plotly

Not asserted from the lazy-import structure — **measured, by actually uninstalling both wheels.**

```
python -m pip uninstall -y streamlit plotly
python -m pytest -m unit
python -m pytest -m sim
python -m pytest -m routing
python -m pytest -m ml
python -m pytest -m slow
python -m pytest -m dash
```

| chunk | with `.[dev,app]` | with streamlit and plotly **UNINSTALLED** |
|---|---|---|
| `unit` | 433 passed | **433 passed** |
| `sim` | 125 passed | **125 passed** |
| `routing` | 107 passed | **107 passed** |
| `ml` | 14 passed | **14 passed** |
| `slow` | 5 passed | **5 passed** |
| `dash` | 17 passed | **5 passed, 12 skipped** |

Identical counts in the five chunks that must not need the extra. The `dash` chunk's 5 surviving
tests are the source-grep firewall checks, which need no streamlit to read a file — so even without
the extra, CI still verifies the dashboard imports no simulation module.

`scripts\make_demo_data.py`, `tools\thingspeak_fetch.py` and `scripts\run_dashboard.py` were all
re-run with the extra uninstalled and produced byte-identical output.

**A second bug was found here and fixed.** `pytest.importorskip("streamlit.testing.v1")` was not
sufficient. `pip uninstall streamlit` leaves the `streamlit/` directory behind whenever anything has
written a `__pycache__` inside it — which a previous test run has — and **an empty directory is a
valid namespace package in Python 3.** So the import succeeded, nothing skipped, and all 12 tests
died on an `AttributeError`:

```
3 failed, 5 passed, 675 deselected, 9 errors in 0.49s
```

Half-uninstalled is the state a developer is actually in right after an uninstall, so the guard now
checks for the attributes it needs (`streamlit.__version__` is not `None`, and
`streamlit.testing.v1.AppTest` exists). Both absent states now behave identically —
`5 passed, 12 skipped` — verified with pip's leftover shell present and after removing it.

### The dashboard was never served in this session

CLAUDE.md's runtime rule and the explicit instruction were both honoured: `streamlit run` was
**not** executed. The only dashboard execution was `streamlit.testing.v1.AppTest`, which runs
`dashboard/app.py` **in-process** — no server, no port, no browser, no network. 17 tests: the app
runs without raising, the banner is on the **rendered** page rather than merely in the source,
`NOT RUN` and `SMOKE RUN` both reach the page, the firmware-unverified text is in the sidebar, the
slider and radio exist with the right ranges, and **all 14 days render** including the three on
which no truck rolled.

---

## Verification (Windows cmd, from the repo root)

```
python scripts\make_demo_data.py --seed 42
python -m pytest -m unit
python -m pytest -m dash
python -m pytest -m sim
python -m pytest -m routing
python -m pytest -m ml
python -m pytest -m slow
python -m ruff check .
python -m ruff format --check .
python -m mypy
python tools\thingspeak_fetch.py --offline tests\data\thingspeak_sample.json
python scripts\run_dashboard.py
```

`ruff check`, `ruff format --check` and `mypy --strict` are clean over **65 source files**.

Run the dashboard yourself:

```
python -m pip install -e ".[dev,app]"
python -m streamlit run dashboard\app.py
```

Determinism:

```
python scripts\make_demo_data.py --seed 42 --out-dir results\d1
python scripts\make_demo_data.py --seed 42 --out-dir results\d2
fc results\d1\kpis.csv results\d2\kpis.csv
```

The independence check, which is the one worth repeating:

```
python -m pip uninstall -y streamlit plotly
python -m pytest -m unit
python -m pytest -m sim
python -m pytest -m routing
python -m pytest -m ml
python -m pytest -m slow
python -m pip install -e ".[dev,app]"
```

Secrets:

```
git ls-files firmware
findstr /S /I "ssid password api_key" firmware\*.ino firmware\*.h
python -m pytest -m unit -k no_secrets
```

`git ls-files firmware` lists exactly three files — `README.md`, `config.example.h`,
`nodemcu_smartbin.ino`. No `config.h` is tracked. The only credential-shaped strings in the tracked
firmware are the macro *uses* `WIFI_SSID` / `WIFI_PASSWORD` in the sketch and the placeholders in
`config.example.h`.

---

## Honest summary

Phase 5 built the three things SPEC section 7 asks for, and the honest reading of each is different:

- **The dashboard works and was run**, headlessly. It displays synthetic demo data and numbers
  parsed from the committed docs, every table under its own verbatim label. Its main design
  property is negative: it cannot start a simulation, because it has no simulator to call.
- **The ThingSpeak path works against a fixture and has never seen a channel.** The code handles an
  empty channel and malformed rows, converts readings into the project's own type, and never leaks
  the read key. Whether a real ThingSpeak channel matches our invented field mapping is untested
  and unknowable from here.
- **The firmware is text.** It has never been compiled. The cross-check verifies the rule we wrote
  down, not the translation unit. This is the weakest deliverable of the phase and is labelled so
  in six places.

Two real bugs were found by the tests written in this phase — a redaction gap that would have
leaked a second API key, and a skip guard that failed on a half-uninstalled package. Both are fixed
and both have committed tests. Neither was in Phases 1–4 code.

Nothing about the project's actual findings changed. Phase 3's benchmark is still NOT RUN, Phase
4's tables are still SMOKE RUNs, Yellow bins are still never collected, and the dashboard's job is
to say all three out loud rather than to make the project look finished.

---

## NOT VERIFIED

New in Phase 5:

- **The firmware has NEVER been compiled and NEVER been flashed.** `arduino-cli` was unavailable
  and no compilation was attempted. No ESP8266 toolchain, no board, no waste, no reading. Every
  statement about what the sketch does is a statement about what its text says.
- **No ThingSpeak channel has ever been contacted.** No channel exists, no real read key or write
  key exists, and no reading from the physical bin has ever been ingested. `--offline` against
  `tests/data/thingspeak_sample.json` is the only path ever executed; the `urlopen_fetcher` branch
  has never run.
- **The ThingSpeak field mapping is invented and uncited.** The paper states none. A real channel
  could silently disagree, and nothing here would detect it.
- **The cross-check tests a Python transcription, not the C++.** It cannot catch a C++ type error, a
  `float`-vs-`double` difference, a pin mis-assignment or a library quirk. The constants test
  applies a regular expression to a text file.
- **The wiring table in `firmware/nodemcu_smartbin/README.md` is untested.** Pin choices, the ECHO
  level-shifter advice and the HX711 calibration procedure are reasoned from datasheets, not from a
  built circuit. The boot-mode warnings about GPIO0/GPIO2 matter and have not been experienced.
- **All dashboard data is synthetic or copied.** The demo snapshot is 200 bins x 14 days of
  generated data; the baseline tables are text parsed out of committed markdown. **Not one figure on
  the dashboard is a hardware reading.**
- **The demo snapshot is not the city benchmark and is comparable with nothing.** 200 bins (4% of
  `sim.n_bins`), 14 days (47% of `sim.days`), `heterogeneous` demand where Phase 2's tables are
  `smooth`, and invented cost constants. Its absolute figures carry no external meaning.
- **The dashboard has only ever been exercised headlessly**, by `AppTest` in-process.
  `streamlit run` was never started, so no browser has rendered the page, and nothing is known
  about its layout at a real viewport, its behaviour on a slow connection, or its appearance in a
  browser's dark mode.
- **Phase 3's strategy benchmark remains NOT RUN and Phase 4's publication runs remain NOT RUN.**
  Phase 5 did not attempt either. The dashboard displays both as such.
- **Yellow-bin opportunistic collection is STILL not built.** SPEC section 7 lists it under Phase 3;
  it was out of scope there, in Phase 4, and here. The demo snapshot shows the consequence — the
  YELLOW count climbing to 57 of 200 bins by day 13, with 11 overflow events — which makes the
  weakness visible without fixing it. **Carried to Phase 6.**
- **GitHub Actions CI has still never executed.** CLAUDE.md forbids `git push`, so nothing has run
  there. Phase 5 adds a sixth `dash` chunk and a fourth extra to the `types` job, both committed and
  **both unproven in CI**. They were verified locally only.
  **Correction 2026-10-15: CI has since run green, see README.**
- **The map's kilometre plot is honest about the grid, which is still not a city.** Bins are
  uniform-random points on a 100 x 100 km square. No road network, no real geography, and
  deliberately no projection onto real coordinates.
- **`.gitattributes` is new and its cross-platform effect is untested.** The `eol=lf` rule is
  verified on this Windows machine with `core.autocrlf=true`; no clone on Linux or macOS has been
  made to confirm the committed demo files survive a round trip there.
- **Phase 5 ran on one machine only.** streamlit 1.66.0 / plotly 7.1.0 / Python 3.11.9 on Windows.
  The stdlib parts keep the project's byte-identical guarantee; the `AppTest` results are
  streamlit-version-dependent and no test pins a rendered pixel or a layout.
