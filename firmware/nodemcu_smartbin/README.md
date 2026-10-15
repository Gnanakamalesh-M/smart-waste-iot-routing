# nodemcu_smartbin — NOT VERIFIED

**This firmware has never been compiled and has never been flashed.**

No `arduino-cli` was available in the session that wrote it, no ESP8266 toolchain was installed,
no board was connected, and no ThingSpeak channel has ever been contacted by this project. Nothing
in this directory is a claim that the sketch builds, runs, or produces correct readings. If you
want it verified, compile it and flash it yourself — then the claim is yours and it is real.

What *is* checked, by `tests/test_firmware_crosscheck.py`:

1. `src/smartwaste/firmware_reference.py` — a Python transcription of the sketch's arithmetic —
   agrees with `src/smartwaste/fusion.py` on the RED/YELLOW/GREEN state for thousands of
   readings, every threshold boundary included.
2. The six `#define`d constants in the sketch equal the matching values in `config/default.yaml`.

Neither runs a C++ compiler. They verify the rule we *wrote down* and the numbers we wrote down.
See `docs/PHASE5_RESULTS.md`.

## Before you build

```
copy firmware\nodemcu_smartbin\config.example.h firmware\nodemcu_smartbin\config.h
```

Then edit `config.h` with your Wi-Fi SSID, Wi-Fi password, ThingSpeak **write** key, your pin
assignments and your HX711 calibration factor.

`config.h` is **gitignored** and must never be committed — it holds a password and a write key,
and a write key lets anyone push data into your channel. `tests/test_no_secrets_tracked.py` fails
the suite if it ever becomes a tracked file.

The **read** key is a different key and lives in `.env` (also gitignored), for
`tools\thingspeak_fetch.py`. The sketch only writes; the fetch script only reads.

## Dependencies (none of these were installed here)

| What | Where |
|---|---|
| ESP8266 board support | Boards Manager URL `https://arduino.esp8266.com/stable/package_esp8266com_index.json`, then install "esp8266 by ESP8266 Community" |
| HX711 Arduino Library | Library Manager, "HX711 Arduino Library" by Bogdan Necula (`bogde/HX711`) |

Board: **NodeMCU 1.0 (ESP-12E Module)**, i.e. `esp8266:esp8266:nodemcuv2`.

## Wiring

Pins are **suggestions in `config.example.h`**, not a tested layout. Change them there, not in
the sketch.

| Component | Signal | Default pin | GPIO | Note |
|---|---|---|---|---|
| HC-SR04 left | TRIG | `D1` | 5 | |
| HC-SR04 left | ECHO | `D2` | 4 | needs a level shift, see below |
| HC-SR04 centre | TRIG | `D5` | 14 | the sensor a single-sensor design would use |
| HC-SR04 centre | ECHO | `D6` | 12 | needs a level shift |
| HC-SR04 right | TRIG | `D7` | 13 | |
| HC-SR04 right | ECHO | `D8` | 15 | needs a level shift; D8 has a pull-down, keep it low at boot |
| HX711 | DOUT | `D3` | 0 | GPIO0 affects boot mode — avoid pulling it low at reset |
| HX711 | SCK | `D4` | 2 | GPIO2 also affects boot mode |
| All | VCC | `VU` / 5V | — | HC-SR04 and HX711 both want 5 V |
| All | GND | `GND` | — | common ground, or the echo timing is nonsense |

**The ECHO lines are the one thing that will damage a board.** HC-SR04 drives ECHO at 5 V and the
ESP8266 is a 3.3 V part. Put a divider on each ECHO line (1 kΩ from ECHO to the pin, 2 kΩ from the
pin to GND gives ~3.3 V) or use a 3.3 V-tolerant module. Three dividers, one per sensor.

Powering three HC-SR04s and an HX711 from a USB-powered NodeMCU's regulator is marginal. A
separate 5 V supply with a common ground is the safer build.

Sensor placement follows the paper: three ultrasonic sensors on the **upper inner surface**
(left / centre / right), load cell **under** the bin, inner side walls angled to push waste toward
the centre.

## The rule

Identical to `src/smartwaste/fusion.py`, which is the paper's own:

```
d_bar    = mean of the valid ultrasonic distances
Fv       = 100 * (1 - d_bar / HEIGHT_CM), clipped to [0, 100]
overload = W >= ALPHA * W_MAX_KG
state    = RED    if overload or Fv >= 80
           YELLOW if Fv >= 60
           GREEN  otherwise
```

Two details are load-bearing, not incidental:

- **Clamp order.** Each distance is clamped to `[MIN_DISTANCE_CM, HEIGHT_CM]` *first*, then
  averaged, then `Fv` is clipped to `[0, 100]`. Averaging before clamping gives different answers
  near a full or empty bin.
- **A timed-out sensor is dropped from the mean**, not averaged in as zero. Zero would read as
  "waste is touching the sensor" — the opposite of "no reading" — and would post a silent false
  RED. With all three dead, the sketch posts nothing.

`ALPHA = 0.85` is **not from the paper**. The paper defines `W_th = alpha * W_max` with
`0 < alpha < 1` and never states alpha. Every constant is recorded in `docs/ASSUMPTIONS.md` under
`bin.*`, and not one has a citation.

## ThingSpeak fields — this mapping is INVENTED

The paper never states which field carries what, so this is ours and uncited:

| Field | Meaning |
|---|---|
| `field1` / `field2` / `field3` | ultrasonic distances, left / centre / right, **cm** |
| `field4` | load-cell weight, **kg** |
| `field5` | `Fv`, fused fill percentage |
| `field6` | state code: 0 GREEN, 1 YELLOW, 2 RED |

`tools\thingspeak_fetch.py` reads fields 1–4 and **ignores 5 and 6 on purpose** — it re-derives
`Fv` and the state with the Python rule, because the point is to check that the two agree rather
than to take the firmware's word for it.

Posting interval is 5 minutes (`POST_INTERVAL_MS`). ThingSpeak's free tier allows one update
every 15 seconds, so this is well inside it.

## If you flash it and it misbehaves

- `all 3 ultrasonic sensors timed out` on the serial monitor at 115200 baud: ECHO wiring, missing
  common ground, or 5 V on a 3.3 V pin.
- Wi-Fi never associates: the ESP8266 is **2.4 GHz only**. A 5 GHz-only SSID looks exactly like a
  wrong password.
- Weight reads nonsense: the HX711 needs calibrating against a known mass. The procedure is in
  the comments in `config.example.h`; the shipped factor is a dummy, not a starting point.
- A board that will not boot after wiring the HX711: GPIO0 (`D3`) and GPIO2 (`D4`) are boot-mode
  pins. Move them in `config.h`.
