/*
 * ===========================================================================================
 *  NOT VERIFIED: this firmware has NEVER been compiled and has NEVER been flashed.
 * ===========================================================================================
 *
 *  No arduino-cli was available in the session that wrote this file, no ESP8266 toolchain was
 *  installed, no board was connected, and no ThingSpeak channel has ever been contacted. Nothing
 *  below is a claim that this code builds, runs, or produces correct readings on real hardware.
 *  Treat every line as UNTESTED until you compile it and flash it yourself.
 *
 *  What *is* checked, by `tests/test_firmware_crosscheck.py`:
 *
 *    1. `src/smartwaste/firmware_reference.py` is a Python transcription of the arithmetic in
 *       this file, and it agrees with `src/smartwaste/fusion.py` on the RED/YELLOW/GREEN state
 *       for thousands of readings, every threshold boundary included.
 *    2. The six #define'd constants below equal the corresponding values in
 *       `config/default.yaml`.
 *
 *  Neither of those runs a C++ compiler. They verify the rule we *wrote down* and the numbers we
 *  wrote down -- not this translation unit. See `docs/PHASE5_RESULTS.md`.
 *
 * -------------------------------------------------------------------------------------------
 *  Hardware (the paper's single physical bin; SPEC section 2)
 * -------------------------------------------------------------------------------------------
 *  NodeMCU (ESP8266), 3 x HC-SR04 ultrasonic on the upper inner surface (left / centre / right),
 *  1 x HX711 + load cell under the bin. Inner side walls angled to push waste to the centre,
 *  which is why the centre sensor is the one a single-sensor design would use -- and why Phase 1
 *  aimed its `uneven_center_peak` skew there.
 *
 *  Pins, Wi-Fi credentials and the ThingSpeak WRITE key live in `config.h`, which is GITIGNORED.
 *  Copy `config.example.h` to `config.h` and fill it in. NEVER commit `config.h`.
 *
 *  Libraries to install (Arduino IDE / arduino-cli), neither of which has been installed here:
 *    - esp8266 board support  (Boards Manager URL in README.md)
 *    - "HX711 Arduino Library" by Bogdan Necula (bogde/HX711)
 *
 * -------------------------------------------------------------------------------------------
 *  The fusion rule -- identical to src/smartwaste/fusion.py, which is identical to the paper
 * -------------------------------------------------------------------------------------------
 *    d_bar   = mean of the valid ultrasonic distances
 *    Fv      = 100 * (1 - d_bar / HEIGHT_CM), clipped to [0, 100]
 *    overload= W >= ALPHA * W_MAX_KG
 *    state   = RED    if overload or Fv >= RED_THRESHOLD_PCT
 *              YELLOW if Fv >= YELLOW_THRESHOLD_PCT
 *              GREEN  otherwise
 *
 *  Clamp order matters and is deliberate: each distance is clamped to
 *  [MIN_DISTANCE_CM, HEIGHT_CM] FIRST, then averaged, then Fv is clipped to [0, 100]. The Python
 *  side does the same two-stage clamp (`noise.SensorArray` clamps the reading,
 *  `fusion.fill_percent` clips Fv), and swapping the order would change the answer near a full
 *  or empty bin.
 *
 *  A sensor that times out is DROPPED from the mean, not averaged in as zero. Averaging in a
 *  zero would read as "the waste is touching the sensor" -- the exact opposite of a missing
 *  reading -- and would drive a silent false RED. With all three dead, nothing is posted.
 *
 *  Road/field caveat: ALPHA = 0.85 is NOT from the paper. The paper defines
 *  W_th = alpha * W_max with 0 < alpha < 1 and never states alpha. Every constant below is
 *  recorded in docs/ASSUMPTIONS.md under `bin.*`, and none has a citation.
 * ===========================================================================================
 */

#include <ESP8266WiFi.h>
#include <ESP8266HTTPClient.h>
#include <WiFiClient.h>
#include <HX711.h>

#include "config.h"

/* --- bin geometry and state thresholds -----------------------------------------------------
 * These six MUST match config/default.yaml. tests/test_firmware_crosscheck.py parses them out
 * of this file with a regex and fails if they drift.
 */
#define HEIGHT_CM             100.0f   /* bin.height_cm            -- inner height H, cm      */
#define W_MAX_KG              50.0f    /* bin.w_max_kg             -- capacity W_max, kg      */
#define ALPHA                 0.85f    /* bin.alpha                -- W_th = ALPHA * W_MAX_KG */
#define YELLOW_THRESHOLD_PCT  60.0f    /* bin.yellow_threshold_pct -- paper's 60              */
#define RED_THRESHOLD_PCT     80.0f    /* bin.red_threshold_pct    -- paper's 80              */
#define MIN_DISTANCE_CM       2.0f     /* sensors.min_distance_cm  -- HC-SR04 minimum range   */

/* --- sensor array ------------------------------------------------------------------------- */
#define N_ULTRASONIC          3        /* sensors.n_ultrasonic                                */

/* Microseconds of round-trip echo per centimetre: sound travels ~343 m/s, the pulse covers the
 * distance twice, so 1 cm ~ 58 us. Standard HC-SR04 arithmetic, not a project constant. */
#define US_PER_CM_ROUND_TRIP  58.0f

/* pulseIn timeout. 100 cm round trip is ~5800 us; 25000 us leaves generous headroom and still
 * bounds the worst case at 3 x 25 ms per cycle. */
#define ECHO_TIMEOUT_US       25000UL

/* Returned by readDistanceCm() when the echo never came back. Negative, so it can never be
 * confused with a real distance. */
#define INVALID_DISTANCE_CM   (-1.0f)

/* --- state codes, as posted to ThingSpeak field6 ------------------------------------------ */
#define STATE_GREEN           0
#define STATE_YELLOW          1
#define STATE_RED             2

/* --- timing -------------------------------------------------------------------------------- */
#define POST_INTERVAL_MS      300000UL /* 5 minutes. ThingSpeak's free tier allows 1 post / 15 s */
#define WIFI_TIMEOUT_MS       20000UL

static const uint8_t TRIG_PINS[N_ULTRASONIC] = { TRIG_PIN_LEFT, TRIG_PIN_CENTER, TRIG_PIN_RIGHT };
static const uint8_t ECHO_PINS[N_ULTRASONIC] = { ECHO_PIN_LEFT, ECHO_PIN_CENTER, ECHO_PIN_RIGHT };

static HX711 scale;

/* ============================== the fusion rule ========================================== */

/* One HC-SR04 ping, clamped to the sensor's usable range.
 *
 * Returns INVALID_DISTANCE_CM on timeout so the caller can drop this sensor from the mean.
 */
static float readDistanceCm(uint8_t trigPin, uint8_t echoPin) {
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);

  unsigned long echoUs = pulseIn(echoPin, HIGH, ECHO_TIMEOUT_US);
  if (echoUs == 0UL) {
    return INVALID_DISTANCE_CM;
  }

  float cm = (float) echoUs / US_PER_CM_ROUND_TRIP;
  /* Clamp FIRST, before any averaging. Mirrors noise.SensorArray on the Python side. */
  if (cm < MIN_DISTANCE_CM) {
    cm = MIN_DISTANCE_CM;
  }
  if (cm > HEIGHT_CM) {
    cm = HEIGHT_CM;
  }
  return cm;
}

/* Mean of the valid distances; INVALID_DISTANCE_CM when every sensor timed out.
 *
 * `nValid` reports how many sensors contributed, so the caller can refuse to post a reading
 * built from nothing.
 */
static float meanValidDistanceCm(const float distances[], uint8_t n, uint8_t *nValid) {
  float sum = 0.0f;
  uint8_t count = 0;
  for (uint8_t i = 0; i < n; i++) {
    if (distances[i] > 0.0f) {
      sum += distances[i];
      count++;
    }
  }
  *nValid = count;
  if (count == 0) {
    return INVALID_DISTANCE_CM;
  }
  return sum / (float) count;
}

/* Fv = 100 * (1 - d_bar / H), clipped to [0, 100]. The paper's equation, verbatim. */
static float fillPercent(float dBarCm) {
  float fv = 100.0f * (1.0f - dBarCm / HEIGHT_CM);
  if (fv < 0.0f) {
    fv = 0.0f;
  }
  if (fv > 100.0f) {
    fv = 100.0f;
  }
  return fv;
}

/* Iw = 1 if W >= alpha * W_max. Note `>=`, not `>`: the boundary is overloaded, and
 * tests/test_fusion.py pins that on the Python side. */
static bool isOverloaded(float weightKg) {
  return weightKg >= (ALPHA * W_MAX_KG);
}

/* The paper's three-state rule. Weight wins: an overloaded bin is RED whatever Fv says. */
static uint8_t classifyState(float fillPct, bool overloaded) {
  if (overloaded || fillPct >= RED_THRESHOLD_PCT) {
    return STATE_RED;
  }
  if (fillPct >= YELLOW_THRESHOLD_PCT) {
    return STATE_YELLOW;
  }
  return STATE_GREEN;
}

static const char *stateName(uint8_t state) {
  switch (state) {
    case STATE_RED:    return "RED";
    case STATE_YELLOW: return "YELLOW";
    default:           return "GREEN";
  }
}

/* ============================== plumbing ================================================== */

static bool connectWiFi() {
  if (WiFi.status() == WL_CONNECTED) {
    return true;
  }
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  unsigned long startedAt = millis();
  while (WiFi.status() != WL_CONNECTED) {
    if (millis() - startedAt > WIFI_TIMEOUT_MS) {
      /* The SSID is printed; the password never is. */
      Serial.print(F("WiFi connect timed out for SSID "));
      Serial.println(WIFI_SSID);
      return false;
    }
    delay(250);
  }
  Serial.print(F("WiFi connected, IP "));
  Serial.println(WiFi.localIP());
  return true;
}

/* Post one reading to ThingSpeak.
 *
 * Field mapping (INVENTED -- the paper never states one; see src/smartwaste/thingspeak.py):
 *   field1..3 distances cm, field4 weight kg, field5 Fv, field6 state code.
 *
 * The WRITE key is in the URL, so the URL is NEVER printed. Only the HTTP status is.
 */
static bool postToThingSpeak(const float distances[], uint8_t n, float weightKg,
                             float fillPct, uint8_t state) {
  WiFiClient client;
  HTTPClient http;

  String url = String("http://api.thingspeak.com/update?api_key=") + THINGSPEAK_WRITE_API_KEY;
  for (uint8_t i = 0; i < n; i++) {
    url += "&field" + String(i + 1) + "=" + String(distances[i], 2);
  }
  url += "&field4=" + String(weightKg, 3);
  url += "&field5=" + String(fillPct, 2);
  url += "&field6=" + String(state);

  if (!http.begin(client, url)) {
    Serial.println(F("http.begin failed"));
    return false;
  }
  int status = http.GET();
  http.end();

  /* Deliberately status-only: printing `url` here would put the write key in the serial log. */
  Serial.print(F("ThingSpeak HTTP status "));
  Serial.println(status);
  return status == 200;
}

void setup() {
  Serial.begin(115200);
  delay(100);
  Serial.println();
  Serial.println(F("nodemcu_smartbin -- NOT VERIFIED, never compiled or flashed by its author"));

  for (uint8_t i = 0; i < N_ULTRASONIC; i++) {
    pinMode(TRIG_PINS[i], OUTPUT);
    digitalWrite(TRIG_PINS[i], LOW);
    pinMode(ECHO_PINS[i], INPUT);
  }

  scale.begin(HX711_DOUT_PIN, HX711_SCK_PIN);
  /* HX711_CALIBRATION_FACTOR and HX711_TARE_OFFSET are per-build and live in config.h: a load
   * cell has to be calibrated against known masses, and no two rigs share a factor. */
  scale.set_scale(HX711_CALIBRATION_FACTOR);
  scale.set_offset(HX711_TARE_OFFSET);

  connectWiFi();
}

void loop() {
  float distances[N_ULTRASONIC];
  for (uint8_t i = 0; i < N_ULTRASONIC; i++) {
    distances[i] = readDistanceCm(TRIG_PINS[i], ECHO_PINS[i]);
    delay(60); /* HC-SR04 datasheet: >= 60 ms between pings or the echoes interfere. */
  }

  uint8_t nValid = 0;
  float dBar = meanValidDistanceCm(distances, N_ULTRASONIC, &nValid);
  if (nValid == 0) {
    /* No ultrasonic reading at all. Posting a fabricated Fv would be worse than posting
     * nothing, because downstream it is indistinguishable from a real measurement. */
    Serial.println(F("all 3 ultrasonic sensors timed out; nothing posted"));
    delay(POST_INTERVAL_MS);
    return;
  }

  float weightKg = scale.get_units(10);
  if (weightKg < 0.0f) {
    weightKg = 0.0f; /* A negative mass is a tare error, not a measurement. */
  }

  float fillPct = fillPercent(dBar);
  bool overloaded = isOverloaded(weightKg);
  uint8_t state = classifyState(fillPct, overloaded);

  Serial.print(F("d_bar "));
  Serial.print(dBar, 2);
  Serial.print(F(" cm ("));
  Serial.print(nValid);
  Serial.print(F("/3 sensors)  W "));
  Serial.print(weightKg, 3);
  Serial.print(F(" kg  Fv "));
  Serial.print(fillPct, 2);
  Serial.print(F("%  overload "));
  Serial.print(overloaded ? F("yes") : F("no"));
  Serial.print(F("  state "));
  Serial.println(stateName(state));

  if (connectWiFi()) {
    postToThingSpeak(distances, N_ULTRASONIC, weightKg, fillPct, state);
  }

  delay(POST_INTERVAL_MS);
}
