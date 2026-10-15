/*
 * ===========================================================================================
 *  config.example.h -- DUMMY VALUES ONLY. This file is tracked in git.
 * ===========================================================================================
 *
 *  Copy it next to nodemcu_smartbin.ino as `config.h` and fill in your real values:
 *
 *      copy firmware\nodemcu_smartbin\config.example.h firmware\nodemcu_smartbin\config.h
 *
 *  `config.h` is GITIGNORED (see .gitignore) and must NEVER be committed. It holds your Wi-Fi
 *  password and your ThingSpeak WRITE key -- a write key lets anyone push data into your
 *  channel, so treat it exactly like a password.
 *
 *  `tests/test_no_secrets_tracked.py` fails the test suite if `config.h` ever becomes a tracked
 *  file, or if a tracked file in here stops looking like a placeholder. That check is the
 *  safety net; not committing the file is the actual protection.
 *
 *  If you think you have committed a real key: revoke it in ThingSpeak first (generate a new
 *  one), then worry about the git history. A key that is still valid in a public repo is the
 *  problem; a key in history that no longer works is not.
 * ===========================================================================================
 */

#ifndef SMARTBIN_CONFIG_H
#define SMARTBIN_CONFIG_H

/* --- Wi-Fi --------------------------------------------------------------------------------
 * The ESP8266 is 2.4 GHz only. A 5 GHz-only SSID will never associate, which looks exactly
 * like a wrong password in the serial log.
 */
#define WIFI_SSID                   "YOUR_WIFI_SSID"
#define WIFI_PASSWORD               "YOUR_WIFI_PASSWORD"

/* --- ThingSpeak ---------------------------------------------------------------------------
 * Channel -> API Keys -> "Write API Key". 16 characters.
 *
 * The READ key is NOT needed here -- the sketch only writes. The read key belongs in `.env`,
 * for tools\thingspeak_fetch.py. Two different keys, two different files, neither tracked.
 */
#define THINGSPEAK_WRITE_API_KEY    "0000000000000000"

/* --- HC-SR04 pins (NodeMCU D-labels) ------------------------------------------------------
 * Avoid D0/GPIO16 for ECHO: it has no interrupt and behaves differently from the other pins.
 * Avoid D3/GPIO0 and D4/GPIO2 for TRIG if you can: both affect boot mode.
 *
 *   left   TRIG D1 / GPIO5    ECHO D2 / GPIO4
 *   centre TRIG D5 / GPIO14   ECHO D6 / GPIO12
 *   right  TRIG D7 / GPIO13   ECHO D8 / GPIO15
 *
 * HC-SR04 ECHO idles at 5 V and the ESP8266 is a 3.3 V part. Use a divider (e.g. 1k / 2k) on
 * each ECHO line, or a 3.3 V-tolerant module. UNTESTED, like everything else here.
 */
#define TRIG_PIN_LEFT               D1
#define ECHO_PIN_LEFT               D2
#define TRIG_PIN_CENTER             D5
#define ECHO_PIN_CENTER             D6
#define TRIG_PIN_RIGHT              D7
#define ECHO_PIN_RIGHT              D8

/* --- HX711 load cell ----------------------------------------------------------------------
 * DOUT and SCK are plain GPIO; the HX711 protocol is bit-banged by the library.
 */
#define HX711_DOUT_PIN              D3
#define HX711_SCK_PIN               D4

/* Per-rig, and there is no sensible default: calibrate against a known mass.
 *
 *   1. Flash with CALIBRATION_FACTOR 1.0 and TARE_OFFSET 0.
 *   2. With the bin empty, note the raw average -> that is your TARE_OFFSET.
 *   3. Put a known mass on it, note the raw average, and
 *      CALIBRATION_FACTOR = (raw - TARE_OFFSET) / known_kg.
 *
 * The placeholder below is a dummy, not a starting point.
 */
#define HX711_CALIBRATION_FACTOR    1.0f
#define HX711_TARE_OFFSET           0L

#endif /* SMARTBIN_CONFIG_H */
