#include <Arduino.h>
#include <string.h>
#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "driver/i2s.h"
#include <WiFi.h>
#include <WiFiUdp.h>

#include "prahari_config.h"
#include "prahari_logmel.h"
#include "prahari_model.h"
#include "prahari_test_clip.h"
#include "prahari_link.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_profiler_interface.h"

#define CAPS (MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT)
#define PDM_CLK_PIN            42
#define PDM_DATA_PIN           41
#define PRIVACY_LED_PIN        21
#define PRIVACY_LED_ACTIVE     LOW
#define ARENA_BYTES            (36 * 1024)
#define INFER_EVERY_FRAMES     12
#define TRIGGER_THRESHOLD      0.85f
#define SMOOTH_N               2
#define REFRACTORY_MS          1500
#define MIC_GAIN               8.0f
#define STATIC_DRAM_BYTES      63480
#define RAM_BUDGET_BYTES       (256 * 1024)
#define VAD_ENABLED            1
#define VAD_TRIGGER_RATIO      3.0f
#define VAD_MIN_RMS            30.0f
#define VAD_HANGOVER_MS        700
#define VAD_OPEN_HOPS          3
#define INFER_SLOW_FRAMES      96
#define BURST_ENABLED_DEFAULT  1
#define BURST_ARM_SCORE        0.50f
#define BURST_EVERY_FRAMES     4
#define BURST_LEN_FRAMES       50
#define STATUS_EVERY           10
#define WIFI_SSID     "YOUR_WIFI_SSID"
#define WIFI_PASS     "YOUR_WIFI_PASSWORD"
#define PRAHARI_SERVER_MODE 0

#if PRAHARI_SERVER_MODE == 1
  #define SERVER_IP     "YOUR_SERVER_PUBLIC_IP"
  #define PRAHARI_USE_DISCOVERY 0
#else
  #define SERVER_IP     "192.168.1.100"
  #define PRAHARI_USE_DISCOVERY 1
#endif

#define SERVER_PORT   5005
#define DISCOVERY_PORT 5006
#define ADPCM_BLOCK_SAMPLES    480
#define ADPCM_BLOCK_MS         (ADPCM_BLOCK_SAMPLES / (PRAHARI_SAMPLE_RATE / 1000))
#define ADPCM_BLOCK_BYTES      (4 + ADPCM_BLOCK_SAMPLES / 2)
#define PREROLL_MS             1500
#define PREROLL_BLOCKS         (PREROLL_MS / ADPCM_BLOCK_MS)
#define PREROLL_SAMPLES        (PREROLL_MS * PRAHARI_SAMPLE_RATE / 1000)
#define CAPTURE_MS             8000
#define CAPTURE_TEST_MS        CAPTURE_MS
#define ENDPOINT_TRACE_DEFAULT 0
#define ENDPOINT_ENABLED       1
#define ENDPOINT_MIN_MS        1000
#define ENDPOINT_SILENCE_MS    800
#define ENDPOINT_PEAK_FRAC     0.18f
#define ENDPOINT_FLOOR_RATIO   1.4f
#define ENDPOINT_NOISE_K       0.0f
#define ENDPOINT_NOISE_BLOCKS  33
#define PROF_MAX_EVENTS 48

class PrahariProfiler : public tflite::MicroProfilerInterface {
 public:
  uint32_t BeginEvent(const char *tag) override {
    if (!armed_ || n_ >= PROF_MAX_EVENTS) return kNone;
    uint32_t h = n_++;
    tags_[h] = tag;
    t0_[h]   = esp_timer_get_time();
    return h;
  }
  void EndEvent(uint32_t h) override {
    if (h >= PROF_MAX_EVENTS) return;
    us_[h] += (uint32_t)(esp_timer_get_time() - t0_[h]);
    if (h + 1 > n_ops_) n_ops_ = h + 1;
  }

  void arm()       { armed_ = true; }
  void disarm()    { armed_ = false; }
  void begin_run() { n_ = 0; }
  void end_run()   { runs_++; }
  void clear() {
    for (uint32_t i = 0; i < PROF_MAX_EVENTS; i++) { us_[i] = 0; tags_[i] = nullptr; }
    n_ = 0; n_ops_ = 0; runs_ = 0;
  }

  void report() const {
    if (!runs_ || !n_ops_) { Serial.println(F("profiler: nothing recorded")); return; }
    uint32_t total = 0;
    for (uint32_t i = 0; i < n_ops_; i++) total += us_[i];
    Serial.println();
    Serial.printf("per-operator breakdown, mean of %lu runs\n", (unsigned long)runs_);
    Serial.println(F("  #  operator                  mean us    %   bar"));
    Serial.println(F("  -- ---------------------- ---------- ----- ------------------------"));
    for (uint32_t i = 0; i < n_ops_; i++) {
      float mean = (float)us_[i] / (float)runs_;
      float pct  = total ? 100.0f * (float)us_[i] / (float)total : 0.0f;
      char bar[25];
      int  nb = (int)(pct / 4.0f + 0.5f);
      if (nb > 24) nb = 24;
      for (int b = 0; b < nb; b++) bar[b] = '#';
      bar[nb] = 0;
      Serial.printf("  %2lu %-22s %10.1f %5.1f  %s\n",
                    (unsigned long)i, tags_[i] ? tags_[i] : "?", mean, pct, bar);
    }
    Serial.printf("  -- total                  %10.1f us  (%.2f ms)\n",
                  (float)total / (float)runs_, (float)total / (float)runs_ / 1000.0f);
    Serial.println();
  }

 private:
  static const uint32_t kNone = 0xFFFFFFFFu;
  const char *tags_[PROF_MAX_EVENTS] = {};
  int64_t     t0_[PROF_MAX_EVENTS]   = {};
  uint32_t    us_[PROF_MAX_EVENTS]   = {};
  uint32_t    n_ = 0, n_ops_ = 0, runs_ = 0;
  bool        armed_ = false;
};

static PrahariProfiler g_prof;

static uint8_t *g_arena = nullptr;
static tflite::MicroInterpreter *g_interp = nullptr;
static TfLiteTensor *g_in = nullptr, *g_out = nullptr;

static bool        g_ready = false;
static const char *g_fail  = nullptr;
static size_t      g_arena_used = 0;

struct ClipResult {
  bool ran = false, invoke_ok = false, match = false;
  int raw = 0, expect_raw = 0;
  float score = 0.0f, expect_score = 0.0f;
  float ms_front = 0.0f, ms_infer = 0.0f;
};
static ClipResult g_pos, g_neg;
static float g_clip_rms = 0.0f;

static int8_t   g_feat[PRAHARI_FEATURE_COUNT];

static uint16_t g_feat_head = 0;

static int16_t  g_win[PRAHARI_WIN_SAMPLES];
static float    g_raw_rms = 0.0f;
static int16_t  g_raw_min = 0, g_raw_max = 0;
static float    g_score_max = 0.0f;
static float    g_level_peak = 0.0f;

static float    g_noise_floor = 8.0f * MIC_GAIN;
static uint32_t g_gate_open_until = 0;
static uint32_t g_hops_total = 0;
static uint32_t g_hops_gated = 0;
static uint32_t g_infer_runs = 0;
static bool     g_burst_on    = BURST_ENABLED_DEFAULT;
static float    g_burst_arm   = BURST_ARM_SCORE;
static uint32_t g_burst_left  = 0;
static uint32_t g_burst_extra = 0;
static uint32_t g_bursts      = 0;

static float    g_thresh   = TRIGGER_THRESHOLD;
static float    g_mic_gain = MIC_GAIN;
static uint32_t g_capture_ms = CAPTURE_TEST_MS;
static bool     g_no_endpoint = false;
static bool     g_ep_trace   = ENDPOINT_TRACE_DEFAULT;
static uint32_t g_gate_opens = 0;
static bool     g_quiet = false;
static uint32_t g_quiet_t0 = 0, g_quiet_opens0 = 0, g_quiet_hops0 = 0,
                g_quiet_runs0 = 0;

static const int16_t ima_step[89] = {
      7,     8,     9,    10,    11,    12,    13,    14,    16,    17,
     19,    21,    23,    25,    28,    31,    34,    37,    41,    45,
     50,    55,    60,    66,    73,    80,    88,    97,   107,   118,
    130,   143,   157,   173,   190,   209,   230,   253,   279,   307,
    337,   371,   408,   449,   494,   544,   598,   658,   724,   796,
    876,   963,  1060,  1166,  1282,  1411,  1552,  1707,  1878,  2066,
   2272,  2499,  2749,  3024,  3327,  3660,  4026,  4428,  4871,  5358,
   5894,  6484,  7132,  7845,  8630,  9493, 10442, 11487, 12635, 13899,
  15289, 16818, 18500, 20350, 22385, 24623, 27086, 29794, 32767 };
static const int8_t ima_index_adj[16] = {
  -1, -1, -1, -1, 2, 4, 6, 8, -1, -1, -1, -1, 2, 4, 6, 8 };

typedef struct { int32_t pred; int8_t idx; } adpcm_t;

static inline uint8_t adpcm_code(adpcm_t *e, int16_t sample) {
  int step = ima_step[e->idx];
  int diff = (int)sample - e->pred;
  uint8_t code = 0;
  if (diff < 0) { code = 8; diff = -diff; }
  int diffq = step >> 3;
  if (diff >= step) { code |= 4; diff -= step; diffq += step; }
  step >>= 1;
  if (diff >= step) { code |= 2; diff -= step; diffq += step; }
  step >>= 1;
  if (diff >= step) { code |= 1;               diffq += step; }
  e->pred += (code & 8) ? -diffq : diffq;
  if (e->pred >  32767) e->pred =  32767;
  if (e->pred < -32768) e->pred = -32768;
  e->idx = (int8_t)(e->idx + ima_index_adj[code]);
  if (e->idx <  0) e->idx =  0;
  if (e->idx > 88) e->idx = 88;
  return code;
}

static inline void adpcm_reset(adpcm_t *e) { e->pred = 0; e->idx = 0; }

static uint8_t g_blk_seq = 0;

static inline void adpcm_stamp(const adpcm_t *e, uint8_t *blk) {
  blk[0] = (uint8_t)(e->pred & 0xFF);
  blk[1] = (uint8_t)((e->pred >> 8) & 0xFF);
  blk[2] = (uint8_t)e->idx;
  blk[3] = g_blk_seq++;
}

static uint8_t  g_pre[PREROLL_BLOCKS * ADPCM_BLOCK_BYTES];
static uint16_t g_pre_blk = 0;
static uint16_t g_pre_off = 0;
static bool     g_pre_full = false;
static adpcm_t  g_pre_enc;

static bool      g_wifi = false;
static IPAddress g_srv_ip;
static uint16_t  g_srv_port = SERVER_PORT;
static bool      g_discovered = false;
static uint32_t g_streams = 0;
static size_t   g_free_after_wifi = 0;
static uint32_t g_frames = 0;
static uint32_t g_infers = 0;
static float    g_hist[SMOOTH_N];
static int      g_hist_n = 0;
static uint32_t g_last_trigger_ms = 0;
static uint32_t g_triggers = 0;
static float    g_live_rms = 0.0f;
static float    g_last_score = 0.0f, g_last_smooth = 0.0f;
static float    g_ms_frontend = 0.0f, g_ms_infer = 0.0f;

static float g_dc_x1 = 0.0f, g_dc_y1 = 0.0f;

static void print_ram_budget() {
  size_t total = heap_caps_get_total_size(CAPS);
  size_t minfree = heap_caps_get_minimum_free_size(CAPS);
  size_t heap_peak = total - minfree;
  size_t used = heap_peak + STATIC_DRAM_BYTES;
  Serial.println(F("--- RAM, against the 256 KB ceiling ---------------------------"));
  Serial.printf("  static DRAM (.data+.bss)  %7u B  %6.1f KB   [from firmware.elf]\n",
                (unsigned)STATIC_DRAM_BYTES, STATIC_DRAM_BYTES / 1024.0);
  Serial.printf("  heap PEAK ever used       %7u B  %6.1f KB   [total %u - min-free %u]\n",
                (unsigned)heap_peak, heap_peak / 1024.0,
                (unsigned)total, (unsigned)minfree);
  Serial.printf("  TOTAL INTERNAL SRAM       %7u B  %6.1f KB   = %.1f%% of 256 KB\n",
                (unsigned)used, used / 1024.0,
                100.0 * used / RAM_BUDGET_BYTES);
  Serial.printf("  headroom left             %7d B  %6.1f KB\n",
                (int)RAM_BUDGET_BYTES - (int)used,
                ((int)RAM_BUDGET_BYTES - (int)used) / 1024.0);
  Serial.printf("  of which: arena %u B, model %u B flash, pre-roll %u B\n",
                (unsigned)g_arena_used, (unsigned)prahari_model_tflite_len,
                (unsigned)sizeof(g_pre));
  Serial.println(F("--------------------------------------------------------------"));
}

static void rule() {
  Serial.println(F("---------------------------------------------------------------"));
}

static float rms_i16(const int16_t *x, int n) {
  double acc = 0;
  for (int i = 0; i < n; i++) acc += (double)x[i] * x[i];
  return (float)sqrt(acc / n);
}

static float hp_rms(const int16_t *x, int n) {
  static const float b0 = 0.945976379f, b1 = -1.891952758f, b2 = 0.945976379f;
  static const float a1 = -1.889032127f, a2 = 0.894873389f;
  static float x1 = 0.0f, x2 = 0.0f, y1 = 0.0f, y2 = 0.0f;
  float acc = 0.0f;
  for (int i = 0; i < n; i++) {
    float v = (float)x[i];
    float y = b0 * v + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
    x2 = x1; x1 = v; y2 = y1; y1 = y;
    acc += y * y;
  }
  return sqrtf(acc / (float)n);
}

static void quantise_row(const float *mel, int8_t *row) {
  for (int i = 0; i < PRAHARI_MEL_BINS; i++) {
    float q = mel[i] / PRAHARI_MODEL_IN_SCALE + (float)PRAHARI_MODEL_IN_ZP;
    int v = (int)(q < 0 ? q - 0.5f : q + 0.5f);
    if (v < -128) v = -128;
    if (v > 127) v = 127;
    row[i] = (int8_t)v;
  }
}

static float score_from_output() {
  int raw = g_out->data.int8[0];
  return (raw - PRAHARI_MODEL_OUT_ZP) * PRAHARI_MODEL_OUT_SCALE;
}

static void run_clip(ClipResult &r, const int16_t *pcm,
                     int expect_raw, float expect_score) {
  r.expect_raw = expect_raw; r.expect_score = expect_score;
  int n_frames = 1 + (PRAHARI_TEST_CLIP_LEN - PRAHARI_WIN_SAMPLES)
                     / PRAHARI_HOP_SAMPLES;
  if (n_frames > PRAHARI_CTX_FRAMES) n_frames = PRAHARI_CTX_FRAMES;

  float mel[PRAHARI_MEL_BINS];
  int8_t *dst = g_in->data.int8;

  int64_t t0 = esp_timer_get_time();
  for (int f = 0; f < n_frames; f++) {
    prahari_logmel_frame(pcm + f * PRAHARI_HOP_SAMPLES, mel);
    quantise_row(mel, dst + f * PRAHARI_MEL_BINS);
  }
  int64_t t1 = esp_timer_get_time();
  TfLiteStatus st = g_interp->Invoke();
  int64_t t2 = esp_timer_get_time();

  r.ms_front = (t1 - t0) / 1000.0f;
  r.ms_infer = (t2 - t1) / 1000.0f;
  r.ran = true;
  if (st != kTfLiteOk) return;
  r.invoke_ok = true;
  r.raw   = g_out->data.int8[0];
  r.score = score_from_output();
  r.match = (r.raw - r.expect_raw <= 1 && r.expect_raw - r.raw <= 1);
}

static void print_clip(const char *label, const ClipResult &r) {
  if (!r.ran)       { Serial.printf("%-9s did not run\n", label);   return; }
  if (!r.invoke_ok) { Serial.printf("%-9s INVOKE FAILED\n", label); return; }
  Serial.printf("%-9s raw %4d (expect %4d)  score %.6f (expect %.6f)  %s\n",
                label, r.raw, r.expect_raw, r.score, r.expect_score,
                !r.match ? "*** MISMATCH ***"
                : (r.raw == r.expect_raw) ? "MATCH"
                : "MATCH (1 LSB - device kernel rounding)");
  Serial.printf("          front-end %6.2f ms (%.3f ms/frame)   inference %6.2f ms\n",
                r.ms_front, r.ms_front / PRAHARI_CTX_FRAMES, r.ms_infer);
}

static bool init_model() {
  const tflite::Model *model = tflite::GetModel(prahari_model_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    g_fail = "schema version mismatch between model and TFLM runtime";
    return false;
  }
  static tflite::MicroMutableOpResolver<10> resolver;
  resolver.AddMul();
  resolver.AddAdd();
  resolver.AddConv2D();
  resolver.AddDepthwiseConv2D();
  resolver.AddMaxPool2D();
  resolver.AddAveragePool2D();
  resolver.AddReshape();
  resolver.AddMean();
  resolver.AddFullyConnected();
  resolver.AddLogistic();

  g_arena = (uint8_t *)heap_caps_aligned_alloc(16, ARENA_BYTES, CAPS);
  if (!g_arena) { g_fail = "could not allocate the tensor arena"; return false; }

  static tflite::MicroInterpreter interp(model, resolver, g_arena, ARENA_BYTES,
                                         nullptr, &g_prof);
  g_interp = &interp;
  if (interp.AllocateTensors() != kTfLiteOk) {
    g_fail = "AllocateTensors failed - see the TFLM message above";
    return false;
  }
  g_arena_used = interp.arena_used_bytes();
  g_in  = interp.input(0);
  g_out = interp.output(0);
  return true;
}

static void init_privacy_led(void) {
  pinMode(PRIVACY_LED_PIN, OUTPUT);
  digitalWrite(PRIVACY_LED_PIN, !PRIVACY_LED_ACTIVE);
}

static bool init_microphone() {
  i2s_config_t cfg = {};
  cfg.mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX | I2S_MODE_PDM);
  cfg.sample_rate = PRAHARI_SAMPLE_RATE;
  cfg.bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT;
  cfg.channel_format = I2S_CHANNEL_FMT_ONLY_LEFT;
  cfg.communication_format = I2S_COMM_FORMAT_STAND_I2S;
  cfg.intr_alloc_flags = ESP_INTR_FLAG_LEVEL1;
  cfg.dma_buf_count = 12;
  cfg.dma_buf_len = 256;
  cfg.use_apll = false;
  cfg.tx_desc_auto_clear = false;
  cfg.fixed_mclk = 0;

  if (i2s_driver_install(I2S_NUM_0, &cfg, 0, NULL) != ESP_OK) {
    g_fail = "i2s_driver_install failed"; return false;
  }
  i2s_pin_config_t pins = {};
  pins.mck_io_num   = I2S_PIN_NO_CHANGE;
  pins.bck_io_num   = I2S_PIN_NO_CHANGE;
  pins.ws_io_num    = PDM_CLK_PIN;
  pins.data_out_num = I2S_PIN_NO_CHANGE;
  pins.data_in_num  = PDM_DATA_PIN;
  if (i2s_set_pin(I2S_NUM_0, &pins) != ESP_OK) {
    g_fail = "i2s_set_pin failed"; return false;
  }
  i2s_set_clk(I2S_NUM_0, PRAHARI_SAMPLE_RATE,
              I2S_BITS_PER_SAMPLE_16BIT, I2S_CHANNEL_MONO);
  i2s_zero_dma_buffer(I2S_NUM_0);
  return true;
}

static bool read_hop(int16_t *dst);
static bool discover_server(void);
static void link_start(void);

static void wifi_begin() {
  Serial.printf("WiFi: connecting to \"%s\" ", WIFI_SSID);
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  uint32_t t0 = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - t0 < 15000) {
    delay(250); Serial.print('.');
  }
  Serial.println();
  g_wifi = (WiFi.status() == WL_CONNECTED);
  if (g_wifi) {
    Serial.printf("WiFi: connected, IP %s, %lu ms\n",
                  WiFi.localIP().toString().c_str(),
                  (unsigned long)(millis() - t0));
  } else {
    Serial.println(F("WiFi: FAILED - detection still runs, but nothing is"));
    Serial.println(F("      streamed. Check WIFI_SSID / WIFI_PASS."));
  }
  if (g_wifi) {
    g_discovered = discover_server();
    if (!g_discovered) {
      g_srv_ip.fromString(SERVER_IP);
      g_srv_port = SERVER_PORT;
      Serial.printf("discovery: no answer - falling back to %s:%d\n",
                    SERVER_IP, SERVER_PORT);
      Serial.println(F("  (is server.py running? is the laptop on THIS"));
      Serial.println(F("   network? is Windows Firewall allowing python?)"));
    }
  }
  if (g_wifi) link_start();
  g_free_after_wifi = heap_caps_get_free_size(CAPS);
  Serial.printf("free SRAM after WiFi: %u B (%.1f KB)\n",
                (unsigned)g_free_after_wifi, g_free_after_wifi / 1024.0);
}

static bool discover_server() {
#if !PRAHARI_USE_DISCOVERY
  Serial.printf("discovery: off (public server %s:%d)\n",
                SERVER_IP, SERVER_PORT);
  return false;
#else
  WiFiUDP udp;
  if (!udp.begin(0)) return false;
  IPAddress bcast = WiFi.localIP();
  bcast[3] = 255;
  Serial.printf("discovery: asking %s:%d ", bcast.toString().c_str(),
                DISCOVERY_PORT);
  for (int attempt = 0; attempt < 6; attempt++) {
    udp.beginPacket(bcast, DISCOVERY_PORT);
    udp.write((const uint8_t *)"PRAHARI?", 8);
    udp.endPacket();
    Serial.print('.');
    uint32_t t0 = millis();
    while (millis() - t0 < 400) {
      int sz = udp.parsePacket();
      if (sz > 0) {
        char buf[64] = {0};
        int n = udp.read((uint8_t *)buf, sizeof(buf) - 1);
        if (n >= 8 && strncmp(buf, "PRAHARI!", 8) == 0) {
          g_srv_ip = udp.remoteIP();
          int p = atoi(buf + 8);
          if (p > 0) g_srv_port = (uint16_t)p;
          udp.stop();
          Serial.printf("\ndiscovery: server at %s:%u\n",
                        g_srv_ip.toString().c_str(), g_srv_port);
          return true;
        }
      }
      delay(5);
    }
  }
  udp.stop();
  Serial.println();
  return false;
#endif
}

static bool preroll_push(const int16_t *hop) {
  uint8_t *blk = g_pre + (uint32_t)g_pre_blk * ADPCM_BLOCK_BYTES;
  if (g_pre_off == 0) adpcm_stamp(&g_pre_enc, blk);
  uint8_t *d = blk + 4 + g_pre_off / 2;
  for (int i = 0; i < PRAHARI_HOP_SAMPLES; i += 2) {
    uint8_t lo = adpcm_code(&g_pre_enc, hop[i]);
    uint8_t hi = adpcm_code(&g_pre_enc, hop[i + 1]);
    *d++ = (uint8_t)(lo | (hi << 4));
  }
  g_pre_off += PRAHARI_HOP_SAMPLES;
  if (g_pre_off >= ADPCM_BLOCK_SAMPLES) {
    g_pre_off = 0;
    if (++g_pre_blk >= PREROLL_BLOCKS) { g_pre_blk = 0; g_pre_full = true; }
    return true;
  }
  return false;
}

static char  g_hello[200];
static float g_us_link_avg = 0.0f;

#define STREAM_HOLD_MS 120

static uint32_t link_millis(void) { return millis(); }
static void     link_log(const char *s) { Serial.printf("  %s\n", s); }

static void link_point(void) {
  pl_set_server(g_srv_ip.toString().c_str(), g_srv_port);
}

static void link_start(void) {
  snprintf(g_hello, sizeof(g_hello),
           "{\"event\":\"hello\",\"proto\":2,\"fw\":\"prahari-warm-1\","
           "\"threshold\":%.2f,\"codec\":\"ima-adpcm\",\"block_bytes\":%d,"
           "\"preroll_ms\":%d}\n",
           g_thresh, ADPCM_BLOCK_BYTES, PREROLL_MS);
  pl_init(link_millis, link_log);
  pl_set_hello(g_hello);
  link_point();
  uint32_t t0 = millis();
  while (!pl_is_up() && millis() - t0 < 1500) { pl_service(); delay(10); }
  if (!pl_is_up())
    Serial.println(F("  link: not up yet - will keep trying in the background"));
}

static const char *link_state_name(void) {
  switch (pl_state()) {
    case PL_UP:         return "UP";
    case PL_CONNECTING: return "connecting";
    default:            return "DOWN";
  }
}

static void link_service(void) {
  if (!g_wifi) return;
  int64_t t0 = esp_timer_get_time();
  pl_service();
  float us = (float)(esp_timer_get_time() - t0);
  g_us_link_avg = 0.99f * g_us_link_avg + 0.01f * us;

#if PRAHARI_USE_DISCOVERY
  static uint32_t last_disc = 0;
  if (pl_state() == PL_DOWN && pl_fail_streak() >= 4 &&
      (int32_t)(millis() - g_gate_open_until) > 0 &&
      millis() - last_disc > 60000) {
    last_disc = millis();
    Serial.println(F("  link: server not answering - asking the network where it is"));
    if (discover_server()) link_point();
  }
#endif
}

static void link_report(void) {
  Serial.println();
  Serial.printf(">> link %s to %s:%u\n", link_state_name(),
                g_srv_ip.toString().c_str(), g_srv_port);
  Serial.printf(">> connects %lu, drops %lu, failed attempts %lu, "
                "last connect %lu ms\n",
                (unsigned long)pl_stats.connects,
                (unsigned long)pl_stats.disconnects,
                (unsigned long)pl_stats.connect_fails,
                (unsigned long)pl_stats.last_connect_ms);
  Serial.printf(">> pings %lu, pongs %lu, service cost %.0f us/hop\n\n",
                (unsigned long)pl_stats.pings, (unsigned long)pl_stats.pongs,
                g_us_link_avg);
}

static void stream_event(float score, uint32_t t_trigger) {
  if (!g_wifi) { Serial.println(F("  (no WiFi - not streamed)")); return; }

  const bool warm = pl_is_up();
  if (!warm) {
    Serial.println(F("  uplink: warm link is down - cold connect (fallback)"));
    if (!pl_connect_blocking(1500)) {
      Serial.printf("  uplink: connect to %s:%u FAILED\n",
                    g_srv_ip.toString().c_str(), g_srv_port);
      return;
    }
  }
  const uint32_t t_ready = millis();

  float d = g_hops_total
      ? (float)(g_infer_runs * INFER_EVERY_FRAMES) / (float)g_hops_total : 1.0f;
  if (d > 1.0f) d = 1.0f;

  char hdr[640];
  int n = snprintf(hdr, sizeof(hdr),
      "{\"event\":\"wake\",\"score\":%.3f,\"threshold\":%.2f,"
      "\"sr\":%d,\"preroll_ms\":%d,\"capture_ms\":%d,"
      "\"codec\":\"ima-adpcm\",\"block_samples\":%d,\"block_bytes\":%d,"
      "\"framing\":\"sentinel\",\"proto\":2,\"warm\":%d,"
      "\"connect_ms\":%lu,\"seq\":%lu,"
      "\"arena_bytes\":%u,\"model_bytes\":%u,\"infer_ms\":%.1f,"
      "\"frontend_ms_per_frame\":%.2f,\"free_sram\":%u,"
      "\"mic_rms\":%.0f,\"clip_rms\":%.0f,"
      "\"ram_used_bytes\":%u,\"ram_budget_bytes\":%u,"
      "\"cpu_pct\":%.0f,\"gate_pct\":%.0f}\n",
      score, g_thresh, PRAHARI_SAMPLE_RATE, PREROLL_MS, (int)g_capture_ms,
      ADPCM_BLOCK_SAMPLES, ADPCM_BLOCK_BYTES, warm ? 1 : 0,
      (unsigned long)(t_ready - t_trigger), (unsigned long)g_streams + 1,
      (unsigned)g_arena_used, (unsigned)prahari_model_tflite_len,
      g_ms_infer, g_ms_frontend, (unsigned)heap_caps_get_free_size(CAPS),
      g_level_peak, g_clip_rms,
      (unsigned)(STATIC_DRAM_BYTES + heap_caps_get_total_size(CAPS)
                 - heap_caps_get_minimum_free_size(CAPS)),
      (unsigned)RAM_BUDGET_BYTES,
      g_ms_frontend / 10.0f * 100.0f
        + g_ms_infer / (INFER_EVERY_FRAMES * 10.0f) * 100.0f * d,
      d * 100.0f);
  if (n <= 0 || n >= (int)sizeof(hdr)) { Serial.println(F("  header overflow")); return; }

  uint16_t rd, avail;
  if (g_pre_full) { rd = (uint16_t)((g_pre_blk + 1) % PREROLL_BLOCKS); avail = PREROLL_BLOCKS - 1; }
  else            { rd = 0;                                          avail = g_pre_blk; }

  digitalWrite(PRIVACY_LED_PIN, PRIVACY_LED_ACTIVE);
  if (pl_stream_begin(g_pre, PREROLL_BLOCKS, ADPCM_BLOCK_BYTES, rd, avail,
                      t_trigger, hdr, (size_t)n) != 0) {
    digitalWrite(PRIVACY_LED_PIN, !PRIVACY_LED_ACTIVE);
    Serial.println(F("  uplink: could not send the header - link lost, event dropped"));
    return;
  }
  pl_stream_pump();

  const uint16_t want_blocks = (uint16_t)(g_capture_ms / ADPCM_BLOCK_MS);
  const uint32_t t_cap = millis() + g_capture_ms * 3;
  const uint32_t t_live0 = millis();
  int16_t  hop[PRAHARI_HOP_SAMPLES];
  uint16_t nblk = 0;
  bool     ok = true;
  uint32_t held_ms = 0;
  uint32_t quiet_ms = 0;
  bool     endpointed = false;
  float    peak_rms  = 0.0f;
  float    blk_ms_acc = 0.0f;
  float    nz_hist[ENDPOINT_NOISE_BLOCKS];
  uint8_t  nz_n = 0, nz_i = 0;
  int      blk_hops = 0;
  const float floor_at = g_noise_floor * ENDPOINT_FLOOR_RATIO;

  while (nblk < want_blocks && millis() < t_cap) {
    if (pl_stream_backlog() >= PREROLL_BLOCKS - 1 && held_ms < STREAM_HOLD_MS) {
      uint32_t t = millis();
      if (pl_stream_wait(10) < 0) { ok = false; break; }
      held_ms += millis() - t;
      continue;
    }
    if (!read_hop(hop)) { ok = false; break; }
    const bool block_done = preroll_push(hop);

    float r = hp_rms(hop, PRAHARI_HOP_SAMPLES);
    blk_ms_acc += r * r;
    blk_hops++;

    if (block_done) {
      pl_stream_block_done();
      nblk++;
#if ENDPOINT_ENABLED
      float br = sqrtf(blk_ms_acc / (float)blk_hops);
      if (br > peak_rms) peak_rms = br;
      float quiet_at = peak_rms * ENDPOINT_PEAK_FRAC;
      if (quiet_at < floor_at) quiet_at = floor_at;
      nz_hist[nz_i] = br;
      nz_i = (uint8_t)((nz_i + 1) % ENDPOINT_NOISE_BLOCKS);
      if (nz_n < ENDPOINT_NOISE_BLOCKS) nz_n++;
      {
        float mn = nz_hist[0];
        for (uint8_t k = 1; k < nz_n; k++) if (nz_hist[k] < mn) mn = nz_hist[k];
        if (quiet_at < mn * ENDPOINT_NOISE_K) quiet_at = mn * ENDPOINT_NOISE_K;
      }
      if (br > quiet_at) quiet_ms = 0;
      else               quiet_ms += ADPCM_BLOCK_MS;
      if (!g_no_endpoint &&
          (uint32_t)nblk * ADPCM_BLOCK_MS >= ENDPOINT_MIN_MS &&
          quiet_ms >= ENDPOINT_SILENCE_MS)
        endpointed = true;
      if (g_ep_trace && (nblk % 10 == 0 || endpointed))
        Serial.printf("  vad %5.1fs  block rms %5.0f | quiet below %4.0f "
                      "(peak %4.0f x %.2f, floor %3.0f x %.1f) | silent run "
                      "%4lu/%d ms%s\n",
                      nblk * ADPCM_BLOCK_MS / 1000.0f, br, quiet_at,
                      peak_rms, ENDPOINT_PEAK_FRAC, g_noise_floor,
                      ENDPOINT_FLOOR_RATIO, (unsigned long)quiet_ms,
                      ENDPOINT_SILENCE_MS, endpointed ? "  -> STOP" : "");
#endif
      blk_ms_acc = 0.0f;
      blk_hops = 0;
    }
    if (pl_stream_pump() < 0) { ok = false; break; }
    if (endpointed) break;
  }
  const uint32_t t_live_end = millis();

  char trailer[200];
  snprintf(trailer, sizeof(trailer),
           "{\"ack_ms\":%ld,\"hdr_ms\":%ld,\"dropped\":%lu,\"backlog_max\":%lu,"
           "\"held_ms\":%lu,\"live_ms\":%u,\"endpointed\":%d,\"warm\":%d}",
           (long)pl_stats.s_ack_ms, (long)pl_stats.s_hdr_ms,
           (unsigned long)pl_stats.s_blocks_dropped,
           (unsigned long)pl_stats.s_max_backlog, (unsigned long)held_ms,
           (unsigned)(nblk * ADPCM_BLOCK_MS), endpointed ? 1 : 0, warm ? 1 : 0);
  const bool ended = ok && pl_stream_end(trailer, 2000) == 0;
  const uint32_t t_done = millis();
  digitalWrite(PRIVACY_LED_PIN, !PRIVACY_LED_ACTIVE);
  g_streams++;

  Serial.printf("  uplink: %s link | %d ms pre-roll + %u ms live | %lu blocks sent"
                " | stream #%lu %s\n",
                warm ? "WARM" : "cold", PREROLL_MS,
                (unsigned)(nblk * ADPCM_BLOCK_MS),
                (unsigned long)pl_stats.s_blocks_sent, (unsigned long)g_streams,
                !ended       ? "TRUNCATED (link lost)"
                : endpointed ? "OK (endpointed on silence)"
                             : "OK (hit the ceiling - VAD never saw 800 ms of quiet)");
  Serial.printf("  endpoint: peak block rms %.0f, quiet below %.0f, "
                "floor %.0f\n", peak_rms,
                (peak_rms * ENDPOINT_PEAK_FRAC > floor_at)
                  ? peak_rms * ENDPOINT_PEAK_FRAC : floor_at,
                g_noise_floor);
  Serial.printf("  latency: keyword -> link ready %lu ms (%s) | header in TCP %ld ms"
                " | server ACK %ld ms (round trip) | END sent %lu ms\n",
                (unsigned long)(t_ready - t_trigger), warm ? "already open" : "cold connect",
                (long)pl_stats.s_hdr_ms, (long)pl_stats.s_ack_ms,
                (unsigned long)(t_done - t_trigger));
  Serial.printf("  capture: %u ms audio in %lu ms wall | backlog peak %lu blocks"
                " (%lu ms) | mic held %lu ms | %s\n",
                (unsigned)(nblk * ADPCM_BLOCK_MS),
                (unsigned long)(t_live_end - t_live0),
                (unsigned long)pl_stats.s_max_backlog,
                (unsigned long)(pl_stats.s_max_backlog * ADPCM_BLOCK_MS),
                (unsigned long)held_ms,
                pl_stats.s_blocks_dropped
                  ? "NETWORK TOO SLOW - oldest audio dropped (see below)"
                  : "clean - nothing lost");
  if (pl_stats.s_blocks_dropped)
    Serial.printf("  capture: %lu ms of the oldest unsent audio was dropped\n",
                  (unsigned long)(pl_stats.s_blocks_dropped * ADPCM_BLOCK_MS));

  memset(g_feat, PRAHARI_MODEL_IN_FILL, sizeof(g_feat));
  g_feat_head = 0;
  memset(g_win, 0, sizeof(g_win));
  g_hist_n = 0;
  for (int i = 0; i < SMOOTH_N; i++) g_hist[i] = 0.0f;
}

static bool read_hop(int16_t *dst) {
  size_t want = PRAHARI_HOP_SAMPLES * sizeof(int16_t), got = 0, n = 0;
  while (got < want) {
    if (i2s_read(I2S_NUM_0, (uint8_t *)dst + got, want - got, &n,
                 portMAX_DELAY) != ESP_OK) return false;
    got += n;
  }
  g_raw_rms = 0.9f * g_raw_rms + 0.1f * rms_i16(dst, PRAHARI_HOP_SAMPLES);
  for (int i = 0; i < PRAHARI_HOP_SAMPLES; i++) {
    if (dst[i] < g_raw_min) g_raw_min = dst[i];
    if (dst[i] > g_raw_max) g_raw_max = dst[i];
  }

  for (int i = 0; i < PRAHARI_HOP_SAMPLES; i++) {
    float x = (float)dst[i];
    float y = x - g_dc_x1 + 0.995f * g_dc_y1;
    g_dc_x1 = x; g_dc_y1 = y;
    float v = y * g_mic_gain;
    if (v >  32767.0f) v =  32767.0f;
    if (v < -32768.0f) v = -32768.0f;
    dst[i] = (int16_t)v;
  }
  return true;
}

void setup() {
  Serial.begin(115200);
  Serial.setTxTimeoutMs(0);
  delay(2500);
}

static void profile_ops(int runs) {
  if (!g_ready) { Serial.println(F("not ready - cannot profile")); return; }

  float mel[PRAHARI_MEL_BINS];
  int n_frames = 1 + (PRAHARI_TEST_CLIP_LEN - PRAHARI_WIN_SAMPLES)
                     / PRAHARI_HOP_SAMPLES;
  if (n_frames > PRAHARI_CTX_FRAMES) n_frames = PRAHARI_CTX_FRAMES;

  static int8_t ref[PRAHARI_CTX_FRAMES * PRAHARI_MEL_BINS];
  memset(ref, PRAHARI_MODEL_IN_FILL, sizeof(ref));
  for (int f = 0; f < n_frames; f++) {
    prahari_logmel_frame(prahari_clip_positive + f * PRAHARI_HOP_SAMPLES, mel);
    quantise_row(mel, ref + f * PRAHARI_MEL_BINS);
  }

  Serial.printf("\nprofiling %d inferences on the embedded positive clip ...\n",
                runs);
  g_prof.clear();
  g_prof.arm();
  int64_t w0 = esp_timer_get_time();
  for (int i = 0; i < runs; i++) {
    memcpy(g_in->data.int8, ref, sizeof(ref));
    g_prof.begin_run();
    if (g_interp->Invoke() != kTfLiteOk) {
      Serial.println(F("INVOKE FAILED during profiling"));
      g_prof.disarm();
      return;
    }
    g_prof.end_run();
  }
  int64_t w1 = esp_timer_get_time();
  g_prof.disarm();

  g_prof.report();
  Serial.printf("wall clock per inference: %.2f ms  (includes a %u B memcpy)\n",
                (float)(w1 - w0) / runs / 1000.0f,
                (unsigned)sizeof(ref));
  Serial.printf("score on the positive clip: %.6f   (expect %.6f)  %s\n\n",
                score_from_output(), (double)PRAHARI_EXPECT_POS_SCORE,
                g_out->data.int8[0] == PRAHARI_EXPECT_POS_RAW
                  ? "MATCH" : "*** MISMATCH ***");
}

static void poll_serial() {
  static char buf[24];
  static int  n = 0;
  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\r') continue;
    if (c != '\n' && n < (int)sizeof(buf) - 1) { buf[n++] = c; continue; }
    buf[n] = 0;
    if (n) {
      float v = (float)atof(buf + 1);
      switch (buf[0]) {
        case 't': case 'T':
          if (v > 0.0f && v < 1.0f) {
            g_thresh = v;
            Serial.printf("\n>> threshold = %.2f\n\n", g_thresh);
          } else Serial.println(F("\n>> threshold must be 0..1\n"));
          break;
        case 'b': case 'B':
          if (v > 0.0f && v < 1.0f) {
            g_burst_arm = v; g_burst_on = true;
            Serial.printf("\n>> burst scanning ON, arms at score %.2f\n\n", g_burst_arm);
          } else {
            g_burst_on = !g_burst_on; g_burst_left = 0;
            Serial.printf("\n>> burst scanning %s (arms at %.2f)\n\n", g_burst_on ? "ON" : "OFF", g_burst_arm);
          }
          break;
        case 'g': case 'G':
          if (v >= 0.1f && v <= 40.0f) {
            g_noise_floor *= v / g_mic_gain;
            g_mic_gain = v;
            Serial.printf("\n>> mic gain = %.2f\n\n", g_mic_gain);
          } else Serial.println(F("\n>> gain must be 0.1..40\n"));
          break;
        case 'p': case 'P':
          profile_ops((v >= 1.0f && v <= 200.0f) ? (int)v : 20);
          break;
        case 'c': case 'C':
          if (v >= 2.0f && v <= 120.0f) {
            g_capture_ms = (uint32_t)(v * 1000.0f);
            Serial.printf("\n>> capture ceiling = %lu s\n\n",
                          (unsigned long)(g_capture_ms / 1000));
          } else Serial.println(F("\n>> ceiling must be 2..120 seconds\n"));
          break;
        case 'v': case 'V':
          g_ep_trace = !g_ep_trace;
          Serial.printf("\n>> VAD trace %s\n\n", g_ep_trace ? "ON" : "OFF");
          break;
        case 'q': case 'Q':
          g_quiet = !g_quiet;
          if (g_quiet) {
            g_quiet_t0 = millis(); g_quiet_opens0 = g_gate_opens;
            g_quiet_hops0 = g_hops_total; g_quiet_runs0 = g_infer_runs;
            Serial.println(F("\n>> QUIET: status lines off. Stay silent, then type q again.\n"));
          } else {
            uint32_t hops = g_hops_total - g_quiet_hops0;
            uint32_t runs = g_infer_runs - g_quiet_runs0;
            float duty = hops ? (float)(runs * INFER_EVERY_FRAMES) / hops : 0.0f;
            float cpu = g_ms_frontend / 10.0f * 100.0f
                      + g_ms_infer / (INFER_EVERY_FRAMES * 10.0f) * 100.0f * duty;
            Serial.printf("\n>> QUIET RESULT: %lu s | gate opened %lu times | "
                          "scan duty %.1f%% | CPU %.1f%%\n\n",
                          (unsigned long)((millis() - g_quiet_t0) / 1000),
                          (unsigned long)(g_gate_opens - g_quiet_opens0),
                          duty * 100.0f, cpu);
          }
          break;
        case 's': case 'S':
          Serial.println(F("\n>> MANUAL CAPTURE (no keyword) - stay silent\n"));
          stream_event(0.0f, millis());
          break;
        case 'n': case 'N': {
          uint32_t keep = g_capture_ms;
          g_capture_ms = 60000; g_no_endpoint = true;
          Serial.println(F("\n>> ROOM RECORDING: 60 s, no keyword needed. Do NOT say prahari.\n"));
          stream_event(0.0f, millis());
          g_no_endpoint = false; g_capture_ms = keep;
          break;
        }
        case 'l': case 'L':
          link_report();
          break;
        case 'r': case 'R':
          g_triggers = 0; g_hist_n = 0;
          g_hops_total = 0; g_hops_gated = 0; g_infer_runs = 0; g_burst_extra = 0; g_bursts = 0;
          Serial.println(F("\n>> hit counter and CPU average reset\n"));
          break;
        default:
          Serial.println();
          Serial.println(F("commands (type, then Enter):"));
          Serial.printf("  t0.70   set trigger threshold   (now %.2f)\n", g_thresh);
          Serial.printf("  g2.0    set microphone gain     (now %.2f)\n", g_mic_gain);
          Serial.printf("  b       burst scanning on/off   (now %s, arms at %.2f; b0.40 sets it)\n",
                        g_burst_on ? "ON" : "OFF", g_burst_arm);
          Serial.println(F("  r       reset the hit counter and the CPU average"));
          Serial.println(F("  l       warm-link report (state, reconnects, pings)"));
          Serial.println(F("  q       quiet test: status lines off/on, reports gate opens + CPU"));
          Serial.println(F("  s       record + send a few seconds now, no keyword needed"));
          Serial.println(F("  n       record + send 60 s of the room (retrain negatives)"));
          Serial.printf("  c60     capture ceiling in seconds  (now %lu)\n",
                        (unsigned long)(g_capture_ms / 1000));
          Serial.printf("  v       VAD trace on/off            (now %s)\n",
                        g_ep_trace ? "ON" : "OFF");
          Serial.println(F("  p20     per-operator timing, 20 inferences"));
          Serial.println(F("  ?       this help"));
          Serial.println(F("Tune in the actual room: say the keyword and say"));
          Serial.println(F("the near-miss word, and find the value that"));
          Serial.println(F("separates them. Then put it in main.cpp.\n"));
      }
    }
    n = 0;
  }
}

void loop() {
  poll_serial();

  static bool booted = false;
  if (!booted) {
    booted = true;
    Serial.println();
    rule();
    Serial.println(F("PRAHARI - LIVE   (any TFLM messages below are real errors)"));
    rule();

    prahari_frontend_init();
    if (!init_model()) { g_ready = false; return; }

    run_clip(g_pos, prahari_clip_positive,
             PRAHARI_EXPECT_POS_RAW, PRAHARI_EXPECT_POS_SCORE);
    run_clip(g_neg, prahari_clip_negative,
             PRAHARI_EXPECT_NEG_RAW, PRAHARI_EXPECT_NEG_SCORE);
    g_clip_rms = rms_i16(prahari_clip_positive, PRAHARI_TEST_CLIP_LEN);

    Serial.printf("model        : %u B flash\n", prahari_model_tflite_len);
    Serial.printf("arena        : %u B used of %d allocated\n",
                  (unsigned)g_arena_used, ARENA_BYTES);
    Serial.printf("free SRAM    : %u B\n",
                  (unsigned)heap_caps_get_free_size(CAPS));
    rule();
    Serial.println(F("SELF-TEST (embedded clips - must match the laptop)"));
    print_clip("POSITIVE", g_pos);
    print_clip("NEGATIVE", g_neg);
    bool ok = g_pos.match && g_neg.match;
    Serial.printf("self-test    : %s\n", ok ? "PASS" : "*** FAIL ***");
    rule();

    wifi_begin();
    rule();

    init_privacy_led();
    if (!init_microphone()) { g_ready = false; return; }

    memset(g_win, 0, sizeof(g_win));
    memset(g_feat, PRAHARI_MODEL_IN_FILL, sizeof(g_feat));
    g_feat_head = 0;
    for (int i = 0; i < SMOOTH_N; i++) g_hist[i] = 0.0f;

    Serial.println(F("MICROPHONE LIVE"));
    Serial.printf("  PDM clock GPIO%d, data GPIO%d, %d Hz mono\n",
                  PDM_CLK_PIN, PDM_DATA_PIN, PRAHARI_SAMPLE_RATE);
    Serial.printf("  inference every %d hops (%d ms), threshold %.2f,"
                  " smoothing %d, refractory %d ms\n",
                  INFER_EVERY_FRAMES, INFER_EVERY_FRAMES * 10,
                  g_thresh, SMOOTH_N, REFRACTORY_MS);
    Serial.printf("  reference loudness (positive clip) RMS %.0f\n", g_clip_rms);
    Serial.println(F("  compare that with the live RMS below - if they are"));
    Serial.println(F("  far apart, set MIC_GAIN and reflash."));
    rule();
    Serial.println(F("microphone check - one second of audio..."));
    {
      int16_t probe[PRAHARI_HOP_SAMPLES];
      double acc = 0; int16_t lo = 32767, hi = -32768; int zero = 0;
      for (int k = 0; k < 100; k++) {
        size_t got = 0, want = sizeof(probe), n = 0;
        while (got < want) {
          if (i2s_read(I2S_NUM_0, (uint8_t *)probe + got, want - got, &n,
                       pdMS_TO_TICKS(200)) != ESP_OK) break;
          if (n == 0) break;
          got += n;
        }
        for (int i = 0; i < PRAHARI_HOP_SAMPLES; i++) {
          acc += (double)probe[i] * probe[i];
          if (probe[i] < lo) lo = probe[i];
          if (probe[i] > hi) hi = probe[i];
          if (probe[i] == 0) zero++;
        }
      }
      float r = (float)sqrt(acc / (100.0 * PRAHARI_HOP_SAMPLES));
      Serial.printf("  raw RMS %.1f   min %d   max %d   exact-zero samples %d/16000\n",
                    r, lo, hi, zero);
      if (hi == lo) {
        Serial.println(F("  *** THE MICROPHONE IS RETURNING A CONSTANT. Not wired,"));
        Serial.println(F("  *** wrong pins, or the Sense expansion board is not"));
        Serial.println(F("  *** seated. Nothing below this line can work."));
      } else if (r < 20.0f) {
        Serial.println(F("  microphone is alive but very quiet - speak close and"));
        Serial.println(F("  watch the raw RMS in the live lines below."));
      } else {
        Serial.println(F("  microphone looks alive."));
      }
      Serial.printf("  for scale, the positive training clip has RMS %.0f\n",
                    g_clip_rms);
    }
    rule();
    print_ram_budget();
    Serial.println(F("Say \"prahari\"."));
    g_ready = true;
    return;
  }

  if (!g_ready) {
    Serial.printf("NOT RUNNING: %s\n", g_fail ? g_fail : "unknown");
    delay(5000);
    return;
  }

  int16_t hop[PRAHARI_HOP_SAMPLES];
  if (!read_hop(hop)) { Serial.println(F("i2s_read failed")); delay(1000); return; }

  preroll_push(hop);
  link_service();

  int64_t t0 = esp_timer_get_time();

  memmove(g_win, g_win + PRAHARI_HOP_SAMPLES,
          (PRAHARI_WIN_SAMPLES - PRAHARI_HOP_SAMPLES) * sizeof(int16_t));
  memcpy(g_win + PRAHARI_WIN_SAMPLES - PRAHARI_HOP_SAMPLES, hop, sizeof(hop));

  float mel[PRAHARI_MEL_BINS];
  prahari_logmel_frame(g_win, mel);
  quantise_row(mel, g_feat + (uint32_t)g_feat_head * PRAHARI_MEL_BINS);
  g_feat_head = (uint16_t)((g_feat_head + 1) % PRAHARI_CTX_FRAMES);

  int64_t t1 = esp_timer_get_time();
  g_ms_frontend = (t1 - t0) / 1000.0f;
  g_live_rms = 0.9f * g_live_rms + 0.1f * rms_i16(hop, PRAHARI_HOP_SAMPLES);
  if (g_live_rms > g_level_peak) g_level_peak = g_live_rms;

  {
    float rms = hp_rms(hop, PRAHARI_HOP_SAMPLES);
    if (rms < g_noise_floor * VAD_TRIGGER_RATIO)
      g_noise_floor = 0.995f * g_noise_floor + 0.005f * rms;
    else if (rms < g_noise_floor)
      g_noise_floor = 0.9f * g_noise_floor + 0.1f * rms;
    else
      g_noise_floor *= 1.0005f;
    if (g_noise_floor < 1.0f) g_noise_floor = 1.0f;

    float open_at = g_noise_floor * VAD_TRIGGER_RATIO;
    if (open_at < VAD_MIN_RMS) open_at = VAD_MIN_RMS;
    static uint8_t loud_run = 0;
    if (rms > open_at) { if (loud_run < 255) loud_run++; }
    else loud_run = 0;
    if (loud_run >= VAD_OPEN_HOPS) {
      if (loud_run == VAD_OPEN_HOPS &&
          (int32_t)(millis() - g_gate_open_until) > 0) g_gate_opens++;
      g_gate_open_until = millis() + VAD_HANGOVER_MS;
    }
  }
  g_hops_total++;
  g_frames++;

#if VAD_ENABLED
  bool active = ((int32_t)(millis() - g_gate_open_until) <= 0);
  uint32_t cadence = active ? INFER_EVERY_FRAMES : INFER_SLOW_FRAMES;
#else
  uint32_t cadence = INFER_EVERY_FRAMES;
#endif
  bool in_burst = g_burst_left > 0;
  if (in_burst) { g_burst_left--; cadence = BURST_EVERY_FRAMES; }
  if (g_frames % cadence) return;
  const bool on_grid = (g_frames % INFER_EVERY_FRAMES) == 0;
  if (on_grid || !in_burst) {
    g_hops_gated += (in_burst ? INFER_EVERY_FRAMES : cadence);
    g_infer_runs++;
  } else {
    g_burst_extra++;
  }

  {
    const uint32_t head      = (uint32_t)g_feat_head;
    const uint32_t tail_rows = PRAHARI_CTX_FRAMES - head;
    memcpy(g_in->data.int8,
           g_feat + head * PRAHARI_MEL_BINS,
           tail_rows * PRAHARI_MEL_BINS);
    if (head)
      memcpy(g_in->data.int8 + tail_rows * PRAHARI_MEL_BINS,
             g_feat,
             head * PRAHARI_MEL_BINS);
  }

  int64_t t2 = esp_timer_get_time();
  if (g_interp->Invoke() != kTfLiteOk) {
    Serial.println(F("Invoke failed"));
    return;
  }
  int64_t t3 = esp_timer_get_time();
  g_ms_infer = (t3 - t2) / 1000.0f;
  g_infers++;

  g_last_score = score_from_output();
  g_hist[g_infers % SMOOTH_N] = g_last_score;
  if (g_hist_n < SMOOTH_N) g_hist_n++;
  g_last_smooth = 0.0f;
  for (int i = 0; i < g_hist_n; i++)
    if (g_hist[i] > g_last_smooth) g_last_smooth = g_hist[i];
  if (g_last_score > g_score_max) g_score_max = g_last_score;

  if (g_burst_on && on_grid && g_last_score >= g_burst_arm) {
    if (!g_burst_left) g_bursts++;
    g_burst_left = BURST_LEN_FRAMES;
  }

  uint32_t now = millis();
  bool refractory = (now - g_last_trigger_ms) < REFRACTORY_MS;

  if (g_last_smooth >= g_thresh && !refractory) {
    g_last_trigger_ms = now;
    g_triggers++;
    Serial.printf("\n  *** PRAHARI ***  #%lu   score %.3f (peak-of-%d %.3f)"
                  "   RMS %.0f   t=%lu ms\n\n",
                  (unsigned long)g_triggers, g_last_score, SMOOTH_N,
                  g_last_smooth, g_live_rms, (unsigned long)now);
    stream_event(g_last_score, now);
    g_last_trigger_ms = millis();
    return;
  }

  if (g_infers % STATUS_EVERY == 0 && !g_quiet) {
    float duty = g_hops_total
        ? (float)(g_infer_runs * INFER_EVERY_FRAMES) / (float)g_hops_total
        : 1.0f;
    if (duty > 1.0f) duty = 1.0f;
    float cpu_front = g_ms_frontend / 10.0f * 100.0f;
    float cpu_infer = g_ms_infer / (INFER_EVERY_FRAMES * 10.0f) * 100.0f * duty;
    if (g_hops_total)
      cpu_infer += g_ms_infer * (float)g_burst_extra / (g_hops_total * 10.0f) * 100.0f;
    float cpu = cpu_front + cpu_infer;
    const float lo = g_clip_rms * 0.40f, hi = g_clip_rms * 2.50f;
    const char *verdict = g_level_peak < lo ? "TOO QUIET"
                        : g_level_peak > hi ? "TOO LOUD" : "ok";
    Serial.printf("[live] score %.3f (best %.3f) trig@%.2f | gate %4.1f%% "
                  "floor %4.0f %s | LOUDEST %5.0f/%.0f [%s] gain %.1f | "
                  "front %.2f infer %.1f | CPU %.0f%% = %.0f front + %.0f net "
                  "| hits %lu | link %s %.0fus | opens %lu | bursts %lu%s\n",
                  g_last_score, g_score_max, g_thresh,
                  duty * 100.0f, g_noise_floor,
                  ((int32_t)(millis() - g_gate_open_until) <= 0) ? "FAST" : "slow",
                  g_level_peak, g_clip_rms, verdict,
                  g_level_peak > 1.0f ? g_clip_rms / g_level_peak : 1.0f,
                  g_ms_frontend, g_ms_infer,
                  cpu, cpu_front, cpu_infer,
                  (unsigned long)g_triggers, link_state_name(),
                  g_us_link_avg, (unsigned long)g_gate_opens,
                  (unsigned long)g_bursts, g_burst_on ? "" : " (OFF)");    g_level_peak = 0.0f;
    if (g_infers % (STATUS_EVERY * 10) == 0) print_ram_budget();
    g_score_max = 0.0f;
    g_raw_min = 0; g_raw_max = 0;
  }
}
