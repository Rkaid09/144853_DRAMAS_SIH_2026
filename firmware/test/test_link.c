#define _GNU_SOURCE
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <errno.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include "prahari_link.h"

#define SR 16000
#define HOP 160
#define BLK_SAMPLES 480
#define BLK_BYTES 244
#define NBLK 50

static uint32_t t_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint32_t)(ts.tv_sec * 1000ull + ts.tv_nsec / 1000000ull);
}
static void logcb(const char *s) { printf("  [%7.3f] %s\n", t_ms() / 1000.0, s); fflush(stdout); }

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
static uint8_t adpcm_code(adpcm_t *e, int16_t sample) {
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
static uint8_t g_blk_seq;
static void adpcm_stamp(const adpcm_t *e, uint8_t *blk) {
  blk[0] = (uint8_t)(e->pred & 0xFF);
  blk[1] = (uint8_t)((e->pred >> 8) & 0xFF);
  blk[2] = (uint8_t)e->idx;
  blk[3] = g_blk_seq++;
}

static uint8_t  g_pre[NBLK * BLK_BYTES];
static uint16_t g_pre_blk, g_pre_off;
static int      g_pre_full;
static adpcm_t  g_enc;
static int      preroll_push(const int16_t *hop) {
  uint8_t *blk = g_pre + (uint32_t)g_pre_blk * BLK_BYTES;
  if (g_pre_off == 0) adpcm_stamp(&g_enc, blk);
  uint8_t *d = blk + 4 + g_pre_off / 2;
  for (int i = 0; i < HOP; i += 2) {
    uint8_t lo = adpcm_code(&g_enc, hop[i]);
    uint8_t hi = adpcm_code(&g_enc, hop[i + 1]);
    *d++ = (uint8_t)(lo | (hi << 4));
  }
  g_pre_off += HOP;
  if (g_pre_off >= BLK_SAMPLES) {
    g_pre_off = 0;
    if (++g_pre_blk >= NBLK) { g_pre_blk = 0; g_pre_full = 1; }
    return 1;
  }
  return 0;
}

static uint64_t g_n;
static double   g_phase;
static uint32_t g_next_hop;
static uint32_t g_dma_lost_ms;

static void read_hop(int16_t *h) {
  for (;;) {
    int32_t d = (int32_t)(g_next_hop - t_ms());
    if (d <= 0) break;
    struct timespec ts = {0, (long)d * 1000000L};
    nanosleep(&ts, NULL);
  }
  while ((int32_t)(t_ms() - g_next_hop) > 512) {
    g_next_hop += 10; g_dma_lost_ms += 10;
    for (int i = 0; i < HOP; i++, g_n++) {
      double f = 400.0 + (double)((g_n / BLK_SAMPLES) % 20) * 100.0;
      g_phase += 2.0 * M_PI * f / SR;
    }
  }
  g_next_hop += 10;
  for (int i = 0; i < HOP; i++, g_n++) {
    uint64_t k = g_n / BLK_SAMPLES;
    double f = 400.0 + (double)(k % 20) * 100.0;
    g_phase += 2.0 * M_PI * f / SR;
    h[i] = (int16_t)(6000.0 * sin(g_phase));
  }
}

static void idle_for(uint32_t ms) {
  int16_t hop[HOP];
  uint32_t end = t_ms() + ms;
  while ((int32_t)(t_ms() - end) < 0) {
    read_hop(hop);
    preroll_push(hop);
    pl_service();
  }
}

static int one_event(int seq, int live_blocks) {
  uint32_t t_trig = t_ms();
  int warm = pl_is_up();
  if (!warm && !pl_connect_blocking(1500)) { printf("  EVENT %d: no link\n", seq); return -1; }
  char hdr[256];
  int n = snprintf(hdr, sizeof hdr,
      "{\"event\":\"wake\",\"score\":0.9,\"codec\":\"ima-adpcm\",\"block_samples\":480,"
      "\"block_bytes\":244,\"framing\":\"sentinel\",\"warm\":%d,\"connect_ms\":%u,\"seq\":%d}\n",
      warm, (unsigned)(t_ms() - t_trig), seq);
  uint16_t rd, avail;
  if (g_pre_full) { rd = (uint16_t)((g_pre_blk + 1) % NBLK); avail = NBLK - 1; }
  else            { rd = 0; avail = g_pre_blk; }
  if (pl_stream_begin(g_pre, NBLK, BLK_BYTES, rd, avail, t_trig, hdr, (size_t)n) != 0) {
    printf("  EVENT %d: header failed\n", seq); return -1;
  }
  int16_t hop[HOP];
  int nblk = 0, ok = 1;
  uint32_t t_live0 = t_ms();
  uint32_t held = 0;
  g_dma_lost_ms = 0;
  while (nblk < live_blocks) {
    if (pl_stream_backlog() >= NBLK - 1 && held < 400) {
      uint32_t t = t_ms();
      if (pl_stream_wait(10) < 0) { ok = 0; break; }
      held += t_ms() - t;
      continue;
    }
    read_hop(hop);
    if (preroll_push(hop)) { pl_stream_block_done(); nblk++; }
    if (pl_stream_pump() < 0) { ok = 0; break; }
  }
  uint32_t wall = t_ms() - t_live0;
  char tr[128];
  snprintf(tr, sizeof tr, "{\"ack_ms\":%d,\"dropped\":%u,\"backlog_max\":%u,\"live_ms\":%d}",
           (int)pl_stats.s_ack_ms, (unsigned)pl_stats.s_blocks_dropped,
           (unsigned)pl_stats.s_max_backlog, nblk * 30);
  int e = ok ? pl_stream_end(tr, 2000) : -1;
  printf("  EVENT %d: %s warm=%d hdr %d ms ack %d ms | sent %u blocks, dropped %u, "
         "backlog max %u | held mic %u ms, DMA overflow %u ms | live %d ms audio in %u ms wall\n",
         seq, (ok && e == 0) ? "OK" : "FAILED", warm, (int)pl_stats.s_hdr_ms,
         (int)pl_stats.s_ack_ms, (unsigned)pl_stats.s_blocks_sent,
         (unsigned)pl_stats.s_blocks_dropped, (unsigned)pl_stats.s_max_backlog, (unsigned)held,
         (unsigned)g_dma_lost_ms, nblk * 30, (unsigned)wall);
  fflush(stdout);
  return (ok && e == 0) ? 0 : -1;
}

static int legacy_write(int fd, const uint8_t *p, size_t n) {
  uint32_t t0 = t_ms();
  while (n) {
    ssize_t w = send(fd, p, n, MSG_NOSIGNAL | MSG_DONTWAIT);
    if (w > 0) { p += w; n -= (size_t)w; t0 = t_ms(); continue; }
    if (w < 0 && errno != EAGAIN) return -1;
    if (t_ms() - t0 > 2000) return -1;
    usleep(1000);
  }
  return 0;
}
static int legacy_event(const char *ip, int port, int seq, int live_blocks) {
  uint32_t t_trig = t_ms();
  int fd = socket(AF_INET, SOCK_STREAM, 0);
  int v = 2880; setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &v, sizeof v);
  int one = 1; setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
  struct sockaddr_in sa; memset(&sa, 0, sizeof sa);
  sa.sin_family = AF_INET; sa.sin_port = htons((uint16_t)port);
  inet_pton(AF_INET, ip, &sa.sin_addr);
  if (connect(fd, (struct sockaddr *)&sa, sizeof sa) != 0) { close(fd); return -1; }
  uint32_t t_conn = t_ms() - t_trig;
  char hdr[256];
  int n = snprintf(hdr, sizeof hdr,
      "{\"event\":\"wake\",\"codec\":\"ima-adpcm\",\"block_bytes\":244,\"seq\":%d}\n", seq);
  int ok = legacy_write(fd, (uint8_t *)hdr, (size_t)n) == 0;
  uint32_t t_pre0 = t_ms();
  if (ok && g_pre_full) {
    uint16_t start = (uint16_t)((g_pre_blk + 1) % NBLK);
    if (start > 0) ok = legacy_write(fd, g_pre + start * BLK_BYTES, (size_t)(NBLK - start) * BLK_BYTES) == 0;
    if (ok && g_pre_blk > 0) ok = legacy_write(fd, g_pre, (size_t)g_pre_blk * BLK_BYTES) == 0;
  }
  uint32_t pre_ms = t_ms() - t_pre0;
  g_dma_lost_ms = 0;
  adpcm_t enc = {0, 0};
  uint8_t out[BLK_BYTES * 10];
  int nblk = 0, batched = 0;
  uint32_t t_live0 = t_ms();
  int16_t pcm[BLK_SAMPLES];
  while (ok && nblk < live_blocks) {
    for (int h = 0; h < 3; h++) read_hop(pcm + h * HOP);
    uint8_t *blk = out + batched * BLK_BYTES;
    adpcm_stamp(&enc, blk);
    for (int i = 0; i < BLK_SAMPLES; i += 2)
      blk[4 + i / 2] = (uint8_t)(adpcm_code(&enc, pcm[i]) | (adpcm_code(&enc, pcm[i + 1]) << 4));
    batched++; nblk++;
    if (batched == 10) { ok = legacy_write(fd, out, (size_t)batched * BLK_BYTES) == 0; batched = 0; }
  }
  if (ok && batched) ok = legacy_write(fd, out, (size_t)batched * BLK_BYTES) == 0;
  close(fd);
  printf("  LEGACY %d: %s connect %u ms | pre-roll write %u ms | DMA overflow (speech lost) %u ms"
         " | live %d ms audio in %u ms wall\n", seq, ok ? "OK" : "TRUNCATED", (unsigned)t_conn,
         (unsigned)pre_ms, (unsigned)g_dma_lost_ms, nblk * 30, (unsigned)(t_ms() - t_live0));
  return ok ? 0 : -1;
}

int main(int argc, char **argv) {
  const char *ip = argc > 1 ? argv[1] : "127.0.0.1";
  int port = argc > 2 ? atoi(argv[2]) : 5005;
  int events = argc > 3 ? atoi(argv[3]) : 3;
  int live = argc > 4 ? atoi(argv[4]) : 60;
  int gap_s = argc > 5 ? atoi(argv[5]) : 3;
  int first_idle = argc > 6 ? atoi(argv[6]) : 3;
  setvbuf(stdout, NULL, _IOLBF, 0);
  pl_init(t_ms, logcb);
  pl_set_hello("{\"event\":\"hello\",\"proto\":2,\"fw\":\"host-test\"}\n");
  pl_set_server(ip, (uint16_t)port);
  g_next_hop = t_ms();
  idle_for((uint32_t)first_idle * 1000);
  int fails = 0;
  int legacy = getenv("LEGACY") != NULL;
  for (int i = 1; i <= events; i++) {
    if ((legacy ? legacy_event(ip, port, i, live) : one_event(i, live)) != 0) fails++;
    idle_for((uint32_t)gap_s * 1000);
  }
  printf("RESULT: %d/%d events ok | connects %u disconnects %u fails %u pings %u pongs %u\n",
         events - fails, events, (unsigned)pl_stats.connects, (unsigned)pl_stats.disconnects,
         (unsigned)pl_stats.connect_fails, (unsigned)pl_stats.pings, (unsigned)pl_stats.pongs);
  return fails ? 1 : 0;
}
