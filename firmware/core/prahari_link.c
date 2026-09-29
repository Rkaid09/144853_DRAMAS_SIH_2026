#include "prahari_link.h"

#include <errno.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>

#if defined(ESP_PLATFORM)
  #include <sys/select.h>
  #include <sys/time.h>
  #include "lwip/sockets.h"
  #include "lwip/netdb.h"
#else
  #include <sys/socket.h>
  #include <sys/select.h>
  #include <sys/time.h>
  #include <netinet/in.h>
  #include <netinet/tcp.h>
#endif

#ifdef MSG_NOSIGNAL
  #define PL_SEND_FLAGS MSG_NOSIGNAL
#else
  #define PL_SEND_FLAGS 0
#endif

#ifndef PL_PING_EVERY_MS
#define PL_PING_EVERY_MS      10000
#endif
#ifndef PL_PONG_TIMEOUT_MS
#define PL_PONG_TIMEOUT_MS     4000
#endif
#define PL_CONNECT_TIMEOUT_MS  3000
#ifndef PL_BACKOFF_MIN_MS
#define PL_BACKOFF_MIN_MS      1000
#endif
#define PL_BACKOFF_MAX_MS      5000
#define PL_IDLE_POLL_EVERY       10
#define PL_STALL_MS            3000
#define PL_MAX_BLOCK            256

pl_stats_t pl_stats;

static uint32_t (*s_now)(void);
static void     (*s_log)(const char *);

static char       s_ip[40];
static uint16_t   s_port;
static const char *s_hello;

static int        s_fd = -1;
static pl_state_t s_state = PL_DOWN;
static uint32_t   s_t_conn0, s_next_try, s_backoff = PL_BACKOFF_MIN_MS;
static uint16_t   s_fail_streak;
static uint32_t   s_last_tx, s_ping_at, s_calls;
static int        s_ping_out;
static char       s_rx[160];
static uint16_t   s_rx_n;

static int        s_streaming;
static uint8_t   *s_ring;
static uint16_t   s_nblk, s_bb, s_rd, s_avail;
static uint8_t    s_tail[PL_MAX_BLOCK];
static uint16_t   s_tail_n, s_tail_off;
static uint32_t   s_t0, s_last_progress;

static uint32_t now_ms(void) { return s_now ? s_now() : 0; }

static void logf_(const char *fmt, ...) {
  if (!s_log) return;
  char b[176];
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(b, sizeof b, fmt, ap);
  va_end(ap);
  s_log(b);
}

static int would_block(int e) { return e == EAGAIN || e == EWOULDBLOCK; }

static void close_fd(void) {
  if (s_fd >= 0) { close(s_fd); s_fd = -1; }
}

static int wait_fd(int for_write, uint32_t ms) {
  if (s_fd < 0) return -1;
  fd_set set;
  FD_ZERO(&set);
  FD_SET(s_fd, &set);
  struct timeval tv;
  tv.tv_sec = ms / 1000;
  tv.tv_usec = (ms % 1000) * 1000;
  return select(s_fd + 1, for_write ? NULL : &set, for_write ? &set : NULL,
                NULL, &tv);
}

static void schedule_retry(void) {
  s_next_try = now_ms() + s_backoff;
  s_backoff = s_backoff * 2 > PL_BACKOFF_MAX_MS ? PL_BACKOFF_MAX_MS
                                                : s_backoff * 2;
}

void pl_drop(const char *why) {
  int was_up = (s_state == PL_UP);
  close_fd();
  s_state = PL_DOWN;
  s_streaming = 0;
  if (was_up) {
    pl_stats.disconnects++;
    s_backoff = PL_BACKOFF_MIN_MS;
    logf_("link: DOWN (%s) - reconnecting in the background",
          why ? why : "?");
  } else {
    pl_stats.connect_fails++;
    s_fail_streak++;
    if ((s_fail_streak & (s_fail_streak - 1)) == 0)
      logf_("link: connect to %s:%u failed (%s), attempt %u, next in %lu ms",
            s_ip, (unsigned)s_port, why ? why : "?", (unsigned)s_fail_streak,
            (unsigned long)s_backoff);
  }
  schedule_retry();
}

static int send_all(const void *p, size_t n, uint32_t timeout_ms) {
  const uint8_t *b = (const uint8_t *)p;
  uint32_t t0 = now_ms();
  while (n) {
    if (s_fd < 0) return -1;
    int r = (int)send(s_fd, b, n, PL_SEND_FLAGS);
    if (r > 0) { b += r; n -= (size_t)r; continue; }
    if (r < 0 && !would_block(errno)) { pl_drop(strerror(errno)); return -1; }
    if (now_ms() - t0 > timeout_ms) { pl_drop("send timeout"); return -1; }
    wait_fd(1, 10);
  }
  s_last_tx = now_ms();
  return 0;
}

static void went_up(void) {
  uint32_t t = now_ms();
  s_state = PL_UP;
  pl_stats.connects++;
  pl_stats.last_connect_ms = t - s_t_conn0;
  s_backoff = PL_BACKOFF_MIN_MS;
  s_fail_streak = 0;
  s_ping_out = 0;
  s_rx_n = 0;
  s_last_tx = t;
  logf_("link: UP to %s:%u in %lu ms - kept open for every wake event",
        s_ip, (unsigned)s_port, (unsigned long)pl_stats.last_connect_ms);
  if (s_hello) send_all(s_hello, strlen(s_hello), 200);
}

static void start_connect(void) {
  struct sockaddr_in sa;
  memset(&sa, 0, sizeof sa);
  sa.sin_family = AF_INET;
  sa.sin_port = htons(s_port);
  s_t_conn0 = now_ms();
  unsigned a, b, c, d;
  if (sscanf(s_ip, "%u.%u.%u.%u", &a, &b, &c, &d) != 4 ||
      a > 255 || b > 255 || c > 255 || d > 255) { pl_drop("bad ip"); return; }
  {
    uint8_t *ip = (uint8_t *)&sa.sin_addr.s_addr;
    ip[0] = (uint8_t)a; ip[1] = (uint8_t)b; ip[2] = (uint8_t)c; ip[3] = (uint8_t)d;
  }

  s_fd = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
  if (s_fd < 0) { pl_drop("no socket"); return; }
  int fl = fcntl(s_fd, F_GETFL, 0);
  fcntl(s_fd, F_SETFL, fl | O_NONBLOCK);

#ifdef PL_TEST_SNDBUF
  { int v = PL_TEST_SNDBUF; setsockopt(s_fd, SOL_SOCKET, SO_SNDBUF, &v, sizeof v); }
#endif
  int one = 1;
  setsockopt(s_fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
  setsockopt(s_fd, SOL_SOCKET, SO_KEEPALIVE, &one, sizeof one);
#ifdef TCP_KEEPIDLE
  { int v = 20; setsockopt(s_fd, IPPROTO_TCP, TCP_KEEPIDLE, &v, sizeof v); }
#endif
#ifdef TCP_KEEPINTVL
  { int v = 5;  setsockopt(s_fd, IPPROTO_TCP, TCP_KEEPINTVL, &v, sizeof v); }
#endif
#ifdef TCP_KEEPCNT
  { int v = 3;  setsockopt(s_fd, IPPROTO_TCP, TCP_KEEPCNT, &v, sizeof v); }
#endif

  int r = connect(s_fd, (struct sockaddr *)&sa, sizeof sa);
  if (r == 0) { went_up(); return; }
  if (errno == EINPROGRESS || would_block(errno)) { s_state = PL_CONNECTING; return; }
  pl_drop(strerror(errno));
}

static void check_connecting(uint32_t wait_ms) {
  int r = wait_fd(1, wait_ms);
  if (r > 0) {
    int err = 0;
    socklen_t len = sizeof err;
    getsockopt(s_fd, SOL_SOCKET, SO_ERROR, &err, &len);
    if (err == 0) went_up();
    else          pl_drop(strerror(err));
  } else if (r < 0) {
    pl_drop("select");
  } else if (now_ms() - s_t_conn0 > PL_CONNECT_TIMEOUT_MS) {
    pl_drop("timeout");
  }
}

static void on_line(const char *line) {
  if (strstr(line, "\"pong\"")) pl_stats.pongs++;
  if (s_streaming && pl_stats.s_ack_ms < 0 && strstr(line, "\"ack\""))
    pl_stats.s_ack_ms = (int32_t)(now_ms() - s_t0);
}

static int poll_rx(void) {
  char buf[96];
  for (;;) {
    if (s_fd < 0) return -1;
    int r = (int)recv(s_fd, buf, sizeof buf, 0);
    if (r > 0) {
      s_ping_out = 0;
      for (int i = 0; i < r; i++) {
        char c = buf[i];
        if (c == '\n') { s_rx[s_rx_n] = 0; on_line(s_rx); s_rx_n = 0; }
        else if (s_rx_n < sizeof s_rx - 1) s_rx[s_rx_n++] = c;
        else s_rx_n = 0;
      }
      if (r < (int)sizeof buf) return 0;
      continue;
    }
    if (r == 0) { pl_drop("server closed the connection"); return -1; }
    if (would_block(errno)) return 0;
    pl_drop(strerror(errno));
    return -1;
  }
}

void pl_init(uint32_t (*millis_fn)(void), void (*log_fn)(const char *)) {
  s_now = millis_fn;
  s_log = log_fn;
  memset(&pl_stats, 0, sizeof pl_stats);
  pl_stats.s_ack_ms = -1;
  pl_stats.s_hdr_ms = -1;
}

void pl_set_server(const char *ip, uint16_t port) {
  if (strncmp(ip, s_ip, sizeof s_ip) == 0 && port == s_port) return;
  if (s_state != PL_DOWN) pl_drop("server address changed");
  strncpy(s_ip, ip, sizeof s_ip - 1);
  s_ip[sizeof s_ip - 1] = 0;
  s_port = port;
  s_backoff = PL_BACKOFF_MIN_MS;
  s_next_try = now_ms();
  s_fail_streak = 0;
}

void pl_set_hello(const char *json_line) { s_hello = json_line; }
pl_state_t pl_state(void)   { return s_state; }
int pl_is_up(void)          { return s_state == PL_UP; }
uint16_t pl_fail_streak(void) { return s_fail_streak; }

void pl_service(void) {
  uint32_t t = now_ms();
  switch (s_state) {
    case PL_DOWN:
      if (s_ip[0] && (int32_t)(t - s_next_try) >= 0) start_connect();
      return;
    case PL_CONNECTING:
      check_connecting(0);
      return;
    case PL_UP:
      if (s_streaming) return;
      if (s_ping_out || (++s_calls % PL_IDLE_POLL_EVERY) == 0)
        if (poll_rx() < 0) return;
      if (s_ping_out && t - s_ping_at > PL_PONG_TIMEOUT_MS) {
        pl_drop("no pong");
        return;
      }
      if (!s_ping_out && t - s_last_tx >= PL_PING_EVERY_MS) {
        static const char ping[] = "{\"event\":\"ping\"}\n";
        if (send_all(ping, sizeof ping - 1, 100) == 0) {
          s_ping_at = t;
          s_ping_out = 1;
          pl_stats.pings++;
        }
      }
      return;
  }
}

int pl_connect_blocking(uint32_t timeout_ms) {
  uint32_t t0 = now_ms();
  if (s_state == PL_DOWN && s_ip[0]) start_connect();
  while (s_state == PL_CONNECTING && now_ms() - t0 < timeout_ms)
    check_connecting(10);
  return s_state == PL_UP;
}

uint16_t pl_stream_backlog(void) { return s_avail; }

void pl_stream_block_done(void) {
  if (!s_streaming) return;
  s_avail++;
  if (s_avail > s_nblk - 1) {
    s_rd = (uint16_t)((s_rd + 1) % s_nblk);
    s_avail--;
    pl_stats.s_blocks_dropped++;
  }
  if (s_avail > pl_stats.s_max_backlog) pl_stats.s_max_backlog = s_avail;
}

int pl_stream_begin(uint8_t *ring, uint16_t nblk, uint16_t bb,
                    uint16_t rd, uint16_t avail, uint32_t t_trigger,
                    const char *hdr, size_t hdr_len) {
  if (bb > PL_MAX_BLOCK || nblk < 2) return -1;
  pl_stats.s_blocks_sent = 0;
  pl_stats.s_blocks_dropped = 0;
  pl_stats.s_max_backlog = avail;
  pl_stats.s_ack_ms = -1;
  pl_stats.s_hdr_ms = -1;
  s_ring = ring; s_nblk = nblk; s_bb = bb;
  s_rd = rd; s_avail = avail;
  s_tail_n = s_tail_off = 0;
  s_t0 = t_trigger;

  if (s_state == PL_UP) poll_rx();
  if (s_state != PL_UP && !pl_connect_blocking(1500)) return -1;

  s_streaming = 1;
  s_rx_n = 0;
  if (send_all(hdr, hdr_len, 1000) != 0) {
    if (!pl_connect_blocking(1500)) return -1;
    s_streaming = 1;
    if (send_all(hdr, hdr_len, 1000) != 0) return -1;
  }
  pl_stats.s_hdr_ms = (int32_t)(now_ms() - t_trigger);
  s_last_progress = now_ms();
  return 0;
}

int pl_stream_pump(void) {
  if (!s_streaming || s_state != PL_UP) return -1;
  if (poll_rx() < 0) return -1;

  while (s_tail_off < s_tail_n) {
    int r = (int)send(s_fd, s_tail + s_tail_off, s_tail_n - s_tail_off,
                      PL_SEND_FLAGS);
    if (r > 0) { s_tail_off += (uint16_t)r; s_last_progress = now_ms(); continue; }
    if (r < 0 && !would_block(errno)) { pl_drop(strerror(errno)); return -1; }
    goto check_stall;
  }
  if (s_tail_n) { s_tail_n = s_tail_off = 0; pl_stats.s_blocks_sent++; }

  while (s_avail) {
    uint16_t cont = s_avail;
    if (cont > s_nblk - s_rd) cont = (uint16_t)(s_nblk - s_rd);
    int r = (int)send(s_fd, s_ring + (size_t)s_rd * s_bb, (size_t)cont * s_bb,
                      PL_SEND_FLAGS);
    if (r < 0) {
      if (would_block(errno)) break;
      pl_drop(strerror(errno));
      return -1;
    }
    if (r == 0) break;
    s_last_progress = now_ms();
    uint16_t whole = (uint16_t)(r / s_bb), part = (uint16_t)(r % s_bb);
    s_rd = (uint16_t)((s_rd + whole) % s_nblk);
    s_avail -= whole;
    pl_stats.s_blocks_sent += whole;
    if (part) {
      memcpy(s_tail, s_ring + (size_t)s_rd * s_bb + part, s_bb - part);
      s_tail_n = (uint16_t)(s_bb - part);
      s_tail_off = 0;
      s_rd = (uint16_t)((s_rd + 1) % s_nblk);
      s_avail--;
      break;
    }
    if (whole < cont) break;
  }

check_stall:
  if ((s_avail || s_tail_off < s_tail_n) &&
      now_ms() - s_last_progress > PL_STALL_MS) {
    pl_drop("stream stalled");
    return -1;
  }
  return 0;
}

int pl_stream_wait(uint32_t ms) {
  if (!s_streaming || s_fd < 0) return -1;
  if (s_avail || s_tail_off < s_tail_n) wait_fd(1, ms);
  return pl_stream_pump();
}

int pl_stream_end(const char *trailer, uint32_t timeout_ms) {
  uint32_t t0 = now_ms();
  while (s_streaming && (s_avail || s_tail_off < s_tail_n)) {
    if (pl_stream_pump() < 0) return -1;
    if (now_ms() - t0 > timeout_ms) { pl_drop("drain timeout"); return -1; }
    if (s_avail || s_tail_off < s_tail_n) wait_fd(1, 10);
  }
  if (!s_streaming) return -1;

  uint8_t end[PL_MAX_BLOCK];
  memset(end, 0, sizeof end);
  end[2] = 0xFF;
  end[3] = 0xFF;
  if (trailer) {
    size_t n = strlen(trailer);
    if (n > (size_t)s_bb - 5) n = (size_t)s_bb - 5;
    memcpy(end + 4, trailer, n);
  }
  if (send_all(end, s_bb, 1000) != 0) return -1;

  uint32_t t1 = now_ms();
  while (pl_stats.s_ack_ms < 0 && s_state == PL_UP && now_ms() - t1 < 300) {
    wait_fd(0, 10);
    if (poll_rx() < 0) break;
  }
  s_streaming = 0;
  s_last_tx = now_ms();
  return 0;
}
