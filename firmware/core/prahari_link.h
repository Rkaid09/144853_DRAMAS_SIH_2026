#ifndef PRAHARI_LINK_H
#define PRAHARI_LINK_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum { PL_DOWN = 0, PL_CONNECTING = 1, PL_UP = 2 } pl_state_t;

typedef struct {
  uint32_t connects;
  uint32_t disconnects;
  uint32_t connect_fails;
  uint32_t pings, pongs;
  uint32_t last_connect_ms;
  uint32_t s_blocks_sent;
  uint32_t s_blocks_dropped;
  uint32_t s_max_backlog;
  int32_t  s_ack_ms;
  int32_t  s_hdr_ms;
} pl_stats_t;

extern pl_stats_t pl_stats;

void       pl_init(uint32_t (*millis_fn)(void), void (*log_fn)(const char *));
void       pl_set_server(const char *ip, uint16_t port);
void       pl_set_hello(const char *json_line);

void       pl_service(void);

pl_state_t pl_state(void);
int        pl_is_up(void);
uint16_t   pl_fail_streak(void);
void       pl_drop(const char *why);

int        pl_connect_blocking(uint32_t timeout_ms);

int  pl_stream_begin(uint8_t *ring, uint16_t nblk, uint16_t bb,
                     uint16_t rd, uint16_t avail, uint32_t t_trigger,
                     const char *hdr, size_t hdr_len);
void pl_stream_block_done(void);
int  pl_stream_pump(void);
int  pl_stream_wait(uint32_t ms);
int  pl_stream_end(const char *trailer, uint32_t timeout_ms);
uint16_t pl_stream_backlog(void);

#ifdef __cplusplus
}
#endif
#endif
