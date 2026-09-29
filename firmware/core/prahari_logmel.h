#ifndef PRAHARI_LOGMEL_H
#define PRAHARI_LOGMEL_H

#include <stdint.h>
#include "prahari_config.h"

#ifdef __cplusplus
extern "C" {
#endif

void prahari_frontend_init(void);

void prahari_logmel_frame(const int16_t *frame, float *out_mel);

int prahari_logmel_window(const int16_t *pcm, int n_samples, float *out);

void prahari_fft512(float *re, float *im);

void prahari_quantise(const float *in, int n, int8_t *out);

#ifdef __cplusplus
}
#endif
#endif
