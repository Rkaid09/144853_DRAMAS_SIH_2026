#include "prahari_logmel.h"
#include "prahari_tables.h"
#include <math.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define NFFT  PRAHARI_FFT_SIZE
#define NHALF (NFFT / 2)
#define NBINS (NFFT / 2 + 1)

#ifndef PRAHARI_REAL_FFT
#define PRAHARI_REAL_FFT 1
#endif

static float fft_re[NFFT];
static float fft_im[NFFT];
static float tw_re[NFFT / 2];
static float tw_im[NFFT / 2];
#if PRAHARI_REAL_FFT
static float pw[NBINS];
#endif
static int   initialised = 0;

void prahari_frontend_init(void)
{
    for (int k = 0; k < NFFT / 2; k++) {
        double a = -2.0 * M_PI * (double)k / (double)NFFT;
        tw_re[k] = (float)cos(a);
        tw_im[k] = (float)sin(a);
    }
    initialised = 1;
}

static void fft_inplace(void)
{
    for (int i = 1, j = 0; i < NFFT; i++) {
        int bit = NFFT >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            float t;
            t = fft_re[i]; fft_re[i] = fft_re[j]; fft_re[j] = t;
            t = fft_im[i]; fft_im[i] = fft_im[j]; fft_im[j] = t;
        }
    }

    for (int len = 2; len <= NFFT; len <<= 1) {
        int half = len >> 1;
        int step = NFFT / len;
        for (int i = 0; i < NFFT; i += len) {
            for (int j = 0; j < half; j++) {
                float wr = tw_re[j * step];
                float wi = tw_im[j * step];
                int a = i + j, b = i + j + half;
                float vr = fft_re[b] * wr - fft_im[b] * wi;
                float vi = fft_re[b] * wi + fft_im[b] * wr;
                fft_re[b] = fft_re[a] - vr;
                fft_im[b] = fft_im[a] - vi;
                fft_re[a] = fft_re[a] + vr;
                fft_im[a] = fft_im[a] + vi;
            }
        }
    }
}

__attribute__((weak)) void prahari_fft512(float *re, float *im)
{
    (void)re; (void)im;
    fft_inplace();
}

#if PRAHARI_REAL_FFT
static void fft256_inplace(void)
{
    for (int i = 1, j = 0; i < NHALF; i++) {
        int bit = NHALF >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            float t;
            t = fft_re[i]; fft_re[i] = fft_re[j]; fft_re[j] = t;
            t = fft_im[i]; fft_im[i] = fft_im[j]; fft_im[j] = t;
        }
    }

    for (int len = 2; len <= NHALF; len <<= 1) {
        int half = len >> 1;
        int step = NFFT / len;
        for (int i = 0; i < NHALF; i += len) {
            for (int j = 0; j < half; j++) {
                float wr = tw_re[j * step];
                float wi = tw_im[j * step];
                int a = i + j, b = i + j + half;
                float vr = fft_re[b] * wr - fft_im[b] * wi;
                float vi = fft_re[b] * wi + fft_im[b] * wr;
                fft_re[b] = fft_re[a] - vr;
                fft_im[b] = fft_im[a] - vi;
                fft_re[a] = fft_re[a] + vr;
                fft_im[a] = fft_im[a] + vi;
            }
        }
    }
}

static void rfft_power(void)
{
    fft256_inplace();

    {
        float ar = fft_re[0], ai = fft_im[0];
        float x0 = ar + ai;
        float xn = ar - ai;
        pw[0]     = x0 * x0;
        pw[NHALF] = xn * xn;
    }

    for (int k = 1; k < NHALF; k++) {
        float ar = fft_re[k],         ai = fft_im[k];
        float br = fft_re[NHALF - k], bi = fft_im[NHALF - k];

        float zer =  0.5f * (ar + br);
        float zei =  0.5f * (ai - bi);
        float zor =  0.5f * (ai + bi);
        float zoi = -0.5f * (ar - br);

        float wr = tw_re[k], wi = tw_im[k];

        float xr = zer + (wr * zor - wi * zoi);
        float xi = zei + (wr * zoi + wi * zor);

        pw[k] = xr * xr + xi * xi;
    }
}
#endif

void prahari_logmel_frame(const int16_t *frame, float *out_mel)
{
    if (!initialised) prahari_frontend_init();

#if PRAHARI_REAL_FFT
    for (int k = 0; k < PRAHARI_WIN_SAMPLES / 2; k++) {
        fft_re[k] = ((float)frame[2 * k]     / 32768.0f) * prahari_hann[2 * k];
        fft_im[k] = ((float)frame[2 * k + 1] / 32768.0f) * prahari_hann[2 * k + 1];
    }
    for (int k = PRAHARI_WIN_SAMPLES / 2; k < NHALF; k++) {
        fft_re[k] = 0.0f;
        fft_im[k] = 0.0f;
    }

    rfft_power();

    for (int m = 0; m < PRAHARI_MEL_BINS; m++) {
        int   s = prahari_mel_start[m];
        int   L = prahari_mel_len[m];
        int   o = prahari_mel_off[m];
        float acc = 0.0f;
        for (int k = 0; k < L; k++) {
            acc += pw[s + k] * prahari_mel_w[o + k];
        }
        out_mel[m] = logf(acc + PRAHARI_LOG_EPSILON);
    }

#else

    for (int n = 0; n < PRAHARI_WIN_SAMPLES; n++) {
        fft_re[n] = ((float)frame[n] / 32768.0f) * prahari_hann[n];
        fft_im[n] = 0.0f;
    }
    for (int n = PRAHARI_WIN_SAMPLES; n < NFFT; n++) {
        fft_re[n] = 0.0f;
        fft_im[n] = 0.0f;
    }

    prahari_fft512(fft_re, fft_im);

    for (int m = 0; m < PRAHARI_MEL_BINS; m++) {
        int   s = prahari_mel_start[m];
        int   L = prahari_mel_len[m];
        int   o = prahari_mel_off[m];
        float acc = 0.0f;
        for (int k = 0; k < L; k++) {
            int   b  = s + k;
            float re = fft_re[b], im = fft_im[b];
            acc += (re * re + im * im) * prahari_mel_w[o + k];
        }
        out_mel[m] = logf(acc + PRAHARI_LOG_EPSILON);
    }
#endif
}

int prahari_logmel_window(const int16_t *pcm, int n_samples, float *out)
{
    if (n_samples < PRAHARI_WIN_SAMPLES) return 0;
    int n_frames = 1 + (n_samples - PRAHARI_WIN_SAMPLES) / PRAHARI_HOP_SAMPLES;
    if (n_frames > PRAHARI_CTX_FRAMES) n_frames = PRAHARI_CTX_FRAMES;

    for (int f = 0; f < n_frames; f++) {
        prahari_logmel_frame(pcm + f * PRAHARI_HOP_SAMPLES,
                             out + f * PRAHARI_MEL_BINS);
    }
    return n_frames;
}

void prahari_quantise(const float *in, int n, int8_t *out)
{
    for (int i = 0; i < n; i++) {
        float q = in[i] / PRAHARI_FEAT_SCALE + (float)PRAHARI_FEAT_ZERO_POINT;
        int   v = (int)(q < 0.0f ? q - 0.5f : q + 0.5f);
        if (v < -128) v = -128;
        if (v >  127) v =  127;
        out[i] = (int8_t)v;
    }
}
