#include "stft.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static int is_pow2(int x) { return x > 0 && (x & (x - 1)) == 0; }

static int bit_reverse(int x, int bits) {
    int y = 0;
    for (int i = 0; i < bits; i++) {
        y = (y << 1) | (x & 1);
        x >>= 1;
    }
    return y;
}

int stft_plan_init(StftPlan *p, int n_fft, int hop, int win_len) {
    if (!is_pow2(n_fft) || win_len > n_fft) return -1;
    memset(p, 0, sizeof(*p));
    p->n_fft     = n_fft;
    p->hop       = hop;
    p->win_len   = win_len;
    p->half_bins = n_fft / 2 + 1;
    p->pad       = (n_fft - hop) / 2;

    p->window       = (float *)malloc((size_t)win_len * sizeof(float));
    p->twiddle_cos  = (float *)malloc((size_t)(n_fft / 2) * sizeof(float));
    p->twiddle_sin  = (float *)malloc((size_t)(n_fft / 2) * sizeof(float));
    p->bit_reverse  = (int   *)malloc((size_t)n_fft * sizeof(int));
    if (!p->window || !p->twiddle_cos || !p->twiddle_sin || !p->bit_reverse) {
        stft_plan_free(p);
        return -1;
    }

    /* Hann window matching torch.hann_window(win_len, periodic=True):
       w[i] = 0.5 - 0.5 * cos(2*pi*i / win_len) */
    for (int i = 0; i < win_len; i++) {
        p->window[i] = 0.5f - 0.5f * cosf((float)(2.0 * M_PI * i / (double)win_len));
    }

    /* Twiddle factors for DIT FFT of length n_fft:
         W_k = exp(-j * 2*pi*k / n_fft) for k in [0, n_fft/2) */
    for (int k = 0; k < n_fft / 2; k++) {
        double ang = -2.0 * M_PI * k / (double)n_fft;
        p->twiddle_cos[k] = (float)cos(ang);
        p->twiddle_sin[k] = (float)sin(ang);
    }

    int bits = 0; while ((1 << bits) < n_fft) bits++;
    for (int i = 0; i < n_fft; i++) p->bit_reverse[i] = bit_reverse(i, bits);

    return 0;
}

void stft_plan_free(StftPlan *p) {
    if (!p) return;
    free(p->window);      p->window = NULL;
    free(p->twiddle_cos); p->twiddle_cos = NULL;
    free(p->twiddle_sin); p->twiddle_sin = NULL;
    free(p->bit_reverse); p->bit_reverse = NULL;
}

int stft_frame_count(const StftPlan *p, int T) {
    int padded = T + 2 * p->pad;
    return (padded - p->n_fft) / p->hop + 1;
}

/* In-place radix-2 DIT complex FFT with precomputed twiddles + bit-reverse table. */
static void fft_inplace(const StftPlan *p, float *re, float *im) {
    const int n = p->n_fft;
    /* Bit-reverse permutation. */
    for (int i = 0; i < n; i++) {
        int j = p->bit_reverse[i];
        if (j > i) {
            float tr = re[i]; re[i] = re[j]; re[j] = tr;
            float ti = im[i]; im[i] = im[j]; im[j] = ti;
        }
    }
    /* Butterflies. */
    for (int size = 2; size <= n; size *= 2) {
        int half = size / 2;
        int step = n / size;   /* index into twiddle tables */
        for (int i = 0; i < n; i += size) {
            for (int k = 0; k < half; k++) {
                int ip = i + k;
                int iq = ip + half;
                float wr = p->twiddle_cos[k * step];
                float wi = p->twiddle_sin[k * step];
                float tr = wr * re[iq] - wi * im[iq];
                float ti = wr * im[iq] + wi * re[iq];
                re[iq] = re[ip] - tr;
                im[iq] = im[ip] - ti;
                re[ip] += tr;
                im[ip] += ti;
            }
        }
    }
}

void stft_magnitude(const StftPlan *p, const float *input, int T,
                    float *output, float *scratch) {
    const int n_fft    = p->n_fft;
    const int hop      = p->hop;
    const int pad      = p->pad;
    const int win      = p->win_len;
    const int halfbins = p->half_bins;
    const int padded   = T + 2 * pad;
    const int frames   = (padded - n_fft) / hop + 1;

    float *padded_buf = scratch;
    float *re         = padded_buf + padded;
    float *im         = re + n_fft;

    /* Reflect pad (mirrors torch.nn.functional.pad(mode='reflect')). */
    for (int i = 0; i < pad; i++) padded_buf[i]          = input[pad - i];
    memcpy(padded_buf + pad, input, (size_t)T * sizeof(float));
    for (int i = 0; i < pad; i++) padded_buf[pad + T + i] = input[T - 2 - i];

    for (int f = 0; f < frames; f++) {
        const float *frame = padded_buf + f * hop;
        /* Apply window into the re buffer, zero im. (win_len == n_fft here,
           so we cover all of re; if win_len < n_fft we'd need to zero the rest.) */
        for (int i = 0; i < win; i++) re[i] = frame[i] * p->window[i];
        for (int i = win; i < n_fft; i++) re[i] = 0.0f;
        memset(im, 0, (size_t)n_fft * sizeof(float));

        fft_inplace(p, re, im);

        /* Magnitude with +1e-6 floor — matches vc/__init__.py spectrogram_torch. */
        for (int k = 0; k < halfbins; k++) {
            float r = re[k], i = im[k];
            output[(size_t)k * frames + f] = sqrtf(r * r + i * i + 1e-6f);
        }
    }
}
