#ifndef LILAC_STFT_H
#define LILAC_STFT_H

#include <stdint.h>

/* Precomputed STFT plan: Hann window, twiddle factors, and buffer sizing.
   Lilac uses a fixed configuration (n_fft=1024, hop=256, win=1024) so the
   plan lives for the lifetime of the engine. */
typedef struct {
    int n_fft;       /* must be a power of 2 */
    int hop;
    int win_len;
    int half_bins;   /* n_fft / 2 + 1 */
    int pad;         /* reflect pad per side = (n_fft - hop) / 2 */

    float   *window;     /* [win_len] Hann */
    float   *twiddle_cos;/* [n_fft / 2] */
    float   *twiddle_sin;/* [n_fft / 2] */
    int     *bit_reverse;/* [n_fft] */
} StftPlan;

int  stft_plan_init(StftPlan *p, int n_fft, int hop, int win_len);
void stft_plan_free(StftPlan *p);

/* Returns number of frames for a given input length (matches torch.stft
   with center=False and reflect-padded input). */
int  stft_frame_count(const StftPlan *p, int T);

/* Compute magnitude spectrogram.
     input:  [T] fp32 (raw audio, no pre-padding required here)
     output: [half_bins, frames]  row-major; magnitude = sqrt(re^2 + im^2 + 1e-6)
   scratch must hold (T + 2*pad) + 2 * n_fft floats. */
void stft_magnitude(const StftPlan *p, const float *input, int T,
                    float *output, float *scratch);

#endif
