#ifndef LILAC_STREAM_H
#define LILAC_STREAM_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---------------------------------------------------------------------
   Streaming conv1d. Each call site maintains its own input state cache
   so that interior positions are computed with real left context — no
   "left window boundary" zero pad that would change between hops.

   State holds the left context of size (K-1)*dilation. Each call with
   n_new new input frames combines state + new_input into a virtual slice
   of size state_size + n_new, runs conv1d_direct on it, and emits n_new
   output frames at buffer positions [pad, pad+n_new). Those positions
   have all their receptive-field inputs fully inside the slice — so the
   output matches what a non-streaming conv would produce for the same
   (stream-absolute) time positions.

   On cold start, state is all zeros (matching dec_forward's window-level
   left zero pad for the very first few hops of the stream).
   --------------------------------------------------------------------- */

typedef struct {
    int  C;
    int  K;
    int  dilation;
    int  state_size;      /* (K-1) * dilation */
    int  pad;             /* state_size / 2 — conv's symmetric pad */
    float *cache;         /* [C, state_size] row-major, zeros initially */
} Conv1DStream;

int  c1d_stream_init (Conv1DStream *s, int C_in, int K, int dilation);
void c1d_stream_free (Conv1DStream *s);
void c1d_stream_reset(Conv1DStream *s);

/* Run streaming conv1d. Weight is in the K-first prepacked layout
   [K, C_out, C_in] produced by conv_prepack_weight.

   Produces n_new frames of output corresponding to the stream positions
   of the n_new new_input frames (shifted by the state's internal lag —
   the caller is responsible for tracking cumulative lag across layers).

   scratch: float buffer of at least 2*(C_in + C_out)*(state_size + n_new)
   floats. Used to build the combined input slice and receive the conv
   output before slicing out the valid n_new frames. */
void conv1d_stream(Conv1DStream *s,
                   const float *weight_kfirst,
                   int   C_out,
                   const float *bias,
                   const float *new_input, int n_new,
                   float *new_output,
                   float *scratch);

/* Streaming ConvTranspose1d (upsample). Symmetric pad, stride u.
   State holds the last `state_in` input frames. Each call produces n_new*u
   output frames of the "interior" slice — positions whose receptive-field
   inputs all lie in the combined (state + new_input) buffer, so no buffer-
   boundary zero-pad artifacts. Emits outputs time-aligned with the LAST
   n_new*u frames of the rolling output stream (i.e., lags by state_in input
   frames relative to new_input arrival). */
typedef struct {
    int  C_in;
    int  K;
    int  u;
    int  pad;
    int  state_in;          /* input frames of left context carried across hops */
    int  interior_start;    /* = K - pad - 1 (first output index with full RF in buf) */
    float *cache;           /* [C_in, state_in] */
} ConvTranspose1DStream;

int  ct1d_stream_init (ConvTranspose1DStream *s, int C_in, int K, int u, int pad);
void ct1d_stream_free (ConvTranspose1DStream *s);
void ct1d_stream_reset(ConvTranspose1DStream *s);

/* Weight layout: PyTorch standard [C_in, C_out, K] row-major, same as
   conv_transpose1d(). scratch must fit:
     C_in*(state_in + n_new)               (input buffer)
   + C_out*((state_in + n_new - 1)*u + K - 2*pad)  (full output)
   + C_out * K * (state_in + n_new)        (col2im scratch inside ct) */
void conv_transpose1d_stream(ConvTranspose1DStream *s,
                             const float *weight,
                             int   C_out,
                             const float *bias,
                             const float *new_input, int n_new,
                             float *new_output,
                             float *scratch);

/* ---------------------------------------------------------------------
   Delay line: holds the last `size` frames of a [C, stream] signal so the
   caller can retrieve x[T - size] when given the current x[T]. Used to
   align the `x` residual with the lagged (conv-stream) output inside a
   resblock step. Initialized to zeros (matches zero-padded warm-up).
   --------------------------------------------------------------------- */
typedef struct {
    int  C;
    int  size;
    float *buf;    /* [C, size] */
} DelayLine;

int  delay_line_init (DelayLine *d, int C, int size);
void delay_line_free (DelayLine *d);
void delay_line_reset(DelayLine *d);

/* Append new_input (n_new frames) to the tail of the line and pop the first
   n_new frames (which are x[T - size .. T - size + n_new)). */
void delay_line_step(DelayLine *d, const float *new_input, int n_new,
                     float *popped);

/* ---------------------------------------------------------------------
   Streaming ResBlock1 (HiFi-GAN style). 3 steps of (conv1 dilated + conv2),
   each with a residual add. Each step carries a per-step DelayLine to
   time-align `x` with the conv2 output. Total output lag across a resblock
   = sum over steps of (c1_pad + c2_pad).
   --------------------------------------------------------------------- */
typedef struct {
    int  C;
    int  K;
    int  step_lag[3];          /* c1_pad + c2_pad per step */
    int  total_lag;            /* sum of step_lag — resblock's overall lag */
    Conv1DStream c1[3];
    Conv1DStream c2[3];
    DelayLine    xdelay[3];
} ResblockStream;

int  rb_stream_init (ResblockStream *s, int C, int K, const int dilations[3]);
void rb_stream_free (ResblockStream *s);
void rb_stream_reset(ResblockStream *s);

void resblock_stream(ResblockStream *s,
                     const float *const c1_w_kf[3], const float *const c1_b[3],
                     const float *const c2_w_kf[3], const float *const c2_b[3],
                     const float *new_input, int n_new,
                     float *new_output,
                     float *scratch);

#ifdef __cplusplus
}
#endif

#endif
