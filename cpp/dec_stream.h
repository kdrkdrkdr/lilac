#ifndef LILAC_DEC_STREAM_H
#define LILAC_DEC_STREAM_H

#include "dec.h"
#include "stream.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Streaming HiFi-GAN generator (same topology as dec_forward in dec.c).
   Each call feeds `n_new` z-frames [192, n_new] and emits n_new*256 audio
   samples. The emitted audio stream lags the z-input stream by
   `cumulative_lag_audio` samples — the first several hops produce meaningless
   warmup data; by the time enough z has been fed to cover cum_lag, later
   outputs are bit-close (fp-noise) to dec_forward's output at the same
   absolute stream positions.

   Per-stage alignment: the three resblocks in a stage have different internal
   lags (12 / 36 / 60 at stage-out rate, for K=3/7/11 with dil={1,3,5}). We
   delay-align them to the maximum before averaging so the sum is time-
   consistent — this is the only new bookkeeping beyond the primitives in
   stream.h. */
typedef struct {
    const Dec *dec;                                 /* weights (not owned) */

    int  initial_channel;                           /* 192 */
    int  upsample_initial;                          /* 512 */
    int  stage_channels[LILAC_DEC_UPSAMPLES];       /* {256,128,64,32} */
    int  stage_rate[LILAC_DEC_UPSAMPLES];           /* cum up at stage out: {8,64,128,256} */
    int  rb_lag_out[LILAC_DEC_KERNELS_PER_UP];      /* {12,36,60} at stage-out rate */
    int  rb_max_lag;                                /* 60 */
    int  ct_lag_out[LILAC_DEC_UPSAMPLES];           /* state_in*u - interior_start */
    long cumulative_lag_audio;                      /* total dec_stream lag in audio samples */

    Conv1DStream           conv_pre;
    Conv1DStream           conv_post;
    ConvTranspose1DStream  ups[LILAC_DEC_UPSAMPLES];
    ResblockStream         rb[LILAC_DEC_UPSAMPLES][LILAC_DEC_KERNELS_PER_UP];
    DelayLine              rb_align[LILAC_DEC_UPSAMPLES][LILAC_DEC_KERNELS_PER_UP];

    float *conv_pre_w_kf;   /* K-first pack owned here (dec.conv_pre_w is orig layout) */
    float *conv_post_w_kf;
} DecStream;

int  dec_stream_init (DecStream *ds, const Dec *d);
void dec_stream_free (DecStream *ds);
void dec_stream_reset(DecStream *ds);

void dec_stream_forward(DecStream *ds,
                        const float *z_new, int n_new,
                        float *audio_out,   /* [n_new*256] */
                        float *scratch);

int  dec_stream_scratch_floats(const DecStream *ds, int n_new_max);

#ifdef __cplusplus
}
#endif

#endif
