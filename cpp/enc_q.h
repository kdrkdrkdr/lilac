#ifndef LILAC_ENC_Q_H
#define LILAC_ENC_Q_H

#include "wn.h"

struct TensorStore;

/* Posterior encoder: spec [spec_ch, T] -> z [inter_channels, T].
   In lilac zero_g is True, so enc_q is conditioned on zeros (not the source
   speaker embedding). With tau=0 the sampled z equals the mean m directly. */
typedef struct {
    int spec_channels;   /* 513 */
    int inter_channels;  /* 192 */
    int hidden_channels; /* 192 */
    int kernel_size;     /* 5 */
    int n_layers;        /* 16 */
    int gin_channels;    /* 256 (but we always pass zeros in forward) */

    const float *pre_w,  *pre_b;     /* Conv1d k=1 [H, spec_ch] */
    const float *proj_w, *proj_b;    /* Conv1d k=1 [2*inter, H] — split into m, logs */

    WNBlock wn;
} EncQ;

int  enc_q_init(EncQ *e, const struct TensorStore *store);
void enc_q_free(EncQ *e);

/* spec : [spec_ch, T]
   z    : [inter_ch, T]  output (the sampled posterior; with tau=0 this is m) */
void enc_q_forward(const EncQ *e, const float *spec, int T, float *z, float *scratch);

int  enc_q_scratch_floats(const EncQ *e, int T_max);

#endif
