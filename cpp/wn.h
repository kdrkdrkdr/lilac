#ifndef LILAC_WN_H
#define LILAC_WN_H

#include <stdint.h>

struct TensorStore;

/* WN block — dilated residual conv stack with gated activation, shared by
   PosteriorEncoder (enc_q.enc) and every ResidualCouplingLayer (flow.*.enc).
   Matches vc/modules.py's WN class. dilation_rate is always 1 in lilac, so
   every in_layer uses dilation=1. */
#define LILAC_WN_MAX_LAYERS 32

typedef struct {
    int hidden_channels;
    int kernel_size;
    int n_layers;
    int gin_channels;

    const float *cond_w;   /* Conv1d k=1: [2H * n_layers, gin]  or NULL if gin=0 */
    const float *cond_b;   /* [2H * n_layers] */

    const float *in_w[LILAC_WN_MAX_LAYERS]; /* Conv1d k=K: [2H, H, K] */
    float       *in_w_kfirst[LILAC_WN_MAX_LAYERS]; /* [K, 2H, H] prepacked for conv1d_direct */
    const float *in_b[LILAC_WN_MAX_LAYERS]; /* [2H] */

    const float *rs_w[LILAC_WN_MAX_LAYERS]; /* Conv1d k=1: last layer [H, H], others [2H, H] */
    const float *rs_b[LILAC_WN_MAX_LAYERS]; /* last [H], others [2H] */
} WNBlock;

int  wn_block_init(WNBlock *wn, const struct TensorStore *store,
                   const char *prefix,  /* e.g. "enc_q.enc" or "flow.flows.0.enc" */
                   int hidden, int kernel, int n_layers, int gin_channels);
void wn_block_free(WNBlock *wn);

/* x is [H, T] in-place; output written to `out` [H, T].
   g (when gin_channels > 0) is [gin, 1] (single-frame speaker embed).
   scratch size is returned by wn_block_scratch_floats(). */
void wn_block_forward(const WNBlock *wn, float *x, int T,
                      const float *g, float *out, float *scratch);

/* Returns the number of floats the caller must supply in scratch. */
int  wn_block_scratch_floats(const WNBlock *wn, int T_max);

#endif
