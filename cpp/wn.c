#include "wn.h"
#include "conv.h"
#include "ops.h"
#include "tensor.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const Tensor *try_get(const TensorStore *store, const char *fmt, ...) {
    char name[256];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(name, sizeof(name), fmt, ap);
    va_end(ap);
    return tensor_get(store, name);
}

int wn_block_init(WNBlock *wn, const TensorStore *store, const char *prefix,
                  int hidden, int kernel, int n_layers, int gin_channels) {
    if (n_layers > LILAC_WN_MAX_LAYERS) return -1;
    memset(wn, 0, sizeof(*wn));
    wn->hidden_channels = hidden;
    wn->kernel_size     = kernel;
    wn->n_layers        = n_layers;
    wn->gin_channels    = gin_channels;

    if (gin_channels > 0) {
        const Tensor *cw = try_get(store, "%s.cond_layer.weight", prefix);
        const Tensor *cb = try_get(store, "%s.cond_layer.bias",   prefix);
        if (!cw || !cb) return -1;
        wn->cond_w = cw->data;
        wn->cond_b = cb->data;
    }
    for (int i = 0; i < n_layers; i++) {
        const Tensor *iw = try_get(store, "%s.in_layers.%d.weight", prefix, i);
        const Tensor *ib = try_get(store, "%s.in_layers.%d.bias",   prefix, i);
        const Tensor *rw = try_get(store, "%s.res_skip_layers.%d.weight", prefix, i);
        const Tensor *rb = try_get(store, "%s.res_skip_layers.%d.bias",   prefix, i);
        if (!iw || !ib || !rw || !rb) return -1;
        wn->in_w[i] = iw->data; wn->in_b[i] = ib->data;
        wn->rs_w[i] = rw->data; wn->rs_b[i] = rb->data;

        /* Prepack in_layer weight to [K, 2H, H] for conv1d_direct. */
        size_t pack = (size_t)(2 * hidden) * hidden * kernel;
        float *kf = (float *)malloc(pack * sizeof(float));
        if (!kf) return -1;
        conv_prepack_weight(iw->data, 2 * hidden, hidden, kernel, kf);
        wn->in_w_kfirst[i] = kf;
    }
    return 0;
}

void wn_block_free(WNBlock *wn) {
    for (int i = 0; i < LILAC_WN_MAX_LAYERS; i++) {
        free(wn->in_w_kfirst[i]);
        wn->in_w_kfirst[i] = NULL;
    }
}

int wn_block_scratch_floats(const WNBlock *wn, int T_max) {
    int H = wn->hidden_channels;
    int L = wn->n_layers;
    /* cond_g (2H*L broadcast over T), in_act [2H, T_max], rs_act [2H, T_max],
       gated [H, T_max]. in_layer now uses conv1d_direct (no im2col). */
    return 2 * H * L
         + 2 * H * T_max
         + 2 * H * T_max
         + H * T_max;
}

void wn_block_forward(const WNBlock *wn, float *x, int T,
                      const float *g, float *out, float *scratch) {
    const int H = wn->hidden_channels;
    const int K = wn->kernel_size;
    const int L = wn->n_layers;
    const int pad = (K - 1) / 2;   /* dilation_rate = 1 */

    float *cond_g   = scratch;                      /* [2H * L, 1] if g, else unused */
    float *in_act   = cond_g + 2 * H * L;           /* [2H, T]   */
    float *rs_act   = in_act + 2 * H * T;           /* [2H, T]   */
    float *gated    = rs_act + 2 * H * T;           /* [H,  T]   */

    /* Project speaker embedding once if present: cond_g = cond_w * g + cond_b */
    if (wn->gin_channels > 0 && g) {
        /* Conv1d k=1 over single frame: output shape [2H*L, 1] */
        conv1d(g, wn->gin_channels, 1, wn->cond_w, 2 * H * L, 1, wn->cond_b,
               0, 1, cond_g, NULL);
    }

    /* Zero output accumulator. */
    memset(out, 0, (size_t)H * T * sizeof(float));

    for (int i = 0; i < L; i++) {
        /* in_act = in_layer_i(x), shape [2H, T] via direct conv1d. */
        conv1d_direct(x, H, T, wn->in_w_kfirst[i], 2 * H, K, wn->in_b[i],
                      pad, 1, in_act);

        if (wn->gin_channels > 0 && g) {
            const float *g_l = cond_g + (size_t)i * 2 * H;
            ops_broadcast_add_ct(in_act, g_l, 2 * H, T);
        }

        /* Gated activation: gated[c, t] = tanh(in_act[c, t]) * sigmoid(in_act[c+H, t])
           Implemented per (c, t) using a scalar pass over contiguous memory. */
        {
            const float *a = in_act;          /* tanh operand (first H channels) */
            const float *b = in_act + (size_t)H * T;  /* sigmoid operand */
            for (int c = 0; c < H; c++) {
                const float *ar = a + (size_t)c * T;
                const float *br = b + (size_t)c * T;
                float *gr = gated + (size_t)c * T;
                for (int t = 0; t < T; t++) {
                    float tv = tanhf(ar[t]);
                    float sv = 1.0f / (1.0f + expf(-br[t]));
                    gr[t] = tv * sv;
                }
            }
        }

        /* res_skip_layer: Conv1d k=1. Output channels differ between non-last and last. */
        int rs_ch = (i < L - 1) ? (2 * H) : H;
        conv1d(gated, H, T, wn->rs_w[i], rs_ch, 1, wn->rs_b[i], 0, 1, rs_act, NULL);

        if (i < L - 1) {
            /* res part into x (first H channels), skip part into out (last H channels). */
            const float *res_part  = rs_act;
            const float *skip_part = rs_act + (size_t)H * T;
            ops_vec_add(x,   res_part,  (size_t)H * T);
            ops_vec_add(out, skip_part, (size_t)H * T);
        } else {
            /* Last layer: entire rs_act is skip. */
            ops_vec_add(out, rs_act, (size_t)H * T);
        }
    }
}
