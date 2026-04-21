#include "enc_q.h"
#include "conv.h"
#include "ops.h"
#include "tensor.h"

#include <stdlib.h>
#include <string.h>

int enc_q_init(EncQ *e, const TensorStore *store) {
    memset(e, 0, sizeof(*e));
    e->spec_channels   = 513;
    e->inter_channels  = 192;
    e->hidden_channels = 192;
    e->kernel_size     = 5;
    e->n_layers        = 16;
    e->gin_channels    = 256;

    const Tensor *t;
    t = tensor_get(store, "enc_q.pre.weight");  if (!t) return -1; e->pre_w  = t->data;
    t = tensor_get(store, "enc_q.pre.bias");    if (!t) return -1; e->pre_b  = t->data;
    t = tensor_get(store, "enc_q.proj.weight"); if (!t) return -1; e->proj_w = t->data;
    t = tensor_get(store, "enc_q.proj.bias");   if (!t) return -1; e->proj_b = t->data;

    return wn_block_init(&e->wn, store, "enc_q.enc",
                         e->hidden_channels, e->kernel_size, e->n_layers, e->gin_channels);
}

void enc_q_free(EncQ *e) { wn_block_free(&e->wn); }

int enc_q_scratch_floats(const EncQ *e, int T_max) {
    int H = e->hidden_channels;
    int x_buf    = H * T_max;                 /* pre output + WN workspace input */
    int wn_out   = H * T_max;                 /* WN output */
    int wn_scr   = wn_block_scratch_floats(&e->wn, T_max);
    int cond_zero = e->gin_channels;          /* we pass zeros-of-gin */
    return x_buf + wn_out + wn_scr + cond_zero + 64;
}

void enc_q_forward(const EncQ *e, const float *spec, int T, float *z, float *scratch) {
    const int H = e->hidden_channels;
    float *x        = scratch;                        /* [H, T] */
    float *wn_out   = x + (size_t)H * T;              /* [H, T] */
    float *wn_scr   = wn_out + (size_t)H * T;
    float *cond_zero = wn_scr + wn_block_scratch_floats(&e->wn, T);
    memset(cond_zero, 0, (size_t)e->gin_channels * sizeof(float));

    /* pre: Conv1d k=1 [H, spec_ch] */
    conv1d(spec, e->spec_channels, T, e->pre_w, H, 1, e->pre_b, 0, 1, x, NULL);

    /* WN with zero conditioning (zero_g=True at PosteriorEncoder level). */
    wn_block_forward(&e->wn, x, T, cond_zero, wn_out, wn_scr);

    /* proj: Conv1d k=1 [2*inter, H]. tau=0 → we only need m (first inter_channels
       rows). Row-major weight layout means rows [0, inter) are a contiguous
       [inter, H] slice — write directly into z, halving both the projection
       compute and the memcpy. */
    conv1d(wn_out, H, T, e->proj_w, e->inter_channels, 1, e->proj_b, 0, 1, z, NULL);
}
