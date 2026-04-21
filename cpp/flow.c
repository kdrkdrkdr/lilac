#include "flow.h"
#include "conv.h"
#include "ops.h"
#include "tensor.h"

#include <stdio.h>
#include <string.h>

int flow_init(Flow *f, const TensorStore *store) {
    memset(f, 0, sizeof(*f));
    f->channels        = 192;
    f->half_channels   = 96;
    f->hidden_channels = 192;
    f->kernel_size     = 5;
    f->n_layers        = 4;
    f->gin_channels    = 256;

    /* ResidualCouplingBlock alternates ResidualCouplingLayer + Flip, so the
       coupling layers live at even indices: flow.flows.{0, 2, 4, 6}. */
    int flow_idx[LILAC_FLOW_N_FLOWS] = {0, 2, 4, 6};

    char prefix[128];
    for (int i = 0; i < LILAC_FLOW_N_FLOWS; i++) {
        int idx = flow_idx[i];
        char name[192];
        const Tensor *t;
        snprintf(name, sizeof(name), "flow.flows.%d.pre.weight", idx);
        t = tensor_get(store, name); if (!t) return -1; f->layers[i].pre_w = t->data;
        snprintf(name, sizeof(name), "flow.flows.%d.pre.bias", idx);
        t = tensor_get(store, name); if (!t) return -1; f->layers[i].pre_b = t->data;
        snprintf(name, sizeof(name), "flow.flows.%d.post.weight", idx);
        t = tensor_get(store, name); if (!t) return -1; f->layers[i].post_w = t->data;
        snprintf(name, sizeof(name), "flow.flows.%d.post.bias", idx);
        t = tensor_get(store, name); if (!t) return -1; f->layers[i].post_b = t->data;

        snprintf(prefix, sizeof(prefix), "flow.flows.%d.enc", idx);
        if (wn_block_init(&f->layers[i].wn, store, prefix,
                          f->hidden_channels, f->kernel_size,
                          f->n_layers, f->gin_channels) != 0) return -1;
    }
    return 0;
}

void flow_free(Flow *f) {
    for (int i = 0; i < LILAC_FLOW_N_FLOWS; i++) wn_block_free(&f->layers[i].wn);
}

int flow_scratch_floats(const Flow *f, int T_max) {
    int H = f->hidden_channels;
    /* h_pre [H, T], wn_out [H, T], stats [half, T], wn_scratch, plus a swap
       buffer [channels, T] for Flip (we do it in-place with a temp). */
    int per = H * T_max + H * T_max + f->half_channels * T_max
            + wn_block_scratch_floats(&f->layers[0].wn, T_max);
    int tmp_flip = f->channels * T_max;
    return per + tmp_flip + 64;
}

static void channel_flip(float *x, int C, int T, float *tmp) {
    /* Flip along channel axis: tmp[c] = x[C-1-c], then copy back. */
    for (int c = 0; c < C; c++) {
        memcpy(tmp + (size_t)c * T,
               x   + (size_t)(C - 1 - c) * T,
               (size_t)T * sizeof(float));
    }
    memcpy(x, tmp, (size_t)C * T * sizeof(float));
}

void flow_forward(const Flow *f, float *x, int T, const float *g,
                  int reverse, float *scratch) {
    const int H = f->hidden_channels;
    const int C = f->channels;
    const int HC = f->half_channels;

    float *h_pre   = scratch;
    float *wn_out  = h_pre + (size_t)H * T;
    float *stats   = wn_out + (size_t)H * T;
    float *wn_scr  = stats + (size_t)HC * T;
    float *tmp_flip = wn_scr + wn_block_scratch_floats(&f->layers[0].wn, T);

    /* ResidualCouplingBlock with n_flows=4 (coupling + flip pairs). In forward
       order we iterate i = 0..3 applying coupling then flip; in reverse order
       we iterate i = 3..0 applying flip-inverse then coupling-inverse (flip is
       its own inverse). */
    if (!reverse) {
        for (int i = 0; i < LILAC_FLOW_N_FLOWS; i++) {
            /* Coupling forward: x0 gets pre -> WN -> post = m, x1 += m. */
            float *x0 = x;                  /* [HC, T] */
            float *x1 = x + (size_t)HC * T; /* [HC, T] */
            conv1d(x0, HC, T, f->layers[i].pre_w, H, 1, f->layers[i].pre_b,
                   0, 1, h_pre, NULL);
            wn_block_forward(&f->layers[i].wn, h_pre, T, g, wn_out, wn_scr);
            conv1d(wn_out, H, T, f->layers[i].post_w, HC, 1, f->layers[i].post_b,
                   0, 1, stats, NULL);
            ops_vec_add(x1, stats, (size_t)HC * T);
            /* Flip. */
            channel_flip(x, C, T, tmp_flip);
        }
    } else {
        for (int i = LILAC_FLOW_N_FLOWS - 1; i >= 0; i--) {
            /* Undo flip first. */
            channel_flip(x, C, T, tmp_flip);
            /* Coupling inverse: x1 -= m. */
            float *x0 = x;
            float *x1 = x + (size_t)HC * T;
            conv1d(x0, HC, T, f->layers[i].pre_w, H, 1, f->layers[i].pre_b,
                   0, 1, h_pre, NULL);
            wn_block_forward(&f->layers[i].wn, h_pre, T, g, wn_out, wn_scr);
            conv1d(wn_out, H, T, f->layers[i].post_w, HC, 1, f->layers[i].post_b,
                   0, 1, stats, NULL);
            ops_vec_sub(x1, stats, (size_t)HC * T);
        }
    }
}
