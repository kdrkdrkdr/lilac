#include "ref_enc.h"
#include "conv.h"
#include "gru.h"
#include "ops.h"
#include "tensor.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const int FILTERS[6] = {32, 32, 64, 64, 128, 128};

#include <immintrin.h>

/* torch.relu in-place (AVX2). */
static void relu_(float *x, size_t n) {
    const __m256 z = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);
        _mm256_storeu_ps(x + i, _mm256_max_ps(v, z));
    }
    for (; i < n; i++) if (x[i] < 0) x[i] = 0;
}

static int conv_dims_after_6_stride2(int L) {
    for (int i = 0; i < 6; i++) L = (L + 2 - 3) / 2 + 1;
    return L;
}

int ref_enc_init(RefEnc *r, const TensorStore *store,
                 int spec_channels, int gin_channels) {
    memset(r, 0, sizeof(*r));
    r->spec_channels = spec_channels;
    r->gin_channels  = gin_channels;
    r->gru_hidden    = 128;
    r->freq_reduced  = conv_dims_after_6_stride2(spec_channels);
    r->gru_input     = 128 * r->freq_reduced;

    const Tensor *t;
    t = tensor_get(store, "ref_enc.layernorm.weight"); if (!t) return -1; r->ln_w = t->data;
    t = tensor_get(store, "ref_enc.layernorm.bias");   if (!t) return -1; r->ln_b = t->data;

    char name[128];
    for (int i = 0; i < 6; i++) {
        snprintf(name, sizeof(name), "ref_enc.convs.%d.weight", i);
        t = tensor_get(store, name); if (!t) return -1; r->conv_w[i] = t->data;
        snprintf(name, sizeof(name), "ref_enc.convs.%d.bias", i);
        t = tensor_get(store, name); if (!t) return -1; r->conv_b[i] = t->data;
    }

    t = tensor_get(store, "ref_enc.gru.weight_ih_l0"); if (!t) return -1; r->gru_W_ih = t->data;
    t = tensor_get(store, "ref_enc.gru.weight_hh_l0"); if (!t) return -1; r->gru_W_hh = t->data;
    t = tensor_get(store, "ref_enc.gru.bias_ih_l0");   if (!t) return -1; r->gru_b_ih = t->data;
    t = tensor_get(store, "ref_enc.gru.bias_hh_l0");   if (!t) return -1; r->gru_b_hh = t->data;

    t = tensor_get(store, "ref_enc.proj.weight"); if (!t) return -1; r->proj_w = t->data;
    t = tensor_get(store, "ref_enc.proj.bias");   if (!t) return -1; r->proj_b = t->data;

    return 0;
}

/* Walk the 6-layer Conv2d chain symbolically to compute max activation and
   max im2col workspace needed. Precisely tracks per-layer C/H/W. */
static void ref_enc_dims(const RefEnc *r, int T_max,
                         long *buf_max, long *im2col_max,
                         int *out_Tp, int *out_WR) {
    int C = 1, H = T_max, W = r->spec_channels;
    long bmax = (long)H * W;       /* initial input holds T*W floats */
    long imax = 0;
    for (int i = 0; i < 6; i++) {
        int H_out = (H + 2 - 3) / 2 + 1;
        int W_out = (W + 2 - 3) / 2 + 1;
        long im_sz  = (long)C * 9 * H_out * W_out;
        long out_sz = (long)FILTERS[i] * H_out * W_out;
        if (im_sz  > imax) imax = im_sz;
        if (out_sz > bmax) bmax = out_sz;
        C = FILTERS[i]; H = H_out; W = W_out;
    }
    *buf_max    = bmax;
    *im2col_max = imax;
    *out_Tp     = H;
    *out_WR     = W;
}

int ref_enc_scratch_floats(const RefEnc *r, int T_max) {
    long bmax, imax;
    int Tp, WR;
    ref_enc_dims(r, T_max, &bmax, &imax, &Tp, &WR);
    long gru_in_sz  = (long)Tp * 128 * WR;
    long gru_scr_sz = (long)Tp * 3 * r->gru_hidden + 3 * r->gru_hidden;
    /* Layout: buf_a, buf_b, im2col, gru_in, h[128], gru_scratch */
    return (int)(bmax + bmax + imax + gru_in_sz + 128 + gru_scr_sz + 1024);
}

void ref_enc_forward(const RefEnc *r, float *spec, int T,
                     float *out, float *scratch) {
    int W0 = r->spec_channels;

    long bmax, imax;
    int  Tp, WR;
    ref_enc_dims(r, T, &bmax, &imax, &Tp, &WR);

    float *buf_a      = scratch;
    float *buf_b      = buf_a  + bmax;
    float *im2col_buf = buf_b  + bmax;
    float *gru_in     = im2col_buf + imax;
    float *h          = gru_in + (size_t)Tp * 128 * WR;
    float *gru_scr    = h + 128;

    /* Transpose spec [spec_ch, T] to buf_a [T, spec_ch] (= [C=1, H=T, W=spec_ch]). */
    for (int c = 0; c < W0; c++) {
        for (int t = 0; t < T; t++) buf_a[(size_t)t * W0 + c] = spec[(size_t)c * T + t];
    }

    /* LayerNorm over last dim. */
    ops_layer_norm(buf_a, r->ln_w, r->ln_b, T, W0, 1e-5f);

    int C_in = 1, H = T, Wd = W0;
    float *cur = buf_a;
    float *nxt = buf_b;
    for (int i = 0; i < 6; i++) {
        int C_out = FILTERS[i];
        int H_out = (H + 2 - 3) / 2 + 1;
        int W_out = (Wd + 2 - 3) / 2 + 1;
        conv2d(cur, C_in, H, Wd, r->conv_w[i], C_out, 3, 3, r->conv_b[i],
               1, 1, 2, 2, nxt, im2col_buf);
        relu_(nxt, (size_t)C_out * H_out * W_out);
        float *tmp = cur; cur = nxt; nxt = tmp;
        C_in = C_out; H = H_out; Wd = W_out;
    }

    /* cur holds [128, Tp, WR]; transpose to gru_in[Tp, 128 * WR]. */
    for (int t = 0; t < Tp; t++) {
        for (int c = 0; c < 128; c++) {
            for (int w = 0; w < WR; w++) {
                gru_in[(size_t)t * 128 * WR + c * WR + w]
                    = cur[((size_t)c * Tp + t) * WR + w];
            }
        }
    }

    memset(h, 0, 128 * sizeof(float));
    gru_forward_last(gru_in, Tp, 128 * WR, 128,
                     r->gru_W_ih, r->gru_W_hh, r->gru_b_ih, r->gru_b_hh,
                     h, gru_scr);

    ops_sgemm(r->proj_w, h, out, r->gin_channels, 1, 128, 1.0f, 0.0f);
    for (int i = 0; i < r->gin_channels; i++) out[i] += r->proj_b[i];
}
