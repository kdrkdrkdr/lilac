#include "conv.h"
#include "ops.h"

#include <cblas.h>
#include <stddef.h>
#include <string.h>

/* im2col for 1-D stride-1 convolution.
   col layout after this fills [C_in*K, T_out] row-major:
     col[(c*K + k) * T_out + t] = input[c*T_in + t + k*dilation - pad]  (0 if OOB) */
static void im2col_1d(const float *input, int C_in, int T_in,
                      int K, int pad, int dilation,
                      float *col, int T_out) {
    for (int c = 0; c < C_in; c++) {
        const float *in_c = input + (size_t)c * T_in;
        for (int k = 0; k < K; k++) {
            float *row = col + ((size_t)c * K + k) * T_out;
            int t_in0 = k * dilation - pad;   /* t_in when t_out = 0 */
            /* Split into three ranges: left OOB zero, in-range copy, right OOB zero */
            int t_out = 0;
            /* left zeros: t_in < 0 → t_out < -t_in0 */
            int left = t_in0 < 0 ? (-t_in0) : 0;
            if (left > T_out) left = T_out;
            for (; t_out < left; t_out++) row[t_out] = 0.0f;
            /* in-range: t_in in [0, T_in) */
            int right_start = T_in - t_in0;   /* first t_out with t_in >= T_in */
            if (right_start > T_out) right_start = T_out;
            if (right_start > t_out) {
                memcpy(row + t_out, in_c + (t_in0 + t_out),
                       (size_t)(right_start - t_out) * sizeof(float));
                t_out = right_start;
            }
            /* right zeros */
            for (; t_out < T_out; t_out++) row[t_out] = 0.0f;
        }
    }
}

void conv1d(const float *input, int C_in, int T_in,
            const float *weight, int C_out, int K,
            const float *bias,
            int pad, int dilation,
            float *output, float *scratch) {
    int T_out = T_in + 2 * pad - dilation * (K - 1);
    if (K == 1 && pad == 0) {
        /* Pointwise: weight [C_out, C_in] @ input [C_in, T_in] = output [C_out, T_in] */
        ops_sgemm(weight, input, output, C_out, T_in, C_in, 1.0f, 0.0f);
    } else {
        im2col_1d(input, C_in, T_in, K, pad, dilation, scratch, T_out);
        ops_sgemm(weight, scratch, output, C_out, T_out, C_in * K, 1.0f, 0.0f);
    }
    if (bias) ops_bias_add_ct(output, bias, C_out, T_out);
}

/* im2col for 2-D stride-S, no dilation. Fills col[C_in*kH*kW, H_out*W_out]:
     col[(c*kH + kh)*kW + kw, h*W_out + w]
       = input[c, h*sH + kh - pH, w*sW + kw - pW]   (0 if OOB) */
static void im2col_2d(const float *input, int C_in, int H_in, int W_in,
                      int kH, int kW, int pH, int pW, int sH, int sW,
                      float *col, int H_out, int W_out) {
    const int col_w = H_out * W_out;
    for (int c = 0; c < C_in; c++) {
        const float *in_c = input + (size_t)c * H_in * W_in;
        for (int kh = 0; kh < kH; kh++) {
            for (int kw = 0; kw < kW; kw++) {
                float *row = col + ((size_t)c * kH * kW + kh * kW + kw) * col_w;
                for (int h = 0; h < H_out; h++) {
                    int h_in = h * sH + kh - pH;
                    if (h_in < 0 || h_in >= H_in) {
                        for (int w = 0; w < W_out; w++) row[h * W_out + w] = 0.0f;
                        continue;
                    }
                    const float *in_row = in_c + (size_t)h_in * W_in;
                    for (int w = 0; w < W_out; w++) {
                        int w_in = w * sW + kw - pW;
                        row[h * W_out + w] = (w_in >= 0 && w_in < W_in) ? in_row[w_in] : 0.0f;
                    }
                }
            }
        }
    }
}

void conv2d(const float *input, int C_in, int H_in, int W_in,
            const float *weight, int C_out, int kH, int kW,
            const float *bias,
            int pH, int pW, int sH, int sW,
            float *output, float *scratch) {
    int H_out = (H_in + 2 * pH - kH) / sH + 1;
    int W_out = (W_in + 2 * pW - kW) / sW + 1;
    im2col_2d(input, C_in, H_in, W_in, kH, kW, pH, pW, sH, sW,
              scratch, H_out, W_out);
    ops_sgemm(weight, scratch, output,
              C_out, H_out * W_out, C_in * kH * kW, 1.0f, 0.0f);
    if (bias) ops_bias_add_ct(output, bias, C_out, H_out * W_out);
}

void conv_prepack_weight(const float *w, int C_out, int C_in, int K, float *out) {
    /* w[c_o, c_i, k] → out[k, c_o, c_i] */
    for (int k = 0; k < K; k++) {
        float *dst = out + (size_t)k * C_out * C_in;
        for (int co = 0; co < C_out; co++) {
            const float *src = w + (size_t)co * C_in * K + k;  /* stride K */
            float *drow = dst + (size_t)co * C_in;
            for (int ci = 0; ci < C_in; ci++) drow[ci] = src[(size_t)ci * K];
        }
    }
}

void conv1d_direct(const float *input, int C_in, int T_in,
                   const float *weight_k_first,
                   int C_out, int K,
                   const float *bias,
                   int pad, int dilation,
                   float *output) {
    int T_out = T_in + 2 * pad - dilation * (K - 1);
    size_t out_n = (size_t)C_out * T_out;

    /* Seed output with the bias broadcast (or zeros). Each k-slice sgemm then
       accumulates with beta=1. */
    if (bias) {
        for (int c = 0; c < C_out; c++) {
            float *row = output + (size_t)c * T_out;
            float b = bias[c];
            for (int t = 0; t < T_out; t++) row[t] = b;
        }
    } else {
        memset(output, 0, out_n * sizeof(float));
    }

    for (int k = 0; k < K; k++) {
        int offset = k * dilation - pad;            /* t_in = t_out + offset */
        int t_start = offset < 0 ? -offset : 0;
        int t_end   = T_in - offset;
        if (t_end > T_out) t_end = T_out;
        if (t_end <= t_start) continue;
        int n_valid = t_end - t_start;

        const float *A = weight_k_first + (size_t)k * C_out * C_in;  /* [C_out, C_in], lda=C_in */
        const float *B = input + (size_t)(t_start + offset);         /* first time col; ldb=T_in */
        float       *C = output + t_start;                           /* ldc=T_out */

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    C_out, n_valid, C_in,
                    1.0f, A, C_in,
                          B, T_in,
                    1.0f, C, T_out);
    }
}

void conv_transpose1d(const float *input, int C_in, int T_in,
                      const float *weight, int C_out, int K,
                      const float *bias,
                      int pad, int stride,
                      float *output, float *scratch) {
    int T_out = (T_in - 1) * stride - 2 * pad + K;
    /* Step 1: Z[C_out*K, T_in] = W^T @ input
       W stored as [C_in, C_out*K] (row-major), read as [C_out*K, C_in] via trans. */
    ops_sgemm_tn(weight, input, scratch,
                 C_out * K, T_in, C_in, 1.0f, 0.0f);

    /* Step 2: col2im — scatter-add Z[c_out, k, t_in] into output[c_out, t_in*stride + k - pad].
       Loop order (k, t_in) outer / c_out inner is cache-friendly: both the
       source row (scratch[:, k, t_in]) and destination column (output[:, t_out])
       are unit-stride across c_out once reshaped, so we can AVX2 the add. */
    size_t out_size = (size_t)C_out * T_out;
    memset(output, 0, out_size * sizeof(float));

    for (int k = 0; k < K; k++) {
        int t_out0 = k - pad;
        for (int t_in = 0; t_in < T_in; t_in++) {
            int t_out = t_out0 + t_in * stride;
            if (t_out < 0 || t_out >= T_out) continue;
            /* accumulate scratch[c_out, k, t_in] into output[c_out, t_out]
               for c_out in [0, C_out). Source stride = K*T_in, dest stride = T_out. */
            for (int c = 0; c < C_out; c++) {
                output[(size_t)c * T_out + t_out]
                    += scratch[(size_t)c * K * T_in + (size_t)k * T_in + t_in];
            }
        }
    }

    if (bias) ops_bias_add_ct(output, bias, C_out, T_out);
}
