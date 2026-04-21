#ifndef LILAC_OPS_H
#define LILAC_OPS_H

#include <stddef.h>

/* ---------------------------------------------------------------------
   Matrix multiply wrappers over OpenBLAS cblas_sgemm, row-major.
   --------------------------------------------------------------------- */

/* C[M,N] = alpha * A[M,K] * B[K,N] + beta * C[M,N] */
void ops_sgemm(const float *A, const float *B, float *C,
               int M, int N, int K, float alpha, float beta);

/* C[M,N] = alpha * A[M,K] * B[N,K]^T + beta * C[M,N]
   (B stored row-major [N,K]; useful when W is laid out [out,in] and we want y = x @ W^T.) */
void ops_sgemm_nt(const float *A, const float *B, float *C,
                  int M, int N, int K, float alpha, float beta);

/* C[M,N] = alpha * A[K,M]^T * B[K,N] + beta * C[M,N]
   (A stored row-major [K,M]; used by ConvTranspose1d.) */
void ops_sgemm_tn(const float *A, const float *B, float *C,
                  int M, int N, int K, float alpha, float beta);

/* ---------------------------------------------------------------------
   Vector primitives — AVX2 fast paths with scalar tail.
   --------------------------------------------------------------------- */

/* dst[i] += src[i] */
void ops_vec_add(float *dst, const float *src, size_t n);

/* dst[i] -= src[i] */
void ops_vec_sub(float *dst, const float *src, size_t n);

/* dst[i] = src[i] * scale */
void ops_vec_scale(float *dst, const float *src, float scale, size_t n);

/* dst[i] *= src[i]  (element-wise, e.g. applying a mask) */
void ops_vec_mul(float *dst, const float *src, size_t n);

/* Single-pass fused leaky_relu + copy: dst[i] = src[i] > 0 ? src[i] : src[i]*slope
   Replaces (memcpy(dst, src) + ops_leaky_relu(dst)) pair with one read+write. */
void ops_leaky_relu_copy(float *dst, const float *src, float slope, size_t n);

/* Clamp to [-1, 1] + turn NaN into 0. In-place. */
void ops_clamp_nan(float *x, size_t n);

/* Broadcast a per-channel scalar add: for c in [0, C), row(c)[0..T) += v[c].
   Same layout as ops_bias_add_ct but takes an arbitrary vector. */
void ops_broadcast_add_ct(float *dst, const float *v, int c_count, int frame_len);

/* In-place tanhf over a flat buffer. */
void ops_tanhf(float *x, size_t n);

/* dst[i] = x[i] + bias[c] where i = c*frame + f, length (c_count * frame)
   Typical use: conv bias broadcast over time dim. */
void ops_bias_add_ct(float *dst, const float *bias, int c_count, int frame_len);

/* LeakyReLU(slope=0.1 default in lilac). In-place. */
void ops_leaky_relu(float *x, float slope, size_t n);

/* out[i] = tanh(a[i] + b[i]) * sigmoid(a[i+N] + b[i+N])  for i in [0, N)
   Inputs a, b have 2*N elements each; output out has N.
   (Matches vc/commons.py's fused_add_tanh_sigmoid_multiply.) */
void ops_gated_tanh_sigmoid(const float *a, const float *b, float *out, size_t n);

/* LayerNorm over the last dim. x is [n_rows, n_features], normalized per row.
   In-place. weight and bias are [n_features] (affine transform). */
void ops_layer_norm(float *x, const float *weight, const float *bias,
                    int n_rows, int n_features, float eps);

#endif
