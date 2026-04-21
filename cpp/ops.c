#include "ops.h"
#include "simd_math.h"

#include <cblas.h>
#include <immintrin.h>
#include <math.h>
#include <stddef.h>

void ops_sgemm(const float *A, const float *B, float *C,
               int M, int N, int K, float alpha, float beta) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, alpha, A, K, B, N, beta, C, N);
}

void ops_sgemm_nt(const float *A, const float *B, float *C,
                  int M, int N, int K, float alpha, float beta) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                M, N, K, alpha, A, K, B, K, beta, C, N);
}

void ops_sgemm_tn(const float *A, const float *B, float *C,
                  int M, int N, int K, float alpha, float beta) {
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                M, N, K, alpha, A, M, B, N, beta, C, N);
}

void ops_vec_add(float *dst, const float *src, size_t n) {
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 d = _mm256_loadu_ps(dst + i);
        __m256 s = _mm256_loadu_ps(src + i);
        _mm256_storeu_ps(dst + i, _mm256_add_ps(d, s));
    }
    for (; i < n; i++) dst[i] += src[i];
}

void ops_vec_sub(float *dst, const float *src, size_t n) {
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 d = _mm256_loadu_ps(dst + i);
        __m256 s = _mm256_loadu_ps(src + i);
        _mm256_storeu_ps(dst + i, _mm256_sub_ps(d, s));
    }
    for (; i < n; i++) dst[i] -= src[i];
}

void ops_vec_scale(float *dst, const float *src, float scale, size_t n) {
    __m256 vs = _mm256_set1_ps(scale);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 s = _mm256_loadu_ps(src + i);
        _mm256_storeu_ps(dst + i, _mm256_mul_ps(s, vs));
    }
    for (; i < n; i++) dst[i] = src[i] * scale;
}

void ops_vec_mul(float *dst, const float *src, size_t n) {
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 d = _mm256_loadu_ps(dst + i);
        __m256 s = _mm256_loadu_ps(src + i);
        _mm256_storeu_ps(dst + i, _mm256_mul_ps(d, s));
    }
    for (; i < n; i++) dst[i] *= src[i];
}

void ops_bias_add_ct(float *dst, const float *bias, int c_count, int frame_len) {
    for (int c = 0; c < c_count; c++) {
        __m256 vb = _mm256_set1_ps(bias[c]);
        float *row = dst + (size_t)c * frame_len;
        int f = 0;
        for (; f + 8 <= frame_len; f += 8) {
            __m256 v = _mm256_loadu_ps(row + f);
            _mm256_storeu_ps(row + f, _mm256_add_ps(v, vb));
        }
        for (; f < frame_len; f++) row[f] += bias[c];
    }
}

void ops_leaky_relu(float *x, float slope, size_t n) {
    __m256 vs = _mm256_set1_ps(slope);
    __m256 vz = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);
        __m256 neg = _mm256_mul_ps(v, vs);
        __m256 mask = _mm256_cmp_ps(v, vz, _CMP_GT_OQ);
        _mm256_storeu_ps(x + i, _mm256_blendv_ps(neg, v, mask));
    }
    for (; i < n; i++) x[i] = x[i] > 0 ? x[i] : x[i] * slope;
}

void ops_leaky_relu_copy(float *dst, const float *src, float slope, size_t n) {
    __m256 vs = _mm256_set1_ps(slope);
    __m256 vz = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(src + i);
        __m256 neg = _mm256_mul_ps(v, vs);
        __m256 mask = _mm256_cmp_ps(v, vz, _CMP_GT_OQ);
        _mm256_storeu_ps(dst + i, _mm256_blendv_ps(neg, v, mask));
    }
    for (; i < n; i++) { float v = src[i]; dst[i] = v > 0 ? v : v * slope; }
}

void ops_clamp_nan(float *x, size_t n) {
    /* NaN check via self-compare (NaN != NaN). Clamp to [-1, 1] with min/max. */
    const __m256 p1 = _mm256_set1_ps( 1.0f);
    const __m256 n1 = _mm256_set1_ps(-1.0f);
    const __m256 zero = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);
        /* nan_mask: NaN lanes → 0xFFFFFFFF */
        __m256 nan_mask = _mm256_cmp_ps(v, v, _CMP_UNORD_Q);
        v = _mm256_blendv_ps(v, zero, nan_mask);
        v = _mm256_min_ps(p1, _mm256_max_ps(n1, v));
        _mm256_storeu_ps(x + i, v);
    }
    for (; i < n; i++) {
        float v = x[i];
        if (v != v) v = 0.0f;
        if (v >  1.0f) v =  1.0f;
        if (v < -1.0f) v = -1.0f;
        x[i] = v;
    }
}

void ops_broadcast_add_ct(float *dst, const float *v, int c_count, int frame_len) {
    for (int c = 0; c < c_count; c++) {
        __m256 vv = _mm256_set1_ps(v[c]);
        float *row = dst + (size_t)c * frame_len;
        int f = 0;
        for (; f + 8 <= frame_len; f += 8) {
            __m256 a = _mm256_loadu_ps(row + f);
            _mm256_storeu_ps(row + f, _mm256_add_ps(a, vv));
        }
        for (; f < frame_len; f++) row[f] += v[c];
    }
}

void ops_tanhf(float *x, size_t n) {
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);
        _mm256_storeu_ps(x + i, simd_tanhf(v));
    }
    for (; i < n; i++) x[i] = tanhf(x[i]);
}

/* Fused gated activation:
     out[i] = tanh(a[i] + b[i]) * sigmoid(a[i+n] + b[i+n])
   SIMD tanh/sigmoid from simd_math.h (matches libm to ~1 ULP). */
void ops_gated_tanh_sigmoid(const float *a, const float *b, float *out, size_t n) {
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 a0 = _mm256_loadu_ps(a + i);
        __m256 b0 = _mm256_loadu_ps(b + i);
        __m256 a1 = _mm256_loadu_ps(a + n + i);
        __m256 b1 = _mm256_loadu_ps(b + n + i);
        __m256 t = simd_tanhf(_mm256_add_ps(a0, b0));
        __m256 s = simd_sigmoidf(_mm256_add_ps(a1, b1));
        _mm256_storeu_ps(out + i, _mm256_mul_ps(t, s));
    }
    for (; i < n; i++) {
        float t = tanhf(a[i] + b[i]);
        float s_x = a[i + n] + b[i + n];
        float s   = 1.0f / (1.0f + expf(-s_x));
        out[i]    = t * s;
    }
}

void ops_layer_norm(float *x, const float *weight, const float *bias,
                    int n_rows, int n_features, float eps) {
    const float inv_n = 1.0f / (float)n_features;
    for (int r = 0; r < n_rows; r++) {
        float *row = x + (size_t)r * n_features;
        /* mean */
        __m256 vsum = _mm256_setzero_ps();
        int i = 0;
        for (; i + 8 <= n_features; i += 8) {
            vsum = _mm256_add_ps(vsum, _mm256_loadu_ps(row + i));
        }
        float sum[8];
        _mm256_storeu_ps(sum, vsum);
        float s = sum[0]+sum[1]+sum[2]+sum[3]+sum[4]+sum[5]+sum[6]+sum[7];
        for (; i < n_features; i++) s += row[i];
        float mean = s * inv_n;

        /* variance */
        __m256 vmean = _mm256_set1_ps(mean);
        __m256 vsq = _mm256_setzero_ps();
        i = 0;
        for (; i + 8 <= n_features; i += 8) {
            __m256 d = _mm256_sub_ps(_mm256_loadu_ps(row + i), vmean);
            vsq = _mm256_add_ps(vsq, _mm256_mul_ps(d, d));
        }
        float sq[8];
        _mm256_storeu_ps(sq, vsq);
        float v = sq[0]+sq[1]+sq[2]+sq[3]+sq[4]+sq[5]+sq[6]+sq[7];
        for (; i < n_features; i++) { float d = row[i] - mean; v += d * d; }
        float var = v * inv_n;
        float inv_std = 1.0f / sqrtf(var + eps);

        /* apply */
        __m256 vinv = _mm256_set1_ps(inv_std);
        i = 0;
        for (; i + 8 <= n_features; i += 8) {
            __m256 vx = _mm256_loadu_ps(row + i);
            __m256 vw = _mm256_loadu_ps(weight + i);
            __m256 vb = _mm256_loadu_ps(bias + i);
            __m256 vd = _mm256_sub_ps(vx, vmean);
            __m256 vn = _mm256_mul_ps(vd, vinv);
            _mm256_storeu_ps(row + i, _mm256_add_ps(_mm256_mul_ps(vn, vw), vb));
        }
        for (; i < n_features; i++) {
            row[i] = (row[i] - mean) * inv_std * weight[i] + bias[i];
        }
    }
}
