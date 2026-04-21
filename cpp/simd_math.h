#ifndef LILAC_SIMD_MATH_H
#define LILAC_SIMD_MATH_H

/* AVX2 transcendentals matched to libm within ~1-2 ULP. Uses the classic
   range-reduction + minimax-polynomial approach (same family as Cephes /
   Intel SVML). Inline so the compiler can fold into caller hot loops. */

#include <immintrin.h>

/* expf for each lane. Valid for x in about [-87.3, 88.7]; saturates outside. */
static inline __m256 simd_expf(__m256 x) {
    const __m256 cmax = _mm256_set1_ps( 88.72283172607421875f);
    const __m256 cmin = _mm256_set1_ps(-87.33654022216796875f);
    x = _mm256_min_ps(_mm256_max_ps(x, cmin), cmax);

    /* Range-reduce: x = n*ln2 + r, |r| <= ln2/2. Use 2-part ln2 for precision. */
    const __m256 log2e  = _mm256_set1_ps(1.4426950408889634f);
    const __m256 half   = _mm256_set1_ps(0.5f);
    __m256 fn = _mm256_floor_ps(_mm256_fmadd_ps(x, log2e, half));
    __m256i ni = _mm256_cvtps_epi32(fn);

    const __m256 ln2_hi = _mm256_set1_ps(0.693359375f);         /* 0x3f317200, exact */
    const __m256 ln2_lo = _mm256_set1_ps(-2.12194440e-4f);      /* remainder */
    __m256 r = _mm256_fnmadd_ps(fn, ln2_hi, x);
    r = _mm256_fnmadd_ps(fn, ln2_lo, r);

    /* Horner polynomial for exp(r). Degree-6, matches expf to ~1 ULP in the
       reduced range [-ln2/2, ln2/2]. Coeffs from Cephes cephes_expf. */
    __m256 p = _mm256_set1_ps(1.9875691500E-4f);
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.3981999507E-3f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(8.3334519073E-3f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(4.1665795894E-2f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.6666665459E-1f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(5.0000001201E-1f));
    __m256 r2 = _mm256_mul_ps(r, r);
    p = _mm256_fmadd_ps(p, r2, r);
    p = _mm256_add_ps(p, _mm256_set1_ps(1.0f));

    /* 2^n via exponent field: (n + 127) << 23. */
    __m256i bias = _mm256_add_epi32(ni, _mm256_set1_epi32(127));
    __m256 pow2  = _mm256_castsi256_ps(_mm256_slli_epi32(bias, 23));
    return _mm256_mul_ps(p, pow2);
}

/* sigmoid(x) = 1 / (1 + exp(-x)). Uses simd_expf; saturation already handled. */
static inline __m256 simd_sigmoidf(__m256 x) {
    const __m256 one = _mm256_set1_ps(1.0f);
    __m256 negx = _mm256_sub_ps(_mm256_setzero_ps(), x);
    __m256 e    = simd_expf(negx);
    return _mm256_div_ps(one, _mm256_add_ps(one, e));
}

/* tanh(x) = 1 - 2 / (exp(2x) + 1). Computed this way to avoid cancellation near
   zero. For |x| >= ~9 the result saturates to ±1 at fp32 precision. */
static inline __m256 simd_tanhf(__m256 x) {
    const __m256 cmax = _mm256_set1_ps( 9.0f);
    const __m256 cmin = _mm256_set1_ps(-9.0f);
    x = _mm256_min_ps(_mm256_max_ps(x, cmin), cmax);
    __m256 e2 = simd_expf(_mm256_add_ps(x, x));
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 two = _mm256_set1_ps(2.0f);
    return _mm256_sub_ps(one, _mm256_div_ps(two, _mm256_add_ps(e2, one)));
}

#endif
