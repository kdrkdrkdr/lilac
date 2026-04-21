/* Validate SIMD tanh/sigmoid/exp against libm. Reports max abs error over a
   dense sweep of the typical activation range [-10, 10]. The VITS decoder
   runs within ±6 in practice. */
#include "simd_math.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static void test_fn(const char *name,
                    float (*ref)(float),
                    __m256 (*simd)(__m256),
                    float lo, float hi, int N, double ulp_fail) {
    float max_abs = 0, max_rel = 0;
    float at_max_abs = 0;
    float x8[8] __attribute__((aligned(32)));
    float y8[8] __attribute__((aligned(32)));
    for (int i = 0; i < N; i += 8) {
        for (int j = 0; j < 8; j++) {
            float t = (float)(i + j) / (float)(N - 1);
            x8[j] = lo + (hi - lo) * t;
        }
        __m256 v = _mm256_load_ps(x8);
        _mm256_store_ps(y8, simd(v));
        for (int j = 0; j < 8; j++) {
            float r = ref(x8[j]);
            float d = fabsf(y8[j] - r);
            float rel = d / (fabsf(r) + 1e-12f);
            if (d > max_abs) { max_abs = d; at_max_abs = x8[j]; }
            if (rel > max_rel) max_rel = rel;
        }
    }
    const char *ok = (max_abs < (float)ulp_fail) ? "OK" : "FAIL";
    printf("  %-9s  max_abs=%.3e (at x=%.3f)  max_rel=%.3e  [%s]\n",
           name, max_abs, at_max_abs, max_rel, ok);
}

static float ref_sigmoidf(float x) { return 1.0f / (1.0f + expf(-x)); }

int main(void) {
    printf("SIMD math vs libm over [-10, 10], 100k samples:\n");
    test_fn("expf",     expf,         simd_expf,     -10.0f,  10.0f, 100000, 5e-5);
    test_fn("tanhf",    tanhf,        simd_tanhf,    -10.0f,  10.0f, 100000, 5e-7);
    test_fn("sigmoidf", ref_sigmoidf, simd_sigmoidf, -10.0f,  10.0f, 100000, 5e-7);
    printf("SIMD math vs libm over [-25, 25]:\n");
    test_fn("expf",     expf,         simd_expf,     -25.0f,  25.0f, 100000, 5e-5);
    test_fn("tanhf",    tanhf,        simd_tanhf,    -25.0f,  25.0f, 100000, 5e-7);
    test_fn("sigmoidf", ref_sigmoidf, simd_sigmoidf, -25.0f,  25.0f, 100000, 5e-7);
    return 0;
}
