/* Exercises ops.c against small reference computations. Run after
   `make test_ops.exe` in cpp/. Prints PASS/FAIL lines. */
#include "ops.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int approx_eq(float a, float b, float tol) {
    return fabsf(a - b) <= tol;
}

static int test_sgemm(void) {
    /* C = A[2,3] * B[3,4]: classic 2x3 times 3x4. */
    float A[6] = {
        1, 2, 3,
        4, 5, 6,
    };
    float B[12] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
    };
    float C[8] = {0};
    ops_sgemm(A, B, C, 2, 4, 3, 1.0f, 0.0f);
    /* expected: row 0 = [38, 44, 50, 56], row 1 = [83, 98, 113, 128] */
    float expect[8] = {38, 44, 50, 56, 83, 98, 113, 128};
    for (int i = 0; i < 8; i++) {
        if (!approx_eq(C[i], expect[i], 1e-4f)) {
            printf("FAIL test_sgemm at i=%d got=%.3f want=%.3f\n", i, C[i], expect[i]);
            return 1;
        }
    }
    printf("PASS test_sgemm\n");
    return 0;
}

static int test_vec_add(void) {
    float dst[17], src[17];
    for (int i = 0; i < 17; i++) { dst[i] = (float)i; src[i] = (float)(i * 2); }
    ops_vec_add(dst, src, 17);
    for (int i = 0; i < 17; i++) {
        if (!approx_eq(dst[i], (float)(i * 3), 1e-6f)) {
            printf("FAIL test_vec_add at %d\n", i); return 1;
        }
    }
    printf("PASS test_vec_add (n=17, tail 1 element)\n");
    return 0;
}

static int test_leaky_relu(void) {
    float x[8] = {-1, 2, -3, 4, -5, 6, -7, 8};
    ops_leaky_relu(x, 0.1f, 8);
    float expect[8] = {-0.1f, 2, -0.3f, 4, -0.5f, 6, -0.7f, 8};
    for (int i = 0; i < 8; i++) {
        if (!approx_eq(x[i], expect[i], 1e-6f)) {
            printf("FAIL test_leaky_relu at %d got=%.3f want=%.3f\n", i, x[i], expect[i]);
            return 1;
        }
    }
    printf("PASS test_leaky_relu\n");
    return 0;
}

static int test_gated(void) {
    /* n=2, so buffers have 4 elements (2*n). */
    float a[4] = {0.5f, -0.3f, 1.0f, 0.2f};
    float b[4] = {0.1f, 0.4f, -0.5f, 0.8f};
    float out[2];
    ops_gated_tanh_sigmoid(a, b, out, 2);
    /* expected: i=0 -> tanh(0.5+0.1) * sigmoid(1.0-0.5)
                 i=1 -> tanh(-0.3+0.4) * sigmoid(0.2+0.8) */
    float e0 = tanhf(0.6f) * (1.0f / (1.0f + expf(-0.5f)));
    float e1 = tanhf(0.1f) * (1.0f / (1.0f + expf(-1.0f)));
    if (!approx_eq(out[0], e0, 1e-5f) || !approx_eq(out[1], e1, 1e-5f)) {
        printf("FAIL test_gated got=[%.5f, %.5f] want=[%.5f, %.5f]\n",
               out[0], out[1], e0, e1);
        return 1;
    }
    printf("PASS test_gated\n");
    return 0;
}

static int test_bias_add_ct(void) {
    float x[12] = {
        1,1,1,1,   /* c=0 */
        2,2,2,2,   /* c=1 */
        3,3,3,3,   /* c=2 */
    };
    float bias[3] = {10, 20, 30};
    ops_bias_add_ct(x, bias, 3, 4);
    float expect[12] = {11,11,11,11, 22,22,22,22, 33,33,33,33};
    for (int i = 0; i < 12; i++) {
        if (!approx_eq(x[i], expect[i], 1e-6f)) {
            printf("FAIL test_bias_add_ct at %d\n", i); return 1;
        }
    }
    printf("PASS test_bias_add_ct\n");
    return 0;
}

int main(void) {
    int fails = 0;
    fails += test_sgemm();
    fails += test_vec_add();
    fails += test_leaky_relu();
    fails += test_gated();
    fails += test_bias_add_ct();
    if (fails) { printf("\n%d test(s) FAILED\n", fails); return 1; }
    printf("\nall ops tests PASS\n");
    return 0;
}
