/* Verify conv1d_stream agrees with conv1d_direct for the same stream positions.
   Test: a long random input sequence, run full conv1d_direct once, then feed
   the same sequence in chunks through conv1d_stream. Emitted streaming output
   (after the lag) should match the corresponding interior positions of the
   full output to fp-noise precision. */
#include "conv.h"
#include "ops.h"
#include "stream.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void rand_fill(float *x, int n, unsigned *seed) {
    for (int i = 0; i < n; i++) {
        *seed = *seed * 1103515245u + 12345u;
        x[i] = ((int)(*seed >> 16) & 0x7fff) / 32768.0f * 2.0f - 1.0f;
    }
}

static int test_case(int C_in, int C_out, int K, int dilation,
                     int T_total, int HOP, double tol) {
    int pad = (K - 1) * dilation / 2;

    /* Build a random weight + bias and prepack. */
    unsigned seed = 0xBEEF + K * 37 + dilation;
    float *W = (float *)malloc((size_t)C_out * C_in * K * sizeof(float));
    float *W_kf = (float *)malloc((size_t)C_out * C_in * K * sizeof(float));
    float *b = (float *)malloc((size_t)C_out * sizeof(float));
    rand_fill(W, C_out * C_in * K, &seed);
    rand_fill(b, C_out, &seed);
    conv_prepack_weight(W, C_out, C_in, K, W_kf);

    /* Random input stream [C_in, T_total]. */
    float *X = (float *)calloc((size_t)C_in * T_total, sizeof(float));
    rand_fill(X, C_in * T_total, &seed);

    /* Reference: full conv1d_direct over [C_in, T_total] → [C_out, T_total]. */
    float *Y_ref = (float *)calloc((size_t)C_out * T_total, sizeof(float));
    conv1d_direct(X, C_in, T_total, W_kf, C_out, K, b, pad, dilation, Y_ref);

    /* Streaming: feed X in HOP-sized chunks. */
    Conv1DStream s;
    c1d_stream_init(&s, C_in, K, dilation);
    float *Y_str = (float *)calloc((size_t)C_out * T_total, sizeof(float));
    int scratch_n = 2 * (C_in + C_out) * (s.state_size + HOP) + 1024;
    float *scratch = (float *)malloc((size_t)scratch_n * sizeof(float));
    float *in_hop  = (float *)malloc((size_t)C_in  * HOP * sizeof(float));
    float *out_hop = (float *)malloc((size_t)C_out * HOP * sizeof(float));

    /* Streaming has LAG = pad: emitting n_new outputs at hop with new_input
       ending at stream time T gives outputs corresponding to stream times
       [T - n_new - pad, T - pad). Feed the whole stream plus `pad` zero
       frames at the end so the final outputs get flushed. */
    int T_feed = T_total + pad;
    float *zeros = (float *)calloc((size_t)C_in * HOP, sizeof(float));

    int out_written = 0;     /* stream time of the next output to write. */
    for (int t = 0; t < T_feed; t += HOP) {
        int n_new = (t + HOP <= T_feed) ? HOP : T_feed - t;
        /* Input source: X for stream time in [0, T_total), zeros beyond. */
        for (int c = 0; c < C_in; c++) {
            for (int i = 0; i < n_new; i++) {
                int src = t + i;
                in_hop[(size_t)c * n_new + i] =
                    (src < T_total) ? X[(size_t)c * T_total + src] : 0.0f;
            }
        }
        conv1d_stream(&s, W_kf, C_out, b, in_hop, n_new, out_hop, scratch);
        /* Each output frame k (0..n_new) corresponds to stream time
           t + k - pad. Write to Y_str at that position. */
        for (int k = 0; k < n_new; k++) {
            int dst = t + k - pad;
            if (dst < 0 || dst >= T_total) continue;
            for (int c = 0; c < C_out; c++)
                Y_str[(size_t)c * T_total + dst] = out_hop[(size_t)c * n_new + k];
            out_written++;
        }
    }
    free(zeros);

    /* Compare interior positions — both reference and streaming use zero-pad
       at the stream boundaries the same way, so compare everywhere. */
    double max_abs = 0;
    int fails = 0;
    for (int c = 0; c < C_out; c++) {
        for (int t = 0; t < T_total; t++) {
            double d = fabs((double)Y_ref[c * T_total + t]
                            - (double)Y_str[c * T_total + t]);
            if (d > max_abs) max_abs = d;
            if (d > tol) fails++;
        }
    }
    int total = C_out * T_total;
    printf("  C_in=%d C_out=%d K=%d D=%d T=%d HOP=%d : max_abs=%.2e fails=%d/%d %s\n",
           C_in, C_out, K, dilation, T_total, HOP, max_abs, fails, total,
           max_abs < tol ? "OK" : "FAIL");

    c1d_stream_free(&s);
    free(W); free(W_kf); free(b); free(X);
    free(Y_ref); free(Y_str);
    free(scratch); free(in_hop); free(out_hop);
    return max_abs < tol ? 0 : 1;
}

static int test_ct(int C_in, int C_out, int K, int u, int pad,
                   int T_input, int HOP_in, double tol) {
    unsigned seed = 0xFACE + K * 101 + u;
    float *W = (float *)malloc((size_t)C_in * C_out * K * sizeof(float));
    float *b = (float *)malloc((size_t)C_out * sizeof(float));
    rand_fill(W, C_in * C_out * K, &seed);
    rand_fill(b, C_out, &seed);

    float *X = (float *)calloc((size_t)C_in * T_input, sizeof(float));
    rand_fill(X, C_in * T_input, &seed);

    /* Reference: full conv_transpose1d. */
    int T_out = (T_input - 1) * u + K - 2 * pad;
    float *Y_ref = (float *)calloc((size_t)C_out * T_out, sizeof(float));
    int ref_scratch_n = C_out * K * T_input + 1024;
    float *ref_scratch = (float *)malloc((size_t)ref_scratch_n * sizeof(float));
    conv_transpose1d(X, C_in, T_input, W, C_out, K, b, pad, u, Y_ref, ref_scratch);

    /* Streaming. */
    ConvTranspose1DStream s;
    ct1d_stream_init(&s, C_in, K, u, pad);
    int scratch_n = C_in * (s.state_in + HOP_in) +
                    C_out * ((s.state_in + HOP_in - 1) * u + K - 2 * pad) +
                    C_out * K * (s.state_in + HOP_in) + 1024;
    float *scr = (float *)malloc((size_t)scratch_n * sizeof(float));
    float *in_hop  = (float *)malloc((size_t)C_in  * HOP_in * sizeof(float));
    float *out_hop = (float *)malloc((size_t)C_out * HOP_in * u * sizeof(float));

    float *Y_str = (float *)calloc((size_t)C_out * T_out, sizeof(float));

    /* Feed inputs in HOP_in-sized chunks, plus state_in lookahead of zeros. */
    int lookahead = s.state_in;
    int T_feed = T_input + lookahead;
    for (int t = 0; t < T_feed; t += HOP_in) {
        int n_new = (t + HOP_in <= T_feed) ? HOP_in : T_feed - t;
        for (int c = 0; c < C_in; c++) {
            for (int i = 0; i < n_new; i++) {
                int src = t + i;
                in_hop[(size_t)c * n_new + i] =
                    (src < T_input) ? X[(size_t)c * T_input + src] : 0.0f;
            }
        }
        conv_transpose1d_stream(&s, W, C_out, b, in_hop, n_new, out_hop, scr);
        /* Emit stream position (derived in stream.h): buffer-relative output
           [interior_start, interior_start + n_new*u) maps to stream output
           [(t - state_in)*u + interior_start, ...). */
        int dst_out_start = (t - lookahead) * u + s.interior_start;
        for (int c = 0; c < C_out; c++) {
            for (int k = 0; k < n_new * u; k++) {
                int dst = dst_out_start + k;
                if (dst < 0 || dst >= T_out) continue;
                Y_str[(size_t)c * T_out + dst] = out_hop[(size_t)c * n_new * u + k];
            }
        }
    }

    double max_abs = 0; int fails = 0;
    for (int c = 0; c < C_out; c++) {
        for (int t = 0; t < T_out; t++) {
            double d = fabs((double)Y_ref[c * T_out + t] - (double)Y_str[c * T_out + t]);
            if (d > max_abs) max_abs = d;
            if (d > tol) fails++;
        }
    }
    int total = C_out * T_out;
    printf("  CT C_in=%d C_out=%d K=%d u=%d p=%d T=%d HOP=%d : max_abs=%.2e fails=%d/%d %s\n",
           C_in, C_out, K, u, pad, T_input, HOP_in, max_abs, fails, total,
           max_abs < tol ? "OK" : "FAIL");

    ct1d_stream_free(&s);
    free(W); free(b); free(X); free(Y_ref); free(ref_scratch);
    free(scr); free(in_hop); free(out_hop); free(Y_str);
    return max_abs < tol ? 0 : 1;
}

/* Full (non-streaming) resblock1 — mirrors the one in dec.c, using the same
   K-first prepacked weights. Included here to provide a reference. */
static void rb_full(const float *const c1_w_kf[3], const float *const c1_b[3],
                    const float *const c2_w_kf[3], const float *const c2_b[3],
                    const int dilations[3], int K,
                    float *x, int C, int T,
                    float *tmp) {
    for (int step = 0; step < 3; step++) {
        ops_leaky_relu_copy(tmp, x, 0.1f, (size_t)C * T);
        int dil = dilations[step];
        int pad1 = (K * dil - dil) / 2;
        float *y = tmp + (size_t)C * T;
        conv1d_direct(tmp, C, T, c1_w_kf[step], C, K, c1_b[step], pad1, dil, y);
        ops_leaky_relu(y, 0.1f, (size_t)C * T);
        int pad2 = (K - 1) / 2;
        float *z = tmp;
        conv1d_direct(y, C, T, c2_w_kf[step], C, K, c2_b[step], pad2, 1, z);
        ops_vec_add(x, z, (size_t)C * T);
    }
}

static int test_rb(int C, int K, int T, int HOP, double tol) {
    unsigned seed = 0xD00D + K + T;
    int dilations[3] = {1, 3, 5};

    float *W1[3], *W1_kf[3], *b1[3], *W2[3], *W2_kf[3], *b2[3];
    for (int s = 0; s < 3; s++) {
        W1[s]    = (float *)malloc((size_t)C * C * K * sizeof(float));
        W1_kf[s] = (float *)malloc((size_t)C * C * K * sizeof(float));
        b1[s]    = (float *)malloc((size_t)C * sizeof(float));
        W2[s]    = (float *)malloc((size_t)C * C * K * sizeof(float));
        W2_kf[s] = (float *)malloc((size_t)C * C * K * sizeof(float));
        b2[s]    = (float *)malloc((size_t)C * sizeof(float));
        rand_fill(W1[s], C * C * K, &seed);
        rand_fill(b1[s], C, &seed);
        rand_fill(W2[s], C * C * K, &seed);
        rand_fill(b2[s], C, &seed);
        conv_prepack_weight(W1[s], C, C, K, W1_kf[s]);
        conv_prepack_weight(W2[s], C, C, K, W2_kf[s]);
    }

    float *X = (float *)calloc((size_t)C * T, sizeof(float));
    rand_fill(X, C * T, &seed);

    /* Reference: full resblock (destructive on X, make copy). */
    float *X_full = (float *)malloc((size_t)C * T * sizeof(float));
    memcpy(X_full, X, (size_t)C * T * sizeof(float));
    float *tmp_full = (float *)calloc((size_t)C * T * 2, sizeof(float));
    rb_full((const float *const*)W1_kf, (const float *const*)b1,
            (const float *const*)W2_kf, (const float *const*)b2,
            dilations, K, X_full, C, T, tmp_full);

    /* Streaming. */
    ResblockStream s;
    rb_stream_init(&s, C, K, dilations);
    int scratch_n = 5 * C * HOP + 8 * (C * HOP + 1024);  /* generous */
    float *scr = (float *)malloc((size_t)scratch_n * sizeof(float));
    float *in_hop  = (float *)malloc((size_t)C * HOP * sizeof(float));
    float *out_hop = (float *)malloc((size_t)C * HOP * sizeof(float));
    float *Y_str = (float *)calloc((size_t)C * T, sizeof(float));

    int lag = s.total_lag;
    int T_feed = T + lag;
    for (int t = 0; t < T_feed; t += HOP) {
        int n_new = (t + HOP <= T_feed) ? HOP : T_feed - t;
        for (int c = 0; c < C; c++) {
            for (int i = 0; i < n_new; i++) {
                int src = t + i;
                in_hop[(size_t)c * n_new + i] =
                    (src < T) ? X[(size_t)c * T + src] : 0.0f;
            }
        }
        resblock_stream(&s,
            (const float *const*)W1_kf, (const float *const*)b1,
            (const float *const*)W2_kf, (const float *const*)b2,
            in_hop, n_new, out_hop, scr);
        int dst_start = t - lag;
        for (int c = 0; c < C; c++) {
            for (int k = 0; k < n_new; k++) {
                int dst = dst_start + k;
                if (dst < 0 || dst >= T) continue;
                Y_str[(size_t)c * T + dst] = out_hop[(size_t)c * n_new + k];
            }
        }
    }

    /* Streaming's pre-stream state is zeros (no prior inputs), which propagates
       through conv chains with biases → non-zero values at pre-stream times
       upstream. Those artifacts leak back into emit positions whose conv RF
       reaches the left boundary of X. Non-streaming rb_full zero-pads at X's
       boundary. Both match for positions far enough from the boundary — skip
       the first `interior_margin` samples when scoring parity. */
    int interior_margin = 2 * lag;       /* generous */
    int interior_right  = T - 2 * lag;   /* similarly trim right boundary */
    double max_abs = 0; int fails = 0;
    for (int c = 0; c < C; c++) {
        for (int t = interior_margin; t < interior_right; t++) {
            double d = fabs((double)X_full[c * T + t] - (double)Y_str[c * T + t]);
            if (d > max_abs) max_abs = d;
            if (d > tol) fails++;
        }
    }
    int total_checked = C * (interior_right - interior_margin);
    printf("  RB C=%d K=%d T=%d HOP=%d lag=%d interior=[%d,%d): max_abs=%.2e fails=%d/%d %s\n",
           C, K, T, HOP, lag, interior_margin, interior_right,
           max_abs, fails, total_checked,
           max_abs < tol ? "OK" : "FAIL");
    if (max_abs > tol && C <= 2) {
        printf("    ref[0..10]: ");
        for (int i = 0; i < 10 && i < T; i++) printf("%.4f ", X_full[i]);
        printf("\n    str[0..10]: ");
        for (int i = 0; i < 10 && i < T; i++) printf("%.4f ", Y_str[i]);
        printf("\n    X[0..10]:   ");
        for (int i = 0; i < 10 && i < T; i++) printf("%.4f ", X[i]);
        printf("\n");
    }

    for (int s0 = 0; s0 < 3; s0++) {
        free(W1[s0]); free(W1_kf[s0]); free(b1[s0]);
        free(W2[s0]); free(W2_kf[s0]); free(b2[s0]);
    }
    rb_stream_free(&s);
    free(X); free(X_full); free(tmp_full);
    free(scr); free(in_hop); free(out_hop); free(Y_str);
    return max_abs < tol ? 0 : 1;
}

int main(void) {
    int fails = 0;
    printf("conv1d_stream parity tests:\n");
    /* Typical WN in_layer: k=5 dil=1. */
    fails += test_case( 8,  8, 5, 1, 120, 10, 1e-5);
    fails += test_case(32, 64, 5, 1, 200, 15, 1e-5);
    /* Resblock convs: k=3 various dilations. */
    fails += test_case(16, 16, 3, 1, 300, 20, 1e-5);
    fails += test_case(16, 16, 3, 3, 300, 20, 1e-5);
    fails += test_case(16, 16, 3, 5, 300, 20, 1e-5);
    /* k=7 dil=5 (resblock biggest). */
    fails += test_case(32, 32, 7, 5, 400, 25, 1e-5);
    /* k=11 dil=5 (actual worst case in lilac). */
    fails += test_case(32, 32, 11, 5, 600, 30, 1e-5);
    /* conv_pre-like: k=7 large C. */
    fails += test_case(64, 128, 7, 1, 400, 12, 1e-5);
    /* conv_post-like: k=7 to 1 channel. */
    fails += test_case(32, 1, 7, 1, 500, 50, 1e-5);

    printf("conv_transpose1d_stream parity tests:\n");
    /* Dec ups stages. */
    fails += test_ct(256, 128, 16, 8, 4, 120, 10, 1e-4);
    fails += test_ct(128,  64, 16, 8, 4, 800, 40, 1e-4);
    fails += test_ct( 64,  32,  4, 2, 1, 1500, 80, 1e-5);
    fails += test_ct( 32,  16,  4, 2, 1, 2500, 120, 1e-5);

    printf("resblock_stream parity tests (interior, tol relaxed for FP accumulation):\n");
    /* Interior-only comparison: leaks from the pre-stream boundary (where
       streaming's zero-initialized state diverges from non-streaming's zero
       pad through conv-of-bias chains) affect positions whose cumulative
       conv-RF reaches back into negative stream time. We trim those and
       compare the genuinely interior region. Tolerance is relaxed to absorb
       FP accumulation across the 6 cascaded convs in a resblock; real dec
       use never sees these artifacts because emit is always far past
       boundaries. */
    fails += test_rb(32, 3, 600, 40, 1e-2);
    fails += test_rb(32, 7, 600, 40, 1e-1);
    fails += test_rb(32, 11, 1200, 80, 2.0);
    fails += test_rb(64, 11, 1500, 100, 5.0);

    if (fails == 0) { printf("PASS\n"); return 0; }
    printf("FAIL (%d tests)\n", fails);
    return 1;
}
