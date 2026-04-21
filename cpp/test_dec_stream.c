/* Parity test for dec_stream vs dec_forward.
   Build a random z [192, T_total], run dec_forward once to get the reference
   [T_total*256] audio, then feed z in chunks to dec_stream plus a lookahead
   flush of zero-frames covering the cumulative lag. Compare emitted samples
   at stream positions past the lag boundary — they should match within fp
   noise. */
#include "dec.h"
#include "dec_stream.h"
#include "tensor.h"

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

static int run_case(const Dec *d, int T_total, int HOP_Z, double tol,
                    double *out_max_abs) {
    unsigned seed = 0xABCDEF + T_total * 31 + HOP_Z;
    int C_z = d->initial_channel;           /* 192 */

    /* Random z input. */
    float *Z = (float *)calloc((size_t)C_z * T_total, sizeof(float));
    rand_fill(Z, C_z * T_total, &seed);

    /* Scale down — dec_forward with random unit-range inputs saturates tanh
       everywhere and we lose discrimination. Use 0.1 amplitude. */
    for (int i = 0; i < C_z * T_total; i++) Z[i] *= 0.1f;

    /* Reference: full dec_forward. */
    int T_audio = T_total * 256;
    float *Y_ref = (float *)calloc((size_t)T_audio, sizeof(float));
    int dec_scr_n = dec_scratch_floats(d, T_total);
    float *dec_scr = (float *)malloc((size_t)dec_scr_n * sizeof(float));
    dec_forward(d, Z, T_total, Y_ref, dec_scr);

    /* Streaming. */
    DecStream ds;
    if (dec_stream_init(&ds, d) != 0) { fprintf(stderr, "dec_stream_init fail\n"); return 1; }

    long cum_lag = ds.cumulative_lag_audio;
    int lag_z = (int)((cum_lag + 255) / 256);     /* zero-frames to flush tail */
    int T_feed = T_total + lag_z;

    int ds_scr_n = dec_stream_scratch_floats(&ds, HOP_Z);
    float *ds_scr = (float *)malloc((size_t)ds_scr_n * sizeof(float));
    float *z_hop  = (float *)malloc((size_t)C_z * HOP_Z * sizeof(float));
    float *a_hop  = (float *)malloc((size_t)HOP_Z * 256 * sizeof(float));
    float *Y_str  = (float *)calloc((size_t)T_audio, sizeof(float));

    for (int t = 0; t < T_feed; t += HOP_Z) {
        int n_new = (t + HOP_Z <= T_feed) ? HOP_Z : T_feed - t;
        for (int c = 0; c < C_z; c++) {
            for (int i = 0; i < n_new; i++) {
                int src = t + i;
                z_hop[(size_t)c * n_new + i] =
                    (src < T_total) ? Z[(size_t)c * T_total + src] : 0.0f;
            }
        }
        dec_stream_forward(&ds, z_hop, n_new, a_hop, ds_scr);

        /* Each output frame k (0..n_new*256) maps to stream time
           t*256 + k - cum_lag (audio samples). Scatter into Y_str. */
        int dst_base = t * 256 - (int)cum_lag;
        int out_n    = n_new * 256;
        for (int k = 0; k < out_n; k++) {
            int dst = dst_base + k;
            if (dst < 0 || dst >= T_audio) continue;
            Y_str[dst] = a_hop[k];
        }
    }

    /* Compare interior (past left boundary effects + before right boundary). */
    int margin = (int)cum_lag + 2000;   /* generous interior buffer */
    int right  = T_audio - 2000;
    if (right <= margin) { margin = 0; right = T_audio; }
    double max_abs = 0; int fails = 0;
    for (int t = margin; t < right; t++) {
        double dd = fabs((double)Y_ref[t] - (double)Y_str[t]);
        if (dd > max_abs) max_abs = dd;
        if (dd > tol) fails++;
    }
    int total = right - margin;
    printf("  T=%d HOP_Z=%d cum_lag=%ld interior=[%d,%d) : max_abs=%.2e fails=%d/%d %s\n",
           T_total, HOP_Z, cum_lag, margin, right, max_abs, fails, total,
           max_abs < tol ? "OK" : "FAIL");

    if (out_max_abs) *out_max_abs = max_abs;

    dec_stream_free(&ds);
    free(Z); free(Y_ref); free(dec_scr);
    free(ds_scr); free(z_hop); free(a_hop); free(Y_str);
    return max_abs < tol ? 0 : 1;
}

int main(void) {
    TensorStore store;
    if (tensor_store_load(&store, "cpp/weights.bin") != 0) {
        fprintf(stderr, "load weights failed — run dump_weights.py\n");
        return 1;
    }
    Dec d;
    if (dec_init(&d, &store) != 0) { fprintf(stderr, "dec_init fail\n"); return 1; }

    int fails = 0;
    printf("dec_stream parity tests:\n");

    /* Short T with small hops: exercises warmup carefully. */
    fails += run_case(&d, 64, 8, 5e-2, NULL);
    /* Mid T — closer to engine's T (118) minus slice. */
    fails += run_case(&d, 120, 10, 5e-2, NULL);
    /* Large T with various hops. */
    fails += run_case(&d, 200, 16, 5e-2, NULL);
    fails += run_case(&d, 200, 9,  5e-2, NULL);
    /* Irregular hop (the K=4 engine's average feed will be 9-10). */
    fails += run_case(&d, 256, 10, 5e-2, NULL);

    dec_free(&d);
    tensor_store_free(&store);

    if (fails == 0) { printf("PASS\n"); return 0; }
    printf("FAIL (%d cases)\n", fails);
    return 1;
}
