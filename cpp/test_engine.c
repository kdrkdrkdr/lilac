/* Smoke test for the full engine: create, run one hop, destroy.
   Uses a synthetic target wav so we don't need audio files. */
#include "engine.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

int main(void) {
    /* Synthesize a 2-second 22050 Hz target wav: 440 Hz sine. */
    int sr = 22050;
    int tgt_len = sr * 2;
    float *tgt = (float *)calloc((size_t)tgt_len, sizeof(float));
    for (int i = 0; i < tgt_len; i++) {
        tgt[i] = 0.3f * sinf((float)(2.0 * M_PI * 440.0 * i / sr));
    }

    printf("[1/4] engine_init ...\n");
    fflush(stdout);
    Engine eng;
    int K = 4;
    if (engine_init(&eng, "cpp/weights.bin", tgt, tgt_len, K) != 0) {
        fprintf(stderr, "engine_init failed\n");
        free(tgt);
        return 1;
    }
    printf("  hop=%d window=%d emit_offset=%d\n", eng.hop, eng.window, eng.emit_offset);

    /* Feed enough hops to fill the window, then one more for a real process. */
    int hops_to_fill = eng.window / eng.hop;           /* 12 when K=4 */
    int total_hops   = hops_to_fill + 4;

    float *in_hop = (float *)calloc((size_t)eng.hop, sizeof(float));
    for (int i = 0; i < eng.hop; i++) {
        in_hop[i] = 0.2f * sinf((float)(2.0 * M_PI * 220.0 * i / sr));
    }

    printf("[2/4] warmup %d hops ...\n", hops_to_fill);
    fflush(stdout);
    for (int i = 0; i < hops_to_fill; i++) {
        const float *out = engine_process_hop(&eng, in_hop);
        (void)out;
    }

    printf("[3/4] %d real hops ...\n", total_hops - hops_to_fill);
    fflush(stdout);
    for (int i = hops_to_fill; i < total_hops; i++) {
        const float *out = engine_process_hop(&eng, in_hop);
        double peak = 0, rms = 0;
        for (int k = 0; k < eng.hop; k++) {
            double v = out[k];
            if (fabs(v) > peak) peak = fabs(v);
            rms += v * v;
        }
        rms = sqrt(rms / eng.hop);
        printf("  hop %2d  peak=%.4f rms=%.4f\n", i, peak, rms);
        fflush(stdout);
    }

    printf("[4/4] engine_free ...\n");
    fflush(stdout);
    engine_free(&eng);
    free(in_hop);
    free(tgt);
    printf("OK\n");
    return 0;
}
