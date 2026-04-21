/* Micro-benchmark for engine_process_hop. Loads weights.bin, builds a synthetic
   target, runs N warm-up hops to fill the window, then times M hops. Reports
   avg / min / median / p95 / max and average per-stage breakdown.

   Diagnostics from model.c / dec.c are routed through lilac_log in log.c and
   end up in bench.log next to bench.exe. */
#include "engine.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

static double wall_ms(void) {
    static LARGE_INTEGER freq = {0};
    if (!freq.QuadPart) QueryPerformanceFrequency(&freq);
    LARGE_INTEGER c; QueryPerformanceCounter(&c);
    return (double)c.QuadPart * 1000.0 / (double)freq.QuadPart;
}

static int cmp_double(const void *a, const void *b) {
    double da = *(const double *)a, db = *(const double *)b;
    return (da > db) - (da < db);
}

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

int main(int argc, char **argv) {
    int K = (argc > 1) ? atoi(argv[1]) : 4;
    int N = (argc > 2) ? atoi(argv[2]) : 50;
    const char *weights = "weights.bin";

    int sr = 22050;
    int tgt_len = sr * 2;
    float *tgt = (float *)calloc((size_t)tgt_len, sizeof(float));
    for (int i = 0; i < tgt_len; i++)
        tgt[i] = 0.3f * sinf((float)(2.0 * M_PI * 440.0 * i / sr));

    Engine eng;
    printf("engine_init K=%d ...\n", K);
    if (engine_init(&eng, weights, tgt, tgt_len, K) != 0) {
        fprintf(stderr, "engine_init failed\n"); return 1;
    }
    printf("  hop=%d window=%d\n", eng.hop, eng.window);

    float *in_hop = (float *)calloc((size_t)eng.hop, sizeof(float));
    for (int i = 0; i < eng.hop; i++)
        in_hop[i] = 0.2f * sinf((float)(2.0 * M_PI * 220.0 * i / sr));

    int warmup = eng.window / eng.hop + 3;
    printf("warmup %d hops ...\n", warmup);
    for (int i = 0; i < warmup; i++) engine_process_hop(&eng, in_hop);

    printf("bench %d hops ...\n", N);
    double *t = (double *)malloc(sizeof(double) * N);
    double t0 = wall_ms();
    for (int i = 0; i < N; i++) {
        double s = wall_ms();
        engine_process_hop(&eng, in_hop);
        t[i] = wall_ms() - s;
    }
    double total = wall_ms() - t0;

    qsort(t, N, sizeof(double), cmp_double);
    double sum = 0; for (int i = 0; i < N; i++) sum += t[i];
    double avg = sum / N;
    double med = t[N / 2];
    double p95 = t[(int)(N * 0.95)];
    double hop_ms = (double)eng.hop * 1000.0 / sr;

    printf("\n--- bench K=%d N=%d hop=%d (%.1f ms of audio/hop) ---\n",
           K, N, eng.hop, hop_ms);
    printf("  avg=%.1f  min=%.1f  med=%.1f  p95=%.1f  max=%.1f ms\n",
           avg, t[0], med, p95, t[N - 1]);
    printf("  RTF = %.2f  (%.1f%% of audio budget)\n",
           avg / hop_ms, 100.0 * avg / hop_ms);
    printf("  total wall=%.1f ms\n", total);

    engine_free(&eng);
    free(in_hop); free(tgt); free(t);
    return 0;
}
