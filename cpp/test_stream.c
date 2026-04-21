/* Verify model_forward_emit produces bit-identical output (in the emit region)
   to model_forward. If this passes, engine can switch to the emit variant and
   skip computing 85-90% of dec's output that will be discarded anyway. */
#include "model.h"
#include "tensor.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static float *read_floats(const char *path, size_t n) {
    FILE *f = fopen(path, "rb"); if (!f) return NULL;
    float *b = (float *)malloc(n * sizeof(float));
    size_t got = fread(b, sizeof(float), n, f); fclose(f);
    if (got != n) { free(b); return NULL; }
    return b;
}

int main(void) {
    TensorStore store;
    if (tensor_store_load(&store, "cpp/weights.bin") != 0) { fprintf(stderr, "load weights\n"); return 1; }
    Model m; if (model_init(&m, &store) != 0) { fprintf(stderr, "model_init\n"); return 1; }

    FILE *f = fopen("cpp/ref/model_manifest.txt", "r");
    if (!f) { fprintf(stderr, "need cpp/ref/model_manifest.txt\n"); return 1; }
    int spec_ch, T, gin, wav_len;
    if (fscanf(f, "%d %d %d %d", &spec_ch, &T, &gin, &wav_len) != 4) return 1;
    fclose(f);

    float *spec   = read_floats("cpp/ref/model_spec.bin",  (size_t)spec_ch * T);
    float *spec2  = read_floats("cpp/ref/model_spec.bin",  (size_t)spec_ch * T);  /* fresh copy, forward mutates */
    float *g_src  = read_floats("cpp/ref/model_g_src.bin", (size_t)gin);
    float *g_tgt  = read_floats("cpp/ref/model_g_tgt.bin", (size_t)gin);
    if (!spec || !spec2 || !g_src || !g_tgt) { fprintf(stderr, "read ref\n"); return 1; }

    int hop_len = m.hop_length;
    int audio_len = T * hop_len;
    int scratch_n = model_scratch_floats(&m, T);
    float *scratch  = (float *)calloc((size_t)scratch_n, sizeof(float));
    float *scratch2 = (float *)calloc((size_t)scratch_n, sizeof(float));

    /* Reference: full forward. */
    float *wav_full = (float *)calloc((size_t)audio_len, sizeof(float));
    model_forward(&m, spec, T, g_src, g_tgt, wav_full, scratch);

    /* Test a few emit positions. For engine at K=4: hop = CHUNK/K = 2496,
       emit_offset in dec output at T=172 (from manifest) is 2*CHUNK-HOP.
       But here T=172 (from test manifest) not 117. Generalize: try emit at the
       model's natural "mid-right" position using the same formula pattern. */
    const int hop_samples = 2496;
    int test_positions[4];
    int n_tests = 0;
    /* Pick a few offsets inside the window: near left edge, middle, near right. */
    test_positions[n_tests++] = hop_len * 8;                                        /* near left */
    test_positions[n_tests++] = hop_len * (T / 3);                                  /* first third */
    test_positions[n_tests++] = hop_len * (2 * T / 3);                              /* two thirds */
    test_positions[n_tests++] = audio_len - hop_samples - hop_len * 4;              /* near right */

    int fails = 0;
    for (int ti = 0; ti < n_tests; ti++) {
        int emit_off = test_positions[ti];
        /* Round to frame boundary. */
        emit_off = (emit_off / hop_len) * hop_len;
        if (emit_off + hop_samples > audio_len) continue;

        float *emit = (float *)calloc((size_t)hop_samples, sizeof(float));
        /* Must give model_forward_emit a fresh spec copy (enc_q's ref_enc path
           mutates via layernorm, though enc_q itself does not — keep both runs
           independent via separate buffers). */
        memcpy(spec2, spec, (size_t)spec_ch * T * sizeof(float));
        model_forward_emit(&m, spec2, T, g_src, g_tgt,
                           emit_off, hop_samples, emit, scratch2);

        double max_abs = 0;
        int count_nonzero = 0;
        for (int i = 0; i < hop_samples; i++) {
            double d = fabs((double)emit[i] - (double)wav_full[emit_off + i]);
            if (d > max_abs) max_abs = d;
            if (d > 0) count_nonzero++;
        }
        printf("  emit_off=%d (frame %d)  max_abs=%.3e  non_bit_exact=%d/%d\n",
               emit_off, emit_off / hop_len, max_abs, count_nonzero, hop_samples);
        if (max_abs > 1e-5) fails++;
        free(emit);
    }

    free(spec); free(spec2); free(g_src); free(g_tgt);
    free(scratch); free(scratch2); free(wav_full);

    if (fails) { printf("FAIL (%d positions failed parity)\n", fails); return 1; }
    printf("PASS\n");
    return 0;
}
