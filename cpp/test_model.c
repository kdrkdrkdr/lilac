/* End-to-end parity: model_forward(spec, g_src, g_tgt) → wav.
   Compares against Python reference produced by gen_model_ref.py. */
#include "model.h"
#include "tensor.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static float *read_floats(const char *path, size_t n) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "open fail: %s\n", path); return NULL; }
    float *buf = (float *)malloc(n * sizeof(float));
    size_t got = fread(buf, sizeof(float), n, f);
    fclose(f);
    if (got != n) { fprintf(stderr, "short read %s got=%zu want=%zu\n", path, got, n); free(buf); return NULL; }
    return buf;
}

int main(void) {
    TensorStore store;
    if (tensor_store_load(&store, "cpp/weights.bin") != 0) return 1;
    Model m;
    if (model_init(&m, &store) != 0) { tensor_store_free(&store); return 1; }

    FILE *f = fopen("cpp/ref/model_manifest.txt", "r");
    if (!f) { fprintf(stderr, "missing cpp/ref/model_manifest.txt — run gen_model_ref.py\n"); return 1; }
    int spec_ch, T, gin, wav_len;
    if (fscanf(f, "%d %d %d %d", &spec_ch, &T, &gin, &wav_len) != 4) { fclose(f); return 1; }
    fclose(f);
    printf("manifest: spec_ch=%d T=%d gin=%d wav_len=%d\n", spec_ch, T, gin, wav_len);

    float *spec    = read_floats("cpp/ref/model_spec.bin",  (size_t)spec_ch * T);
    float *g_src   = read_floats("cpp/ref/model_g_src.bin", (size_t)gin);
    float *g_tgt   = read_floats("cpp/ref/model_g_tgt.bin", (size_t)gin);
    float *wav_ref = read_floats("cpp/ref/model_wav.bin",   (size_t)wav_len);
    if (!spec || !g_src || !g_tgt || !wav_ref) return 1;

    float *wav_c = (float *)calloc((size_t)T * 256, sizeof(float));
    float *scratch = (float *)calloc((size_t)model_scratch_floats(&m, T), sizeof(float));
    if (!wav_c || !scratch) { fprintf(stderr, "alloc fail\n"); return 1; }

    model_forward(&m, spec, T, g_src, g_tgt, wav_c, scratch);

    int n = T * 256 < wav_len ? T * 256 : wav_len;
    double max_abs = 0, mean_abs = 0, sum_ref = 0, sum_c = 0;
    for (int i = 0; i < n; i++) {
        double d = fabs((double)(wav_c[i] - wav_ref[i]));
        if (d > max_abs) max_abs = d;
        mean_abs += d;
        sum_ref += fabs((double)wav_ref[i]);
        sum_c   += fabs((double)wav_c[i]);
    }
    mean_abs /= (double)n;

    printf("compare over n=%d samples:\n", n);
    printf("  max_abs  = %.4f\n", max_abs);
    printf("  mean_abs = %.4f\n", mean_abs);
    printf("  mean|ref|= %.4f\n", sum_ref / n);
    printf("  mean|c|  = %.4f\n", sum_c   / n);
    printf("  c_first_8  = "); for (int i = 0; i < 8; i++) printf("%.4f ", wav_c[i]);   printf("\n");
    printf("  ref_first_8= "); for (int i = 0; i < 8; i++) printf("%.4f ", wav_ref[i]); printf("\n");
    printf("  c_last_8   = "); for (int i = n - 8; i < n; i++) printf("%.4f ", wav_c[i]);   printf("\n");
    printf("  ref_last_8 = "); for (int i = n - 8; i < n; i++) printf("%.4f ", wav_ref[i]); printf("\n");

    int pass = max_abs < 5e-2;   /* loose; after a deep chain even ULP-level layer errors accumulate */
    printf("\n%s (max_abs < 5e-2 target)\n", pass ? "PASS" : "FAIL");

    free(spec); free(g_src); free(g_tgt); free(wav_ref);
    free(wav_c); free(scratch);
    model_init(&m, &store);   /* no-op, just to mirror the init pattern */
    tensor_store_free(&store);
    return pass ? 0 : 1;
}
