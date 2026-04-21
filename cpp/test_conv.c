/* Parity test: run conv1d on cases produced by gen_conv_ref.py and compare
   against F.conv1d reference outputs. */
#include "conv.h"
#include "ops.h"
#include "tensor.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static float *read_floats(const char *path, size_t expect_n) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "  open fail: %s\n", path); return NULL; }
    float *buf = (float *)malloc(expect_n * sizeof(float));
    if (!buf) { fclose(f); return NULL; }
    size_t got = fread(buf, sizeof(float), expect_n, f);
    fclose(f);
    if (got != expect_n) {
        fprintf(stderr, "  short read: %s got=%zu want=%zu\n", path, got, expect_n);
        free(buf);
        return NULL;
    }
    return buf;
}

static int run_case(const TensorStore *store,
                    const char *tag, const char *name,
                    int K, int pad, int dil,
                    int C_in, int T_in, int C_out, int T_out,
                    int has_bias) {
    char wname[256], bname[256], xpath[256], ypath[256];
    snprintf(wname, sizeof(wname), "%s.weight", name);
    snprintf(bname, sizeof(bname), "%s.bias", name);
    snprintf(xpath, sizeof(xpath), "cpp/ref/%s_x.bin", tag);
    snprintf(ypath, sizeof(ypath), "cpp/ref/%s_y.bin", tag);

    const Tensor *wt = tensor_get(store, wname);
    const Tensor *bt = has_bias ? tensor_get(store, bname) : NULL;
    if (!wt || (has_bias && !bt)) {
        printf("FAIL %s: missing weight/bias\n", tag);
        return 1;
    }

    float *x = read_floats(xpath, (size_t)C_in * T_in);
    float *y_ref = read_floats(ypath, (size_t)C_out * T_out);
    if (!x || !y_ref) { free(x); free(y_ref); return 1; }

    float *y = (float *)malloc((size_t)C_out * T_out * sizeof(float));
    float *scratch = (K == 1 && pad == 0) ? NULL
                     : (float *)malloc((size_t)C_in * K * T_out * sizeof(float));

    conv1d(x, C_in, T_in, wt->data, C_out, K, has_bias ? bt->data : NULL,
           pad, dil, y, scratch);

    double max_abs = 0.0, mean_abs = 0.0;
    size_t n = (size_t)C_out * T_out;
    for (size_t i = 0; i < n; i++) {
        double d = fabs((double)(y[i] - y_ref[i]));
        if (d > max_abs) max_abs = d;
        mean_abs += d;
    }
    mean_abs /= (double)n;

    int pass = max_abs < 1e-3;
    printf("%s %-10s  max_abs=%.2e  mean_abs=%.2e  shape=[%d,%d]\n",
           pass ? "PASS" : "FAIL", tag, max_abs, mean_abs, C_out, T_out);

    free(x); free(y_ref); free(y); free(scratch);
    return pass ? 0 : 1;
}

int main(void) {
    TensorStore store;
    if (tensor_store_load(&store, "cpp/weights.bin") != 0) return 1;

    FILE *f = fopen("cpp/ref/manifest.txt", "r");
    if (!f) {
        fprintf(stderr, "manifest.txt missing — run `python cpp/gen_conv_ref.py` first\n");
        tensor_store_free(&store);
        return 1;
    }

    int fails = 0, total = 0;
    char tag[64], name[128];
    int K, pad, dil, C_in, T_in, C_out, T_out, has_bias;
    while (fscanf(f, "%63s %127s %d %d %d %d %d %d %d %d",
                  tag, name, &K, &pad, &dil, &C_in, &T_in, &C_out, &T_out, &has_bias) == 10) {
        fails += run_case(&store, tag, name, K, pad, dil, C_in, T_in, C_out, T_out, has_bias);
        total++;
    }
    fclose(f);
    tensor_store_free(&store);

    printf("\n%d/%d PASS\n", total - fails, total);
    return fails ? 1 : 0;
}
