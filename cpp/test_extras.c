/* Parity tests for Conv2d, LayerNorm, and GRU. */
#include "conv.h"
#include "gru.h"
#include "ops.h"
#include "tensor.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static float *read_floats(const char *path, size_t n) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    float *buf = (float *)malloc(n * sizeof(float));
    if (!buf) { fclose(f); return NULL; }
    size_t got = fread(buf, sizeof(float), n, f);
    fclose(f);
    if (got != n) { free(buf); return NULL; }
    return buf;
}

static int run_conv2d_case(const TensorStore *store,
                           const char *tag, const char *name,
                           int C_in, int H_in, int W_in,
                           int C_out, int H_out, int W_out) {
    char wname[256], bname[256], xpath[256], ypath[256];
    snprintf(wname, sizeof(wname), "%s.weight", name);
    snprintf(bname, sizeof(bname), "%s.bias",   name);
    snprintf(xpath, sizeof(xpath), "cpp/ref/%s_x.bin", tag);
    snprintf(ypath, sizeof(ypath), "cpp/ref/%s_y.bin", tag);

    const Tensor *wt = tensor_get(store, wname);
    const Tensor *bt = tensor_get(store, bname);
    if (!wt || !bt) { printf("FAIL %s: missing weight/bias\n", tag); return 1; }

    float *x     = read_floats(xpath, (size_t)C_in  * H_in  * W_in);
    float *y_ref = read_floats(ypath, (size_t)C_out * H_out * W_out);
    float *y     = (float *)malloc((size_t)C_out * H_out * W_out * sizeof(float));
    float *scratch = (float *)malloc((size_t)C_in * 3 * 3 * H_out * W_out * sizeof(float));

    conv2d(x, C_in, H_in, W_in, wt->data, C_out, 3, 3, bt->data,
           1, 1, 2, 2, y, scratch);

    double max_abs = 0, mean_abs = 0;
    size_t n = (size_t)C_out * H_out * W_out;
    for (size_t i = 0; i < n; i++) {
        double d = fabs((double)(y[i] - y_ref[i]));
        if (d > max_abs) max_abs = d;
        mean_abs += d;
    }
    mean_abs /= (double)n;

    int pass = max_abs < 1e-3;
    printf("%s %-6s  max_abs=%.2e  mean_abs=%.2e  out=[%d,%d,%d]\n",
           pass ? "PASS" : "FAIL", tag, max_abs, mean_abs, C_out, H_out, W_out);

    free(x); free(y_ref); free(y); free(scratch);
    return pass ? 0 : 1;
}

static int run_layernorm(const TensorStore *store) {
    const Tensor *w = tensor_get(store, "ref_enc.layernorm.weight");
    const Tensor *b = tensor_get(store, "ref_enc.layernorm.bias");
    if (!w || !b) { printf("FAIL ln: missing weight/bias\n"); return 1; }

    FILE *f = fopen("cpp/ref/ln_manifest.txt", "r");
    if (!f) { printf("FAIL ln: manifest missing\n"); return 1; }
    int n_rows, n_feat;
    if (fscanf(f, "%d %d", &n_rows, &n_feat) != 2) { fclose(f); return 1; }
    fclose(f);

    float *x     = read_floats("cpp/ref/ln_x.bin", (size_t)n_rows * n_feat);
    float *y_ref = read_floats("cpp/ref/ln_y.bin", (size_t)n_rows * n_feat);
    ops_layer_norm(x, w->data, b->data, n_rows, n_feat, 1e-5f);

    double max_abs = 0, mean_abs = 0;
    size_t n = (size_t)n_rows * n_feat;
    for (size_t i = 0; i < n; i++) {
        double d = fabs((double)(x[i] - y_ref[i]));
        if (d > max_abs) max_abs = d;
        mean_abs += d;
    }
    mean_abs /= (double)n;

    int pass = max_abs < 1e-3;
    printf("%s ln      max_abs=%.2e  mean_abs=%.2e  rows=%d feat=%d\n",
           pass ? "PASS" : "FAIL", max_abs, mean_abs, n_rows, n_feat);

    free(x); free(y_ref);
    return pass ? 0 : 1;
}

int main(void) {
    TensorStore store;
    if (tensor_store_load(&store, "cpp/weights.bin") != 0) return 1;

    int fails = 0, total = 0;
    FILE *f = fopen("cpp/ref/conv2d_manifest.txt", "r");
    if (f) {
        char tag[64], name[128];
        int C_in, H_in, W_in, C_out, H_out, W_out;
        while (fscanf(f, "%63s %127s %d %d %d %d %d %d",
                      tag, name, &C_in, &H_in, &W_in, &C_out, &H_out, &W_out) == 8) {
            fails += run_conv2d_case(&store, tag, name, C_in, H_in, W_in, C_out, H_out, W_out);
            total++;
        }
        fclose(f);
    } else {
        printf("WARN conv2d manifest missing — run gen_extras_ref.py\n");
    }

    fails += run_layernorm(&store); total++;

    /* GRU test */
    {
        const Tensor *W_ih = tensor_get(&store, "ref_enc.gru.weight_ih_l0");
        const Tensor *W_hh = tensor_get(&store, "ref_enc.gru.weight_hh_l0");
        const Tensor *b_ih = tensor_get(&store, "ref_enc.gru.bias_ih_l0");
        const Tensor *b_hh = tensor_get(&store, "ref_enc.gru.bias_hh_l0");

        f = fopen("cpp/ref/gru_manifest.txt", "r");
        int T, input_size, H;
        if (f && fscanf(f, "%d %d %d", &T, &input_size, &H) == 3 &&
            W_ih && W_hh && b_ih && b_hh) {
            if (f) fclose(f);
            float *x     = read_floats("cpp/ref/gru_x.bin", (size_t)T * input_size);
            float *h_ref = read_floats("cpp/ref/gru_h.bin", (size_t)H);
            float *h     = (float *)calloc(H, sizeof(float));
            float *scratch = (float *)malloc(((size_t)T * 3 * H + 3 * H) * sizeof(float));

            gru_forward_last(x, T, input_size, H,
                             W_ih->data, W_hh->data, b_ih->data, b_hh->data,
                             h, scratch);

            double max_abs = 0, mean_abs = 0;
            for (int i = 0; i < H; i++) {
                double d = fabs((double)(h[i] - h_ref[i]));
                if (d > max_abs) max_abs = d;
                mean_abs += d;
            }
            mean_abs /= (double)H;
            int pass = max_abs < 1e-3;
            printf("%s gru     max_abs=%.2e  mean_abs=%.2e  T=%d input=%d H=%d\n",
                   pass ? "PASS" : "FAIL", max_abs, mean_abs, T, input_size, H);
            total++;
            if (!pass) fails++;

            free(x); free(h_ref); free(h); free(scratch);
        } else {
            if (f) fclose(f);
            printf("WARN gru: missing manifest or weights\n");
        }
    }

    tensor_store_free(&store);
    printf("\n%d/%d PASS\n", total - fails, total);
    return fails ? 1 : 0;
}
