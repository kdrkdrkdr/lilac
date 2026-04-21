#include "dec.h"
#include "conv.h"
#include "ops.h"
#include "pool.h"
#include "tensor.h"

#include <immintrin.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include "log.h"

/* Global 2-worker pool. Main thread runs one of the three resblocks, so with
   two auxiliary workers we achieve 3-way parallelism per upsample stage.
   Created lazily on first dec_forward. Lifetime = process. */
static Pool *g_pool = NULL;
static INIT_ONCE g_pool_once = INIT_ONCE_STATIC_INIT;
static BOOL CALLBACK init_pool(PINIT_ONCE o, PVOID p, PVOID *ctx) {
    (void)o; (void)p; (void)ctx;
    g_pool = pool_create(2);
    return TRUE;
}

static double _dec_ms(void) {
    static LARGE_INTEGER freq = {0};
    if (!freq.QuadPart) QueryPerformanceFrequency(&freq);
    LARGE_INTEGER c; QueryPerformanceCounter(&c);
    return (double)c.QuadPart * 1000.0 / (double)freq.QuadPart;
}

static const int UP_RATES[LILAC_DEC_UPSAMPLES]   = {8, 8, 2, 2};
static const int UP_KERNELS[LILAC_DEC_UPSAMPLES] = {16, 16, 4, 4};
static const int RB_KERNELS[LILAC_DEC_KERNELS_PER_UP] = {3, 7, 11};
static const int RB_DIL[LILAC_DEC_KERNELS_PER_UP][LILAC_DEC_RESBLOCK_CONVS] = {
    {1, 3, 5}, {1, 3, 5}, {1, 3, 5},
};

int dec_init(Dec *d, const TensorStore *store) {
    memset(d, 0, sizeof(*d));
    d->initial_channel  = 192;
    d->upsample_initial = 512;
    for (int i = 0; i < LILAC_DEC_UPSAMPLES; i++) {
        d->upsample_rates[i]   = UP_RATES[i];
        d->upsample_kernels[i] = UP_KERNELS[i];
    }
    for (int i = 0; i < LILAC_DEC_KERNELS_PER_UP; i++) {
        d->resblock_kernels[i] = RB_KERNELS[i];
        for (int j = 0; j < LILAC_DEC_RESBLOCK_CONVS; j++)
            d->resblock_dilations[i][j] = RB_DIL[i][j];
    }

    const Tensor *t;
    t = tensor_get(store, "dec.conv_pre.weight"); if (!t) return -1; d->conv_pre_w = t->data;
    t = tensor_get(store, "dec.conv_pre.bias");   if (!t) return -1; d->conv_pre_b = t->data;
    t = tensor_get(store, "dec.conv_post.weight");if (!t) return -1; d->conv_post_w = t->data;
    t = tensor_get(store, "dec.cond.weight");     if (!t) return -1; d->cond_w = t->data;
    t = tensor_get(store, "dec.cond.bias");       if (!t) return -1; d->cond_b = t->data;

    char name[128];
    for (int i = 0; i < LILAC_DEC_UPSAMPLES; i++) {
        snprintf(name, sizeof(name), "dec.ups.%d.weight", i);
        t = tensor_get(store, name); if (!t) return -1; d->ups_w[i] = t->data;
        snprintf(name, sizeof(name), "dec.ups.%d.bias", i);
        t = tensor_get(store, name); if (!t) return -1; d->ups_b[i] = t->data;
    }

    /* resblocks[i*3 + j]: i = upsample stage, j = kernel index in {3,7,11} */
    int stage_channels[LILAC_DEC_UPSAMPLES];
    {
        int c = d->upsample_initial;
        for (int i = 0; i < LILAC_DEC_UPSAMPLES; i++) { c /= 2; stage_channels[i] = c; }
    }
    for (int i = 0; i < LILAC_DEC_UPSAMPLES; i++) {
        int C = stage_channels[i];
        for (int j = 0; j < LILAC_DEC_KERNELS_PER_UP; j++) {
            int flat = i * LILAC_DEC_KERNELS_PER_UP + j;
            int K = d->resblock_kernels[j];
            size_t pack = (size_t)C * C * K;
            for (int c = 0; c < LILAC_DEC_RESBLOCK_CONVS; c++) {
                snprintf(name, sizeof(name), "dec.resblocks.%d.convs1.%d.weight", flat, c);
                t = tensor_get(store, name); if (!t) return -1; d->rb_c1_w[flat][c] = t->data;
                snprintf(name, sizeof(name), "dec.resblocks.%d.convs1.%d.bias", flat, c);
                t = tensor_get(store, name); if (!t) return -1; d->rb_c1_b[flat][c] = t->data;
                snprintf(name, sizeof(name), "dec.resblocks.%d.convs2.%d.weight", flat, c);
                t = tensor_get(store, name); if (!t) return -1; d->rb_c2_w[flat][c] = t->data;
                snprintf(name, sizeof(name), "dec.resblocks.%d.convs2.%d.bias", flat, c);
                t = tensor_get(store, name); if (!t) return -1; d->rb_c2_b[flat][c] = t->data;

                float *kf1 = (float *)malloc(pack * sizeof(float));
                float *kf2 = (float *)malloc(pack * sizeof(float));
                if (!kf1 || !kf2) { free(kf1); free(kf2); return -1; }
                conv_prepack_weight(d->rb_c1_w[flat][c], C, C, K, kf1);
                conv_prepack_weight(d->rb_c2_w[flat][c], C, C, K, kf2);
                d->rb_c1_w_kfirst[flat][c] = kf1;
                d->rb_c2_w_kfirst[flat][c] = kf2;
            }
        }
    }
    return 0;
}

void dec_free(Dec *d) {
    for (int flat = 0; flat < LILAC_DEC_UPSAMPLES * LILAC_DEC_KERNELS_PER_UP; flat++) {
        for (int c = 0; c < LILAC_DEC_RESBLOCK_CONVS; c++) {
            free(d->rb_c1_w_kfirst[flat][c]);
            free(d->rb_c2_w_kfirst[flat][c]);
            d->rb_c1_w_kfirst[flat][c] = NULL;
            d->rb_c2_w_kfirst[flat][c] = NULL;
        }
    }
}

/* Walks the dec stages once to compute the exact worst-case buffer sizes.
   Stages: pre-conv at [C=initial=512, T_in], then 4 upsamples with rates
   [8,8,2,2] halving channels each time. Within each upsample we run
   ConvTranspose1d (k=[16,16,4,4]), then 3 resblocks (kernels [3,7,11]). */
static void dec_dims(int T_in, long *buf_max, long *rb_tmp_max, long *im2col_max) {
    int  C = 512;
    long T = T_in;
    long bmax = (long)C * T;                    /* conv_pre output */
    long rmax = 0;
    long imax = (long)C * 7 * T;                /* conv_pre k=7 im2col */
    for (int i = 0; i < LILAC_DEC_UPSAMPLES; i++) {
        int u   = UP_RATES[i];
        int k_u = UP_KERNELS[i];
        int C_out = C / 2;
        long T_out = T * u;
        /* ConvTranspose1d scratch = C_out * k_u * T_in. */
        long im_up = (long)C_out * k_u * T;
        if (im_up > imax) imax = im_up;
        /* ups output occupies buf_b: C_out * T_out */
        long buf_stage = (long)C_out * T_out;
        if (buf_stage > bmax) bmax = buf_stage;
        /* resblocks — each uses im2col of size C_out * K * T_out where K∈{3,7,11}
           (k=11 is the largest). rb_tmp holds 2 * C_out * T_out. */
        long im_rb = (long)C_out * 11 * T_out;
        if (im_rb > imax) imax = im_rb;
        long rb_tmp = 2L * C_out * T_out;
        if (rb_tmp > rmax) rmax = rb_tmp;
        C = C_out; T = T_out;
    }
    /* conv_post k=7. */
    long im_post = (long)C * 7 * T;
    if (im_post > imax) imax = im_post;
    *buf_max    = bmax;
    *rb_tmp_max = rmax;
    *im2col_max = imax;
}

int dec_scratch_floats(const Dec *d, int T_max) {
    (void)d;
    long bmax, rbmax, imax;
    dec_dims(T_max, &bmax, &rbmax, &imax);
    /* Layout for parallel resblock dispatch with direct conv1d:
         buf_a, buf_b                           2*bmax
         rb_out[3] (per-kernel outputs)         3*bmax
         rb_tmp[3] (per-worker [2C, T] scratch) 3*rbmax
         im2col_main (conv_pre/ups/conv_post)   imax */
    return (int)(5 * bmax + 3 * rbmax + imax + 1024);
}

/* ResBlock1 forward: three (dilated conv + non-dilated conv) blocks with
   LeakyReLU before each conv. All convs are C_in=C_out=C with kernel 3/7/11,
   so we use conv1d_direct (multi-sgemm, no im2col buffer) on the K-first
   prepacked weights. */
static void resblock1(const float *const c1_w_kf[], const float *const c1_b[],
                      const float *const c2_w_kf[], const float *const c2_b[],
                      const int dilations[], int kernel,
                      float *x, int C, int T,
                      float *tmp) {
    for (int step = 0; step < LILAC_DEC_RESBLOCK_CONVS; step++) {
        ops_leaky_relu_copy(tmp, x, 0.1f, (size_t)C * T);
        int dil  = dilations[step];
        int pad1 = (kernel * dil - dil) / 2;
        float *y = tmp + (size_t)C * T;
        conv1d_direct(tmp, C, T, c1_w_kf[step], C, kernel, c1_b[step], pad1, dil, y);
        ops_leaky_relu(y, 0.1f, (size_t)C * T);
        int pad2 = (kernel - 1) / 2;
        float *z = tmp;
        conv1d_direct(y, C, T, c2_w_kf[step], C, kernel, c2_b[step], pad2, 1, z);
        ops_vec_add(x, z, (size_t)C * T);
    }
}

typedef struct {
    const float *const *c1_w_kf;
    const float *const *c1_b;
    const float *const *c2_w_kf;
    const float *const *c2_b;
    const int *dilations;
    int kernel, C, T;
    const float *input;   /* upsample output (read-only, shared across 3 jobs) */
    float *out;           /* C*T — starts as copy of input, resblock writes in place */
    float *tmp;           /* 2*C*T */
} ResblockJob;

static void resblock_task(void *arg) {
    ResblockJob *j = (ResblockJob *)arg;
    size_t n = (size_t)j->C * j->T;
    memcpy(j->out, j->input, n * sizeof(float));
    resblock1(j->c1_w_kf, j->c1_b, j->c2_w_kf, j->c2_b,
              j->dilations, j->kernel,
              j->out, j->C, j->T, j->tmp);
}

void dec_forward(const Dec *d, const float *z, int T, float *out_audio, float *scratch) {
    long T_cur = T;
    long C_cur = d->upsample_initial;

    long bmax, rbmax, imax;
    dec_dims(T, &bmax, &rbmax, &imax);

    float *buf_a    = scratch;
    float *buf_b    = buf_a    + (size_t)bmax;
    float *rb_out[3] = {
        buf_b    + (size_t)bmax,
        buf_b    + (size_t)(2 * bmax),
        buf_b    + (size_t)(3 * bmax),
    };
    float *rb_tmp[3] = {
        rb_out[2] + (size_t)bmax,
        rb_out[2] + (size_t)bmax + (size_t)rbmax,
        rb_out[2] + (size_t)bmax + (size_t)(2 * rbmax),
    };
    /* Only conv_pre/ups/conv_post still materialize im2col; resblocks use
       conv1d_direct and need no scratch beyond rb_tmp. */
    float *im2col_main = rb_tmp[2] + (size_t)rbmax;
    (void)imax;

    /* Lazy-init the shared worker pool. */
    InitOnceExecuteOnce(&g_pool_once, init_pool, NULL, NULL);

    /* conv_pre: [192, T] → [512, T] k=7 pad=3 */
    conv1d(z, d->initial_channel, T, d->conv_pre_w, d->upsample_initial, 7,
           d->conv_pre_b, 3, 1, buf_a, im2col_main);
    /* Python Generator adds cond(g) after conv_pre. With zero_g=True, cond(0)
       collapses to the bias broadcast over time. */
    ops_bias_add_ct(buf_a, d->cond_b, d->upsample_initial, (int)T_cur);

    float *cur = buf_a;
    float *nxt = buf_b;

    static int dec_tick = 0;
    int log_now = ((dec_tick++ & 31) == 0);
    double st_t[LILAC_DEC_UPSAMPLES][2];

    for (int i = 0; i < LILAC_DEC_UPSAMPLES; i++) {
        ops_leaky_relu(cur, 0.1f, (size_t)C_cur * T_cur);

        int u = d->upsample_rates[i];
        int k = d->upsample_kernels[i];
        int pad = (k - u) / 2;
        int C_out = (int)(C_cur / 2);
        int T_out = (int)((T_cur - 1) * u - 2 * pad + k);
        double ts0 = _dec_ms();
        conv_transpose1d(cur, (int)C_cur, (int)T_cur, d->ups_w[i], C_out, k,
                         d->ups_b[i], pad, u, nxt, im2col_main);
        double ts1 = _dec_ms();

        /* Dispatch the 3 resblocks in parallel. Main thread runs job 0; pool
           workers run jobs 1 and 2. Each writes its own output buffer. */
        ResblockJob jobs[3];
        for (int j = 0; j < 3; j++) {
            int flat = i * LILAC_DEC_KERNELS_PER_UP + j;
            jobs[j].c1_w_kf = (const float *const *)d->rb_c1_w_kfirst[flat];
            jobs[j].c1_b    = d->rb_c1_b[flat];
            jobs[j].c2_w_kf = (const float *const *)d->rb_c2_w_kfirst[flat];
            jobs[j].c2_b    = d->rb_c2_b[flat];
            jobs[j].dilations = d->resblock_dilations[j];
            jobs[j].kernel = d->resblock_kernels[j];
            jobs[j].C = C_out;
            jobs[j].T = T_out;
            jobs[j].input = nxt;
            jobs[j].out = rb_out[j];
            jobs[j].tmp = rb_tmp[j];
        }
        pool_submit(g_pool, 0, resblock_task, &jobs[1]);
        pool_submit(g_pool, 1, resblock_task, &jobs[2]);
        resblock_task(&jobs[0]);
        pool_wait(g_pool);

        /* nxt = (rb_out[0] + rb_out[1] + rb_out[2]) / 3, SIMD in one pass. */
        {
            size_t n = (size_t)C_out * T_out;
            const __m256 inv3 = _mm256_set1_ps(1.0f / 3.0f);
            size_t p = 0;
            for (; p + 8 <= n; p += 8) {
                __m256 a = _mm256_loadu_ps(rb_out[0] + p);
                __m256 b = _mm256_loadu_ps(rb_out[1] + p);
                __m256 c = _mm256_loadu_ps(rb_out[2] + p);
                __m256 s = _mm256_add_ps(_mm256_add_ps(a, b), c);
                _mm256_storeu_ps(nxt + p, _mm256_mul_ps(s, inv3));
            }
            for (; p < n; p++)
                nxt[p] = (rb_out[0][p] + rb_out[1][p] + rb_out[2][p]) * (1.0f / 3.0f);
        }
        double ts2 = _dec_ms();
        st_t[i][0] = ts1 - ts0;
        st_t[i][1] = ts2 - ts1;

        float *tmp = cur; cur = nxt; nxt = tmp;
        C_cur = C_out;
        T_cur = T_out;
    }

    if (log_now) {
        lilac_log("[dec]  s0 ups=%.1f rb=%.1f | s1 ups=%.1f rb=%.1f | "
                  "s2 ups=%.1f rb=%.1f | s3 ups=%.1f rb=%.1f\n",
                  st_t[0][0], st_t[0][1], st_t[1][0], st_t[1][1],
                  st_t[2][0], st_t[2][1], st_t[3][0], st_t[3][1]);
    }

    ops_leaky_relu(cur, 0.1f, (size_t)C_cur * T_cur);
    conv1d(cur, (int)C_cur, (int)T_cur, d->conv_post_w, 1, 7, NULL, 3, 1,
           nxt, im2col_main);

    memcpy(out_audio, nxt, (size_t)T_cur * sizeof(float));
    ops_tanhf(out_audio, (size_t)T_cur);
}
