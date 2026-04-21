#include "dec_stream.h"
#include "conv.h"
#include "ops.h"
#include "pool.h"

#include <immintrin.h>
#include <stdlib.h>
#include <string.h>

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

/* Shared 2-worker pool so the 3 resblocks per stage run concurrently (main +
   2 workers). Lazily initialized; lifetime = process. */
static Pool *g_ds_pool = NULL;
static INIT_ONCE g_ds_pool_once = INIT_ONCE_STATIC_INIT;
static BOOL CALLBACK init_ds_pool(PINIT_ONCE o, PVOID p, PVOID *ctx) {
    (void)o; (void)p; (void)ctx;
    g_ds_pool = pool_create(2);
    return TRUE;
}

int dec_stream_init(DecStream *ds, const Dec *d) {
    memset(ds, 0, sizeof(*ds));
    ds->dec = d;
    ds->initial_channel  = d->initial_channel;   /* 192 */
    ds->upsample_initial = d->upsample_initial;  /* 512 */

    /* Per-stage channels and cumulative upsample at stage output. */
    int C = d->upsample_initial;
    int rate = 1;
    for (int i = 0; i < LILAC_DEC_UPSAMPLES; i++) {
        C /= 2;
        rate *= d->upsample_rates[i];
        ds->stage_channels[i] = C;
        ds->stage_rate[i]     = rate;
    }

    /* Per-kernel resblock lag: sum_{step} (c1_pad + c2_pad) at stage-out rate.
       Same across stages since K/dilations are stage-independent. */
    ds->rb_max_lag = 0;
    for (int j = 0; j < LILAC_DEC_KERNELS_PER_UP; j++) {
        int K = d->resblock_kernels[j];
        int lag = 0;
        for (int step = 0; step < 3; step++) {
            int dil    = d->resblock_dilations[j][step];
            int c1_pad = (K - 1) * dil / 2;
            int c2_pad = (K - 1) / 2;
            lag += c1_pad + c2_pad;
        }
        ds->rb_lag_out[j] = lag;
        if (lag > ds->rb_max_lag) ds->rb_max_lag = lag;
    }

    /* ConvTranspose1d stream lag (in output-rate samples) per stage. */
    for (int i = 0; i < LILAC_DEC_UPSAMPLES; i++) {
        int K   = d->upsample_kernels[i];
        int u   = d->upsample_rates[i];
        int pad = (K - u) / 2;
        int state_in       = (K + u - 1) / u + 1;
        int interior_start = K - pad - 1;
        ds->ct_lag_out[i]  = state_in * u - interior_start;
    }

    /* Init primitives. */
    if (c1d_stream_init(&ds->conv_pre, d->initial_channel, 7, 1) != 0) goto fail;
    if (c1d_stream_init(&ds->conv_post,
                        ds->stage_channels[LILAC_DEC_UPSAMPLES - 1], 7, 1) != 0) goto fail;

    int C_in_stage = d->upsample_initial;
    for (int i = 0; i < LILAC_DEC_UPSAMPLES; i++) {
        int K = d->upsample_kernels[i];
        int u = d->upsample_rates[i];
        int pad = (K - u) / 2;
        if (ct1d_stream_init(&ds->ups[i], C_in_stage, K, u, pad) != 0) goto fail;
        int C_out = ds->stage_channels[i];
        for (int j = 0; j < LILAC_DEC_KERNELS_PER_UP; j++) {
            int K_rb = d->resblock_kernels[j];
            if (rb_stream_init(&ds->rb[i][j], C_out, K_rb, d->resblock_dilations[j]) != 0)
                goto fail;
            int align_size = ds->rb_max_lag - ds->rb_lag_out[j];
            if (delay_line_init(&ds->rb_align[i][j], C_out, align_size) != 0) goto fail;
        }
        C_in_stage = C_out;
    }

    /* Cumulative lag in audio samples.
         conv_pre : 3 z-frames (K=7 pad) × 256
         stage i  : (ct_lag + rb_max_lag) stage-out-rate × (256 / stage_rate[i])
         conv_post: 3 audio samples (K=7 pad) */
    long cum = (long)3 * 256;
    for (int i = 0; i < LILAC_DEC_UPSAMPLES; i++) {
        int stage_lag = ds->ct_lag_out[i] + ds->rb_max_lag;
        int per       = 256 / ds->stage_rate[i];
        cum += (long)stage_lag * per;
    }
    cum += 3;
    ds->cumulative_lag_audio = cum;

    /* Pack conv_pre / conv_post K-first (dec_init doesn't pack these). */
    {
        size_t pre_n  = (size_t)7 * 512 * 192;
        size_t post_n = (size_t)7 * 1 * ds->stage_channels[LILAC_DEC_UPSAMPLES - 1];
        ds->conv_pre_w_kf  = (float *)malloc(pre_n  * sizeof(float));
        ds->conv_post_w_kf = (float *)malloc(post_n * sizeof(float));
        if (!ds->conv_pre_w_kf || !ds->conv_post_w_kf) goto fail;
        conv_prepack_weight(d->conv_pre_w,  512, 192, 7, ds->conv_pre_w_kf);
        conv_prepack_weight(d->conv_post_w, 1,
                            ds->stage_channels[LILAC_DEC_UPSAMPLES - 1], 7,
                            ds->conv_post_w_kf);
    }

    return 0;
fail:
    dec_stream_free(ds);
    return -1;
}

void dec_stream_free(DecStream *ds) {
    if (!ds) return;
    c1d_stream_free(&ds->conv_pre);
    c1d_stream_free(&ds->conv_post);
    for (int i = 0; i < LILAC_DEC_UPSAMPLES; i++) {
        ct1d_stream_free(&ds->ups[i]);
        for (int j = 0; j < LILAC_DEC_KERNELS_PER_UP; j++) {
            rb_stream_free(&ds->rb[i][j]);
            delay_line_free(&ds->rb_align[i][j]);
        }
    }
    free(ds->conv_pre_w_kf);  ds->conv_pre_w_kf  = NULL;
    free(ds->conv_post_w_kf); ds->conv_post_w_kf = NULL;
}

void dec_stream_reset(DecStream *ds) {
    if (!ds) return;
    c1d_stream_reset(&ds->conv_pre);
    c1d_stream_reset(&ds->conv_post);
    for (int i = 0; i < LILAC_DEC_UPSAMPLES; i++) {
        ct1d_stream_reset(&ds->ups[i]);
        for (int j = 0; j < LILAC_DEC_KERNELS_PER_UP; j++) {
            rb_stream_reset(&ds->rb[i][j]);
            delay_line_reset(&ds->rb_align[i][j]);
        }
    }
}

/* Worst-case working buf = max over layers of C_layer_out * n_out_rate.
   conv_pre out : 512 * n_new
   stage i out  : stage_channels[i] * n_new * stage_rate[i]
   audio        : 1 * n_new * 256 */
static long dec_stream_max_buf(const DecStream *ds, int n_new) {
    long m = 512L * n_new;
    for (int i = 0; i < LILAC_DEC_UPSAMPLES; i++) {
        long sz = (long)ds->stage_channels[i] * n_new * ds->stage_rate[i];
        if (sz > m) m = sz;
    }
    long aud = 256L * n_new;
    if (aud > m) m = aud;
    return m;
}

int dec_stream_scratch_floats(const DecStream *ds, int n_new_max) {
    long m = dec_stream_max_buf(ds, n_new_max);
    /* Layout: cur + nxt + rb_out[3] + rb_al[3] + inner_rb[3] + inner_main.
       Each inner_rb needs 5*C_out*T_out + conv1d_stream inner; at stage 3
       (C=32, T_out=256*n, K=11 dil=5) that is 5m + 2m = 7m (m = 8192*n).
       Budget 8m per worker for headroom. inner_main handles conv_pre/ups/
       conv_post which are smaller. */
    long layout = 8 * m;
    long inner  = 3 * 8L * m + 4L * m;    /* 24m + 4m = 28m */
    return (int)(layout + inner + 16384);
}

/* Work item for one of the 3 parallel resblocks + its delay alignment. */
typedef struct {
    DecStream *ds;
    const Dec *d;
    int stage_i;
    int kernel_j;
    const float *input;    /* shared ups output */
    int n_in;
    float *out;            /* destination rb_al[j] (post-align) */
    float *tmp_rb_out;     /* scratch: rb_stream output before delay align */
    float *inner_scr;      /* rb_stream's internal scratch */
} RbTask;

static void rb_task(void *arg) {
    RbTask *t = (RbTask *)arg;
    int flat = t->stage_i * LILAC_DEC_KERNELS_PER_UP + t->kernel_j;
    resblock_stream(&t->ds->rb[t->stage_i][t->kernel_j],
        (const float *const *)t->d->rb_c1_w_kfirst[flat], t->d->rb_c1_b[flat],
        (const float *const *)t->d->rb_c2_w_kfirst[flat], t->d->rb_c2_b[flat],
        t->input, t->n_in, t->tmp_rb_out, t->inner_scr);
    delay_line_step(&t->ds->rb_align[t->stage_i][t->kernel_j],
                    t->tmp_rb_out, t->n_in, t->out);
}

void dec_stream_forward(DecStream *ds,
                        const float *z_new, int n_new,
                        float *audio_out, float *scratch) {
    const Dec *d = ds->dec;

    long m = dec_stream_max_buf(ds, n_new);
    float *cur      = scratch;
    float *nxt      = cur + m;
    float *rb_out[3] = { nxt + m, nxt + 2*m, nxt + 3*m };
    float *rb_al[3]  = { nxt + 4*m, nxt + 5*m, nxt + 6*m };
    /* Per-worker rb inner scratch (6m each) + one main inner for ct/conv_pre/
       conv_post (4m). */
    float *inner_rb[3] = {
        nxt + 7*m,
        nxt + 7*m + 8*m,
        nxt + 7*m + 16*m,
    };
    float *inner    = nxt + 7*m + 24*m;

    InitOnceExecuteOnce(&g_ds_pool_once, init_ds_pool, NULL, NULL);

    /* conv_pre: [192, n_new] -> [512, n_new] */
    conv1d_stream(&ds->conv_pre, ds->conv_pre_w_kf, 512,
                  d->conv_pre_b, z_new, n_new, cur, inner);
    /* cond(g=0) collapses to bias broadcast over time. */
    ops_bias_add_ct(cur, d->cond_b, 512, n_new);

    int C_cur = 512;
    int T_cur = n_new;

    for (int i = 0; i < LILAC_DEC_UPSAMPLES; i++) {
        int u     = d->upsample_rates[i];
        int C_out = ds->stage_channels[i];
        int T_out = T_cur * u;

        ops_leaky_relu(cur, 0.1f, (size_t)C_cur * T_cur);

        /* ups: [C_cur, T_cur] -> nxt [C_out, T_out] */
        conv_transpose1d_stream(&ds->ups[i], d->ups_w[i], C_out,
                                d->ups_b[i], cur, T_cur, nxt, inner);

        /* 3 resblocks in parallel: main thread runs j=0, workers run j=1,2.
           Each worker writes its own rb_out scratch and the final aligned
           result into rb_al[j]. */
        RbTask tasks[3];
        for (int j = 0; j < LILAC_DEC_KERNELS_PER_UP; j++) {
            tasks[j].ds         = ds;
            tasks[j].d          = d;
            tasks[j].stage_i    = i;
            tasks[j].kernel_j   = j;
            tasks[j].input      = nxt;
            tasks[j].n_in       = T_out;
            tasks[j].out        = rb_al[j];
            tasks[j].tmp_rb_out = rb_out[j];
            tasks[j].inner_scr  = inner_rb[j];
        }
        pool_submit(g_ds_pool, 0, rb_task, &tasks[1]);
        pool_submit(g_ds_pool, 1, rb_task, &tasks[2]);
        rb_task(&tasks[0]);
        pool_wait(g_ds_pool);

        /* Average rb_al[0..2] → nxt (SIMD over C_out*T_out). */
        {
            size_t n = (size_t)C_out * T_out;
            const __m256 inv3 = _mm256_set1_ps(1.0f / 3.0f);
            size_t p = 0;
            for (; p + 8 <= n; p += 8) {
                __m256 a = _mm256_loadu_ps(rb_al[0] + p);
                __m256 b = _mm256_loadu_ps(rb_al[1] + p);
                __m256 c = _mm256_loadu_ps(rb_al[2] + p);
                __m256 s = _mm256_add_ps(_mm256_add_ps(a, b), c);
                _mm256_storeu_ps(nxt + p, _mm256_mul_ps(s, inv3));
            }
            for (; p < n; p++)
                nxt[p] = (rb_al[0][p] + rb_al[1][p] + rb_al[2][p]) * (1.0f / 3.0f);
        }

        float *t = cur; cur = nxt; nxt = t;
        C_cur = C_out; T_cur = T_out;
    }

    ops_leaky_relu(cur, 0.1f, (size_t)C_cur * T_cur);
    /* conv_post: [32, T_audio] -> [1, T_audio] (bias=False) */
    conv1d_stream(&ds->conv_post, ds->conv_post_w_kf, 1, NULL,
                  cur, T_cur, audio_out, inner);
    ops_tanhf(audio_out, (size_t)T_cur);
}
