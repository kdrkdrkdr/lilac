#include "engine.h"
#include "conv.h"
#include "log.h"
#include "ops.h"

#include <cblas.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define RATE  22050
#define CHUNK 9984

static float *alloc_floats(size_t n) {
    float *p = (float *)malloc(n * sizeof(float));
    if (p) memset(p, 0, n * sizeof(float));
    return p;
}

int engine_init(Engine *e, const char *weights_path,
                const float *target_wav, int target_len, int K) {
    memset(e, 0, sizeof(*e));
    if (CHUNK % K != 0) return -1;

    /* OpenBLAS thread count. The outer 3-way parallel resblock dispatch in
       dec.c multiplies with OpenBLAS's per-sgemm threading, so we keep the
       inner count modest. Env LILAC_BLAS_THREADS overrides. */
    {
        SYSTEM_INFO si; GetSystemInfo(&si);
        int n_cores = (int)si.dwNumberOfProcessors;
        /* Floor at 4, ceiling at n_cores. 5 was the empirically-tuned sweet
           spot on 16-core; it lets 3-way parallel rb dispatch fit realtime
           comfortably. Anything below 4 starves the sgemm. */
        int n_blas  = n_cores < 4 ? n_cores : (n_cores < 5 ? 4 : 5);
        const char *env = getenv("LILAC_BLAS_THREADS");
        if (env && *env) { int v = atoi(env); if (v > 0 && v <= 64) n_blas = v; }
        openblas_set_num_threads(n_blas);
        lilac_log("[lilac] cores=%d  openblas_threads=%d  parallel_mode=%d\n",
                  n_cores, openblas_get_num_threads(), openblas_get_parallel());
    }

    if (tensor_store_load(&e->store, weights_path) != 0) return -1;
    if (model_init(&e->model, &e->store) != 0) { tensor_store_free(&e->store); return -1; }
    if (stft_plan_init(&e->stft, 1024, 256, 1024) != 0) { tensor_store_free(&e->store); return -1; }
    if (dec_stream_init(&e->ds, &e->model.dec) != 0) { tensor_store_free(&e->store); return -1; }

    /* hop is z-frame-aligned so dec_stream sees a continuous, integer-indexed
       z-frame stream across engine calls. */
    e->chunk       = CHUNK;
    e->window      = 3 * CHUNK;
    int frames_per_chunk = CHUNK / 256;                /* 39 */
    e->hop_frames  = (frames_per_chunk + K - 1) / K;   /* ceil-div */
    e->hop         = e->hop_frames * 256;
    e->emit_offset = 2 * CHUNK - e->hop;
    e->fade_len    = (int)(RATE * 0.005f);

    /* Discard the first N hops post window-fill while dec_stream's internal
       cumulative lag is being primed with real z context. After N hops we've
       fed N*hop samples of output through dec_stream; the earliest valid
       emit sample is at stream position cumulative_lag_audio, so we need
       N*hop >= cumulative_lag_audio. */
    long cl = e->ds.cumulative_lag_audio;
    e->priming_hops_needed = (int)((cl + e->hop - 1) / e->hop);
    e->priming_hops_done   = 0;

    e->window_buf = alloc_floats((size_t)e->window);
    e->fade       = alloc_floats((size_t)e->fade_len);
    if (!e->window_buf || !e->fade) return -1;
    for (int i = 0; i < e->fade_len; i++) {
        float c = cosf((float)(M_PI / 2.0 * i / (double)e->fade_len));
        e->fade[i] = c * c;
    }

    int frames_max = stft_frame_count(&e->stft, e->window);
    e->spec          = alloc_floats((size_t)e->stft.half_bins * frames_max);
    e->wav_out       = alloc_floats((size_t)e->window);
    e->stft_scratch  = alloc_floats((size_t)(e->window + 2 * e->stft.pad) + 2 * e->stft.n_fft);
    e->model_scratch = alloc_floats((size_t)model_scratch_floats(&e->model, frames_max));
    e->target_se     = alloc_floats((size_t)e->model.gin_channels);
    e->source_se     = alloc_floats((size_t)e->model.gin_channels);
    if (!e->spec || !e->wav_out || !e->stft_scratch || !e->model_scratch
        || !e->target_se || !e->source_se) return -1;

    /* Extract target SE once from the provided reference clip. */
    int tgt_frames = stft_frame_count(&e->stft, target_len);
    float *tgt_spec = alloc_floats((size_t)e->stft.half_bins * tgt_frames);
    float *tgt_stft_scr = alloc_floats((size_t)(target_len + 2 * e->stft.pad) + 2 * e->stft.n_fft);
    float *tgt_ref_scr  = alloc_floats((size_t)ref_enc_scratch_floats(&e->model.ref_enc, tgt_frames));
    if (!tgt_spec || !tgt_stft_scr || !tgt_ref_scr) {
        free(tgt_spec); free(tgt_stft_scr); free(tgt_ref_scr);
        return -1;
    }
    stft_magnitude(&e->stft, target_wav, target_len, tgt_spec, tgt_stft_scr);
    model_extract_se(&e->model, tgt_spec, tgt_frames, e->target_se, tgt_ref_scr);
    free(tgt_spec); free(tgt_stft_scr); free(tgt_ref_scr);
    return 0;
}

void engine_free(Engine *e) {
    dec_stream_free(&e->ds);
    model_free(&e->model);
    free(e->window_buf);
    free(e->fade);
    free(e->spec);
    free(e->wav_out);
    free(e->stft_scratch);
    free(e->model_scratch);
    free(e->target_se);
    free(e->source_se);
    stft_plan_free(&e->stft);
    tensor_store_free(&e->store);
    memset(e, 0, sizeof(*e));
}

void engine_reset_source(Engine *e) {
    e->source_se_ready    = 0;
    e->has_prev_last      = 0;
    dec_stream_reset(&e->ds);
    e->priming_hops_done  = 0;
}

static void smooth_boundary(Engine *e, float *out, int n) {
    if (!e->has_prev_last) {
        e->prev_last = out[n - 1];
        e->has_prev_last = 1;
        return;
    }
    float delta = e->prev_last - out[0];
    int m = e->fade_len < n ? e->fade_len : n;
    for (int i = 0; i < m; i++) out[i] += delta * e->fade[i];
    e->prev_last = out[n - 1];
}

const float *engine_process_hop(Engine *e, const float *input_hop) {
    /* Slide window left by hop, append new hop at the tail. */
    memmove(e->window_buf, e->window_buf + e->hop,
            (size_t)(e->window - e->hop) * sizeof(float));
    memcpy(e->window_buf + (e->window - e->hop), input_hop,
           (size_t)e->hop * sizeof(float));

    if (e->window_filled < e->window) e->window_filled += e->hop;
    if (e->window_filled < e->window) {
        /* Warmup: emit silence until window is full. */
        memset(e->wav_out, 0, (size_t)e->hop * sizeof(float));
        return e->wav_out;
    }

    /* STFT → spec */
    int frames = stft_frame_count(&e->stft, e->window);
    stft_magnitude(&e->stft, e->window_buf, e->window, e->spec, e->stft_scratch);

    /* Cache source SE on the first ready window. */
    if (!e->source_se_ready) {
        /* ref_enc mutates spec via layernorm — we run it on a separate copy so we
           can reuse `spec` for enc_q. */
        size_t spec_n = (size_t)e->stft.half_bins * frames;
        float *spec_copy = alloc_floats(spec_n);
        memcpy(spec_copy, e->spec, spec_n * sizeof(float));
        /* Reuse model_scratch for ref_enc (fits since model_scratch is the max). */
        model_extract_se(&e->model, spec_copy, frames, e->source_se, e->model_scratch);
        free(spec_copy);
        e->source_se_ready = 1;
    }

    /* Streaming dec: feed `hop_frames` new z-frames per call. The feed region
       sits inside the RF-stable middle of the current window, with
       LILAC_SPEC_SLICE_RIGHT_FRAMES of right context kept after it. */
    float *emit = e->wav_out + e->emit_offset;
    int z_feed_end_frame = frames - 50;   /* matches LILAC_SPEC_SLICE_RIGHT_FRAMES */
    model_forward_stream(&e->model, &e->ds, e->spec, frames,
                         e->source_se, e->target_se,
                         z_feed_end_frame, e->hop_frames,
                         emit, e->model_scratch);

    if (e->priming_hops_done < e->priming_hops_needed) {
        e->priming_hops_done++;
        memset(emit, 0, (size_t)e->hop * sizeof(float));
        return emit;
    }

    ops_clamp_nan(emit, (size_t)e->hop);
    smooth_boundary(e, emit, e->hop);
    return emit;
}
