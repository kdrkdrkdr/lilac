#include "vc.h"

#include "audio_io.h"
#include "engine.h"
#include "miniaudio.h"
#include "rnnoise.h"

#include <cblas.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include "log.h"

static double wall_ms(void) {
    static LARGE_INTEGER freq = {0};
    if (!freq.QuadPart) QueryPerformanceFrequency(&freq);
    LARGE_INTEGER c; QueryPerformanceCounter(&c);
    return (double)c.QuadPart * 1000.0 / (double)freq.QuadPart;
}

/* ---------------------------------------------------------------------------
   Audio path:
     mic (48 kHz, device side) ──► in48_ring ──► worker:
         rnnoise_process_frame (480-sample chunks, VAD + denoise)
             ├── gate: vad < threshold → zero frame (engine will see silence)
             └── denoise_on ? denoised : raw  → resampler 48 → 22050
         engine_process_hop (22050, HOP = eng.hop)
         resampler 22050 → 48  → out48_ring ──► playback (48 kHz, device side)

   Device is opened at 48 kHz so RNNoise (which is 48 k-only) runs directly
   on mic audio. VC stays at 22050 internally — resampling happens on both
   sides inside the worker.
--------------------------------------------------------------------------- */

#define DEVICE_RATE      48000
#define DEVICE_BLOCK     1920      /* 40 ms @ 48 k per audio callback */
#define RNN_FRAME        480       /* 10 ms @ 48 k */
#define INT16_SCALE      32768.0f
#define VAD_HOLD_FRAMES     8      /* ~80 ms hangover after last active frame */
#define OUT_SILENT_HOLD_HOPS 4     /* ~600 ms of output pass-through to drain
                                      dec_stream / flow lookahead buffers
                                      before the silence-gate kills bias. */

/* AGC tuning — one-pole coefs applied per 10 ms frame (100 frames/s).
   coef = exp(-1/(tau*rate)). Attack (gain-down) faster than release (up). */
#define AGC_ATTACK_COEF   0.82f    /* ≈ 50 ms  time constant */
#define AGC_RELEASE_COEF  0.975f   /* ≈ 400 ms time constant */
#define AGC_MAX_GAIN      10.0f    /* +20 dB ceiling, don't boost noise */
#define AGC_NOISE_FLOOR   1e-4f    /* RMS below this → leave frame untouched */

/* Ring cap ≈ 0.5 s of 48 k audio — plenty for worker/callback jitter. */
#define RING48_CAP       24000

struct LilacVC {
    Engine      eng;
    AudioIO    *io;

    /* 48 kHz rings across audio-cb ↔ rnn-thread boundary and
       vc-thread ↔ audio-cb boundary. */
    float      *in48_ring;   size_t in48_cap, in48_r, in48_w, in48_cnt;
    float      *out48_ring;  size_t out48_cap, out48_r, out48_w, out48_cnt;
    CRITICAL_SECTION in48_cs, out48_cs;

    /* 22 kHz ring between rnn thread (producer) and vc thread (consumer). */
    float      *vc22_ring;   size_t vc22_cap, vc22_r, vc22_w, vc22_cnt;
    CRITICAL_SECTION vc22_cs;

    /* Two workers. rnn thread: 48 k → rnnoise → 48→22 resample → vc22_ring.
       vc thread:  vc22_ring → engine_process_hop → 22→48 resample → out48_ring. */
    HANDLE     rnn_thread;
    HANDLE     vc_thread;
    HANDLE     rnn_work_event;    /* signalled by audio_cb when in48 has data */
    HANDLE     vc_work_event;     /* signalled by rnn thread when vc22 has >= HOP */
    HANDLE     stop_event;

    /* rnn-thread-owned scratch. */
    float     *work48_in;         /* drained from in48_ring each iter (≤ RING48_CAP) */
    float     *rnn_tail;          /* rnnoise-frame remainder across iters */
    int        rnn_tail_n;
    float     *rnn_in_scaled;     /* RNN_FRAME, ×32768 for rnnoise */
    float     *rnn_out;           /* RNN_FRAME denoised */
    float     *rnn_raw;           /* RNN_FRAME raw (pre-scale) for denoise_off path */
    float     *rnn_22_out;        /* RNN_FRAME → 22 kHz per-frame (≤ 222 samples) */
    size_t     rnn_22_out_cap;

    /* vc-thread-owned scratch. */
    float     *hop22_in;          /* HOP samples popped from vc22_ring */
    float     *hop_out22;         /* HOP copy of engine output */
    float     *out48_block;       /* engine output resampled back to 48 k */
    size_t     out48_block_cap;

    /* Rate converters (mono, f32). 48→22 owned by rnn thread,
       22→48 owned by vc thread. State persists across iters. */
    ma_linear_resampler r48_to_22;
    ma_linear_resampler r22_to_48;

    /* Engine guard (target swap / source reset vs worker). */
    CRITICAL_SECTION engine_cs;

    /* RNNoise state + config. */
    DenoiseState *rn;
    float         vad_threshold;
    int           denoise_on;
    float         last_vad;
    int           vad_hold_remaining;   /* 10 ms frames left in hangover */

    /* AGC (operates on 10 ms frames at 48 kHz between VAD gate and 48→22
       resample). One-pole smoothing; fast gain-down (attack), slow gain-up
       (release). gain stays put when VAD silences the frame. */
    int           agc_on;
    float         agc_target_linear;   /* 10^(target_db/20) */
    float         agc_gain;            /* smoothed, current-frame gain */

    /* Output silence-gate hangover. dec_stream + flow lookahead buffers
       up to ~500 ms of input inside the engine; zeroing the first silent
       hop would throw that trailing audio away. We keep output flowing
       for out_silent_hold_remaining HOPs after input goes silent, then
       blank to kill bias residue. */
    int        out_silent_hold_remaining;

    /* Stats. */
    int        in_dropped;
    int        out_dropped;
    float      in_rms;
    float      out_rms;
    float      avg_process_ms;
    float      recent_proc_ms;
    double     sum_process_ms;
    long       n_blocks;
};

static float block_rms(const float *x, unsigned n) {
    if (!x || n == 0) return 0.0f;
    double s = 0.0;
    for (unsigned i = 0; i < n; i++) s += (double)x[i] * x[i];
    return (float)sqrt(s / (double)n);
}

/* AGC: adjust gain towards target RMS. In-place on an RNN_FRAME block.
   Writes *out_buf* (may alias in) with the scaled + soft-clipped signal.
   Gain is smoothed across frames via one-pole filtering. */
static void agc_process(LilacVC *vc, float *buf) {
    float rms = block_rms(buf, RNN_FRAME);
    if (rms < AGC_NOISE_FLOOR) return;            /* silence → leave alone */

    float desired = vc->agc_target_linear / rms;
    if (desired > AGC_MAX_GAIN) desired = AGC_MAX_GAIN;

    float coef = desired < vc->agc_gain ? AGC_ATTACK_COEF : AGC_RELEASE_COEF;
    vc->agc_gain = coef * vc->agc_gain + (1.0f - coef) * desired;

    float g = vc->agc_gain;
    for (int i = 0; i < RNN_FRAME; i++) {
        float v = buf[i] * g;
        if (v >  1.0f) v =  1.0f;
        if (v < -1.0f) v = -1.0f;
        buf[i] = v;
    }
}

/* ---- Ring helpers ---- */

static size_t ring_push_in48(LilacVC *vc, const float *src, size_t n) {
    size_t accepted = 0;
    for (size_t i = 0; i < n; i++) {
        if (vc->in48_cnt >= vc->in48_cap) { vc->in_dropped++; continue; }
        vc->in48_ring[vc->in48_w] = src[i];
        vc->in48_w = (vc->in48_w + 1) % vc->in48_cap;
        vc->in48_cnt++;
        accepted++;
    }
    return accepted;
}

static size_t ring_drain_in48(LilacVC *vc, float *dst, size_t max_n) {
    size_t n = vc->in48_cnt < max_n ? vc->in48_cnt : max_n;
    for (size_t i = 0; i < n; i++) {
        dst[i] = vc->in48_ring[vc->in48_r];
        vc->in48_r = (vc->in48_r + 1) % vc->in48_cap;
    }
    vc->in48_cnt -= n;
    return n;
}

static void ring_push_out48(LilacVC *vc, const float *src, size_t n) {
    for (size_t i = 0; i < n; i++) {
        vc->out48_ring[vc->out48_w] = src[i];
        vc->out48_w = (vc->out48_w + 1) % vc->out48_cap;
        if (vc->out48_cnt < vc->out48_cap) vc->out48_cnt++;
        else vc->out48_r = (vc->out48_r + 1) % vc->out48_cap;   /* overwrite oldest */
    }
}

static size_t ring_pop_out48(LilacVC *vc, float *dst, size_t n) {
    size_t take = n < vc->out48_cnt ? n : vc->out48_cnt;
    for (size_t i = 0; i < take; i++) {
        dst[i] = vc->out48_ring[vc->out48_r];
        vc->out48_r = (vc->out48_r + 1) % vc->out48_cap;
    }
    vc->out48_cnt -= take;
    return take;
}

static void ring_push_vc22(LilacVC *vc, const float *src, size_t n) {
    for (size_t i = 0; i < n; i++) {
        vc->vc22_ring[vc->vc22_w] = src[i];
        vc->vc22_w = (vc->vc22_w + 1) % vc->vc22_cap;
        if (vc->vc22_cnt < vc->vc22_cap) vc->vc22_cnt++;
        else vc->vc22_r = (vc->vc22_r + 1) % vc->vc22_cap;  /* overwrite oldest if full */
    }
}

static size_t ring_pop_vc22(LilacVC *vc, float *dst, size_t n) {
    if (vc->vc22_cnt < n) return 0;
    for (size_t i = 0; i < n; i++) {
        dst[i] = vc->vc22_ring[vc->vc22_r];
        vc->vc22_r = (vc->vc22_r + 1) % vc->vc22_cap;
    }
    vc->vc22_cnt -= n;
    return n;
}

/* ---- RNNoise worker: 48 k input → denoise + VAD gate → 48→22 resample
        → vc22_ring. Signals vc_work_event when vc22_ring has >= HOP. ---- */

static DWORD WINAPI rnn_worker_proc(LPVOID arg) {
    LilacVC *vc = (LilacVC *)arg;
    const int HOP = vc->eng.hop;
    HANDLE objs[2] = { vc->rnn_work_event, vc->stop_event };

    for (;;) {
        /* Drain in48_ring → work48_in. */
        EnterCriticalSection(&vc->in48_cs);
        size_t n_in48 = ring_drain_in48(vc, vc->work48_in, RING48_CAP);
        LeaveCriticalSection(&vc->in48_cs);

        if (n_in48 == 0) {
            DWORD r = WaitForMultipleObjects(2, objs, FALSE, INFINITE);
            if (r == WAIT_OBJECT_0 + 1) return 0;
            continue;
        }

        int consume_src = 0;
        int pushed_this_iter = 0;
        for (;;) {
            int avail = vc->rnn_tail_n + ((int)n_in48 - consume_src);
            if (avail < RNN_FRAME) break;

            /* Assemble one 480-sample frame from (rnn_tail ++ work48_in). */
            int take_from_tail = vc->rnn_tail_n;
            if (take_from_tail > RNN_FRAME) take_from_tail = RNN_FRAME;
            int take_from_src  = RNN_FRAME - take_from_tail;

            memcpy(vc->rnn_raw, vc->rnn_tail, (size_t)take_from_tail * sizeof(float));
            memcpy(vc->rnn_raw + take_from_tail,
                   vc->work48_in + consume_src,
                   (size_t)take_from_src * sizeof(float));

            int tail_remaining = vc->rnn_tail_n - take_from_tail;
            if (tail_remaining > 0)
                memmove(vc->rnn_tail,
                        vc->rnn_tail + take_from_tail,
                        (size_t)tail_remaining * sizeof(float));
            vc->rnn_tail_n = tail_remaining;
            consume_src += take_from_src;

            /* rnnoise: scale to int16 range, denoise, scale back. */
            for (int i = 0; i < RNN_FRAME; i++)
                vc->rnn_in_scaled[i] = vc->rnn_raw[i] * INT16_SCALE;
            float vad = rnnoise_process_frame(vc->rn, vc->rnn_out, vc->rnn_in_scaled);
            vc->last_vad = vad;
            for (int i = 0; i < RNN_FRAME; i++)
                vc->rnn_out[i] *= 1.0f / INT16_SCALE;

            /* VAD gate with short hangover: once the frame is active (vad >=
               threshold) we refill the hold counter. Subsequent frames below
               threshold keep passing until the counter drains, which lets
               speech tails (decaying consonants, breath) survive instead of
               getting clipped. */
            float *src_for_vc = vc->denoise_on ? vc->rnn_out : vc->rnn_raw;
            if (vad >= vc->vad_threshold) {
                vc->vad_hold_remaining = VAD_HOLD_FRAMES;
            } else if (vc->vad_hold_remaining > 0) {
                vc->vad_hold_remaining--;
            } else {
                memset(vc->rnn_out, 0, RNN_FRAME * sizeof(float));
                src_for_vc = vc->rnn_out;
            }

            /* AGC on the gated/denoised frame before it enters the VC path.
               agc_process is a no-op on silent frames (RMS < noise floor)
               so zeroed VAD-gated frames stay zero. */
            if (vc->agc_on) agc_process(vc, src_for_vc);

            /* 48 → 22050 resample this frame; push to vc22_ring. */
            ma_uint64 in_f  = RNN_FRAME;
            ma_uint64 out_f = vc->rnn_22_out_cap;
            ma_linear_resampler_process_pcm_frames(
                &vc->r48_to_22,
                src_for_vc, &in_f,
                vc->rnn_22_out, &out_f);

            if (out_f > 0) {
                EnterCriticalSection(&vc->vc22_cs);
                ring_push_vc22(vc, vc->rnn_22_out, (size_t)out_f);
                pushed_this_iter = 1;
                int enough = vc->vc22_cnt >= (size_t)HOP;
                LeaveCriticalSection(&vc->vc22_cs);
                if (enough) SetEvent(vc->vc_work_event);
            }
        }

        /* Save source leftover to rnn_tail for next iter. */
        int leftover_src = (int)n_in48 - consume_src;
        if (leftover_src > 0) {
            memcpy(vc->rnn_tail + vc->rnn_tail_n,
                   vc->work48_in + consume_src,
                   (size_t)leftover_src * sizeof(float));
            vc->rnn_tail_n += leftover_src;
        }
        (void)pushed_this_iter;

        if (WaitForSingleObject(vc->stop_event, 0) == WAIT_OBJECT_0) return 0;
    }
}

/* ---- VC worker: vc22_ring → engine_process_hop → 22→48 resample
        → out48_ring. Waits on vc_work_event signaled by rnn thread. ---- */

static DWORD WINAPI vc_worker_proc(LPVOID arg) {
    LilacVC *vc = (LilacVC *)arg;
    const int HOP = vc->eng.hop;
    HANDLE objs[2] = { vc->vc_work_event, vc->stop_event };

    for (;;) {
        /* Drain as many full HOPs as currently buffered — no re-wait in between. */
        for (;;) {
            if (WaitForSingleObject(vc->stop_event, 0) == WAIT_OBJECT_0) return 0;
            EnterCriticalSection(&vc->vc22_cs);
            size_t got = ring_pop_vc22(vc, vc->hop22_in, (size_t)HOP);
            LeaveCriticalSection(&vc->vc22_cs);
            if (got != (size_t)HOP) break;

            /* Silence-gate with a HOP-level hangover. When input goes
               silent we keep letting output through for
               OUT_SILENT_HOLD_HOPS hops so the dec_stream / flow
               lookahead buffers can flush the real speech tail
               (several hundred ms) that would otherwise be discarded.
               Only once the counter drains do we force output to zero
               to kill the faint bias residue. Active input refills it. */
            int input_silent = block_rms(vc->hop22_in, (unsigned)HOP) < 1e-5f;
            int force_zero   = 0;
            if (!input_silent) {
                vc->out_silent_hold_remaining = OUT_SILENT_HOLD_HOPS;
            } else if (vc->out_silent_hold_remaining > 0) {
                vc->out_silent_hold_remaining--;
            } else {
                force_zero = 1;
            }

            double t0 = wall_ms();
            EnterCriticalSection(&vc->engine_cs);
            const float *out22 = engine_process_hop(&vc->eng, vc->hop22_in);
            LeaveCriticalSection(&vc->engine_cs);
            double dt = wall_ms() - t0;

            memcpy(vc->hop_out22, out22, (size_t)HOP * sizeof(float));

            vc->sum_process_ms += dt;
            vc->n_blocks       += 1;
            vc->avg_process_ms  = (float)(vc->sum_process_ms / (double)vc->n_blocks);
            vc->recent_proc_ms  = vc->recent_proc_ms == 0.0f
                                ? (float)dt
                                : 0.9f * vc->recent_proc_ms + 0.1f * (float)dt;

            ma_uint64 in_f  = (ma_uint64)HOP;
            ma_uint64 out_f = vc->out48_block_cap;
            ma_linear_resampler_process_pcm_frames(
                &vc->r22_to_48,
                vc->hop_out22, &in_f,
                vc->out48_block, &out_f);

            if (force_zero)
                memset(vc->out48_block, 0, (size_t)out_f * sizeof(float));

            EnterCriticalSection(&vc->out48_cs);
            ring_push_out48(vc, vc->out48_block, (size_t)out_f);
            LeaveCriticalSection(&vc->out48_cs);
        }

        DWORD r = WaitForMultipleObjects(2, objs, FALSE, INFINITE);
        if (r == WAIT_OBJECT_0 + 1) return 0;
    }
}

/* ---- Audio callback (48 kHz, device-native) ---- */

static void audio_cb(const float *in, float *out, unsigned n, void *user) {
    LilacVC *vc = (LilacVC *)user;

    EnterCriticalSection(&vc->in48_cs);
    ring_push_in48(vc, in, n);
    int should_signal = vc->in48_cnt >= RNN_FRAME;
    LeaveCriticalSection(&vc->in48_cs);
    if (should_signal) SetEvent(vc->rnn_work_event);

    EnterCriticalSection(&vc->out48_cs);
    size_t got = ring_pop_out48(vc, out, n);
    LeaveCriticalSection(&vc->out48_cs);
    if (got < n) {
        memset(out + got, 0, (size_t)(n - got) * sizeof(float));
        vc->out_dropped += (int)(n - got);
    }

    vc->in_rms  = block_rms(in,  n);
    vc->out_rms = block_rms(out, n);
}

/* ---- Public API ---- */

LilacVC *lilac_create(const char *weights_path,
                      const float *target_wav, int target_len,
                      int K) {
    if (!weights_path || !target_wav || target_len <= 0 || K <= 0) return NULL;

    LilacVC *vc = (LilacVC *)calloc(1, sizeof(LilacVC));
    if (!vc) return NULL;
    if (engine_init(&vc->eng, weights_path, target_wav, target_len, K) != 0) {
        free(vc); return NULL;
    }
    vc->io = audio_io_create();
    if (!vc->io) { engine_free(&vc->eng); free(vc); return NULL; }

    InitializeCriticalSection(&vc->in48_cs);
    InitializeCriticalSection(&vc->out48_cs);
    InitializeCriticalSection(&vc->vc22_cs);
    InitializeCriticalSection(&vc->engine_cs);

    const int HOP = vc->eng.hop;

    vc->in48_cap  = RING48_CAP;
    vc->out48_cap = RING48_CAP;
    vc->vc22_cap  = (size_t)(HOP * 4);         /* ~1 s @ 22050 between threads */
    vc->in48_ring  = (float *)calloc(vc->in48_cap,  sizeof(float));
    vc->out48_ring = (float *)calloc(vc->out48_cap, sizeof(float));
    vc->vc22_ring  = (float *)calloc(vc->vc22_cap,  sizeof(float));
    if (!vc->in48_ring || !vc->out48_ring || !vc->vc22_ring) goto fail;

    /* rnn-thread scratch. */
    vc->work48_in     = (float *)calloc(RING48_CAP, sizeof(float));
    vc->rnn_tail      = (float *)calloc(RING48_CAP + RNN_FRAME, sizeof(float));
    vc->rnn_in_scaled = (float *)calloc(RNN_FRAME, sizeof(float));
    vc->rnn_out       = (float *)calloc(RNN_FRAME, sizeof(float));
    vc->rnn_raw       = (float *)calloc(RNN_FRAME, sizeof(float));
    vc->rnn_22_out_cap = RNN_FRAME;          /* downsample 48→22 ⇒ ≤ 222 frames, 480 is generous */
    vc->rnn_22_out    = (float *)calloc(vc->rnn_22_out_cap, sizeof(float));

    /* vc-thread scratch. */
    vc->hop22_in      = (float *)calloc((size_t)HOP, sizeof(float));
    vc->hop_out22     = (float *)calloc((size_t)HOP, sizeof(float));
    /* 22k → 48k upsamples by factor 48/22.05 ≈ 2.177; round up + margin. */
    vc->out48_block_cap = (size_t)((long long)HOP * DEVICE_RATE / 22050) + 1024;
    vc->out48_block   = (float *)calloc(vc->out48_block_cap, sizeof(float));

    if (!vc->work48_in || !vc->rnn_tail || !vc->rnn_in_scaled
        || !vc->rnn_out || !vc->rnn_raw || !vc->rnn_22_out
        || !vc->hop22_in || !vc->hop_out22 || !vc->out48_block) goto fail;

    /* Resamplers. */
    {
        ma_linear_resampler_config c1 = ma_linear_resampler_config_init(
            ma_format_f32, 1, DEVICE_RATE, 22050);
        if (ma_linear_resampler_init(&c1, NULL, &vc->r48_to_22) != MA_SUCCESS) goto fail;
        ma_linear_resampler_config c2 = ma_linear_resampler_config_init(
            ma_format_f32, 1, 22050, DEVICE_RATE);
        if (ma_linear_resampler_init(&c2, NULL, &vc->r22_to_48) != MA_SUCCESS) goto fail;
    }

    /* RNNoise with default built-in model. */
    vc->rn = rnnoise_create(NULL);
    if (!vc->rn) goto fail;

    vc->vad_threshold     = 0.90f;
    vc->denoise_on        = 0;   /* raw → VC; rnnoise still runs for VAD */
    vc->last_vad          = 0.0f;
    vc->vad_hold_remaining = 0;
    vc->out_silent_hold_remaining = 0;
    vc->agc_on            = 1;
    vc->agc_target_linear = 0.1f;    /* -20 dBFS */
    vc->agc_gain          = 1.0f;

    vc->rnn_work_event = CreateEventW(NULL, FALSE, FALSE, NULL);
    vc->vc_work_event  = CreateEventW(NULL, FALSE, FALSE, NULL);
    vc->stop_event     = CreateEventW(NULL, TRUE,  FALSE, NULL);
    if (!vc->rnn_work_event || !vc->vc_work_event || !vc->stop_event) goto fail;

    vc->rnn_thread = CreateThread(NULL, 0, rnn_worker_proc, vc, 0, NULL);
    vc->vc_thread  = CreateThread(NULL, 0, vc_worker_proc,  vc, 0, NULL);
    if (!vc->rnn_thread || !vc->vc_thread) goto fail;

    return vc;

fail:
    lilac_destroy(vc);
    return NULL;
}

void lilac_destroy(LilacVC *vc) {
    if (!vc) return;
    if (vc->io) audio_io_stop(vc->io);

    if (vc->stop_event) SetEvent(vc->stop_event);
    if (vc->rnn_work_event) SetEvent(vc->rnn_work_event);
    if (vc->vc_work_event)  SetEvent(vc->vc_work_event);
    if (vc->rnn_thread) {
        WaitForSingleObject(vc->rnn_thread, INFINITE);
        CloseHandle(vc->rnn_thread);
    }
    if (vc->vc_thread) {
        WaitForSingleObject(vc->vc_thread, INFINITE);
        CloseHandle(vc->vc_thread);
    }
    if (vc->rnn_work_event) CloseHandle(vc->rnn_work_event);
    if (vc->vc_work_event)  CloseHandle(vc->vc_work_event);
    if (vc->stop_event) CloseHandle(vc->stop_event);

    if (vc->io) audio_io_destroy(vc->io);

    if (vc->rn) rnnoise_destroy(vc->rn);
    ma_linear_resampler_uninit(&vc->r48_to_22, NULL);
    ma_linear_resampler_uninit(&vc->r22_to_48, NULL);

    DeleteCriticalSection(&vc->in48_cs);
    DeleteCriticalSection(&vc->out48_cs);
    DeleteCriticalSection(&vc->vc22_cs);
    DeleteCriticalSection(&vc->engine_cs);

    free(vc->in48_ring);
    free(vc->out48_ring);
    free(vc->vc22_ring);
    free(vc->work48_in);
    free(vc->rnn_tail);
    free(vc->rnn_in_scaled);
    free(vc->rnn_out);
    free(vc->rnn_raw);
    free(vc->rnn_22_out);
    free(vc->hop22_in);
    free(vc->hop_out22);
    free(vc->out48_block);
    engine_free(&vc->eng);
    free(vc);
}

static int list_devices_from(AudioIO *io, LilacDevice *out, int max_count) {
    AudioDeviceInfo tmp[64];
    int n = audio_io_list_devices(io, tmp, (int)(sizeof(tmp) / sizeof(tmp[0])));
    if (out && max_count > 0) {
        int m = n < max_count ? n : max_count;
        for (int i = 0; i < m; i++) {
            out[i].id         = tmp[i].id;
            out[i].is_input   = tmp[i].is_input;
            out[i].is_default = tmp[i].is_default;
            memcpy(out[i].name, tmp[i].name, sizeof(out[i].name));
        }
    }
    return n;
}

int lilac_list_devices(LilacVC *vc, LilacDevice *out, int max_count) {
    if (!vc) return 0;
    return list_devices_from(vc->io, out, max_count);
}

int lilac_enum_devices(LilacDevice *out, int max_count) {
    AudioIO *io = audio_io_create();
    if (!io) return 0;
    int n = list_devices_from(io, out, max_count);
    audio_io_destroy(io);
    return n;
}

int lilac_start(LilacVC *vc, int input_id, int output_id) {
    if (!vc) return -1;
    /* Device side runs at 48 kHz so rnnoise gets its native rate. */
    return audio_io_start(vc->io, input_id, output_id,
                          DEVICE_RATE, DEVICE_BLOCK, audio_cb, vc);
}

void lilac_stop(LilacVC *vc) {
    if (!vc) return;
    audio_io_stop(vc->io);
}

int lilac_sample_rate(const LilacVC *vc) { (void)vc; return DEVICE_RATE; }
int lilac_hop_samples(const LilacVC *vc) { return vc ? vc->eng.hop : 0; }

int lilac_set_target(LilacVC *vc, const float *wav, int len) {
    if (!vc || !wav || len <= 0) return -1;
    int frames = stft_frame_count(&vc->eng.stft, len);
    float *tmp_spec = (float *)calloc((size_t)vc->eng.stft.half_bins * frames, sizeof(float));
    float *tmp_sscr = (float *)calloc((size_t)(len + 2 * vc->eng.stft.pad) + 2 * vc->eng.stft.n_fft, sizeof(float));
    float *tmp_mscr = (float *)calloc((size_t)ref_enc_scratch_floats(&vc->eng.model.ref_enc, frames), sizeof(float));
    float *new_se   = (float *)calloc((size_t)vc->eng.model.gin_channels, sizeof(float));
    if (!tmp_spec || !tmp_sscr || !tmp_mscr || !new_se) {
        free(tmp_spec); free(tmp_sscr); free(tmp_mscr); free(new_se);
        return -1;
    }
    stft_magnitude(&vc->eng.stft, wav, len, tmp_spec, tmp_sscr);
    model_extract_se(&vc->eng.model, tmp_spec, frames, new_se, tmp_mscr);

    EnterCriticalSection(&vc->engine_cs);
    memcpy(vc->eng.target_se, new_se, (size_t)vc->eng.model.gin_channels * sizeof(float));
    LeaveCriticalSection(&vc->engine_cs);

    free(tmp_spec); free(tmp_sscr); free(tmp_mscr); free(new_se);
    return 0;
}

void lilac_reset_source(LilacVC *vc) {
    if (!vc) return;
    EnterCriticalSection(&vc->engine_cs);
    engine_reset_source(&vc->eng);
    LeaveCriticalSection(&vc->engine_cs);
}

void lilac_get_stats(const LilacVC *vc, LilacStats *out) {
    if (!vc || !out) return;
    out->input_rms      = vc->in_rms;
    out->output_rms     = vc->out_rms;
    out->dropped_frames = vc->in_dropped + vc->out_dropped;
    out->avg_process_ms = vc->avg_process_ms;
    out->recent_proc_ms = vc->recent_proc_ms;
    out->is_running     = audio_io_is_running(vc->io);
    out->last_vad       = vc->last_vad;
}

void lilac_set_vad_threshold(LilacVC *vc, float threshold) {
    if (!vc) return;
    if (threshold < 0.0f) threshold = 0.0f;
    if (threshold > 1.0f) threshold = 1.0f;
    vc->vad_threshold = threshold;
}

float lilac_get_vad_threshold(const LilacVC *vc) {
    return vc ? vc->vad_threshold : 0.0f;
}

void lilac_set_denoise(LilacVC *vc, int enabled) {
    if (!vc) return;
    vc->denoise_on = enabled ? 1 : 0;
}

int lilac_get_denoise(const LilacVC *vc) {
    return vc ? vc->denoise_on : 0;
}

void lilac_set_agc(LilacVC *vc, int enabled) {
    if (!vc) return;
    vc->agc_on = enabled ? 1 : 0;
    if (!vc->agc_on) vc->agc_gain = 1.0f;   /* reset so re-enable starts neutral */
}

int lilac_get_agc(const LilacVC *vc) {
    return vc ? vc->agc_on : 0;
}

void lilac_set_agc_target_db(LilacVC *vc, float db) {
    if (!vc) return;
    if (db >  0.0f)  db =  0.0f;    /* target above 0 dBFS makes no sense */
    if (db < -60.0f) db = -60.0f;
    vc->agc_target_linear = powf(10.0f, db / 20.0f);
}

float lilac_get_agc_target_db(const LilacVC *vc) {
    if (!vc) return 0.0f;
    return 20.0f * log10f(vc->agc_target_linear);
}
