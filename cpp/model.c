#include "model.h"
#include "tensor.h"

#include <stdlib.h>
#include <string.h>

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include "log.h"

static double _ms(void) {
    static LARGE_INTEGER freq = {0};
    if (!freq.QuadPart) QueryPerformanceFrequency(&freq);
    LARGE_INTEGER c; QueryPerformanceCounter(&c);
    return (double)c.QuadPart * 1000.0 / (double)freq.QuadPart;
}

int model_init(Model *m, const TensorStore *store) {
    memset(m, 0, sizeof(*m));
    m->spec_channels  = 513;
    m->inter_channels = 192;
    m->gin_channels   = 256;
    m->hop_length     = 256;
    if (ref_enc_init(&m->ref_enc, store, m->spec_channels, m->gin_channels) != 0) return -1;
    if (enc_q_init(&m->enc_q, store) != 0) return -1;
    if (flow_init(&m->flow, store) != 0) return -1;
    if (dec_init(&m->dec, store) != 0) return -1;
    return 0;
}

void model_free(Model *m) {
    enc_q_free(&m->enc_q);
    flow_free(&m->flow);
    dec_free(&m->dec);
}

int model_scratch_floats(const Model *m, int T_max) {
    int ref  = ref_enc_scratch_floats(&m->ref_enc, T_max);
    int enc  = enc_q_scratch_floats(&m->enc_q, T_max);
    int flw  = flow_scratch_floats(&m->flow, T_max);
    int dec_s = dec_scratch_floats(&m->dec, T_max);
    /* Persistent across stages: z [inter, T], z_hat reuses z, plus scratch-per-stage.
       We pick max since stages run sequentially and reuse the same scratch region. */
    int staged = ref;
    if (enc > staged) staged = enc;
    if (flw > staged) staged = flw;
    if (dec_s > staged) staged = dec_s;
    int z = m->inter_channels * T_max;
    return z + staged + 1024;
}

void model_extract_se(Model *m, float *spec, int T, float *out_se, float *scratch) {
    ref_enc_forward(&m->ref_enc, spec, T, out_se, scratch);
}

/* Number of spec frames of left/right context to keep around the emit region
   when slicing z for dec. Must exceed dec's receptive field at z-input to keep
   emit bit-exact. Dec's theoretical RF at output ≈ 121 audio samples < 1 spec
   frame, but cascaded stage RFs multiply upward; 4/2 is measured-safe. */
/* Dec receptive field in z-frames measured empirically via probe_rf.exe:
   error drops to ~5e-7 (fp32 noise floor) at left=right=15. Used on the
   z (post-flow) slice fed to dec. */
#define LILAC_DEC_SLICE_LEFT_FRAMES   15
#define LILAC_DEC_SLICE_RIGHT_FRAMES  15

/* Full-pipeline (enc_q + flow + dec) receptive field at the spec input,
   measured via probe_full.exe. Left/right=40 drives emit error into the
   fp-noise floor (~6e-6) across all interior emit positions. */
#define LILAC_SPEC_SLICE_LEFT_FRAMES  50
#define LILAC_SPEC_SLICE_RIGHT_FRAMES 50

/* Internal helper also exposed via extern for RF probing. */
void model_forward_emit_probe(Model *m, float *spec, int T,
                              const float *g_src, const float *g_tgt,
                              int emit_audio_start, int emit_audio_len,
                              float *emit_out, float *scratch,
                              int left_frames, int right_frames);

void model_forward_emit_probe(Model *m, float *spec, int T,
                              const float *g_src, const float *g_tgt,
                              int emit_audio_start, int emit_audio_len,
                              float *emit_out, float *scratch,
                              int left_frames, int right_frames) {
    const int hop_len = m->hop_length;
    float *z      = scratch;
    float *staged = z + (size_t)m->inter_channels * T;
    enc_q_forward(&m->enc_q, spec, T, z, staged);
    flow_forward(&m->flow, z, T, g_src, 0, staged);
    flow_forward(&m->flow, z, T, g_tgt, 1, staged);

    int emit_start_frame = emit_audio_start / hop_len;
    int emit_end_frame   = (emit_audio_start + emit_audio_len + hop_len - 1) / hop_len;
    int T_slice_start    = emit_start_frame - left_frames;
    int T_slice_end      = emit_end_frame   + right_frames;
    if (T_slice_start < 0) T_slice_start = 0;
    if (T_slice_end > T)   T_slice_end   = T;
    int T_slice = T_slice_end - T_slice_start;

    float *z_slice   = staged;
    float *out_slice = z_slice   + (size_t)m->inter_channels * T_slice;
    float *staged_d  = out_slice + (size_t)T_slice * hop_len;
    for (int c = 0; c < m->inter_channels; c++) {
        memcpy(z_slice + (size_t)c * T_slice,
               z       + (size_t)c * T + T_slice_start,
               (size_t)T_slice * sizeof(float));
    }
    dec_forward(&m->dec, z_slice, T_slice, out_slice, staged_d);
    int emit_offset_in_slice = emit_audio_start - T_slice_start * hop_len;
    memcpy(emit_out, out_slice + emit_offset_in_slice,
           (size_t)emit_audio_len * sizeof(float));
}

void model_forward_emit(Model *m, float *spec, int T,
                        const float *g_src, const float *g_tgt,
                        int emit_audio_start, int emit_audio_len,
                        float *emit_out, float *scratch) {
    const int hop_len = m->hop_length;

    int emit_start_frame = emit_audio_start / hop_len;
    int emit_end_frame   = (emit_audio_start + emit_audio_len + hop_len - 1) / hop_len;

    /* Outer slice — input to enc_q + flow. Their combined RF spans ~40 spec
       frames each side (measured); outside that the emit region is stable. */
    int T_spec_start = emit_start_frame - LILAC_SPEC_SLICE_LEFT_FRAMES;
    int T_spec_end   = emit_end_frame   + LILAC_SPEC_SLICE_RIGHT_FRAMES;
    if (T_spec_start < 0) T_spec_start = 0;
    if (T_spec_end > T)   T_spec_end   = T;
    int T_spec = T_spec_end - T_spec_start;

    /* Inner slice — input to dec, taken from the flow output (already sliced
       by the outer step). Uses a tighter buffer because dec's RF is smaller. */
    int T_slice_start = (emit_start_frame - T_spec_start) - LILAC_DEC_SLICE_LEFT_FRAMES;
    int T_slice_end   = (emit_end_frame   - T_spec_start) + LILAC_DEC_SLICE_RIGHT_FRAMES;
    if (T_slice_start < 0)       T_slice_start = 0;
    if (T_slice_end   > T_spec)  T_slice_end   = T_spec;
    int T_slice = T_slice_end - T_slice_start;

    /* Scratch layout: spec_sliced, z, staged_enc_flow, z_slice, out_slice,
       staged_dec. Each fits within model_scratch_floats(m, T). */
    float *spec_s = scratch;
    float *z      = spec_s + (size_t)m->spec_channels * T_spec;
    float *staged = z      + (size_t)m->inter_channels * T_spec;

    /* Build the sliced spec (row-major per channel). */
    for (int c = 0; c < m->spec_channels; c++) {
        memcpy(spec_s + (size_t)c * T_spec,
               spec   + (size_t)c * T + T_spec_start,
               (size_t)T_spec * sizeof(float));
    }

    enc_q_forward(&m->enc_q, spec_s, T_spec, z, staged);
    flow_forward(&m->flow, z, T_spec, g_src, 0, staged);
    flow_forward(&m->flow, z, T_spec, g_tgt, 1, staged);

    /* Slice z for dec, reusing the staged region (enc_q/flow have finished). */
    float *z_slice   = staged;
    float *out_slice = z_slice   + (size_t)m->inter_channels * T_slice;
    float *staged_d  = out_slice + (size_t)T_slice * hop_len;

    for (int c = 0; c < m->inter_channels; c++) {
        memcpy(z_slice + (size_t)c * T_slice,
               z       + (size_t)c * T_spec + T_slice_start,
               (size_t)T_slice * sizeof(float));
    }

    dec_forward(&m->dec, z_slice, T_slice, out_slice, staged_d);

    int emit_offset_in_slice =
        emit_audio_start - (T_spec_start + T_slice_start) * hop_len;
    memcpy(emit_out,
           out_slice + emit_offset_in_slice,
           (size_t)emit_audio_len * sizeof(float));
}

void model_forward_stream(Model *m, DecStream *ds,
                          float *spec, int T,
                          const float *g_src, const float *g_tgt,
                          int z_feed_end_frame, int n_z_to_feed,
                          float *audio_out, float *scratch) {
    int feed_start = z_feed_end_frame - n_z_to_feed;
    int feed_end   = z_feed_end_frame;

    int T_spec_start = feed_start - LILAC_SPEC_SLICE_LEFT_FRAMES;
    int T_spec_end   = feed_end   + LILAC_SPEC_SLICE_RIGHT_FRAMES;
    if (T_spec_start < 0) T_spec_start = 0;
    if (T_spec_end > T)   T_spec_end   = T;
    int T_spec = T_spec_end - T_spec_start;

    float *spec_s = scratch;
    float *z      = spec_s + (size_t)m->spec_channels  * T_spec;
    float *staged = z      + (size_t)m->inter_channels * T_spec;

    for (int c = 0; c < m->spec_channels; c++) {
        memcpy(spec_s + (size_t)c * T_spec,
               spec   + (size_t)c * T + T_spec_start,
               (size_t)T_spec * sizeof(float));
    }

    double t0 = _ms();
    enc_q_forward(&m->enc_q, spec_s, T_spec, z, staged);
    double t1 = _ms();
    flow_forward(&m->flow, z, T_spec, g_src, 0, staged);
    double t2 = _ms();
    flow_forward(&m->flow, z, T_spec, g_tgt, 1, staged);
    double t3 = _ms();

    /* Extract the n_z_to_feed feed-region frames and hand to dec_stream. */
    int feed_slice_start = feed_start - T_spec_start;
    float *z_feed     = staged;
    float *ds_scratch = z_feed + (size_t)m->inter_channels * n_z_to_feed;
    for (int c = 0; c < m->inter_channels; c++) {
        memcpy(z_feed + (size_t)c * n_z_to_feed,
               z      + (size_t)c * T_spec + feed_slice_start,
               (size_t)n_z_to_feed * sizeof(float));
    }
    dec_stream_forward(ds, z_feed, n_z_to_feed, audio_out, ds_scratch);
    double t4 = _ms();

    static int tick = 0;
    if ((tick++ & 31) == 0) {
        lilac_log("[model_s] T_spec=%d n_z=%d enc_q=%.1f flow_s=%.1f flow_t=%.1f dec_s=%.1f total=%.1f ms\n",
                  T_spec, n_z_to_feed, t1 - t0, t2 - t1, t3 - t2, t4 - t3, t4 - t0);
    }
}

void model_forward(Model *m, float *spec, int T,
                   const float *g_src, const float *g_tgt,
                   float *out_wav, float *scratch) {
    float *z       = scratch;
    float *staged  = z + (size_t)m->inter_channels * T;

    double t0 = _ms();
    enc_q_forward(&m->enc_q, spec, T, z, staged);
    double t1 = _ms();
    flow_forward(&m->flow, z, T, g_src, 0, staged);
    double t2 = _ms();
    flow_forward(&m->flow, z, T, g_tgt, 1, staged);
    double t3 = _ms();
    dec_forward(&m->dec, z, T, out_wav, staged);
    double t4 = _ms();

    static int tick = 0;
    if ((tick++ & 31) == 0) {
        lilac_log("[model] T=%d enc_q=%.1f flow_s=%.1f flow_t=%.1f dec=%.1f total=%.1f ms\n",
                  T, t1 - t0, t2 - t1, t3 - t2, t4 - t3, t4 - t0);
    }
}
