#ifndef LILAC_MODEL_H
#define LILAC_MODEL_H

#include "dec.h"
#include "dec_stream.h"
#include "enc_q.h"
#include "flow.h"
#include "ref_enc.h"
#include "stft.h"

struct TensorStore;

/* Full lilac VC pipeline — equivalent of vc/models.py SynthesizerTrn.forward
   with tau=0 and zero_g=True baked in:
       g_src = ref_enc(spec)    (optional: caller-provided cached embedding)
       z     = enc_q(spec, g=0)           // tau=0 -> z = m
       z_p   = flow(z,   g=g_src)         // forward (source removal)
       z_hat = flow(z_p, g=g_tgt, reverse)// target injection
       wav   = dec(z_hat, g=0) */
typedef struct {
    EncQ    enc_q;
    Flow    flow;
    Dec     dec;
    RefEnc  ref_enc;
    int     spec_channels;     /* 513 */
    int     inter_channels;    /* 192 */
    int     gin_channels;      /* 256 */
    int     hop_length;        /* 256 */
} Model;

int  model_init(Model *m, const struct TensorStore *store);
void model_free(Model *m);

/* Extract a speaker embedding from a spectrogram.
     spec : [spec_ch, T] (mutated during layernorm)
     out_se : [gin, 1] */
void model_extract_se(Model *m, float *spec, int T, float *out_se, float *scratch);

/* Full VC forward.
     spec    : [spec_ch, T] — mutated
     g_src   : [gin, 1] cached source SE  (caller extracts once)
     g_tgt   : [gin, 1] target SE         (caller extracts once)
     out_wav : [T * hop_length] fp32 mono
     scratch : see model_scratch_floats() */
void model_forward(Model *m, float *spec, int T,
                   const float *g_src, const float *g_tgt,
                   float *out_wav, float *scratch);

/* Emit-region forward. enc_q + flow run on full T (their receptive fields span
   the whole window), but dec runs only on a tight z-slice around the emit
   region. Output is bit-exact to model_forward inside the emit region provided
   the slice leaves enough RF buffer on both sides of the emit range.

   emit_audio_start : absolute sample offset (in the full T*hop_length output)
   emit_audio_len   : samples to emit into emit_out (typically engine.hop)
   emit_out         : [emit_audio_len] — receives the emit-region audio
   scratch          : same size as model_scratch_floats(m, T) */
void model_forward_emit(Model *m, float *spec, int T,
                        const float *g_src, const float *g_tgt,
                        int emit_audio_start, int emit_audio_len,
                        float *emit_out, float *scratch);

/* Streaming forward. enc_q + flow run on a spec slice centered on the feed
   region (RF-padded); the resulting z frames at window positions
   [z_feed_end_frame - n_z_to_feed, z_feed_end_frame) are fed incrementally
   to `ds` (the caller's persistent DecStream state). Emits n_z_to_feed*256
   audio samples into `audio_out` — warmup/lag-region output is produced on
   first few calls and should be discarded by the caller until
   ds->cumulative_lag_audio samples have been produced. */
void model_forward_stream(Model *m, DecStream *ds,
                          float *spec, int T,
                          const float *g_src, const float *g_tgt,
                          int z_feed_end_frame, int n_z_to_feed,
                          float *audio_out, float *scratch);

int  model_scratch_floats(const Model *m, int T_max);

#endif
