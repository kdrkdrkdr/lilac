#ifndef LILAC_ENGINE_H
#define LILAC_ENGINE_H

#include "dec_stream.h"
#include "model.h"
#include "stft.h"
#include "tensor.h"

/* Streaming VC engine tying together: weights store, STFT, model, sliding
   window, DC-offset boundary smoothing, cached source/target speaker embeds.

   Audio contract: fp32 mono at 22050 Hz. Each call to engine_process_hop
   consumes `hop` samples and emits `hop` samples. During warmup (window not
   yet full), output is silence. Once filled, each emitted block carries the
   VC output corresponding to samples CHUNK + (K-1)*HOP behind the window's
   most recent sample — i.e., max latency ≈ CHUNK + HOP per call. */
typedef struct {
    TensorStore store;
    Model       model;
    StftPlan    stft;

    int chunk;         /* 9984 */
    int window;        /* 3 * chunk */
    int hop;           /* hop_frames * 256  (rounded up from chunk/K so hop is 256-aligned) */
    int hop_frames;    /* hop / 256 — z-frames consumed per call by dec_stream */
    int emit_offset;   /* 2 * chunk - hop */
    int fade_len;      /* 5 ms @ 22050 Hz */

    DecStream ds;              /* streaming HiFi-GAN state */
    int       priming_hops_done;
    int       priming_hops_needed;

    float *window_buf;     /* [window] rolling */
    int    window_filled;  /* samples accumulated so far (capped at window) */

    float *fade;           /* [fade_len] cos² envelope */
    int    has_prev_last;
    float  prev_last;

    float *spec;           /* [half_bins, frames_max] */
    float *wav_out;        /* [window] */
    float *stft_scratch;
    float *model_scratch;

    float *target_se;      /* [gin, 1] — extracted once at init */
    float *source_se;      /* [gin, 1] — cached on first full window */
    int    source_se_ready;
} Engine;

int           engine_init(Engine *e, const char *weights_path,
                          const float *target_wav, int target_len, int K);
void          engine_free(Engine *e);
void          engine_reset_source(Engine *e);
const float  *engine_process_hop(Engine *e, const float *input_hop);

#endif
