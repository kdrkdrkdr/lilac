#ifndef LILAC_VC_H
#define LILAC_VC_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _WIN32
#  ifdef LILAC_BUILD_DLL
#    define LILAC_API __declspec(dllexport)
#  else
#    define LILAC_API __declspec(dllimport)
#  endif
#else
#  define LILAC_API
#endif

/* Opaque engine handle. Owns: weights, STFT plan, model, miniaudio context. */
typedef struct LilacVC LilacVC;

typedef struct {
    int  id;
    int  is_input;     /* 1 = capture, 0 = playback */
    int  is_default;
    char name[256];
} LilacDevice;

typedef struct {
    float input_rms;
    float output_rms;
    int   dropped_frames;
    float avg_process_ms;      /* cumulative since start */
    float recent_proc_ms;      /* EMA over recent blocks */
    int   is_running;
    float last_vad;            /* RNNoise VAD probability of most recent 10 ms frame */
} LilacStats;

/* Create engine. target_wav is mono fp32 at 22050 Hz; len is sample count.
   K is the sub-chunk factor (CHUNK / HOP); typical values 1/2/4/8. */
LILAC_API LilacVC *lilac_create(const char *weights_path,
                                const float *target_wav, int target_len,
                                int K);
LILAC_API void     lilac_destroy(LilacVC *vc);

/* Refresh and return device list. Pass out=NULL to just query count. */
LILAC_API int      lilac_list_devices(LilacVC *vc, LilacDevice *out, int max_count);

/* Standalone device enumeration — works without a VC handle, so the UI can
   populate dropdowns before the engine is initialised. */
LILAC_API int      lilac_enum_devices(LilacDevice *out, int max_count);

/* Start the audio pipeline. Pass -1 for either id to use the system default. */
LILAC_API int      lilac_start(LilacVC *vc, int input_id, int output_id);
LILAC_API void     lilac_stop(LilacVC *vc);

LILAC_API int      lilac_sample_rate(const LilacVC *vc);
LILAC_API int      lilac_hop_samples(const LilacVC *vc);

/* Swap the target speaker at runtime. Safe to call while running. */
LILAC_API int      lilac_set_target(LilacVC *vc, const float *wav, int len);

/* Reset source-speaker cache (re-learn on next full window). */
LILAC_API void     lilac_reset_source(LilacVC *vc);

/* Copy stats into *out. Thread-safe (reads are atomic-ish; tiny tears OK for UI). */
LILAC_API void     lilac_get_stats(const LilacVC *vc, LilacStats *out);

/* VAD gate threshold (RNNoise probability in [0,1]). Default 0.88. Frames with
   probability below this threshold are zeroed before hitting the VC engine —
   so "silent" input produces silent output, at modest CPU savings. */
LILAC_API void     lilac_set_vad_threshold(LilacVC *vc, float threshold);
LILAC_API float    lilac_get_vad_threshold(const LilacVC *vc);

/* When enabled (default), rnnoise's denoised output feeds the VC; when
   disabled, raw mic audio feeds the VC and rnnoise still runs to produce
   the VAD gate signal. */
LILAC_API void     lilac_set_denoise(LilacVC *vc, int enabled);
LILAC_API int      lilac_get_denoise(const LilacVC *vc);

/* Auto gain control on the 10 ms frame after VAD gating, before the VC
   model sees it. Default enabled, target = -20 dBFS. */
LILAC_API void     lilac_set_agc(LilacVC *vc, int enabled);
LILAC_API int      lilac_get_agc(const LilacVC *vc);
LILAC_API void     lilac_set_agc_target_db(LilacVC *vc, float db);
LILAC_API float    lilac_get_agc_target_db(const LilacVC *vc);

#ifdef __cplusplus
}
#endif

#endif
