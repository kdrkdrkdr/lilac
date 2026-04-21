#ifndef LILAC_AUDIO_IO_H
#define LILAC_AUDIO_IO_H

#include <stdint.h>

/* Thin wrapper around miniaudio for the lilac engine:
     * enumerate input/output devices (for the UI)
     * start/stop a duplex stream bound to a user-supplied callback
     * owns its own miniaudio context, device, and cached id table. */

typedef struct {
    int  id;                /* index into internal cache; pass to audio_io_start */
    int  is_input;          /* 1 = capture, 0 = playback */
    int  is_default;
    char name[256];
} AudioDeviceInfo;

typedef void (*AudioDuplexCallback)(const float *in_frames,
                                    float       *out_frames,
                                    unsigned     n_frames,
                                    void        *user);

typedef struct AudioIO AudioIO;

AudioIO *audio_io_create(void);
void     audio_io_destroy(AudioIO *io);

/* Refresh and return device list. out may be NULL to just query count via
   return value; otherwise writes up to max_count entries. */
int      audio_io_list_devices(AudioIO *io, AudioDeviceInfo *out, int max_count);

/* Start a duplex stream. input_id / output_id come from audio_io_list_devices;
   pass -1 to use the system default. sample_rate and block_frames configure
   the stream. Returns 0 on success. */
int      audio_io_start(AudioIO *io, int input_id, int output_id,
                        unsigned sample_rate, unsigned block_frames,
                        AudioDuplexCallback cb, void *user);

void     audio_io_stop(AudioIO *io);

int      audio_io_is_running(const AudioIO *io);

#endif
