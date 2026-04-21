#define MINIAUDIO_IMPLEMENTATION
#define MA_NO_DECODING
#define MA_NO_ENCODING
#include "miniaudio.h"

#include "audio_io.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_DEVICES 64

typedef struct {
    ma_device_id native;
    int          is_input;
    int          is_default;
    char         name[256];
    int          valid;
} DevSlot;

struct AudioIO {
    ma_context    ctx;
    int           ctx_ready;
    ma_device     device;
    int           device_ready;

    DevSlot       cache[MAX_DEVICES];
    int           cache_count;

    AudioDuplexCallback  user_cb;
    void                *user_data;
    unsigned             block_frames;
};

static void data_callback(ma_device *device, void *output, const void *input,
                          ma_uint32 frame_count) {
    AudioIO *io = (AudioIO *)device->pUserData;
    if (!io || !io->user_cb) return;
    io->user_cb((const float *)input, (float *)output, frame_count, io->user_data);
}

AudioIO *audio_io_create(void) {
    AudioIO *io = (AudioIO *)calloc(1, sizeof(AudioIO));
    if (!io) return NULL;
    if (ma_context_init(NULL, 0, NULL, &io->ctx) != MA_SUCCESS) {
        free(io);
        return NULL;
    }
    io->ctx_ready = 1;
    return io;
}

void audio_io_destroy(AudioIO *io) {
    if (!io) return;
    if (io->device_ready) {
        ma_device_uninit(&io->device);
        io->device_ready = 0;
    }
    if (io->ctx_ready) {
        ma_context_uninit(&io->ctx);
        io->ctx_ready = 0;
    }
    free(io);
}

int audio_io_list_devices(AudioIO *io, AudioDeviceInfo *out, int max_count) {
    if (!io || !io->ctx_ready) return 0;

    ma_device_info *pb_info, *cap_info;
    ma_uint32       pb_count,  cap_count;
    if (ma_context_get_devices(&io->ctx, &pb_info, &pb_count, &cap_info, &cap_count) != MA_SUCCESS)
        return 0;

    io->cache_count = 0;
    for (ma_uint32 i = 0; i < cap_count && io->cache_count < MAX_DEVICES; i++) {
        DevSlot *s = &io->cache[io->cache_count++];
        s->native     = cap_info[i].id;
        s->is_input   = 1;
        s->is_default = cap_info[i].isDefault;
        snprintf(s->name, sizeof(s->name), "%s", cap_info[i].name);
        s->valid = 1;
    }
    for (ma_uint32 i = 0; i < pb_count && io->cache_count < MAX_DEVICES; i++) {
        DevSlot *s = &io->cache[io->cache_count++];
        s->native     = pb_info[i].id;
        s->is_input   = 0;
        s->is_default = pb_info[i].isDefault;
        snprintf(s->name, sizeof(s->name), "%s", pb_info[i].name);
        s->valid = 1;
    }

    if (out) {
        int n = io->cache_count < max_count ? io->cache_count : max_count;
        for (int i = 0; i < n; i++) {
            out[i].id         = i;
            out[i].is_input   = io->cache[i].is_input;
            out[i].is_default = io->cache[i].is_default;
            snprintf(out[i].name, sizeof(out[i].name), "%s", io->cache[i].name);
        }
    }
    return io->cache_count;
}

int audio_io_start(AudioIO *io, int input_id, int output_id,
                   unsigned sample_rate, unsigned block_frames,
                   AudioDuplexCallback cb, void *user) {
    if (!io || !io->ctx_ready || io->device_ready) return -1;

    ma_device_config cfg = ma_device_config_init(ma_device_type_duplex);
    cfg.sampleRate         = sample_rate;
    cfg.periodSizeInFrames = block_frames;
    cfg.capture.format     = ma_format_f32;
    cfg.capture.channels   = 1;
    cfg.playback.format    = ma_format_f32;
    cfg.playback.channels  = 1;
    cfg.dataCallback       = data_callback;
    cfg.pUserData          = io;

    if (input_id >= 0 && input_id < io->cache_count
        && io->cache[input_id].valid && io->cache[input_id].is_input) {
        cfg.capture.pDeviceID = &io->cache[input_id].native;
    }
    if (output_id >= 0 && output_id < io->cache_count
        && io->cache[output_id].valid && !io->cache[output_id].is_input) {
        cfg.playback.pDeviceID = &io->cache[output_id].native;
    }

    io->user_cb      = cb;
    io->user_data    = user;
    io->block_frames = block_frames;

    if (ma_device_init(&io->ctx, &cfg, &io->device) != MA_SUCCESS) return -1;
    if (ma_device_start(&io->device) != MA_SUCCESS) {
        ma_device_uninit(&io->device);
        return -1;
    }
    io->device_ready = 1;
    return 0;
}

void audio_io_stop(AudioIO *io) {
    if (!io || !io->device_ready) return;
    ma_device_stop(&io->device);
    ma_device_uninit(&io->device);
    io->device_ready = 0;
}

int audio_io_is_running(const AudioIO *io) {
    return io && io->device_ready;
}
