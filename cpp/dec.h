#ifndef LILAC_DEC_H
#define LILAC_DEC_H

struct TensorStore;

/* HiFi-GAN v1 style generator. Matches vc/models.py's Generator.
     * conv_pre  : Conv1d k=7
     * cond      : Conv1d k=1   (unused when zero_g=True — we just skip it)
     * 4 upsample stages, rates [8, 8, 2, 2], kernels [16, 16, 4, 4]
         ups[i]      : ConvTranspose1d
         for each of 3 resblocks (kernel 3/7/11):
             ResBlock1: 3 dilated conv (dil 1/3/5) + 3 non-dilated conv, all k=3
         sum(resblock outputs) / 3
     * conv_post : Conv1d k=7, bias=False
     * tanh on output */
#define LILAC_DEC_UPSAMPLES      4
#define LILAC_DEC_KERNELS_PER_UP 3
#define LILAC_DEC_RESBLOCK_CONVS 3

typedef struct {
    int initial_channel;       /* inter_channels = 192 */
    int upsample_initial;      /* 512 */

    int upsample_rates[LILAC_DEC_UPSAMPLES];
    int upsample_kernels[LILAC_DEC_UPSAMPLES];
    int resblock_kernels[LILAC_DEC_KERNELS_PER_UP];       /* {3,7,11} */
    int resblock_dilations[LILAC_DEC_KERNELS_PER_UP][LILAC_DEC_RESBLOCK_CONVS]; /* {1,3,5} */

    const float *conv_pre_w, *conv_pre_b;
    const float *conv_post_w;  /* bias=False */
    const float *cond_w,     *cond_b;   /* dec.cond — applied even when g=0 (bias broadcast) */

    const float *ups_w[LILAC_DEC_UPSAMPLES];
    const float *ups_b[LILAC_DEC_UPSAMPLES];

    /* resblocks flattened: [stage_i][kernel_j] → index stage_i*3 + kernel_j.
       Pointers into the TensorStore for the original [C_out, C_in, K] weights
       plus owned K-first prepacked copies for conv1d_direct. */
    const float *rb_c1_w[LILAC_DEC_UPSAMPLES * LILAC_DEC_KERNELS_PER_UP][LILAC_DEC_RESBLOCK_CONVS];
    const float *rb_c1_b[LILAC_DEC_UPSAMPLES * LILAC_DEC_KERNELS_PER_UP][LILAC_DEC_RESBLOCK_CONVS];
    const float *rb_c2_w[LILAC_DEC_UPSAMPLES * LILAC_DEC_KERNELS_PER_UP][LILAC_DEC_RESBLOCK_CONVS];
    const float *rb_c2_b[LILAC_DEC_UPSAMPLES * LILAC_DEC_KERNELS_PER_UP][LILAC_DEC_RESBLOCK_CONVS];

    float *rb_c1_w_kfirst[LILAC_DEC_UPSAMPLES * LILAC_DEC_KERNELS_PER_UP][LILAC_DEC_RESBLOCK_CONVS];
    float *rb_c2_w_kfirst[LILAC_DEC_UPSAMPLES * LILAC_DEC_KERNELS_PER_UP][LILAC_DEC_RESBLOCK_CONVS];
} Dec;

int  dec_init(Dec *d, const struct TensorStore *store);
void dec_free(Dec *d);

/* Input z: [initial_channel, T]. Output audio: [T * total_upsample] (mono). */
void dec_forward(const Dec *d, const float *z, int T, float *out_audio, float *scratch);

int  dec_scratch_floats(const Dec *d, int T_max);

#endif
