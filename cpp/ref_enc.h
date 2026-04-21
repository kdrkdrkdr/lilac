#ifndef LILAC_REF_ENC_H
#define LILAC_REF_ENC_H

struct TensorStore;

/* Reference encoder: spec [spec_ch, T] -> speaker embedding [gin, 1].
   Layout matches vc/models.py's ReferenceEncoder:
     * LayerNorm over the last dim (freq)
     * 6 × (Conv2d k=3 s=2 p=1 + ReLU), channels 1→32→32→64→64→128→128
     * Flatten to GRU input [T', 128 * freq_after] and run 1-layer GRU (H=128)
     * Linear proj from 128 → gin_channels (256) */
typedef struct {
    int spec_channels;   /* 513 */
    int gin_channels;    /* 256 */

    const float *ln_w, *ln_b;                 /* [spec_channels] each */
    const float *conv_w[6], *conv_b[6];       /* ref_enc.convs.i weights */
    const float *gru_W_ih, *gru_W_hh;
    const float *gru_b_ih, *gru_b_hh;
    const float *proj_w, *proj_b;             /* [gin, 128], [gin] */

    int gru_hidden;      /* 128 */
    int gru_input;       /* 128 * freq_reduced */
    int freq_reduced;    /* freq / 2^6 after 6 stride-2 convs */
} RefEnc;

int  ref_enc_init(RefEnc *r, const struct TensorStore *store,
                  int spec_channels, int gin_channels);

/* spec [spec_channels, T] (will be temporarily mutated by layernorm);
   out is [gin, 1]. scratch must satisfy ref_enc_scratch_floats(). */
void ref_enc_forward(const RefEnc *r, float *spec, int T,
                     float *out, float *scratch);

int  ref_enc_scratch_floats(const RefEnc *r, int T_max);

#endif
