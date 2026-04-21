#ifndef LILAC_FLOW_H
#define LILAC_FLOW_H

#include "wn.h"

struct TensorStore;

/* Normalizing flow: 4 ResidualCouplingLayer + Flip pairs. Each coupling layer
   splits x [inter, T] into halves, runs a WN-conditioned network on the first
   half to produce an additive offset for the second half (mean_only=True, so
   no log-scale factor). Flip swaps the halves. Used with reverse=False for
   source-identity removal, reverse=True for target-identity injection. */
#define LILAC_FLOW_N_FLOWS 4

typedef struct {
    int channels;        /* inter_channels = 192 */
    int half_channels;   /* 96 */
    int hidden_channels; /* 192 */
    int kernel_size;     /* 5 */
    int n_layers;        /* 4 */
    int gin_channels;    /* 256 */

    struct {
        const float *pre_w,  *pre_b;    /* Conv1d k=1 [H, half] */
        const float *post_w, *post_b;   /* Conv1d k=1 [half, H]  (mean_only => out=half) */
        WNBlock      wn;
    } layers[LILAC_FLOW_N_FLOWS];
} Flow;

int  flow_init(Flow *f, const struct TensorStore *store);
void flow_free(Flow *f);

/* Forward or inverse flow. x is [channels, T] in-place. g is [gin, 1]. */
void flow_forward(const Flow *f, float *x, int T, const float *g,
                  int reverse, float *scratch);

int  flow_scratch_floats(const Flow *f, int T_max);

#endif
