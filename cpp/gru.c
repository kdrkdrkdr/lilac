#include "gru.h"
#include "ops.h"

#include <math.h>
#include <stddef.h>

void gru_forward_last(const float *x, int T, int input_size, int H,
                      const float *W_ih, const float *W_hh,
                      const float *b_ih, const float *b_hh,
                      float *h, float *scratch) {
    const int G  = 3 * H;
    float   *ih  = scratch;               /* [T, 3H] */
    float   *hh  = scratch + (size_t)T * G;  /* [3H] */

    /* Batched input projection: ih = x @ W_ih^T  (T × 3H). */
    ops_sgemm_nt(x, W_ih, ih, T, G, input_size, 1.0f, 0.0f);
    /* Add b_ih broadcast over T. */
    for (int t = 0; t < T; t++) {
        float *row = ih + (size_t)t * G;
        for (int g = 0; g < G; g++) row[g] += b_ih[g];
    }

    for (int t = 0; t < T; t++) {
        /* hh = h @ W_hh^T  (1 × 3H) */
        ops_sgemm_nt(h, W_hh, hh, 1, G, H, 1.0f, 0.0f);
        for (int g = 0; g < G; g++) hh[g] += b_hh[g];

        const float *ih_t = ih + (size_t)t * G;
        const float *ih_r = ih_t;
        const float *ih_z = ih_t + H;
        const float *ih_n = ih_t + 2 * H;
        const float *hh_r = hh;
        const float *hh_z = hh + H;
        const float *hh_n = hh + 2 * H;

        for (int i = 0; i < H; i++) {
            float r = 1.0f / (1.0f + expf(-(ih_r[i] + hh_r[i])));
            float z = 1.0f / (1.0f + expf(-(ih_z[i] + hh_z[i])));
            float n = tanhf(ih_n[i] + r * hh_n[i]);
            h[i]    = (1.0f - z) * n + z * h[i];
        }
    }
}
