#include "stream.h"
#include "conv.h"
#include "ops.h"

#include <stdlib.h>
#include <string.h>

int c1d_stream_init(Conv1DStream *s, int C_in, int K, int dilation) {
    memset(s, 0, sizeof(*s));
    s->C          = C_in;
    s->K          = K;
    s->dilation   = dilation;
    s->state_size = (K - 1) * dilation;
    s->pad        = s->state_size / 2;         /* symmetric pad */
    if (s->state_size > 0) {
        s->cache = (float *)calloc((size_t)C_in * s->state_size, sizeof(float));
        if (!s->cache) return -1;
    }
    return 0;
}

void c1d_stream_free(Conv1DStream *s) {
    if (!s) return;
    free(s->cache);
    s->cache = NULL;
}

void c1d_stream_reset(Conv1DStream *s) {
    if (s && s->cache)
        memset(s->cache, 0,
               (size_t)s->C * s->state_size * sizeof(float));
}

void conv1d_stream(Conv1DStream *s,
                   const float *weight_kfirst,
                   int   C_out,
                   const float *bias,
                   const float *new_input, int n_new,
                   float *new_output,
                   float *scratch) {
    /* Special case: K == 1 (pointwise). No state needed; just run conv1d. */
    if (s->K == 1) {
        conv1d_direct(new_input, s->C, n_new, weight_kfirst,
                      C_out, 1, bias, 0, 1, new_output);
        return;
    }

    int S = s->state_size;
    int L = S + n_new;

    /* Layout scratch: buf [C_in, L], out_full [C_out, L]. */
    float *buf      = scratch;
    float *out_full = buf + (size_t)s->C * L;

    /* Build buffer = state ++ new_input, per channel. */
    for (int c = 0; c < s->C; c++) {
        if (S > 0)
            memcpy(buf + (size_t)c * L,
                   s->cache + (size_t)c * S,
                   (size_t)S * sizeof(float));
        memcpy(buf + (size_t)c * L + S,
               new_input + (size_t)c * n_new,
               (size_t)n_new * sizeof(float));
    }

    /* Run full conv on the virtual slice. Output at positions [pad, pad+n_new)
       of the slice has all its RF inputs inside the slice, no zero-pad. */
    conv1d_direct(buf, s->C, L, weight_kfirst,
                  C_out, s->K, bias, s->pad, s->dilation, out_full);

    /* Extract [C_out, n_new] from out_full starting at pad column. */
    for (int c = 0; c < C_out; c++) {
        memcpy(new_output + (size_t)c * n_new,
               out_full   + (size_t)c * L + s->pad,
               (size_t)n_new * sizeof(float));
    }

    /* Update cache: last S columns of buf become new state. */
    if (S > 0) {
        for (int c = 0; c < s->C; c++) {
            memcpy(s->cache + (size_t)c * S,
                   buf      + (size_t)c * L + L - S,
                   (size_t)S * sizeof(float));
        }
    }
}

int ct1d_stream_init(ConvTranspose1DStream *s, int C_in, int K, int u, int pad) {
    memset(s, 0, sizeof(*s));
    s->C_in   = C_in;
    s->K      = K;
    s->u      = u;
    s->pad    = pad;
    /* Enough input-side context so interior_start + n_new*u always stays in
       the full_output range: state_in = ceil(K/u) is the smallest that works
       with a margin of one u. */
    s->state_in = (K + u - 1) / u + 1;
    s->interior_start = K - pad - 1;
    s->cache = (float *)calloc((size_t)C_in * s->state_in, sizeof(float));
    if (!s->cache) return -1;
    return 0;
}

void ct1d_stream_free(ConvTranspose1DStream *s) { if (s) { free(s->cache); s->cache = NULL; } }
void ct1d_stream_reset(ConvTranspose1DStream *s) {
    if (s && s->cache)
        memset(s->cache, 0, (size_t)s->C_in * s->state_in * sizeof(float));
}

void conv_transpose1d_stream(ConvTranspose1DStream *s,
                             const float *weight,
                             int   C_out,
                             const float *bias,
                             const float *new_input, int n_new,
                             float *new_output,
                             float *scratch) {
    int S = s->state_in;
    int L = S + n_new;
    int L_out = (L - 1) * s->u + s->K - 2 * s->pad;
    int emit_n = n_new * s->u;

    float *buf      = scratch;
    float *full_out = buf + (size_t)s->C_in * L;
    float *ct_scr   = full_out + (size_t)C_out * L_out;

    /* Assemble state + new_input. */
    for (int c = 0; c < s->C_in; c++) {
        memcpy(buf + (size_t)c * L,
               s->cache + (size_t)c * S,
               (size_t)S * sizeof(float));
        memcpy(buf + (size_t)c * L + S,
               new_input + (size_t)c * n_new,
               (size_t)n_new * sizeof(float));
    }

    conv_transpose1d(buf, s->C_in, L, weight, C_out, s->K, bias,
                     s->pad, s->u, full_out, ct_scr);

    /* Emit interior slice [interior_start, interior_start + emit_n). */
    for (int c = 0; c < C_out; c++) {
        memcpy(new_output + (size_t)c * emit_n,
               full_out   + (size_t)c * L_out + s->interior_start,
               (size_t)emit_n * sizeof(float));
    }

    /* Update cache: last S columns of buf become new state. */
    for (int c = 0; c < s->C_in; c++) {
        memcpy(s->cache + (size_t)c * S,
               buf      + (size_t)c * L + L - S,
               (size_t)S * sizeof(float));
    }
}

/* ------------------------ DelayLine ------------------------ */
int delay_line_init(DelayLine *d, int C, int size) {
    memset(d, 0, sizeof(*d));
    d->C    = C;
    d->size = size;
    if (size > 0) {
        d->buf = (float *)calloc((size_t)C * size, sizeof(float));
        if (!d->buf) return -1;
    }
    return 0;
}

void delay_line_free(DelayLine *d) { if (d) { free(d->buf); d->buf = NULL; } }
void delay_line_reset(DelayLine *d) {
    if (d && d->buf)
        memset(d->buf, 0, (size_t)d->C * d->size * sizeof(float));
}

void delay_line_step(DelayLine *d, const float *new_input, int n_new,
                     float *popped) {
    int S = d->size;
    if (S == 0) {                          /* pass-through */
        size_t n = (size_t)d->C * n_new * sizeof(float);
        memcpy(popped, new_input, n);
        return;
    }
    /* Pop first n_new of (buf ++ new_input). */
    for (int c = 0; c < d->C; c++) {
        const float *b_row = d->buf + (size_t)c * S;
        const float *n_row = new_input + (size_t)c * n_new;
        float       *p_row = popped + (size_t)c * n_new;
        if (n_new <= S) {
            memcpy(p_row, b_row, (size_t)n_new * sizeof(float));
        } else {
            memcpy(p_row, b_row, (size_t)S * sizeof(float));
            memcpy(p_row + S, n_row, (size_t)(n_new - S) * sizeof(float));
        }
    }
    /* Update buf: keep the last S frames of the extended sequence. */
    int extended = S + n_new;              /* conceptual length of buf++new */
    int keep_from_new = n_new >= S ? S : n_new;
    int keep_from_buf = S - keep_from_new;
    for (int c = 0; c < d->C; c++) {
        float       *b_row = d->buf + (size_t)c * S;
        const float *n_row = new_input + (size_t)c * n_new;
        if (keep_from_buf > 0) {
            /* Move buf[n_new .. n_new + keep_from_buf) to start. */
            memmove(b_row, b_row + (extended - S), (size_t)keep_from_buf * sizeof(float));
        }
        /* Place keep_from_new from new_input at the tail. */
        memcpy(b_row + keep_from_buf,
               n_row + (n_new - keep_from_new),
               (size_t)keep_from_new * sizeof(float));
    }
}

/* ------------------------ Streaming ResBlock ------------------------ */
int rb_stream_init(ResblockStream *s, int C, int K, const int dilations[3]) {
    memset(s, 0, sizeof(*s));
    s->C = C;
    s->K = K;
    s->total_lag = 0;
    for (int step = 0; step < 3; step++) {
        int c1_pad = (K - 1) * dilations[step] / 2;
        int c2_pad = (K - 1) * 1 / 2;
        s->step_lag[step] = c1_pad + c2_pad;
        s->total_lag += s->step_lag[step];
        if (c1d_stream_init(&s->c1[step], C, K, dilations[step]) != 0) return -1;
        if (c1d_stream_init(&s->c2[step], C, K, 1) != 0) return -1;
        if (delay_line_init(&s->xdelay[step], C, s->step_lag[step]) != 0) return -1;
    }
    return 0;
}

void rb_stream_free(ResblockStream *s) {
    if (!s) return;
    for (int i = 0; i < 3; i++) {
        c1d_stream_free(&s->c1[i]);
        c1d_stream_free(&s->c2[i]);
        delay_line_free(&s->xdelay[i]);
    }
}

void rb_stream_reset(ResblockStream *s) {
    if (!s) return;
    for (int i = 0; i < 3; i++) {
        c1d_stream_reset(&s->c1[i]);
        c1d_stream_reset(&s->c2[i]);
        delay_line_reset(&s->xdelay[i]);
    }
}

void resblock_stream(ResblockStream *s,
                     const float *const c1_w_kf[3], const float *const c1_b[3],
                     const float *const c2_w_kf[3], const float *const c2_b[3],
                     const float *new_input, int n_new,
                     float *new_output,
                     float *scratch) {
    size_t xbytes = (size_t)s->C * n_new * sizeof(float);

    float *tmp         = scratch;
    float *y           = tmp       + (size_t)s->C * n_new;
    float *z           = y         + (size_t)s->C * n_new;
    float *x_delayed   = z         + (size_t)s->C * n_new;
    float *x_work      = x_delayed + (size_t)s->C * n_new;
    float *inner_scr   = x_work    + (size_t)s->C * n_new;

    memcpy(x_work, new_input, xbytes);

    for (int step = 0; step < 3; step++) {
        ops_leaky_relu_copy(tmp, x_work, 0.1f, (size_t)s->C * n_new);
        conv1d_stream(&s->c1[step], c1_w_kf[step], s->C, c1_b[step],
                      tmp, n_new, y, inner_scr);
        ops_leaky_relu(y, 0.1f, (size_t)s->C * n_new);
        conv1d_stream(&s->c2[step], c2_w_kf[step], s->C, c2_b[step],
                      y, n_new, z, inner_scr);
        delay_line_step(&s->xdelay[step], x_work, n_new, x_delayed);
        memcpy(x_work, x_delayed, xbytes);
        ops_vec_add(x_work, z, (size_t)s->C * n_new);
    }
    memcpy(new_output, x_work, xbytes);
}
