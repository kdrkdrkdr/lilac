#ifndef LILAC_GRU_H
#define LILAC_GRU_H

/* Single-layer batch_first GRU, matches PyTorch nn.GRU update equations:
     r = sigmoid(W_ir x + b_ir + W_hr h + b_hr)
     z = sigmoid(W_iz x + b_iz + W_hz h + b_hz)
     n = tanh(W_in x + b_in + r * (W_hn h + b_hn))
     h = (1 - z) * n + z * h
   Weights are stored in PyTorch's concat-gate order [r, z, n]:
     W_ih: [3H, input_size]
     W_hh: [3H, H]
     b_ih, b_hh: [3H]

   x      : [T, input_size]
   h      : [H]  — initial hidden state (caller zero-inits for default); overwritten
                   with final hidden state on return.
   scratch: at least (T * 3H + 3H) floats. */
void gru_forward_last(const float *x, int T, int input_size, int H,
                      const float *W_ih, const float *W_hh,
                      const float *b_ih, const float *b_hh,
                      float *h, float *scratch);

#endif
