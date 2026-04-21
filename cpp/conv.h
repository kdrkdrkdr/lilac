#ifndef LILAC_CONV_H
#define LILAC_CONV_H

/* Conv1d with stride=1, zero padding, optional dilation.
     input:  [C_in,  T_in]   row-major
     weight: [C_out, C_in, K] row-major
     bias:   [C_out]          (nullable)
     output: [C_out, T_out]   T_out = T_in + 2*pad - dilation*(K-1)

   scratch: caller-provided workspace of at least
   (C_in * K * T_out) float elements. Unused when K == 1 && pad == 0
   (pointwise fast path goes directly through sgemm). */
void conv1d(const float *input, int C_in, int T_in,
            const float *weight, int C_out, int K,
            const float *bias,
            int pad, int dilation,
            float *output, float *scratch);

/* Reorder a conv1d weight tensor from PyTorch layout [C_out, C_in, K] to
   K-first layout [K, C_out, C_in]. The K-first slice at offset k*C_out*C_in
   is a contiguous [C_out, C_in] matrix ready for sgemm — which is exactly
   what conv1d_direct needs. Caller owns `out` (size C_out*C_in*K floats). */
void conv_prepack_weight(const float *w, int C_out, int C_in, int K, float *out);

/* Direct Conv1d via K BLAS calls, no im2col materialization. Weight must be
   in the K-first layout produced by conv_prepack_weight. Expected to be a
   big win when K*C_out*C_in*T_out memory traffic would dominate. */
void conv1d_direct(const float *input, int C_in, int T_in,
                   const float *weight_k_first,   /* [K, C_out, C_in] */
                   int C_out, int K,
                   const float *bias,
                   int pad, int dilation,
                   float *output);

/* Conv2d, no dilation.
     input:  [C_in,  H_in, W_in]          row-major (channels-first, no batch)
     weight: [C_out, C_in, kH, kW]        row-major
     bias:   [C_out]                      nullable
     output: [C_out, H_out, W_out]
        H_out = (H_in + 2*padH - kH)/strideH + 1
        W_out = (W_in + 2*padW - kW)/strideW + 1
   scratch: at least C_in*kH*kW * H_out*W_out floats. */
void conv2d(const float *input, int C_in, int H_in, int W_in,
            const float *weight, int C_out, int kH, int kW,
            const float *bias,
            int padH, int padW, int strideH, int strideW,
            float *output, float *scratch);

/* ConvTranspose1d. PyTorch weight layout: [C_in, C_out, K].
     input : [C_in,  T_in]
     weight: [C_in,  C_out, K]        row-major
     bias  : [C_out]                  nullable
     output: [C_out, T_out]  where T_out = (T_in - 1)*stride - 2*pad + K
   scratch: at least (C_out * K * T_in) floats. */
void conv_transpose1d(const float *input, int C_in, int T_in,
                      const float *weight, int C_out, int K,
                      const float *bias,
                      int pad, int stride,
                      float *output, float *scratch);

#endif
