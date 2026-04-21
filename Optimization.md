# Lilac — Optimization Notes

How the voice-conversion engine hit sub-realtime on CPU. Listed in the order
each change was applied; each entry states the mechanism, the measured
impact, and the files that carry the logic.

## 1. PyTorch → hand-written C engine

The Python prototype was a straight `torch.nn` port of the OpenVoice v2
voice-conversion graph. Rewriting the forward pass in C against OpenBLAS
removed interpreter overhead and per-op CUDA/TH dispatch cost — the engine
is now one linear sequence of `sgemm` / activation calls over preallocated
buffers, no reference counting, no autograd, no Python GIL.

Files: `cpp/model.c`, `cpp/enc_q.c`, `cpp/flow.c`, `cpp/dec.c`, `cpp/wn.c`,
`cpp/gru.c`, `cpp/conv.c`, `cpp/ref_enc.c`.

## 2. Streaming HiFi-GAN generator (`dec_stream`)

The non-streaming decoder in the Python prototype materialises the full
audio output for every z-frame chunk the flow produces. That's wasted work
as the hop-aligned output region is a tiny slice of the window.

`dec_stream` instead keeps per-layer state:

- `Conv1DStream` / `ConvTranspose1DStream` carry their own padding and
  left-context tail so each forward only touches `n_new` frames.
- `ResblockStream` does the same for the dilated residual stacks.
- `DelayLine[max_lag − this_lag]` aligns the three parallel resblocks per
  upsample stage so their outputs add in phase across streaming calls.

Net effect: only `n_new × 256` audio samples are produced per hop instead
of the full `3 × CHUNK` window, and there is no recomputation of frames
already emitted.

Files: `cpp/dec_stream.c`, `cpp/dec_stream.h`.

## 3. Z-frame-aligned hop

`hop_frames = ceil(CHUNK / 256 / K)`. Because the hop is an integer number
of z-frames, `dec_stream` sees a continuous, integer-indexed z-frame stream
across engine calls. No fractional-frame bookkeeping, no re-priming on the
boundary between two hops.

File: `cpp/engine.c`.

## 4. Pool-based 3-way parallel resblock dispatch

Each upsample stage has three resblocks. They are independent for a given
frame so we dispatch two of them to worker threads and run the third on the
main thread, using a fixed-size thread pool (`cpp/pool.c`) with condition
variables. Net speedup on the dec forward: ≈1.8× on 4+ core machines.

Pairs with OpenBLAS's own per-sgemm threading — we keep `openblas_threads`
modest (default 5, clamped to `[4, n_cores]`) so the outer 3-way pool and
the inner BLAS parallelism don't oversubscribe the CPU.

Files: `cpp/pool.c`, `cpp/pool.h`, `cpp/dec.c`, `cpp/dec_stream.c`,
`cpp/engine.c:34-50`.

## 5. Static linking of OpenBLAS + gfortran

OpenBLAS ships as a fat DLL (≈30 MB) with runtime dependencies on
`libgfortran`, `libquadmath`, `libwinmm`, etc. Distributing all of those
alongside `libvc.dll` was brittle.

Switched to `-Wl,-Bstatic -lopenblas -lgfortran -lquadmath -lpthread -Wl,-Bdynamic`
plus `-static-libgcc`. Result: `libvc.dll` is 49 MB but self-contained — no
sibling DLLs need to ship, and Electron/Koffi loads it from any directory.

File: `cpp/Makefile`.

## 6. RNNoise vendored into the same DLL

Denoise + VAD runs inline with the audio callback, no separate process,
no JNI / WASM hop. Compiled directly into `libvc.dll` with
`-DRNN_ENABLE_X86_RTCD -DCPU_INFO_BY_C -DFLOAT_APPROX`, picking up the
SSE4.1 / AVX2 dispatch paths automatically.

Files: `cpp/rnnoise/` (vendored), `cpp/Makefile` `RNNOISE_SRCS` block.

## 7. Two-thread audio pipeline with SPSC rings

The capture callback is 48 kHz device-rate; RNNoise wants 48 kHz 10 ms
frames; VC runs at 22.05 kHz on a much larger hop. Rather than force a
single thread to do all three, the pipeline splits:

```
 audio_cb ──push──► in48_ring (48k, 40 ms blocks)
 rnn_worker  pop──► denoise + VAD + AGC + 48→22 resample
             push─► vc22_ring (22k, 10 ms frames)
 vc_worker   pop──► engine_process_hop (hop-sized batches)
             push─► out48_ring (48k)
 audio_cb   pop──► device output
```

Each ring has its own `CRITICAL_SECTION`; producer/consumer signal via
`SetEvent` rather than polling. Keeps the high-priority audio callback
completely free of VC model work.

File: `cpp/vc.c`.

## 8. VAD gate with hangover

Per-frame RNNoise VAD probability gates the 10 ms block as early as
possible: low-probability frames are zeroed in `rnn_worker_proc` so the
VC side never sees them.

A **hangover counter** (`VAD_HOLD_FRAMES = 8`, ≈80 ms) keeps the gate
open briefly after the last active frame so decaying consonants and
breath tails aren't clipped.

Silence that survives the gate reaches the VC engine as actual zeros,
which `engine_process_hop` processes in the normal way — we don't skip
the call because the streaming decoder's internal state must stay in
sync with wall time.

File: `cpp/vc.c:267-283`.

## 9. Output silence-gate with hangover

`dec_stream` and the flow carry several hundred milliseconds of input
lookahead inside their state. If we blanked the output the moment the
input went silent, the real speech tail trapped in the pipeline would
be discarded.

The output silence-gate therefore has its own per-HOP hangover
(`OUT_SILENT_HOLD_HOPS = 4`, ≈600 ms): the first silent hops keep
emitting so the engine's internal buffers can drain; only once the
counter runs out do we `memset` the block to kill the residual bias
hum of a fully-silent engine.

File: `cpp/vc.c:344-362`.

## 10. Stateful dual-rate resamplers

48 ↔ 22 resampling uses miniaudio's `ma_linear_resampler` as a pair of
long-lived instances (`r48_to_22`, `r22_to_48`). Keeping state across
calls eliminates the discontinuity you get from resetting a resampler
at every block boundary, and the stateful variant's carry buffers are
what make arbitrary-rate 10 ms → variable-output resampling work
cleanly inside the rnn_worker loop.

File: `cpp/vc.c`.

## 11. AGC after VAD gate, before VC

A simple one-pole AGC (`AGC_ATTACK_COEF 0.82` ≈50 ms attack,
`AGC_RELEASE_COEF 0.975` ≈400 ms release, `AGC_MAX_GAIN 10×`) runs on
the 10 ms frame after the VAD gate and before the 48→22 resample.
Target is −20 dBFS by default, user-adjustable via the Electron slider.

Keeping AGC off the silence frames (the VAD has already zeroed them)
means the gain estimate doesn't drift while the speaker is quiet, and
the converted output stays at consistent loudness regardless of mic
distance.

File: `cpp/vc.c` `agc_process`, `vc_worker_proc`.

## 12. Raw (non-denoised) audio path into VC

RNNoise is aggressive enough that feeding its denoised output into the
VC model flattens the converted voice. The current default feeds the
*raw* 10 ms frame into the VC path while still using RNNoise only for
the VAD probability. Keeps the clean Focusrite-style mic characteristics
and produces a more natural converted timbre. `denoise_on` is still a
build-time flag if aggressive denoise is preferred.

File: `cpp/vc.c`, `denoise_on` default.

## 13. Compile flags

- `-O2 -mavx2 -mfma` — AVX2 + FMA throughout the engine, OpenBLAS picks
  matching kernels via its runtime detection.
- `-DFLOAT_APPROX` for rnnoise — uses fast math approximations for
  tanh / sigmoid / exp on the hot path.

## Current budget on 16-core Ryzen

- `hop = 3328 samples` at 22.05 kHz (≈151 ms) with `K = 3`.
- `recent_proc_ms` typically 100–120 ms → real-time factor ≈0.7–0.8.
- End-to-end latency (mic → ear): ≈270–330 ms dominated by the 151 ms
  HOP accumulation + ≈120 ms engine forward + ≈40 ms device blocks.

## Intentional non-optimisations

- Runtime stays **fp32 everywhere**. Quantisation was evaluated and
  rejected — the quality hit on speaker-embedding extraction was more
  audible than any CPU saving was worth.
- No weight pruning. The streaming decoder restructuring was enough to
  hit realtime without touching model shapes.
- Phase-vocoder pitch-shift pre-pass was considered for source/target
  F0 mismatch but not implemented — the added STFT+iSTFT would eat the
  margin we won back in 2+4, and the OpenVoice SE path already handles
  moderate pitch gaps acceptably.
