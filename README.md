# Lilac

Realtime zero-shot voice conversion. One short reference clip of the target
voice (5 s+) is enough — no retraining. Runs on CPU with a hand-written C
engine that streams through a HiFi-GAN generator and keeps RTF < 1 on modest
hardware.

<p align="center">
  <img src="assets/program.png" alt="Lilac desktop app" width="420">
</p>

The desktop app is an Electron shell over `libvc.dll`.

- C engine (`cpp/`) — production backend (OpenBLAS + RNNoise + miniaudio,
  streaming decoder, pool-based parallel resblocks).
- Electron UI (`electron/`) — device picker, meters, AGC slider, VAD gate.

## Download

Prebuilt Windows zip on the
[Releases page](https://github.com/kdrkdrkdr/lilac/releases). Unzip and
run `Lilac.exe` — `libvc.dll`, `weights.bin`, and `tsu_10s.wav` sit
alongside the executable and must stay in the same folder.

## Build — native engine

Requires MSYS2 (`mingw-w64-x86_64-toolchain`, `mingw-w64-x86_64-gcc-fortran`)
plus a prebuilt OpenBLAS tree at `cpp/openblas/` and `cpp/weights.bin`.

```sh
cd cpp
mingw32-make libvc.dll
```

Output `cpp/libvc.dll` has OpenBLAS, gfortran, and RNNoise statically linked
— no sibling DLLs need to ship.

## Run — Electron app (dev)

```sh
cd electron
npm install
npm start
```

## Package

```sh
cd electron
npm run dist
```

Output lands in `electron/dist/win-unpacked/` with `Lilac.exe`, `libvc.dll`,
`weights.bin`, and `tsu_10s.wav` at the top level; zip the folder to
distribute.

## Architecture

- 48 kHz device I/O via miniaudio.
- RNNoise runs on the capture thread (10 ms frames) for VAD; raw audio
  feeds the VC path, with an 80 ms VAD hangover to preserve speech tails.
- AGC (target −20 dBFS default) between VAD and VC.
- 48 ↔ 22.05 kHz stateful resampling.
- VC worker runs the streaming HiFi-GAN generator; output has a 600 ms
  silence-gate hangover so the engine's internal lookahead can flush
  before bias is blanked.
- Two-thread pipeline (rnn + vc) connected by SPSC ring buffers.

See [Optimization.md](Optimization.md) for the full list of perf measures.

## Links

Source: <https://github.com/kdrkdrkdr/lilac>
