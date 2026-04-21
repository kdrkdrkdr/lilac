# Lilac

Realtime zero-shot voice conversion. One short reference clip of the target
voice (5 s+) is enough — no retraining. Runs on CPU with a hand-written C
engine that streams through a HiFi-GAN generator and keeps RTF < 1 on modest
hardware.

The desktop app is an Electron shell over `libvc.dll`.

- Python reference (`core.py`, `main.py`, `vc/`) — prototype kept for parity
  tests.
- C engine (`cpp/`) — production backend (OpenBLAS + RNNoise + miniaudio,
  streaming decoder, pool-based parallel resblocks).
- Electron UI (`electron/`) — device picker, meters, AGC, VAD gate.

## Build — native engine

Requires MSYS2 (`mingw-w64-x86_64-toolchain`, `mingw-w64-x86_64-gcc-fortran`)
plus a prebuilt OpenBLAS dropped at `cpp/openblas/`.

```sh
cd cpp
mingw32-make libvc.dll
```

Output `cpp/libvc.dll` has OpenBLAS, gfortran, and RNNoise statically linked,
so nothing else needs to ship with it.

You also need `cpp/weights.bin`. Export it from the Python checkpoint via
`cpp/dump_weights.py`.

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

Output lands in `electron/dist/win-unpacked/`:

```
Lilac.exe
libvc.dll
weights.bin
resources/
```

Zip the folder to distribute. All three top-level files must stay together.

## Architecture

- 48 kHz device I/O via miniaudio.
- RNNoise denoise + VAD gate on the capture thread (10 ms frames).
- AGC (target −20 dBFS default) between VAD and VC.
- 48 ↔ 22.05 kHz resampling (stateful linear).
- VC worker runs the streaming HiFi-GAN generator; per-hop output is
  zeroed when the input hop was VAD-silenced (no bias-residue hum).
- Two-thread pipeline (rnn + vc) connected by SPSC ring buffers.

## Links

Source: <https://github.com/kdrkdrkdr/lilac>
