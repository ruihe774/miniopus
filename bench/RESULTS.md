# miniopus vs libopus 1.6.1 encoder benchmark

- miniopus main @ `7eae0be`; libopus `v1.6.1` (22244de5), submodule `third_party/opus`
- Host: Intel(R) Xeon(R) Processor @ 2.10GHz, 4 vCPU, Linux 6.18 (cloud VM, shared; expect a few % noise)
- Runtimes: Deno deno 2.9.7 (V8), Bun 1.4.2 (JSC), Wasmtime crate 48.0.7 (Cranelift), Wasmer crate 7.5 (Cranelift; embedded via Rust, default backend)
- Builds: miniopus wasm2 = `+simd128`, wasm3 = `+simd128,+relaxed-simd,+extended-const,+tail-call` (`scripts/build.sh`, release)
  - libopus scalar: `OPUS_DISABLE_INTRINSICS`, baseline x86-64
  - libopus AVX: SSE4.1 intrinsics presumed, AVX2 kernels disabled (`OPUS_X86_MAY_HAVE_AVX2=OFF`), whole library `-mavx`
  - libopus AVX2: `OPUS_X86_PRESUME_AVX2` (AVX2+FMA kernels), whole library `-mavx2 -mfma`
  - all libopus builds Release, `-ffp-contract=off`; CELT reference without tonality analysis (as in minicelt's tests)
- Workload: `test/audio/*.wav` concatenated (12.8 s, 639 x 20 ms frames), mono, `opus_encode_float`. SILK: 16 kHz, RESTRICTED_SILK, wideband, 24 kbit/s VBR, DTX/FEC off. CELT: 48 kHz (ffmpeg-resampled), RESTRICTED_CELT, 64 kbit/s constrained VBR, DTX off.
- Method: each pass encodes the whole file with a fresh encoder (init untimed); 3 warm-up passes, then 15 timed passes; **median** pass time reported (min in the second table). Wasm hosts copy each frame into linear memory inside the timed loop. Runs are sequential, one process per cell.
- Output size in bytes is identical across all builds/runtimes for a given codec/complexity (checked), so every cell does the same work.

Values are **ms per pass** (lower is better); in parentheses, speed as multiple of real time (12.78 s of audio / pass time).


## SILK

| complexity | libopus scalar | libopus AVX | libopus AVX2 | Deno (V8) wasm2 | Deno (V8) wasm3 | Bun (JSC) wasm2 | Bun (JSC) wasm3 | Wasmtime wasm2 | Wasmtime wasm3 | Wasmer wasm2 | Wasmer wasm3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 32.5 (394x) | 31.0 (412x) | 25.2 (507x) | 47.9 (267x) | 48.7 (262x) | 45.3 (282x) | 44.9 (285x) | 42.7 (300x) | 47.5 (269x) | 52.6 (243x) | n/a |
| 5 | 60.8 (210x) | 70.4 (182x) | 71.0 (180x) | 79.4 (161x) | 95.8 (133x) | 75.7 (169x) | 83.0 (154x) | 93.9 (136x) | 93.7 (136x) | 98.3 (130x) | n/a |
| 9 | 140.1 (91x) | 132.0 (97x) | 86.0 (149x) | 140.2 (91x) | 131.5 (97x) | 105.0 (122x) | 106.8 (120x) | 112.8 (113x) | 127.9 (100x) | 133.1 (96x) | n/a |

Minimum pass time (ms):

| complexity | libopus scalar | libopus AVX | libopus AVX2 | Deno (V8) wasm2 | Deno (V8) wasm3 | Bun (JSC) wasm2 | Bun (JSC) wasm3 | Wasmtime wasm2 | Wasmtime wasm3 | Wasmer wasm2 | Wasmer wasm3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 31.3 | 29.6 | 20.0 | 47.1 | 46.7 | 44.6 | 44.6 | 37.7 | 46.6 | 49.0 | n/a |
| 5 | 58.4 | 69.0 | 68.5 | 76.1 | 93.9 | 72.7 | 76.9 | 93.5 | 92.8 | 89.1 | n/a |
| 9 | 138.7 | 111.7 | 84.9 | 129.9 | 128.2 | 100.3 | 100.9 | 104.7 | 126.8 | 130.8 | n/a |

Relative to libopus scalar (median; >1 = slower than scalar C):

| complexity | libopus AVX | libopus AVX2 | Deno (V8) wasm2 | Deno (V8) wasm3 | Bun (JSC) wasm2 | Bun (JSC) wasm3 | Wasmtime wasm2 | Wasmtime wasm3 | Wasmer wasm2 | Wasmer wasm3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.95 | 0.78 | 1.47 | 1.50 | 1.39 | 1.38 | 1.31 | 1.46 | 1.62 | n/a |
| 5 | 1.16 | 1.17 | 1.31 | 1.58 | 1.25 | 1.37 | 1.55 | 1.54 | 1.62 | n/a |
| 9 | 0.94 | 0.61 | 1.00 | 0.94 | 0.75 | 0.76 | 0.80 | 0.91 | 0.95 | n/a |

## CELT

| complexity | libopus scalar | libopus AVX | libopus AVX2 | Deno (V8) wasm2 | Deno (V8) wasm3 | Bun (JSC) wasm2 | Bun (JSC) wasm3 | Wasmtime wasm2 | Wasmtime wasm3 | Wasmer wasm2 | Wasmer wasm3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 24.5 (521x) | 27.0 (474x) | 25.4 (503x) | 31.0 (413x) | 25.3 (505x) | 24.7 (517x) | 27.9 (458x) | 24.4 (525x) | 29.5 (433x) | 30.9 (413x) | 31.0 (413x) |
| 5 | 54.7 (234x) | 37.7 (339x) | 37.8 (338x) | 44.9 (285x) | 44.1 (290x) | 41.5 (308x) | 43.4 (294x) | 43.6 (293x) | 44.3 (289x) | 35.9 (356x) | 48.4 (264x) |
| 9 | 55.6 (230x) | 31.4 (406x) | 39.1 (327x) | 46.0 (278x) | 44.2 (289x) | 42.3 (302x) | 35.1 (364x) | 46.5 (275x) | 46.0 (278x) | 47.0 (272x) | 49.4 (259x) |

Minimum pass time (ms):

| complexity | libopus scalar | libopus AVX | libopus AVX2 | Deno (V8) wasm2 | Deno (V8) wasm3 | Bun (JSC) wasm2 | Bun (JSC) wasm3 | Wasmtime wasm2 | Wasmtime wasm3 | Wasmer wasm2 | Wasmer wasm3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 24.4 | 26.1 | 25.1 | 29.8 | 23.5 | 24.3 | 22.3 | 23.7 | 29.4 | 29.1 | 26.9 |
| 5 | 54.3 | 33.1 | 37.6 | 44.6 | 42.8 | 41.3 | 42.0 | 42.0 | 43.3 | 35.3 | 47.2 |
| 9 | 47.0 | 31.3 | 38.4 | 45.9 | 39.9 | 41.3 | 34.3 | 45.5 | 45.8 | 45.1 | 48.0 |

Relative to libopus scalar (median; >1 = slower than scalar C):

| complexity | libopus AVX | libopus AVX2 | Deno (V8) wasm2 | Deno (V8) wasm3 | Bun (JSC) wasm2 | Bun (JSC) wasm3 | Wasmtime wasm2 | Wasmtime wasm3 | Wasmer wasm2 | Wasmer wasm3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.10 | 1.04 | 1.26 | 1.03 | 1.01 | 1.14 | 0.99 | 1.20 | 1.26 | 1.26 |
| 5 | 0.69 | 0.69 | 0.82 | 0.81 | 0.76 | 0.79 | 0.80 | 0.81 | 0.66 | 0.89 |
| 9 | 0.57 | 0.70 | 0.83 | 0.80 | 0.76 | 0.63 | 0.84 | 0.83 | 0.85 | 0.89 |

## Notes

- **Noise:** this is a shared cloud VM; median and min differ by up to ~25% in some cells (e.g. CELT complexity 9 libopus scalar: 55.6 vs 47.0 ms), so differences under ~15% should not be read as real. A single run per cell, not repeated across sessions.

- **Wasmer + SILK wasm3: n/a.** Wasmer's Cranelift backend rejects the module: `proposed tail-call operator ReturnCall` (unsupported). The CELT wasm3 build compiles and runs on Wasmer.
- Wasmtime and Wasmer are driven through their Rust embedding APIs (the CLIs cannot call exports with linear-memory I/O); typed-call overhead per frame is one dynamic `Func::call`/`Function::call` with a pre-resolved handle.
- Wasmer was not built with its LLVM backend (not installed); results reflect Cranelift.
- "AVX" is not a distinct upstream configuration: libopus x86 has SSE/SSE4.1 and AVX2 kernels only, so the AVX row is the SSE4.1-intrinsics build compiled with VEX encoding (`-mavx`), without AVX2 kernels.
- Reproduce: `bench/gen-pcm.sh; scripts/build.sh all all release; bench/build-native.sh;` build `bench/host` with `--features wasmtime|wasmer` (pass `--target x86_64-unknown-linux-gnu`, since the repo `.cargo/config.toml` defaults to wasm32), then `bench/run-all.sh 15`.
