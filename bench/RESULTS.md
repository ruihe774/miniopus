# miniopus vs libopus 1.6.1 encoder benchmark

- miniopus main @ `6324a3a` (release builds post-processed by `wasm-opt` version_133: `-O2 -Os -c -lmu`, stripped); libopus `v1.6.1` (22244de5), submodule `third_party/opus`
- Host: Intel(R) Core(TM) i5-14600KF (desktop, 20 threads), Linux 7.0, otherwise idle
- Runtimes: Deno 2.9.7 (V8), Bun 1.4.4 (JSC). Wasmtime and Wasmer are not in this run (the arms are still in the harness: `RUNTIMES="deno bun"` selects the wasm arms); their results on a different host are in `8ebc26e`.
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

| complexity | libopus scalar | libopus AVX | libopus AVX2 | Deno (V8) wasm2 | Deno (V8) wasm3 | Bun (JSC) wasm2 | Bun (JSC) wasm3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 20.0 (640x) | 19.6 (652x) | 15.0 (851x) | 29.0 (441x) | 28.3 (451x) | 28.0 (457x) | 27.7 (461x) |
| 5 | 46.2 (277x) | 42.9 (298x) | 43.1 (296x) | 61.5 (208x) | 60.6 (211x) | 57.9 (221x) | 57.3 (223x) |
| 9 | 86.2 (148x) | 80.7 (158x) | 54.0 (237x) | 85.9 (149x) | 85.0 (150x) | 78.6 (163x) | 79.7 (160x) |

Minimum pass time (ms):

| complexity | libopus scalar | libopus AVX | libopus AVX2 | Deno (V8) wasm2 | Deno (V8) wasm3 | Bun (JSC) wasm2 | Bun (JSC) wasm3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 19.9 | 19.5 | 15.0 | 28.9 | 28.2 | 27.9 | 27.7 |
| 5 | 46.0 | 42.9 | 43.0 | 61.5 | 60.4 | 57.7 | 57.2 |
| 9 | 86.1 | 80.7 | 53.9 | 85.7 | 84.8 | 78.5 | 79.4 |

Relative to libopus scalar (median; >1 = slower than scalar C):

| complexity | libopus AVX | libopus AVX2 | Deno (V8) wasm2 | Deno (V8) wasm3 | Bun (JSC) wasm2 | Bun (JSC) wasm3 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0.98 | 0.75 | 1.45 | 1.42 | 1.40 | 1.39 |
| 5 | 0.93 | 0.93 | 1.33 | 1.31 | 1.25 | 1.24 |
| 9 | 0.94 | 0.63 | 1.00 | 0.99 | 0.91 | 0.92 |

## CELT

| complexity | libopus scalar | libopus AVX | libopus AVX2 | Deno (V8) wasm2 | Deno (V8) wasm3 | Bun (JSC) wasm2 | Bun (JSC) wasm3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 20.1 (637x) | 15.6 (818x) | 15.6 (817x) | 18.7 (683x) | 18.8 (682x) | 17.6 (726x) | 17.7 (722x) |
| 5 | 34.3 (373x) | 24.3 (525x) | 23.5 (544x) | 28.0 (457x) | 27.4 (467x) | 25.9 (494x) | 25.9 (493x) |
| 9 | 34.7 (369x) | 24.7 (517x) | 23.9 (534x) | 28.5 (448x) | 28.3 (452x) | 26.8 (476x) | 26.6 (481x) |

Minimum pass time (ms):

| complexity | libopus scalar | libopus AVX | libopus AVX2 | Deno (V8) wasm2 | Deno (V8) wasm3 | Bun (JSC) wasm2 | Bun (JSC) wasm3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 20.0 | 15.6 | 15.6 | 18.7 | 18.7 | 17.5 | 17.7 |
| 5 | 34.2 | 24.3 | 23.4 | 27.9 | 27.3 | 25.8 | 25.8 |
| 9 | 34.7 | 24.7 | 23.9 | 28.5 | 28.0 | 26.7 | 26.5 |

Relative to libopus scalar (median; >1 = slower than scalar C):

| complexity | libopus AVX | libopus AVX2 | Deno (V8) wasm2 | Deno (V8) wasm3 | Bun (JSC) wasm2 | Bun (JSC) wasm3 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0.78 | 0.78 | 0.93 | 0.94 | 0.88 | 0.88 |
| 5 | 0.71 | 0.68 | 0.82 | 0.80 | 0.75 | 0.76 |
| 9 | 0.71 | 0.69 | 0.82 | 0.82 | 0.77 | 0.77 |

## Notes

- **Noise:** the host was idle and median and min agree within about 1% in every cell, but this is one run per cell in one session. The libopus numbers and the earlier table in `8ebc26e` come from different machines (a 4 vCPU cloud VM there), so compare cells within a table, not across the two files.
- **Effect of `wasm-opt`:** an interleaved A/B on this host (unoptimized release vs `-O2 -Os -c -lmu` vs `-O2 -Oz -c -lmu`, 3 rounds of 15 passes, Deno and Bun, wasm2 and wasm3, both codecs, complexity 1/5/9) found the optimized builds within about 3% of the unoptimized ones in every cell: CELT 0 to 3% faster, SILK on Deno at complexity 5 and 9 about 2 to 3% slower, everything else within 1%. Geometric mean over all cells: `-Os` +0.0%, `-Oz` -0.3%. Binaries are about 9% (CELT) and 8% (SILK) smaller. Output bytes are identical in all builds. The A/B raw data is not kept in the repo.
- "AVX" is not a distinct upstream configuration: libopus x86 has SSE/SSE4.1 and AVX2 kernels only, so the AVX row is the SSE4.1-intrinsics build compiled with VEX encoding (`-mavx`), without AVX2 kernels.
- Reproduce (`wasm-opt` must be in PATH for the build): `bench/gen-pcm.sh; scripts/build.sh all all release; bench/build-native.sh; RUNTIMES="deno bun" bench/run-all.sh 15 > target/bench/results.jsonl`. For Wasmtime and Wasmer, build `bench/host` with `--features wasmtime|wasmer` (pass `--target x86_64-unknown-linux-gnu`, since the repo `.cargo/config.toml` defaults to wasm32) and omit `RUNTIMES`. Wasmer's Cranelift backend rejects the SILK wasm3 module (`proposed tail-call operator ReturnCall`).
