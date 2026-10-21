# minisilk: configuration, trimming, tests

minisilk is a Rust port of the libopus SILK-only float encoder
(`OPUS_APPLICATION_RESTRICTED_SILK`), for wasm32 only. C source: libopus 1.6.1
plus a fork's WASM SIMD / relaxed SIMD kernels (https://github.com/ruihe774/opus-wasm-simd,
branch `v1.6.1-wasm-simd`; not vendored here, fetch it to consult). Do not use the fork's `wasm-simd`
branch: it is based on post-1.6.1 upstream, whose scalar code behaves differently. The kernels are bit-exact with the scalar C, so the
reference for testing is plain upstream libopus (`third_party/opus`), compared bit-exactly. Upstream's own
SIMD builds are bit-exact with scalar for this configuration too (`scripts/variance.ts`: 0 differing runs
in about 4000 on aarch64 NEON and on x86-64 AVX2).

## Fixed configuration

- 16 kHz mono input, wideband (internal rate is always 16 kHz), 20 ms frames
  (`nb_subfr == 4`, `frame_length == 320`), VBR.
- DTX is supported. No FEC/LBRR, no CBR, no stereo, no resampler (replaced by an
  11-sample input delay line: the resampler's 10 samples plus the one sample of C's mono
  "stereo mid" buffer), no 10/40/60 ms frames.
- The fixed values are compile-time constants (`silk::define`: `FS_KHZ`, `MAX_NB_SUBFR`,
  `MAX_FRAME_LENGTH`, `MAX_SUB_FRAME_LENGTH`, `LTP_MEM_LENGTH`, `LA_PITCH_MAX`,
  `FIND_PITCH_LPC_WIN_MAX`, `MAX_LPC_ORDER` = predictLPCOrder, `MAX_FRAMES_PER_PACKET` = 1), not
  state fields: `sCmn.fs_kHz`, `nb_subfr`, `frame_length`, ... do not exist. Every frame is
  coded independently (`CODE_INDEPENDENTLY`, the first and only frame of a packet), so the
  conditional-coding paths, the pitch-lag delta coding and `ec_prevLagIndex`/`ec_prevSignalType`
  are gone.
- The state structs keep only what the configuration reads or writes (no `silk_EncControlStruct`
  - its fields are arguments of `silk_Encode` -, stereo state, LBRR buffers, resampler state,
  bandwidth-switch state, `sLP`, ...). They are plain Rust structs (no `#[repr(C)]`) and do not
  mirror the C layout; nothing depends on field order or offsets.
- Float path only (`silk/float`, no `FIXED_POINT`).
- Where the fork has a WASM SIMD kernel (`silk/wasm`, `silk/float/wasm`,
  `celt/wasm`), the crate uses **that** kernel with `core::arch::wasm32` and has no
  scalar version. Relaxed SIMD is `#[cfg(target_feature = "relaxed-simd")]` (C's
  `#if defined(__wasm_relaxed_simd__)`). Scalar fallbacks for
  configurations we never use (e.g. `nStatesDelayedDecision > 4`) are a
  `debug_assert!`.
- Branches that are unreachable under the fixed configuration (fs_kHz != 16,
  nb_subfr == 2, stereo, LBRR, CBR) are removed, state included, with a `debug_assert!` of
  the assumption where it is cheap. Size (code and memory) is the main optimization, so do not
  add them back "for completeness"; git has them.

## Trimming

The crate is trimmed to the fixed configuration (see "Fixed configuration" above for what was
removed or specialised: other rates, frame sizes, stereo, FEC/LBRR, CBR, the resampler, the
conditional-coding paths and the state that went with them, the scalar versions of kernels the fork
has in SIMD). Size (code and memory) is the main optimization, and git has what was removed. Rule for
what to keep when changing it: delete what produces no difference (numbers or bitstream) in the
configuration, and check with the bit-exact comparison (see Tests).

## Layout

One Rust module per C file, named after it (`silk/NLSF_VQ.c` →
`src/silk/NLSF_VQ.rs`). Function, struct, field and constant names are mostly the C ones
(`silk_NLSF_VQ`, `sCmn`); the crate allows the naming lints.

| C | Rust |
|---|---|
| `silk/define.h`, `errors.h`, `tuning_parameters.h`, `pitch_est_defines.h` | `silk::{define, tuning_parameters, pitch_est_defines}` |
| `SigProc_FIX.h` macros, `macros.h`, `Inlines.h` | `silk::macros` |
| `SigProc_FLP.h` inlines, `float2int` | `silk::float::sigproc_flp` |
| `structs.h`, `structs_FLP.h` | `silk::structs`, `silk::float::structs_flp` (trimmed, see above) |
| `tables.h` | `silk::tables` (re-exports all `tables_*` modules) |
| `entcode.h` `ec_ctx` | `celt::entcode` |
| `stack_alloc.h` | `stack` |

## Conventions

- Numerics: bit-exactness with the scalar oracle is the only criterion, so types and the order and
  precision of float operations matter only through it. Rust does not contract `a * b + c` into an FMA
  and the reference is built with `-ffp-contract=off`, so plain scalar expressions match (with
  contraction on, clang fuses multiply-adds in loops it auto-vectorizes under `-mrelaxed-simd`, which
  would make the reference depend on the optimizer). Hand-vectorized loops are fine as long as the output
  stays bit-exact. `x as i32` truncates and saturates; `silk_float2int`/`float2int` round half to even
  (`f32.nearest`).
- Bounds checks: indexing with run-time indices goes through `macros::Ix` (`x.at(i)`, `*x.at_mut(i) = v`,
  `x.rg(a..b)`, `x.rg_mut(a..b)`; the trait lives in `silk::macros`, which every module already imports;
  minisilk has no `arch` module): unchecked in release, checked in the `checked` profile (debug
  assertions). Fixed arrays work through method-call auto-unsizing. Plain `x[i]` stays where the index is
  a compile-time-bounded loop variable or constant (the compiler already drops the check) and in the SIMD
  tails that use raw pointers. For 2D arrays only the innermost index goes through `Ix`
  (`*m[j].at_mut(i)`), since `m.at(j)` would copy the row. Files with run-time-indexed slices (`NSQ`,
  `NSQ_del_dec`, `noise_shape_analysis_FLP`, `pitch_analysis_core_FLP`, `burg_modified_FLP`,
  `NLSF_del_dec_quant`, `encode_pulses`, `wrappers_FLP`, `NLSF2A`, `A2NLSF`, `corrMatrix_FLP`) use it,
  which drops panic sites from the release wasm; files whose loops have constant bounds (`VAD`,
  `encode_frame_FLP`) gain nothing and use plain indexing. New code should follow the same split, and run
  the `checked` tests. `unsafe` is allowed anywhere.
- Assertions are `debug_assert!` (active in the `checked` profile).
- Plain arithmetic uses plain Rust operators, so debug builds trap on overflow. Where the algorithm
  relies on wraparound, use the `_ovflw` macros / `wrapping_*` and say so in a comment.
- Scratch memory: `let st = stack::save();` and `st.alloc::<i32>(n)` (freed when `st` goes out of scope;
  not cleared).
- No `std`, no `alloc` in non-test code.

## Memory

Everything is static (`no_std`, no allocator), so the linear memory is stack + data + bss, with
the stack first (overflow traps at address < 0):

- Shadow stack (the Rust call stack): `wasm/build.rs` sets it to 48 KiB (default 1 MiB; the
  `MINISILK_SHADOW_STACK` env var overrides it). Deepest use measured: 32432 bytes (release) and
  36232 (`checked`), both variants.
- Pseudostack (`src/stack.rs`, `GLOBAL_STACK_SIZE`): 5504 bytes. Measured peak 5436 bytes: the
  `ec_buf_copy` of `silk_encode_frame_FLP` (1276 with alignment) plus the `silk_NSQ` scratch (4160).
- Encoder state 8564 bytes, I/O buffers 3196 bytes, tables ~10 KB: 2 wasm pages in total.

To re-measure after a change (do it whenever `silk_encode_frame_FLP`, the NSQs, the VAD, `encode_pulses`
or `NLSF_encode` change), build with the `stack-peak` feature (adds the `minisilk_stack_peak` export and the
high-water mark in `stack.rs`) and a 1 MiB shadow stack, then run `test/stack_peak.ts`, which sweeps
every bitrate (500 .. 750000, `OPUS_AUTO`, `OPUS_BITRATE_MAX`) x complexity 0-10 x DTX x packet loss
x i16/f32 over the corpus and the synthetic signals, painting the stack region to find the shadow
stack depth:

```sh
MINISILK_SHADOW_STACK=1048576 cargo build --release -p minisilk-wasm --target wasm32-unknown-unknown \
  --features stack-peak --target-dir target/stack-peak \
  --config 'target.wasm32-unknown-unknown.rustflags=["-Ctarget-feature=+simd128"]'
deno run --allow-read minisilk/test/stack_peak.ts target/stack-peak/wasm32-unknown-unknown/release/minisilk_wasm.wasm
```

(`--profile checked` for the checked build; the WASM 3.0 flags are in `scripts/build.sh`.)
The checked builds overflow-panic if the pseudostack is too small, and `scripts/test.sh` runs them.

## Tests

There are no Rust unit tests. `minisilk/test/compare.ts` (Deno) drives the wasm builds
through their exports next to natively built upstream libopus (via FFI and the
non-variadic wrappers in `minisilk/test/shim.c`) and compares the output frame by frame:

```sh
git submodule update --init      # third_party/opus (libopus v1.6.1)
scripts/test.sh silk [--quick] [audio files...]
```

Requirements: Rust with the `wasm32-unknown-unknown` target, a C compiler,
CMake, Ninja and Deno. Paths are relative to the workspace root. `scripts/test.sh`
runs `scripts/build-native.sh` once (rerun it after changing `test/shim.c` or the
submodule), `scripts/build.sh silk all both`, then the comparison.

- `scripts/build-native.sh` builds `target/native/minisilk-scalar.so` (intrinsics
  off) and `minisilk-rtcd.so` (upstream defaults, SSE/AVX2 on x86), both with
  `-ffp-contract=off`.
- `scripts/build.sh silk all both` builds the release wasm and the `checked` ones
  (the `checked` Cargo profile: release code with `debug_assert!`s and overflow
  checks, which trap on failure).
- Inputs: the speech recordings in `test/audio` (see
  `THIRD_PARTY_NOTICES.md`) at two levels with silent and noisy pauses, synthetic
  speech, noise at several levels, a chirp, a clipped square wave and clicks,
  tones and DC. Configurations: every bitrate x complexity, DTX on/off, i16/f32
  input, hot f32 input, and random mid-stream ctl changes.
- Every wasm build must be bit-exact with the scalar build (packets, final
  range, DTX state). The rtcd build's float kernels sum in a different order,
  so it is compared for stream validity and coding quality only. Every minisilk
  packet is also decoded natively and its decoder final range checked.
- A mismatch shows only where the stream diverges. The per-function
  differential tests against a wasm build of the C (one per ported function,
  comparing whole encoder states) are kept at the tag `minisilk-oracle-tests`; check it out to
  localize a regression.
