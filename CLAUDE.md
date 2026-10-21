# minisilk / minicelt workspace

Two independent Rust ports of the libopus 1.6.1 float encoder for wasm32: `minisilk/` (SILK-only)
and `minicelt/` (CELT-only). Each crate has its own `CLAUDE.md` (scope) and `PORTING.md`
(configuration, trimming, tests); read them before changing that crate.

## Rules

- Keep the crates separate: no shared crate, no cross-crate dependency. Copy and adapt instead.
- wasm32 only; variants WASM 2.0 (`+simd128`) and 3.0 (`+relaxed-simd`). `no_std`, no alloc.
- The oracle is upstream libopus 1.6.1 built natively as scalar C: minisilk is tested for bit-exactness
  against it, minicelt by tolerance (its SIMD kernels sum in lane order; see minicelt/PORTING.md).
  C reference builds use `-ffp-contract=off`.

- Porting is finished. Closeness to the C code is no longer a criterion, either for or against a change:
  idiomatic Rust rewrites (renamed/specialised functions, constants hardcoded for the only call site,
  restructured loops) are fine as long as output stays correct (bit-exact for minisilk, within tolerance
  for minicelt). Judge changes by bounds checks, size, speed and test results. Each `PORTING.md`
  describes the fixed configuration, what was trimmed and why, the bounds-check conventions and the tests.

## Layout

- `minisilk/`, `minicelt/`: `src/` (library), `wasm/` (cdylib with the `mini*_*` exports),
  `test/compare.ts` (Deno FFI) + `test/*.c`.
- `third_party/opus`: submodule, upstream libopus v1.6.1, test reference only.
- `test/audio`: speech corpus (GPL-2.0, test-only).

## Commands

- `scripts/build.sh [silk|celt|all] [2|3|all] [release|checked|both]`
  -> `target/mini{silk,celt}-wasm{2,3}[-checked].wasm`
- `scripts/build-native.sh` builds the native libopus references (`target/native/`).
- `scripts/test.sh [silk|celt|all] [--quick]` builds everything and runs the comparisons.
- `scripts/bench.sh [-e ENGINE]... <audio dir> label=path.wasm...`: minicelt encode-time A/B benchmark,
  AOT-compiled by Wasmtime / Wasmer (`minicelt/bench`, a native crate outside the wasm workspace).
- `scripts/variance.ts` (Deno; after `build-native.sh`) measures how much upstream libopus' scalar and SIMD
  encoders differ (SILK, CELT); `scripts/select-clips.ts` cuts test clips from long recordings for it.
