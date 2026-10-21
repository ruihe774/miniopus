# minicelt

Rust port of the libopus CELT-only float encoder for wasm32. Read `PORTING.md` before changing
`src/`; it holds the configuration, trimming and test details.

## Scope (fixed, do not widen without asking)

- wasm32 only (no native build); variants WASM 2.0 (`+simd128`) and 3.0 (`+relaxed-simd`).
- 48 kHz, mono or stereo, 20 ms frames, VBR with constrained VBR on/off, DTX.
  No CBR, no tonality analysis, no other rates or frame sizes, no QEXT/DRED/custom modes.
- `no_std`, no alloc. Scratch memory is the static pseudostack in `src/stack.rs` (16 KiB, sized to
  the measured peak: see its doc comment and PORTING.md "Trimming" before changing allocations).
- Trimmed to the scope above: code, state and tables that cannot matter in it are removed, size
  (code and memory) is the main optimization (the module is one wasm page). Trims must keep the
  packets identical (PORTING.md "Trimming").

## Layout

- `src/celt`, `src/opus`: one Rust module per C file.
- `wasm/`: the `minicelt-wasm` cdylib with the `minicelt_*` exports.
- `test/compare.ts` (Deno) + `test/shim.c` + `test/analysis_stub.c`: differential test against native
  `third_party/opus` (scalar C as the oracle, by tolerance). `scripts/gen_tables.py` generates
  `src/celt/static_modes_float.rs` (trimmed to the scope).
- `bench/` (native, own workspace; run with `scripts/bench.sh`): A/B encode-time benchmark of builds,
  compiled ahead of time by Wasmtime / Wasmer (Cranelift, LLVM) (PORTING.md "Trimming").
  `test/bench_maxabs.ts`: A/B microbenchmark of `mathops::maxabs` (builds with `--features bench-maxabs`).
- `test/coverage.ts` + `scripts/coverage.sh`: function / branch-arm coverage printed to the log (CI job `coverage`).

## Commands (from the workspace root)

- `scripts/build.sh celt [2|3|all] [release|checked|both]` -> `target/minicelt-wasm{2,3}[-checked].wasm`
- `scripts/test.sh celt [--quick]` builds everything and runs the comparison. There are no Rust
  unit tests (the per-function C oracle is kept at the tag `minicelt-oracle-tests`).

## Rules

- The test goal is the tolerance gate against upstream libopus 1.6.1 scalar C (see `PORTING.md`): the
  SIMD kernels sum in lane order, so packets are not bit-exact with scalar, but their effects must stay
  within the gate. How that is reached is free (hand-vectorized loops, other types or precision): a change
  that keeps the packets identical to the previous code is the easy case, any other must pass the gate.
- The SIMD kernels come from the WASM SIMD fork (https://github.com/ruihe774/opus-wasm-simd, branch
  `v1.6.1-wasm-simd`) and use `core::arch::wasm32`, with no scalar fallback.
  Relaxed SIMD uses `cfg(target_feature = "relaxed-simd")`.
- Bounds checks: plain indexing where provable, inline `get_unchecked` behind a `debug_assert!` only for the `IX-HARD` sites (PORTING.md
  "Bounds checks"); no check added in a per-sample loop, and bench encode time (`scripts/bench.sh`).
- `src/` is BSD-3 (libopus derivative).
