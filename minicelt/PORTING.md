# minicelt: configuration, trimming, tests

minicelt is a Rust port of the libopus CELT-only float encoder
(`OPUS_APPLICATION_RESTRICTED_CELT`), for wasm32 only. C source: the fork
https://github.com/ruihe774/opus-wasm-simd, branch `v1.6.1-wasm-simd` (upstream
libopus 1.6.1 plus WASM SIMD / relaxed SIMD kernels in `celt/wasm`; its scalar
encoder code is identical to 1.6.1; not vendored here, fetch it to consult). Do not
use the fork's `wasm-simd` branch: it is based on post-1.6.1 upstream, whose
scalar code behaves differently. The oracle for testing is natively built
upstream libopus 1.6.1 (`third_party/opus`, scalar C), compared by tolerance
because the kernels sum in lane order and are not bit-exact with scalar C. See
Tests.

## Fixed configuration

- 48 kHz input (`upsample == 1`), mono or stereo (`channels` 1 or 2, chosen at
  init), 20 ms frames (`LM == 3`, `frame_size == 960`; short blocks are still
  used for transients). The only mode is `mode48000_960_120`, whose fields are
  the constants and statics of `celt::modes` (`modes::nbEBands`, `modes::eBands`, ...).
- VBR, with constrained VBR on (the libopus default) or off. No CBR
  (`vbr_rate == 0` paths are removed with a `debug_assert!`).
- DTX is supported (Opus layer).
- No tonality analysis (`src/analysis.c`, `mlp.c`): `AnalysisInfo::valid` is
  always 0, and code under `if (analysis.valid)` / `st->analysis.valid` is
  removed with a `debug_assert!`. The test reference is built without
  `analysis.c` to match (see Tests).
- Float path only (no `FIXED_POINT`), no `FLOAT_APPROX`, no `ENABLE_QEXT`,
  `ENABLE_DRED`, `ENABLE_PFA`, `CUSTOM_MODES`, `RESYNTH`, no LFE, no energy
  mask (surround), no `signalling` (Opus custom header), `start == 0`.
- Where the fork has a WASM SIMD kernel (`celt/wasm`), the crate uses **that** kernel with
  `core::arch::wasm32` and has no scalar version. Relaxed SIMD is
  `#[cfg(target_feature = "relaxed-simd")]` (C's `#if defined(__wasm_relaxed_simd__)`). Kernels that
  matter here: `op_pvq_search_wasm` (vq_wasm.c), `celt_inner_prod_wasm`,
  `dual_inner_prod_wasm`, `celt_pitch_xcorr_wasm` (with its `xcorr_kernel_wasm` part),
  `comb_filter_const_wasm` (pitch_wasm.c). The fork has no kernels for the MDCT, FFT and `bands.c`; they
  are plain scalar code.
- Everything that cannot matter in this configuration is removed (see
  "Trimming" below): other modes or rates, CBR, QEXT, decoder-only
  `encode == 0` paths, state and parameters that are constant. Functions shared
  with the decoder lose their `encode` parameter (always 1). Where an assumption
  is not obvious from the code, a `debug_assert!` states it.

## Layout

One Rust module per C file, named after it (`celt/quant_bands.c` →
`src/celt/quant_bands.rs`, `src/opus_encoder.c` → `src/opus/encoder.rs`).
Function, struct, field and constant names are mostly the C ones (`quant_coarse_energy`,
`st.prefilter_period`); the crate allows the naming lints. Header-only inlines
are in the module of the matching `.c` file.

| C | Rust |
|---|---|
| `arch.h` (types, `MIN16`/`MAX32`/`IMIN`/..., `celt_isnan`, `ABS16`) | `celt::arch` |
| `mathops.h`, `mathops.c` (float) | `celt::mathops` |
| `entcode.h`/`entcode.c`, `entenc.c` | `celt::entcode`, `celt::entenc` |
| `modes.h`, `static_modes_float.h` | `celt::modes` (the mode's constants), `celt::static_modes_float` (generated, trimmed) |
| `kiss_fft.h`/`.c`, `mdct.h`/`.c` | `celt::kiss_fft`, `celt::mdct` |
| `pitch.h`/`.c` + `wasm/pitch_wasm.c` | `celt::pitch` |
| `vq.h`/`.c` + `wasm/vq_wasm.c` | `celt::vq` |
| `celt.h`, `celt.c` | `celt::celt` |
| `struct OpusCustomEncoder` | `celt::celt_encoder::CELTEncoder` (`#[repr(C)]`) |
| `opus_defines.h`, `opus_private.h` constants | `opus::defines` |
| `stack_alloc.h` | `stack` |

`CELTEncoder`'s trailing variable-size arrays (`in_mem`,
`prefilter_mem`, `oldBandE`, `oldLogE`, `oldLogE2`, `energyError`) are fixed
arrays sized for two channels and indexed with the C per-channel strides
(`c*overlap`, `c*COMBFILTER_MAXPERIOD`, `c*nbEBands`), so for stereo the
bytes equal the C allocation. The struct has no pointers and no non-zero initial value, so the
static encoder is in `.bss` and costs nothing in the `.wasm` file.

## Trimming

The crate is trimmed to the fixed configuration: size (code and memory) is the main optimization,
and git has what was removed. Rules for what to keep when changing it: delete what produces no
difference (performance, numbers or bitstream) in the configuration, and check with the
packet-identity regression check below. What was removed or specialized:

- Unreachable configurations: modes other than the 48 kHz/960 one (and their `CELTMode`
  struct, now constants; allocation, cache, twiddle and window tables of other frame sizes), LM
  other than 3 (functions lost their `LM`, `M`, `start`, `m` parameters; `tf_select_table`,
  `e_prob_model`, `pred_coef`/`beta_coef`, `cache_caps` keep only the LM 3 data; `compute_mdcts`
  and `clt_mdct_forward` only know the shifts 0 and 3), `effEnd == end`, `start == 0`
  (`special_hybrid_folding` and the `start`-dependent code are gone), `upsample`, hybrid-only code
  (weak transients, ...), the `OPUS_SET_*` ctls the exports do not offer.
- State that is constant: `CELTEncoder` lost `mode`, `start`, `vbr`, `clip`, `upsample`,
  `force_intra`, `disable_pf`, `disable_inv`, `signalling`, `lfe`, `arch`, `energy_mask`, `analysis`,
  `silk_info`, `preemph_memD`; `OpusEncoder` lost `silk_mode`, `signal_type`, `force_channels`,
  `max_bandwidth`, `lsb_depth`, `voice_ratio`, `peak_signal_energy`, `prev_mode`, ... The CELT setters
  are plain field writes. The Opus layer is straight-line code with the constants folded in
  (`voice_est == 48`, so the mono/stereo threshold and the bandwidth thresholds are constants;
  `max_data_bytes == 1276`; the activity decision, which only the unreachable SILK path used, is
  gone; the DTX decision only ever sees digital silence).
- Tables that are not stored: the `CELT_PVQ_U_DATA` table (1272 `u32`) is built at init by the
  recurrence (`cwrs::init_cwrs`), the FFT twiddles from their first octant by symmetry and the
  bit-reversal tables by `compute_bitrev_table` (`kiss_fft::init_fft`). They live in `.bss`.
  `scripts/gen_tables.py` checks that the rules reproduce libopus' tables bit for bit. The MDCT twiddles and the
  window are not derivable bit-exactly (the C tables are not correctly rounded libm values) and stay.
- libm (no longer a dependency): `sqrt` and `floor` are the wasm instructions (`mathops::sqrt64`,
  `mathops::floor64`); `mathops::cos_norm` is the part of libm's `cos` that its angles reach, `acos64`
  libm's `acos` with the hardware `sqrt`, `mathops::log2` libm's `log` for positive normal `f32` (reduced
  on the `f32` bits) and `mathops::exp2` libm's `exp` for `0 <= x <= 4` (no special cases, no `scalbn`).
  All four are bit for bit equal to `libm`'s, checked exhaustively over the inputs they can see.
- `mathops::maxabs` is vectorized (`f32x4_pmax`/`f32x4_pmin`, which are the C ternaries `fmax`/`fmin` with the
  operands swapped; min and max do not depend on the order): packets identical to the scalar loop, 2-5 % less
  encode time in `bench.ts` under V8 (it scans every frame three times). `minicelt/test/bench_maxabs.ts` microbenchmarks it.
- C `amp2Log2` always follows `compute_band_energies` and is folded into it (one loop computes `bandE` and
  `bandLogE`): -205 B, same speed. Taking `sqrt` and `log2` two bands at a time (`f32x4_sqrt`, `log2` on `f64x2`)
  gives the same packets but no measurable speedup (21-42 `log2` calls per frame) for +650 B, so it is not done.
- Vectorized with the same packets: the `transient_analysis` floor loop (four samples per `f32x4_floor` and
  saturating conversion, table lookups scalar) and the dynalloc importance (`mathops::exp2x2`, libm's `exp` on
  `f64x2`, replaces the scalar `exp2`; then `f32x4_floor`): together -656 B and 0.2-1.4 % less encode time.
  Batching the `remove_doubling` pitch gains four at a time was +1.5 KB and no faster, so it is not done.
- Bounds checks: where the compiler can be given a proof (constant indices into fixed arrays, masks,
  `match` on a bounded value, iterators, `chunks_exact`, a re-slice hoisted out of the loop) the code uses
  plain indexing, which is checked in every profile and adds no check in a per-sample loop. Where it
  cannot (indices read out of tables or computed from coded values, caller-sized slices at offsets from the
  band table, strided per-sample loops whose iterator form measured slower) indexing is inlined
  as `get_unchecked`/`get_unchecked_mut` in an `unsafe` block behind a `debug_assert!` on the indices
  (unchecked like C in release, checked in the `checked` profile). There is no indexing helper type any
  more (`arch::Ix` was removed). Every function that still does this has an `IX-HARD` comment
  giving the reason and, where a conversion was tried, its measured cost (bytes, encode time). The rule
  for a conversion: no bounds check added inside a per-sample loop (a check hoisted out of the loop, or in
  a band-level loop that runs once per frame, is fine), size is a soft constraint, and the encode time
  measured on the bench backends (Deno, Bun, Wasmtime; "Bench backends" below) must stay within noise. `haar1`,
  `interleave_hadamard` and the `CELT_PVQ_U` lookup were tried and kept unchecked on that measure (their
  `IX-HARD` comments list the attempts). New code
  should follow the same split, and run the `checked` tests.
- Memory: the pseudostack is 16 KiB and the shadow stack 8 KiB (the module stays one page up to about
  10 KiB), so the whole module is one 64 KiB wasm page. What keeps the pseudostack small: `norm` of
  `quant_all_bands` only exists when the theta RDO resynthesises (without it the folding source
  is never read), the C `X` buffer is the `in` buffer, which with the PCM copy is the caller's static
  input buffer (`wasm/src/lib.rs`: `IN_BUF_FRAME` samples per channel, clobbered by encoding, and the
  i16 input is converted in place in it), `run_prefilter` reads the pitch analysis input in place and
  only allocates its `_pre` buffer after the analysis, and the MDCT output is scoped.

Measuring the memory: `minicelt/test/stack_peak.ts` (build and run commands in its header). It sweeps
2592 configurations (mono/stereo x complexity x bitrates x constrained VBR x packet loss x bandwidth,
i16/f32 input, speech corpus and synthetic signals) and prints the pseudostack high-water mark, read
from `minicelt_stack_peak()` of a `--features stack-peak` build, and the shadow-stack depth, found
by painting the first `MINICELT_SHADOW_STACK` bytes of linear memory (the build script of
`minicelt/wasm`; use 16384 to measure) and scanning for the lowest byte written. Current numbers,
for both variants and both profiles: pseudostack peak 16376 bytes (the stereo prefilter, every
frame), shadow stack 708 bytes.

Regression check for trims that must not change anything: encode a corpus with a previous build and
the new one and require identical packets, final ranges and `in_dtx` (the tolerance gate below
does not see small changes). Mono/stereo, complexity 0..10, bitrates from 500 to 750000, constrained VBR
on/off, packet loss, bandwidths, DTX, i16/f32, dynamic ctl changes and the speech corpus were used,
both variants.

Speed regressions: one benchmark on three backends ("Bench backends" below). `minicelt/test/bench.ts`
(usage in its header) runs under Deno (V8) and Bun (JavaScriptCore), whose tiering JITs can run different
code from run to run. `scripts/bench.sh` builds and runs `minicelt/bench`, a native harness (usage in
`minicelt/bench/src/main.rs`) that compiles the builds ahead of time with Wasmtime (Cranelift), so the code
that runs is the same on every run. Both encode the concatenated speech corpus with several builds,
interleaved, and print min/median time and output size per configuration. Measure the noise with an A/A
run (one build against a copy of itself) and run nothing else meanwhile. Run it for any change that touches a hot loop
(it covers mono at 24 and 64 kbps and stereo at 6 to 128 kbps, complexity 5 and 10, i.e. every function
changed in the bounds-check conversion; `stereo_fade` only runs for stereo at 24 kbps and below. It does not
cover packet loss, bandwidth limits, DTX or constrained VBR, nor low complexity).

Coverage: `scripts/coverage.sh [--quick]` builds an unoptimized module (so functions are not inlined
away), instruments it with `wasm-opt --log-execution` and encodes the speech corpus in 19 configurations
(mono/stereo, 6 to 510 kbps, complexity 0 to 10, constrained VBR, DTX, packet loss, bandwidth limits)
through it. It prints, to stdout only, the functions never entered and the functions with unhit logging
points (function entries, loop headers, branch arms, returns; not source lines), per module. CI runs it
as the `coverage` job (log only, never fails the workflow, nothing uploaded). The workload is its own, not
`compare.ts`'s, so a function it reports as never entered may still be reached by the test suite
(`quant_band_n1`, for one, is not reached by the speech corpus).

## Bench backends

Speed is measured on Deno (V8) and Bun (JavaScriptCore) with `minicelt/test/bench.ts`, and on Wasmtime
(Cranelift, ahead of time) with `scripts/bench.sh`. Wasmer is excluded: its Cranelift and LLVM engines stay
in `minicelt/bench` (`-e wasmer-cranelift`, `-e wasmer-llvm`), but changes are not judged by them.

Why Wasmer is excluded: Wasmer-Cranelift adds nothing over Wasmtime (same code generator, within -1 to +6 %
of it), and Wasmer-LLVM 7.5.0 (LLVM 22.1.2) is 1.4-4x slower than Wasmtime because of how it translates
SIMD into LLVM IR, not because of minicelt:

- `v128.bitselect` and `*.relaxed_laneselect` become `select <128 x i1>` over `i128` bitcasts
  (`wasmer-compiler-llvm` `translator/code.rs`, the `V128Bitselect` arm). The AArch64 backend scalarises
  that into one bit extract and one byte-lane insert per bit, instead of one `bsl`. It hits the PVQ pulse
  search (`vq.rs`, the `laneselect` calls, inlined into `quant_partition`): 81 % of that loop's ~1800
  instructions, `quant_partition` 11x slower than under Cranelift, and the gap grows with the bitrate
  (more pulses). rustc canonicalises every mask-merge form back to `v128.bitselect`, so the source cannot
  avoid it. Patching that arm to `and`/`andnot`/`or` on `<4 x i32>` (or rewriting the module, table below)
made Wasmer-LLVM 2-3.6x faster from 24 kbps up, 1.05x at stereo 6 kbps (few pulses).
- Every `v128` is an `i128` (locals, loop phis, loads), which AArch64 keeps in X-register pairs:
  loop-carried accumulators cross between GPRs and NEON registers on every operation. `celt_pitch_xcorr`,
  `celt_inner_prod` and `dual_inner_prod` are 4-7x slower than under Cranelift (24-30 % of their
  instructions are such moves).
- Memory accesses are `volatile`, with a dead load before every store, unless `non_volatile_memops(true)`
  (not spec-exact): 3-16 % slower as shipped, 10-18 % once `bitselect` is patched.

Results of 2026-10-21. Same machine and session throughout: aarch64, pinned to one Cortex-X925 core
(`taskset -c 7`), Deno 2.9.5, Bun 1.4.4, Wasmtime 50.0.0, Wasmer 7.5.0. Min ms per run (639 frames,
12.8 s of audio); the medians are within 0.5 % of the minimums.

Engines, builds of 9931d8e (release, wasm-opt as in `scripts/build.sh`; re-encoded by `wasm-tools`, same
code). The rewritten modules ("no bitselect") replace each `v128.bitselect` / `relaxed_laneselect` with
`v128.and`/`v128.andnot`/`v128.or` on locals; packets are identical.

| config | Deno w2 | Bun w2 | Wasmtime w2 | Deno w3 | Bun w3 | Wasmtime w3 | Wasmer-LLVM w2 | w2 no bitselect |
|---|---|---|---|---|---|---|---|---|
| ch=1 24k cx10 | 20.4 | 18.6 | 20.4 | 19.7 | 21.5 | 19.5 | 73.0 | 33.6 |
| ch=1 64k cx10 | 24.7 | 22.9 | 24.9 | 23.8 | 25.5 | 23.8 | 108.7 | 38.9 |
| ch=2 64k cx10 | 43.7 | 40.7 | 43.6 | 42.3 | 43.4 | 42.1 | 162.2 | 59.5 |
| ch=2 128k cx10 | 63.7 | 60.0 | 64.0 | 61.2 | 62.0 | 61.5 | 305.3 | 83.7 |
| ch=1 64k cx5 | 24.3 | 22.6 | 24.5 | 23.4 | 25.2 | 23.4 | 108.3 | 38.6 |
| ch=2 6k cx10 | 20.0 | 18.1 | 19.5 | 19.4 | 21.0 | 18.8 | 33.6 | 31.7 |
| ch=2 16k cx10 | 23.7 | 21.6 | 23.3 | 23.1 | 24.5 | 22.5 | 53.8 | 36.1 |
| ch=2 24k cx10 | 33.3 | 30.8 | 33.3 | 32.4 | 33.8 | 32.2 | 85.8 | 47.6 |

- Removing `bitselect` does not matter on the bench backends: Deno, Bun and Wasmtime are within -0.2 to
  +0.6 % of the original, both variants (they emit one `bsl`).
- WASM 3.0 is 3-4 % faster than 2.0 on Deno and Wasmtime but 3-16 % slower on Bun. Not `laneselect`
  (the rewritten w3 is as slow); not investigated further.

Release against `-checked` (debug assertions and overflow checks), builds of 1e1bfe5, w2: +27-35 % on
Wasmtime (e.g. 26.3 -> 35.4 ms at ch=1 24k, 77.6 -> 98.4 at ch=2 128k), +5-8 % on Wasmer-LLVM (whose
release build is already 1.4-4x slower). Same builds, Wasmtime against Wasmer-Cranelift and Wasmer-LLVM:
26.3 / 26.7 / 77.1 ms at ch=1 24k, 77.6 / 76.8 / 313.1 at ch=2 128k, 28.1 / 29.8 / 40.3 at ch=2 6k.

## Conventions

- Numerics: correct output is what the tests decide (tolerance gate below), not the C types or operation
  order. Rust does not contract `a * b + c` into an FMA and the reference is built with `-ffp-contract=off`,
  so plain scalar expressions round like the reference. Hand-vectorized loops are fine: packets identical
  to the previous code is the easy case, anything else must pass the gate. `x as i32` truncates and
  saturates.
- Assertions are `debug_assert!` (active in the `checked` profile).
- Plain arithmetic uses plain Rust operators, so `checked` builds trap on integer overflow. Use `wrapping_*`
  where wraparound is intended (unsigned arithmetic in `entenc`, `celt_lcg_rand`, ...) and say so.
- Scratch memory: `let st = stack::save();` and `st.alloc::<i32>(n)` (freed when `st` goes out of scope; not
  cleared). The pseudostack (`GLOBAL_STACK_SIZE`) is sized to the measured peak: allocate in scopes that
  end as early as possible (see Trimming) and re-measure after changes.
- No `std`, no `alloc` in non-test code.

## Tests

There are no Rust unit tests; the test suite is end to end only. A per-function bit-exact oracle
(a wasm32-wasip1 build of the fork's C, linked into `cargo test` and run under Node's WASI, so C and
Rust run the same SIMD in the same engine) is kept at the git tag `minicelt-oracle-tests`
(`minicelt/scripts/oracle-test.sh [2|3|all]` there); check it out to localize a regression.

**End to end** (`minicelt/test/compare.ts`, Deno; `scripts/test.sh celt [--quick] [--all]
[--strict-scalar] [--filter=text] [audio files...]`). Every implementation sits behind one
adapter with the `test/shim.c` API (create, ctl setters, encode i16/f32, final range, in_dtx,
lookahead; decoder for the bitstream check):
- `scalar`, `rtcd`: upstream libopus 1.6.1 natively (`scripts/build-native.sh`, Deno FFI).
- `wasm2`, `wasm2-checked`, `wasm3`, `wasm3-checked`: the minicelt builds (`minicelt_*` exports).

The same input and ctl schedule (static settings: bitrate, complexity, constrained VBR, DTX, packet
loss, bandwidth, i16/f32 input, hot f32 x2.5; dynamic runs with random mid-stream changes; mono and
stereo; 48 kHz speech upsampled x3, synthetic music, transients, noise, chirp, DTX patterns, DC, clipping,
hard-panned/anti-phase/mono-compatible stereo) goes through each implementation once; the traces
(packets, final range, in_dtx, ctl return codes) are then compared pairwise. **Exact pairs** (mismatch =
failure): `wasm2` = `wasm2-checked`, `wasm3` = `wasm3-checked`. **Informational**: scalar and rtcd vs the
wasm builds. Native scalar is *not* an exact reference for the minicelt builds (unlike SILK): the float
SIMD kernels (`celt_inner_prod`, `dual_inner_prod`, `xcorr_kernel`/`celt_pitch_xcorr`, `op_pvq_search`, ...)
accumulate in lane order, not in the C order, so packets differ in a share of the frames (identical SNR).
The fork's C built with its kernels disabled is bit-exact with scalar, so the harness, the libm and
the signal path are not the cause. `--strict-scalar`
makes scalar vs `wasm2`/`wasm2-checked` exact pairs. Any trap, native decoder error, bad TOC or
encoder/decoder final range mismatch fails every run. Every packet is also decoded natively (scalar
decoder with the right channel count) and the SNR of the decoded streams is reported.

**Tolerance gate against upstream scalar** (the oracle for the minicelt builds). `scalar` vs `wasm2`,
`wasm2-checked`, `wasm3`, `wasm3-checked` is judged by effects, because SIMD summation order and FMA change
packets in a share of the frames (upstream's own SIMD
builds do too: `scripts/variance.ts` measured 0.15-1.3 % of frames on aarch64 NEON and 41-45 % on x86-64 AVX2, whose
`celt_pitch_xcorr` fuses with FMA, with SILK bit-exact on both) without a measurable effect on quality or size.
Checks (`GATE` in `compare.ts`), pooled over all runs and again for mono and stereo: ctl return codes and lookahead
equal; `|mean dSNR|` <= 0.02 dB and `|mean dbytes|` <= 0.01 % (a mean fails only if it is also more than 4 standard
errors from zero: noise is symmetric, a defect is not); every run within 1 dB and 2 % of the size (4 bytes floor);
`in_dtx` differing in <= 0.5 % of the frames; per CELT band (21 bands, Hann-windowed 20 ms frames of the decoded
streams) the mean log-spectral distance B-A within 0.02 dB (same significance rule) and the mean absolute distance
within 0.1 dB, over all decoded frames. The band check is there because SNR is dominated by the loud low bands and
cannot see a defect in the top ones. Packet divergence rate is reported, not gated: it is uninformative on x86. The
same checks run for `scalar` vs `rtcd` (upstream's own SIMD) as the baseline of the bounds: if it fails on a host,
the bounds are too tight for it. In `--quick` the band checks are informational (too few frames: the baseline fails
them there). Sensitivity: a 2.4 % change of the encoder's pre-emphasis fails nearly every check; a 10 % change of the
constrained-VBR modulation factor (0.67 to 0.60) fails only the mean band distance, and smaller decision tweaks are
not detectable.

**Relaxed SIMD and FMA**: `f32x4.relaxed_madd` is fused on x86 with FMA (V8
emits `vfmadd`), so the WASM 3.0 build's float kernels round differently from
scalar C, and the float kernels of both builds sum in a different order than the C
code. Both are therefore judged against native scalar by the tolerance gate above, not
bit-exactly. The WASM 3.0 build is the one most likely to approach the gate's bounds
(its FMA behaviour is that of upstream's x86 AVX2 build, which diverges most).
