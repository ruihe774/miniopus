# minicelt porting guide

minicelt is a mechanical Rust port of the libopus CELT-only float encoder
(`OPUS_APPLICATION_RESTRICTED_CELT`), for wasm32 only. C source: the fork
https://github.com/ruihe774/opus-wasm-simd, branch `v1.6.1-wasm-simd` (upstream
libopus 1.6.1 plus WASM SIMD / relaxed SIMD kernels in `celt/wasm`; its scalar
encoder code is identical to 1.6.1; not vendored here, fetch it to port). Do not
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
- Where the fork has a WASM SIMD kernel (`celt/wasm`), port **that** kernel with
  `core::arch::wasm32` and drop the scalar C version. Relaxed SIMD: C's
  `#if defined(__wasm_relaxed_simd__)` becomes
  `#[cfg(target_feature = "relaxed-simd")]`. Kernels that matter here:
  `op_pvq_search_wasm` (vq_wasm.c), `celt_inner_prod_wasm`,
  `dual_inner_prod_wasm`, `celt_pitch_xcorr_wasm` (with its `xcorr_kernel_wasm` part),
  `comb_filter_const_wasm` (pitch_wasm.c). The MDCT, FFT and `bands.c` have no
  kernels and are ported as plain scalar code.
- Everything that cannot matter in this configuration is removed (see
  "Trimming" below), not ported: other modes or rates, CBR, QEXT, decoder-only
  `encode == 0` paths, state and parameters that are constant. Functions shared
  with the decoder lose their `encode` parameter (always 1). Where an assumption
  is not obvious from the code, a `debug_assert!` states it.

## Layout

One Rust module per C file, with the C file name (`celt/quant_bands.c` →
`src/celt/quant_bands.rs`, `src/opus_encoder.c` → `src/opus/encoder.rs`).
Function, struct, field and constant names stay as in C (`quant_coarse_energy`,
`st->prefilter_period`); the crate allows the naming lints. Header-only inlines
go into the module of the matching `.c` file.

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

`CELTEncoder` keeps the C layout where it can; its trailing variable-size arrays (`in_mem`,
`prefilter_mem`, `oldBandE`, `oldLogE`, `oldLogE2`, `energyError`) are fixed
arrays sized for two channels and indexed with the C per-channel strides
(`c*overlap`, `c*COMBFILTER_MAXPERIOD`, `c*nbEBands`), so for stereo the
bytes equal the C allocation. The struct has no pointers and no non-zero initial value, so the
static encoder is in `.bss` and costs nothing in the `.wasm` file.

## Trimming

The crate is trimmed to the fixed configuration: size (code and memory) is the main optimization,
and git has what was removed. Rules for what to keep when changing it: delete what produces no
difference (performance, numbers or bitstream) in the configuration, and check with the
bit-exact comparison below. What was removed or specialized:

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
  encode time in `bench.ts` (it scans every frame three times). `minicelt/test/bench_maxabs.ts` microbenchmarks it.
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
  measured with `minicelt/test/bench.ts` must stay within noise (about 0.2 %). `haar1`,
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

Speed regressions: `minicelt/test/bench.ts` (Deno; usage in its header) encodes the concatenated
speech corpus with two builds, interleaved, and prints min/median time and output size per
configuration. Differences below about 0.2 % are noise. Run it for any change that touches a hot loop
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

## Translation rules

- Types: `int`/`opus_int32` → `i32`, `unsigned`/`opus_uint32` → `u32`,
  `opus_int16` → `i16`, `unsigned char` → `u8`, `signed char` → `i8`,
  `opus_val16`/`opus_val32`/`celt_sig`/`celt_norm`/`celt_ener`/`celt_glog`/
  `opus_res`/`celt_coef`/`float` → `f32`, `double` → `f64`.
- **Float macros**: in the float build the fixed-point macros are identities or
  plain operators. Write them out: `MULT16_32_Q15(a,b)` → `a*b`,
  `MAC16_16(c,a,b)` → `c + a*b`, `SHR32(a,s)`/`SHL32`/`PSHR32`/`VSHR32`/
  `EXTEND32`/`EXTRACT16`/`SATURATE*`/`ROUND16` → `a`, `QCONST16(x,b)`/
  `QCONST32`/`GCONST(x)` → `x` (as an `f32` literal), `HALF32(x)` → `0.5*x`,
  `ADD32`/`SUB32` → `+`/`-`, `DIV32_16(a,b)` → `a/b`. Keep the order of
  operations exactly (`(c)+(a)*(b)`, left to right). Use the `arch` functions
  for `MIN16`/`MAX16`/`MIN32`/`MAX32`/`MING`/`MAXG`/`FMIN`/`FMAX` (C ternary
  semantics, *not* `f32::min`/`max`) and `IMIN`/`IMAX`.
- **Integer promotion**: C computes `opus_int16`/`unsigned char` arithmetic in
  `int`. Cast explicitly (`a as i32 * b as i32`). Watch unsigned arithmetic
  (`opus_uint32`, `unsigned`): mixing `int` and `unsigned` converts to
  `unsigned` in C.
- **Float promotion**: C evaluates an expression in `double` as soon as one
  operand is `double` (a `1.0` or `.5` literal without `f`, a `double`
  variable, `floor()`, `log()`, `sqrt()`, `exp()`, `cos()`, ...). Reproduce the
  exact precision of every operation with `as f64`/`as f32`; `int` operands
  combined with `float` become `float`. Getting this wrong shows up as a
  bit-exactness failure. libm functions: the `mathops` equivalents (see Trimming),
  exactly the precision C calls.
- Rust does not contract `a * b + c` into an FMA, and the reference is built
  with `-ffp-contract=off`, so plain expressions match. Never hand-vectorize a
  plain C loop to imitate compiler output.
- C `(int)x` on a float truncates; Rust `x as i32` truncates and saturates.
  `float2int` (`lrintf`) would be round half to even (`libm::rintf`); the encoder does not
  use it.
  `floor(.5+x)` stays exactly that.
- Arrays: prefer slices (`&[f32]`, `&mut [i32]`), indexed plainly where the bounds are provable and
  with `get_unchecked` behind a `debug_assert!` (unchecked in release, checked in the `checked` profile)
  where they are not (see "Bounds checks" above).
  Where C uses negative offsets or pointer arithmetic into a larger buffer,
  pass the buffer plus an offset, or use raw pointers; `unsafe` is allowed anywhere. C argument
  order is kept; `const T *` → `&[T]`, `T *` output → `&mut [T]` / `&mut T`.
- `celt_assert`/`celt_sig_assert` → `debug_assert!`. Drop `arch` parameters.
- Plain C arithmetic uses plain Rust operators, so `checked` builds trap on
  integer overflow. Use `wrapping_*` exactly where C relies on wraparound
  (unsigned arithmetic in `entenc`, `celt_lcg_rand`, ...) and say so.
- `ALLOC` / `SAVE_STACK` / `RESTORE_STACK`:
  ```rust
  let st = stack::save();          // SAVE_STACK
  let x = st.alloc::<i32>(n);      // ALLOC(x, n, opus_int32)
  // RESTORE_STACK when st goes out of scope
  ```
  Allocated memory is not cleared, as in C. `ALLOC_NONE` → nothing. The pseudostack
  (`GLOBAL_STACK_SIZE`) is sized to the measured peak: allocate in scopes that end as early as
  possible (see Trimming) and re-measure after changes.
- `celt_udiv`/`celt_sudiv`: `USE_SMALL_DIV_TABLE` is not defined (it is ARM
  only), so they are plain `/`.
- Keep the C comments that explain the algorithm; drop the boilerplate.
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
