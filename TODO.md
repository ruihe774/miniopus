# Open items

Found by comparing minisilk and minicelt (idioms, trimming, docs) and not done. Each is optional;
none changes the output. Sizes are release `.wasm` bytes.

## minisilk

- **`Ix` in the remaining files.** `macros::Ix` is used in 11 files (see `minisilk/PORTING.md`,
  "Bounds checks"). About 65 are left, mostly with constant-bound loops where the compiler already
  drops the checks. The largest candidates with run-time indices: `sort`, `sort_FLP`, `NLSF_encode`,
  `LPC_inv_pred_gain`, `VQ_WMat_EC`, `gain_quant`, `NLSF_decode`, `quant_LTP_gains`,
  `warped_autocorrelation_FLP`, `LPC_analysis_filter`, `find_pred_coefs_FLP`. Expect 100-300 bytes
  each; convert one at a time and compare sizes. `VAD` and `encode_frame_FLP` were tried and gained
  nothing.
- **One wasm page.** minicelt fits in one 64 KiB page; minisilk takes two. The shadow stack
  (32-36 KB deep) dominates, so this needs buffers moved into the pseudostack (design change).
- **Tables.** minicelt derives its PVQ, FFT and bit-reversal tables at init. minisilk's tables
  (NLSF codebook, LTP, pulses) are mostly irreducible; probably not worth it.
- **`SAFETY` comments.** Only the SIMD kernels have them; about 96 `unsafe` sites in total.
  Add them to new code.
- **Checked-build shadow stack grew** from 35248 to 36232 bytes with `Ix` (release shrank from
  36280 to 32432). Not investigated; the 48 KiB setting still leaves about 12 KB.

## Both crates (stylistic, left as is)

- Comment density: minisilk has about 1400 `/* */` comments to minicelt's 470, both inherited from C.
- `#[inline(always)]` counts differ with the number of kernels (123 vs 61).
- Small helpers duplicated per crate on purpose (`celt_maxabs_res`, the DTX constants,
  `decide_dtx_mode`, the `wasm_f32x4_hsum`/`WASM_F32X4_MADD` kernels helpers): the workspace rule is
  to copy, not share.
- `OpusEncoder` init style (`mem::zeroed()` + `write_bytes` in minisilk, a field-by-field `const fn`
  in minicelt) and the output buffer type (`&mut [u8]` vs `&mut [u8; 1276]`).
- Module visibility (`pub mod celt`/`silk` in minisilk, private `celt` in minicelt).
- minisilk's `min`/`max` on floats are hand-written C ternaries; minicelt has `arch::MIN16` etc.
  Only matters if float min/max is added to minisilk.
