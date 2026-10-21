//! Port of the fixed-point macro layer: silk/macros.h, the macro part of silk/SigProc_FIX.h,
//! and silk/Inlines.h.
//!
//! Conventions:
//! - C `opus_int`/`int` arguments are `i32`. Callers do the C integer promotions explicitly
//!   (e.g. `silk_SMULBB(a as i32, b as i32)` for `opus_int16` operands).
//! - Plain C arithmetic (`silk_ADD32`, `silk_MUL`, ...) uses plain Rust operators, so debug
//!   builds trap where C would hit signed overflow. The `_ovflw` variants wrap.
//! - The SMULWx/SMLAWx macros use the `OPUS_FAST_INT64` (64-bit) formulations. C's
//!   wasm32 build picks the 32-bit ones (arch.h keys the choice on LP64), but wasm has
//!   native i64 arithmetic and both formulations give identical results, including on
//!   wraparound; the 64-bit ones also avoid an intermediate i32 overflow.
//! - Type-generic C macros (`silk_min`, `silk_max`, `silk_LIMIT`, `silk_abs`, `silk_sign`) are
//!   `macro_rules!` macros; everything else is an `#[inline(always)]` function.


/* Indexing (no C counterpart; lives here because every SILK module already imports `macros::*`) */

/// Indexing and range slicing as in C: no bounds check (`at`/`at_mut` replace `x[i]`,
/// `rg`/`rg_mut` replace `x[a..b]`, `&x[a..]`, ...). With debug assertions on (the `checked`
/// profile that the tests run) they are the ordinary bounds-checked operations, so an index that
/// is out of range traps there. The callers' indices are in range by construction, as in the C
/// code; the release build thereby drops the panic sites and their source locations.
/// (Same trait as minicelt's `arch::Ix`; minisilk has no `arch` module.)
pub trait Ix<T: Copy> {
    fn at(&self, i: usize) -> T;

    fn at_mut(&mut self, i: usize) -> &mut T;

    fn rg<R: core::slice::SliceIndex<[T], Output = [T]>>(&self, r: R) -> &[T];

    fn rg_mut<R: core::slice::SliceIndex<[T], Output = [T]>>(&mut self, r: R) -> &mut [T];
}

impl<T: Copy> Ix<T> for [T] {
    #[inline(always)]
    fn at(&self, i: usize) -> T {
        if cfg!(debug_assertions) {
            self[i]
        } else {
            // SAFETY: see the trait documentation.
            unsafe { *self.get_unchecked(i) }
        }
    }

    #[inline(always)]
    fn rg<R: core::slice::SliceIndex<[T], Output = [T]>>(&self, r: R) -> &[T] {
        // SAFETY: see the trait documentation.
        if cfg!(debug_assertions) { &self[r] } else { unsafe { self.get_unchecked(r) } }
    }

    #[inline(always)]
    fn rg_mut<R: core::slice::SliceIndex<[T], Output = [T]>>(&mut self, r: R) -> &mut [T] {
        // SAFETY: see the trait documentation.
        if cfg!(debug_assertions) { &mut self[r] } else { unsafe { self.get_unchecked_mut(r) } }
    }

    #[inline(always)]
    fn at_mut(&mut self, i: usize) -> &mut T {
        if cfg!(debug_assertions) {
            &mut self[i]
        } else {
            // SAFETY: see the trait documentation.
            unsafe { self.get_unchecked_mut(i) }
        }
    }
}

pub const silk_int32_MAX: i32 = i32::MAX;
pub const silk_int32_MIN: i32 = i32::MIN;
pub const silk_int16_MAX: i32 = i16::MAX as i32;
pub const silk_int16_MIN: i32 = i16::MIN as i32;
pub const silk_uint8_MAX: i32 = u8::MAX as i32;
pub const silk_float_MAX: f32 = f32::MAX;

pub const RAND_MULTIPLIER: i32 = 196314165;
pub const RAND_INCREMENT: i32 = 907633515;

/* silk/macros.h */

/// (a32 * (opus_int32)((opus_int16)(b32))) >> 16
#[inline(always)]
pub fn silk_SMULWB(a32: i32, b32: i32) -> i32 {
    ((a32 as i64 * (b32 as i16 as i64)) >> 16) as i32
}

/// a32 + (b32 * (opus_int32)((opus_int16)(c32))) >> 16
#[inline(always)]
pub fn silk_SMLAWB(a32: i32, b32: i32, c32: i32) -> i32 {
    (a32 as i64 + ((b32 as i64 * (c32 as i16 as i64)) >> 16)) as i32
}

/// a32 + (b32 * (c32 >> 16)) >> 16
#[inline(always)]
pub fn silk_SMLAWT(a32: i32, b32: i32, c32: i32) -> i32 {
    (a32 as i64 + ((b32 as i64 * (c32 as i64 >> 16)) >> 16)) as i32
}

/// (opus_int32)((opus_int16)(a32)) * (opus_int32)((opus_int16)(b32))
#[inline(always)]
pub fn silk_SMULBB(a32: i32, b32: i32) -> i32 {
    (a32 as i16 as i32) * (b32 as i16 as i32)
}

/// a32 + (opus_int32)((opus_int16)(b32)) * (opus_int32)((opus_int16)(c32))
#[inline(always)]
pub fn silk_SMLABB(a32: i32, b32: i32, c32: i32) -> i32 {
    a32 + (b32 as i16 as i32) * (c32 as i16 as i32)
}

/// (a32 * b32) >> 16
#[inline(always)]
pub fn silk_SMULWW(a32: i32, b32: i32) -> i32 {
    ((a32 as i64 * b32 as i64) >> 16) as i32
}

/// a32 + ((b32 * c32) >> 16)
#[inline(always)]
pub fn silk_SMLAWW(a32: i32, b32: i32, c32: i32) -> i32 {
    (a32 as i64 + ((b32 as i64 * c32 as i64) >> 16)) as i32
}

/// add with output saturated
#[inline(always)]
pub fn silk_ADD_SAT32(a: i32, b: i32) -> i32 {
    a.saturating_add(b)
}

/// subtract with output saturated
#[inline(always)]
pub fn silk_SUB_SAT32(a: i32, b: i32) -> i32 {
    a.saturating_sub(b)
}

#[inline(always)]
pub fn silk_CLZ32(in32: i32) -> i32 {
    (in32 as u32).leading_zeros() as i32
}

/* silk/SigProc_FIX.h */

/// Rotate a32 right by 'rot' bits. Negative rot values result in rotating left.
#[inline(always)]
pub fn silk_ROR32(a32: i32, rot: i32) -> i32 {
    let x = a32 as u32;
    let r = rot as u32;
    let m = (rot as u32).wrapping_neg();
    if rot == 0 {
        a32
    } else if rot < 0 {
        ((x << m) | (x >> (32 - m))) as i32
    } else {
        ((x << (32 - r)) | (x >> r)) as i32
    }
}

/// (a32 * b32) output have to be 32bit int
#[inline(always)]
pub fn silk_MUL(a32: i32, b32: i32) -> i32 {
    a32 * b32
}

/// a32 + (b32 * c32) output have to be 32bit int
#[inline(always)]
pub fn silk_MLA(a32: i32, b32: i32, c32: i32) -> i32 {
    silk_ADD32(a32, b32 * c32)
}

/// (a32 * b32)
#[inline(always)]
pub fn silk_SMULL(a32: i32, b32: i32) -> i64 {
    (a32 as i64) * (b32 as i64)
}

/// Adds two signed 32-bit values in a way that can overflow
#[inline(always)]
pub fn silk_ADD32_ovflw(a: i32, b: i32) -> i32 {
    a.wrapping_add(b)
}

/// Subtracts two signed 32-bit values in a way that can overflow
#[inline(always)]
pub fn silk_SUB32_ovflw(a: i32, b: i32) -> i32 {
    a.wrapping_sub(b)
}

/// Multiply-accumulate that allows overflow in the addition
#[inline(always)]
pub fn silk_MLA_ovflw(a32: i32, b32: i32, c32: i32) -> i32 {
    silk_ADD32_ovflw(a32, (b32 as u32).wrapping_mul(c32 as u32) as i32)
}

#[inline(always)]
pub fn silk_SMLABB_ovflw(a32: i32, b32: i32, c32: i32) -> i32 {
    silk_ADD32_ovflw(a32, (b32 as i16 as i32) * (c32 as i16 as i32))
}

#[inline(always)]
pub fn silk_DIV32_16(a32: i32, b16: i32) -> i32 {
    a32 / b16
}

#[inline(always)]
pub fn silk_DIV32(a32: i32, b32: i32) -> i32 {
    a32 / b32
}

#[inline(always)]
pub fn silk_ADD32(a: i32, b: i32) -> i32 {
    a + b
}

#[inline(always)]
pub fn silk_SUB32(a: i32, b: i32) -> i32 {
    a - b
}

#[inline(always)]
pub fn silk_SAT16(a: i32) -> i32 {
    a.clamp(silk_int16_MIN, silk_int16_MAX)
}

#[inline(always)]
pub fn silk_ADD_SAT16(a: i32, b: i32) -> i16 {
    silk_SAT16(silk_ADD32(a, b)) as i16
}

#[inline(always)]
pub fn silk_ADD_POS_SAT32(a: i32, b: i32) -> i32 {
    let s = (a as u32).wrapping_add(b as u32);
    if s & 0x80000000 != 0 { silk_int32_MAX } else { s as i32 }
}

/// shift >= 0, shift < 16
#[inline(always)]
pub fn silk_LSHIFT16(a: i32, shift: i32) -> i16 {
    ((a as u16) << shift) as i16
}

/// shift >= 0, shift < 32
#[inline(always)]
pub fn silk_LSHIFT32(a: i32, shift: i32) -> i32 {
    ((a as u32) << shift) as i32
}

/// shift >= 0, shift < 32
#[inline(always)]
pub fn silk_LSHIFT(a: i32, shift: i32) -> i32 {
    silk_LSHIFT32(a, shift)
}

/// shift >= 0, shift < 32
#[inline(always)]
pub fn silk_RSHIFT32(a: i32, shift: i32) -> i32 {
    a >> shift
}

/// shift >= 0, shift < 64
#[inline(always)]
pub fn silk_RSHIFT64(a: i64, shift: i32) -> i64 {
    a >> shift
}

/// shift >= 0, shift < 32
#[inline(always)]
pub fn silk_RSHIFT(a: i32, shift: i32) -> i32 {
    a >> shift
}

/// saturates before shifting
#[inline(always)]
pub fn silk_LSHIFT_SAT32(a: i32, shift: i32) -> i32 {
    silk_LSHIFT32(silk_LIMIT!(a, silk_RSHIFT32(silk_int32_MIN, shift), silk_RSHIFT32(silk_int32_MAX, shift)), shift)
}

/// shift >= 0, allowed to overflow
#[inline(always)]
pub fn silk_LSHIFT_ovflw(a: i32, shift: i32) -> i32 {
    ((a as u32) << shift) as i32
}

#[inline(always)]
pub fn silk_ADD_LSHIFT(a: i32, b: i32, shift: i32) -> i32 {
    a + silk_LSHIFT(b, shift)
}

#[inline(always)]
pub fn silk_ADD_LSHIFT32(a: i32, b: i32, shift: i32) -> i32 {
    silk_ADD32(a, silk_LSHIFT32(b, shift))
}

#[inline(always)]
pub fn silk_ADD_RSHIFT(a: i32, b: i32, shift: i32) -> i32 {
    a + silk_RSHIFT(b, shift)
}

#[inline(always)]
pub fn silk_ADD_RSHIFT32(a: i32, b: i32, shift: i32) -> i32 {
    silk_ADD32(a, silk_RSHIFT32(b, shift))
}

#[inline(always)]
pub fn silk_SUB_LSHIFT32(a: i32, b: i32, shift: i32) -> i32 {
    silk_SUB32(a, silk_LSHIFT32(b, shift))
}

#[inline(always)]
pub fn silk_SUB_RSHIFT32(a: i32, b: i32, shift: i32) -> i32 {
    silk_SUB32(a, silk_RSHIFT32(b, shift))
}

/// Requires that shift > 0
#[inline(always)]
pub fn silk_RSHIFT_ROUND(a: i32, shift: i32) -> i32 {
    if shift == 1 { (a >> 1) + (a & 1) } else { ((a >> (shift - 1)) + 1) >> 1 }
}

/// Requires that shift > 0
#[inline(always)]
pub fn silk_RSHIFT_ROUND64(a: i64, shift: i32) -> i64 {
    if shift == 1 { (a >> 1) + (a & 1) } else { ((a >> (shift - 1)) + 1) >> 1 }
}

/// Macro to convert floating-point constants to fixed-point.
/// For a C `float` literal (`0.1f`), pass `0.1f32 as f64` so the constant is rounded as in C.
#[inline(always)]
pub const fn SILK_FIX_CONST(c: f64, q: i32) -> i32 {
    (c * ((1i64 << q) as f64) + 0.5) as i32
}

#[inline(always)]
pub fn silk_min_int(a: i32, b: i32) -> i32 {
    if a < b { a } else { b }
}

#[inline(always)]
pub fn silk_min_32(a: i32, b: i32) -> i32 {
    if a < b { a } else { b }
}

#[inline(always)]
pub fn silk_max_int(a: i32, b: i32) -> i32 {
    if a > b { a } else { b }
}

#[inline(always)]
pub fn silk_max_32(a: i32, b: i32) -> i32 {
    if a > b { a } else { b }
}

/// PSEUDO-RANDOM GENERATOR. Store the result as the seed for the next call.
#[inline(always)]
pub fn silk_RAND(seed: i32) -> i32 {
    silk_MLA_ovflw(RAND_INCREMENT, seed, RAND_MULTIPLIER)
}

/// Signed top word multiply.
#[inline(always)]
pub fn silk_SMMUL(a32: i32, b32: i32) -> i32 {
    silk_RSHIFT64(silk_SMULL(a32, b32), 32) as i32
}

/* Type-generic macros */

#[macro_export]
macro_rules! silk_min {
    ($a:expr, $b:expr) => {{
        let (a, b) = ($a, $b);
        if a < b { a } else { b }
    }};
}

#[macro_export]
macro_rules! silk_max {
    ($a:expr, $b:expr) => {{
        let (a, b) = ($a, $b);
        if a > b { a } else { b }
    }};
}

/// Clamps `a` between the two limits, in whichever order they are given.
#[macro_export]
macro_rules! silk_LIMIT {
    ($a:expr, $limit1:expr, $limit2:expr) => {{
        let (a, l1, l2) = ($a, $limit1, $limit2);
        if l1 > l2 {
            if a > l1 { l1 } else if a < l2 { l2 } else { a }
        } else if a > l2 {
            l2
        } else if a < l1 {
            l1
        } else {
            a
        }
    }};
}

/// Be careful, silk_abs returns wrong when input equals to silk_intXX_MIN
#[macro_export]
macro_rules! silk_abs {
    ($a:expr) => {{
        let a = $a;
        if a > 0 { a } else { -a }
    }};
}

#[macro_export]
macro_rules! silk_sign {
    ($a:expr) => {{
        let a = $a;
        if a > 0 { 1 } else if a < 0 { -1 } else { 0 }
    }};
}

pub use crate::{silk_LIMIT as silk_LIMIT_int, silk_LIMIT as silk_LIMIT_16, silk_LIMIT as silk_LIMIT_32};
pub use crate::{silk_LIMIT, silk_abs, silk_max, silk_min, silk_sign};

/* silk/Inlines.h */

/// get number of leading zeros and fractional part (the bits right after the leading one)
#[inline(always)]
pub fn silk_CLZ_FRAC(in_: i32, lz: &mut i32, frac_Q7: &mut i32) {
    let lzeros = silk_CLZ32(in_);
    *lz = lzeros;
    *frac_Q7 = silk_ROR32(in_, 24 - lzeros) & 0x7f;
}

/// Approximation of square root.
/// Accuracy: < +/- 10% for output values > 15, < +/- 2.5% for output values > 120
#[inline(always)]
pub fn silk_SQRT_APPROX(x: i32) -> i32 {
    let (mut lz, mut frac_Q7) = (0, 0);
    if x <= 0 {
        return 0;
    }
    silk_CLZ_FRAC(x, &mut lz, &mut frac_Q7);
    let mut y: i32 = if lz & 1 != 0 { 32768 } else { 46214 }; /* 46214 = sqrt(2) * 32768 */
    /* get scaling right */
    y >>= silk_RSHIFT(lz, 1);
    /* increment using fractional part of input */
    silk_SMLAWB(y, y, silk_SMULBB(213, frac_Q7))
}

/// Divide two int32 values and return result as int32 in a given Q-domain.
/// Returns a good approximation of "(a32 << Qres) / b32".
#[inline(always)]
pub fn silk_DIV32_varQ(a32: i32, b32: i32, Qres: i32) -> i32 {
    debug_assert!(b32 != 0);
    debug_assert!(Qres >= 0);

    /* Compute number of bits head room and normalize inputs */
    let a_headrm = silk_CLZ32(silk_abs!(a32)) - 1;
    let mut a32_nrm = silk_LSHIFT(a32, a_headrm); /* Q: a_headrm                  */
    let b_headrm = silk_CLZ32(silk_abs!(b32)) - 1;
    let b32_nrm = silk_LSHIFT(b32, b_headrm); /* Q: b_headrm                  */

    /* Inverse of b32, with 14 bits of precision */
    let b32_inv = silk_DIV32_16(silk_int32_MAX >> 2, silk_RSHIFT(b32_nrm, 16)); /* Q: 29 + 16 - b_headrm        */

    /* First approximation */
    let mut result = silk_SMULWB(a32_nrm, b32_inv); /* Q: 29 + a_headrm - b_headrm  */

    /* Compute residual by subtracting product of denominator and first approximation */
    /* It's OK to overflow because the final value of a32_nrm should always be small */
    a32_nrm = silk_SUB32_ovflw(a32_nrm, silk_LSHIFT_ovflw(silk_SMMUL(b32_nrm, result), 3)); /* Q: a_headrm   */

    /* Refinement */
    result = silk_SMLAWB(result, a32_nrm, b32_inv); /* Q: 29 + a_headrm - b_headrm  */

    /* Convert to Qres domain */
    let lshift = 29 + a_headrm - b_headrm - Qres;
    if lshift < 0 {
        silk_LSHIFT_SAT32(result, -lshift)
    } else if lshift < 32 {
        silk_RSHIFT(result, lshift)
    } else {
        /* Avoid undefined result */
        0
    }
}

/// Invert int32 value and return result as int32 in a given Q-domain.
/// Returns a good approximation of "(1 << Qres) / b32".
#[inline(always)]
pub fn silk_INVERSE32_varQ(b32: i32, Qres: i32) -> i32 {
    debug_assert!(b32 != 0);
    debug_assert!(Qres > 0);

    /* Compute number of bits head room and normalize input */
    let b_headrm = silk_CLZ32(silk_abs!(b32)) - 1;
    let b32_nrm = silk_LSHIFT(b32, b_headrm); /* Q: b_headrm                */

    /* Inverse of b32, with 14 bits of precision */
    let b32_inv = silk_DIV32_16(silk_int32_MAX >> 2, silk_RSHIFT(b32_nrm, 16)); /* Q: 29 + 16 - b_headrm    */

    /* First approximation */
    let mut result = silk_LSHIFT(b32_inv, 16); /* Q: 61 - b_headrm            */

    /* Compute residual by subtracting product of denominator and first approximation from one */
    let err_Q32 = silk_LSHIFT((1i32 << 29) - silk_SMULWB(b32_nrm, b32_inv), 3); /* Q32                        */

    /* Refinement */
    result = silk_SMLAWW(result, err_Q32, b32_inv); /* Q: 61 - b_headrm            */

    /* Convert to Qres domain */
    let lshift = 61 - b_headrm - Qres;
    if lshift <= 0 {
        silk_LSHIFT_SAT32(result, -lshift)
    } else if lshift < 32 {
        silk_RSHIFT(result, lshift)
    } else {
        /* Avoid undefined result */
        0
    }
}
