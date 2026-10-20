//! Port of celt/arch.h (float build).
//!
//! The float build's arithmetic macros (`MULT16_32_Q15(a,b)` = `a*b`, `SHR32(a,s)` = `a`,
//! `QCONST16(x,b)` = `x`, `ADD32`, `EXTEND32`, `HALF32(x)` = `.5f*x`, ...) are written out as
//! plain f32 expressions at the call site. Only the macros whose semantics differ from the
//! obvious Rust spelling (C ternary min/max, which are not `f32::min`/`max` for NaN and signed
//! zeros) are functions here. The float types (`opus_val16`, `celt_sig`, `celt_norm`, ...) are
//! all `f32`.

use core::arch::wasm32::{f64x2_extract_lane, f64x2_floor, f64x2_splat, f64x2_sqrt};

pub const CELT_SIG_SCALE: f32 = 32768.0;

pub const Q31ONE: f32 = 1.0;
pub const COEF_ONE: f32 = 1.0;
pub const EPSILON: f32 = 1e-15;
pub const VERY_SMALL: f32 = 1e-30;

/// MIN16/MIN32/MING/FMIN: `a < b ? a : b`
#[inline(always)]
pub fn MIN16(a: f32, b: f32) -> f32 {
    if a < b { a } else { b }
}
/// MAX16/MAX32/MAXG/FMAX: `a > b ? a : b`
#[inline(always)]
pub fn MAX16(a: f32, b: f32) -> f32 {
    if a > b { a } else { b }
}
#[inline(always)]
pub fn MIN32(a: f32, b: f32) -> f32 {
    if a < b { a } else { b }
}
#[inline(always)]
pub fn MAX32(a: f32, b: f32) -> f32 {
    if a > b { a } else { b }
}
#[inline(always)]
pub fn IMIN(a: i32, b: i32) -> i32 {
    if a < b { a } else { b }
}
#[inline(always)]
pub fn IMAX(a: i32, b: i32) -> i32 {
    if a > b { a } else { b }
}

/// celt_isnan (non-FLOAT_APPROX): `x != x`
#[inline(always)]
pub fn celt_isnan(x: f32) -> bool {
    x != x
}

/// ABS16/ABS32: `(float)fabs(x)`
#[inline(always)]
pub fn ABS16(x: f32) -> f32 {
    x.abs()
}
#[inline(always)]
pub fn ABS32(x: f32) -> f32 {
    x.abs()
}

/// `sqrt(x)` (double): the single wasm instruction (correctly rounded, like libm's).
#[inline(always)]
pub fn sqrt64(x: f64) -> f64 {
    f64x2_extract_lane::<0>(f64x2_sqrt(f64x2_splat(x)))
}

/// `floor(x)` (double): the single wasm instruction.
#[inline(always)]
pub fn floor64(x: f64) -> f64 {
    f64x2_extract_lane::<0>(f64x2_floor(f64x2_splat(x)))
}

/// Indexing and range slicing as in C: no bounds check (`at`/`at_mut` replace `x[i]`,
/// `rg`/`rg_mut` replace `x[a..b]`, `&x[a..]`, ...). With debug assertions on (the `checked`
/// profile that the tests run) they are the ordinary bounds-checked operations, so an index that
/// is out of range traps there. The callers' indices are in range by construction, as in the C
/// code.
///
/// Use it only where the compiler cannot be given a proof: indices read out of tables or computed
/// from coded values, caller-sized slices indexed at offsets from the band table, and per-sample
/// loops where an iterator or re-slice form measured slower. Everywhere else use plain indexing,
/// iterators or re-slices, which are checked in every profile. Every function that still uses `Ix`
/// carries an `IX-HARD` comment with the reason (and the measured cost where a conversion was tried).
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
        if cfg!(debug_assertions) { &self[r] } else { unsafe { self.get_unchecked(r) } }
    }
    #[inline(always)]
    fn rg_mut<R: core::slice::SliceIndex<[T], Output = [T]>>(&mut self, r: R) -> &mut [T] {
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
