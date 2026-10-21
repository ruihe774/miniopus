//! Shared wasm SIMD helpers. Every `cfg(target_feature = "relaxed-simd")` choice between a relaxed
//! and a plain instruction lives here, so kernels call these instead of selecting by hand.

use core::arch::wasm32::*;

/// Multiply-add a*b+c. With relaxed SIMD this may be fused (single rounding), which is fine for
/// the float kernels: like the SSE/NEON versions they do not match the C summation order anyway.
#[cfg(target_feature = "relaxed-simd")]
#[inline(always)]
pub fn madd(a: v128, b: v128, c: v128) -> v128 {
    f32x4_relaxed_madd(a, b, c)
}

#[cfg(not(target_feature = "relaxed-simd"))]
#[inline(always)]
pub fn madd(a: v128, b: v128, c: v128) -> v128 {
    f32x4_add(f32x4_mul(a, b), c)
}

/// Lane select: `a` where `m` is set, else `b`. With relaxed SIMD this is the single-instruction
/// lane select (e.g. blendvps on x86); every mask passed to it must be a comparison result, i.e.
/// all-ones or all-zeros per lane, for which relaxed_laneselect is fully defined.
#[cfg(target_feature = "relaxed-simd")]
#[inline(always)]
pub fn laneselect(a: v128, b: v128, m: v128) -> v128 {
    i32x4_relaxed_laneselect(a, b, m)
}

#[cfg(not(target_feature = "relaxed-simd"))]
#[inline(always)]
pub fn laneselect(a: v128, b: v128, m: v128) -> v128 {
    v128_bitselect(a, b, m)
}

/// f32x4 -> i32x4 truncation. The caller guarantees every lane is finite and in i32 range (not NaN),
/// where the relaxed truncation is exact too.
#[cfg(target_feature = "relaxed-simd")]
#[inline(always)]
pub fn trunc_i32(a: v128) -> v128 {
    i32x4_relaxed_trunc_f32x4(a)
}

#[cfg(not(target_feature = "relaxed-simd"))]
#[inline(always)]
pub fn trunc_i32(a: v128) -> v128 {
    i32x4_trunc_sat_f32x4(a)
}

/// Lane-wise maximum. With relaxed SIMD the result is implementation-defined when a lane is NaN or
/// the operands are zeros of opposite sign (one native `maxps`/`fmax`); otherwise it is the plain
/// `a < b ? b : a`, which is what the plain `f32x4_pmax` computes. The caller guarantees no NaN, and
/// that a +-0 tie cannot change its result.
#[cfg(target_feature = "relaxed-simd")]
#[inline(always)]
pub fn max4(a: v128, b: v128) -> v128 {
    f32x4_relaxed_max(a, b)
}

#[cfg(not(target_feature = "relaxed-simd"))]
#[inline(always)]
pub fn max4(a: v128, b: v128) -> v128 {
    f32x4_pmax(a, b)
}

/// Lane-wise minimum, the counterpart of [`max4`] (plain: `f32x4_pmin`, `b < a ? b : a`).
#[cfg(target_feature = "relaxed-simd")]
#[inline(always)]
pub fn min4(a: v128, b: v128) -> v128 {
    f32x4_relaxed_min(a, b)
}

#[cfg(not(target_feature = "relaxed-simd"))]
#[inline(always)]
pub fn min4(a: v128, b: v128) -> v128 {
    f32x4_pmin(a, b)
}

/// Unaligned 16-byte load (four `f32` or `i32`). The caller guarantees `p[0..4]` is readable.
// UNSAFE-MAND: `v128_load` is an unsafe intrinsic (a safe load would need a bounds-checked copy or `f32x4(..)`).
#[inline(always)]
pub unsafe fn load4<T>(p: *const T) -> v128 {
    unsafe { v128_load(p as *const v128) }
}

/// Unaligned 16-byte store (four `f32` or `i32`). The caller guarantees `p[0..4]` is writable.
// UNSAFE-MAND: `v128_store` is an unsafe intrinsic.
#[inline(always)]
pub unsafe fn store4<T>(p: *mut T, v: v128) {
    unsafe { v128_store(p as *mut v128, v) }
}

/// Sums the lanes of a, b, c and d: returns {sum(a), sum(b), sum(c), sum(d)}.
#[inline(always)]
pub fn hadd4(a: v128, b: v128, c: v128, d: v128) -> v128 {
    let ab = f32x4_add(
        i32x4_shuffle::<0, 4, 1, 5>(a, b),
        i32x4_shuffle::<2, 6, 3, 7>(a, b),
    );
    let cd = f32x4_add(
        i32x4_shuffle::<0, 4, 1, 5>(c, d),
        i32x4_shuffle::<2, 6, 3, 7>(c, d),
    );
    f32x4_add(i64x2_shuffle::<0, 2>(ab, cd), i64x2_shuffle::<1, 3>(ab, cd))
}

#[inline(always)]
pub fn hsum_f32(a: v128) -> f32 {
    let a = f32x4_add(a, i64x2_shuffle::<1, 0>(a, a));
    let a = f32x4_add(a, i32x4_shuffle::<1, 0, 3, 2>(a, a));
    f32x4_extract_lane::<0>(a)
}

#[inline(always)]
pub fn hsum_i32(a: v128) -> i32 {
    let a = i32x4_add(a, i64x2_shuffle::<1, 0>(a, a));
    let a = i32x4_add(a, i32x4_shuffle::<1, 0, 3, 2>(a, a));
    i32x4_extract_lane::<0>(a)
}

/// Maximum of the four lanes, in every lane of the result (see [`max4`] for the NaN / +-0 condition).
#[inline(always)]
pub fn hmax4(a: v128) -> v128 {
    let a = max4(a, i64x2_shuffle::<1, 0>(a, a));
    max4(a, i32x4_shuffle::<1, 0, 3, 2>(a, a))
}

#[inline(always)]
pub fn hmin_i32(a: v128) -> i32 {
    let a = i32x4_min(a, i64x2_shuffle::<1, 0>(a, a));
    let a = i32x4_min(a, i32x4_shuffle::<1, 0, 3, 2>(a, a));
    i32x4_extract_lane::<0>(a)
}
