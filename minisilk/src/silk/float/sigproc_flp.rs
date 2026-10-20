//! Port of the inline helpers of silk/float/SigProc_FLP.h, plus `float2int` from
//! celt/float_cast.h (wasm path: `__builtin_rintf`, i.e. round half to even, saturating).


use crate::silk::macros::silk_SAT16;

pub const PI: f32 = 3.1415926536;

/// celt/float_cast.h `float2int` for wasm with nontrapping-fptoint.
#[inline(always)]
pub fn float2int(x: f32) -> i32 {
    libm::rintf(x) as i32
}

/// Correctly rounded `sqrt` (C `sqrt()`), as the wasm `f64.sqrt` instruction (libm's is a software
/// implementation).
#[inline(always)]
pub fn sqrt(x: f64) -> f64 {
    core::arch::wasm32::f64x2_extract_lane::<0>(core::arch::wasm32::f64x2_sqrt(core::arch::wasm32::f64x2_splat(x)))
}

/// sigmoid function
#[inline(always)]
pub fn silk_sigmoid(x: f32) -> f32 {
    (1.0 / (1.0 + libm::exp(-x as f64))) as f32
}

/// floating-point to integer conversion (rounding)
#[inline(always)]
pub fn silk_float2int(x: f32) -> i32 {
    float2int(x)
}

/// floating-point to integer conversion (rounding)
#[inline(always)]
pub fn silk_float2short_array(out: &mut [i16], in_: &[f32], length: i32) {
    let mut k = length - 1;
    while k >= 0 {
        out[k as usize] = silk_SAT16(float2int(in_[k as usize])) as i16;
        k -= 1;
    }
}

/// integer to floating-point conversion
#[inline(always)]
pub fn silk_short2float_array(out: &mut [f32], in_: &[i16], length: i32) {
    let mut k = length - 1;
    while k >= 0 {
        out[k as usize] = in_[k as usize] as f32;
        k -= 1;
    }
}

/// using log2() helps the fixed-point conversion
#[inline(always)]
pub fn silk_log2(x: f64) -> f32 {
    (3.32192809488736 * libm::log10(x)) as f32
}

#[inline(always)]
pub fn silk_abs_float(a: f32) -> f32 {
    a.abs()
}
