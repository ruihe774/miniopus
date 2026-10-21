//! Port of silk/float/wasm/inner_product_FLP_wasm.c.

use core::arch::wasm32::*;

/* acc + a * b in double precision. The product of two floats is exact in double,
   so a fused multiply-add gives exactly the same result as a separate multiply and add. */
#[cfg(target_feature = "relaxed-simd")]
#[inline(always)]
fn MADD_F64X2(a: v128, b: v128, acc: v128) -> v128 {
    f64x2_relaxed_madd(a, b, acc)
}

#[cfg(not(target_feature = "relaxed-simd"))]
#[inline(always)]
fn MADD_F64X2(a: v128, b: v128, acc: v128) -> v128 {
    f64x2_add(acc, f64x2_mul(a, b))
}

/* Promotes the upper two floats of a vector */
#[inline(always)]
fn PROMOTE_HIGH_F32X4(x: v128) -> v128 {
    f64x2_promote_low_f32x4(i32x4_shuffle::<2, 3, 2, 3>(x, x))
}

/// inner product of two silk_float arrays, with result as double
pub fn silk_inner_product_FLP(data1: &[f32], data2: &[f32], dataSize: i32) -> f64 {
    let n = dataSize.max(0) as usize;
    debug_assert!(data1.len() >= n && data2.len() >= n);
    let (p1, p2) = (data1.as_ptr(), data2.as_ptr());
    // SAFETY: every vector load below reads 4 floats at an index i with i + 4 <= n.
    unsafe {
        let ld = |p: *const f32, i: usize| v128_load(p.add(i) as *const v128);
        let mut i = 0usize;
        let mut acc0 = f64x2_splat(0.0);
        let mut acc1 = acc0;
        let mut acc2 = acc0;
        let mut acc3 = acc0;
        while i + 7 < n {
            let x = ld(p1, i);
            let y = ld(p2, i);
            acc0 = MADD_F64X2(f64x2_promote_low_f32x4(x), f64x2_promote_low_f32x4(y), acc0);
            acc1 = MADD_F64X2(PROMOTE_HIGH_F32X4(x), PROMOTE_HIGH_F32X4(y), acc1);
            let x = ld(p1, i + 4);
            let y = ld(p2, i + 4);
            acc2 = MADD_F64X2(f64x2_promote_low_f32x4(x), f64x2_promote_low_f32x4(y), acc2);
            acc3 = MADD_F64X2(PROMOTE_HIGH_F32X4(x), PROMOTE_HIGH_F32X4(y), acc3);
            i += 8;
        }
        if i + 3 < n {
            let x = ld(p1, i);
            let y = ld(p2, i);
            acc0 = MADD_F64X2(f64x2_promote_low_f32x4(x), f64x2_promote_low_f32x4(y), acc0);
            acc1 = MADD_F64X2(PROMOTE_HIGH_F32X4(x), PROMOTE_HIGH_F32X4(y), acc1);
            i += 4;
        }
        acc0 = f64x2_add(f64x2_add(acc0, acc1), f64x2_add(acc2, acc3));
        let mut result = f64x2_extract_lane::<0>(acc0) + f64x2_extract_lane::<1>(acc0);

        /* add any remaining products */
        while i < n {
            result += data1[i] as f64 * data2[i] as f64;
            i += 1;
        }
        result
    }
}
