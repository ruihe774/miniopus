//! Port of celt/wasm/pitch_wasm.c (float build): `celt_pitch_xcorr` and `celt_inner_prod`,
//! the only pitch kernels SILK uses (opus_val16 = opus_val32 = f32).

use core::arch::wasm32::*;

/* Multiply-add a*b+c. With relaxed SIMD this may be fused (single rounding),
   which is fine for these float kernels: like the SSE/NEON versions they do
   not match the C summation order anyway. */
#[cfg(target_feature = "relaxed-simd")]
#[inline(always)]
fn WASM_F32X4_MADD(a: v128, b: v128, c: v128) -> v128 {
    f32x4_relaxed_madd(a, b, c)
}

#[cfg(not(target_feature = "relaxed-simd"))]
#[inline(always)]
fn WASM_F32X4_MADD(a: v128, b: v128, c: v128) -> v128 {
    f32x4_add(f32x4_mul(a, b), c)
}

/// Unaligned 4-float load. The caller guarantees `p[0..4]` is readable.
#[inline(always)]
unsafe fn ld(p: *const f32) -> v128 {
    unsafe { v128_load(p as *const v128) }
}

/* Sums the lanes of a, b, c and d: returns {sum(a), sum(b), sum(c), sum(d)}. */
#[inline(always)]
fn wasm_f32x4_hadd4(a: v128, b: v128, c: v128, d: v128) -> v128 {
    let ab = f32x4_add(i32x4_shuffle::<0, 4, 1, 5>(a, b), i32x4_shuffle::<2, 6, 3, 7>(a, b));
    let cd = f32x4_add(i32x4_shuffle::<0, 4, 1, 5>(c, d), i32x4_shuffle::<2, 6, 3, 7>(c, d));
    f32x4_add(i64x2_shuffle::<0, 2>(ab, cd), i64x2_shuffle::<1, 3>(ab, cd))
}

#[inline(always)]
fn wasm_f32x4_hsum(a: v128) -> f32 {
    let a = f32x4_add(a, i64x2_shuffle::<1, 0>(a, a));
    let a = f32x4_add(a, i32x4_shuffle::<1, 0, 3, 2>(a, a));
    f32x4_extract_lane::<0>(a)
}

pub fn celt_inner_prod(x: &[f32], y: &[f32], N: i32) -> f32 {
    let n = N.max(0) as usize;
    debug_assert!(x.len() >= n && y.len() >= n);
    let (px, py) = (x.as_ptr(), y.as_ptr());
    // SAFETY: every vector load reads 4 floats at an index i with i + 4 <= n.
    unsafe {
        let mut i = 0usize;
        let mut sum0 = f32x4_splat(0.0);
        let mut sum1 = f32x4_splat(0.0);
        while i + 7 < n {
            sum0 = WASM_F32X4_MADD(ld(px.add(i)), ld(py.add(i)), sum0);
            sum1 = WASM_F32X4_MADD(ld(px.add(i + 4)), ld(py.add(i + 4)), sum1);
            i += 8;
        }
        if i + 3 < n {
            sum0 = WASM_F32X4_MADD(ld(px.add(i)), ld(py.add(i)), sum0);
            i += 4;
        }
        let mut xy = wasm_f32x4_hsum(f32x4_add(sum0, sum1));
        while i < n {
            xy = xy + x[i] * y[i];
            i += 1;
        }
        xy
    }
}

/* Returns {sum_j x[j]*y[j+k], k=0..3}. Reads y[0..len+2]. */
#[inline(always)]
unsafe fn xcorr4_float(x: *const f32, y: *const f32, len: usize) -> v128 {
    // SAFETY: the caller guarantees x[0..len] and y[0..len + 3] are readable. The vector loop
    // loads x[j..j + 4] and y[j + k..j + k + 4] (k <= 3) with j + 3 < len, so the last element is
    // y[len + 2]; the tail loop loads x[j] and y[j..j + 4] with j < len, the same bound.
    unsafe {
        let mut sum0 = f32x4_splat(0.0);
        let mut sum1 = sum0;
        let mut sum2 = sum0;
        let mut sum3 = sum0;
        /* Same lag-per-lane layout as the C kernel and SSE port: each x[j] is
        broadcast and multiplied with y[j..j+3]. */
        let mut j = 0usize;
        while j + 3 < len {
            let xj = ld(x.add(j));
            sum0 = WASM_F32X4_MADD(i32x4_shuffle::<0, 0, 0, 0>(xj, xj), ld(y.add(j)), sum0);
            sum1 = WASM_F32X4_MADD(i32x4_shuffle::<1, 1, 1, 1>(xj, xj), ld(y.add(j + 1)), sum1);
            sum2 = WASM_F32X4_MADD(i32x4_shuffle::<2, 2, 2, 2>(xj, xj), ld(y.add(j + 2)), sum2);
            sum3 = WASM_F32X4_MADD(i32x4_shuffle::<3, 3, 3, 3>(xj, xj), ld(y.add(j + 3)), sum3);
            j += 4;
        }
        let mut sum = f32x4_add(f32x4_add(sum0, sum1), f32x4_add(sum2, sum3));
        while j < len {
            sum = WASM_F32X4_MADD(f32x4_splat(*x.add(j)), ld(y.add(j)), sum);
            j += 1;
        }
        sum
    }
}

/// `xcorr[i] = sum_j _x[j] * _y[i + j]` for i in 0..max_pitch. Reads `_y[0..len + max_pitch - 1]`.
pub fn celt_pitch_xcorr(_x: &[f32], _y: &[f32], xcorr: &mut [f32], len: i32, max_pitch: i32) {
    debug_assert!(max_pitch > 0);
    let (len, max_pitch) = (len.max(0) as usize, max_pitch.max(0) as usize);
    debug_assert!(_x.len() >= len && xcorr.len() >= max_pitch);
    debug_assert!(_y.len() + 1 >= len + max_pitch);
    let (px, py, po) = (_x.as_ptr(), _y.as_ptr(), xcorr.as_mut_ptr());
    // SAFETY: the C kernel's accesses stay within _x[0..len], _y[0..len+max_pitch-1] and
    // xcorr[0..max_pitch]; checked by the asserts above.
    unsafe {
        let mut i = 0usize;
        /* Eight lags at a time, like the AVX2 version: one partial inner product
        per lag (no broadcasts in the inner loop), then a transpose-add. */
        while i + 7 < max_pitch {
            let y = py.add(i);
            let mut sum0 = f32x4_splat(0.0);
            let mut sum1 = sum0;
            let mut sum2 = sum0;
            let mut sum3 = sum0;
            let mut sum4 = sum0;
            let mut sum5 = sum0;
            let mut sum6 = sum0;
            let mut sum7 = sum0;
            let mut j = 0usize;
            while j + 3 < len {
                let xj = ld(px.add(j));
                sum0 = WASM_F32X4_MADD(xj, ld(y.add(j)), sum0);
                sum1 = WASM_F32X4_MADD(xj, ld(y.add(j + 1)), sum1);
                sum2 = WASM_F32X4_MADD(xj, ld(y.add(j + 2)), sum2);
                sum3 = WASM_F32X4_MADD(xj, ld(y.add(j + 3)), sum3);
                sum4 = WASM_F32X4_MADD(xj, ld(y.add(j + 4)), sum4);
                sum5 = WASM_F32X4_MADD(xj, ld(y.add(j + 5)), sum5);
                sum6 = WASM_F32X4_MADD(xj, ld(y.add(j + 6)), sum6);
                sum7 = WASM_F32X4_MADD(xj, ld(y.add(j + 7)), sum7);
                j += 4;
            }
            let mut lo = wasm_f32x4_hadd4(sum0, sum1, sum2, sum3);
            let mut hi = wasm_f32x4_hadd4(sum4, sum5, sum6, sum7);
            while j < len {
                let xj = f32x4_splat(*px.add(j));
                lo = WASM_F32X4_MADD(xj, ld(y.add(j)), lo);
                hi = WASM_F32X4_MADD(xj, ld(y.add(j + 4)), hi);
                j += 1;
            }
            v128_store(po.add(i) as *mut v128, lo);
            v128_store(po.add(i + 4) as *mut v128, hi);
            i += 8;
        }
        if i + 3 < max_pitch {
            v128_store(po.add(i) as *mut v128, xcorr4_float(px, py.add(i), len));
            i += 4;
        }
        /* In case max_pitch isn't a multiple of 4, do non-unrolled version. */
        while i < max_pitch {
            xcorr[i] = celt_inner_prod(_x, &_y[i..], len as i32);
            i += 1;
        }
    }
}
