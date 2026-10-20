//! Port of silk/wasm/LPC_inv_pred_gain_wasm.c.

use core::arch::wasm32::*;

use crate::silk::define::*;
use crate::silk::macros::*;

const QA: i32 = 24;
const A_LIMIT: i32 = SILK_FIX_CONST(0.99975, QA);
/* SILK_FIX_CONST( 1.0f / MAX_PREDICTION_POWER_GAIN, 30 ); the scaling by 2^30 is exact in float */
const MIN_INV_GAIN_Q30: i32 = SILK_FIX_CONST((1.0f32 / MAX_PREDICTION_POWER_GAIN) as f64, 30);

#[inline(always)]
fn MUL32_FRAC_Q(a32: i32, b32: i32, Q: i32) -> i32 {
    silk_RSHIFT_ROUND64(silk_SMULL(a32, b32), Q) as i32
}

/// High 32 bits of the four 64-bit lanes of lo and hi.
#[inline(always)]
fn silk_high32_i64_wasm(lo: v128, hi: v128) -> v128 {
    i32x4_shuffle::<1, 3, 5, 7>(lo, hi)
}

/// MUL32_FRAC_Q( a, b, 31 ) on each lane, for b != silk_int32_MIN.
/// ( p + 2^30 ) >> 31 is taken as the upper half of 2 * p + 2^31, which cannot overflow,
/// since 64-bit arithmetic right shifts are expensive with SIMD128.
#[inline(always)]
fn silk_MUL32_FRAC_Q31_wasm(a: v128, b: v128) -> v128 {
    let rnd = i64x2_splat(1i64 << 31);
    let mut lo = i64x2_extmul_low_i32x4(a, b);
    let mut hi = i64x2_extmul_high_i32x4(a, b);
    lo = i64x2_add(i64x2_add(lo, lo), rnd);
    hi = i64x2_add(i64x2_add(hi, hi), rnd);
    silk_high32_i64_wasm(lo, hi)
}

/// silk_SUB_SAT32( a, b ) on each lane
#[inline(always)]
fn silk_SUB_SAT32_wasm(a: v128, b: v128) -> v128 {
    let r = i32x4_sub(a, b);
    /* All ones in lanes where a and b have different signs and the sign of the result differs from a */
    let ovfl = i32x4_shr(v128_and(v128_xor(a, b), v128_xor(a, r)), 31);
    let sat = v128_xor(i32x4_shr(a, 31), i32x4_splat(silk_int32_MAX));
    #[cfg(target_feature = "relaxed-simd")]
    {
        /* ovfl is all ones or all zeros in each lane, so this is exact */
        i32x4_relaxed_laneselect(sat, r, ovfl)
    }
    #[cfg(not(target_feature = "relaxed-simd"))]
    {
        v128_bitselect(sat, r, ovfl)
    }
}

/// silk_RSHIFT_ROUND64( silk_SMULL( a, b ), shift ) on each lane, for 1 <= shift <= 31.
/// *bad gets all ones in lanes where the result does not fit in 32 bits; otherwise
/// ( p + 2^( shift - 1 ) ) << ( 32 - shift ) does not overflow, and its upper half is the result.
#[inline(always)]
fn silk_SMULL_RSHIFT_ROUND_wasm(a: v128, b: v128, shift: i32, bad: &mut v128) -> v128 {
    let rnd = i64x2_splat(1i64 << (shift - 1));
    let max = i64x2_splat((1i64 << (31 + shift)) - 1);
    let min = i64x2_splat(-(1i64 << (31 + shift)));
    /* |a * b| < 2^62, so adding the rounding offset cannot overflow */
    let mut lo = i64x2_add(i64x2_extmul_low_i32x4(a, b), rnd);
    let mut hi = i64x2_add(i64x2_extmul_high_i32x4(a, b), rnd);
    *bad = v128_or(*bad, v128_or(i64x2_gt(lo, max), i64x2_lt(lo, min)));
    *bad = v128_or(*bad, v128_or(i64x2_gt(hi, max), i64x2_lt(hi, min)));
    lo = i64x2_shl(lo, (32 - shift) as u32);
    hi = i64x2_shl(hi, (32 - shift) as u32);
    silk_high32_i64_wasm(lo, hi)
}

/// Compute inverse of LPC prediction gain, and
/// test if LPC coefficients are stable (all poles within unit circle)
/// Returns inverse prediction gain in energy domain, Q30
fn LPC_inverse_pred_gain_QA_wasm(A_QA: &mut [i32; MAX_LPC_ORDER as usize], order: i32) -> i32 {
    let mut k: i32;
    let mut n: i32;
    let mut invGain_Q30 = SILK_FIX_CONST(1.0, 30);
    k = order - 1;
    while k > 0 {
        /* Check for stability */
        if (A_QA[k as usize] > A_LIMIT) || (A_QA[k as usize] < -A_LIMIT) {
            return 0;
        }

        /* Set RC equal to negated AR coef */
        let rc_Q31 = -silk_LSHIFT(A_QA[k as usize], 31 - QA);

        /* rc_mult1_Q30 range: [ 1 : 2^30 ] */
        let rc_mult1_Q30 = silk_SUB32(SILK_FIX_CONST(1.0, 30), silk_SMMUL(rc_Q31, rc_Q31));
        debug_assert!(rc_mult1_Q30 > (1 << 15)); /* reduce A_LIMIT if fails */
        debug_assert!(rc_mult1_Q30 <= (1 << 30));

        /* Update inverse gain */
        /* invGain_Q30 range: [ 0 : 2^30 ] */
        invGain_Q30 = silk_LSHIFT(silk_SMMUL(invGain_Q30, rc_mult1_Q30), 2);
        debug_assert!(invGain_Q30 >= 0);
        debug_assert!(invGain_Q30 <= (1 << 30));
        if invGain_Q30 < MIN_INV_GAIN_Q30 {
            return 0;
        }

        /* rc_mult2 range: [ 2^30 : silk_int32_MAX ] */
        let mult2Q = 32 - silk_CLZ32(silk_abs!(rc_mult1_Q30));
        let rc_mult2 = silk_INVERSE32_varQ(rc_mult1_Q30, mult2Q + 30);

        /* Update AR coefficient */
        let rc_Q31_v = i32x4_splat(rc_Q31);
        let rc_mult2_v = i32x4_splat(rc_mult2);
        let mut bad = i32x4_splat(0);
        n = 0;
        while n < ((k + 1) >> 1) - 3 {
            /* tmp1 = A_QA[ n + {0,1,2,3} ], tmp2 = A_QA[ k - n - 1 - {0,1,2,3} ].
            When k is odd, the middle coefficient may be in both vectors; both then
            compute and store the same value. */
            // SAFETY: the loop condition gives n + 4 <= (k + 1) >> 1, so p1 reads/writes
            // A_QA[n..n + 4] and p2 reads/writes A_QA[k - n - 4..k - n] with k - n - 4 >=
            // k - ((k + 1) >> 1) >= 0 and k - n <= k < order <= MAX_LPC_ORDER = A_QA.len().
            unsafe {
                let p1 = A_QA.as_mut_ptr().add(n as usize);
                let p2 = A_QA.as_mut_ptr().add((k - n - 4) as usize);
                let t1 = v128_load(p1 as *const v128);
                let mut t2 = v128_load(p2 as *const v128);
                t2 = i32x4_shuffle::<3, 2, 1, 0>(t2, t2);
                let mut u1 = silk_SUB_SAT32_wasm(t1, silk_MUL32_FRAC_Q31_wasm(t2, rc_Q31_v));
                let mut u2 = silk_SUB_SAT32_wasm(t2, silk_MUL32_FRAC_Q31_wasm(t1, rc_Q31_v));
                u1 = silk_SMULL_RSHIFT_ROUND_wasm(u1, rc_mult2_v, mult2Q, &mut bad);
                u2 = silk_SMULL_RSHIFT_ROUND_wasm(u2, rc_mult2_v, mult2Q, &mut bad);
                v128_store(p1 as *mut v128, u1);
                v128_store(p2 as *mut v128, i32x4_shuffle::<3, 2, 1, 0>(u2, u2));
            }
            n += 4;
        }
        /* The C code returns as soon as one coefficient overflows; the result is the same */
        if v128_any_true(bad) {
            return 0;
        }
        while n < (k + 1) >> 1 {
            let tmp1 = A_QA[n as usize];
            let tmp2 = A_QA[(k - n - 1) as usize];
            let mut tmp64 = silk_RSHIFT_ROUND64(
                silk_SMULL(silk_SUB_SAT32(tmp1, MUL32_FRAC_Q(tmp2, rc_Q31, 31)), rc_mult2),
                mult2Q,
            );
            if tmp64 > silk_int32_MAX as i64 || tmp64 < silk_int32_MIN as i64 {
                return 0;
            }
            A_QA[n as usize] = tmp64 as i32;
            tmp64 = silk_RSHIFT_ROUND64(
                silk_SMULL(silk_SUB_SAT32(tmp2, MUL32_FRAC_Q(tmp1, rc_Q31, 31)), rc_mult2),
                mult2Q,
            );
            if tmp64 > silk_int32_MAX as i64 || tmp64 < silk_int32_MIN as i64 {
                return 0;
            }
            A_QA[(k - n - 1) as usize] = tmp64 as i32;
            n += 1;
        }
        k -= 1;
    }

    /* Check for stability */
    if (A_QA[k as usize] > A_LIMIT) || (A_QA[k as usize] < -A_LIMIT) {
        return 0;
    }

    /* Set RC equal to negated AR coef */
    let rc_Q31 = -silk_LSHIFT(A_QA[0], 31 - QA);

    /* Range: [ 1 : 2^30 ] */
    let rc_mult1_Q30 = silk_SUB32(SILK_FIX_CONST(1.0, 30), silk_SMMUL(rc_Q31, rc_Q31));

    /* Update inverse gain */
    /* Range: [ 0 : 2^30 ] */
    invGain_Q30 = silk_LSHIFT(silk_SMMUL(invGain_Q30, rc_mult1_Q30), 2);
    debug_assert!(invGain_Q30 >= 0);
    debug_assert!(invGain_Q30 <= (1 << 30));
    if invGain_Q30 < MIN_INV_GAIN_Q30 {
        return 0;
    }

    invGain_Q30
}

/// For input in Q12 domain. Returns inverse prediction gain in energy domain, Q30
/// (`silk_LPC_inverse_pred_gain_wasm` in C).
pub fn silk_LPC_inverse_pred_gain(A_Q12: &[i16], order: i32) -> i32 {
    debug_assert!(order >= 1 && order <= MAX_LPC_ORDER && A_Q12.len() >= order as usize);
    let mut Atmp_QA = [0i32; MAX_LPC_ORDER as usize];
    let mut k: i32 = 0;

    /* Increase Q domain of the AR coefficients */
    let mut dc = i32x4_splat(0);
    while k < order - 7 {
        // SAFETY: k + 8 <= order <= A_Q12.len() (debug_assert above) and order <= MAX_LPC_ORDER =
        // Atmp_QA.len(), so the 8-lane load from A_Q12 and the two 4-lane stores to Atmp_QA
        // (k..k + 4, k + 4..k + 8) are in range.
        unsafe {
            let a =v128_load(A_Q12.as_ptr().add(k as usize) as *const v128);
            dc = i32x4_add(dc, i32x4_dot_i16x8(a, i16x8_splat(1)));
            v128_store(
                Atmp_QA.as_mut_ptr().add(k as usize) as *mut v128,
                i32x4_shl(i32x4_extend_low_i16x8(a), (QA - 12) as u32),
            );
            v128_store(
                Atmp_QA.as_mut_ptr().add(k as usize + 4) as *mut v128,
                i32x4_shl(i32x4_extend_high_i16x8(a), (QA - 12) as u32),
            );
        }
        k += 8;
    }
    dc = i32x4_add(dc, i32x4_shuffle::<2, 3, 0, 1>(dc, dc));
    dc = i32x4_add(dc, i32x4_shuffle::<1, 0, 3, 2>(dc, dc));
    let mut DC_resp = i32x4_extract_lane::<0>(dc);
    while k < order {
        DC_resp += A_Q12[k as usize] as i32;
        Atmp_QA[k as usize] = silk_LSHIFT32(A_Q12[k as usize] as i32, QA - 12);
        k += 1;
    }
    /* If the DC is unstable, we don't even need to do the full calculations */
    if DC_resp >= 4096 {
        return 0;
    }
    LPC_inverse_pred_gain_QA_wasm(&mut Atmp_QA, order)
}
