//! Port of celt/mathops.h and celt/mathops.c (float build, without FLOAT_APPROX).

use core::arch::wasm32::{
    f32x4_add, f32x4_demote_f64x2_zero, f32x4_extract_lane, f32x4_neg, f32x4_pmax, f32x4_pmin,
    f32x4_splat, f64x2_add, f64x2_convert_low_i32x4, f64x2_div, f64x2_extract_lane, f64x2_floor,
    f64x2_mul, f64x2_promote_low_f32x4, f64x2_splat, f64x2_sqrt, f64x2_sub, i32x4_shuffle,
    i32x4_trunc_sat_f32x4, i64x2_add, i64x2_extend_low_i32x4, i64x2_shl, i64x2_shuffle,
    i64x2_splat, v128, v128_load,
};

pub const CELT_SIG_SCALE: f32 = 32768.0;

pub const Q31ONE: f32 = 1.0;
pub const COEF_ONE: f32 = 1.0;
pub const EPSILON: f32 = 1e-15;
pub const VERY_SMALL: f32 = 1e-30;

/// C `a < b ? a : b` and `a > b ? a : b`; `f32::min`/`max` differ for NaN and compile to a helper call.
#[inline(always)]
pub fn fmin(a: f32, b: f32) -> f32 {
    if a < b { a } else { b }
}
#[inline(always)]
pub fn fmax(a: f32, b: f32) -> f32 {
    if a > b { a } else { b }
}
#[inline(always)]
pub fn imin(a: i32, b: i32) -> i32 {
    if a < b { a } else { b }
}
#[inline(always)]
pub fn imax(a: i32, b: i32) -> i32 {
    if a > b { a } else { b }
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

/// PI from celt/mathops.h (a double constant).
pub const PI: f64 = 3.1415926535897931;

/// C celt_maxabs16, celt_maxabs_res. Vectorized: `f32x4_pmax(v, acc)` is `acc > v ? acc : v`, i.e.
/// `fmax(acc, v)` (and `f32x4_pmin(v, acc)` is `fmin(acc, v)`), and the extrema do not depend on the
/// order, so the result is that of the scalar loop for any input without NaN (up to the sign of a
/// zero result, which is only compared).
pub fn maxabs(x: &[f32]) -> f32 {
    let (chunks, tail) = x.as_chunks::<8>();
    let mut max0 = f32x4_splat(0.0);
    let mut max1 = f32x4_splat(0.0);
    let mut min0 = f32x4_splat(0.0);
    let mut min1 = f32x4_splat(0.0);
    for c in chunks {
        // UNSAFE-MAND: `v128_load` is an unsafe intrinsic; `c` is 8 floats.
        let (v0, v1) = unsafe {
            (
                v128_load(c.as_ptr() as *const v128),
                v128_load(c[4..].as_ptr() as *const v128),
            )
        };
        max0 = f32x4_pmax(v0, max0);
        max1 = f32x4_pmax(v1, max1);
        min0 = f32x4_pmin(v0, min0);
        min1 = f32x4_pmin(v1, min1);
    }
    let m = f32x4_pmax(f32x4_pmax(max0, max1), f32x4_neg(f32x4_pmin(min0, min1)));
    let m = f32x4_pmax(m, i64x2_shuffle::<1, 0>(m, m));
    let m = f32x4_pmax(m, i32x4_shuffle::<1, 0, 3, 2>(m, m));
    let mut maxval = f32x4_extract_lane::<0>(m);
    for &v in tail {
        maxval = fmax(maxval, v);
        maxval = fmax(maxval, -v);
    }
    maxval
}

/// Calculates the arctangent of x using a Remez approximation of order 15,
/// incorporating only odd-powered terms.
#[inline]
pub fn atan_norm(x: f32) -> f32 {
    const ATAN2_2_OVER_PI: f32 = 0.636619772367581;
    let x_sq = x * x;
    /* Polynomial coefficients approximated in the [0, 1] range.
     * Lolremez command: lolremez --degree 6 --range "0:1"
     *                   "(atan(sqrt(x))-sqrt(x))/(x*sqrt(x))" "1/(sqrt(x)*x)"
     * Please note that ATAN2_COEFF_A01 is fixed to 1.0f. */
    const ATAN2_COEFF_A03: f32 = -3.3331659436225891113281250000e-01;
    const ATAN2_COEFF_A05: f32 = 1.99627041816711425781250000000e-01;
    const ATAN2_COEFF_A07: f32 = -1.3976582884788513183593750000e-01;
    const ATAN2_COEFF_A09: f32 = 9.79423448443412780761718750000e-02;
    const ATAN2_COEFF_A11: f32 = -5.7773590087890625000000000000e-02;
    const ATAN2_COEFF_A13: f32 = 2.30401363223791122436523437500e-02;
    const ATAN2_COEFF_A15: f32 = -4.3554059229791164398193359375e-03;
    ATAN2_2_OVER_PI
        * (x + x
            * x_sq
            * (ATAN2_COEFF_A03
                + x_sq
                    * (ATAN2_COEFF_A05
                        + x_sq
                            * (ATAN2_COEFF_A07
                                + x_sq
                                    * (ATAN2_COEFF_A09
                                        + x_sq
                                            * (ATAN2_COEFF_A11
                                                + x_sq
                                                    * (ATAN2_COEFF_A13
                                                        + x_sq * ATAN2_COEFF_A15)))))))
}

/// Calculates the arctangent of y/x, returning an approximate value in radians.
#[inline]
pub fn atan2p_norm(y: f32, x: f32) -> f32 {
    debug_assert!(x >= 0.0 && y >= 0.0);
    /* For very small values, we don't care about the answer. */
    if (x * x + y * y) < 1e-18 {
        return 0.0;
    }
    if y < x {
        atan_norm(y / x)
    } else {
        1.0 - atan_norm(x / y)
    }
}

/// `(float)sqrt(x)`; C celt_sqrt, celt_sqrt32.
#[inline(always)]
pub fn sqrt(x: f32) -> f32 {
    sqrt64(x as f64) as f32
}
/// `1.f/sqrt(x)`; C celt_rsqrt, celt_rsqrt_norm, celt_rsqrt_norm32.
#[inline(always)]
pub fn rsqrt(x: f32) -> f32 {
    1.0 / sqrt(x)
}
/// `(float)cos((.5f*PI)*(x))` for `0 < x < 1` (its only use, exp_rotation, passes `theta` and
/// `1 - theta` with `theta < .5`). This is libm's `cos` (FreeBSD s_cos.c) restricted to the
/// angles `0 < a < pi/2` that this gives, without the general argument reduction: up to pi/4
/// `k_cos(a, 0)`, above it the `n == 1` case of `rem_pio2` (a - pi/2 in two parts) and
/// `-k_sin`. The result is bit for bit that of `libm::cos` (checked for every `(len, K, spread)`
/// that exp_rotation can see, and far beyond).
pub fn cos_norm(x: f32) -> f32 {
    const C: [f64; 6] = [
        4.16666666666666019037e-02,
        -1.38888888888741095749e-03,
        2.48015872894767294178e-05,
        -2.75573143513906633035e-07,
        2.08757232129817482790e-09,
        -1.13596475577881948265e-11,
    ];
    const S1: f64 = -1.66666666666666324348e-01;
    const S: [f64; 5] = [
        8.33333333332248946124e-03,
        -1.98412698298579493134e-04,
        2.75573137070700676789e-06,
        -2.50507602534068634195e-08,
        1.58969099521155010221e-10,
    ];
    const PIO2_1: f64 = 1.57079632673412561417e+00;
    const PIO2_1T: f64 = 6.07710050650619224932e-11;
    let a = (0.5f32 as f64 * PI) * x as f64;
    debug_assert!(a > 0.0 && a < 1.5707);
    if (a.to_bits() >> 32) as u32 <= 0x3fe921fb {
        /* |a| ~< pi/4: k_cos(a, 0) */
        let z = a * a;
        let w = z * z;
        let r = z * (C[0] + z * (C[1] + z * C[2])) + w * w * (C[3] + z * (C[4] + z * C[5]));
        let hz = 0.5 * z;
        let w = 1.0 - hz;
        (w + (((1.0 - w) - hz) + z * r)) as f32
    } else {
        /* rem_pio2 for pi/4 < a <= 3pi/4: n = 1, then cos(a) = -k_sin(y0, y1, 1) */
        let z = a - PIO2_1;
        let y0 = z - PIO2_1T;
        let y1 = (z - y0) - PIO2_1T;
        let z = y0 * y0;
        let w = z * z;
        let r = S[0] + z * (S[1] + z * S[2]) + z * w * (S[3] + z * S[4]);
        let v = z * y0;
        (-(y0 - ((z * (0.5 * y1 - v * r) - y1) - v * S1))) as f32
    }
}
/// `1.f/(x)`; C celt_rcp.
#[inline(always)]
pub fn rcp(x: f32) -> f32 {
    1.0 / x
}
/// `(float)(1.442695040888963387*log(x))` for a positive normal finite `x` (band energies, which
/// are at least `sqrt(1e-27)`, and `1.001 - c*c` with `|c| <= 1`); C celt_log2, celt_log2_db. This is
/// libm's `log` (FreeBSD e_log.c) without the special cases, with the argument reduction done on
/// the `f32` bits: the split at `sqrt(2)` uses the 20 mantissa bits libm compares, so `k` and `f`
/// are libm's. The result is bit for bit that of `libm::log` (checked for every positive normal
/// finite `f32`).
pub fn log2(x: f32) -> f32 {
    const LN2_HI: f64 = 6.93147180369123816490e-01;
    const LN2_LO: f64 = 1.90821492927058770002e-10;
    const LG: [f64; 7] = [
        6.666666666666735130e-01,
        3.999999999940941908e-01,
        2.857142874366239149e-01,
        2.222219843214978396e-01,
        1.818357216161805012e-01,
        1.531383769920937332e-01,
        1.479819860511658591e-01,
    ];
    debug_assert!(x >= f32::MIN_POSITIVE && x < f32::INFINITY);
    /* x = 2^k*(1+f) with sqrt(2)/2 < 1+f < sqrt(2) */
    let ix = x.to_bits() + (0x3f800000 - 0x3f3504f0);
    let k = (ix >> 23) as i32 - 0x7f;
    let f = f32::from_bits((ix & 0x7fffff) + 0x3f3504f0) as f64 - 1.0;
    /* log(1+f) = 2s + s*R(s*s) with s = f/(2+f) */
    let hfsq = 0.5 * f * f;
    let s = f / (2.0 + f);
    let z = s * s;
    let w = z * z;
    let t1 = w * (LG[1] + w * (LG[3] + w * LG[5]));
    let t2 = z * (LG[0] + w * (LG[2] + w * (LG[4] + w * LG[6])));
    let r = t2 + t1;
    let dk = k as f64;
    (1.442695040888963387 * (s * (hfsq + r) + dk * LN2_LO - hfsq + f + dk * LN2_HI)) as f32
}
/// `(float)exp(0.6931471805599453094*(x))` for `0 <= x <= 4` on the `f32` lanes 0 and 1 of `x`
/// (results in lanes 0 and 1, lanes 2 and 3 zero); C celt_exp2, celt_exp2_db. Its only use, the
/// dynalloc importance, passes `MIN32(4, follower)` with `follower >= 0`. This is libm's `exp`
/// (FreeBSD e_exp.c) on `f64x2` without the special cases and `scalbn`: the reduction
/// `x*ln2 = k*ln2 + r` takes `k` as `x` rounded half up (`0 <= k <= 4`), and `2^k` is exact.
/// Each lane is bit for bit `libm::exp` (no fused multiply-add; checked for every `f32` in `[0, 4]`).
pub fn exp2x2(x: v128) -> v128 {
    const LN2_HI: f64 = 6.93147180369123816490e-01;
    const LN2_LO: f64 = 1.90821492927058770002e-10;
    const P: [f64; 5] = [
        1.66666666666666019037e-01,
        -2.77777777770155933842e-03,
        6.61375632143793436117e-05,
        -1.65339022054652515390e-06,
        4.13813679705723846039e-08,
    ];
    let k = i32x4_trunc_sat_f32x4(f32x4_add(x, f32x4_splat(0.5)));
    let dk = f64x2_convert_low_i32x4(k);
    let hi = f64x2_sub(
        f64x2_mul(
            f64x2_splat(0.6931471805599453094),
            f64x2_promote_low_f32x4(x),
        ),
        f64x2_mul(dk, f64x2_splat(LN2_HI)),
    );
    let lo = f64x2_mul(dk, f64x2_splat(LN2_LO));
    let r = f64x2_sub(hi, lo);
    let rr = f64x2_mul(r, r);
    /* c + rr*a for the Horner steps */
    let p = |c: f64, a: v128| f64x2_add(f64x2_splat(c), f64x2_mul(rr, a));
    let c = f64x2_sub(
        r,
        f64x2_mul(rr, p(P[0], p(P[1], p(P[2], p(P[3], f64x2_splat(P[4])))))),
    );
    let y = f64x2_div(f64x2_mul(r, c), f64x2_sub(f64x2_splat(2.0), c));
    let y = f64x2_add(f64x2_splat(1.0), f64x2_add(f64x2_sub(y, lo), hi));
    let scale = i64x2_shl(i64x2_add(i64x2_extend_low_i32x4(k), i64x2_splat(0x3ff)), 52);
    f32x4_demote_f64x2_zero(f64x2_mul(y, scale))
}

/// `acos(x)` (double) for `|x| < 1`: libm's `acos` (FreeBSD e_acos.c) with the hardware
/// `sqrt` and without the special cases. Bit for bit that of `libm::acos` (checked for every
/// `f32` with `|x| < 1`, which covers the `0.5f * lpc[0]` its only caller passes).
pub fn acos64(x: f64) -> f64 {
    const PIO2_HI: f64 = 1.57079632679489655800e+00;
    const PIO2_LO: f64 = 6.12323399573676603587e-17;
    const PS: [f64; 6] = [
        1.66666666666666657415e-01,
        -3.25565818622400915405e-01,
        2.01212532134862925881e-01,
        -4.00555345006794114027e-02,
        7.91534994289814532176e-04,
        3.47933107596021167570e-05,
    ];
    const QS: [f64; 4] = [
        -2.40339491173441421878e+00,
        2.02094576023350569471e+00,
        -6.88283971605453293030e-01,
        7.70381505559019352791e-02,
    ];
    let r = |z: f64| -> f64 {
        let p = z * (PS[0] + z * (PS[1] + z * (PS[2] + z * (PS[3] + z * (PS[4] + z * PS[5])))));
        let q = 1.0 + z * (QS[0] + z * (QS[1] + z * (QS[2] + z * QS[3])));
        p / q
    };
    let hx = (x.to_bits() >> 32) as u32;
    let ix = hx & 0x7fffffff;
    debug_assert!(ix < 0x3ff00000);
    if ix < 0x3fe00000 {
        /* |x| < 0.5 */
        if ix <= 0x3c600000 {
            /* |x| < 2**-57 (+ the inexact flag of libm) */
            return PIO2_HI + f64::from_bits(0x3870000000000000);
        }
        return PIO2_HI - (x - (PIO2_LO - x * r(x * x)));
    }
    if (hx >> 31) != 0 {
        /* x < -0.5 */
        let z = (1.0 + x) * 0.5;
        let s = sqrt64(z);
        let w = r(z) * s - PIO2_LO;
        return 2. * (PIO2_HI - (s + w));
    }
    /* x > 0.5 */
    let z = (1.0 - x) * 0.5;
    let s = sqrt64(z);
    let df = f64::from_bits(s.to_bits() & 0xff_ff_ff_ff_00_00_00_00);
    let c = (z - df * df) / (s + df);
    let w = r(z) * s + c;
    2. * (df + w)
}
