//! Port of celt/mathops.h and celt/mathops.c (float build, without FLOAT_APPROX).

use super::arch::{MAX16, MAX32, MIN16};

/// PI from celt/mathops.h (a double constant).
pub const PI: f64 = 3.1415926535897931;

pub fn celt_maxabs16(x: &[f32], len: i32) -> f32 {
    let mut maxval: f32 = 0.0;
    let mut minval: f32 = 0.0;
    for &v in &x[..len as usize] {
        maxval = MAX16(maxval, v);
        minval = MIN16(minval, v);
    }
    MAX32(maxval, -minval)
}

/// celt_maxabs_res (float build) = celt_maxabs16.
#[inline(always)]
pub fn celt_maxabs_res(x: &[f32], len: i32) -> f32 {
    celt_maxabs16(x, len)
}

/// Calculates the arctangent of x using a Remez approximation of order 15,
/// incorporating only odd-powered terms.
#[inline]
pub fn celt_atan_norm(x: f32) -> f32 {
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
                                                + x_sq * (ATAN2_COEFF_A13 + x_sq * ATAN2_COEFF_A15)))))))
}

/// Calculates the arctangent of y/x, returning an approximate value in radians.
#[inline]
pub fn celt_atan2p_norm(y: f32, x: f32) -> f32 {
    debug_assert!(x >= 0.0 && y >= 0.0);
    /* For very small values, we don't care about the answer. */
    if (x * x + y * y) < 1e-18 {
        return 0.0;
    }
    if y < x { celt_atan_norm(y / x) } else { 1.0 - celt_atan_norm(x / y) }
}

/// `(float)sqrt(x)`; also celt_sqrt32.
#[inline(always)]
pub fn celt_sqrt(x: f32) -> f32 {
    crate::celt::arch::sqrt64(x as f64) as f32
}
#[inline(always)]
pub fn celt_sqrt32(x: f32) -> f32 {
    celt_sqrt(x)
}
/// `1.f/celt_sqrt(x)`; also celt_rsqrt_norm, celt_rsqrt_norm32.
#[inline(always)]
pub fn celt_rsqrt(x: f32) -> f32 {
    1.0 / celt_sqrt(x)
}
#[inline(always)]
pub fn celt_rsqrt_norm(x: f32) -> f32 {
    celt_rsqrt(x)
}
#[inline(always)]
pub fn celt_rsqrt_norm32(x: f32) -> f32 {
    celt_rsqrt(x)
}
/// `(float)cos((.5f*PI)*(x))` for `0 < x < 1` (its only use, exp_rotation, passes `theta` and
/// `1 - theta` with `theta < .5`). This is libm's `cos` (FreeBSD s_cos.c) restricted to the
/// angles `0 < a < pi/2` that this gives, without the general argument reduction: up to pi/4
/// `k_cos(a, 0)`, above it the `n == 1` case of `rem_pio2` (a - pi/2 in two parts) and
/// `-k_sin`. The result is bit for bit that of `libm::cos` (checked for every `(len, K, spread)`
/// that exp_rotation can see, and far beyond).
pub fn celt_cos_norm(x: f32) -> f32 {
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
/// `1.f/(x)`
#[inline(always)]
pub fn celt_rcp(x: f32) -> f32 {
    1.0 / x
}
/// `(float)(1.442695040888963387*log(x))`; also celt_log2_db.
#[inline(always)]
pub fn celt_log2(x: f32) -> f32 {
    (1.442695040888963387 * libm::log(x as f64)) as f32
}
/// `(float)exp(0.6931471805599453094*(x))`; also celt_exp2_db.
#[inline(always)]
pub fn celt_exp2(x: f32) -> f32 {
    libm::exp(0.6931471805599453094 * x as f64) as f32
}
#[inline(always)]
pub fn celt_log2_db(x: f32) -> f32 {
    celt_log2(x)
}
#[inline(always)]
pub fn celt_exp2_db(x: f32) -> f32 {
    celt_exp2(x)
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
        let s = super::arch::sqrt64(z);
        let w = r(z) * s - PIO2_LO;
        return 2. * (PIO2_HI - (s + w));
    }
    /* x > 0.5 */
    let z = (1.0 - x) * 0.5;
    let s = super::arch::sqrt64(z);
    let df = f64::from_bits(s.to_bits() & 0xff_ff_ff_ff_00_00_00_00);
    let c = (z - df * df) / (s + df);
    let w = r(z) * s + c;
    2. * (df + w)
}
