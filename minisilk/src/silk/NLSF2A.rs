//! Port of silk/NLSF2A.c.

/* conversion between prediction filter coefficients and LSFs   */
/* order should be even                                         */
/* a piecewise linear approximation maps LSF <-> cos(LSF)       */
/* therefore the result is not accurate LSFs, but the two       */
/* functions are accurate inverses of each other                */

use crate::silk::LPC_fit::silk_LPC_fit;
use crate::silk::LPC_inv_pred_gain::silk_LPC_inverse_pred_gain;
use crate::silk::bwexpander_32::silk_bwexpander_32;
use crate::silk::define::*;
use crate::silk::macros::*;
use crate::silk::tables::silk_LSFCosTab_FIX_Q12;

const QA: i32 = 16;

const HALF_MAX: usize = (MAX_LPC_ORDER / 2 + 1) as usize;

/// helper function for NLSF2A(..)
/// `out`: intermediate polynomial, QA [dd+1]; `cLSF`: vector of interleaved 2*cos(LSFs), QA [d];
/// `dd`: polynomial order (= 1/2 * filter order)
#[inline(always)]
fn silk_NLSF2A_find_poly(out: &mut [i32; HALF_MAX], cLSF: &[i32], dd: i32) {
    out[0] = silk_LSHIFT(1, QA);
    out[1] = -cLSF.at(0);
    for k in 1..dd as usize {
        let ftmp = cLSF.at(2 * k); /* QA*/
        *out.at_mut(k + 1) = silk_LSHIFT(out.at(k - 1), 1) - silk_RSHIFT_ROUND64(silk_SMULL(ftmp, out.at(k)), QA) as i32;
        for n in (2..=k).rev() {
            *out.at_mut(n) += out.at(n - 2) - silk_RSHIFT_ROUND64(silk_SMULL(ftmp, out.at(n - 1)), QA) as i32;
        }
        out[1] -= ftmp;
    }
}

/// compute whitening filter coefficients from normalized line spectral frequencies
/// `a_Q12`: monic whitening filter coefficients in Q12, [ d ] (O);
/// `NLSF`: normalized line spectral frequencies in Q15, [ d ];
/// `d`: filter order (should be even)
pub fn silk_NLSF2A(a_Q12: &mut [i16], NLSF: &[i16], d: i32) {
    /* This ordering was found to maximize quality. It improves numerical accuracy of
    silk_NLSF2A_find_poly() compared to "standard" ordering. */
    static ordering16: [u8; 16] = [0, 15, 8, 7, 4, 11, 12, 3, 2, 13, 10, 5, 6, 9, 14, 1];
    static ordering10: [u8; 10] = [0, 9, 6, 3, 4, 5, 8, 1, 2, 7];

    let mut cos_LSF_QA = [0i32; MAX_LPC_ORDER as usize];
    let mut P = [0i32; HALF_MAX];
    let mut Q = [0i32; HALF_MAX];
    let mut a32_QA1 = [0i32; MAX_LPC_ORDER as usize];

    debug_assert!(LSF_COS_TAB_SZ_FIX == 128);
    debug_assert!(d == 10 || d == 16);

    /* convert LSFs to 2*cos(LSF), using piecewise linear curve from table */
    let ordering: &[u8] = if d == 16 { &ordering16 } else { &ordering10 };
    for k in 0..d as usize {
        debug_assert!(NLSF.at(k) >= 0);

        /* f_int on a scale 0-127 (rounded down) */
        let f_int = silk_RSHIFT(NLSF.at(k) as i32, 15 - 7);

        /* f_frac, range: 0..255 */
        let f_frac = NLSF.at(k) as i32 - silk_LSHIFT(f_int, 15 - 7);

        debug_assert!(f_int >= 0);
        debug_assert!(f_int < LSF_COS_TAB_SZ_FIX);

        /* Read start and end value from table */
        let cos_val = silk_LSFCosTab_FIX_Q12.at(f_int as usize) as i32; /* Q12 */
        let delta = silk_LSFCosTab_FIX_Q12.at(f_int as usize + 1) as i32 - cos_val; /* Q12, with a range of 0..200 */

        /* Linear interpolation */
        *cos_LSF_QA.at_mut(ordering.at(k) as usize) =
            silk_RSHIFT_ROUND(silk_LSHIFT(cos_val, 8) + silk_MUL(delta, f_frac), 20 - QA); /* QA */
    }

    let dd = silk_RSHIFT(d, 1);

    /* generate even and odd polynomials using convolution */
    silk_NLSF2A_find_poly(&mut P, cos_LSF_QA.rg(0..), dd);
    silk_NLSF2A_find_poly(&mut Q, cos_LSF_QA.rg(1..), dd);

    /* convert even and odd polynomials to opus_int32 Q12 filter coefs */
    for k in 0..dd as usize {
        let Ptmp = P.at(k + 1) + P.at(k);
        let Qtmp = Q.at(k + 1) - Q.at(k);

        /* the Ptmp and Qtmp values at this stage need to fit in int32 */
        *a32_QA1.at_mut(k) = -Qtmp - Ptmp; /* QA+1 */
        *a32_QA1.at_mut(d as usize - k - 1) = Qtmp - Ptmp; /* QA+1 */
    }

    /* Convert int32 coefficients to Q12 int16 coefs */
    silk_LPC_fit(a_Q12, &mut a32_QA1, 12, QA + 1, d);

    let mut i = 0;
    while silk_LPC_inverse_pred_gain(a_Q12, d) == 0 && i < MAX_LPC_STABILIZE_ITERATIONS {
        /* Prediction coefficients are (too close to) unstable; apply bandwidth expansion   */
        /* on the unscaled coefficients, convert to Q12 and measure again                   */
        silk_bwexpander_32(&mut a32_QA1, d, 65536 - silk_LSHIFT(2, i));
        for k in 0..d as usize {
            *a_Q12.at_mut(k) = silk_RSHIFT_ROUND(a32_QA1.at(k), QA + 1 - 12) as i16; /* QA+1 -> Q12 */
        }
        i += 1;
    }
}
