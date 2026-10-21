//! Port of silk/A2NLSF.c.

/* Conversion between prediction filter coefficients and NLSFs  */
/* Requires the order to be an even number                      */
/* A piecewise linear approximation maps LSF <-> cos(LSF)       */
/* Therefore the result is not accurate NLSFs, but the two      */
/* functions are accurate inverses of each other                */

use crate::silk::bwexpander_32::silk_bwexpander_32;
use crate::silk::define::*;
use crate::silk::macros::*;
use crate::silk::tables::silk_LSFCosTab_FIX_Q12;

/* Number of binary divisions, when not in low complexity mode */
const BIN_DIV_STEPS_A2NLSF_FIX: i32 = 3; /* must be no higher than 16 - log2( LSF_COS_TAB_SZ_FIX ) */
const MAX_ITERATIONS_A2NLSF_FIX: i32 = 16;

const HALF_MAX: usize = (MAX_LPC_ORDER / 2 + 1) as usize;

/// Helper function for A2NLSF(..)
/// Transforms polynomials from cos(n*f) to cos(f)^n
#[inline(always)]
fn silk_A2NLSF_trans_poly(p: &mut [i32; HALF_MAX], dd: i32) {
    for k in 2..=dd as usize {
        for n in ((k + 1)..=dd as usize).rev() {
            *p.at_mut(n - 2) -= p.at(n);
        }
        *p.at_mut(k - 2) -= silk_LSHIFT(p.at(k), 1);
    }
}

/// Helper function for A2NLSF(..)
/// Polynomial evaluation. Returns the polynomial evaluation, in Q16.
/// `p`: polynomial, Q16; `x`: evaluation point, Q12; `dd`: order.
#[inline(always)]
fn silk_A2NLSF_eval_poly(p: &[i32; HALF_MAX], x: i32, dd: i32) -> i32 {
    let mut y32 = p.at(dd as usize); /* Q16 */
    let x_Q16 = silk_LSHIFT(x, 4);

    if 8 == dd {
        y32 = silk_SMLAWW(p[7], y32, x_Q16);
        y32 = silk_SMLAWW(p[6], y32, x_Q16);
        y32 = silk_SMLAWW(p[5], y32, x_Q16);
        y32 = silk_SMLAWW(p[4], y32, x_Q16);
        y32 = silk_SMLAWW(p[3], y32, x_Q16);
        y32 = silk_SMLAWW(p[2], y32, x_Q16);
        y32 = silk_SMLAWW(p[1], y32, x_Q16);
        y32 = silk_SMLAWW(p[0], y32, x_Q16);
    } else {
        for n in (0..dd as usize).rev() {
            y32 = silk_SMLAWW(p.at(n), y32, x_Q16); /* Q16 */
        }
    }
    y32
}

#[inline(always)]
fn silk_A2NLSF_init(a_Q16: &[i32], P: &mut [i32; HALF_MAX], Q: &mut [i32; HALF_MAX], dd: i32) {
    let dd = dd as usize;

    /* Convert filter coefs to even and odd polynomials */
    P[dd] = silk_LSHIFT(1, 16);
    Q[dd] = silk_LSHIFT(1, 16);
    for k in 0..dd {
        *P.at_mut(k) = -a_Q16.at(dd - k - 1) - a_Q16.at(dd + k); /* Q16 */
        *Q.at_mut(k) = -a_Q16.at(dd - k - 1) + a_Q16.at(dd + k); /* Q16 */
    }

    /* Divide out zeros as we have that for even filter orders, */
    /* z =  1 is always a root in Q, and                        */
    /* z = -1 is always a root in P                             */
    for k in (1..=dd).rev() {
        *P.at_mut(k - 1) -= P.at(k);
        *Q.at_mut(k - 1) += Q.at(k);
    }

    /* Transform polynomials from cos(n*f) to cos(f)^n */
    silk_A2NLSF_trans_poly(P, dd as i32);
    silk_A2NLSF_trans_poly(Q, dd as i32);
}

/// Compute Normalized Line Spectral Frequencies (NLSFs) from whitening filter coefficients.
/// If not all roots are found, the a_Q16 coefficients are bandwidth expanded until convergence.
/// `NLSF`: normalized line spectral frequencies in Q15 (0..2^15-1) [d] (O);
/// `a_Q16`: monic whitening filter coefficients in Q16 [d] (I/O);
/// `d`: filter order (must be even).
pub fn silk_A2NLSF(NLSF: &mut [i16], a_Q16: &mut [i32], d: i32) {
    let mut P = [0i32; HALF_MAX];
    let mut Q = [0i32; HALF_MAX];
    /* PQ[0] = P, PQ[1] = Q: `pq` selects the polynomial currently in use */
    let mut pq: usize;

    let dd = silk_RSHIFT(d, 1);

    silk_A2NLSF_init(a_Q16, &mut P, &mut Q, dd);

    /* Find roots, alternating between P and Q */
    pq = 0; /* Pointer to polynomial */

    let mut xlo = silk_LSFCosTab_FIX_Q12[0] as i32; /* Q12*/
    let mut ylo = silk_A2NLSF_eval_poly(&P, xlo, dd);

    let mut root_ix: i32;
    if ylo < 0 {
        /* Set the first NLSF to zero and move on to the next */
        NLSF[0] = 0;
        pq = 1; /* Pointer to polynomial */
        ylo = silk_A2NLSF_eval_poly(&Q, xlo, dd);
        root_ix = 1; /* Index of current root */
    } else {
        root_ix = 0; /* Index of current root */
    }
    let mut k: i32 = 1; /* Loop counter */
    let mut i: i32 = 0; /* Counter for bandwidth expansions applied */
    let mut thr: i32 = 0;
    loop {
        /* Evaluate polynomial */
        let mut xhi = silk_LSFCosTab_FIX_Q12.at(k as usize) as i32; /* Q12 */
        let mut yhi = silk_A2NLSF_eval_poly(if pq == 0 { &P } else { &Q }, xhi, dd);

        /* Detect zero crossing */
        if (ylo <= 0 && yhi >= thr) || (ylo >= 0 && yhi <= -thr) {
            if yhi == 0 {
                /* If the root lies exactly at the end of the current       */
                /* interval, look for the next root in the next interval    */
                thr = 1;
            } else {
                thr = 0;
            }
            /* Binary division */
            let mut ffrac: i32 = -256;
            for m in 0..BIN_DIV_STEPS_A2NLSF_FIX {
                /* Evaluate polynomial */
                let xmid = silk_RSHIFT_ROUND(xlo + xhi, 1);
                let ymid = silk_A2NLSF_eval_poly(if pq == 0 { &P } else { &Q }, xmid, dd);

                /* Detect zero crossing */
                if (ylo <= 0 && ymid >= 0) || (ylo >= 0 && ymid <= 0) {
                    /* Reduce frequency */
                    xhi = xmid;
                    yhi = ymid;
                } else {
                    /* Increase frequency */
                    xlo = xmid;
                    ylo = ymid;
                    ffrac = silk_ADD_RSHIFT(ffrac, 128, m);
                }
            }

            /* Interpolate */
            if silk_abs!(ylo) < 65536 {
                /* Avoid dividing by zero */
                let den = ylo - yhi;
                let nom = silk_LSHIFT(ylo, 8 - BIN_DIV_STEPS_A2NLSF_FIX) + silk_RSHIFT(den, 1);
                if den != 0 {
                    // div_s: guarded by den != 0
                    ffrac += silk_DIV32(nom, den);
                }
            } else {
                /* No risk of dividing by zero because abs(ylo - yhi) >= abs(ylo) >= 65536 */
                // div_s: divisor magnitude >= 65536 >> (8 - BIN_DIV_STEPS_A2NLSF_FIX) > 0
                ffrac += silk_DIV32(ylo, silk_RSHIFT(ylo - yhi, 8 - BIN_DIV_STEPS_A2NLSF_FIX));
            }
            *NLSF.at_mut(root_ix as usize) = silk_min_32(silk_LSHIFT(k, 8) + ffrac, silk_int16_MAX) as i16;

            debug_assert!(NLSF.at(root_ix as usize) >= 0);

            root_ix += 1; /* Next root */
            if root_ix >= d {
                /* Found all roots */
                break;
            }
            /* Alternate pointer to polynomial */
            pq = (root_ix & 1) as usize;

            /* Evaluate polynomial */
            xlo = silk_LSFCosTab_FIX_Q12.at((k - 1) as usize) as i32; /* Q12*/
            ylo = silk_LSHIFT(1 - (root_ix & 2), 12);
        } else {
            /* Increment loop counter */
            k += 1;
            xlo = xhi;
            ylo = yhi;
            thr = 0;

            if k > LSF_COS_TAB_SZ_FIX {
                i += 1;
                if i > MAX_ITERATIONS_A2NLSF_FIX {
                    /* Set NLSFs to white spectrum and exit */
                    // div_s: d >= 0, so d + 1 >= 1
                    NLSF[0] = silk_DIV32_16(1 << 15, d + 1) as i16;
                    for k in 1..d as usize {
                        *NLSF.at_mut(k) = NLSF.at(k - 1).wrapping_add(NLSF.at(0));
                    }
                    return;
                }

                /* Error: Apply progressively more bandwidth expansion and run again */
                silk_bwexpander_32(a_Q16, d, 65536 - silk_LSHIFT(1, i));

                silk_A2NLSF_init(a_Q16, &mut P, &mut Q, dd);
                pq = 0; /* Pointer to polynomial */
                xlo = silk_LSFCosTab_FIX_Q12[0] as i32; /* Q12*/
                ylo = silk_A2NLSF_eval_poly(&P, xlo, dd);
                if ylo < 0 {
                    /* Set the first NLSF to zero and move on to the next */
                    NLSF[0] = 0;
                    pq = 1; /* Pointer to polynomial */
                    ylo = silk_A2NLSF_eval_poly(&Q, xlo, dd);
                    root_ix = 1; /* Index of current root */
                } else {
                    root_ix = 0; /* Index of current root */
                }
                k = 1; /* Reset loop counter */
            }
        }
    }
}
