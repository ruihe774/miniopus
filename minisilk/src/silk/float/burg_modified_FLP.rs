//! Port of silk/float/burg_modified_FLP.c.

use crate::silk::float::energy_FLP::silk_energy_FLP;
use crate::silk::float::inner_product_FLP::silk_inner_product_FLP;
use crate::silk::macros::Ix;
use crate::silk::float::schur_FLP::SILK_MAX_ORDER_LPC;
use crate::silk::tuning_parameters::FIND_LPC_COND_FAC;

const MAX_FRAME_SIZE: i32 = 384; /* subfr_length * nb_subfr = ( 0.005 * 16000 + 16 ) * 4 = 384*/

/// Compute reflection coefficients from input signal. Returns residual energy.
pub fn silk_burg_modified_FLP(
    A: &mut [f32],       /* O    prediction coefficients (length order)                      */
    x: &[f32],           /* I    input signal, length: nb_subfr*(D+L_sub)                    */
    minInvGain: f32,     /* I    minimum inverse prediction gain                             */
    subfr_length: i32,   /* I    input signal subframe length (incl. D preceding samples)    */
    nb_subfr: i32,       /* I    number of subframes stacked in x                            */
    D: i32,              /* I    order                                                       */
) -> f32 {
    const MO: usize = SILK_MAX_ORDER_LPC;
    let mut C_first_row = [0f64; MO];
    let mut CAf = [0f64; MO + 1];
    let mut CAb = [0f64; MO + 1];
    let mut Af = [0f64; MO];

    debug_assert!(subfr_length * nb_subfr <= MAX_FRAME_SIZE);
    let (sl, nsf, d) = (subfr_length as usize, nb_subfr as usize, D as usize);
    let cond_fac = FIND_LPC_COND_FAC as f64;

    /* Compute autocorrelations, added over subframes */
    let mut C0: f64 = silk_energy_FLP(x, nb_subfr * subfr_length);
    for s in 0..nsf {
        let x_ptr = x.rg(s * sl..);
        for n in 1..d + 1 {
            *C_first_row.at_mut(n - 1) += silk_inner_product_FLP(x_ptr, x_ptr.rg(n..), subfr_length - n as i32);
        }
    }
    let mut C_last_row = C_first_row;

    /* Initialize */
    CAf[0] = C0 + cond_fac * C0 + 1e-9f32 as f64;
    CAb[0] = CAf[0];
    let mut invGain: f64 = 1.0;
    let mut reached_max_gain = false;
    for n in 0..d {
        /* Update first row of correlation matrix (without first element) */
        /* Update last row of correlation matrix (without last element, stored in reversed order) */
        /* Update C * Af */
        /* Update C * flipud(Af) (stored in reversed order) */
        for s in 0..nsf {
            let x_ptr = x.rg(s * sl..);
            let mut tmp1: f64 = x_ptr.at(n) as f64;
            let mut tmp2: f64 = x_ptr.at(sl - n - 1) as f64;
            for k in 0..n {
                /* the products of two floats are computed in float */
                *C_first_row.at_mut(k) -= (x_ptr.at(n) * x_ptr.at(n - k - 1)) as f64;
                *C_last_row.at_mut(k) -= (x_ptr.at(sl - n - 1) * x_ptr.at(sl - n + k)) as f64;
                let Atmp = Af.at(k);
                tmp1 += x_ptr.at(n - k - 1) as f64 * Atmp;
                tmp2 += x_ptr.at(sl - n + k) as f64 * Atmp;
            }
            for k in 0..=n {
                *CAf.at_mut(k) -= tmp1 * x_ptr.at(n - k) as f64;
                *CAb.at_mut(k) -= tmp2 * x_ptr.at(sl - n + k - 1) as f64;
            }
        }
        let mut tmp1 = C_first_row.at(n);
        let mut tmp2 = C_last_row.at(n);
        for k in 0..n {
            let Atmp = Af.at(k);
            tmp1 += C_last_row.at(n - k - 1) * Atmp;
            tmp2 += C_first_row.at(n - k - 1) * Atmp;
        }
        *CAf.at_mut(n + 1) = tmp1;
        *CAb.at_mut(n + 1) = tmp2;

        /* Calculate nominator and denominator for the next order reflection (parcor) coefficient */
        let mut num = CAb.at(n + 1);
        let mut nrg_b = CAb[0];
        let mut nrg_f = CAf[0];
        for k in 0..n {
            let Atmp = Af.at(k);
            num += CAb.at(n - k) * Atmp;
            nrg_b += CAb.at(k + 1) * Atmp;
            nrg_f += CAf.at(k + 1) * Atmp;
        }
        debug_assert!(nrg_f > 0.0);
        debug_assert!(nrg_b > 0.0);

        /* Calculate the next order reflection (parcor) coefficient */
        let mut rc = -2.0 * num / (nrg_f + nrg_b);
        debug_assert!(rc > -1.0 && rc < 1.0);

        /* Update inverse prediction gain */
        let tmp1 = invGain * (1.0 - rc * rc);
        if tmp1 <= minInvGain as f64 {
            /* Max prediction gain exceeded; set reflection coefficient such that max prediction gain is exactly hit */
            rc = crate::silk::float::sigproc_flp::sqrt(1.0 - minInvGain as f64 / invGain);
            if num > 0.0 {
                /* Ensure adjusted reflection coefficients has the original sign */
                rc = -rc;
            }
            invGain = minInvGain as f64;
            reached_max_gain = true;
        } else {
            invGain = tmp1;
        }

        /* Update the AR coefficients */
        for k in 0..(n + 1) >> 1 {
            let tmp1 = Af.at(k);
            let tmp2 = Af.at(n - k - 1);
            *Af.at_mut(k) = tmp1 + rc * tmp2;
            *Af.at_mut(n - k - 1) = tmp2 + rc * tmp1;
        }
        *Af.at_mut(n) = rc;

        if reached_max_gain {
            /* Reached max prediction gain; set remaining coefficients to zero and exit loop */
            for k in n + 1..d {
                *Af.at_mut(k) = 0.0;
            }
            break;
        }

        /* Update C * Af and C * Ab */
        for k in 0..=n + 1 {
            let tmp1 = CAf.at(k);
            *CAf.at_mut(k) += rc * CAb.at(n + 1 - k);
            *CAb.at_mut(n + 1 - k) += rc * tmp1;
        }
    }

    let nrg_f: f64;
    if reached_max_gain {
        /* Convert to silk_float */
        for k in 0..d {
            *A.at_mut(k) = (-Af.at(k)) as f32;
        }
        /* Subtract energy of preceding samples from C0 */
        for s in 0..nsf {
            C0 -= silk_energy_FLP(x.rg(s * sl..), D);
        }
        /* Approximate residual energy */
        nrg_f = C0 * invGain;
    } else {
        /* Compute residual energy and store coefficients as silk_float */
        let mut nrg = CAf[0];
        let mut tmp1 = 1.0f64;
        for k in 0..d {
            let Atmp = Af.at(k);
            nrg += CAf.at(k + 1) * Atmp;
            tmp1 += Atmp * Atmp;
            *A.at_mut(k) = (-Atmp) as f32;
        }
        nrg -= cond_fac * C0 * tmp1;
        nrg_f = nrg;
    }

    /* Return residual energy */
    nrg_f as f32
}
