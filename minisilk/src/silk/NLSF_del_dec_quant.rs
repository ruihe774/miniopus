//! Port of silk/NLSF_del_dec_quant.c.

use crate::silk::define::*;
use crate::silk::macros::*;

const NSTATES: usize = NLSF_QUANT_DEL_DEC_STATES as usize;
const MAXO: usize = MAX_LPC_ORDER as usize;

/// Delayed-decision quantizer for NLSF residuals. Returns RD value in Q25.
/// `indices`: quantization indices [ order ] (O); `x_Q10`: input [ order ]; `w_Q5`: weights
/// [ order ]; `pred_coef_Q8`: backward predictor coefs [ order ]; `ec_ix`: indices to entropy
/// coding tables [ order ]; `ec_rates_Q5`: rates []; `quant_step_size_Q16`: quantization step
/// size; `inv_quant_step_size_Q6`: inverse quantization step size; `mu_Q20`: R/D tradeoff;
/// `order`: number of input values.
pub fn silk_NLSF_del_dec_quant(
    indices: &mut [i8],
    x_Q10: &[i16],
    w_Q5: &[i16],
    pred_coef_Q8: &[u8],
    ec_ix: &[i16],
    ec_rates_Q5: &[u8],
    quant_step_size_Q16: i32,
    inv_quant_step_size_Q6: i16,
    mu_Q20: i32,
    order: i16,
) -> i32 {
    let mut ind_sort = [0i32; NSTATES];
    /* zero-initialized (C leaves it uninitialized, but every value read has been written) */
    let mut ind = [[0i8; MAXO]; NSTATES];
    let mut prev_out_Q10 = [0i16; 2 * NSTATES];
    let mut RD_Q25 = [0i32; 2 * NSTATES];
    let mut RD_min_Q25 = [0i32; NSTATES];
    let mut RD_max_Q25 = [0i32; NSTATES];

    let mut out0_Q10_table = [0i32; (2 * NLSF_QUANT_MAX_AMPLITUDE_EXT) as usize];
    let mut out1_Q10_table = [0i32; (2 * NLSF_QUANT_MAX_AMPLITUDE_EXT) as usize];

    for i in -NLSF_QUANT_MAX_AMPLITUDE_EXT..=NLSF_QUANT_MAX_AMPLITUDE_EXT - 1 {
        let adj = SILK_FIX_CONST(NLSF_QUANT_LEVEL_ADJ, 10);
        let mut out0_Q10 = silk_LSHIFT(i, 10) as i16;
        let mut out1_Q10 = out0_Q10.wrapping_add(1024);
        if i > 0 {
            out0_Q10 = out0_Q10.wrapping_sub(adj as i16);
            out1_Q10 = out1_Q10.wrapping_sub(adj as i16);
        } else if i == 0 {
            out1_Q10 = out1_Q10.wrapping_sub(adj as i16);
        } else if i == -1 {
            out0_Q10 = out0_Q10.wrapping_add(adj as i16);
        } else {
            out0_Q10 = out0_Q10.wrapping_add(adj as i16);
            out1_Q10 = out1_Q10.wrapping_add(adj as i16);
        }
        *out0_Q10_table.at_mut((i + NLSF_QUANT_MAX_AMPLITUDE_EXT) as usize) =
            silk_RSHIFT(silk_SMULBB(out0_Q10 as i32, quant_step_size_Q16), 16);
        *out1_Q10_table.at_mut((i + NLSF_QUANT_MAX_AMPLITUDE_EXT) as usize) =
            silk_RSHIFT(silk_SMULBB(out1_Q10 as i32, quant_step_size_Q16), 16);
    }

    debug_assert!((NLSF_QUANT_DEL_DEC_STATES & (NLSF_QUANT_DEL_DEC_STATES - 1)) == 0); /* must be power of two */

    let mut nStates: usize = 1;
    RD_Q25[0] = 0;
    prev_out_Q10[0] = 0;
    for i in (0..order as usize).rev() {
        let rates_Q5 = ec_rates_Q5.rg(ec_ix.at(i) as usize..);
        let in_Q10 = x_Q10.at(i) as i32;
        for j in 0..nStates {
            let pred_Q10 = silk_RSHIFT(silk_SMULBB(pred_coef_Q8.at(i) as i32, prev_out_Q10.at(j) as i32), 8);
            let res_Q10 = (in_Q10 - pred_Q10) as i16 as i32; /* silk_SUB16 */
            let mut ind_tmp = silk_RSHIFT(silk_SMULBB(inv_quant_step_size_Q6 as i32, res_Q10), 16);
            ind_tmp = silk_LIMIT!(ind_tmp, -NLSF_QUANT_MAX_AMPLITUDE_EXT, NLSF_QUANT_MAX_AMPLITUDE_EXT - 1);
            *ind[j].at_mut(i) = ind_tmp as i8;

            /* compute outputs for ind_tmp and ind_tmp + 1 */
            let mut out0_Q10 = out0_Q10_table.at((ind_tmp + NLSF_QUANT_MAX_AMPLITUDE_EXT) as usize) as i16;
            let mut out1_Q10 = out1_Q10_table.at((ind_tmp + NLSF_QUANT_MAX_AMPLITUDE_EXT) as usize) as i16;

            out0_Q10 = out0_Q10.wrapping_add(pred_Q10 as i16);
            out1_Q10 = out1_Q10.wrapping_add(pred_Q10 as i16);
            *prev_out_Q10.at_mut(j) = out0_Q10;
            *prev_out_Q10.at_mut(j + nStates) = out1_Q10;

            /* compute RD for ind_tmp and ind_tmp + 1 */
            let rate0_Q5: i32;
            let rate1_Q5: i32;
            if ind_tmp + 1 >= NLSF_QUANT_MAX_AMPLITUDE {
                if ind_tmp + 1 == NLSF_QUANT_MAX_AMPLITUDE {
                    rate0_Q5 = rates_Q5.at((ind_tmp + NLSF_QUANT_MAX_AMPLITUDE) as usize) as i32;
                    rate1_Q5 = 280;
                } else {
                    rate0_Q5 = silk_SMLABB(280 - 43 * NLSF_QUANT_MAX_AMPLITUDE, 43, ind_tmp);
                    rate1_Q5 = (rate0_Q5 + 43) as i16 as i32; /* silk_ADD16 */
                }
            } else if ind_tmp <= -NLSF_QUANT_MAX_AMPLITUDE {
                if ind_tmp == -NLSF_QUANT_MAX_AMPLITUDE {
                    rate0_Q5 = 280;
                    rate1_Q5 = rates_Q5.at((ind_tmp + 1 + NLSF_QUANT_MAX_AMPLITUDE) as usize) as i32;
                } else {
                    rate0_Q5 = silk_SMLABB(280 - 43 * NLSF_QUANT_MAX_AMPLITUDE, -43, ind_tmp);
                    rate1_Q5 = (rate0_Q5 - 43) as i16 as i32; /* silk_SUB16 */
                }
            } else {
                rate0_Q5 = rates_Q5.at((ind_tmp + NLSF_QUANT_MAX_AMPLITUDE) as usize) as i32;
                rate1_Q5 = rates_Q5.at((ind_tmp + 1 + NLSF_QUANT_MAX_AMPLITUDE) as usize) as i32;
            }
            let RD_tmp_Q25 = RD_Q25.at(j);
            let mut diff_Q10 = (in_Q10 - out0_Q10 as i32) as i16 as i32; /* silk_SUB16 */
            *RD_Q25.at_mut(j) = silk_SMLABB(
                silk_MLA(RD_tmp_Q25, silk_SMULBB(diff_Q10, diff_Q10), w_Q5.at(i) as i32),
                mu_Q20,
                rate0_Q5,
            );
            diff_Q10 = (in_Q10 - out1_Q10 as i32) as i16 as i32; /* silk_SUB16 */
            *RD_Q25.at_mut(j + nStates) = silk_SMLABB(
                silk_MLA(RD_tmp_Q25, silk_SMULBB(diff_Q10, diff_Q10), w_Q5.at(i) as i32),
                mu_Q20,
                rate1_Q5,
            );
        }

        if nStates <= NSTATES / 2 {
            /* double number of states and copy */
            for j in 0..nStates {
                *ind[j + nStates].at_mut(i) = ind[j].at(i) + 1;
            }
            nStates <<= 1;
            for j in nStates..NSTATES {
                *ind[j].at_mut(i) = ind[j - nStates].at(i);
            }
        } else {
            /* sort lower and upper half of RD_Q25, pairwise */
            for j in 0..NSTATES {
                if RD_Q25[j] > RD_Q25[j + NSTATES] {
                    RD_max_Q25[j] = RD_Q25[j];
                    RD_min_Q25[j] = RD_Q25[j + NSTATES];
                    RD_Q25[j] = RD_min_Q25[j];
                    RD_Q25[j + NSTATES] = RD_max_Q25[j];
                    /* swap prev_out values */
                    let out0_Q10 = prev_out_Q10[j];
                    prev_out_Q10[j] = prev_out_Q10[j + NSTATES];
                    prev_out_Q10[j + NSTATES] = out0_Q10;
                    ind_sort[j] = (j + NSTATES) as i32;
                } else {
                    RD_min_Q25[j] = RD_Q25[j];
                    RD_max_Q25[j] = RD_Q25[j + NSTATES];
                    ind_sort[j] = j as i32;
                }
            }
            /* compare the highest RD values of the winning half with the lowest one in the losing half, and copy if necessary */
            /* afterwards ind_sort[] will contain the indices of the NLSF_QUANT_DEL_DEC_STATES winning RD values */
            loop {
                let mut min_max_Q25 = silk_int32_MAX;
                let mut max_min_Q25: i32 = 0;
                let mut ind_min_max = 0usize;
                let mut ind_max_min = 0usize;
                for j in 0..NSTATES {
                    if min_max_Q25 > RD_max_Q25[j] {
                        min_max_Q25 = RD_max_Q25[j];
                        ind_min_max = j;
                    }
                    if max_min_Q25 < RD_min_Q25[j] {
                        max_min_Q25 = RD_min_Q25[j];
                        ind_max_min = j;
                    }
                }
                if min_max_Q25 >= max_min_Q25 {
                    break;
                }
                /* copy ind_min_max to ind_max_min */
                ind_sort[ind_max_min] = ind_sort[ind_min_max] ^ NLSF_QUANT_DEL_DEC_STATES;
                RD_Q25[ind_max_min] = RD_Q25[ind_min_max + NSTATES];
                prev_out_Q10[ind_max_min] = prev_out_Q10[ind_min_max + NSTATES];
                RD_min_Q25[ind_max_min] = 0;
                RD_max_Q25[ind_min_max] = silk_int32_MAX;
                ind[ind_max_min] = ind[ind_min_max];
            }
            /* increment index if it comes from the upper half */
            for j in 0..NSTATES {
                *ind[j].at_mut(i) += silk_RSHIFT(ind_sort[j], NLSF_QUANT_DEL_DEC_STATES_LOG2) as i8;
            }
        }
    }

    /* last sample: find winner, copy indices and return RD value */
    let mut ind_tmp: usize = 0;
    let mut min_Q25 = silk_int32_MAX;
    for j in 0..2 * NSTATES {
        if min_Q25 > RD_Q25[j] {
            min_Q25 = RD_Q25[j];
            ind_tmp = j;
        }
    }
    for j in 0..order as usize {
        *indices.at_mut(j) = ind[ind_tmp & (NSTATES - 1)].at(j);
        debug_assert!(indices[j] as i32 >= -NLSF_QUANT_MAX_AMPLITUDE_EXT);
        debug_assert!(indices[j] as i32 <= NLSF_QUANT_MAX_AMPLITUDE_EXT);
    }
    *indices.at_mut(0) += silk_RSHIFT(ind_tmp as i32, NLSF_QUANT_DEL_DEC_STATES_LOG2) as i8;
    debug_assert!(indices[0] as i32 <= NLSF_QUANT_MAX_AMPLITUDE_EXT);
    debug_assert!(min_Q25 >= 0);
    min_Q25
}
