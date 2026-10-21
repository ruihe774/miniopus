//! Port of silk/NLSF_decode.c.

use crate::silk::NLSF_stabilize::silk_NLSF_stabilize;
use crate::silk::NLSF_unpack::silk_NLSF_unpack;
use crate::silk::define::*;
use crate::silk::macros::*;
use crate::silk::structs::silk_NLSF_CB_struct;

/// Predictive dequantizer for NLSF residuals.
/// `x_Q10`: output [ order ]; `indices`: quantization indices [ order ];
/// `pred_coef_Q8`: backward predictor coefs [ order ]; `quant_step_size_Q16`: quantization step size.
#[inline(always)]
fn silk_NLSF_residual_dequant(
    x_Q10: &mut [i16],
    indices: &[i8],
    pred_coef_Q8: &[u8],
    quant_step_size_Q16: i32,
    order: i16,
) {
    let mut out_Q10: i32 = 0;
    for i in (0..order as usize).rev() {
        let pred_Q10 = silk_RSHIFT(silk_SMULBB(out_Q10, pred_coef_Q8[i] as i32), 8);
        out_Q10 = silk_LSHIFT(indices[i] as i32, 10);
        if out_Q10 > 0 {
            out_Q10 = (out_Q10 - SILK_FIX_CONST(NLSF_QUANT_LEVEL_ADJ, 10)) as i16 as i32; /* silk_SUB16 */
        } else if out_Q10 < 0 {
            out_Q10 = (out_Q10 + SILK_FIX_CONST(NLSF_QUANT_LEVEL_ADJ, 10)) as i16 as i32; /* silk_ADD16 */
        }
        out_Q10 = silk_SMLAWB(pred_Q10, out_Q10, quant_step_size_Q16);
        x_Q10[i] = out_Q10 as i16;
    }
}

/// NLSF vector decoder.
/// `pNLSF_Q15`: quantized NLSF vector [ LPC_ORDER ] (O); `NLSFIndices`: codebook path vector
/// [ LPC_ORDER + 1 ]; `psNLSF_CB`: codebook object.
pub fn silk_NLSF_decode(pNLSF_Q15: &mut [i16], NLSFIndices: &[i8], psNLSF_CB: &silk_NLSF_CB_struct) {
    let mut pred_Q8 = [0u8; MAX_LPC_ORDER as usize];
    let mut ec_ix = [0i16; MAX_LPC_ORDER as usize];
    let mut res_Q10 = [0i16; MAX_LPC_ORDER as usize];
    let order = psNLSF_CB.order as i32;

    /* Unpack entropy table indices and predictor for current CB1 index */
    silk_NLSF_unpack(&mut ec_ix, &mut pred_Q8, psNLSF_CB, NLSFIndices[0] as i32);

    /* Predictive residual dequantizer */
    silk_NLSF_residual_dequant(
        &mut res_Q10,
        &NLSFIndices[1..],
        &pred_Q8,
        psNLSF_CB.quantStepSize_Q16 as i32,
        psNLSF_CB.order,
    );

    /* Apply inverse square-rooted weights to first stage and add to output */
    unsafe {
        let pCB_element = psNLSF_CB.CB1_NLSF_Q8.add((NLSFIndices[0] as i32 * order) as usize);
        let pCB_Wght_Q9 = psNLSF_CB.CB1_Wght_Q9.add((NLSFIndices[0] as i32 * order) as usize);
        for i in 0..order as usize {
            let NLSF_Q15_tmp = silk_ADD_LSHIFT32(
                // div_s: divisor is a codebook weight, positive in the tables
                silk_DIV32_16(silk_LSHIFT(res_Q10[i] as i32, 14), *pCB_Wght_Q9.add(i) as i32),
                *pCB_element.add(i) as i32,
                7,
            );
            pNLSF_Q15[i] = silk_LIMIT!(NLSF_Q15_tmp, 0, 32767) as i16;
        }
    }

    /* NLSF stabilization */
    silk_NLSF_stabilize(
        pNLSF_Q15,
        unsafe { core::slice::from_raw_parts(psNLSF_CB.deltaMin_Q15, order as usize + 1) },
        order,
    );
}
