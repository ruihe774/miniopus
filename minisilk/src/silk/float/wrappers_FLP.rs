//! Port of silk/float/wrappers_FLP.c.

use crate::silk::A2NLSF::silk_A2NLSF;
use crate::silk::NLSF2A::silk_NLSF2A;
use crate::silk::NSQ::silk_NSQ;
use crate::silk::NSQ_del_dec::silk_NSQ_del_dec;
use crate::silk::define::*;
use crate::silk::float::sigproc_flp::silk_float2int;
use crate::silk::float::structs_flp::{silk_encoder_control_FLP, silk_encoder_state_FLP};
use crate::silk::macros::*;
use crate::silk::process_NLSFs::silk_process_NLSFs;
use crate::silk::quant_LTP_gains::silk_quant_LTP_gains;
use crate::silk::structs::{SideInfoIndices, silk_encoder_state, silk_nsq_state};
use crate::silk::tables::silk_LTPScales_table_Q14;

const MAX_LPC_ORDER_: usize = MAX_LPC_ORDER as usize;
const MAX_NB_SUBFR_: usize = MAX_NB_SUBFR as usize;
const LTP_ORDER_: usize = LTP_ORDER as usize;
const MAX_SHAPE_LPC_ORDER_: usize = MAX_SHAPE_LPC_ORDER as usize;

/* Wrappers. Calls flp / fix code */

/// Convert AR filter coefficients to NLSF parameters
pub fn silk_A2NLSF_FLP(
    NLSF_Q15: &mut [i16], /* O    NLSF vector      [ LPC_order ]              */
    pAR: &[f32],          /* I    LPC coefficients [ LPC_order ]              */
    LPC_order: i32,       /* I    LPC order                                   */
) {
    let mut a_fix_Q16 = [0i32; MAX_LPC_ORDER_];

    for i in 0..LPC_order as usize {
        *a_fix_Q16.at_mut(i) = silk_float2int(pAR.at(i) * 65536.0f32);
    }

    silk_A2NLSF(NLSF_Q15, &mut a_fix_Q16, LPC_order);
}

/// Convert LSF parameters to AR prediction filter coefficients
pub fn silk_NLSF2A_FLP(
    pAR: &mut [f32],   /* O    LPC coefficients [ LPC_order ]              */
    NLSF_Q15: &[i16],  /* I    NLSF vector      [ LPC_order ]              */
    LPC_order: i32,    /* I    LPC order                                   */
) {
    let mut a_fix_Q12 = [0i16; MAX_LPC_ORDER_];

    silk_NLSF2A(&mut a_fix_Q12, NLSF_Q15, LPC_order);

    for i in 0..LPC_order as usize {
        *pAR.at_mut(i) = a_fix_Q12.at(i) as f32 * (1.0f32 / 4096.0f32);
    }
}

/// Floating-point NLSF processing wrapper
pub fn silk_process_NLSFs_FLP(
    psEncC: &mut silk_encoder_state,              /* I/O  Encoder state                               */
    PredCoef: &mut [[f32; MAX_LPC_ORDER_]; 2],    /* O    Prediction coefficients                     */
    NLSF_Q15: &mut [i16],                         /* I/O  Normalized LSFs (quant out) (0 - (2^15-1))  */
    prev_NLSF_Q15: &[i16],                        /* I    Previous Normalized LSFs (0 - (2^15-1))     */
) {
    let mut PredCoef_Q12 = [[0i16; MAX_LPC_ORDER_]; 2];

    silk_process_NLSFs(psEncC, &mut PredCoef_Q12, NLSF_Q15, prev_NLSF_Q15);

    for j in 0..2 {
        for i in 0..MAX_LPC_ORDER as usize {
            PredCoef[j][i] = PredCoef_Q12[j][i] as f32 * (1.0f32 / 4096.0f32);
        }
    }
}

/// Floating-point Silk NSQ wrapper.
///
/// # Safety
/// As in C, `psIndices`, `psNSQ`, `pulses` and `x` may point into `*psEnc` (they do in
/// silk_encode_frame_FLP); all must be valid: `pulses` for frame_length writes, `x` for
/// frame_length reads.
pub unsafe fn silk_NSQ_wrapper_FLP(
    psEnc: *mut silk_encoder_state_FLP,          /* I/O  Encoder state FLP                           */
    psEncCtrl: *const silk_encoder_control_FLP,  /* I/O  Encoder control FLP                         */
    psIndices: *mut SideInfoIndices,             /* I/O  Quantization indices                        */
    psNSQ: *mut silk_nsq_state,                  /* I/O  Noise Shaping Quantzation state             */
    pulses: *mut i8,                             /* O    Quantized pulse signal                      */
    x: *const f32,                               /* I    Prefiltered input signal                    */
) {
    unsafe {
        let mut x16 = [0i16; MAX_FRAME_LENGTH as usize];
        let mut Gains_Q16 = [0i32; MAX_NB_SUBFR_];
        let mut PredCoef_Q12 = [[0i16; MAX_LPC_ORDER_]; 2];
        let mut LTPCoef_Q14 = [0i16; LTP_ORDER_ * MAX_NB_SUBFR_];

        /* Noise shaping parameters */
        let mut AR_Q13 = [0i16; MAX_NB_SUBFR_ * MAX_SHAPE_LPC_ORDER_];
        let mut LF_shp_Q14 = [0i32; MAX_NB_SUBFR_]; /* Packs two int16 coefficients per int32 value             */
        let mut Tilt_Q14 = [0i32; MAX_NB_SUBFR_];
        let mut HarmShapeGain_Q14 = [0i32; MAX_NB_SUBFR_];

        let psEncC: *const silk_encoder_state = core::ptr::addr_of!((*psEnc).sCmn);
        let nb_subfr = MAX_NB_SUBFR as usize;
        let psEncCtrl = &*psEncCtrl;

        /* Convert control struct to fix control struct */
        /* Noise shape parameters */
        for i in 0..nb_subfr {
            for j in 0..(*psEncC).shapingLPCOrder as usize {
                AR_Q13[i * MAX_SHAPE_LPC_ORDER_ + j] =
                    silk_float2int(psEncCtrl.AR[i * MAX_SHAPE_LPC_ORDER_ + j] * 8192.0f32) as i16;
            }
        }

        for i in 0..nb_subfr {
            LF_shp_Q14[i] = silk_LSHIFT32(silk_float2int(psEncCtrl.LF_AR_shp[i] * 16384.0f32), 16)
                | silk_float2int(psEncCtrl.LF_MA_shp[i] * 16384.0f32) as u16 as i32;
            Tilt_Q14[i] = silk_float2int(psEncCtrl.Tilt[i] * 16384.0f32);
            HarmShapeGain_Q14[i] = silk_float2int(psEncCtrl.HarmShapeGain[i] * 16384.0f32);
        }
        let Lambda_Q10 = silk_float2int(psEncCtrl.Lambda * 1024.0f32);

        /* prediction and coding parameters */
        for i in 0..nb_subfr * LTP_ORDER_ {
            LTPCoef_Q14[i] = silk_float2int(psEncCtrl.LTPCoef[i] * 16384.0f32) as i16;
        }

        for j in 0..2 {
            for i in 0..MAX_LPC_ORDER as usize {
                PredCoef_Q12[j][i] = silk_float2int(psEncCtrl.PredCoef[j][i] * 4096.0f32) as i16;
            }
        }

        for i in 0..nb_subfr {
            Gains_Q16[i] = silk_float2int(psEncCtrl.Gains[i] * 65536.0f32);
            debug_assert!(Gains_Q16[i] > 0);
        }

        let LTP_scale_Q14 = if (*psIndices).signalType as i32 == TYPE_VOICED {
            silk_LTPScales_table_Q14[(*psIndices).LTP_scaleIndex as usize] as i32
        } else {
            0
        };

        /* Convert input to fix */
        for i in 0..MAX_FRAME_LENGTH as usize {
            x16[i] = silk_float2int(*x.add(i)) as i16;
        }

        /* PredCoef_Q12[ 0 ] in C, read as [ 2 * MAX_LPC_ORDER ] */
        let PredCoef_flat: &[i16] = core::slice::from_raw_parts(PredCoef_Q12.as_ptr() as *const i16, 2 * MAX_LPC_ORDER_);

        /* Call NSQ */
        if (*psEncC).nStatesDelayedDecision > 1 || (*psEncC).warping_Q16 > 0 {
            silk_NSQ_del_dec(
                psEncC,
                psNSQ,
                psIndices,
                &x16,
                pulses,
                PredCoef_flat,
                &LTPCoef_Q14,
                &AR_Q13,
                &HarmShapeGain_Q14,
                &Tilt_Q14,
                &LF_shp_Q14,
                &Gains_Q16,
                &psEncCtrl.pitchL,
                Lambda_Q10,
                LTP_scale_Q14,
            );
        } else {
            silk_NSQ(
                psEncC,
                psNSQ,
                psIndices,
                &x16,
                pulses,
                PredCoef_flat,
                &LTPCoef_Q14,
                &AR_Q13,
                &HarmShapeGain_Q14,
                &Tilt_Q14,
                &LF_shp_Q14,
                &Gains_Q16,
                &psEncCtrl.pitchL,
                Lambda_Q10,
                LTP_scale_Q14,
            );
        }
    }
}

/// Floating-point Silk LTP quantiation wrapper
pub fn silk_quant_LTP_gains_FLP(
    B: &mut [f32; MAX_NB_SUBFR_ * LTP_ORDER_], /* O    Quantized LTP gains                         */
    cbk_index: &mut [i8; MAX_NB_SUBFR_],       /* O    Codebook index                              */
    periodicity_index: &mut i8,                /* O    Periodicity index                           */
    sum_log_gain_Q7: &mut i32,                 /* I/O  Cumulative max prediction gain              */
    pred_gain_dB: &mut f32,                    /* O    LTP prediction gain                         */
    XX: &[f32],                                /* I    Correlation matrix [ MAX_NB_SUBFR * LTP_ORDER * LTP_ORDER ] */
    xX: &[f32],                                /* I    Correlation vector [ MAX_NB_SUBFR * LTP_ORDER ] */
    subfr_len: i32,                            /* I    Number of samples per subframe              */
    nb_subfr: i32,                             /* I    Number of subframes                         */
) {
    let mut pred_gain_dB_Q7 = 0i32;
    let mut B_Q14 = [0i16; MAX_NB_SUBFR_ * LTP_ORDER_];
    let mut XX_Q17 = [0i32; MAX_NB_SUBFR_ * LTP_ORDER_ * LTP_ORDER_];
    let mut xX_Q17 = [0i32; MAX_NB_SUBFR_ * LTP_ORDER_];

    let nb_subfr_ = nb_subfr as usize;
    for i in 0..nb_subfr_ * LTP_ORDER_ * LTP_ORDER_ {
        *XX_Q17.at_mut(i) = silk_float2int(XX.at(i) * 131072.0f32);
    }
    for i in 0..nb_subfr_ * LTP_ORDER_ {
        *xX_Q17.at_mut(i) = silk_float2int(xX.at(i) * 131072.0f32);
    }

    silk_quant_LTP_gains(
        &mut B_Q14,
        cbk_index,
        periodicity_index,
        sum_log_gain_Q7,
        &mut pred_gain_dB_Q7,
        &XX_Q17,
        &xX_Q17,
        subfr_len,
        nb_subfr,
    );

    for i in 0..nb_subfr_ * LTP_ORDER_ {
        *B.at_mut(i) = B_Q14.at(i) as f32 * (1.0f32 / 16384.0f32);
    }

    *pred_gain_dB = pred_gain_dB_Q7 as f32 * (1.0f32 / 128.0f32);
}
