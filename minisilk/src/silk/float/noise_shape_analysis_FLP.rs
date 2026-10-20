//! Port of silk/float/noise_shape_analysis_FLP.c.

use crate::silk::define::*;
use crate::silk::float::apply_sine_window_FLP::silk_apply_sine_window_FLP;
use crate::silk::float::autocorrelation_FLP::silk_autocorrelation_FLP;
use crate::silk::float::bwexpander_FLP::silk_bwexpander_FLP;
use crate::silk::float::energy_FLP::silk_energy_FLP;
use crate::silk::float::k2a_FLP::silk_k2a_FLP;
use crate::silk::float::schur_FLP::silk_schur_FLP;
use crate::silk::float::sigproc_flp::{silk_abs_float, silk_log2, silk_sigmoid};
use crate::silk::float::structs_flp::{silk_encoder_control_FLP, silk_encoder_state_FLP};
use crate::silk::float::warped_autocorrelation_FLP::silk_warped_autocorrelation_FLP;
use crate::silk::macros::*;
use crate::silk::tuning_parameters::*;

/* Compute gain to make warped filter coefficients have a zero mean log frequency response on a   */
/* non-warped frequency scale. (So that it can be implemented with a minimum-phase monic filter.) */
/* Note: A monic filter is one with the first coefficient equal to 1.0. In Silk we omit the first */
/* coefficient in an array of coefficients, for monic filters.                                    */
#[inline(always)]
fn warped_gain(coefs: &[f32], mut lambda: f32, order: i32) -> f32 {
    lambda = -lambda;
    let mut gain = coefs.at((order - 1) as usize);
    let mut i = order - 2;
    while i >= 0 {
        gain = lambda * gain + coefs.at(i as usize);
        i -= 1;
    }
    1.0f32 / (1.0f32 - lambda * gain)
}

/* Convert warped filter coefficients to monic pseudo-warped coefficients and limit maximum     */
/* amplitude of monic warped coefficients by using bandwidth expansion on the true coefficients */
#[inline(always)]
fn warped_true2monic_coefs(coefs: &mut [f32], lambda: f32, limit: f32, order: i32) {
    let order = order as usize;
    let mut ind = 0usize;

    /* Convert to monic coefficients */
    for i in (1..order).rev() {
        *coefs.at_mut(i - 1) -= lambda * coefs.at(i);
    }
    let mut gain = (1.0f32 - lambda * lambda) / (1.0f32 + lambda * coefs.at(0));
    for i in 0..order {
        *coefs.at_mut(i) *= gain;
    }

    /* Limit */
    for iter in 0..10 {
        /* Find maximum absolute value */
        let mut maxabs = -1.0f32;
        for i in 0..order {
            let tmp = silk_abs_float(coefs.at(i));
            if tmp > maxabs {
                maxabs = tmp;
                ind = i;
            }
        }
        if maxabs <= limit {
            /* Coefficients are within range - done */
            return;
        }

        /* Convert back to true warped coefficients */
        for i in 1..order {
            *coefs.at_mut(i - 1) += lambda * coefs.at(i);
        }
        gain = 1.0f32 / gain;
        for i in 0..order {
            *coefs.at_mut(i) *= gain;
        }

        /* Apply bandwidth expansion */
        let chirp = 0.99f32 - (0.8f32 + 0.1f32 * iter as f32) * (maxabs - limit) / (maxabs * (ind + 1) as f32);
        silk_bwexpander_FLP(coefs, order as i32, chirp);

        /* Convert to monic warped coefficients */
        for i in (1..order).rev() {
            *coefs.at_mut(i - 1) -= lambda * coefs.at(i);
        }
        gain = (1.0f32 - lambda * lambda) / (1.0f32 + lambda * coefs.at(0));
        for i in 0..order {
            *coefs.at_mut(i) *= gain;
        }
    }
    debug_assert!(false);
}

#[inline(always)]
fn limit_coefs(coefs: &mut [f32], limit: f32, order: i32) {
    let mut ind = 0usize;

    for iter in 0..10 {
        /* Find maximum absolute value */
        let mut maxabs = -1.0f32;
        for i in 0..order as usize {
            let tmp = silk_abs_float(coefs.at(i));
            if tmp > maxabs {
                maxabs = tmp;
                ind = i;
            }
        }
        if maxabs <= limit {
            /* Coefficients are within range - done */
            return;
        }

        /* Apply bandwidth expansion */
        let chirp = 0.99f32 - (0.8f32 + 0.1f32 * iter as f32) * (maxabs - limit) / (maxabs * (ind + 1) as f32);
        silk_bwexpander_FLP(coefs, order, chirp);
    }
    debug_assert!(false);
}

/// Compute noise shaping coefficients and initial gain values.
/// The input signal `x` of C is `&psEnc.x_buf[x_off..]` (frame_length + la_shape samples,
/// preceded by at least la_shape samples).
pub fn silk_noise_shape_analysis_FLP(
    psEnc: &mut silk_encoder_state_FLP,        /* I/O  Encoder state FLP                           */
    psEncCtrl: &mut silk_encoder_control_FLP, /* I/O  Encoder control FLP                         */
    pitch_res: &[f32],                        /* I    LPC residual from pitch analysis            */
    x_off: usize,                             /* I    Input signal [frame_length + la_shape]: psEnc->x_buf + x_off */
) {
    let mut x_windowed = [0f32; SHAPE_LPC_WIN_MAX as usize];
    let mut auto_corr = [0f32; (MAX_SHAPE_LPC_ORDER + 1) as usize];
    let mut rc = [0f32; (MAX_SHAPE_LPC_ORDER + 1) as usize];
    let psShapeSt = &mut psEnc.sShape;
    let sCmn = &mut psEnc.sCmn;
    let x_buf = &psEnc.x_buf;

    /* Point to start of first LPC analysis block */
    let mut x_ptr = x_off - sCmn.la_shape as usize;

    /****************/
    /* GAIN CONTROL */
    /****************/
    let mut SNR_adj_dB: f32 = sCmn.SNR_dB_Q7 as f32 * (1.0f32 / 128.0);

    /* Input quality is the average of the quality in the lowest two VAD bands */
    psEncCtrl.input_quality =
        0.5f32 * (sCmn.input_quality_bands_Q15[0] + sCmn.input_quality_bands_Q15[1]) as f32 * (1.0f32 / 32768.0);

    /* Coding quality level, between 0.0 and 1.0 */
    psEncCtrl.coding_quality = silk_sigmoid(0.25f32 * (SNR_adj_dB - 20.0));

    /* Reduce coding SNR during low speech activity */
    {
        let b = 1.0f32 - sCmn.speech_activity_Q8 as f32 * (1.0f32 / 256.0);
        SNR_adj_dB -= BG_SNR_DECR_dB * psEncCtrl.coding_quality * (0.5f32 + 0.5f32 * psEncCtrl.input_quality) * b * b;
    }

    if sCmn.indices.signalType as i32 == TYPE_VOICED {
        /* Reduce gains for periodic signals */
        SNR_adj_dB += HARM_SNR_INCR_dB * psEnc.LTPCorr;
    } else {
        /* For unvoiced signals and low-quality input, adjust the quality slower than SNR_dB setting */
        SNR_adj_dB +=
            (-0.4f32 * sCmn.SNR_dB_Q7 as f32 * (1.0f32 / 128.0) + 6.0f32) * (1.0f32 - psEncCtrl.input_quality);
    }

    /*************************/
    /* SPARSENESS PROCESSING */
    /*************************/
    /* Set quantizer offset */
    if sCmn.indices.signalType as i32 == TYPE_VOICED {
        /* Initially set to 0; may be overruled in process_gains(..) */
        sCmn.indices.quantOffsetType = 0;
    } else {
        /* Sparseness measure, based on relative fluctuations of energy per 2 milliseconds */
        let nSamples = 2 * FS_KHZ;
        let mut energy_variation: f32 = 0.0;
        let mut log_energy_prev: f32 = 0.0;
        let mut pitch_res_ptr = 0usize;
        let nSegs = silk_SMULBB(SUB_FRAME_LENGTH_MS, MAX_NB_SUBFR) / 2;
        for k in 0..nSegs {
            let nrg = nSamples as f32 + silk_energy_FLP(pitch_res.rg(pitch_res_ptr..), nSamples) as f32;
            let log_energy = silk_log2(nrg as f64);
            if k > 0 {
                energy_variation += silk_abs_float(log_energy - log_energy_prev);
            }
            log_energy_prev = log_energy;
            pitch_res_ptr += nSamples as usize;
        }

        /* Set quantization offset depending on sparseness measure */
        if energy_variation > ENERGY_VARIATION_THRESHOLD_QNT_OFFSET * (nSegs - 1) as f32 {
            sCmn.indices.quantOffsetType = 0;
        } else {
            sCmn.indices.quantOffsetType = 1;
        }
    }

    /*******************************/
    /* Control bandwidth expansion */
    /*******************************/
    /* More BWE for signals with high prediction gain */
    let mut strength: f32 = FIND_PITCH_WHITE_NOISE_FRACTION * psEncCtrl.predGain; /* between 0.0 and 1.0 */
    let BWExp: f32 = BANDWIDTH_EXPANSION / (1.0f32 + strength * strength);

    /* Slightly more warping in analysis will move quantization noise up in frequency, where it's better masked */
    let warping: f32 = sCmn.warping_Q16 as f32 / 65536.0f32 + 0.01f32 * psEncCtrl.coding_quality;

    /********************************************/
    /* Compute noise shaping AR coefs and gains */
    /********************************************/
    let order = sCmn.shapingLPCOrder;
    for k in 0..MAX_NB_SUBFR as usize {
        /* Apply window: sine slope followed by flat part followed by cosine slope */
        let flat_part = (FS_KHZ * 3) as usize;
        let slope_part = ((sCmn.shapeWinLength - flat_part as i32) / 2) as usize;

        silk_apply_sine_window_FLP(&mut x_windowed, x_buf.rg(x_ptr..), 1, slope_part as i32);
        let mut shift = slope_part;
        x_windowed.rg_mut(shift..shift + flat_part).copy_from_slice(x_buf.rg(x_ptr + shift..x_ptr + shift + flat_part));
        shift += flat_part;
        silk_apply_sine_window_FLP(x_windowed.rg_mut(shift..), x_buf.rg(x_ptr + shift..), 2, slope_part as i32);

        /* Update pointer: next LPC analysis block */
        x_ptr += MAX_SUB_FRAME_LENGTH as usize;

        if sCmn.warping_Q16 > 0 {
            /* Calculate warped auto correlation */
            silk_warped_autocorrelation_FLP(&mut auto_corr, &x_windowed, warping, sCmn.shapeWinLength, order);
        } else {
            /* Calculate regular auto correlation */
            silk_autocorrelation_FLP(&mut auto_corr, &x_windowed, sCmn.shapeWinLength, order + 1);
        }

        /* Add white noise, as a fraction of energy */
        auto_corr[0] += auto_corr[0] * SHAPE_WHITE_NOISE_FRACTION + 1.0f32;

        /* Convert correlations to prediction coefficients, and compute residual energy */
        let nrg = silk_schur_FLP(&mut rc, &auto_corr, order);
        let AR = psEncCtrl.AR.rg_mut(k * MAX_SHAPE_LPC_ORDER as usize..(k + 1) * MAX_SHAPE_LPC_ORDER as usize);
        silk_k2a_FLP(AR, &rc, order);
        psEncCtrl.Gains[k] = crate::silk::float::sigproc_flp::sqrt(nrg as f64) as f32;

        if sCmn.warping_Q16 > 0 {
            /* Adjust gain for warping */
            psEncCtrl.Gains[k] *= warped_gain(AR, warping, order);
        }

        /* Bandwidth expansion for synthesis filter shaping */
        silk_bwexpander_FLP(AR, order, BWExp);

        if sCmn.warping_Q16 > 0 {
            /* Convert to monic warped prediction coefficients and limit absolute values */
            warped_true2monic_coefs(AR, warping, 3.999f32, order);
        } else {
            /* Limit absolute values */
            limit_coefs(AR, 3.999f32, order);
        }
    }

    /*****************/
    /* Gain tweaking */
    /*****************/
    /* Increase gains during low speech activity */
    let gain_mult = libm::pow(2.0, (-0.16f32 * SNR_adj_dB) as f64) as f32;
    let gain_add = libm::pow(2.0, (0.16f32 * MIN_QGAIN_DB as f32) as f64) as f32;
    for k in 0..MAX_NB_SUBFR as usize {
        psEncCtrl.Gains[k] *= gain_mult;
        psEncCtrl.Gains[k] += gain_add;
    }

    /************************************************/
    /* Control low-frequency shaping and noise tilt */
    /************************************************/
    /* Less low frequency shaping for noisy inputs */
    strength = LOW_FREQ_SHAPING
        * (1.0f32
            + LOW_QUALITY_LOW_FREQ_SHAPING_DECR * (sCmn.input_quality_bands_Q15[0] as f32 * (1.0f32 / 32768.0) - 1.0f32));
    strength *= sCmn.speech_activity_Q8 as f32 * (1.0f32 / 256.0);
    let Tilt: f32;
    if sCmn.indices.signalType as i32 == TYPE_VOICED {
        /* Reduce low frequencies quantization noise for periodic signals, depending on pitch lag */
        /*f = 400; freqz([1, -0.98 + 2e-4 * f], [1, -0.97 + 7e-4 * f], 2^12, Fs); axis([0, 1000, -10, 1])*/
        for k in 0..MAX_NB_SUBFR as usize {
            let b = 0.2f32 / FS_KHZ as f32 + 3.0f32 / psEncCtrl.pitchL[k] as f32;
            psEncCtrl.LF_MA_shp[k] = -1.0f32 + b;
            psEncCtrl.LF_AR_shp[k] = 1.0f32 - b - b * strength;
        }
        Tilt = -HP_NOISE_COEF
            - (1.0f32 - HP_NOISE_COEF) * HARM_HP_NOISE_COEF * sCmn.speech_activity_Q8 as f32 * (1.0f32 / 256.0);
    } else {
        let b = 1.3f32 / FS_KHZ as f32;
        psEncCtrl.LF_MA_shp[0] = -1.0f32 + b;
        psEncCtrl.LF_AR_shp[0] = 1.0f32 - b - b * strength * 0.6f32;
        for k in 1..MAX_NB_SUBFR as usize {
            psEncCtrl.LF_MA_shp[k] = psEncCtrl.LF_MA_shp[0];
            psEncCtrl.LF_AR_shp[k] = psEncCtrl.LF_AR_shp[0];
        }
        Tilt = -HP_NOISE_COEF;
    }

    /****************************/
    /* HARMONIC SHAPING CONTROL */
    /****************************/
    let mut HarmShapeGain: f32;
    if sCmn.indices.signalType as i32 == TYPE_VOICED {
        /* Harmonic noise shaping */
        HarmShapeGain = HARMONIC_SHAPING;

        /* More harmonic noise shaping for high bitrates or noisy input */
        HarmShapeGain += HIGH_RATE_OR_LOW_QUALITY_HARMONIC_SHAPING
            * (1.0f32 - (1.0f32 - psEncCtrl.coding_quality) * psEncCtrl.input_quality);

        /* Less harmonic noise shaping for less periodic signals */
        HarmShapeGain *= crate::silk::float::sigproc_flp::sqrt(psEnc.LTPCorr as f64) as f32;
    } else {
        HarmShapeGain = 0.0f32;
    }

    /*************************/
    /* Smooth over subframes */
    /*************************/
    for k in 0..MAX_NB_SUBFR as usize {
        psShapeSt.HarmShapeGain_smth += SUBFR_SMTH_COEF * (HarmShapeGain - psShapeSt.HarmShapeGain_smth);
        psEncCtrl.HarmShapeGain[k] = psShapeSt.HarmShapeGain_smth;
        psShapeSt.Tilt_smth += SUBFR_SMTH_COEF * (Tilt - psShapeSt.Tilt_smth);
        psEncCtrl.Tilt[k] = psShapeSt.Tilt_smth;
    }
}
