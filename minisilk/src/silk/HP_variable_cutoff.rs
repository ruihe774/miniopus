//! Port of silk/HP_variable_cutoff.c.

use crate::silk::define::*;
use crate::silk::structs::silk_encoder_state;
use crate::silk::lin2log::silk_lin2log;
use crate::silk::macros::*;
use crate::silk::tuning_parameters::*;

/// High-pass filter with cutoff frequency adaptation based on pitch lag statistics
pub fn silk_HP_variable_cutoff(
    psEncC1: &mut silk_encoder_state, /* I/O  Encoder state                                      */
) {

    /* Adaptive cutoff frequency: estimate low end of pitch frequency range */
    if psEncC1.prevSignalType as i32 == TYPE_VOICED {
        /* difference, in log domain */
        // div_s: prevLag > 0 for a voiced frame
        let pitch_freq_Hz_Q16 = silk_DIV32_16(silk_LSHIFT(FS_KHZ * 1000, 16), psEncC1.prevLag);
        let mut pitch_freq_log_Q7 = silk_lin2log(pitch_freq_Hz_Q16) - (16 << 7);

        /* adjustment based on quality */
        let quality_Q15 = psEncC1.input_quality_bands_Q15[0];
        pitch_freq_log_Q7 = silk_SMLAWB(
            pitch_freq_log_Q7,
            silk_SMULWB(silk_LSHIFT(-quality_Q15, 2), quality_Q15),
            pitch_freq_log_Q7 - (silk_lin2log(SILK_FIX_CONST(VARIABLE_HP_MIN_CUTOFF_HZ as f64, 16)) - (16 << 7)),
        );

        /* delta_freq = pitch_freq_log - psEnc->variable_HP_smth1; */
        let mut delta_freq_Q7 = pitch_freq_log_Q7 - silk_RSHIFT(psEncC1.variable_HP_smth1_Q15, 8);
        if delta_freq_Q7 < 0 {
            /* less smoothing for decreasing pitch frequency, to track something close to the minimum */
            delta_freq_Q7 = silk_MUL(delta_freq_Q7, 3);
        }

        /* limit delta, to reduce impact of outliers in pitch estimation */
        delta_freq_Q7 = silk_LIMIT_32!(
            delta_freq_Q7,
            -SILK_FIX_CONST(VARIABLE_HP_MAX_DELTA_FREQ as f64, 7),
            SILK_FIX_CONST(VARIABLE_HP_MAX_DELTA_FREQ as f64, 7)
        );

        /* update smoother */
        psEncC1.variable_HP_smth1_Q15 = silk_SMLAWB(
            psEncC1.variable_HP_smth1_Q15,
            silk_SMULBB(psEncC1.speech_activity_Q8, delta_freq_Q7),
            SILK_FIX_CONST(VARIABLE_HP_SMTH_COEF1 as f64, 16),
        );

        /* limit frequency range */
        psEncC1.variable_HP_smth1_Q15 = silk_LIMIT_32!(
            psEncC1.variable_HP_smth1_Q15,
            silk_LSHIFT(silk_lin2log(VARIABLE_HP_MIN_CUTOFF_HZ), 8),
            silk_LSHIFT(silk_lin2log(VARIABLE_HP_MAX_CUTOFF_HZ), 8)
        );
    }
}
