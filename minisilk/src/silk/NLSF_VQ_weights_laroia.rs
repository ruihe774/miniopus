//! Port of silk/NLSF_VQ_weights_laroia.c.

use crate::silk::define::*;
use crate::silk::macros::*;

/*
R. Laroia, N. Phamdo and N. Farvardin, "Robust and Efficient Quantization of Speech LSP
Parameters Using Structured Vector Quantization", Proc. IEEE Int. Conf. Acoust., Speech,
Signal Processing, pp. 641-644, 1991.
*/

/// Laroia low complexity NLSF weights.
/// `pNLSFW_Q_OUT`: input vector weights [D] (O); `pNLSF_Q15`: input vector [D];
/// `D`: input vector dimension (even).
pub fn silk_NLSF_VQ_weights_laroia(pNLSFW_Q_OUT: &mut [i16], pNLSF_Q15: &[i16], D: i32) {
    debug_assert!(D > 0);
    debug_assert!((D & 1) == 0);
    let D = D as usize;

    /* First value */
    let mut tmp1_int = silk_max_int(pNLSF_Q15[0] as i32, 1);
    // div_s: divisor clamped to >= 1 above
    tmp1_int = silk_DIV32_16(1i32 << (15 + NLSF_W_Q), tmp1_int);
    let mut tmp2_int = silk_max_int(pNLSF_Q15[1] as i32 - pNLSF_Q15[0] as i32, 1);
    // div_s: divisor clamped to >= 1 above
    tmp2_int = silk_DIV32_16(1i32 << (15 + NLSF_W_Q), tmp2_int);
    pNLSFW_Q_OUT[0] = silk_min_int(tmp1_int + tmp2_int, silk_int16_MAX) as i16;
    debug_assert!(pNLSFW_Q_OUT[0] > 0);

    /* Main loop */
    let mut k = 1;
    while k < D - 1 {
        tmp1_int = silk_max_int(pNLSF_Q15[k + 1] as i32 - pNLSF_Q15[k] as i32, 1);
        // div_s: divisor clamped to >= 1 above
        tmp1_int = silk_DIV32_16(1i32 << (15 + NLSF_W_Q), tmp1_int);
        pNLSFW_Q_OUT[k] = silk_min_int(tmp1_int + tmp2_int, silk_int16_MAX) as i16;
        debug_assert!(pNLSFW_Q_OUT[k] > 0);

        tmp2_int = silk_max_int(pNLSF_Q15[k + 2] as i32 - pNLSF_Q15[k + 1] as i32, 1);
        // div_s: divisor clamped to >= 1 above
        tmp2_int = silk_DIV32_16(1i32 << (15 + NLSF_W_Q), tmp2_int);
        pNLSFW_Q_OUT[k + 1] = silk_min_int(tmp1_int + tmp2_int, silk_int16_MAX) as i16;
        debug_assert!(pNLSFW_Q_OUT[k + 1] > 0);
        k += 2;
    }

    /* Last value */
    tmp1_int = silk_max_int((1 << 15) - pNLSF_Q15[D - 1] as i32, 1);
    // div_s: divisor clamped to >= 1 above
    tmp1_int = silk_DIV32_16(1i32 << (15 + NLSF_W_Q), tmp1_int);
    pNLSFW_Q_OUT[D - 1] = silk_min_int(tmp1_int + tmp2_int, silk_int16_MAX) as i16;
    debug_assert!(pNLSFW_Q_OUT[D - 1] > 0);
}
