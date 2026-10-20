//! Port of silk/encode_pulses.c.

use crate::celt::entcode::ec_enc;
use crate::celt::entenc::ec_enc_icdf;
use crate::silk::code_signs::silk_encode_signs;
use crate::silk::define::*;
use crate::silk::macros::*;
use crate::silk::shell_coder::silk_shell_encoder;
use crate::silk::tables::*;
use crate::stack;

/*********************************************/
/* Encode quantization indices of excitation */
/*********************************************/

/// `pulses_comb` and `pulses_in` may be the same buffer (in-place combination; the write
/// index never passes the read index).
#[inline(always)]
unsafe fn combine_and_check(
    pulses_comb: *mut i32,   /* O                                   */
    pulses_in: *const i32,   /* I                                   */
    max_pulses: i32,         /* I    max value for sum of pulses    */
    len: usize,              /* I    number of output values        */
) -> i32 {
    unsafe {
        for k in 0..len {
            let sum = *pulses_in.add(2 * k) + *pulses_in.add(2 * k + 1);
            if sum > max_pulses {
                return 1;
            }
            *pulses_comb.add(k) = sum;
        }
    }
    0
}

/// Encode quantization indices of excitation.
///
/// `pulses` holds `frame_length` quantization indices. (The C code zero-pads
/// `pulses[frame_length..frame_length+16]` for 10 ms frames at 12 kHz; that case does not
/// occur here and `frame_length` must be a multiple of 16.)
pub fn silk_encode_pulses(psRangeEnc: &mut ec_enc, signalType: i32, quantOffsetType: i32, pulses: &[i8], frame_length: i32) {
    let mut RateLevelIndex = 0usize;
    let mut pulses_comb = [0i32; 8];

    let st = stack::save();

    /****************************/
    /* Prepare for shell coding */
    /****************************/
    /* Calculate number of shell blocks */
    debug_assert!(1 << LOG2_SHELL_CODEC_FRAME_LENGTH == SHELL_CODEC_FRAME_LENGTH);
    let iter = silk_RSHIFT(frame_length, LOG2_SHELL_CODEC_FRAME_LENGTH) as usize;
    debug_assert!(iter as i32 * SHELL_CODEC_FRAME_LENGTH == frame_length);

    /* Take the absolute value of the pulses */
    let abs_pulses = st.alloc::<i32>(iter * SHELL_CODEC_FRAME_LENGTH as usize);
    debug_assert!(SHELL_CODEC_FRAME_LENGTH & 3 == 0);
    for i in 0..iter * SHELL_CODEC_FRAME_LENGTH as usize {
        *abs_pulses.at_mut(i) = (pulses.at(i) as i32).abs();
    }

    /* Calc sum pulses per shell code frame */
    let sum_pulses = st.alloc::<i32>(iter);
    let nRshifts = st.alloc::<i32>(iter);
    let mut abs_pulses_off = 0usize;
    for i in 0..iter {
        *nRshifts.at_mut(i) = 0;

        loop {
            let abs_pulses_ptr = abs_pulses.rg_mut(abs_pulses_off..abs_pulses_off + SHELL_CODEC_FRAME_LENGTH as usize);
            let comb = pulses_comb.as_mut_ptr();
            let scale_down = unsafe {
                /* 1+1 -> 2 */
                let mut sd = combine_and_check(comb, abs_pulses_ptr.as_ptr(), silk_max_pulses_table[0] as i32, 8);
                /* 2+2 -> 4 */
                sd += combine_and_check(comb, comb, silk_max_pulses_table[1] as i32, 4);
                /* 4+4 -> 8 */
                sd += combine_and_check(comb, comb, silk_max_pulses_table[2] as i32, 2);
                /* 8+8 -> 16 */
                sd += combine_and_check(sum_pulses.at_mut(i), comb, silk_max_pulses_table[3] as i32, 1);
                sd
            };

            if scale_down != 0 {
                /* We need to downscale the quantization signal */
                *nRshifts.at_mut(i) += 1;
                for k in 0..SHELL_CODEC_FRAME_LENGTH as usize {
                    *abs_pulses_ptr.at_mut(k) = silk_RSHIFT(abs_pulses_ptr.at(k), 1);
                }
            } else {
                /* Jump out of while(1) loop and go to next shell coding frame */
                break;
            }
        }
        abs_pulses_off += SHELL_CODEC_FRAME_LENGTH as usize;
    }

    /**************/
    /* Rate level */
    /**************/
    /* find rate level that leads to fewest bits for coding of pulses per block info */
    let mut minSumBits_Q5 = silk_int32_MAX;
    for k in 0..(N_RATE_LEVELS - 1) as usize {
        let nBits_ptr = &silk_pulses_per_block_BITS_Q5[k];
        let mut sumBits_Q5 = silk_rate_levels_BITS_Q5[(signalType >> 1) as usize][k] as i32;
        for i in 0..iter {
            if nRshifts.at(i) > 0 {
                sumBits_Q5 += nBits_ptr.at(SILK_MAX_PULSES as usize + 1) as i32;
            } else {
                sumBits_Q5 += nBits_ptr.at(sum_pulses.at(i) as usize) as i32;
            }
        }
        if sumBits_Q5 < minSumBits_Q5 {
            minSumBits_Q5 = sumBits_Q5;
            RateLevelIndex = k;
        }
    }
    ec_enc_icdf(psRangeEnc, RateLevelIndex as i32, &silk_rate_levels_iCDF[(signalType >> 1) as usize], 8);

    /***************************************************/
    /* Sum-Weighted-Pulses Encoding                    */
    /***************************************************/
    let cdf_ptr = &silk_pulses_per_block_iCDF[RateLevelIndex];
    for i in 0..iter {
        if nRshifts.at(i) == 0 {
            ec_enc_icdf(psRangeEnc, sum_pulses.at(i), cdf_ptr, 8);
        } else {
            ec_enc_icdf(psRangeEnc, SILK_MAX_PULSES + 1, cdf_ptr, 8);
            for _k in 0..nRshifts.at(i) - 1 {
                ec_enc_icdf(psRangeEnc, SILK_MAX_PULSES + 1, &silk_pulses_per_block_iCDF[(N_RATE_LEVELS - 1) as usize], 8);
            }
            ec_enc_icdf(psRangeEnc, sum_pulses.at(i), &silk_pulses_per_block_iCDF[(N_RATE_LEVELS - 1) as usize], 8);
        }
    }

    /******************/
    /* Shell Encoding */
    /******************/
    for i in 0..iter {
        if sum_pulses.at(i) > 0 {
            silk_shell_encoder(psRangeEnc, abs_pulses.rg(i * SHELL_CODEC_FRAME_LENGTH as usize..));
        }
    }

    /****************/
    /* LSB Encoding */
    /****************/
    for i in 0..iter {
        if nRshifts.at(i) > 0 {
            let pulses_ptr = pulses.rg(i * SHELL_CODEC_FRAME_LENGTH as usize..);
            let nLS = nRshifts.at(i) - 1;
            for k in 0..SHELL_CODEC_FRAME_LENGTH as usize {
                let abs_q = (pulses_ptr.at(k) as i32).abs() as i8 as i32;
                for j in (1..=nLS).rev() {
                    let bit = silk_RSHIFT(abs_q, j) & 1;
                    ec_enc_icdf(psRangeEnc, bit, &silk_lsb_iCDF, 8);
                }
                let bit = abs_q & 1;
                ec_enc_icdf(psRangeEnc, bit, &silk_lsb_iCDF, 8);
            }
        }
    }

    /****************/
    /* Encode signs */
    /****************/
    silk_encode_signs(psRangeEnc, pulses, frame_length, signalType, quantOffsetType, sum_pulses);
}
