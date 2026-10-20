//! Port of silk/wasm/NSQ_del_dec_wasm.c.
//!
//! `silk_NSQ_del_dec` is the WASM kernel `silk_NSQ_del_dec_wasm`. The C kernel falls back to
//! the scalar `silk_NSQ_del_dec_c` when `nStatesDelayedDecision > 4`; the encoder never
//! configures more than 4 states (silk/control_codec.c), so that fallback is a
//! `debug_assert!` here and the scalar version is not ported.
//!
//! The delayed-decision states are processed in parallel, one state per
//! 32-bit lane, so up to four states are supported. Unused lanes are
//! computed along but masked out of all decisions. Everything is bit-exact
//! with silk_NSQ_del_dec_c(), including the wrap-around and saturation
//! behaviour of the C macros.

use core::arch::wasm32::*;

use crate::silk::LPC_analysis_filter::silk_LPC_analysis_filter;
use crate::silk::NSQ::{LTP_ORDER_, MAX_LPC_ORDER_, MAX_SHAPE_LPC_ORDER_, NSQ_LPC_BUF_LENGTH_};
use crate::silk::define::*;
use crate::silk::macros::*;
use crate::silk::structs::{SideInfoIndices, silk_encoder_state, silk_nsq_state};
use crate::silk::tables::silk_Quantization_Offsets_Q10;
use crate::stack;

const WASM_DEL_DEC_STATES: usize = 4;
const DECISION_DELAY_: usize = DECISION_DELAY as usize;
const MAX_SUB_FRAME_LENGTH_: usize = MAX_SUB_FRAME_LENGTH as usize;

/* silk_noise_shape_quantizer_del_dec_wasm() assumes QUANT_LEVEL_ADJUST_Q10 == 80 */
const _: () = assert!(QUANT_LEVEL_ADJUST_Q10 == 80);

#[derive(Clone, Copy)]
struct NSQ_del_decs_wasm_struct {
    sLPC_Q14: [[i32; WASM_DEL_DEC_STATES]; MAX_SUB_FRAME_LENGTH_ + NSQ_LPC_BUF_LENGTH_],
    RandState: [[i32; WASM_DEL_DEC_STATES]; DECISION_DELAY_],
    Q_Q10: [[i32; WASM_DEL_DEC_STATES]; DECISION_DELAY_],
    Xq_Q14: [[i32; WASM_DEL_DEC_STATES]; DECISION_DELAY_],
    Pred_Q15: [[i32; WASM_DEL_DEC_STATES]; DECISION_DELAY_],
    Shape_Q14: [[i32; WASM_DEL_DEC_STATES]; DECISION_DELAY_],
    sAR2_Q14: [[i32; WASM_DEL_DEC_STATES]; MAX_SHAPE_LPC_ORDER_],
    LF_AR_Q14: [i32; WASM_DEL_DEC_STATES],
    Diff_Q14: [i32; WASM_DEL_DEC_STATES],
    Seed: [i32; WASM_DEL_DEC_STATES],
    SeedInit: [i32; WASM_DEL_DEC_STATES],
    RD_Q10: [i32; WASM_DEL_DEC_STATES],
}

#[inline(always)]
fn LD(p: &[i32; WASM_DEL_DEC_STATES]) -> v128 {
    // SAFETY: `p` is a reference to exactly 4 i32 (WASM_DEL_DEC_STATES == 4), i.e. 16 readable
    // bytes; v128_load has no alignment requirement.
    unsafe { v128_load(p.as_ptr() as *const v128) }
}

#[inline(always)]
fn ST(p: &mut [i32; WASM_DEL_DEC_STATES], v: v128) {
    // SAFETY: as in `LD`: `p` is exactly 16 writable bytes.
    unsafe { v128_store(p.as_mut_ptr() as *mut v128, v) }
}

/// p = silk_SMULWW_wasm( p, g )
#[inline(always)]
fn scale_row(p: &mut [i32; WASM_DEL_DEC_STATES], g: v128) {
    let v = LD(p);
    ST(p, silk_SMULWW_wasm(v, g));
}

/// mask ? a : b, for masks whose lanes are all ones or all zeros. Under that
/// condition relaxed_laneselect gives the same result on every
/// implementation.
#[inline(always)]
fn silk_select_wasm(a: v128, b: v128, mask: v128) -> v128 {
    #[cfg(target_feature = "relaxed-simd")]
    {
        i32x4_relaxed_laneselect(a, b, mask)
    }
    #[cfg(not(target_feature = "relaxed-simd"))]
    {
        v128_bitselect(a, b, mask)
    }
}

/// Byte shuffle with a run-time selector whose indices are all in 0..15, for
/// which relaxed_swizzle is fully specified.
#[inline(always)]
fn silk_swizzle_wasm(a: v128, sel: v128) -> v128 {
    #[cfg(target_feature = "relaxed-simd")]
    {
        i8x16_relaxed_swizzle(a, sel)
    }
    #[cfg(not(target_feature = "relaxed-simd"))]
    {
        i8x16_swizzle(a, sel)
    }
}

/// silk_SMULWB() on each lane: ( a * (opus_int16)b ) >> 16, with b_sx holding
/// the sign-extended 16-bit coefficient. Split a into its signed high and
/// unsigned low halves; both partial products fit in 32 bits.
#[inline(always)]
fn silk_SMULWB_wasm(a: v128, b_sx: v128) -> v128 {
    let hi = i32x4_mul(i32x4_shr(a, 16), b_sx);
    let lo = i32x4_mul(v128_and(a, i32x4_splat(0xFFFF)), b_sx);
    i32x4_add(hi, i32x4_shr(lo, 16))
}

/// Same as silk_SMULWB_wasm(), with shorter latency for the serial allpass
/// chain: i32x4.dot_i16x8_s multiplies the 16-bit halves directly.
/// b_hi = b << 16 and b_lo = b & 0xFFFF place the coefficient next to the
/// high or low half of each lane. The low half is taken as signed by the dot
/// product, which is corrected by adding b when its top bit is set.
#[inline(always)]
fn silk_SMULWB_dot_wasm(a: v128, b_hi: v128, b_lo: v128, b_sx: v128) -> v128 {
    let hi = i32x4_dot_i16x8(a, b_hi);
    let lo = i32x4_shr(i32x4_dot_i16x8(a, b_lo), 16);
    let adj = v128_and(i32x4_shr(i32x4_shl(a, 16), 31), b_sx);
    i32x4_add(i32x4_add(hi, lo), adj)
}

/// silk_SMULWW() on each lane: ( a * b ) >> 16, truncated to 32 bits.
#[inline(always)]
fn silk_SMULWW_wasm(a: v128, b: v128) -> v128 {
    let lo = i64x2_shr(i64x2_extmul_low_i32x4(a, b), 16);
    let hi = i64x2_shr(i64x2_extmul_high_i32x4(a, b), 16);
    i32x4_shuffle::<0, 2, 4, 6>(lo, hi)
}

/// silk_ADD_SAT32() and silk_SUB_SAT32()
#[inline(always)]
fn silk_ADD_SAT32_wasm(a: v128, b: v128) -> v128 {
    let r = i32x4_add(a, b);
    let ovf = i32x4_shr(v128_and(v128_xor(a, r), v128_xor(b, r)), 31);
    let sat = v128_xor(i32x4_shr(a, 31), i32x4_splat(silk_int32_MAX));
    silk_select_wasm(sat, r, ovf)
}

#[inline(always)]
fn silk_SUB_SAT32_wasm(a: v128, b: v128) -> v128 {
    let r = i32x4_sub(a, b);
    let ovf = i32x4_shr(v128_and(v128_xor(a, r), v128_xor(a, b)), 31);
    let sat = v128_xor(i32x4_shr(a, 31), i32x4_splat(silk_int32_MAX));
    silk_select_wasm(sat, r, ovf)
}

/// silk_RSHIFT_ROUND() for shift > 1, which cannot overflow
#[inline(always)]
fn silk_RSHIFT_ROUND_wasm(a: v128, shift: u32) -> v128 {
    i32x4_shr(i32x4_add(i32x4_shr(a, shift - 1), i32x4_splat(1)), 1)
}

/// Index of the first lane holding the minimum (maximum) of num, which is
/// what the strict comparisons of the C search loops find. Lanes set in mask
/// (unused states) are ignored; lane 0 is always in use.
#[inline(always)]
fn silk_first_min_wasm(mut num: v128, mask: v128, val: &mut i32) -> usize {
    num = silk_select_wasm(i32x4_splat(silk_int32_MAX), num, mask);
    let mut m = i32x4_min(num, i32x4_shuffle::<2, 3, 0, 1>(num, num));
    m = i32x4_min(m, i32x4_shuffle::<1, 0, 3, 2>(m, m));
    *val = i32x4_extract_lane::<0>(m);
    (i32x4_bitmask(i32x4_eq(num, m)) as u32).trailing_zeros() as usize
}

#[inline(always)]
fn silk_first_max_wasm(mut num: v128, mask: v128, val: &mut i32) -> usize {
    num = silk_select_wasm(i32x4_splat(silk_int32_MIN), num, mask);
    let mut m = i32x4_max(num, i32x4_shuffle::<2, 3, 0, 1>(num, num));
    m = i32x4_max(m, i32x4_shuffle::<1, 0, 3, 2>(m, m));
    *val = i32x4_extract_lane::<0>(m);
    (i32x4_bitmask(i32x4_eq(num, m)) as u32).trailing_zeros() as usize
}

/// Noise shape quantizer for one subframe
#[inline(always)]
unsafe fn silk_noise_shape_quantizer_del_dec_wasm(
    NSQ: &mut silk_nsq_state,                   /* I/O  NSQ state                           */
    psDelDec: &mut NSQ_del_decs_wasm_struct,    /* I/O  Delayed decision states             */
    signalType: i32,                            /* I    Signal type                         */
    x_Q10: &[i32],                              /* I                                        */
    pulses: *mut i8,                            /* O                                        */
    xq_off: usize,                              /* O    xq = &NSQ->xq[ xq_off ]             */
    sLTP_Q15: &mut [i32],                       /* I/O  LTP filter state                    */
    delayedGain_Q10: &mut [i32; DECISION_DELAY_], /* I/O  Gain delay buffer                 */
    a_Q12: &[i16],                              /* I    Short term prediction coefs         */
    b_Q14: &[i16],                              /* I    Long term prediction coefs          */
    AR_shp_Q13: &[i16],                         /* I    Noise shaping coefs                 */
    lag: i32,                                   /* I    Pitch lag                           */
    HarmShapeFIRPacked_Q14: i32,                /* I                                        */
    Tilt_Q14: i32,                              /* I    Spectral tilt                       */
    LF_shp_Q14: i32,                            /* I                                        */
    Gain_Q16: i32,                              /* I                                        */
    Lambda_Q10: i32,                            /* I                                        */
    offset_Q10: i32,                            /* I                                        */
    length: i32,                                /* I    Input length                        */
    subfr: i32,                                 /* I    Subframe number                     */
    shapingLPCOrder: i32,                       /* I    Shaping LPC filter order            */
    predictLPCOrder: i32,                       /* I    Prediction filter order             */
    warping_Q16: i32,                           /* I                                        */
    MaskDelDec: v128,                           /* I    All ones in the lanes of unused states */
    smpl_buf_idx: &mut i32,                     /* I/O  Index to newest samples in buffers  */
    decisionDelay: i32,                         /* I                                        */
) {
    // SAFETY: all vector accesses go through `LD`/`ST` on `[i32; 4]` rows of `psDelDec`, indexed
    // with ordinary (bounds-checked) slice indexing. The raw pointers are bounded by the
    // debug_asserts below: sLTP_shp / sLTP_Q15 / xq are written at the running buffer indices,
    // which stay within their arrays for `length` samples (sLTP_buf_idx + length <=
    // sLTP_Q15.len(), sLTP_shp_buf_idx + length <= sLTP_shp_Q14.len(), xq_off + length <=
    // xq.len()). `pulses` and `xq` are written at offsets i - decisionDelay in
    // [-decisionDelay, length - decisionDelay) (only once i >= decisionDelay or subfr > 0, and
    // xq_off >= decisionDelay); `pulses` has room for that as the caller passes the subframe
    // start of a frame_length buffer, and negative offsets only occur after the first
    // decisionDelay samples of the frame (subfr > 0). The lag pointers read up to 4 entries below their start,
    // non-negative offsets by the asserts on lag.
    unsafe {
        let mut Winner_ind: usize;
        let mut RDmin_ind: usize;
        let mut RDmax_ind: usize;
        let mut last_smple_idx: i32;
        let mut LTP_pred_Q14: i32;
        let mut n_LTP_Q14: i32;
        let mut RDmin_Q10: i32 = 0;
        let mut RDmax_Q10: i32 = 0;
        let mut a_Q12_sx = [i32x4_splat(0); MAX_LPC_ORDER_];
        let mut AR_shp_sx = [i32x4_splat(0); MAX_SHAPE_LPC_ORDER_];
        let mut Seed: v128;
        let mut LF_AR_Q14: v128;
        let mut Diff_Q14: v128;
        let mut RD_Q10: v128;
        let lane_idx = i32x4(0, 1, 2, 3);
        let byte_idx = i8x16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let warped = warping_Q16 as i16 != 0;

        debug_assert!((shapingLPCOrder & 1) == 0); /* check that order is even */
        debug_assert!(shapingLPCOrder >= 2 && shapingLPCOrder <= MAX_SHAPE_LPC_ORDER);
        debug_assert!(predictLPCOrder == 10 || predictLPCOrder == 16);
        debug_assert!(NSQ.sLTP_shp_buf_idx - lag + HARM_SHAPE_FIR_TAPS / 2 - 2 >= 0 || lag <= 0);
        debug_assert!(NSQ.sLTP_buf_idx - lag + LTP_ORDER / 2 - 4 >= 0 || signalType != TYPE_VOICED);
        debug_assert!(NSQ.sLTP_buf_idx + length <= sLTP_Q15.len() as i32);
        debug_assert!(NSQ.sLTP_shp_buf_idx + length <= NSQ.sLTP_shp_Q14.len() as i32);
        debug_assert!(xq_off + length as usize <= NSQ.xq.len() && xq_off >= decisionDelay as usize);

        let sLTP_shp = NSQ.sLTP_shp_Q14.as_mut_ptr();
        let sLTP_Q15p = sLTP_Q15.as_mut_ptr();
        let xq = NSQ.xq.as_mut_ptr().add(xq_off);

        let mut shp_lag_ptr = sLTP_shp.wrapping_offset((NSQ.sLTP_shp_buf_idx - lag + HARM_SHAPE_FIR_TAPS / 2) as isize);
        let mut pred_lag_ptr = sLTP_Q15p.wrapping_offset((NSQ.sLTP_buf_idx - lag + LTP_ORDER / 2) as isize);
        let Gain_Q10 = silk_RSHIFT(Gain_Q16, 6);

        for j in 0..predictLPCOrder as usize {
            a_Q12_sx[j] = i32x4_splat(a_Q12[j] as i32);
        }
        for j in 0..shapingLPCOrder as usize {
            AR_shp_sx[j] = i32x4_splat(AR_shp_Q13[j] as i32);
        }
        let warp_hi = i32x4_splat(((warping_Q16 as u32) << 16) as i32);
        let warp_lo = i32x4_splat(warping_Q16 & 0xFFFF);
        let warp_sx = i32x4_splat(warping_Q16 as i16 as i32);
        let Tilt_sx = i32x4_splat(Tilt_Q14 as i16 as i32);
        let LF_shp_lo_sx = i32x4_splat(LF_shp_Q14 as i16 as i32);
        let LF_shp_hi_sx = i32x4_splat(silk_RSHIFT(LF_shp_Q14, 16));
        /* Lambda in the low half of each lane, zero in the high half, so that
           i32x4.dot_i16x8_s computes silk_SMULBB( x, Lambda_Q10 ) */
        let Lambda_lo = i32x4_splat(Lambda_Q10 & 0xFFFF);

        Seed = LD(&psDelDec.Seed);
        LF_AR_Q14 = LD(&psDelDec.LF_AR_Q14);
        Diff_Q14 = LD(&psDelDec.Diff_Q14);
        RD_Q10 = LD(&psDelDec.RD_Q10);

        for i in 0..length as usize {
            let LPC_pred_Q14: v128;
            let mut n_AR_Q14: v128;
            let mut n_LF_Q14: v128;
            let mut tmp1: v128;
            let mut tmp2: v128;
            let mut r_Q10: v128;
            let mut q1_Q0: v128;
            let mut q1_Q10: v128;
            let mut rd1_Q10: v128;
            let mut rd2_Q10: v128;
            let mut mask: v128;
            let mut exc_Q14: v128;

            /* Perform common calculations used in all states */

            /* Long-term prediction */
            if signalType == TYPE_VOICED {
                /* Unrolled loop */
                /* Avoids introducing a bias because silk_SMLAWB() always rounds to -inf */
                LTP_pred_Q14 = 2;
                LTP_pred_Q14 = silk_SMLAWB(LTP_pred_Q14, *pred_lag_ptr, b_Q14[0] as i32);
                LTP_pred_Q14 = silk_SMLAWB(LTP_pred_Q14, *pred_lag_ptr.sub(1), b_Q14[1] as i32);
                LTP_pred_Q14 = silk_SMLAWB(LTP_pred_Q14, *pred_lag_ptr.sub(2), b_Q14[2] as i32);
                LTP_pred_Q14 = silk_SMLAWB(LTP_pred_Q14, *pred_lag_ptr.sub(3), b_Q14[3] as i32);
                LTP_pred_Q14 = silk_SMLAWB(LTP_pred_Q14, *pred_lag_ptr.sub(4), b_Q14[4] as i32);
                LTP_pred_Q14 = silk_LSHIFT(LTP_pred_Q14, 1); /* Q13 -> Q14 */
                pred_lag_ptr = pred_lag_ptr.add(1);
            } else {
                LTP_pred_Q14 = 0;
            }

            /* Long-term shaping */
            if lag > 0 {
                /* Symmetric, packed FIR coefficients */
                n_LTP_Q14 = silk_SMULWB(silk_ADD_SAT32(*shp_lag_ptr, *shp_lag_ptr.sub(2)), HarmShapeFIRPacked_Q14);
                n_LTP_Q14 = silk_SMLAWT(n_LTP_Q14, *shp_lag_ptr.sub(1), HarmShapeFIRPacked_Q14);
                n_LTP_Q14 = silk_SUB_LSHIFT32(LTP_pred_Q14, n_LTP_Q14, 2); /* Q12 -> Q14 */
                shp_lag_ptr = shp_lag_ptr.add(1);
            } else {
                n_LTP_Q14 = 0;
            }

            /* Generate dither */
            Seed = i32x4_add(i32x4_mul(Seed, i32x4_splat(RAND_MULTIPLIER)), i32x4_splat(RAND_INCREMENT));

            /* Short-term prediction */
            let psLPC_Q14 = |j: usize| LD(&psDelDec.sLPC_Q14[NSQ_LPC_BUF_LENGTH_ - 1 + i - j]);
            /* Avoids introducing a bias because silk_SMLAWB() always rounds to -inf */
            tmp1 = i32x4_add(i32x4_splat(silk_RSHIFT(predictLPCOrder, 1)), silk_SMULWB_wasm(psLPC_Q14(0), a_Q12_sx[0]));
            tmp2 = silk_SMULWB_wasm(psLPC_Q14(1), a_Q12_sx[1]);
            tmp1 = i32x4_add(tmp1, silk_SMULWB_wasm(psLPC_Q14(2), a_Q12_sx[2]));
            tmp2 = i32x4_add(tmp2, silk_SMULWB_wasm(psLPC_Q14(3), a_Q12_sx[3]));
            tmp1 = i32x4_add(tmp1, silk_SMULWB_wasm(psLPC_Q14(4), a_Q12_sx[4]));
            tmp2 = i32x4_add(tmp2, silk_SMULWB_wasm(psLPC_Q14(5), a_Q12_sx[5]));
            tmp1 = i32x4_add(tmp1, silk_SMULWB_wasm(psLPC_Q14(6), a_Q12_sx[6]));
            tmp2 = i32x4_add(tmp2, silk_SMULWB_wasm(psLPC_Q14(7), a_Q12_sx[7]));
            tmp1 = i32x4_add(tmp1, silk_SMULWB_wasm(psLPC_Q14(8), a_Q12_sx[8]));
            tmp2 = i32x4_add(tmp2, silk_SMULWB_wasm(psLPC_Q14(9), a_Q12_sx[9]));
            if predictLPCOrder == 16 {
                tmp1 = i32x4_add(tmp1, silk_SMULWB_wasm(psLPC_Q14(10), a_Q12_sx[10]));
                tmp2 = i32x4_add(tmp2, silk_SMULWB_wasm(psLPC_Q14(11), a_Q12_sx[11]));
                tmp1 = i32x4_add(tmp1, silk_SMULWB_wasm(psLPC_Q14(12), a_Q12_sx[12]));
                tmp2 = i32x4_add(tmp2, silk_SMULWB_wasm(psLPC_Q14(13), a_Q12_sx[13]));
                tmp1 = i32x4_add(tmp1, silk_SMULWB_wasm(psLPC_Q14(14), a_Q12_sx[14]));
                tmp2 = i32x4_add(tmp2, silk_SMULWB_wasm(psLPC_Q14(15), a_Q12_sx[15]));
            }
            LPC_pred_Q14 = i32x4_shl(i32x4_add(tmp1, tmp2), 4); /* Q10 -> Q14 */

            /* Noise shape feedback */
            let sAR2 = &mut psDelDec.sAR2_Q14;
            if warped {
                let (mut s0, mut s1): (v128, v128);
                /* Output of lowpass section */
                s0 = LD(&sAR2[0]);
                tmp2 = i32x4_add(Diff_Q14, silk_SMULWB_dot_wasm(s0, warp_hi, warp_lo, warp_sx));
                /* Output of allpass section */
                s1 = LD(&sAR2[1]);
                tmp1 = i32x4_add(s0, silk_SMULWB_dot_wasm(i32x4_sub(s1, tmp2), warp_hi, warp_lo, warp_sx));
                ST(&mut sAR2[0], tmp2);
                n_AR_Q14 = i32x4_add(i32x4_splat(silk_RSHIFT(shapingLPCOrder, 1)), silk_SMULWB_wasm(tmp2, AR_shp_sx[0]));
                /* Loop over allpass sections */
                let mut j = 2usize;
                while j < shapingLPCOrder as usize {
                    /* Output of allpass section */
                    s0 = LD(&sAR2[j]);
                    tmp2 = i32x4_add(s1, silk_SMULWB_dot_wasm(i32x4_sub(s0, tmp1), warp_hi, warp_lo, warp_sx));
                    ST(&mut sAR2[j - 1], tmp1);
                    n_AR_Q14 = i32x4_add(n_AR_Q14, silk_SMULWB_wasm(tmp1, AR_shp_sx[j - 1]));
                    /* Output of allpass section */
                    s1 = LD(&sAR2[j + 1]);
                    tmp1 = i32x4_add(s0, silk_SMULWB_dot_wasm(i32x4_sub(s1, tmp2), warp_hi, warp_lo, warp_sx));
                    ST(&mut sAR2[j], tmp2);
                    n_AR_Q14 = i32x4_add(n_AR_Q14, silk_SMULWB_wasm(tmp2, AR_shp_sx[j]));
                    j += 2;
                }
                ST(&mut sAR2[shapingLPCOrder as usize - 1], tmp1);
                n_AR_Q14 = i32x4_add(n_AR_Q14, silk_SMULWB_wasm(tmp1, AR_shp_sx[shapingLPCOrder as usize - 1]));
            } else {
                /* Without warping the allpass sections reduce to a delay line */
                tmp1 = Diff_Q14;
                n_AR_Q14 = i32x4_splat(silk_RSHIFT(shapingLPCOrder, 1));
                tmp2 = i32x4_splat(0);
                let mut j = 0usize;
                while j < shapingLPCOrder as usize {
                    let s0 = LD(&sAR2[j]);
                    let s1 = LD(&sAR2[j + 1]);
                    ST(&mut sAR2[j], tmp1);
                    ST(&mut sAR2[j + 1], s0);
                    n_AR_Q14 = i32x4_add(n_AR_Q14, silk_SMULWB_wasm(tmp1, AR_shp_sx[j]));
                    tmp2 = i32x4_add(tmp2, silk_SMULWB_wasm(s0, AR_shp_sx[j + 1]));
                    tmp1 = s1;
                    j += 2;
                }
                n_AR_Q14 = i32x4_add(n_AR_Q14, tmp2);
            }

            n_AR_Q14 = i32x4_shl(n_AR_Q14, 1); /* Q11 -> Q12 */
            n_AR_Q14 = i32x4_add(n_AR_Q14, silk_SMULWB_wasm(LF_AR_Q14, Tilt_sx)); /* Q12 */
            n_AR_Q14 = i32x4_shl(n_AR_Q14, 2); /* Q12 -> Q14 */

            n_LF_Q14 = silk_SMULWB_wasm(LD(&psDelDec.Shape_Q14[*smpl_buf_idx as usize]), LF_shp_lo_sx); /* Q12 */
            n_LF_Q14 = i32x4_add(n_LF_Q14, silk_SMULWB_wasm(LF_AR_Q14, LF_shp_hi_sx)); /* Q12 */
            n_LF_Q14 = i32x4_shl(n_LF_Q14, 2); /* Q12 -> Q14 */

            /* Input minus prediction plus noise feedback                       */
            /* r = x[ i ] - LTP_pred - LPC_pred + n_AR + n_Tilt + n_LF + n_LTP  */
            tmp1 = silk_ADD_SAT32_wasm(n_AR_Q14, n_LF_Q14); /* Q14 */
            tmp2 = i32x4_add(i32x4_splat(n_LTP_Q14), LPC_pred_Q14); /* Q13 */
            tmp1 = silk_SUB_SAT32_wasm(tmp2, tmp1); /* Q13 */
            tmp1 = silk_RSHIFT_ROUND_wasm(tmp1, 4); /* Q10 */

            r_Q10 = i32x4_sub(i32x4_splat(x_Q10[i]), tmp1); /* residual error Q10 */

            /* Flip sign depending on dither */
            let sign = i32x4_shr(Seed, 31);
            r_Q10 = i32x4_sub(v128_xor(r_Q10, sign), sign);
            r_Q10 = i32x4_max(i32x4_min(r_Q10, i32x4_splat(30 << 10)), i32x4_splat(-(31 << 10)));

            /* Find two quantization level candidates and measure their rate-distortion */
            q1_Q10 = i32x4_sub(r_Q10, i32x4_splat(offset_Q10));
            q1_Q0 = i32x4_shr(q1_Q10, 10);
            if Lambda_Q10 > 2048 {
                /* For aggressive RDO, the bias becomes more than one pulse. */
                let rdo_offset = Lambda_Q10 / 2 - 512;
                let gt = i32x4_gt(q1_Q10, i32x4_splat(rdo_offset));
                let lt = i32x4_lt(q1_Q10, i32x4_splat(-rdo_offset));
                tmp1 = i32x4_shr(i32x4_sub(q1_Q10, i32x4_splat(rdo_offset)), 10);
                tmp2 = i32x4_shr(i32x4_add(q1_Q10, i32x4_splat(rdo_offset)), 10);
                q1_Q0 = silk_select_wasm(tmp1, silk_select_wasm(tmp2, i32x4_shr(q1_Q10, 31), lt), gt);
            }
            /* q1_Q10 = q1_Q0 * 1024 + offset_Q10, moved QUANT_LEVEL_ADJUST_Q10 towards zero */
            tmp1 = i32x4_max(i32x4_min(q1_Q0, i32x4_splat(1)), i32x4_splat(-1));
            tmp1 = i32x4_add(i32x4_shl(tmp1, 6), i32x4_shl(tmp1, 4)); /* * QUANT_LEVEL_ADJUST_Q10 */
            q1_Q10 = i32x4_add(i32x4_sub(i32x4_shl(q1_Q0, 10), tmp1), i32x4_splat(offset_Q10));
            /* q2_Q10 is one level above, closer when q1_Q0 is 0 or -1 */
            mask = i32x4_eq(i32x4_add(q1_Q0, u32x4_shr(q1_Q0, 31)), i32x4_splat(0));
            let q2_Q10 = i32x4_add(
                q1_Q10,
                i32x4_sub(i32x4_splat(1024), v128_and(mask, i32x4_splat(QUANT_LEVEL_ADJUST_Q10))),
            );
            /* The rate term is Lambda times the magnitude of the level: q1_Q10 and q2_Q10
               are negative exactly when the C code negates them, since 0 < offset_Q10 < 944. */
            rd1_Q10 = i32x4_dot_i16x8(i32x4_abs(q1_Q10), Lambda_lo);
            rd2_Q10 = i32x4_dot_i16x8(i32x4_abs(q2_Q10), Lambda_lo);
            tmp1 = i32x4_sub(r_Q10, q1_Q10);
            rd1_Q10 = i32x4_add(rd1_Q10, i32x4_dot_i16x8(tmp1, v128_and(tmp1, i32x4_splat(0xFFFF))));
            rd1_Q10 = i32x4_shr(rd1_Q10, 10);
            tmp2 = i32x4_sub(r_Q10, q2_Q10);
            rd2_Q10 = i32x4_add(rd2_Q10, i32x4_dot_i16x8(tmp2, v128_and(tmp2, i32x4_splat(0xFFFF))));
            rd2_Q10 = i32x4_shr(rd2_Q10, 10);

            mask = i32x4_lt(rd1_Q10, rd2_Q10);
            let mut SS0_RD_Q10 = i32x4_add(RD_Q10, i32x4_min(rd1_Q10, rd2_Q10));
            let mut SS1_RD_Q10 = i32x4_add(RD_Q10, i32x4_max(rd1_Q10, rd2_Q10));
            let mut SS0_Q_Q10 = silk_select_wasm(q1_Q10, q2_Q10, mask);
            let SS1_Q_Q10 = silk_select_wasm(q2_Q10, q1_Q10, mask);

            let x_Q14 = i32x4_splat(((x_Q10[i] as u32) << 4) as i32);

            /* Update states for best quantization */

            /* Quantized excitation */
            exc_Q14 = i32x4_shl(SS0_Q_Q10, 4);
            exc_Q14 = i32x4_sub(v128_xor(exc_Q14, sign), sign);

            /* Add predictions */
            let mut SS0_LPC_exc_Q14 = i32x4_add(exc_Q14, i32x4_splat(LTP_pred_Q14));
            let mut SS0_xq_Q14 = i32x4_add(SS0_LPC_exc_Q14, LPC_pred_Q14);

            /* Update states */
            let mut SS0_Diff_Q14 = i32x4_sub(SS0_xq_Q14, x_Q14);
            let mut SS0_LF_AR_Q14 = i32x4_sub(SS0_Diff_Q14, n_AR_Q14);
            let mut SS0_sLTP_shp_Q14 = silk_SUB_SAT32_wasm(SS0_LF_AR_Q14, n_LF_Q14);

            /* Update states for second best quantization */

            /* Quantized excitation */
            exc_Q14 = i32x4_shl(SS1_Q_Q10, 4);
            exc_Q14 = i32x4_sub(v128_xor(exc_Q14, sign), sign);

            /* Add predictions */
            let SS1_LPC_exc_Q14 = i32x4_add(exc_Q14, i32x4_splat(LTP_pred_Q14));
            let SS1_xq_Q14 = i32x4_add(SS1_LPC_exc_Q14, LPC_pred_Q14);

            /* Update states */
            let SS1_Diff_Q14 = i32x4_sub(SS1_xq_Q14, x_Q14);
            let SS1_LF_AR_Q14 = i32x4_sub(SS1_Diff_Q14, n_AR_Q14);
            let SS1_sLTP_shp_Q14 = silk_SUB_SAT32_wasm(SS1_LF_AR_Q14, n_LF_Q14);

            *smpl_buf_idx = if *smpl_buf_idx != 0 { *smpl_buf_idx - 1 } else { DECISION_DELAY - 1 };
            last_smple_idx = *smpl_buf_idx + decisionDelay;
            if last_smple_idx >= DECISION_DELAY {
                last_smple_idx -= DECISION_DELAY;
            }
            let last = last_smple_idx as usize;

            /* Find winner */
            Winner_ind = silk_first_min_wasm(SS0_RD_Q10, MaskDelDec, &mut RDmin_Q10);

            /* Increase RD values of expired states */
            tmp1 = LD(&psDelDec.RandState[last]);
            mask = i32x4_eq(tmp1, i32x4_splat(psDelDec.RandState[last][Winner_ind]));
            tmp1 = v128_andnot(i32x4_splat(silk_int32_MAX >> 4), mask);
            SS0_RD_Q10 = i32x4_add(SS0_RD_Q10, tmp1);
            SS1_RD_Q10 = i32x4_add(SS1_RD_Q10, tmp1);

            /* Find worst in first set and best in second set */
            RDmax_ind = silk_first_max_wasm(SS0_RD_Q10, MaskDelDec, &mut RDmax_Q10);
            RDmin_ind = silk_first_min_wasm(SS1_RD_Q10, MaskDelDec, &mut RDmin_Q10);

            /* Replace a state if best from second set outperforms worst in first set */
            if RDmin_Q10 < RDmax_Q10 {
                /* Byte selector moving lane RDmin_ind into lane RDmax_ind */
                let lane_mask = i32x4_eq(lane_idx, i32x4_splat(RDmax_ind as i32));
                let bcast = i32x4_splat(RDmin_ind as i32 * 0x04040404 + 0x03020100);
                let sel = silk_select_wasm(bcast, byte_idx, lane_mask);

                /* Only the last ( NSQ_LPC_BUF_LENGTH - 1 ) values of sLPC_Q14 are used again */
                for j in i + 1..i + NSQ_LPC_BUF_LENGTH_ {
                    let v = silk_swizzle_wasm(LD(&psDelDec.sLPC_Q14[j]), sel);
                    ST(&mut psDelDec.sLPC_Q14[j], v);
                }
                for j in 0..shapingLPCOrder as usize {
                    let v = silk_swizzle_wasm(LD(&psDelDec.sAR2_Q14[j]), sel);
                    ST(&mut psDelDec.sAR2_Q14[j], v);
                }
                /* RandState, Q_Q10, Xq_Q14, Pred_Q15 and Shape_Q14 are mostly read
                   as scalars later, so copy them lane by lane */
                let ptr = psDelDec.RandState.as_mut_ptr();
                for j in 0..5 * DECISION_DELAY_ {
                    (*ptr.add(j))[RDmax_ind] = (*ptr.add(j))[RDmin_ind];
                }
                Seed = silk_swizzle_wasm(Seed, sel);
                let v = silk_swizzle_wasm(LD(&psDelDec.SeedInit), sel);
                ST(&mut psDelDec.SeedInit, v);
                /* LF_AR_Q14, Diff_Q14 and RD_Q10 are overwritten below */

                SS0_Q_Q10 = silk_select_wasm(silk_swizzle_wasm(SS1_Q_Q10, bcast), SS0_Q_Q10, lane_mask);
                SS0_RD_Q10 = silk_select_wasm(silk_swizzle_wasm(SS1_RD_Q10, bcast), SS0_RD_Q10, lane_mask);
                SS0_xq_Q14 = silk_select_wasm(silk_swizzle_wasm(SS1_xq_Q14, bcast), SS0_xq_Q14, lane_mask);
                SS0_LF_AR_Q14 = silk_select_wasm(silk_swizzle_wasm(SS1_LF_AR_Q14, bcast), SS0_LF_AR_Q14, lane_mask);
                SS0_Diff_Q14 = silk_select_wasm(silk_swizzle_wasm(SS1_Diff_Q14, bcast), SS0_Diff_Q14, lane_mask);
                SS0_sLTP_shp_Q14 =
                    silk_select_wasm(silk_swizzle_wasm(SS1_sLTP_shp_Q14, bcast), SS0_sLTP_shp_Q14, lane_mask);
                SS0_LPC_exc_Q14 =
                    silk_select_wasm(silk_swizzle_wasm(SS1_LPC_exc_Q14, bcast), SS0_LPC_exc_Q14, lane_mask);
            }

            /* Write samples from winner to output and long-term filter states */
            if subfr > 0 || i as i32 >= decisionDelay {
                let o = i as isize - decisionDelay as isize;
                *pulses.offset(o) = silk_RSHIFT_ROUND(psDelDec.Q_Q10[last][Winner_ind], 10) as i8;
                *xq.offset(o) = silk_SAT16(silk_RSHIFT_ROUND(
                    silk_SMULWW(psDelDec.Xq_Q14[last][Winner_ind], delayedGain_Q10[last]),
                    8,
                )) as i16;
                *sLTP_shp.add((NSQ.sLTP_shp_buf_idx - decisionDelay) as usize) = psDelDec.Shape_Q14[last][Winner_ind];
                *sLTP_Q15p.add((NSQ.sLTP_buf_idx - decisionDelay) as usize) = psDelDec.Pred_Q15[last][Winner_ind];
            }
            NSQ.sLTP_shp_buf_idx += 1;
            NSQ.sLTP_buf_idx += 1;

            /* Update states */
            LF_AR_Q14 = SS0_LF_AR_Q14;
            Diff_Q14 = SS0_Diff_Q14;
            RD_Q10 = SS0_RD_Q10;
            let idx = *smpl_buf_idx as usize;
            ST(&mut psDelDec.sLPC_Q14[NSQ_LPC_BUF_LENGTH_ + i], SS0_xq_Q14);
            ST(&mut psDelDec.Xq_Q14[idx], SS0_xq_Q14);
            ST(&mut psDelDec.Q_Q10[idx], SS0_Q_Q10);
            ST(&mut psDelDec.Pred_Q15[idx], i32x4_shl(SS0_LPC_exc_Q14, 1));
            ST(&mut psDelDec.Shape_Q14[idx], SS0_sLTP_shp_Q14);
            Seed = i32x4_add(Seed, silk_RSHIFT_ROUND_wasm(SS0_Q_Q10, 10));
            ST(&mut psDelDec.RandState[idx], Seed);
            delayedGain_Q10[idx] = Gain_Q10;
        }
        ST(&mut psDelDec.Seed, Seed);
        ST(&mut psDelDec.LF_AR_Q14, LF_AR_Q14);
        ST(&mut psDelDec.Diff_Q14, Diff_Q14);
        ST(&mut psDelDec.RD_Q10, RD_Q10);

        /* Update LPC states */
        psDelDec.sLPC_Q14.copy_within(length as usize..length as usize + NSQ_LPC_BUF_LENGTH_, 0);
    }
}

/// Copy the final part of the signals from the winner state to the output and
/// long-term filter states.
#[inline(always)]
unsafe fn silk_copy_winner_state_wasm(
    psDelDec: &NSQ_del_decs_wasm_struct,
    NSQ: &mut silk_nsq_state,
    pulses: *mut i8,
    pxq_off: usize, /* pxq = &NSQ->xq[ pxq_off ] */
    smpl_buf_idx: i32,
    decisionDelay: i32,
    Winner_ind: usize,
    gain: i32,
    shift: i32,
) {
    // SAFETY: the only calls are at the start of subframe 2 and after the last subframe, so
    // `pulses` is at least 2 * MAX_SUB_FRAME_LENGTH entries into the frame's pulse buffer and
    // the offsets [-decisionDelay, 0) (decisionDelay <= DECISION_DELAY = 40 <= 160) are in
    // range; the xq / sLTP_shp writes are bounds-checked slice indexing.
    unsafe {
        debug_assert!(pxq_off >= decisionDelay as usize);
        let mut last_smple_idx = smpl_buf_idx + decisionDelay;
        for i in 0..decisionDelay {
            last_smple_idx = (last_smple_idx - 1) % DECISION_DELAY;
            if last_smple_idx < 0 {
                last_smple_idx += DECISION_DELAY;
            }
            let last = last_smple_idx as usize;
            let o = (i - decisionDelay) as isize;
            *pulses.offset(o) = silk_RSHIFT_ROUND(psDelDec.Q_Q10[last][Winner_ind], 10) as i8;
            NSQ.xq[(pxq_off as isize + o) as usize] =
                silk_SAT16(silk_RSHIFT_ROUND(silk_SMULWW(psDelDec.Xq_Q14[last][Winner_ind], gain), shift)) as i16;
            NSQ.sLTP_shp_Q14[(NSQ.sLTP_shp_buf_idx - decisionDelay + i) as usize] = psDelDec.Shape_Q14[last][Winner_ind];
        }
    }
}

/// Delayed-decision noise shaping quantization (silk_NSQ_del_dec_wasm).
///
/// Raw pointers as in C: the encoder passes `&psEnc->sCmn.sNSQ`, `&psEnc->sCmn.indices` and
/// `psEnc->sCmn.pulses`, which all lie inside `*psEncC`. Only configuration fields are read
/// through `psEncC`. `pulses` receives `frame_length` values.
pub unsafe fn silk_NSQ_del_dec(
    psEncC: *const silk_encoder_state, /* I    Encoder State                   */
    NSQ: *mut silk_nsq_state,          /* I/O  NSQ state                       */
    psIndices: *mut SideInfoIndices,   /* I/O  Quantization Indices            */
    x16: &[i16],                       /* I    Input                           */
    pulses: *mut i8,                   /* O    Quantized pulse signal          */
    PredCoef_Q12: &[i16],              /* I    Short term prediction coefs     [ 2 * MAX_LPC_ORDER ] */
    LTPCoef_Q14: &[i16],               /* I    Long term prediction coefs      [ LTP_ORDER * MAX_NB_SUBFR ] */
    AR_Q13: &[i16],                    /* I    Noise shaping coefs             [ MAX_NB_SUBFR * MAX_SHAPE_LPC_ORDER ] */
    HarmShapeGain_Q14: &[i32],         /* I    Long term shaping coefs         [ MAX_NB_SUBFR ] */
    Tilt_Q14: &[i32],                  /* I    Spectral tilt                   [ MAX_NB_SUBFR ] */
    LF_shp_Q14: &[i32],                /* I    Low frequency shaping coefs     [ MAX_NB_SUBFR ] */
    Gains_Q16: &[i32],                 /* I    Quantization step sizes         [ MAX_NB_SUBFR ] */
    pitchL: &[i32],                    /* I    Pitch lags                      [ MAX_NB_SUBFR ] */
    Lambda_Q10: i32,                   /* I    Rate/distortion tradeoff        */
    LTP_scale_Q14: i32,                /* I    LTP state scaling               */
) {
    // SAFETY: `psEncC`, `NSQ` and `psIndices` are valid (see above) and `pulses` has frame_length
    // entries: it is advanced by subfr_length once per subframe, nb_subfr times. The scratch
    // buffers hold (ltp_mem_length + frame_length) entries, which bounds the sLTP indices used by
    // the kernels called below; the `LD`/`ST` calls here are on `[i32; 4]` rows.
    unsafe {
        let subfr_length = MAX_SUB_FRAME_LENGTH;
        let ltp_mem_length = LTP_MEM_LENGTH;
        let frame_length = MAX_FRAME_LENGTH;
        let nb_subfr = MAX_NB_SUBFR;
        let predictLPCOrder = MAX_LPC_ORDER;
        let shapingLPCOrder = (*psEncC).shapingLPCOrder;
        let warping_Q16 = (*psEncC).warping_Q16;
        let nStatesDelayedDecision = (*psEncC).nStatesDelayedDecision;
        let NSQ = &mut *NSQ;
        let signalType = (*psIndices).signalType as i32;
        let quantOffsetType = (*psIndices).quantOffsetType as i32;
        let NLSFInterpCoef_Q2 = (*psIndices).NLSFInterpCoef_Q2 as i32;

        let mut Winner_ind: usize;
        let mut subfr: i32;
        let mut smpl_buf_idx: i32;
        let mut decisionDelay: i32;
        let mut RDmin_Q10: i32 = 0;
        let mut x_sc_Q10 = [0i32; MAX_SUB_FRAME_LENGTH_];
        let mut delayedGain_Q10 = [0i32; DECISION_DELAY_];
        let mut t: v128;

        /* The scalar silk_NSQ_del_dec_c() fallback for more states is not ported: the encoder
           never configures more than 4 states */
        debug_assert!(nStatesDelayedDecision >= 1 && nStatesDelayedDecision as usize <= WASM_DEL_DEC_STATES);
        debug_assert!(nb_subfr == MAX_NB_SUBFR && subfr_length == MAX_SUB_FRAME_LENGTH);
        debug_assert!(frame_length == nb_subfr * subfr_length && x16.len() >= frame_length as usize);

        /* Lanes of states beyond nStatesDelayedDecision never take part in decisions */
        let MaskDelDec = i32x4_gt(i32x4(0, 1, 2, 3), i32x4_splat(nStatesDelayedDecision - 1));

        /* Set unvoiced lag to the previous one, overwrite later for voiced */
        let mut lag = NSQ.lagPrev;

        debug_assert!(NSQ.prev_gain_Q16 != 0);

        /* Initialize delayed decision states */
        let mut psDelDec: NSQ_del_decs_wasm_struct = core::mem::zeroed();
        t = v128_and(i32x4_add(i32x4(0, 1, 2, 3), i32x4_splat((*psIndices).Seed as i32)), i32x4_splat(3));
        ST(&mut psDelDec.Seed, t);
        ST(&mut psDelDec.SeedInit, t);
        ST(&mut psDelDec.LF_AR_Q14, i32x4_splat(NSQ.sLF_AR_shp_Q14));
        ST(&mut psDelDec.Diff_Q14, i32x4_splat(NSQ.sDiff_shp_Q14));
        ST(&mut psDelDec.Shape_Q14[0], i32x4_splat(NSQ.sLTP_shp_Q14[ltp_mem_length as usize - 1]));
        for i in 0..NSQ_LPC_BUF_LENGTH_ {
            ST(&mut psDelDec.sLPC_Q14[i], i32x4_splat(NSQ.sLPC_Q14[i]));
        }
        for i in 0..MAX_SHAPE_LPC_ORDER_ {
            ST(&mut psDelDec.sAR2_Q14[i], i32x4_splat(NSQ.sAR2_Q14[i]));
        }

        let offset_Q10 = silk_Quantization_Offsets_Q10[(signalType >> 1) as usize][quantOffsetType as usize] as i32;
        smpl_buf_idx = 0; /* index of oldest samples */

        decisionDelay = silk_min_int(DECISION_DELAY, subfr_length);

        /* For voiced frames limit the decision delay to lower than the pitch lag */
        if signalType == TYPE_VOICED {
            for k in 0..nb_subfr as usize {
                decisionDelay = silk_min_int(decisionDelay, pitchL[k] - LTP_ORDER / 2 - 1);
            }
        } else if lag > 0 {
            decisionDelay = silk_min_int(decisionDelay, lag - LTP_ORDER / 2 - 1);
        }

        let LSF_interpolation_flag: usize = if NLSFInterpCoef_Q2 == 4 { 0 } else { 1 };

        let stk = stack::save();
        let sLTP_Q15 = stk.alloc::<i32>((ltp_mem_length + frame_length) as usize);
        let sLTP = stk.alloc::<i16>((ltp_mem_length + frame_length) as usize);
        /* Set up pointers to start of sub frame */
        let mut pxq = ltp_mem_length as usize; /* pxq = &NSQ->xq[ pxq ] */
        NSQ.sLTP_shp_buf_idx = ltp_mem_length;
        NSQ.sLTP_buf_idx = ltp_mem_length;
        let mut x16_off = 0usize;
        let mut pulses = pulses;
        subfr = 0;
        for k in 0..nb_subfr as usize {
            let A_Q12 = &PredCoef_Q12[(((k >> 1) | (1 - LSF_interpolation_flag)) * MAX_LPC_ORDER_)..][..MAX_LPC_ORDER_];
            let B_Q14 = &LTPCoef_Q14[k * LTP_ORDER_..][..LTP_ORDER_];
            let AR_shp_Q13 = &AR_Q13[k * MAX_SHAPE_LPC_ORDER_..][..MAX_SHAPE_LPC_ORDER_];

            /* Noise shape parameters */
            debug_assert!(HarmShapeGain_Q14[k] >= 0);
            let mut HarmShapeFIRPacked_Q14 = silk_RSHIFT(HarmShapeGain_Q14[k], 2);
            HarmShapeFIRPacked_Q14 |= silk_LSHIFT(silk_RSHIFT(HarmShapeGain_Q14[k], 1), 16);

            NSQ.rewhite_flag = 0;
            if signalType == TYPE_VOICED {
                /* Voiced */
                lag = pitchL[k];

                /* Re-whitening */
                if (k as i32 & (3 - silk_LSHIFT(LSF_interpolation_flag as i32, 1))) == 0 {
                    if k == 2 {
                        /* RESET DELAYED DECISIONS */
                        /* Find winner */
                        t = LD(&psDelDec.RD_Q10);
                        Winner_ind = silk_first_min_wasm(t, MaskDelDec, &mut RDmin_Q10);
                        t = i32x4_add(
                            t,
                            v128_andnot(
                                i32x4_splat(silk_int32_MAX >> 4),
                                i32x4_eq(i32x4(0, 1, 2, 3), i32x4_splat(Winner_ind as i32)),
                            ),
                        );
                        ST(&mut psDelDec.RD_Q10, t);

                        /* Copy final part of signals from winner state to output and long-term filter states */
                        silk_copy_winner_state_wasm(
                            &psDelDec, NSQ, pulses, pxq, smpl_buf_idx, decisionDelay, Winner_ind, Gains_Q16[1], 14,
                        );

                        subfr = 0;
                    }

                    /* Rewhiten with new A coefs */
                    let start_idx = ltp_mem_length - lag - predictLPCOrder - LTP_ORDER / 2;
                    debug_assert!(start_idx > 0);

                    silk_LPC_analysis_filter(
                        &mut sLTP[start_idx as usize..],
                        &NSQ.xq[(start_idx + k as i32 * subfr_length) as usize..],
                        A_Q12,
                        ltp_mem_length - start_idx,
                        predictLPCOrder,
                    );

                    NSQ.sLTP_buf_idx = ltp_mem_length;
                    NSQ.rewhite_flag = 1;
                }
            }

            silk_nsq_del_dec_scale_states_wasm(
                subfr_length,
                ltp_mem_length,
                NSQ,
                &mut psDelDec,
                &x16[x16_off..],
                &mut x_sc_Q10,
                sLTP,
                sLTP_Q15,
                k,
                LTP_scale_Q14,
                Gains_Q16,
                pitchL,
                signalType,
                decisionDelay,
            );

            silk_noise_shape_quantizer_del_dec_wasm(
                NSQ,
                &mut psDelDec,
                signalType,
                &x_sc_Q10,
                pulses,
                pxq,
                sLTP_Q15,
                &mut delayedGain_Q10,
                A_Q12,
                B_Q14,
                AR_shp_Q13,
                lag,
                HarmShapeFIRPacked_Q14,
                Tilt_Q14[k],
                LF_shp_Q14[k],
                Gains_Q16[k],
                Lambda_Q10,
                offset_Q10,
                subfr_length,
                subfr,
                shapingLPCOrder,
                predictLPCOrder,
                warping_Q16,
                MaskDelDec,
                &mut smpl_buf_idx,
                decisionDelay,
            );
            subfr += 1;

            x16_off += subfr_length as usize;
            pulses = pulses.add(subfr_length as usize);
            pxq += subfr_length as usize;
        }

        /* Find winner */
        Winner_ind = silk_first_min_wasm(LD(&psDelDec.RD_Q10), MaskDelDec, &mut RDmin_Q10);

        /* Copy final part of signals from winner state to output and long-term filter states */
        (*psIndices).Seed = psDelDec.SeedInit[Winner_ind] as i8;
        let Gain_Q10 = silk_RSHIFT32(Gains_Q16[nb_subfr as usize - 1], 6);
        silk_copy_winner_state_wasm(&psDelDec, NSQ, pulses, pxq, smpl_buf_idx, decisionDelay, Winner_ind, Gain_Q10, 8);
        for i in 0..NSQ_LPC_BUF_LENGTH_ {
            NSQ.sLPC_Q14[i] = psDelDec.sLPC_Q14[subfr_length as usize + i][Winner_ind];
        }
        for i in 0..MAX_SHAPE_LPC_ORDER_ {
            NSQ.sAR2_Q14[i] = psDelDec.sAR2_Q14[i][Winner_ind];
        }

        /* Update states */
        NSQ.sLF_AR_shp_Q14 = psDelDec.LF_AR_Q14[Winner_ind];
        NSQ.sDiff_shp_Q14 = psDelDec.Diff_Q14[Winner_ind];
        NSQ.lagPrev = pitchL[nb_subfr as usize - 1];

        /* Save quantized speech signal */
        NSQ.xq.copy_within(frame_length as usize..(frame_length + ltp_mem_length) as usize, 0);
        NSQ.sLTP_shp_Q14.copy_within(frame_length as usize..(frame_length + ltp_mem_length) as usize, 0);
    }
}

#[inline(always)]
unsafe fn silk_nsq_del_dec_scale_states_wasm(
    subfr_length: i32,                       /* I    psEncC->subfr_length                */
    ltp_mem_length: i32,                     /* I    psEncC->ltp_mem_length              */
    NSQ: &mut silk_nsq_state,                /* I/O  NSQ state                           */
    psDelDec: &mut NSQ_del_decs_wasm_struct, /* I/O  Delayed decision states             */
    x16: &[i16],                             /* I    Input                               */
    x_sc_Q10: &mut [i32],                    /* O    Input scaled with 1/Gain in Q10     */
    sLTP: &[i16],                            /* I    Re-whitened LTP state in Q0         */
    sLTP_Q15: &mut [i32],                    /* O    LTP state matching scaled input     */
    subfr: usize,                            /* I    Subframe number                     */
    LTP_scale_Q14: i32,                      /* I    LTP state scaling                   */
    Gains_Q16: &[i32],                       /* I                                        */
    pitchL: &[i32],                          /* I    Pitch lag                           */
    signal_type: i32,                        /* I    Signal type                         */
    decisionDelay: i32,                      /* I    Decision delay                      */
) {
    // SAFETY: as in `silk_nsq_scale_states_wasm` (NSQ.rs): the x16 / x_sc_Q10 pointers come from
    // slices cut to `subfr_length` and the 4-lane loops run only while i < limit - 3; the sLTP,
    // sLTP_Q15 and sLTP_shp_Q14 ranges start at i >= 0 and end at or below the buffer lengths
    // (the debug_asserts below).
    unsafe {
        let mut i: i32;
        let mut g: v128;

        let lag = pitchL[subfr];
        let mut inv_gain_Q31 = silk_INVERSE32_varQ(silk_max!(Gains_Q16[subfr], 1), 47);
        debug_assert!(inv_gain_Q31 != 0);

        /* Scale input */
        let inv_gain_Q26 = silk_RSHIFT_ROUND(inv_gain_Q31, 5);
        g = i32x4_splat(inv_gain_Q26);
        let x16p = x16[..subfr_length as usize].as_ptr();
        let x_sc = x_sc_Q10[..subfr_length as usize].as_mut_ptr();
        i = 0;
        while i < subfr_length - 3 {
            v128_store(x_sc.add(i as usize) as *mut v128, silk_SMULWW_wasm(i32x4_load_extend_i16x4(x16p.add(i as usize)), g));
            i += 4;
        }
        while i < subfr_length {
            *x_sc.add(i as usize) = silk_SMULWW(*x16p.add(i as usize) as i32, inv_gain_Q26);
            i += 1;
        }

        let sLTP_len = sLTP.len().min(sLTP_Q15.len());
        let sLTPp = sLTP.as_ptr();
        let sLTP_Q15p = sLTP_Q15.as_mut_ptr();

        /* After rewhitening the LTP state is un-scaled, so scale with inv_gain_Q16 */
        if NSQ.rewhite_flag != 0 {
            if subfr == 0 {
                /* Do LTP downscaling */
                inv_gain_Q31 = silk_LSHIFT(silk_SMULWB(inv_gain_Q31, LTP_scale_Q14), 2);
            }
            g = i32x4_splat(inv_gain_Q31);
            i = NSQ.sLTP_buf_idx - lag - LTP_ORDER / 2;
            debug_assert!(i >= 0 && NSQ.sLTP_buf_idx as usize <= sLTP_len);
            while i < NSQ.sLTP_buf_idx - 3 {
                debug_assert!(i < MAX_FRAME_LENGTH);
                v128_store(
                    sLTP_Q15p.add(i as usize) as *mut v128,
                    silk_SMULWB_wasm(g, i32x4_load_extend_i16x4(sLTPp.add(i as usize))),
                );
                i += 4;
            }
            while i < NSQ.sLTP_buf_idx {
                *sLTP_Q15p.add(i as usize) = silk_SMULWB(inv_gain_Q31, *sLTPp.add(i as usize) as i32);
                i += 1;
            }
        }

        /* Adjust for changing gain */
        if Gains_Q16[subfr] != NSQ.prev_gain_Q16 {
            let gain_adj_Q16 = silk_DIV32_varQ(NSQ.prev_gain_Q16, Gains_Q16[subfr], 16);
            g = i32x4_splat(gain_adj_Q16);

            /* Scale long-term shaping state */
            let shp = NSQ.sLTP_shp_Q14.as_mut_ptr();
            i = NSQ.sLTP_shp_buf_idx - ltp_mem_length;
            debug_assert!(i >= 0 && NSQ.sLTP_shp_buf_idx as usize <= NSQ.sLTP_shp_Q14.len());
            while i < NSQ.sLTP_shp_buf_idx - 3 {
                let p = shp.add(i as usize) as *mut v128;
                v128_store(p, silk_SMULWW_wasm(v128_load(p), g));
                i += 4;
            }
            while i < NSQ.sLTP_shp_buf_idx {
                *shp.add(i as usize) = silk_SMULWW(gain_adj_Q16, *shp.add(i as usize));
                i += 1;
            }

            /* Scale long-term prediction state */
            if signal_type == TYPE_VOICED && NSQ.rewhite_flag == 0 {
                i = NSQ.sLTP_buf_idx - lag - LTP_ORDER / 2;
                debug_assert!(i >= 0 && NSQ.sLTP_buf_idx as usize <= sLTP_len);
                while i < NSQ.sLTP_buf_idx - decisionDelay - 3 {
                    let p = sLTP_Q15p.add(i as usize) as *mut v128;
                    v128_store(p, silk_SMULWW_wasm(v128_load(p), g));
                    i += 4;
                }
                while i < NSQ.sLTP_buf_idx - decisionDelay {
                    *sLTP_Q15p.add(i as usize) = silk_SMULWW(gain_adj_Q16, *sLTP_Q15p.add(i as usize));
                    i += 1;
                }
            }

            /* Scale scalar states */
            scale_row(&mut psDelDec.LF_AR_Q14, g);
            scale_row(&mut psDelDec.Diff_Q14, g);

            /* Scale short-term prediction and shaping states */
            for i in 0..NSQ_LPC_BUF_LENGTH_ {
                scale_row(&mut psDelDec.sLPC_Q14[i], g);
            }
            for i in 0..MAX_SHAPE_LPC_ORDER_ {
                scale_row(&mut psDelDec.sAR2_Q14[i], g);
            }
            for i in 0..DECISION_DELAY_ {
                scale_row(&mut psDelDec.Pred_Q15[i], g);
                scale_row(&mut psDelDec.Shape_Q14[i], g);
            }

            /* Save inverse gain */
            NSQ.prev_gain_Q16 = Gains_Q16[subfr];
        }
    }
}
