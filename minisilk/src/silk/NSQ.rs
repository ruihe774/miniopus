//! Port of silk/NSQ.h and silk/wasm/NSQ_wasm.c.
//!
//! `silk_NSQ` is the WASM kernel `silk_NSQ_wasm`. The kernel does not use any of the shared
//! helpers in silk/NSQ.h (it has its own vectorized short-term prediction and noise shape
//! feedback), and the scalar `silk_NSQ_c` is not ported.
//!
//! The quantizer runs one sample at a time, so the filters are vectorized
//! along their taps. The short-term prediction and noise shaping filter
//! states are kept in registers and shifted by one lane per sample, which
//! avoids reloading values that were just stored. The newest tap of both
//! filters, which depends on the previous sample, is computed separately in
//! scalar code, so that the vector products of the older taps stay off the
//! critical path from one sample to the next. Bit-exact with silk_NSQ_c().

use core::arch::wasm32::*;

use crate::silk::LPC_analysis_filter::silk_LPC_analysis_filter;
use crate::silk::define::*;
use crate::silk::macros::*;
use crate::silk::structs::{SideInfoIndices, silk_encoder_state, silk_nsq_state};
use crate::silk::tables::silk_Quantization_Offsets_Q10;
use crate::stack;

pub(crate) const MAX_LPC_ORDER_: usize = MAX_LPC_ORDER as usize;
pub(crate) const MAX_SHAPE_LPC_ORDER_: usize = MAX_SHAPE_LPC_ORDER as usize;
pub(crate) const NSQ_LPC_BUF_LENGTH_: usize = NSQ_LPC_BUF_LENGTH as usize;
pub(crate) const LTP_ORDER_: usize = LTP_ORDER as usize;

/// silk_SMULWB() on each lane: ( a * (opus_int16)b ) >> 16, with b_sx holding
/// the sign-extended 16-bit coefficient. Split a into its signed high and
/// unsigned low halves; both partial products fit in 32 bits.
#[inline(always)]
fn silk_SMULWB_wasm(a: v128, b_sx: v128) -> v128 {
    let hi = i32x4_mul(i32x4_shr(a, 16), b_sx);
    let lo = i32x4_mul(v128_and(a, i32x4_splat(0xFFFF)), b_sx);
    i32x4_add(hi, i32x4_shr(lo, 16))
}

/// silk_SMULWW() on each lane: ( a * b ) >> 16, truncated to 32 bits.
#[inline(always)]
fn silk_SMULWW_wasm(a: v128, b: v128) -> v128 {
    let lo = i64x2_shr(i64x2_extmul_low_i32x4(a, b), 16);
    let hi = i64x2_shr(i64x2_extmul_high_i32x4(a, b), 16);
    i32x4_shuffle::<0, 2, 4, 6>(lo, hi)
}

#[inline(always)]
unsafe fn ld(p: *const i32) -> v128 {
    unsafe { v128_load(p as *const v128) }
}

#[inline(always)]
unsafe fn st(p: *mut i32, v: v128) {
    unsafe { v128_store(p as *mut v128, v) }
}

#[inline(always)]
unsafe fn silk_nsq_scale_states_wasm(
    subfr_length: i32,        /* I    psEncC->subfr_length            */
    ltp_mem_length: i32,      /* I    psEncC->ltp_mem_length          */
    NSQ: &mut silk_nsq_state, /* I/O  NSQ state                       */
    x16: &[i16],              /* I    input                           */
    x_sc_Q10: &mut [i32],     /* O    input scaled with 1/Gain        */
    sLTP: &[i16],             /* I    re-whitened LTP state in Q0     */
    sLTP_Q15: &mut [i32],     /* O    LTP state matching scaled input */
    subfr: usize,             /* I    subframe number                 */
    LTP_scale_Q14: i32,       /* I                                    */
    Gains_Q16: &[i32],        /* I                                    */
    pitchL: &[i32],           /* I    Pitch lag                       */
    signal_type: i32,         /* I    Signal type                     */
) {
    // SAFETY: the x16 / x_sc_Q10 pointers come from slices cut to `subfr_length`, and the vector
    // loops run while i < subfr_length - 3 (4 lanes in range). The sLTP / sLTP_Q15 / sLTP_shp_Q14
    // loops cover [start, buf_idx) with start >= 0 and buf_idx within the buffer length (the
    // debug_asserts below), again in steps of 4 only while i < buf_idx - 3. The sLPC_Q14 (16
    // entries scaled) and sAR2_Q14 (24) loops have bounds that are multiples of 4 and lie within
    // the arrays (sLPC_Q14 holds MAX_SUB_FRAME_LENGTH + NSQ_LPC_BUF_LENGTH entries).
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
            st(x_sc.add(i as usize), silk_SMULWW_wasm(i32x4_load_extend_i16x4(x16p.add(i as usize)), g));
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
                st(sLTP_Q15p.add(i as usize), silk_SMULWB_wasm(g, i32x4_load_extend_i16x4(sLTPp.add(i as usize))));
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
                st(shp.add(i as usize), silk_SMULWW_wasm(ld(shp.add(i as usize)), g));
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
                while i < NSQ.sLTP_buf_idx - 3 {
                    st(sLTP_Q15p.add(i as usize), silk_SMULWW_wasm(ld(sLTP_Q15p.add(i as usize)), g));
                    i += 4;
                }
                while i < NSQ.sLTP_buf_idx {
                    *sLTP_Q15p.add(i as usize) = silk_SMULWW(gain_adj_Q16, *sLTP_Q15p.add(i as usize));
                    i += 1;
                }
            }

            NSQ.sLF_AR_shp_Q14 = silk_SMULWW(gain_adj_Q16, NSQ.sLF_AR_shp_Q14);
            NSQ.sDiff_shp_Q14 = silk_SMULWW(gain_adj_Q16, NSQ.sDiff_shp_Q14);

            /* Scale short-term prediction and shaping states */
            let lpc = NSQ.sLPC_Q14.as_mut_ptr();
            for i in (0..NSQ_LPC_BUF_LENGTH_).step_by(4) {
                st(lpc.add(i), silk_SMULWW_wasm(ld(lpc.add(i)), g));
            }
            let ar2 = NSQ.sAR2_Q14.as_mut_ptr();
            for i in (0..MAX_SHAPE_LPC_ORDER_).step_by(4) {
                st(ar2.add(i), silk_SMULWW_wasm(ld(ar2.add(i)), g));
            }

            /* Save inverse gain */
            NSQ.prev_gain_Q16 = Gains_Q16[subfr];
        }
    }
}

/// Noise shape quantizer for one subframe. AR_VECS is a compile-time constant
/// (4 or 6): the number of vectors that hold the noise shaping filter.
#[inline(always)]
unsafe fn silk_noise_shape_quantizer_wasm<const AR_VECS: usize>(
    NSQ: &mut silk_nsq_state,    /* I/O  NSQ state                       */
    signalType: i32,             /* I    Signal type                     */
    x_sc_Q10: &[i32],            /* I                                    */
    pulses: *mut i8,             /* O                                    */
    xq_off: usize,               /* O    xq = &NSQ->xq[ xq_off ]         */
    sLTP_Q15: &mut [i32],        /* I/O  LTP state                       */
    a_Q12: &[i16],               /* I    Short term prediction coefs     */
    b_Q14: &[i16],               /* I    Long term prediction coefs      */
    AR_shp_Q13: &[i16],          /* I    Noise shaping AR coefs          */
    lag: i32,                    /* I    Pitch lag                       */
    HarmShapeFIRPacked_Q14: i32, /* I                                    */
    Tilt_Q14: i32,               /* I    Spectral tilt                   */
    LF_shp_Q14: i32,             /* I                                    */
    Gain_Q16: i32,               /* I                                    */
    Lambda_Q10: i32,             /* I                                    */
    offset_Q10: i32,             /* I                                    */
    length: i32,                 /* I    Input length                    */
    shapingLPCOrder: i32,        /* I    Noise shaping AR filter order   */
    predictLPCOrder: i32,        /* I    Prediction filter order         */
) {
    // SAFETY: the caller passes `pulses` with `length` writable entries, and the debug_asserts
    // below state the buffer bounds: xq_off + length <= xq.len(); sLTP_buf_idx + length <=
    // sLTP_Q15.len(); sLTP_shp_buf_idx + length <= sLTP_shp_Q14.len(); the lag pointers start at
    // non-negative offsets (for lag > 0 / voiced). The 4-lane loads of `coef` (24 entries), sAR2_Q14
    // (24) and sLPC_Q14 (96 = 16 history + 80 samples, read at most up to index 16 + length) stay
    // within their arrays; the LPC history loads read psLPC_Q14[-15 ..= 0], which is inside the
    // first NSQ_LPC_BUF_LENGTH entries. The final xq stores write xq[0..length] in 8/1-lane steps.
    unsafe {
        let mut LTP_pred_Q13: i32;
        let mut LPC_pred_Q10: i32;
        let mut n_AR_Q12: i32;
        let mut n_LTP_Q13: i32;
        let mut n_LF_Q12: i32;
        let mut r_Q10: i32;
        let mut rr_Q10: i32;
        let mut q1_Q0: i32;
        let mut q1_Q10: i32;
        let mut q2_Q10: i32;
        let mut rd1_Q20: i32;
        let mut rd2_Q20: i32;
        let mut exc_Q14: i32;
        let mut LPC_exc_Q14: i32;
        let mut xq_Q14: i32;
        let mut tmp1: i32;
        let mut tmp2: i32;
        let mut sLF_AR_shp_Q14: i32;
        let mut sDiff_shp_Q14: i32;
        let mut xq_prev_Q14: i32;
        let mut coef = [0i32; MAX_SHAPE_LPC_ORDER_];
        let (mut lpc0, mut lpc1, mut lpc2, mut lpc3): (v128, v128, v128, v128);
        let (mut ar0, mut ar1, mut ar2, mut ar3, mut ar4, mut ar5): (v128, v128, v128, v128, v128, v128);
        let (c4, c5): (v128, v128);

        debug_assert!(predictLPCOrder == 10 || predictLPCOrder == 16);
        debug_assert!(shapingLPCOrder as usize <= 4 * AR_VECS);
        debug_assert!(AR_VECS == 4 || AR_VECS == 6);

        debug_assert!(xq_off + length as usize <= NSQ.xq.len());
        debug_assert!(NSQ.sLTP_shp_buf_idx - lag + HARM_SHAPE_FIR_TAPS / 2 - 2 >= 0 || lag <= 0);
        debug_assert!(NSQ.sLTP_buf_idx - lag + LTP_ORDER / 2 - 4 >= 0 || signalType != TYPE_VOICED);
        debug_assert!(NSQ.sLTP_buf_idx + length <= sLTP_Q15.len() as i32);
        debug_assert!(NSQ.sLTP_shp_buf_idx + length <= NSQ.sLTP_shp_Q14.len() as i32);
        debug_assert!(NSQ.sLTP_shp_buf_idx >= 1);
        let sLTP_shp = NSQ.sLTP_shp_Q14.as_mut_ptr();
        let sLPC = NSQ.sLPC_Q14.as_mut_ptr();
        let sLTP_Q15p = sLTP_Q15.as_mut_ptr();
        let xq = NSQ.xq.as_mut_ptr().add(xq_off);

        let mut shp_lag_ptr = sLTP_shp.wrapping_offset((NSQ.sLTP_shp_buf_idx - lag + HARM_SHAPE_FIR_TAPS / 2) as isize);
        let mut pred_lag_ptr = sLTP_Q15p.wrapping_offset((NSQ.sLTP_buf_idx - lag + LTP_ORDER / 2) as isize);
        let Gain_Q10 = silk_RSHIFT(Gain_Q16, 6);

        /* Set up short term AR state */
        let mut psLPC_Q14 = sLPC.add(NSQ_LPC_BUF_LENGTH_ - 1);

        /* Short-term prediction: lpc0..lpc3 = psLPC_Q14[ -16 .. -1 ] against a_Q12[ 16 .. 1 ],
           with zeros above predictLPCOrder; psLPC_Q14[ 0 ] is xq_prev_Q14 */
        coef[0] = 0;
        for i in 1..MAX_LPC_ORDER_ {
            coef[MAX_LPC_ORDER_ - i] = if (i as i32) < predictLPCOrder { a_Q12[i] as i32 } else { 0 };
        }
        let a0 = ld(coef.as_ptr().add(0));
        let a1 = ld(coef.as_ptr().add(4));
        let a2 = ld(coef.as_ptr().add(8));
        let a3 = ld(coef.as_ptr().add(12));
        lpc0 = ld(psLPC_Q14.sub(15));
        lpc1 = ld(psLPC_Q14.sub(11));
        lpc2 = ld(psLPC_Q14.sub(7));
        lpc3 = ld(psLPC_Q14.sub(3));
        lpc3 = i32x4_shuffle::<3, 4, 5, 6>(lpc2, lpc3);
        lpc2 = i32x4_shuffle::<3, 4, 5, 6>(lpc1, lpc2);
        lpc1 = i32x4_shuffle::<3, 4, 5, 6>(lpc0, lpc1);
        lpc0 = i32x4_shuffle::<3, 4, 5, 6>(i32x4_splat(0), lpc0);
        xq_prev_Q14 = *psLPC_Q14;

        /* Noise shape feedback: the filter input is { sDiff_shp_Q14, sAR2_Q14[ 0 ], sAR2_Q14[ 1 ], ... }.
           ar0..ar5 hold sAR2_Q14[], against AR_shp_Q13[ 1 .. ] with zeros from shapingLPCOrder on. */
        for i in 0..4 * AR_VECS {
            coef[i] = if (i as i32) + 1 < shapingLPCOrder { AR_shp_Q13[i + 1] as i32 } else { 0 };
        }
        let c0 = ld(coef.as_ptr().add(0));
        let c1 = ld(coef.as_ptr().add(4));
        let c2 = ld(coef.as_ptr().add(8));
        let c3 = ld(coef.as_ptr().add(12));
        let sAR2 = NSQ.sAR2_Q14.as_mut_ptr();
        ar0 = ld(sAR2.add(0));
        ar1 = ld(sAR2.add(4));
        ar2 = ld(sAR2.add(8));
        ar3 = ld(sAR2.add(12));
        if AR_VECS > 4 {
            c4 = ld(coef.as_ptr().add(16));
            c5 = ld(coef.as_ptr().add(20));
            ar4 = ld(sAR2.add(16));
            ar5 = ld(sAR2.add(20));
        } else {
            c4 = i32x4_splat(0);
            c5 = c4;
            ar4 = c4;
            ar5 = c4;
        }

        /* Long-term prediction: taps 1..4 in a vector, against pred_lag_ptr[ -4 .. -1 ] */
        let b_sx = i32x4(b_Q14[4] as i32, b_Q14[3] as i32, b_Q14[2] as i32, b_Q14[1] as i32);

        sDiff_shp_Q14 = NSQ.sDiff_shp_Q14;
        sLF_AR_shp_Q14 = NSQ.sLF_AR_shp_Q14;

        for i in 0..length as usize {
            let ltp: v128;
            let mut ar: v128;
            let mut t0: v128;

            /* Generate dither */
            NSQ.rand_seed = silk_RAND(NSQ.rand_seed);

            /* Short-term prediction */
            let lpc = i32x4_add(
                i32x4_add(silk_SMULWB_wasm(lpc0, a0), silk_SMULWB_wasm(lpc1, a1)),
                i32x4_add(silk_SMULWB_wasm(lpc2, a2), silk_SMULWB_wasm(lpc3, a3)),
            );

            /* Long-term prediction */
            if signalType == TYPE_VOICED {
                ltp = silk_SMULWB_wasm(ld(pred_lag_ptr.sub(4)), b_sx);
            } else {
                ltp = i32x4_splat(0);
            }

            /* Noise shape feedback */
            debug_assert!((shapingLPCOrder & 1) == 0); /* check that order is even */
            ar = i32x4_add(
                i32x4_add(silk_SMULWB_wasm(ar0, c0), silk_SMULWB_wasm(ar1, c1)),
                i32x4_add(silk_SMULWB_wasm(ar2, c2), silk_SMULWB_wasm(ar3, c3)),
            );
            if AR_VECS > 4 {
                ar = i32x4_add(ar, i32x4_add(silk_SMULWB_wasm(ar4, c4), silk_SMULWB_wasm(ar5, c5)));
            }

            /* Horizontal sums: lane 0 = LPC, lane 1 = AR, lane 2 = LTP */
            t0 = i32x4_add(i32x4_shuffle::<0, 1, 4, 5>(lpc, ar), i32x4_shuffle::<2, 3, 6, 7>(lpc, ar));
            let t1 = i32x4_add(ltp, i32x4_shuffle::<2, 3, 0, 1>(ltp, ltp));
            t0 = i32x4_add(i32x4_shuffle::<0, 2, 4, 6>(t0, t1), i32x4_shuffle::<1, 3, 5, 7>(t0, t1));

            /* Avoids introducing a bias because silk_SMLAWB() always rounds to -inf */
            LPC_pred_Q10 = silk_ADD32_ovflw(silk_RSHIFT(predictLPCOrder, 1), i32x4_extract_lane::<0>(t0));
            LPC_pred_Q10 = silk_SMLAWB(LPC_pred_Q10, xq_prev_Q14, a_Q12[0] as i32);
            if signalType == TYPE_VOICED {
                LTP_pred_Q13 = silk_ADD32_ovflw(2, i32x4_extract_lane::<2>(t0));
                LTP_pred_Q13 = silk_SMLAWB(LTP_pred_Q13, *pred_lag_ptr, b_Q14[0] as i32);
                pred_lag_ptr = pred_lag_ptr.add(1);
            } else {
                LTP_pred_Q13 = 0;
            }
            n_AR_Q12 = silk_ADD32_ovflw(silk_RSHIFT(shapingLPCOrder, 1), i32x4_extract_lane::<1>(t0));
            n_AR_Q12 = silk_SMLAWB(n_AR_Q12, sDiff_shp_Q14, AR_shp_Q13[0] as i32);
            /* Q11 -> Q12 */
            n_AR_Q12 = silk_LSHIFT32(n_AR_Q12, 1);

            n_AR_Q12 = silk_SMLAWB(n_AR_Q12, sLF_AR_shp_Q14, Tilt_Q14);

            n_LF_Q12 = silk_SMULWB(*sLTP_shp.add(NSQ.sLTP_shp_buf_idx as usize - 1), LF_shp_Q14);
            n_LF_Q12 = silk_SMLAWT(n_LF_Q12, sLF_AR_shp_Q14, LF_shp_Q14);

            debug_assert!(lag > 0 || signalType != TYPE_VOICED);

            /* Combine prediction and noise shaping signals */
            tmp1 = silk_SUB32_ovflw(silk_LSHIFT32(LPC_pred_Q10, 2), n_AR_Q12); /* Q12 */
            tmp1 = silk_SUB32_ovflw(tmp1, n_LF_Q12); /* Q12 */
            if lag > 0 {
                /* Symmetric, packed FIR coefficients */
                n_LTP_Q13 = silk_SMULWB(silk_ADD_SAT32(*shp_lag_ptr, *shp_lag_ptr.sub(2)), HarmShapeFIRPacked_Q14);
                n_LTP_Q13 = silk_SMLAWT(n_LTP_Q13, *shp_lag_ptr.sub(1), HarmShapeFIRPacked_Q14);
                n_LTP_Q13 = silk_LSHIFT(n_LTP_Q13, 1);
                shp_lag_ptr = shp_lag_ptr.add(1);

                tmp2 = silk_SUB32(LTP_pred_Q13, n_LTP_Q13); /* Q13 */
                tmp1 = silk_ADD32_ovflw(tmp2, silk_LSHIFT32(tmp1, 1)); /* Q13 */
                tmp1 = silk_RSHIFT_ROUND(tmp1, 3); /* Q10 */
            } else {
                tmp1 = silk_RSHIFT_ROUND(tmp1, 2); /* Q10 */
            }

            r_Q10 = silk_SUB32(x_sc_Q10[i], tmp1); /* residual error Q10 */

            /* Flip sign depending on dither */
            if NSQ.rand_seed < 0 {
                r_Q10 = -r_Q10;
            }
            r_Q10 = silk_LIMIT_32!(r_Q10, -(31 << 10), 30 << 10);

            /* Find two quantization level candidates and measure their rate-distortion */
            q1_Q10 = silk_SUB32(r_Q10, offset_Q10);
            q1_Q0 = silk_RSHIFT(q1_Q10, 10);
            if Lambda_Q10 > 2048 {
                /* For aggressive RDO, the bias becomes more than one pulse. */
                let rdo_offset = Lambda_Q10 / 2 - 512;
                if q1_Q10 > rdo_offset {
                    q1_Q0 = silk_RSHIFT(q1_Q10 - rdo_offset, 10);
                } else if q1_Q10 < -rdo_offset {
                    q1_Q0 = silk_RSHIFT(q1_Q10 + rdo_offset, 10);
                } else if q1_Q10 < 0 {
                    q1_Q0 = -1;
                } else {
                    q1_Q0 = 0;
                }
            }
            if q1_Q0 > 0 {
                q1_Q10 = silk_SUB32(silk_LSHIFT(q1_Q0, 10), QUANT_LEVEL_ADJUST_Q10);
                q1_Q10 = silk_ADD32(q1_Q10, offset_Q10);
                q2_Q10 = silk_ADD32(q1_Q10, 1024);
                rd1_Q20 = silk_SMULBB(q1_Q10, Lambda_Q10);
                rd2_Q20 = silk_SMULBB(q2_Q10, Lambda_Q10);
            } else if q1_Q0 == 0 {
                q1_Q10 = offset_Q10;
                q2_Q10 = silk_ADD32(q1_Q10, 1024 - QUANT_LEVEL_ADJUST_Q10);
                rd1_Q20 = silk_SMULBB(q1_Q10, Lambda_Q10);
                rd2_Q20 = silk_SMULBB(q2_Q10, Lambda_Q10);
            } else if q1_Q0 == -1 {
                q2_Q10 = offset_Q10;
                q1_Q10 = silk_SUB32(q2_Q10, 1024 - QUANT_LEVEL_ADJUST_Q10);
                rd1_Q20 = silk_SMULBB(-q1_Q10, Lambda_Q10);
                rd2_Q20 = silk_SMULBB(q2_Q10, Lambda_Q10);
            } else {
                /* Q1_Q0 < -1 */
                q1_Q10 = silk_ADD32(silk_LSHIFT(q1_Q0, 10), QUANT_LEVEL_ADJUST_Q10);
                q1_Q10 = silk_ADD32(q1_Q10, offset_Q10);
                q2_Q10 = silk_ADD32(q1_Q10, 1024);
                rd1_Q20 = silk_SMULBB(-q1_Q10, Lambda_Q10);
                rd2_Q20 = silk_SMULBB(-q2_Q10, Lambda_Q10);
            }
            rr_Q10 = silk_SUB32(r_Q10, q1_Q10);
            rd1_Q20 = silk_SMLABB(rd1_Q20, rr_Q10, rr_Q10);
            rr_Q10 = silk_SUB32(r_Q10, q2_Q10);
            rd2_Q20 = silk_SMLABB(rd2_Q20, rr_Q10, rr_Q10);

            if rd2_Q20 < rd1_Q20 {
                q1_Q10 = q2_Q10;
            }

            let pulse = silk_RSHIFT_ROUND(q1_Q10, 10) as i8;
            *pulses.add(i) = pulse;

            /* Excitation */
            exc_Q14 = silk_LSHIFT(q1_Q10, 4);
            if NSQ.rand_seed < 0 {
                exc_Q14 = -exc_Q14;
            }

            /* Add predictions */
            LPC_exc_Q14 = silk_ADD_LSHIFT32(exc_Q14, LTP_pred_Q13, 1);
            xq_Q14 = silk_ADD32_ovflw(LPC_exc_Q14, silk_LSHIFT32(LPC_pred_Q10, 4));

            /* Update states */
            psLPC_Q14 = psLPC_Q14.add(1);
            *psLPC_Q14 = xq_Q14;
            lpc0 = i32x4_shuffle::<1, 2, 3, 4>(lpc0, lpc1);
            lpc1 = i32x4_shuffle::<1, 2, 3, 4>(lpc1, lpc2);
            lpc2 = i32x4_shuffle::<1, 2, 3, 4>(lpc2, lpc3);
            lpc3 = i32x4_shuffle::<1, 2, 3, 4>(lpc3, i32x4_splat(xq_prev_Q14));
            xq_prev_Q14 = xq_Q14;
            if AR_VECS > 4 {
                ar5 = i32x4_shuffle::<3, 4, 5, 6>(ar4, ar5);
                ar4 = i32x4_shuffle::<3, 4, 5, 6>(ar3, ar4);
            }
            ar3 = i32x4_shuffle::<3, 4, 5, 6>(ar2, ar3);
            ar2 = i32x4_shuffle::<3, 4, 5, 6>(ar1, ar2);
            ar1 = i32x4_shuffle::<3, 4, 5, 6>(ar0, ar1);
            ar0 = i32x4_shuffle::<3, 4, 5, 6>(i32x4_splat(sDiff_shp_Q14), ar0);
            sDiff_shp_Q14 = silk_SUB32_ovflw(xq_Q14, silk_LSHIFT32(x_sc_Q10[i], 4));
            sLF_AR_shp_Q14 = silk_SUB32_ovflw(sDiff_shp_Q14, silk_LSHIFT32(n_AR_Q12, 2));

            *sLTP_shp.add(NSQ.sLTP_shp_buf_idx as usize) =
                silk_SUB32_ovflw(sLF_AR_shp_Q14, silk_LSHIFT32(n_LF_Q12, 2));
            *sLTP_Q15p.add(NSQ.sLTP_buf_idx as usize) = silk_LSHIFT(LPC_exc_Q14, 1);
            NSQ.sLTP_shp_buf_idx += 1;
            NSQ.sLTP_buf_idx += 1;

            /* Make dither dependent on quantized signal */
            NSQ.rand_seed = silk_ADD32_ovflw(NSQ.rand_seed, pulse as i32);
        }
        NSQ.sDiff_shp_Q14 = sDiff_shp_Q14;
        NSQ.sLF_AR_shp_Q14 = sLF_AR_shp_Q14;

        /* Save the noise shaping state; entries at and above shapingLPCOrder are unchanged */
        st(coef.as_mut_ptr().add(0), ar0);
        st(coef.as_mut_ptr().add(4), ar1);
        st(coef.as_mut_ptr().add(8), ar2);
        st(coef.as_mut_ptr().add(12), ar3);
        if AR_VECS > 4 {
            st(coef.as_mut_ptr().add(16), ar4);
            st(coef.as_mut_ptr().add(20), ar5);
        }
        core::ptr::copy_nonoverlapping(coef.as_ptr(), sAR2, shapingLPCOrder as usize);

        /* Scale XQ back to normal level before saving */
        {
            let gain = i32x4_splat(Gain_Q10);
            psLPC_Q14 = sLPC.add(NSQ_LPC_BUF_LENGTH_);
            let mut i: i32 = 0;
            while i < length - 7 {
                let mut xq0 = silk_SMULWW_wasm(ld(psLPC_Q14.add(i as usize)), gain);
                let mut xq1 = silk_SMULWW_wasm(ld(psLPC_Q14.add(i as usize + 4)), gain);
                /* silk_RSHIFT_ROUND( xq, 8 ), saturated to 16 bits */
                xq0 = i32x4_shr(i32x4_add(i32x4_shr(xq0, 7), i32x4_splat(1)), 1);
                xq1 = i32x4_shr(i32x4_add(i32x4_shr(xq1, 7), i32x4_splat(1)), 1);
                v128_store(xq.add(i as usize) as *mut v128, i16x8_narrow_i32x4(xq0, xq1));
                i += 8;
            }
            while i < length {
                *xq.add(i as usize) =
                    silk_SAT16(silk_RSHIFT_ROUND(silk_SMULWW(*psLPC_Q14.add(i as usize), Gain_Q10), 8)) as i16;
                i += 1;
            }
        }

        /* Update LPC synth buffer */
        core::ptr::copy(sLPC.add(length as usize), sLPC, NSQ_LPC_BUF_LENGTH_);
    }
}

/// Noise shaping quantization (silk_NSQ_wasm).
///
/// Raw pointers as in C: the encoder passes `&psEnc->sCmn.sNSQ`, `&psEnc->sCmn.indices` and
/// `psEnc->sCmn.pulses`, which all lie inside `*psEncC`. Only configuration fields are read
/// through `psEncC`. `pulses` receives `frame_length` values.
pub unsafe fn silk_NSQ(
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
    // the kernels called below.
    unsafe {
        let subfr_length = MAX_SUB_FRAME_LENGTH;
        let ltp_mem_length = LTP_MEM_LENGTH;
        let frame_length = MAX_FRAME_LENGTH;
        let nb_subfr = MAX_NB_SUBFR;
        let predictLPCOrder = MAX_LPC_ORDER;
        let shapingLPCOrder = (*psEncC).shapingLPCOrder;
        let NSQ = &mut *NSQ;
        let signalType = (*psIndices).signalType as i32;
        let quantOffsetType = (*psIndices).quantOffsetType as i32;
        let NLSFInterpCoef_Q2 = (*psIndices).NLSFInterpCoef_Q2 as i32;

        debug_assert!(nb_subfr == MAX_NB_SUBFR && subfr_length == MAX_SUB_FRAME_LENGTH);
        debug_assert!(frame_length == nb_subfr * subfr_length && x16.len() >= frame_length as usize);

        NSQ.rand_seed = (*psIndices).Seed as i32;

        /* Set unvoiced lag to the previous one, overwrite later for voiced */
        let mut lag = NSQ.lagPrev;

        debug_assert!(NSQ.prev_gain_Q16 != 0);

        let offset_Q10 = silk_Quantization_Offsets_Q10[(signalType >> 1) as usize][quantOffsetType as usize] as i32;

        let LSF_interpolation_flag = if NLSFInterpCoef_Q2 == 4 { 0 } else { 1 };

        let stk = stack::save();
        let sLTP_Q15 = stk.alloc::<i32>((ltp_mem_length + frame_length) as usize);
        let sLTP = stk.alloc::<i16>((ltp_mem_length + frame_length) as usize);
        let x_sc_Q10 = stk.alloc::<i32>(subfr_length as usize);
        /* Set up pointers to start of sub frame */
        NSQ.sLTP_shp_buf_idx = ltp_mem_length;
        NSQ.sLTP_buf_idx = ltp_mem_length;
        let mut pxq = ltp_mem_length as usize; /* pxq = &NSQ->xq[ pxq ] */
        let mut x16_off = 0usize;
        let mut pulses = pulses;
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

                    NSQ.rewhite_flag = 1;
                    NSQ.sLTP_buf_idx = ltp_mem_length;
                }
            }

            silk_nsq_scale_states_wasm(
                subfr_length,
                ltp_mem_length,
                NSQ,
                &x16[x16_off..],
                x_sc_Q10,
                sLTP,
                sLTP_Q15,
                k,
                LTP_scale_Q14,
                Gains_Q16,
                pitchL,
                signalType,
            );

            if shapingLPCOrder <= 16 {
                silk_noise_shape_quantizer_wasm::<4>(
                    NSQ, signalType, x_sc_Q10, pulses, pxq, sLTP_Q15, A_Q12, B_Q14, AR_shp_Q13, lag,
                    HarmShapeFIRPacked_Q14, Tilt_Q14[k], LF_shp_Q14[k], Gains_Q16[k], Lambda_Q10, offset_Q10,
                    subfr_length, shapingLPCOrder, predictLPCOrder,
                );
            } else {
                silk_noise_shape_quantizer_wasm::<6>(
                    NSQ, signalType, x_sc_Q10, pulses, pxq, sLTP_Q15, A_Q12, B_Q14, AR_shp_Q13, lag,
                    HarmShapeFIRPacked_Q14, Tilt_Q14[k], LF_shp_Q14[k], Gains_Q16[k], Lambda_Q10, offset_Q10,
                    subfr_length, shapingLPCOrder, predictLPCOrder,
                );
            }

            x16_off += subfr_length as usize;
            pulses = pulses.add(subfr_length as usize);
            pxq += subfr_length as usize;
        }

        /* Update lagPrev for next frame */
        NSQ.lagPrev = pitchL[nb_subfr as usize - 1];

        /* Save quantized speech and noise shaping signals */
        NSQ.xq.copy_within(frame_length as usize..(frame_length + ltp_mem_length) as usize, 0);
        NSQ.sLTP_shp_Q14.copy_within(frame_length as usize..(frame_length + ltp_mem_length) as usize, 0);
    }
}
