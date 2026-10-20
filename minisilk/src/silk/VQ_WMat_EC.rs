//! Port of silk/wasm/VQ_WMat_EC_wasm.c.

use core::arch::wasm32::*;

use crate::silk::define::*;
use crate::silk::lin2log::silk_lin2log;
use crate::silk::macros::*;

const LTP_ORDER_: usize = LTP_ORDER as usize;

/// silk_SMULWB() on each lane; the lanes of b hold 16-bit values.
#[inline(always)]
fn silk_SMULWB_wasm(a: v128, b: v128) -> v128 {
    let hi = i32x4_mul(i32x4_shr(a, 16), b);
    let lo = i32x4_mul(v128_and(a, i32x4_splat(0x0000FFFF)), b);
    i32x4_add(hi, i32x4_shr(lo, 16))
}

/// Entropy constrained matrix-weighted VQ, hard-coded to 5-element vectors, for a single input
/// data vector (silk_VQ_WMat_EC_wasm).
pub fn silk_VQ_WMat_EC(
    ind: &mut i8,                /* O    index of best codebook vector               */
    res_nrg_Q15: &mut i32,       /* O    best residual energy                        */
    rate_dist_Q8: &mut i32,      /* O    best total bitrate                          */
    gain_Q7: &mut i32,           /* O    sum of absolute LTP coefficients            */
    XX_Q17: &[i32],              /* I    correlation matrix                          */
    xX_Q17: &[i32],              /* I    correlation vector                          */
    cb_Q7: &[[i8; LTP_ORDER_]],  /* I    codebook                                    */
    cb_gain_Q7: &[u8],           /* I    codebook effective gain                     */
    cl_Q5: &[u8],                /* I    code length for each codebook vector        */
    subfr_len: i32,              /* I    number of samples per subframe              */
    max_gain_Q7: i32,            /* I    maximum sum of absolute LTP coefficients    */
    L: i32,                      /* I    number of vectors in codebook               */
) {
    let mut gain_tmp_Q7: i32;
    let mut neg_xX_Q24 = [0i32; 5];
    let mut sum1_Q15: i32;
    let mut sum2_Q24: i32;
    let mut bits_res_Q8: i32;
    let mut bits_tot_Q8: i32;
    let mut sum1_Q15_buf = [0i32; 4];

    debug_assert!(XX_Q17.len() >= 25 && xX_Q17.len() >= 5);
    debug_assert!(cb_Q7.len() >= L as usize && cb_gain_Q7.len() >= L as usize && cl_Q5.len() >= L as usize);
    let cb_flat: &[i8] = cb_Q7.as_flattened();

    /* Negate and convert to new Q domain */
    neg_xX_Q24[0] = -silk_LSHIFT32(xX_Q17[0], 7);
    neg_xX_Q24[1] = -silk_LSHIFT32(xX_Q17[1], 7);
    neg_xX_Q24[2] = -silk_LSHIFT32(xX_Q17[2], 7);
    neg_xX_Q24[3] = -silk_LSHIFT32(xX_Q17[3], 7);
    neg_xX_Q24[4] = -silk_LSHIFT32(xX_Q17[4], 7);

    /* Loop over codebook */
    *rate_dist_Q8 = silk_int32_MAX;
    *res_nrg_Q15 = silk_int32_MAX;
    let mut cb_row = 0usize; /* cb_row_Q7 = &cb_Q7[ cb_row ] */
    /* If things go really bad, at least *ind is set to something safe. */
    *ind = 0;
    let mut k: i32 = 0;
    while k < L {
        let n: i32;
        if k < L - 3 {
            /* Quantization error of four codebook vectors at once: lane i handles vector k + i.
               All sums wrap modulo 2^32 exactly like the 32-bit C arithmetic, so the order of the
               additions does not matter. */

            /* Transpose the 4 x 5 codebook block (20 bytes) to one vector per coefficient */
            let row = &cb_flat[cb_row..cb_row + 20];
            // SAFETY: `row` is a 20-byte slice (bounds-checked above): the loads read row[0..16]
            // and row[4..20].
            let (v0, v1) = unsafe {
                (v128_load(row.as_ptr() as *const v128), v128_load(row.as_ptr().add(4) as *const v128))
            };
            let b0123 = i8x16_shuffle::<0, 5, 10, 15, 1, 6, 11, 28, 2, 7, 12, 29, 3, 8, 13, 30>(v0, v1);
            let b4 = i8x16_shuffle::<4, 9, 14, 31, 4, 9, 14, 31, 4, 9, 14, 31, 4, 9, 14, 31>(v0, v1);
            let mut t = i16x8_extend_low_i8x16(b0123);
            let cb0 = i32x4_extend_low_i16x8(t);
            let cb1 = i32x4_extend_high_i16x8(t);
            t = i16x8_extend_high_i8x16(b0123);
            let cb2 = i32x4_extend_low_i16x8(t);
            let cb3 = i32x4_extend_high_i16x8(t);
            let cb4 = i32x4_extend_low_i16x8(i16x8_extend_low_i8x16(b4));

            /* Weighted rate */
            /* Quantization error: 1 - 2 * xX * cb + cb' * XX * cb */
            let mut sum1 = i32x4_splat(SILK_FIX_CONST(1.001, 15));

            /* first row of XX_Q17 */
            let mut sum2 = i32x4_add(i32x4_splat(neg_xX_Q24[0]), i32x4_mul(i32x4_splat(XX_Q17[1]), cb1));
            sum2 = i32x4_add(sum2, i32x4_mul(i32x4_splat(XX_Q17[2]), cb2));
            sum2 = i32x4_add(sum2, i32x4_mul(i32x4_splat(XX_Q17[3]), cb3));
            sum2 = i32x4_add(sum2, i32x4_mul(i32x4_splat(XX_Q17[4]), cb4));
            sum2 = i32x4_shl(sum2, 1);
            sum2 = i32x4_add(sum2, i32x4_mul(i32x4_splat(XX_Q17[0]), cb0));
            sum1 = i32x4_add(sum1, silk_SMULWB_wasm(sum2, cb0));

            /* second row of XX_Q17 */
            sum2 = i32x4_add(i32x4_splat(neg_xX_Q24[1]), i32x4_mul(i32x4_splat(XX_Q17[7]), cb2));
            sum2 = i32x4_add(sum2, i32x4_mul(i32x4_splat(XX_Q17[8]), cb3));
            sum2 = i32x4_add(sum2, i32x4_mul(i32x4_splat(XX_Q17[9]), cb4));
            sum2 = i32x4_shl(sum2, 1);
            sum2 = i32x4_add(sum2, i32x4_mul(i32x4_splat(XX_Q17[6]), cb1));
            sum1 = i32x4_add(sum1, silk_SMULWB_wasm(sum2, cb1));

            /* third row of XX_Q17 */
            sum2 = i32x4_add(i32x4_splat(neg_xX_Q24[2]), i32x4_mul(i32x4_splat(XX_Q17[13]), cb3));
            sum2 = i32x4_add(sum2, i32x4_mul(i32x4_splat(XX_Q17[14]), cb4));
            sum2 = i32x4_shl(sum2, 1);
            sum2 = i32x4_add(sum2, i32x4_mul(i32x4_splat(XX_Q17[12]), cb2));
            sum1 = i32x4_add(sum1, silk_SMULWB_wasm(sum2, cb2));

            /* fourth row of XX_Q17 */
            sum2 = i32x4_add(i32x4_splat(neg_xX_Q24[3]), i32x4_mul(i32x4_splat(XX_Q17[19]), cb4));
            sum2 = i32x4_shl(sum2, 1);
            sum2 = i32x4_add(sum2, i32x4_mul(i32x4_splat(XX_Q17[18]), cb3));
            sum1 = i32x4_add(sum1, silk_SMULWB_wasm(sum2, cb3));

            /* last row of XX_Q17 */
            sum2 = i32x4_splat(silk_LSHIFT32(neg_xX_Q24[4], 1));
            sum2 = i32x4_add(sum2, i32x4_mul(i32x4_splat(XX_Q17[24]), cb4));
            sum1 = i32x4_add(sum1, silk_SMULWB_wasm(sum2, cb4));

            // SAFETY: `sum1_Q15_buf` is a [i32; 4], exactly 16 writable bytes.
            unsafe { v128_store(sum1_Q15_buf.as_mut_ptr() as *mut v128, sum1) };
            n = 4;
        } else {
            /* Leftover codebook vectors, same as the C code */
            let cb_row_Q7 = &cb_flat[cb_row..cb_row + LTP_ORDER_];
            let cb = |j: usize| cb_row_Q7[j] as i32;
            sum1_Q15 = SILK_FIX_CONST(1.001, 15);

            /* first row of XX_Q17 */
            sum2_Q24 = silk_MLA(neg_xX_Q24[0], XX_Q17[1], cb(1));
            sum2_Q24 = silk_MLA(sum2_Q24, XX_Q17[2], cb(2));
            sum2_Q24 = silk_MLA(sum2_Q24, XX_Q17[3], cb(3));
            sum2_Q24 = silk_MLA(sum2_Q24, XX_Q17[4], cb(4));
            sum2_Q24 = silk_LSHIFT32(sum2_Q24, 1);
            sum2_Q24 = silk_MLA(sum2_Q24, XX_Q17[0], cb(0));
            sum1_Q15 = silk_SMLAWB(sum1_Q15, sum2_Q24, cb(0));

            /* second row of XX_Q17 */
            sum2_Q24 = silk_MLA(neg_xX_Q24[1], XX_Q17[7], cb(2));
            sum2_Q24 = silk_MLA(sum2_Q24, XX_Q17[8], cb(3));
            sum2_Q24 = silk_MLA(sum2_Q24, XX_Q17[9], cb(4));
            sum2_Q24 = silk_LSHIFT32(sum2_Q24, 1);
            sum2_Q24 = silk_MLA(sum2_Q24, XX_Q17[6], cb(1));
            sum1_Q15 = silk_SMLAWB(sum1_Q15, sum2_Q24, cb(1));

            /* third row of XX_Q17 */
            sum2_Q24 = silk_MLA(neg_xX_Q24[2], XX_Q17[13], cb(3));
            sum2_Q24 = silk_MLA(sum2_Q24, XX_Q17[14], cb(4));
            sum2_Q24 = silk_LSHIFT32(sum2_Q24, 1);
            sum2_Q24 = silk_MLA(sum2_Q24, XX_Q17[12], cb(2));
            sum1_Q15 = silk_SMLAWB(sum1_Q15, sum2_Q24, cb(2));

            /* fourth row of XX_Q17 */
            sum2_Q24 = silk_MLA(neg_xX_Q24[3], XX_Q17[19], cb(4));
            sum2_Q24 = silk_LSHIFT32(sum2_Q24, 1);
            sum2_Q24 = silk_MLA(sum2_Q24, XX_Q17[18], cb(3));
            sum1_Q15 = silk_SMLAWB(sum1_Q15, sum2_Q24, cb(3));

            /* last row of XX_Q17 */
            sum2_Q24 = silk_LSHIFT32(neg_xX_Q24[4], 1);
            sum2_Q24 = silk_MLA(sum2_Q24, XX_Q17[24], cb(4));
            sum1_Q15 = silk_SMLAWB(sum1_Q15, sum2_Q24, cb(4));

            sum1_Q15_buf[0] = sum1_Q15;
            n = 1;
        }

        /* find best, in codebook order */
        for i in 0..n as usize {
            sum1_Q15 = sum1_Q15_buf[i];
            gain_tmp_Q7 = cb_gain_Q7[k as usize] as i32;

            /* Penalty for too large gain */
            let penalty = silk_LSHIFT32(silk_max!(silk_SUB32(gain_tmp_Q7, max_gain_Q7), 0), 11);

            if sum1_Q15 >= 0 {
                /* Translate residual energy to bits using high-rate assumption (6 dB ==> 1 bit/sample) */
                bits_res_Q8 = silk_SMULBB(subfr_len, silk_lin2log(sum1_Q15 + penalty) - (15 << 7));
                /* In the following line we reduce the codelength component by half ("-1"); seems to slightly improve quality */
                bits_tot_Q8 = silk_ADD_LSHIFT32(bits_res_Q8, cl_Q5[k as usize] as i32, 3 - 1);
                if bits_tot_Q8 <= *rate_dist_Q8 {
                    *rate_dist_Q8 = bits_tot_Q8;
                    *res_nrg_Q15 = sum1_Q15 + penalty;
                    *ind = k as i8;
                    *gain_Q7 = gain_tmp_Q7;
                }
            }
            k += 1;
        }

        /* Go to next cbk vector */
        cb_row += n as usize * LTP_ORDER_;
    }
}
