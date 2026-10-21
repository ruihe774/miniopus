//! Port of silk/VAD.c and silk/wasm/VAD_wasm.c.
//!
//! The WASM build exports `silk_VAD_Init`, `silk_VAD_GetNoiseLevels` (non-static when
//! `OPUS_WASM_SIMD128` is defined) and the kernel `silk_VAD_GetSA_Q8_wasm`, which is what
//! `silk_VAD_GetSA_Q8` resolves to. The scalar `silk_VAD_GetSA_Q8_c` is not ported.

use core::arch::wasm32::*;

use crate::silk::ana_filt_bank_1::silk_ana_filt_bank_1;
use crate::silk::define::*;
use crate::silk::lin2log::silk_lin2log;
use crate::silk::macros::*;
use crate::silk::sigm_Q15::silk_sigm_Q15;
use crate::silk::structs::{silk_VAD_state, silk_encoder_state};
use crate::stack;

const VAD_N_BANDS_: usize = VAD_N_BANDS as usize;

/// Initialization of the Silk VAD
pub fn silk_VAD_Init(
    psSilk_VAD: &mut silk_VAD_state, /* I/O  Pointer to Silk VAD state                   */
) {
    /* reset state memory */
    *psSilk_VAD = unsafe { core::mem::zeroed() };

    /* init noise levels */
    /* Initialize array with approx pink noise levels (psd proportional to inverse of frequency) */
    for b in 0..VAD_N_BANDS_ {
        // div_s: b + 1 >= 1
        psSilk_VAD.NoiseLevelBias[b] = silk_max_32(silk_DIV32_16(VAD_NOISE_LEVELS_BIAS, b as i32 + 1), 1);
    }

    /* Initialize state */
    for b in 0..VAD_N_BANDS_ {
        psSilk_VAD.NL[b] = silk_MUL(100, psSilk_VAD.NoiseLevelBias[b]);
        // div_s: NL[b] > 0 (initialised positive, updated with saturating adds)
        psSilk_VAD.inv_NL[b] = silk_DIV32(silk_int32_MAX, psSilk_VAD.NL[b]);
    }
    psSilk_VAD.counter = 15;

    /* init smoothed energy-to-noise ratio*/
    for b in 0..VAD_N_BANDS_ {
        psSilk_VAD.NrgRatioSmth_Q8[b] = 100 * 256; /* 100 * 256 --> 20 dB SNR */
    }
}

/* Weighting factors for tilt measure */
static tiltWeights: [i32; VAD_N_BANDS_] = [30000, 6000, -12000, -12000];

/// Sum of ( x[ i ] >> 3 )^2. Each square is at most 2^24, so (as in the C code)
/// the 32-bit accumulation cannot overflow for len <= 128.
#[inline(always)]
unsafe fn silk_VAD_energy_wasm(x: *const i16, len: i32) -> i32 {
    // SAFETY: the caller guarantees x[0..len] is readable. The 8-lane load reads x[i..i + 8] with
    // i < len - 7, the 4-lane `load64_zero` reads x[i..i + 4] with i < len - 3, and the scalar tail
    // reads x[i] with i < len.
    unsafe {
        let mut i: i32;
        let mut sum: i32;
        let mut acc: v128;
        let mut v: v128;

        acc = i32x4_splat(0);
        i = 0;
        while i < len - 7 {
            v = i16x8_shr(v128_load(x.add(i as usize) as *const v128), 3);
            acc = i32x4_add(acc, i32x4_dot_i16x8(v, v));
            i += 8;
        }
        if i < len - 3 {
            /* Upper four lanes are zero and contribute nothing */
            v = i16x8_shr(v128_load64_zero(x.add(i as usize) as *const u64), 3);
            acc = i32x4_add(acc, i32x4_dot_i16x8(v, v));
            i += 4;
        }
        acc = i32x4_add(acc, i32x4_shuffle::<2, 3, 0, 1>(acc, acc));
        acc = i32x4_add(acc, i32x4_shuffle::<1, 0, 3, 2>(acc, acc));
        sum = i32x4_extract_lane::<0>(acc);

        while i < len {
            let x_tmp = silk_RSHIFT(*x.add(i as usize) as i32, 3);
            sum = silk_SMLABB(sum, x_tmp, x_tmp);
            i += 1;
        }
        sum
    }
}

/// Get the speech activity level in Q8 (silk_VAD_GetSA_Q8_wasm).
///
/// Raw pointers as in C: the encoder passes `psEncC->inputBuf` as `pIn`, which lies
/// inside `*psEncC`. Only `sVAD`, `speech_activity_Q8`, `input_tilt_Q15` and
/// `input_quality_bands_Q15` are written.
pub unsafe fn silk_VAD_GetSA_Q8(
    psEncC: *mut silk_encoder_state, /* I/O  Encoder state                               */
    pIn: *const i16,                 /* I    PCM input                                   */
) {
    // SAFETY: `psEncC` is a valid, exclusive encoder state and `pIn` points to `frame_length`
    // (MAX_FRAME_LENGTH) readable samples (its only caller passes `inputBuf`, an array of that
    // length). `X` is a scratch buffer of X_offset[3] + frame_length / 2 samples; the three
    // analysis filter banks write `frame_length / 2`, `/ 4` and `/ 8` samples to the offsets
    // above, which fit in it by construction of `X_offset`.
    unsafe {
        let mut SA_Q15: i32;
        let pSNR_dB_Q7: i32;
        let mut input_tilt: i32;
        let mut decimated_framelength: i32;
        let mut dec_subframe_length: i32;
        let mut dec_subframe_offset: i32;
        let mut SNR_Q7: i32;
        let mut sumSquared: i32;
        let smooth_coef_Q16: i32;
        let mut Xnrg = [0i32; VAD_N_BANDS_];
        let mut NrgToNoiseRatio_Q8 = [0i32; VAD_N_BANDS_];
        let mut speech_nrg: i32;
        let mut X_offset = [0i32; VAD_N_BANDS_];
        let frame_length = MAX_FRAME_LENGTH;
        let psSilk_VAD: &mut silk_VAD_state = &mut (*psEncC).sVAD;

        let st = stack::save();

        /* Safety checks */
        debug_assert!(VAD_N_BANDS == 4);

        /***********************/
        /* Filter and Decimate */
        /***********************/
        let decimated_framelength1 = silk_RSHIFT(frame_length, 1);
        let decimated_framelength2 = silk_RSHIFT(frame_length, 2);
        decimated_framelength = silk_RSHIFT(frame_length, 3);
        /* Decimate into 4 bands:
           0       L      3L       L              3L                             5L
                   -      --       -              --                             --
                   8       8       2               4                              4

           [0-1 kHz| temp. |1-2 kHz|    2-4 kHz    |            4-8 kHz           |

           They're arranged to allow the minimal ( frame_length / 4 ) extra
           scratch space during the downsampling process */
        X_offset[0] = 0;
        X_offset[1] = decimated_framelength + decimated_framelength2;
        X_offset[2] = X_offset[1] + decimated_framelength;
        X_offset[3] = X_offset[2] + decimated_framelength2;
        let X_buf = st.alloc::<i16>((X_offset[3] + decimated_framelength1) as usize);
        let X = X_buf.as_mut_ptr();

        /* 0-8 kHz to 0-4 kHz and 4-8 kHz */
        silk_ana_filt_bank_1(pIn, &mut psSilk_VAD.AnaState, X, X.add(X_offset[3] as usize), frame_length);

        /* 0-4 kHz to 0-2 kHz and 2-4 kHz */
        silk_ana_filt_bank_1(X, &mut psSilk_VAD.AnaState1, X, X.add(X_offset[2] as usize), decimated_framelength1);

        /* 0-2 kHz to 0-1 kHz and 1-2 kHz */
        silk_ana_filt_bank_1(X, &mut psSilk_VAD.AnaState2, X, X.add(X_offset[1] as usize), decimated_framelength2);

        /*********************************************/
        /* HP filter on lowest band (differentiator) */
        /*********************************************/
        /* X[ i ] = ( X[ i ] >> 1 ) - ( X[ i - 1 ] >> 1 ), processed from the end so
           that every load still sees the unmodified input. The difference of two
           halved 16-bit values always fits in 16 bits. */
        let HPstateTmp = silk_RSHIFT(*X.add(decimated_framelength as usize - 1) as i32, 1) as i16;
        let mut i = decimated_framelength - 8;
        while i > 0 {
            let cur = i16x8_shr(v128_load(X.add(i as usize) as *const v128), 1);
            let prev = i16x8_shr(v128_load(X.add(i as usize - 1) as *const v128), 1);
            v128_store(X.add(i as usize) as *mut v128, i16x8_sub(cur, prev));
            i -= 8;
        }
        i += 7;
        while i > 0 {
            *X.add(i as usize) =
                (silk_RSHIFT(*X.add(i as usize) as i32, 1) - silk_RSHIFT(*X.add(i as usize - 1) as i32, 1)) as i16;
            i -= 1;
        }
        *X = (silk_RSHIFT(*X as i32, 1) - psSilk_VAD.HPstate as i32) as i16;
        psSilk_VAD.HPstate = HPstateTmp;

        /*************************************/
        /* Calculate the energy in each band */
        /*************************************/
        for b in 0..VAD_N_BANDS_ {
            /* Find the decimated framelength in the non-uniformly divided bands */
            decimated_framelength = silk_RSHIFT(frame_length, silk_min_int(VAD_N_BANDS - b as i32, VAD_N_BANDS - 1));

            /* Split length into subframe lengths */
            dec_subframe_length = silk_RSHIFT(decimated_framelength, VAD_INTERNAL_SUBFRAMES_LOG2);
            dec_subframe_offset = 0;

            /* Compute energy per sub-frame */
            /* initialize with summed energy of last subframe */
            Xnrg[b] = psSilk_VAD.XnrgSubfr[b];
            sumSquared = 0;
            for s in 0..VAD_INTERNAL_SUBFRAMES {
                sumSquared =
                    silk_VAD_energy_wasm(X.add((X_offset[b] + dec_subframe_offset) as usize), dec_subframe_length);

                /* Safety check */
                debug_assert!(sumSquared >= 0);

                /* Add/saturate summed energy of current subframe */
                if s < VAD_INTERNAL_SUBFRAMES - 1 {
                    Xnrg[b] = silk_ADD_POS_SAT32(Xnrg[b], sumSquared);
                } else {
                    /* Look-ahead subframe */
                    Xnrg[b] = silk_ADD_POS_SAT32(Xnrg[b], silk_RSHIFT(sumSquared, 1));
                }

                dec_subframe_offset += dec_subframe_length;
            }
            psSilk_VAD.XnrgSubfr[b] = sumSquared;
        }

        /********************/
        /* Noise estimation */
        /********************/
        silk_VAD_GetNoiseLevels(&Xnrg, psSilk_VAD);

        /***********************************************/
        /* Signal-plus-noise to noise ratio estimation */
        /***********************************************/
        sumSquared = 0;
        input_tilt = 0;
        for b in 0..VAD_N_BANDS_ {
            speech_nrg = Xnrg[b] - psSilk_VAD.NL[b];
            if speech_nrg > 0 {
                /* Divide, with sufficient resolution */
                if (Xnrg[b] as u32 & 0xFF800000) == 0 {
                    // div_s: NL + 1 >= 1 since NL >= 0
                    NrgToNoiseRatio_Q8[b] = silk_DIV32(silk_LSHIFT(Xnrg[b], 8), psSilk_VAD.NL[b] + 1);
                } else {
                    // div_s: (NL >> 8) + 1 >= 1 since NL >= 0
                    NrgToNoiseRatio_Q8[b] = silk_DIV32(Xnrg[b], silk_RSHIFT(psSilk_VAD.NL[b], 8) + 1);
                }

                /* Convert to log domain */
                SNR_Q7 = silk_lin2log(NrgToNoiseRatio_Q8[b]) - 8 * 128;

                /* Sum-of-squares */
                sumSquared = silk_SMLABB(sumSquared, SNR_Q7, SNR_Q7); /* Q14 */

                /* Tilt measure */
                if speech_nrg < (1i32 << 20) {
                    /* Scale down SNR value for small subband speech energies */
                    SNR_Q7 = silk_SMULWB(silk_LSHIFT(silk_SQRT_APPROX(speech_nrg), 6), SNR_Q7);
                }
                input_tilt = silk_SMLAWB(input_tilt, tiltWeights[b], SNR_Q7);
            } else {
                NrgToNoiseRatio_Q8[b] = 256;
            }
        }

        /* Mean-of-squares */
        sumSquared = silk_DIV32_16(sumSquared, VAD_N_BANDS); /* Q14 */

        /* Root-mean-square approximation, scale to dBs, and write to output pointer */
        pSNR_dB_Q7 = (3 * silk_SQRT_APPROX(sumSquared)) as i16 as i32; /* Q7 */

        /*********************************/
        /* Speech Probability Estimation */
        /*********************************/
        SA_Q15 = silk_sigm_Q15(silk_SMULWB(VAD_SNR_FACTOR_Q16, pSNR_dB_Q7) - VAD_NEGATIVE_OFFSET_Q5);

        /**************************/
        /* Frequency Tilt Measure */
        /**************************/
        (*psEncC).input_tilt_Q15 = silk_LSHIFT(silk_sigm_Q15(input_tilt) - 16384, 1);

        /**************************************************/
        /* Scale the sigmoid output based on power levels */
        /**************************************************/
        speech_nrg = 0;
        for b in 0..VAD_N_BANDS_ {
            /* Accumulate signal-without-noise energies, higher frequency bands have more weight */
            speech_nrg += (b as i32 + 1) * silk_RSHIFT(Xnrg[b] - psSilk_VAD.NL[b], 4);
        }

        /* 20 ms frames */
        speech_nrg = silk_RSHIFT32(speech_nrg, 1);
        /* Power scaling */
        if speech_nrg <= 0 {
            SA_Q15 = silk_RSHIFT(SA_Q15, 1);
        } else if speech_nrg < 16384 {
            speech_nrg = silk_LSHIFT32(speech_nrg, 16);

            /* square-root */
            speech_nrg = silk_SQRT_APPROX(speech_nrg);
            SA_Q15 = silk_SMULWB(32768 + speech_nrg, SA_Q15);
        }

        /* Copy the resulting speech activity in Q8 */
        (*psEncC).speech_activity_Q8 = silk_min_int(silk_RSHIFT(SA_Q15, 7), silk_uint8_MAX);

        /***********************************/
        /* Energy Level and SNR estimation */
        /***********************************/
        /* Smoothing coefficient */
        smooth_coef_Q16 = silk_SMULWB(VAD_SNR_SMOOTH_COEF_Q18, silk_SMULWB(SA_Q15, SA_Q15));

        for b in 0..VAD_N_BANDS_ {
            /* compute smoothed energy-to-noise ratio per band */
            psSilk_VAD.NrgRatioSmth_Q8[b] = silk_SMLAWB(
                psSilk_VAD.NrgRatioSmth_Q8[b],
                NrgToNoiseRatio_Q8[b] - psSilk_VAD.NrgRatioSmth_Q8[b],
                smooth_coef_Q16,
            );

            /* signal to noise ratio in dB per band */
            SNR_Q7 = 3 * (silk_lin2log(psSilk_VAD.NrgRatioSmth_Q8[b]) - 8 * 128);
            /* quality = sigmoid( 0.25 * ( SNR_dB - 16 ) ); */
            (*psEncC).input_quality_bands_Q15[b] = silk_sigm_Q15(silk_RSHIFT(SNR_Q7 - 16 * 128, 4));
        }
    }
}

/// Noise level estimation
pub fn silk_VAD_GetNoiseLevels(
    pX: &[i32; VAD_N_BANDS_],        /* I    subband energies                            */
    psSilk_VAD: &mut silk_VAD_state, /* I/O  Pointer to Silk VAD state                   */
) {
    let mut nl: i32;
    let mut nrg: i32;
    let mut inv_nrg: i32;
    let mut coef: i32;
    let min_coef: i32;

    /* Initially faster smoothing */
    if psSilk_VAD.counter < 1000 {
        /* 1000 = 20 sec */
        // div_s: (counter >> 4) + 1 >= 1
        min_coef = silk_DIV32_16(silk_int16_MAX, silk_RSHIFT(psSilk_VAD.counter, 4) + 1);
        /* Increment frame counter */
        psSilk_VAD.counter += 1;
    } else {
        min_coef = 0;
    }

    for k in 0..VAD_N_BANDS_ {
        /* Get old noise level estimate for current band */
        nl = psSilk_VAD.NL[k];
        debug_assert!(nl >= 0);

        /* Add bias */
        nrg = silk_ADD_POS_SAT32(pX[k], psSilk_VAD.NoiseLevelBias[k]);
        debug_assert!(nrg > 0);

        /* Invert energies */
        // div_s: nrg > 0 (saturating add of positive terms)
        inv_nrg = silk_DIV32(silk_int32_MAX, nrg);
        debug_assert!(inv_nrg >= 0);

        /* Less update when subband energy is high */
        if nrg > silk_LSHIFT(nl, 3) {
            coef = VAD_NOISE_LEVEL_SMOOTH_COEF_Q16 >> 3;
        } else if nrg < nl {
            coef = VAD_NOISE_LEVEL_SMOOTH_COEF_Q16;
        } else {
            coef = silk_SMULWB(silk_SMULWW(inv_nrg, nl), VAD_NOISE_LEVEL_SMOOTH_COEF_Q16 << 1);
        }

        /* Initially faster smoothing */
        coef = silk_max_int(coef, min_coef);

        /* Smooth inverse energies */
        psSilk_VAD.inv_NL[k] = silk_SMLAWB(psSilk_VAD.inv_NL[k], inv_nrg - psSilk_VAD.inv_NL[k], coef);
        debug_assert!(psSilk_VAD.inv_NL[k] >= 0);

        /* Compute noise level by inverting again */
        // div_s: inv_NL[k] > 0
        nl = silk_DIV32(silk_int32_MAX, psSilk_VAD.inv_NL[k]);
        debug_assert!(nl >= 0);

        /* Limit noise levels (guarantee 7 bits of head room) */
        nl = silk_min!(nl, 0x00FFFFFF);

        /* Store as part of state */
        psSilk_VAD.NL[k] = nl;
    }
}
