//! Port of silk/float/pitch_analysis_core_FLP.c.
//!
//! Only the 16 kHz / 4 subframe (20 ms) configuration is ported: the 8 kHz and 12 kHz input
//! paths and the 10 ms (`nb_subfr == 2`) codebooks are removed (`debug_assert!`ed).

/*****************************************************************************
* Pitch analyser function
******************************************************************************/

use crate::celt::pitch::celt_pitch_xcorr;
use crate::silk::float::energy_FLP::silk_energy_FLP;
use crate::silk::float::inner_product_FLP::silk_inner_product_FLP;
use crate::silk::float::sigproc_flp::*;
use crate::silk::float::sort_FLP::silk_insertion_sort_decreasing_FLP;
use crate::silk::macros::*;
use crate::silk::pitch_est_defines::*;
use crate::silk::pitch_est_tables::*;

const SCRATCH_SIZE: usize = 22;

const NB_SUBFR: usize = PE_MAX_NB_SUBFR as usize;
const CBKS_STAGE3: usize = PE_NB_CBKS_STAGE3_MAX as usize;
const STAGE3_LAGS: usize = PE_NB_STAGE3_LAGS as usize;
const C_LEN: usize = ((PE_MAX_LAG >> 1) + 5) as usize;

/* silk/resampler_down2.c: allpass coefficients from silk/resampler_rom.h */
const silk_resampler_down2_0: i32 = 9872;
const silk_resampler_down2_1: i32 = 39809 - 65536;

/// Downsample by a factor 2 (silk/resampler_down2.c; only used here).
fn silk_resampler_down2(
    S: &mut [i32],     /* I/O  State vector [ 2 ]                                          */
    out: &mut [i16],   /* O    Output signal [ floor(len/2) ]                              */
    in_: &[i16],       /* I    Input signal [ len ]                                        */
    inLen: i32,        /* I    Number of input samples                                     */
) {
    let len2 = silk_RSHIFT32(inLen, 1);

    /* Internal variables and state are in Q10 format */
    for k in 0..len2 as usize {
        /* Convert to Q10 */
        let in32 = silk_LSHIFT(in_.at(2 * k) as i32, 10);

        /* All-pass section for even input sample */
        let Y = silk_SUB32(in32, S[0]);
        let X = silk_SMLAWB(Y, Y, silk_resampler_down2_1);
        let mut out32 = silk_ADD32(S[0], X);
        S[0] = silk_ADD32(in32, X);

        /* Convert to Q10 */
        let in32 = silk_LSHIFT(in_.at(2 * k + 1) as i32, 10);

        /* All-pass section for odd input sample, and add to output of previous section */
        let Y = silk_SUB32(in32, S[1]);
        let X = silk_SMULWB(Y, silk_resampler_down2_0);
        out32 = silk_ADD32(out32, S[1]);
        out32 = silk_ADD32(out32, X);
        S[1] = silk_ADD32(in32, X);

        /* Add, convert back to int16 and store to output */
        *out.at_mut(k) = silk_SAT16(silk_RSHIFT_ROUND(out32, 11)) as i16;
    }
}

/// CORE PITCH ANALYSIS FUNCTION
///
/// Returns the voicing estimate: 0 voiced, 1 unvoiced.
/// Only `Fs_kHz == 16 && nb_subfr == 4` is supported.
#[allow(unused_assignments)]
pub fn silk_pitch_analysis_core_FLP(
    frame: &[f32],            /* I    Signal of length PE_FRAME_LENGTH_MS*Fs_kHz                  */
    pitch_out: &mut [i32],    /* O    Pitch lag values [nb_subfr]                                 */
    lagIndex: &mut i16,       /* O    Lag Index                                                   */
    contourIndex: &mut i8,    /* O    Pitch contour Index                                         */
    LTPCorr: &mut f32,        /* I/O  Normalized correlation; input: value from previous frame    */
    mut prevLag: i32,         /* I    Last lag of previous frame; set to zero is unvoiced         */
    search_thres1: f32,       /* I    First stage threshold for lag candidates 0 - 1              */
    search_thres2: f32,       /* I    Final threshold for lag candidates 0 - 1                    */
    Fs_kHz: i32,              /* I    sample frequency (kHz)                                      */
    complexity: i32,          /* I    Complexity setting, 0-2, where 2 is highest                 */
    nb_subfr: i32,            /* I    Number of 5 ms subframes                                    */
) -> i32 {
    let mut frame_8kHz = [0f32; (PE_MAX_FRAME_LENGTH_MS * 8) as usize];
    let mut frame_4kHz = [0f32; (PE_MAX_FRAME_LENGTH_MS * 4) as usize];
    let mut frame_8_FIX = [0i16; (PE_MAX_FRAME_LENGTH_MS * 8) as usize];
    let mut frame_4_FIX = [0i16; (PE_MAX_FRAME_LENGTH_MS * 4) as usize];
    let mut filt_state = [0i32; 6];
    let mut C = [[0f32; C_LEN]; NB_SUBFR];
    let mut xcorr = [0f32; (PE_MAX_LAG_MS * 4 - PE_MIN_LAG_MS * 4 + 1) as usize];
    let mut CC = [0f32; PE_NB_CBKS_STAGE2_EXT as usize];
    let mut d_srch = [0i32; PE_D_SRCH_LENGTH as usize];
    let mut d_comp = [0i16; C_LEN];

    /* Only the 16 kHz, 20 ms configuration is supported */
    debug_assert!(Fs_kHz == 16);
    debug_assert!(nb_subfr == PE_MAX_NB_SUBFR);

    /* Check for valid complexity setting */
    debug_assert!(complexity >= SILK_PE_MIN_COMPLEX);
    debug_assert!(complexity <= SILK_PE_MAX_COMPLEX);

    debug_assert!(search_thres1 >= 0.0f32 && search_thres1 <= 1.0f32);
    debug_assert!(search_thres2 >= 0.0f32 && search_thres2 <= 1.0f32);

    /* Set up frame lengths max / min lag for the sampling frequency */
    let frame_length = (PE_LTP_MEM_LENGTH_MS + nb_subfr * PE_SUBFR_LENGTH_MS) * Fs_kHz;
    let frame_length_4kHz = (PE_LTP_MEM_LENGTH_MS + nb_subfr * PE_SUBFR_LENGTH_MS) * 4;
    let frame_length_8kHz = (PE_LTP_MEM_LENGTH_MS + nb_subfr * PE_SUBFR_LENGTH_MS) * 8;
    let sf_length = PE_SUBFR_LENGTH_MS * Fs_kHz;
    let sf_length_4kHz = PE_SUBFR_LENGTH_MS * 4;
    let sf_length_8kHz = PE_SUBFR_LENGTH_MS * 8;
    let min_lag = PE_MIN_LAG_MS * Fs_kHz;
    let min_lag_4kHz = PE_MIN_LAG_MS * 4;
    let min_lag_8kHz = PE_MIN_LAG_MS * 8;
    let max_lag = PE_MAX_LAG_MS * Fs_kHz - 1;
    let max_lag_4kHz = PE_MAX_LAG_MS * 4;
    let max_lag_8kHz = PE_MAX_LAG_MS * 8 - 1;

    /* Resample from input sampled at Fs_kHz to 8 kHz (Fs_kHz == 16) */
    {
        /* Resample to 16 -> 8 khz */
        let mut frame_16_FIX = [0i16; (16 * PE_MAX_FRAME_LENGTH_MS) as usize];
        silk_float2short_array(&mut frame_16_FIX, frame, frame_length);
        filt_state[..2].fill(0);
        silk_resampler_down2(&mut filt_state, &mut frame_8_FIX, &frame_16_FIX, frame_length);
        silk_short2float_array(&mut frame_8kHz, &frame_8_FIX, frame_length_8kHz);
    }

    /* Decimate again to 4 kHz */
    filt_state[..2].fill(0);
    silk_resampler_down2(&mut filt_state, &mut frame_4_FIX, &frame_8_FIX, frame_length_8kHz);
    silk_short2float_array(&mut frame_4kHz, &frame_4_FIX, frame_length_4kHz);

    /* Low-pass filter */
    for i in (1..frame_length_4kHz as usize).rev() {
        *frame_4kHz.at_mut(i) = silk_ADD_SAT16(frame_4kHz.at(i) as i32, frame_4kHz.at(i - 1) as i32) as f32;
    }

    /******************************************************************************
    * FIRST STAGE, operating in 4 khz
    ******************************************************************************/
    let mut target_ptr = silk_LSHIFT(sf_length_4kHz, 2) as usize;
    for _k in 0..nb_subfr >> 1 {
        /* Check that we are within range of the array */
        debug_assert!(target_ptr + sf_length_8kHz as usize <= frame_length_4kHz as usize);

        let mut basis_ptr = target_ptr - min_lag_4kHz as usize;

        /* Check that we are within range of the array */
        debug_assert!(basis_ptr + sf_length_8kHz as usize <= frame_length_4kHz as usize);

        celt_pitch_xcorr(
            frame_4kHz.rg(target_ptr..),
            frame_4kHz.rg(target_ptr - max_lag_4kHz as usize..),
            &mut xcorr,
            sf_length_8kHz,
            max_lag_4kHz - min_lag_4kHz + 1,
        );

        /* Calculate first vector products before loop */
        let mut cross_corr: f64 = xcorr.at((max_lag_4kHz - min_lag_4kHz) as usize) as f64;
        let mut normalizer: f64 = silk_energy_FLP(frame_4kHz.rg(target_ptr..), sf_length_8kHz)
            + silk_energy_FLP(frame_4kHz.rg(basis_ptr..), sf_length_8kHz)
            + (sf_length_8kHz as f32 * 4000.0f32) as f64;

        *C[0].at_mut(min_lag_4kHz as usize) += (2.0 * cross_corr / normalizer) as f32;

        /* From now on normalizer is computed recursively */
        for d in min_lag_4kHz + 1..=max_lag_4kHz {
            basis_ptr -= 1;

            cross_corr = xcorr.at((max_lag_4kHz - d) as usize) as f64;

            /* Add contribution of new sample and remove contribution from oldest sample */
            let b0 = frame_4kHz.at(basis_ptr) as f64;
            let b1 = frame_4kHz.at(basis_ptr + sf_length_8kHz as usize) as f64;
            normalizer += b0 * b0 - b1 * b1;
            *C[0].at_mut(d as usize) += (2.0 * cross_corr / normalizer) as f32;
        }
        /* Update target pointer */
        target_ptr += sf_length_8kHz as usize;
    }

    /* Apply short-lag bias */
    for i in (min_lag_4kHz..=max_lag_4kHz).rev() {
        *C[0].at_mut(i as usize) -= C[0].at(i as usize) * i as f32 / 4096.0f32;
    }

    /* Sort */
    let mut length_d_srch = 4 + 2 * complexity;
    debug_assert!(3 * length_d_srch <= PE_D_SRCH_LENGTH);
    silk_insertion_sort_decreasing_FLP(
        C[0].rg_mut(min_lag_4kHz as usize..),
        &mut d_srch,
        max_lag_4kHz - min_lag_4kHz + 1,
        length_d_srch,
    );

    /* Escape if correlation is very low already here */
    let Cmax: f32 = C[0].at(min_lag_4kHz as usize);
    if Cmax < 0.2f32 {
        pitch_out[..nb_subfr as usize].fill(0);
        *LTPCorr = 0.0f32;
        *lagIndex = 0;
        *contourIndex = 0;
        return 1;
    }

    let threshold: f32 = search_thres1 * Cmax;
    for i in 0..length_d_srch {
        /* Convert to 8 kHz indices for the sorted correlation that exceeds the threshold */
        if C[0].at((min_lag_4kHz + i) as usize) > threshold {
            *d_srch.at_mut(i as usize) = silk_LSHIFT(d_srch.at(i as usize) + min_lag_4kHz, 1);
        } else {
            length_d_srch = i;
            break;
        }
    }
    debug_assert!(length_d_srch > 0);

    for i in min_lag_8kHz - 5..max_lag_8kHz + 5 {
        *d_comp.at_mut(i as usize) = 0;
    }
    for i in 0..length_d_srch {
        *d_comp.at_mut(d_srch.at(i as usize) as usize) = 1;
    }

    /* Convolution */
    for i in (min_lag_8kHz..=max_lag_8kHz + 3).rev() {
        let i = i as usize;
        *d_comp.at_mut(i) += d_comp.at(i - 1) + d_comp.at(i - 2);
    }

    length_d_srch = 0;
    for i in min_lag_8kHz..max_lag_8kHz + 1 {
        if d_comp.at((i + 1) as usize) > 0 {
            *d_srch.at_mut(length_d_srch as usize) = i;
            length_d_srch += 1;
        }
    }

    /* Convolution */
    for i in (min_lag_8kHz..=max_lag_8kHz + 3).rev() {
        let i = i as usize;
        *d_comp.at_mut(i) += d_comp.at(i - 1) + d_comp.at(i - 2) + d_comp.at(i - 3);
    }

    let mut length_d_comp = 0usize;
    for i in min_lag_8kHz..max_lag_8kHz + 4 {
        if d_comp.at(i as usize) > 0 {
            *d_comp.at_mut(length_d_comp) = (i - 2) as i16;
            length_d_comp += 1;
        }
    }

    /**********************************************************************************
    ** SECOND STAGE, operating at 8 kHz, on lag sections with high correlation
    *************************************************************************************/
    /*********************************************************************************
    * Find energy of each subframe projected onto its history, for a range of delays
    *********************************************************************************/
    C = [[0f32; C_LEN]; NB_SUBFR];

    let mut target_ptr = (PE_LTP_MEM_LENGTH_MS * 8) as usize;
    for k in 0..nb_subfr as usize {
        let energy_tmp: f64 = silk_energy_FLP(frame_8kHz.rg(target_ptr..), sf_length_8kHz) + 1.0;
        for j in 0..length_d_comp {
            let d = d_comp.at(j) as usize;
            let basis_ptr = target_ptr - d;
            let cross_corr: f64 = silk_inner_product_FLP(frame_8kHz.rg(basis_ptr..), frame_8kHz.rg(target_ptr..), sf_length_8kHz);
            if cross_corr > 0.0 {
                let energy: f64 = silk_energy_FLP(frame_8kHz.rg(basis_ptr..), sf_length_8kHz);
                *C[k].at_mut(d) = (2.0 * cross_corr / (energy + energy_tmp)) as f32;
            } else {
                *C[k].at_mut(d) = 0.0f32;
            }
        }
        target_ptr += sf_length_8kHz as usize;
    }

    /* search over lag range and lags codebook */
    /* scale factor for lag codebook, as a function of center lag */

    let mut CCmax: f32 = 0.0f32; /* This value doesn't matter */
    let mut CCmax_b: f32 = -1000.0f32;

    let mut CBimax: i32 = 0; /* To avoid returning undefined lag values */
    let mut lag: i32 = -1; /* To check if lag with strong enough correlation has been found */

    let prevLag_log2: f32;
    if prevLag > 0 {
        /* Fs_kHz == 16 */
        prevLag = silk_RSHIFT(prevLag, 1);
        prevLag_log2 = silk_log2(prevLag as f32 as f64);
    } else {
        prevLag_log2 = 0.0;
    }

    /* Set up stage 2 codebook based on number of subframes (nb_subfr == 4) */
    let Lag_CB = &silk_CB_lags_stage2;
    let nb_cbk_search = PE_NB_CBKS_STAGE2 as usize; /* Fs_kHz != 8 */

    for k in 0..length_d_srch as usize {
        let d = d_srch.at(k);
        for j in 0..nb_cbk_search {
            *CC.at_mut(j) = 0.0f32;
            for i in 0..nb_subfr as usize {
                /* Try all codebooks */
                *CC.at_mut(j) += C[i].at((d + Lag_CB[i].at(j) as i32) as usize);
            }
        }
        /* Find best codebook */
        let mut CCmax_new: f32 = -1000.0f32;
        let mut CBimax_new: i32 = 0;
        for i in 0..nb_cbk_search {
            if CC.at(i) > CCmax_new {
                CCmax_new = CC.at(i);
                CBimax_new = i as i32;
            }
        }

        /* Bias towards shorter lags */
        let lag_log2: f32 = silk_log2(d as f32 as f64);
        let mut CCmax_new_b: f32 = CCmax_new - PE_SHORTLAG_BIAS * nb_subfr as f32 * lag_log2;

        /* Bias towards previous lag */
        if prevLag > 0 {
            let mut delta_lag_log2_sqr: f32 = lag_log2 - prevLag_log2;
            delta_lag_log2_sqr *= delta_lag_log2_sqr;
            CCmax_new_b -= PE_PREVLAG_BIAS * nb_subfr as f32 * (*LTPCorr) * delta_lag_log2_sqr / (delta_lag_log2_sqr + 0.5f32);
        }

        if CCmax_new_b > CCmax_b                        /* Find maximum biased correlation                  */
            && CCmax_new > nb_subfr as f32 * search_thres2 /* Correlation needs to be high enough to be voiced */
        {
            CCmax_b = CCmax_new_b;
            CCmax = CCmax_new;
            lag = d;
            CBimax = CBimax_new;
        }
    }

    if lag == -1 {
        /* No suitable candidate found */
        pitch_out[..PE_MAX_NB_SUBFR as usize].fill(0);
        *LTPCorr = 0.0f32;
        *lagIndex = 0;
        *contourIndex = 0;
        return 1;
    }

    /* Output normalized correlation */
    *LTPCorr = CCmax / nb_subfr as f32;
    debug_assert!(*LTPCorr >= 0.0f32);

    /* Fs_kHz > 8: search in original signal */
    {
        /* Compensate for decimation */
        debug_assert!(lag == silk_SAT16(lag));
        /* Fs_kHz == 16 */
        lag = silk_LSHIFT(lag, 1);

        lag = lag.clamp(min_lag, max_lag);
        let start_lag = silk_max_int(lag - 2, min_lag);
        let end_lag = silk_min_int(lag + 2, max_lag);
        let mut lag_new = lag; /* to avoid undefined lag */
        CBimax = 0; /* to avoid undefined lag */

        CCmax = -1000.0f32;

        /* Calculate the correlations and energies needed in stage 3 */
        let mut cross_corr_st3 = [[[0f32; STAGE3_LAGS]; CBKS_STAGE3]; NB_SUBFR];
        let mut energies_st3 = [[[0f32; STAGE3_LAGS]; CBKS_STAGE3]; NB_SUBFR];
        silk_P_Ana_calc_corr_st3(&mut cross_corr_st3, frame, start_lag, sf_length, nb_subfr, complexity);
        silk_P_Ana_calc_energy_st3(&mut energies_st3, frame, start_lag, sf_length, nb_subfr, complexity);

        let mut lag_counter = 0usize;
        debug_assert!(lag == silk_SAT16(lag));
        let contour_bias: f32 = PE_FLATCONTOUR_BIAS / lag as f32;

        /* Set up cbk parameters according to complexity setting and frame length (nb_subfr == 4) */
        let nb_cbk_search = silk_nb_cbk_searchs_stage3[complexity as usize] as usize;
        let Lag_CB = &silk_CB_lags_stage3;

        let target_ptr = (PE_LTP_MEM_LENGTH_MS * Fs_kHz) as usize;
        let energy_tmp: f64 = silk_energy_FLP(frame.rg(target_ptr..), nb_subfr * sf_length) + 1.0;
        for d in start_lag..=end_lag {
            for j in 0..nb_cbk_search {
                let mut cross_corr: f64 = 0.0;
                let mut energy: f64 = energy_tmp;
                for k in 0..nb_subfr as usize {
                    cross_corr += cross_corr_st3[k][j].at(lag_counter) as f64;
                    energy += energies_st3[k][j].at(lag_counter) as f64;
                }
                let mut CCmax_new: f32;
                if cross_corr > 0.0 {
                    CCmax_new = (2.0 * cross_corr / energy) as f32;
                    /* Reduce depending on flatness of contour */
                    CCmax_new *= 1.0f32 - contour_bias * j as f32;
                } else {
                    CCmax_new = 0.0f32;
                }

                if CCmax_new > CCmax && (d + silk_CB_lags_stage3[0][j] as i32) <= max_lag {
                    CCmax = CCmax_new;
                    lag_new = d;
                    CBimax = j as i32;
                }
            }
            lag_counter += 1;
        }

        for k in 0..nb_subfr as usize {
            *pitch_out.at_mut(k) = lag_new + Lag_CB[k].at(CBimax as usize) as i32;
            *pitch_out.at_mut(k) = pitch_out.at(k).clamp(min_lag, PE_MAX_LAG_MS * Fs_kHz);
        }
        *lagIndex = (lag_new - min_lag) as i16;
        *contourIndex = CBimax as i8;
    }
    debug_assert!(*lagIndex >= 0);
    /* return as voiced */
    0
}

/***********************************************************************
 * Calculates the correlations used in stage 3 search. In order to cover
 * the whole lag codebook for all the searched offset lags (lag +- 2),
 * the following correlations are needed in each sub frame:
 *
 * sf1: lag range [-8,...,7] total 16 correlations
 * sf2: lag range [-4,...,4] total 9 correlations
 * sf3: lag range [-3,....4] total 8 correltions
 * sf4: lag range [-6,....8] total 15 correlations
 *
 * In total 48 correlations. The direct implementation computed in worst
 * case 4*12*5 = 240 correlations, but more likely around 120.
 ***********************************************************************/
fn silk_P_Ana_calc_corr_st3(
    cross_corr_st3: &mut [[[f32; STAGE3_LAGS]; CBKS_STAGE3]; NB_SUBFR], /* O 3 DIM correlation array */
    frame: &[f32],      /* I vector to correlate                                            */
    start_lag: i32,     /* I start lag                                                      */
    sf_length: i32,     /* I sub frame length                                               */
    nb_subfr: i32,      /* I number of subframes                                            */
    complexity: i32,    /* I Complexity setting                                             */
) {
    let mut scratch_mem = [0f32; SCRATCH_SIZE];
    let mut xcorr = [0f32; SCRATCH_SIZE];

    debug_assert!(complexity >= SILK_PE_MIN_COMPLEX);
    debug_assert!(complexity <= SILK_PE_MAX_COMPLEX);
    debug_assert!(nb_subfr == PE_MAX_NB_SUBFR);

    let Lag_range = &silk_Lag_range_stage3[complexity as usize];
    let Lag_CB = &silk_CB_lags_stage3;
    let nb_cbk_search = silk_nb_cbk_searchs_stage3[complexity as usize] as usize;

    let mut target_ptr = silk_LSHIFT(sf_length, 2); /* Pointer to middle of frame */
    for k in 0..nb_subfr as usize {
        let mut lag_counter = 0usize;

        /* Calculate the correlations for each subframe */
        let lag_low = Lag_range[k][0] as i32;
        let lag_high = Lag_range[k][1] as i32;
        debug_assert!(lag_high - lag_low + 1 <= SCRATCH_SIZE as i32);
        celt_pitch_xcorr(
            frame.rg(target_ptr as usize..),
            frame.rg((target_ptr - start_lag - lag_high) as usize..),
            &mut xcorr,
            sf_length,
            lag_high - lag_low + 1,
        );
        for j in lag_low..=lag_high {
            debug_assert!(lag_counter < SCRATCH_SIZE);
            *scratch_mem.at_mut(lag_counter) = xcorr.at((lag_high - j) as usize);
            lag_counter += 1;
        }

        let delta = Lag_range[k][0] as i32;
        for i in 0..nb_cbk_search {
            /* Fill out the 3 dim array that stores the correlations for */
            /* each code_book vector for each start lag */
            let idx = (Lag_CB[k].at(i) as i32 - delta) as usize;
            for j in 0..STAGE3_LAGS {
                debug_assert!(idx + j < SCRATCH_SIZE);
                debug_assert!(idx + j < lag_counter);
                *cross_corr_st3[k][i].at_mut(j) = scratch_mem.at(idx + j);
            }
        }
        target_ptr += sf_length;
    }
}

/********************************************************************/
/* Calculate the energies for first two subframes. The energies are */
/* calculated recursively.                                          */
/********************************************************************/
fn silk_P_Ana_calc_energy_st3(
    energies_st3: &mut [[[f32; STAGE3_LAGS]; CBKS_STAGE3]; NB_SUBFR], /* O 3 DIM correlation array */
    frame: &[f32],      /* I vector to correlate                                            */
    start_lag: i32,     /* I start lag                                                      */
    sf_length: i32,     /* I sub frame length                                               */
    nb_subfr: i32,      /* I number of subframes                                            */
    complexity: i32,    /* I Complexity setting                                             */
) {
    let mut scratch_mem = [0f32; SCRATCH_SIZE];

    debug_assert!(complexity >= SILK_PE_MIN_COMPLEX);
    debug_assert!(complexity <= SILK_PE_MAX_COMPLEX);
    debug_assert!(nb_subfr == PE_MAX_NB_SUBFR);

    let Lag_range = &silk_Lag_range_stage3[complexity as usize];
    let Lag_CB = &silk_CB_lags_stage3;
    let nb_cbk_search = silk_nb_cbk_searchs_stage3[complexity as usize] as usize;

    let mut target_ptr = silk_LSHIFT(sf_length, 2);
    for k in 0..nb_subfr as usize {
        let mut lag_counter = 0usize;

        /* Calculate the energy for first lag */
        let basis_ptr = (target_ptr - (start_lag + Lag_range[k][0] as i32)) as usize;
        let mut energy: f64 = silk_energy_FLP(frame.rg(basis_ptr..), sf_length) + 1e-3;
        debug_assert!(energy >= 0.0);
        *scratch_mem.at_mut(lag_counter) = energy as f32;
        lag_counter += 1;

        let lag_diff = Lag_range[k][1] as i32 - Lag_range[k][0] as i32 + 1;
        for i in 1..lag_diff as usize {
            /* remove part outside new window */
            let a = frame.at(basis_ptr + sf_length as usize - i) as f64;
            energy -= a * a;
            debug_assert!(energy >= 0.0);

            /* add part that comes into window */
            let b = frame.at(basis_ptr - i) as f64;
            energy += b * b;
            debug_assert!(energy >= 0.0);
            debug_assert!(lag_counter < SCRATCH_SIZE);
            *scratch_mem.at_mut(lag_counter) = energy as f32;
            lag_counter += 1;
        }

        let delta = Lag_range[k][0] as i32;
        for i in 0..nb_cbk_search {
            /* Fill out the 3 dim array that stores the correlations for    */
            /* each code_book vector for each start lag                     */
            let idx = (Lag_CB[k].at(i) as i32 - delta) as usize;
            for j in 0..STAGE3_LAGS {
                debug_assert!(idx + j < SCRATCH_SIZE);
                debug_assert!(idx + j < lag_counter);
                *energies_st3[k][i].at_mut(j) = scratch_mem.at(idx + j);
                debug_assert!(energies_st3[k][i].at(j) >= 0.0f32);
            }
        }
        target_ptr += sf_length;
    }
}
