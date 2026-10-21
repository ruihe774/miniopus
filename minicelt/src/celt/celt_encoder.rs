//! Port of celt/celt_encoder.c (48 kHz, 20 ms frames, mono/stereo, VBR / constrained VBR).

use super::bands::{
    SPREAD_NONE, SPREAD_NORMAL, compute_band_energies, haar1, hysteresis_decision, normalise_bands,
    quant_all_bands, spreading_decision,
};
use super::celt::{
    COMBFILTER_MAXPERIOD, COMBFILTER_MINPERIOD, comb_filter, init_caps, spread_icdf, tapset_icdf,
    tf_select_table, trim_icdf,
};
use super::entcode::{BITRES, EC_ILOG, ec_enc, ec_get_error, ec_tell, ec_tell_frac};
use super::entenc::{
    ec_enc_bit_logp, ec_enc_bits, ec_enc_done, ec_enc_icdf, ec_enc_shrink, ec_enc_uint,
};
use super::mathops::{
    CELT_SIG_SCALE, EPSILON, PI, acos64, exp2, floor64, log2, maxabs, rcp, sqrt64,
};
use super::mdct::clt_mdct_forward;
use super::modes;
use super::pitch::{celt_inner_prod, pitch_downsample_factor2, pitch_search, remove_doubling};
use super::quant_bands::{
    amp2Log2, eMeans, quant_coarse_energy, quant_energy_finalise, quant_fine_energy,
};
use super::rate::clt_compute_allocation;
use crate::opus::defines::OPUS_INTERNAL_ERROR;
use crate::stack;

/// LM: log2 of the number of short blocks per frame (20 ms frames only).
const LM: i32 = 3;

/// Maximum channel count; the trailing arrays are sized for it.
pub const MAX_CHANNELS: usize = 2;
/// modes::overlap
pub const OVERLAP: usize = 120;
/// modes::nbEBands
pub const NB_EBANDS: usize = 21;

/// OpusCustomEncoder / CELTEncoder. The C struct ends in variable-size arrays laid out after
/// `in_mem` (each `channels` times its per-channel size); here they are fixed-size fields
/// sized for `MAX_CHANNELS` and indexed with the same per-channel strides as in C. The fields
/// that are constant in the supported configuration (`start`, `vbr`, `clip`, `upsample`,
/// `force_intra`, `disable_pf`, `disable_inv`, `signalling`, `lfe`, `arch`, `energy_mask`, `analysis`,
/// `silk_info`, `mode`) are not stored. The all-zero state has no pointers, so the static
/// encoder lives in `.bss`.
pub struct CELTEncoder {
    pub channels: i32,
    pub stream_channels: i32,

    pub complexity: i32,
    pub end: i32,

    pub bitrate: i32,
    /// If zero, VBR can do whatever it likes with the rate
    pub constrained_vbr: i32,
    pub loss_rate: i32,
    pub lsb_depth: i32,

    /* Everything beyond this point gets cleared on a reset */
    pub rng: u32,
    pub spread_decision: i32,
    pub delayedIntra: f32,
    pub tonal_average: i32,
    pub lastCodedBands: i32,
    pub hf_average: i32,
    pub tapset_decision: i32,

    pub prefilter_period: i32,
    pub prefilter_gain: f32,
    pub prefilter_tapset: i32,
    pub consec_transient: i32,

    pub preemph_memE: [f32; 2],

    /* VBR-related parameters */
    pub vbr_reservoir: i32,
    pub vbr_drift: i32,
    pub vbr_offset: i32,
    pub vbr_count: i32,
    pub overlap_max: f32,
    pub stereo_saving: f32,
    pub intensity: i32,
    pub spec_avg: f32,

    /// Size = channels*overlap
    pub in_mem: [f32; MAX_CHANNELS * OVERLAP],
    /// Size = channels*COMBFILTER_MAXPERIOD
    pub prefilter_mem: [f32; MAX_CHANNELS * COMBFILTER_MAXPERIOD as usize],
    /// Size = channels*nbEBands
    pub oldBandE: [f32; MAX_CHANNELS * NB_EBANDS],
    /// Size = channels*nbEBands
    pub oldLogE: [f32; MAX_CHANNELS * NB_EBANDS],
    /// Size = channels*nbEBands
    pub oldLogE2: [f32; MAX_CHANNELS * NB_EBANDS],
    /// Size = channels*nbEBands
    pub energyError: [f32; MAX_CHANNELS * NB_EBANDS],
}

impl CELTEncoder {
    /// All-zero state (as after the C memset).
    pub const fn new_zeroed() -> Self {
        CELTEncoder {
            channels: 0,
            stream_channels: 0,
            complexity: 0,
            end: 0,
            bitrate: 0,
            constrained_vbr: 0,
            loss_rate: 0,
            lsb_depth: 0,
            rng: 0,
            spread_decision: 0,
            delayedIntra: 0.0,
            tonal_average: 0,
            lastCodedBands: 0,
            hf_average: 0,
            tapset_decision: 0,
            prefilter_period: 0,
            prefilter_gain: 0.0,
            prefilter_tapset: 0,
            consec_transient: 0,
            preemph_memE: [0.0; 2],
            vbr_reservoir: 0,
            vbr_drift: 0,
            vbr_offset: 0,
            vbr_count: 0,
            overlap_max: 0.0,
            stereo_saving: 0.0,
            intensity: 0,
            spec_avg: 0.0,
            in_mem: [0.0; MAX_CHANNELS * OVERLAP],
            prefilter_mem: [0.0; MAX_CHANNELS * COMBFILTER_MAXPERIOD as usize],
            oldBandE: [0.0; MAX_CHANNELS * NB_EBANDS],
            oldLogE: [0.0; MAX_CHANNELS * NB_EBANDS],
            oldLogE2: [0.0; MAX_CHANNELS * NB_EBANDS],
            energyError: [0.0; MAX_CHANNELS * NB_EBANDS],
        }
    }
}

/// celt_encoder_init(st, 48000, channels) = opus_custom_encoder_init_arch(st, mode48000_960_120,
/// channels) with the OPUS_RESET_STATE of its end. `st` is the all-zero state (the Opus layer
/// clears it first) and `channels` is 1 or 2.
pub fn celt_encoder_init(st: &mut CELTEncoder, channels: i32) {
    st.channels = channels;
    st.stream_channels = channels;

    st.end = modes::effEBands;

    st.constrained_vbr = 1;

    st.complexity = 5;
    st.lsb_depth = 24;

    /* celt_encoder_reset_state(): everything from rng on is zero already, except: */
    let n = (channels * modes::nbEBands) as usize;
    debug_assert!(n <= st.oldLogE.len() && n <= st.oldLogE2.len());
    for v in st.oldLogE2.iter_mut().take(n) {
        *v = -28.0f32;
    }
    for v in st.oldLogE.iter_mut().take(n) {
        *v = -28.0f32;
    }
    st.delayedIntra = 1.0;
    st.spread_decision = SPREAD_NORMAL;
    st.tonal_average = 256;
}

// Static helpers of celt_encoder.c, made `pub`. Conventions shared by all of them:
// - `upsample` is always 1 (debug_assert in the callers), `arch` is dropped.
// - `AnalysisInfo::valid` is always 0, so the `analysis` parameters and the code under
//   `analysis->valid` are removed (alloc_trim_analysis, dynalloc_analysis, run_prefilter,
//   compute_vbr).
// - LFE is always 0, `energy_mask` (surround) is always null: `surround_trim`,
//   `surround_dynalloc`, `surround_masking`, `has_surround_mask` and `lfe` are removed.
// - `normalize_tone_input` and `acos_approx` exist only under `FIXED_POINT` in C; the float
//   `tone_detect` calls `acos()` directly, so they are not ported.
// - M_PI is `mathops::PI` (the exact double constant).

/// Table of 6*64/x, trained on real data to minimize the average error
static inv_table: [u8; 128] = [
    255, 255, 156, 110, 86, 70, 59, 51, 45, 40, 37, 33, 31, 28, 26, 25, //
    23, 22, 21, 20, 19, 18, 17, 16, 16, 15, 15, 14, 13, 13, 12, 12, //
    12, 12, 11, 11, 11, 10, 10, 10, 9, 9, 9, 9, 9, 9, 8, 8, //
    8, 8, 8, 7, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 6, //
    6, 6, 6, 6, 6, 6, 6, 6, 6, 5, 5, 5, 5, 5, 5, 5, //
    5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, //
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, //
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, //
];

/// `in_` holds `C` channels of `len = N+overlap = 1080` samples each (`in_[c*len + i]`, i.e. the
/// `in` buffer of celt_encode_with_ec). `*tf_chan` is only written when a channel's
/// metric exceeds 0 (the caller initializes it); `*tf_estimate` is always written. Returns
/// `is_transient`.
pub fn transient_analysis(
    in_: &[f32],
    C: i32,
    tf_estimate: &mut f32,
    tf_chan: &mut i32,
    tone_freq: f32,
    toneishness: f32,
) -> i32 {
    let mut mask_metric: i32 = 0;
    /* Forward masking: 6.7 dB/ms. (The 3.3 dB/ms "weak transient" decay is hybrid only.) */
    let forward_decay: f32 = 0.0625;
    let sg = stack::save();
    const LEN: usize = 960 + modes::overlap as usize;
    const LEN2: usize = LEN / 2;
    let tmp = sg.alloc::<f32>(LEN);
    let len2 = LEN2 as i32;
    for c in 0..C {
        let mut unmask: i32;
        let mut mem0: f32 = 0.0;
        let mut mem1: f32 = 0.0;
        /* High-pass filter: (1 - 2*z^-1 + z^-2) / (1 - z^-1 + .5*z^-2) */
        for (t, &x) in tmp.iter_mut().zip(&in_[c as usize * LEN..][..LEN]) {
            let y = mem0 + x;
            /* Original code:
            mem0 = mem1 + y - 2*x;
            mem1 = x - .5f*y;
            Modified code to shorten dependency chains: */
            let mem00 = mem0;
            mem0 = mem0 - x + 0.5 * mem1;
            mem1 = x - mem00;
            *t = y;
        }
        /* First few samples are bad because we don't propagate the memory */
        tmp[..12].fill(0.0);

        let mut mean: f32 = 0.0;
        mem0 = 0.0;
        /* Grouping by two to reduce complexity */
        /* Forward pass to compute the post-echo threshold*/
        /* in place: pair i is read before tmp[i] (i <= 2i) is written, hence the cells */
        let cells = core::cell::Cell::from_mut(&mut tmp[..]).as_slice_of_cells();
        for (t, [a, b]) in cells.iter().zip(cells.as_chunks::<2>().0) {
            let x2 = a.get() * a.get() + b.get() * b.get();
            mean += x2;
            mem0 = x2 + (1.0 - forward_decay) * mem0;
            t.set(forward_decay * mem0);
        }

        mem0 = 0.0;
        let mut maxE: f32 = 0.0;
        /* Backward pass to compute the pre-echo threshold */
        for t in tmp[..LEN2].iter_mut().rev() {
            /* Backward masking: 13.9 dB/ms. */
            mem0 = *t + 0.875 * mem0;
            *t = 0.125 * mem0;
            maxE = maxE.max(0.125 * mem0);
        }

        /* Compute the ratio of the "frame energy" over the harmonic mean of the energy.
        This essentially corresponds to a bitrate-normalized temporal noise-to-mask
        ratio */

        /* As a compromise with the old transient detector, frame energy is the
        geometric mean of the energy and half the max */
        /* `mean * maxE*.5*len2`: float product, then double (.5 is a double literal) */
        mean = sqrt64((mean * maxE) as f64 * 0.5 * len2 as f64) as f32;
        /* Inverse of the mean energy in Q15+6 */
        let norm: f32 = len2 as f32 / (EPSILON + mean);
        /* Compute harmonic mean discarding the unreliable boundaries
        The data is smooth, so we only take 1/4th of the samples */
        unmask = 0;
        /* We should never see NaNs here. If we find any, then something really bad happened and we better abort
        before it does any damage later on. If these asserts are disabled (no hardening), then the table
        lookup a few lines below (id = ...) is likely to crash dur to an out-of-bounds read. DO NOT FIX
        that crash on NaN since it could result in a worse issue later on. */
        debug_assert!(!tmp[0].is_nan());
        debug_assert!(!norm.is_nan());
        /* i = 12, 16, .., < len2-5 */
        for q in tmp[12..LEN2 - 4].chunks_exact(4) {
            /* Do not round to nearest. floor() is a double operation; the clamp to 0..127 (MIN32/MAX32 in
            C) is done on the saturating integer conversion (also maps NaN to 0 like C's `(int)NaN` here) */
            let v: f64 = floor64((64.0 * norm * (q[0] + EPSILON)) as f64);
            let id = (v as usize).min(127);
            unmask += inv_table[id] as i32;
        }
        /* Normalize, compensate for the 1/4th of the sample and the factor of 6 in the inverse table */
        unmask = 64 * unmask * 4 / (6 * (len2 - 17));
        if unmask > mask_metric {
            *tf_chan = c;
            mask_metric = unmask;
        }
    }
    let mut is_transient = (mask_metric > 200) as i32;
    /* Prevent the transient detector from confusing the partial cycle of a
    very low frequency tone with a transient. */
    if toneishness > 0.98 && tone_freq < 0.026 {
        is_transient = 0;
        mask_metric = 0;
    }
    /* ("weak transients" only exist with allow_weak_transients, hybrid mode) */
    /* Arbitrary metric for VBR boost */
    let tf_max: f32 = 0.0f32.max(sqrt64((27 * mask_metric) as f64) as f32 - 42.0);
    /* *tf_estimate = 1 + IMIN(1, sqrt(IMAX(0, tf_max-30))/20); */
    /* QCONST32(0.139,28) is a double literal, so the subtraction and MAX32 are in double. */
    let d: f64 = (0.0069f32 * 163.0f32.min(tf_max)) as f64 - 0.139;
    let d = if 0.0 > d { 0.0 } else { d };
    *tf_estimate = sqrt64(d) as f32;
    is_transient
}

/// Looks for sudden increases of energy to decide whether we need to patch
/// the transient decision. `newE`/`oldE` are the band log energies (`C` channels of `nbEBands`).
pub fn patch_transient_decision(
    newE: &[f32],
    oldE: &[f32],
    nbEBands: i32,
    start: i32,
    end: i32,
    C: i32,
) -> i32 {
    let mut mean_diff: f32 = 0.0;
    let mut spread_old = [0.0f32; 26];
    let nb = nbEBands as usize;
    let (start, end) = (start as usize, end as usize);
    let spread_old = &mut spread_old[..end];
    /* Apply an aggressive (-6 dB/Bark) spreading function to the old frame to
    avoid false detection caused by irrelevant bands */
    if C == 1 {
        spread_old[start] = oldE[start];
        for i in start + 1..end {
            spread_old[i] = (spread_old[i - 1] - 1.0).max(oldE[i]);
        }
    } else {
        let (o0, o1) = (&oldE[..end], &oldE[nb..nb + end]);
        spread_old[start] = o0[start].max(o1[start]);
        for i in start + 1..end {
            spread_old[i] = (spread_old[i - 1] - 1.0).max(o0[i].max(o1[i]));
        }
    }
    for i in (start..end - 1).rev() {
        spread_old[i] = spread_old[i].max(spread_old[i + 1] - 1.0);
    }
    /* Compute mean increase */
    for c in 0..C as usize {
        let new = &newE[c * nb..c * nb + end];
        for i in 2i32.max(start as i32) as usize..end - 1 {
            let x1 = 0.0f32.max(new[i]);
            let x2 = 0.0f32.max(spread_old[i]);
            mean_diff = mean_diff + 0.0f32.max(x1 - x2);
        }
    }
    mean_diff = mean_diff / (C * (end as i32 - 1 - 2i32.max(start as i32))) as f32;
    (mean_diff > 1.0) as i32
}

/// Apply window and compute the MDCT for all sub-frames and all channels in a frame.
/// `in_` is the whole `in` buffer (`CC*(B*N+overlap)` samples, channel `c` at `c*(B*N+overlap)`),
/// `out` the `C*B*N` coefficients. (`upsample` is always 1, LM is 3.)
pub fn compute_mdcts(shortBlocks: i32, in_: &[f32], out: &mut [f32], C: i32, CC: i32) {
    let overlap = modes::overlap;
    let N: i32;
    let B: i32;
    let shift: i32;
    if shortBlocks != 0 {
        B = shortBlocks;
        N = modes::shortMdctSize;
        shift = modes::maxLM;
    } else {
        B = 1;
        N = modes::shortMdctSize << 3;
        shift = 0;
    }
    for c in 0..CC {
        for b in 0..B {
            /* Interleaving the sub-frames while doing the MDCTs */
            clt_mdct_forward(
                in_,
                (c * (B * N + overlap) + b * N) as usize,
                out,
                (b + c * N * B) as usize,
                shift,
                B,
            );
        }
    }
    if CC == 2 && C == 1 {
        let n = (B * N) as usize;
        debug_assert!(out.len() >= 2 * n);
        if let Some((lo, hi)) = out.split_at_mut_checked(n) {
            for (l, &h) in lo.iter_mut().zip(hi.iter()) {
                *l = 0.5 * *l + 0.5 * h;
            }
        }
    }
}

/// `pcm` holds the `CC`-interleaved input samples, `pcm[CC*i + c]` for i in 0..960 being those of
/// channel `c`; `inp[0..960]` receives the pre-emphasized signal (`in+c*(N+overlap)+overlap` of the
/// caller); `*mem` is `preemph_memE[c]`. The coefficient is `mode->preemph[0]` (`preemph[1] == 0`;
/// `upsample` is always 1, `clip` is always set, N is 960).
#[inline(never)]
pub fn celt_preemphasis(
    pcm: &[f32],
    c: usize,
    inp: &mut [f32],
    CC: i32,
    mem: &mut f32,
    need_clip: bool,
) {
    let inp = &mut inp[..960];
    if CC == 2 {
        preemph_loop(
            pcm.as_chunks::<2>().0.iter().map(|p| p[c & 1]),
            inp,
            mem,
            need_clip,
        );
    } else {
        preemph_loop(pcm.iter().copied(), inp, mem, need_clip);
    }
}

#[inline(always)]
fn preemph_loop(pcmp: impl Iterator<Item = f32>, inp: &mut [f32], mem: &mut f32, need_clip: bool) {
    let coef0 = modes::preemph0;
    let mut m = *mem;
    /* Fast path for the normal 48kHz case and no clipping */
    if !need_clip {
        for (o, p) in inp.iter_mut().zip(pcmp) {
            let x = CELT_SIG_SCALE * p;
            /* Apply pre-emphasis */
            *o = x - m;
            m = coef0 * x;
        }
    } else {
        for (o, p) in inp.iter_mut().zip(pcmp) {
            /* Clip input to avoid encoding non-portable files */
            let x = (-65536.0f32).max(65536.0f32.min(CELT_SIG_SCALE * p));
            /* Apply pre-emphasis */
            *o = x - m;
            m = coef0 * x;
        }
    }
    *mem = m;
}

pub fn l1_metric(tmp: &[f32], N: i32, lm: i32, bias: f32) -> f32 {
    let mut L1: f32 = 0.0;
    for &t in &tmp[..N as usize] {
        L1 += t.abs();
    }
    /* When in doubt, prefer good freq resolution */
    L1 = L1 + (lm as f32 * bias) * L1;
    L1
}

/// `X` holds the channels of normalized spectrum (`N0` coefficients each); `tf_res[0..len]` and
/// `importance[0..len]` are indexed by band. Returns `tf_select`.
#[inline(never)]
pub fn tf_analysis(
    len: i32,
    isTransient: i32,
    tf_res: &mut [i32],
    lambda: i32,
    X: &[f32],
    N0: i32,
    tf_estimate: f32,
    tf_chan: i32,
    importance: &[i32],
) -> i32 {
    let mut tf_select: i32;
    let mut selcost = [0i32; 2];
    let sg = stack::save();
    let bias: f32 = 0.04 * (-0.25f32).max(0.5 - tf_estimate);

    let lenu = len as usize;
    let eb = &modes::eBands[..lenu + 1];
    let nlast = ((eb[lenu] as i32 - eb[lenu - 1] as i32) << LM) as usize;
    let metric = sg.alloc::<i32>(lenu);
    let tmp = sg.alloc::<f32>(nlast);
    let tmp_1 = sg.alloc::<f32>(nlast);
    let path0 = sg.alloc::<i32>(lenu);
    let path1 = sg.alloc::<i32>(lenu);
    let importance = &importance[..lenu];
    let tf_res = &mut tf_res[..lenu];
    let X = &X[(tf_chan * N0) as usize..];

    for (m, e) in metric.iter_mut().zip(eb.windows(2)) {
        let mut best_level: i32 = 0;
        let N = (e[1] as i32 - e[0] as i32) << LM;
        let Nu = N as usize;
        /* band is too narrow to be split down to LM=-1 */
        let narrow = (e[1] as i32 - e[0] as i32) == 1;
        let off = (e[0] as usize) << LM;
        let tmp = &mut tmp[..Nu];
        tmp.copy_from_slice(&X[off..off + Nu]);
        /* Just add the right channel if we're in stereo */
        let mut L1 = l1_metric(tmp, N, if isTransient != 0 { LM } else { 0 }, bias);
        let mut best_L1 = L1;
        /* Check the -1 case for transients */
        if isTransient != 0 && !narrow {
            tmp_1[..Nu].copy_from_slice(tmp);
            haar1(tmp_1, N >> LM, 1 << LM);
            L1 = l1_metric(tmp_1, N, LM + 1, bias);
            if L1 < best_L1 {
                best_L1 = L1;
                best_level = -1;
            }
        }
        for k in 0..LM + !(isTransient != 0 || narrow) as i32 {
            let B: i32;

            if isTransient != 0 {
                B = LM - k - 1;
            } else {
                B = k + 1;
            }

            haar1(tmp, N >> k, 1 << k);

            L1 = l1_metric(tmp, N, B, bias);

            if L1 < best_L1 {
                best_L1 = L1;
                best_level = k + 1;
            }
        }
        /* metric is in Q1 to be able to select the mid-point (-0.5) for narrower bands */
        if isTransient != 0 {
            *m = 2 * best_level;
        } else {
            *m = -2 * best_level;
        }
        /* For bands that can't be split to -1, set the metric to the half-way point to avoid
        biasing the decision */
        if narrow && (*m == 0 || *m == -2 * LM) {
            *m -= 1;
        }
    }
    /* 2*tf_select_table[LM][4*isTransient+k] */
    let tfs: [i32; 4] = {
        let t = &tf_select_table[4 * (isTransient & 1) as usize..][..4];
        [
            2 * t[0] as i32,
            2 * t[1] as i32,
            2 * t[2] as i32,
            2 * t[3] as i32,
        ]
    };
    /* Search for the optimal tf resolution, including tf_select */
    tf_select = 0;
    for (sel, sc) in selcost.iter_mut().enumerate() {
        let (t0, t1) = (tfs[2 * sel], tfs[2 * sel + 1]);
        let mut cost0 = importance[0] * (metric[0] - t0).abs();
        let mut cost1 =
            importance[0] * (metric[0] - t1).abs() + if isTransient != 0 { 0 } else { lambda };
        for (&imp, &m) in importance[1..].iter().zip(&metric[1..]) {
            let curr0 = cost0.min(cost1 + lambda);
            let curr1 = (cost0 + lambda).min(cost1);
            cost0 = curr0 + imp * (m - t0).abs();
            cost1 = curr1 + imp * (m - t1).abs();
        }
        *sc = cost0.min(cost1);
    }
    /* For now, we're conservative and only allow tf_select=1 for transients.
     * If tests confirm it's useful for non-transients, we could allow it. */
    if selcost[1] < selcost[0] && isTransient != 0 {
        tf_select = 1;
    }
    let (t0, t1) = (tfs[2 * tf_select as usize], tfs[2 * tf_select as usize + 1]);
    let mut cost0 = importance[0] * (metric[0] - t0).abs();
    let mut cost1 =
        importance[0] * (metric[0] - t1).abs() + if isTransient != 0 { 0 } else { lambda };
    /* Viterbi forward pass */
    for i in 1..lenu {
        let (imp, m) = (importance[i], metric[i]);
        let curr0: i32;
        let curr1: i32;

        let from0 = cost0;
        let from1 = cost1 + lambda;
        if from0 < from1 {
            curr0 = from0;
            path0[i] = 0;
        } else {
            curr0 = from1;
            path0[i] = 1;
        }

        let from0 = cost0 + lambda;
        let from1 = cost1;
        if from0 < from1 {
            curr1 = from0;
            path1[i] = 0;
        } else {
            curr1 = from1;
            path1[i] = 1;
        }
        cost0 = curr0 + imp * (m - t0).abs();
        cost1 = curr1 + imp * (m - t1).abs();
    }
    let mut next = if cost0 < cost1 { 0 } else { 1 };
    tf_res[lenu - 1] = next;
    /* Viterbi backward pass to check the decisions */
    for i in (0..lenu - 1).rev() {
        next = if next == 1 {
            path1[i + 1]
        } else {
            path0[i + 1]
        };
        tf_res[i] = next;
    }
    tf_select
}

/// Codes the tf_res flags of bands `start..end` and `tf_select`, then converts `tf_res[start..end]`
/// to the final per-band tf change via `tf_select_table`.
pub fn tf_encode(
    start: i32,
    end: i32,
    isTransient: i32,
    tf_res: &mut [i32],
    mut tf_select: i32,
    enc: &mut ec_enc,
) {
    let mut budget: u32 = enc.storage * 8;
    let mut tell: u32 = ec_tell(enc) as u32;
    let mut logp: i32 = if isTransient != 0 { 2 } else { 4 };
    /* Reserve space to code the tf_select decision. */
    let tf_select_rsv: u32 = (tell + logp as u32 + 1 <= budget) as u32;
    budget -= tf_select_rsv;
    let mut curr: i32 = 0;
    let mut tf_changed: i32 = 0;
    let tf_res = &mut tf_res[start as usize..end as usize];
    for r in tf_res.iter_mut() {
        if tell + logp as u32 <= budget {
            ec_enc_bit_logp(enc, *r ^ curr, logp as u32);
            tell = ec_tell(enc) as u32;
            curr = *r;
            tf_changed |= curr;
        } else {
            *r = curr;
        }
        logp = if isTransient != 0 { 4 } else { 5 };
    }
    /* Only code tf_select if it would actually make a difference. */
    /* tf_select_table[LM][4*isTransient + ..]; isTransient, tf_select and the flags are 0 or 1 */
    let t: &[i8; 4] = tf_select_table[4 * (isTransient & 1) as usize..][..4]
        .try_into()
        .unwrap();
    if tf_select_rsv != 0 && t[(tf_changed & 1) as usize] != t[(2 + (tf_changed & 1)) as usize] {
        ec_enc_bit_logp(enc, tf_select, 1);
    } else {
        tf_select = 0;
    }
    for r in tf_res.iter_mut() {
        *r = t[((2 * tf_select + *r) & 3) as usize] as i32;
    }
}

/// `X` holds the two channels of normalized spectrum (`N0` coefficients each) when `C == 2`.
/// `bandLogE` is `C*nbEBands`. `analysis` and `surround_trim` are dropped (see section header).
#[inline(never)]
pub fn alloc_trim_analysis(
    X: &[f32],
    bandLogE: &[f32],
    end: i32,
    C: i32,
    N0: i32,
    stereo_saving: &mut f32,
    tf_estimate: f32,
    intensity: i32,
    equiv_rate: i32,
) -> i32 {
    let mut diff: f32 = 0.0;
    let mut trim: f32 = 5.0;
    let eb = modes::eBands;
    /* At low bitrate, reducing the trim seems to help. At higher bitrates, it's less
    clear what's best, so we're keeping it as it was before, at least for now. */
    if equiv_rate < 64000 {
        trim = 4.0;
    } else if equiv_rate < 80000 {
        let frac: i32 = (equiv_rate - 64000) >> 10;
        trim = 4.0 + (1.0f32 / 16.0) * frac as f32;
    }
    if C == 2 {
        let mut sum: f32 = 0.0; /* Q10 */
        let mut minXC: f32; /* Q10 */
        let (XL, XR) = X.split_at(N0 as usize);
        /* inner product of the two channels over band `e[0]..e[1]` */
        let band_prod = |e: &[i16]| {
            let (a, b) = ((e[0] as usize) << LM, (e[1] as usize) << LM);
            celt_inner_prod(&XL[a..b], &XR[a..b], (b - a) as i32)
        };
        /* Compute inter-channel correlation for low frequencies */
        for e in eb[..9].windows(2) {
            sum = sum + band_prod(e);
        }
        sum = (1.0f32 / 8.0) * sum;
        sum = 1.0f32.min(sum.abs());
        minXC = sum;
        for e in eb.get(8..intensity as usize + 1).unwrap_or(&[]).windows(2) {
            minXC = minXC.min(band_prod(e).abs());
        }
        minXC = 1.0f32.min(minXC.abs());
        /* mid-side savings estimations based on the LF average*/
        let logXC = log2(1.001 - sum * sum);
        /* mid-side savings estimations based on min correlation */
        let logXC2 = (0.5 * logXC).max(log2(1.001 - minXC * minXC));

        trim += (-4.0f32).max(0.75 * logXC);
        *stereo_saving = (*stereo_saving + 0.25).min(-(0.5 * logXC2));
    }

    /* Estimate spectral tilt */
    for c in 0..C as usize {
        for (i, &b) in bandLogE[c * modes::nbEBands as usize..][..end as usize - 1]
            .iter()
            .enumerate()
        {
            diff += b * (2 + 2 * i as i32 - end) as f32;
        }
    }
    diff /= (C * (end - 1)) as f32;
    trim -= (-2.0f32).max(2.0f32.min((diff + 1.0) / 6.0));
    trim -= 2.0 * tf_estimate;
    /* (analysis->valid is always 0; surround_trim is always 0) */

    let trim_index = floor64((0.5f32 + trim) as f64) as i32;
    0i32.max(10i32.min(trim_index))
}

/// Returns `dual_stereo` (0/1). `X` holds the two channels of normalized spectrum (`N0` each).
pub fn stereo_analysis(X: &[f32], N0: i32) -> i32 {
    let mut sumLR: f32 = EPSILON;
    let mut sumMS: f32 = EPSILON;
    /* bands 0..13 are contiguous: bins 0..eBands[13] << LM */
    let n13 = (modes::eBands[13] as usize) << LM;

    /* Use the L1 norm to model the entropy of the L/R signal vs the M/S signal */
    for (&L, &R) in X[..n13].iter().zip(&X[N0 as usize..][..n13]) {
        /* We cast to 32-bit first because of the -32768 case */
        let M = L + R;
        let S = L - R;
        sumLR = sumLR + (L.abs() + R.abs());
        sumMS = sumMS + (M.abs() + S.abs());
    }
    sumMS = 0.707107 * sumMS;
    let thetas: i32 = 13;
    let n = (n13 as i32) << 1;
    (((n + thetas) as f32 * sumMS) > (n as f32 * sumLR)) as i32
}

/// Median of `x[0..5]`.
pub fn median_of_5(x: &[f32]) -> f32 {
    debug_assert!(x.len() >= 5);
    let Some(&[x0, x1, x2, x3, x4]) = x.first_chunk::<5>() else {
        return 0.0;
    };
    let (mut t0, mut t1, t2, mut t3, mut t4): (f32, f32, f32, f32, f32);
    t2 = x2;
    if x0 > x1 {
        t0 = x1;
        t1 = x0;
    } else {
        t0 = x0;
        t1 = x1;
    }
    if x3 > x4 {
        t3 = x4;
        t4 = x3;
    } else {
        t3 = x3;
        t4 = x4;
    }
    if t0 > t3 {
        core::mem::swap(&mut t0, &mut t3);
        core::mem::swap(&mut t1, &mut t4);
    }
    if t2 > t1 {
        if t1 < t3 { t2.min(t3) } else { t4.min(t1) }
    } else {
        if t2 < t3 { t1.min(t3) } else { t2.min(t4) }
    }
}

/// Median of `x[0..3]`.
pub fn median_of_3(x: &[f32]) -> f32 {
    debug_assert!(x.len() >= 3);
    let Some(&[x0, x1, t2]) = x.first_chunk::<3>() else {
        return 0.0;
    };
    let (t0, t1): (f32, f32);
    if x0 > x1 {
        t0 = x1;
        t1 = x0;
    } else {
        t0 = x0;
        t1 = x1;
    }
    if t1 < t2 {
        t1
    } else if t0 < t2 {
        t2
    } else {
        t0
    }
}

/// Dynamic allocation analysis. `bandLogE`/`bandLogE2` are `C*nbEBands`; `offsets`, `importance`,
/// `spread_weight` are indexed by band (`offsets` is cleared for `nbEBands`). Returns `maxDepth`.
/// Dropped parameters: `oldBandE` (only used for `LM == 0`), `vbr` (always 1: CBR removed),
/// `lfe`, `surround_dynalloc` (all-zero without energy mask, folded into a `MAXG(.., 0)`),
/// `analysis`, `qext_scale`.
pub fn dynalloc_analysis(
    bandLogE: &[f32],
    bandLogE2: &[f32],
    nbEBands: i32,
    start: i32,
    end: i32,
    C: i32,
    offsets: &mut [i32],
    lsb_depth: i32,
    isTransient: i32,
    constrained_vbr: i32,
    effectiveBytes: i32,
    tot_boost_: &mut i32,
    importance: &mut [i32],
    spread_weight: &mut [i32],
    tone_freq: f32,
    toneishness: f32,
) -> f32 {
    let eBands = modes::eBands;
    let logN = modes::logN;
    let mut tot_boost: i32 = 0;
    let sg = stack::save();
    let nb = nbEBands as usize;
    let endu = end as usize;
    let follower = sg.alloc::<f32>(C as usize * nb);
    let noise_floor = sg.alloc::<f32>(C as usize * nb);
    let bandLogE3 = sg.alloc::<f32>(nb);
    offsets[..nb].fill(0);
    let noise_floor = &mut noise_floor[..endu];
    let eBands = &eBands[..endu + 1];
    /* Dynamic allocation code */
    let mut maxDepth: f32 = -31.9;
    for (i, (nf, (&n, &m))) in noise_floor
        .iter_mut()
        .zip(logN.iter().zip(eMeans.iter()))
        .enumerate()
    {
        /* Noise floor must take into account eMeans, the depth, the width of the bands
        and the preemphasis filter (approx. square of bark band ID) */
        *nf = 0.0625 * n as f32 + 0.5 + (9 - lsb_depth) as f32 - m
            + 0.0062 * (i as i32 + 5) as f32 * (i as i32 + 5) as f32;
    }
    for c in 0..C as usize {
        for (&b, &nf) in bandLogE[c * nb..c * nb + endu]
            .iter()
            .zip(noise_floor.iter())
        {
            maxDepth = maxDepth.max(b - nf);
        }
    }
    {
        /* Compute a really simple masking model to avoid taking into account completely masked
        bands when computing the spreading decision. */
        let mask = &mut sg.alloc::<f32>(nb)[..endu];
        let sig = &mut sg.alloc::<f32>(nb)[..endu];
        for ((m, &b), &nf) in mask
            .iter_mut()
            .zip(&bandLogE[..endu])
            .zip(noise_floor.iter())
        {
            *m = b - nf;
        }
        if C == 2 {
            for ((m, &b), &nf) in mask
                .iter_mut()
                .zip(&bandLogE[nb..nb + endu])
                .zip(noise_floor.iter())
            {
                *m = (*m).max(b - nf);
            }
        }
        sig.copy_from_slice(mask);
        for i in 1..endu {
            mask[i] = mask[i].max(mask[i - 1] - 2.0);
        }
        for i in (0..endu - 1).rev() {
            mask[i] = mask[i].max(mask[i + 1] - 3.0);
        }
        for ((sw, &s), &m) in spread_weight[..endu]
            .iter_mut()
            .zip(sig.iter())
            .zip(mask.iter())
        {
            /* Compute SMR: Mask is never more than 72 dB below the peak and never below the noise floor.*/
            let smr: f32 = s - (0.0f32.max(maxDepth - 12.0)).max(m);
            /* Clamp SMR to make sure we're not shifting by something negative or too large. */
            let shift: i32 = 5i32.min(0i32.max(-(floor64((0.5f32 + smr) as f64) as i32)));
            *sw = 32 >> shift;
        }
    }
    /* Make sure that dynamic allocation can't make us bust the budget.
    We enable the feature starting at 24 kb/s for 20-ms frames
    and 96 kb/s for 2.5 ms frames.  */
    if effectiveBytes >= (30 + 5 * LM) {
        let mut last: usize = 0;
        for c in 0..C as usize {
            let bandLogE3 = &mut bandLogE3[..endu];
            bandLogE3.copy_from_slice(&bandLogE2[c * nb..c * nb + endu]);
            /* (LM == 0 special case removed: LM is always 3 here) */
            let f = &mut follower[c * nb..c * nb + endu];
            f[0] = bandLogE3[0];
            for i in 1..endu {
                /* The last band to be at least 3 dB higher than the previous one
                is the last we'll consider. Otherwise, we run into problems on
                bandlimited signals. */
                if bandLogE3[i] > bandLogE3[i - 1] + 0.5 {
                    last = i;
                }
                f[i] = (f[i - 1] + 1.5).min(bandLogE3[i]);
            }
            for i in (0..last).rev() {
                f[i] = f[i].min((f[i + 1] + 2.0).min(bandLogE3[i]));
            }

            /* Combine with a median filter to avoid dynalloc triggering unnecessarily.
            The "offset" value controls how conservative we are -- a higher offset
            reduces the impact of the median filter and makes dynalloc use more bits. */
            let offset: f32 = 1.0;
            for i in 2..endu - 2 {
                f[i] = f[i].max(median_of_5(&bandLogE3[i - 2..]) - offset);
            }
            let tmp = median_of_3(bandLogE3) - offset;
            f[0] = f[0].max(tmp);
            f[1] = f[1].max(tmp);
            let tmp = median_of_3(&bandLogE3[endu - 3..]) - offset;
            f[endu - 2] = f[endu - 2].max(tmp);
            f[endu - 1] = f[endu - 1].max(tmp);

            for (v, &nf) in f.iter_mut().zip(noise_floor.iter()) {
                *v = (*v).max(nf);
            }
        }
        let start = start as usize;
        let (f0, f1) = follower.split_at_mut(nb);
        let f0 = &mut f0[..endu];
        if C == 2 {
            let f1 = &mut f1[..endu];
            for i in start..endu {
                /* Consider 24 dB "cross-talk" */
                f1[i] = f1[i].max(f0[i] - 4.0);
                f0[i] = f0[i].max(f1[i] - 4.0);
                f0[i] =
                    0.5 * (0.0f32.max(bandLogE[i] - f0[i]) + 0.0f32.max(bandLogE[nb + i] - f1[i]));
            }
        } else {
            for i in start..endu {
                f0[i] = 0.0f32.max(bandLogE[i] - f0[i]);
            }
        }
        for v in &mut f0[start..] {
            /* surround_dynalloc[i] == 0 */
            *v = (*v).max(0.0);
        }
        for (imp, &v) in importance[start..endu].iter_mut().zip(&f0[start..]) {
            *imp = floor64((0.5f32 + 13.0 * exp2(v.min(4.0))) as f64) as i32;
        }
        /* For non-transient CBR/CVBR frames, halve the dynalloc contribution */
        if constrained_vbr != 0 && isTransient == 0 {
            for v in &mut f0[start..] {
                *v = 0.5 * *v;
            }
        }
        for i in start..endu {
            if i < 8 {
                f0[i] *= 2.0;
            }
            if i >= 12 {
                f0[i] = 0.5 * f0[i];
            }
        }
        /* Compensate for Opus' under-allocation on tones. */
        if toneishness > 0.98 {
            /* `.5 + tone_freq*120/M_PI`: float product, then double */
            let freq_bin: i32 = floor64(0.5 + (tone_freq * 120.0) as f64 / PI) as i32;
            for i in start..endu {
                let (lo, hi) = (eBands[i] as i32, eBands[i + 1] as i32);
                if freq_bin >= lo && freq_bin <= hi {
                    f0[i] += 2.0;
                }
                if freq_bin >= lo - 1 && freq_bin <= hi + 1 {
                    f0[i] += 1.0;
                }
                if freq_bin >= lo - 2 && freq_bin <= hi + 2 {
                    f0[i] += 1.0;
                }
                if freq_bin >= lo - 3 && freq_bin <= hi + 3 {
                    f0[i] += 0.5;
                }
            }
            if freq_bin >= eBands[endu] as i32 {
                f0[endu - 1] += 2.0;
                f0[endu - 2] += 1.0;
            }
        }
        /* (analysis->valid is always 0: no leak_boost) */
        for i in start..endu {
            let width: i32;
            let boost: i32;
            let boost_bits: i32;

            f0[i] = f0[i].min(4.0);

            /* (SHR32 by 8 is the identity in the float build) */
            width = C * (eBands[i + 1] as i32 - eBands[i] as i32) << LM;
            if width < 6 {
                boost = f0[i] as i32;
                boost_bits = boost * width << BITRES;
            } else if width > 48 {
                boost = (f0[i] * 8.0) as i32;
                boost_bits = (boost * width << BITRES) / 8;
            } else {
                boost = (f0[i] * width as f32 / 6.0) as i32;
                boost_bits = boost * 6 << BITRES;
            }
            /* For CBR and non-transient CVBR frames, limit dynalloc to 2/3 of the bits */
            if (constrained_vbr != 0 && isTransient == 0)
                && (tot_boost + boost_bits) >> BITRES >> 3 > 2 * effectiveBytes / 3
            {
                let cap: i32 = (2 * effectiveBytes / 3) << BITRES << 3;
                offsets[i] = cap - tot_boost;
                tot_boost = cap;
                break;
            } else {
                offsets[i] = boost;
                tot_boost += boost_bits;
            }
        }
    } else {
        importance[start as usize..end as usize].fill(13);
    }
    *tot_boost_ = tot_boost;
    maxDepth
}

/// Compute the LPC coefficients using a least-squares fit for both forward and backward
/// prediction. `x[0..len]`; returns 1 on failure (`lpc` untouched), else 0.
pub fn tone_lpc(x: &[f32], len: i32, delay: i32, lpc: &mut [f32; 2]) -> i32 {
    let mut r00: f32 = 0.0;
    let mut r01: f32 = 0.0;
    let mut r11: f32;
    let mut r02: f32 = 0.0;
    let r12: f32;
    let r22: f32;
    let mut edges: f32;
    debug_assert!(len > 2 * delay);
    let (len, delay) = (len as usize, delay as usize);
    /* Compute correlations as if using the forward prediction covariance method. */
    let x = &x[..len];
    let n = len - 2 * delay;
    for ((&a, &b), &c) in x[..n].iter().zip(&x[delay..delay + n]).zip(&x[2 * delay..]) {
        r00 += a * a;
        r01 += a * b;
        r02 += a * c;
    }
    let head0 = &x[..delay];
    let head1 = &x[delay..2 * delay];
    let tail0 = &x[len - 2 * delay..len - delay];
    let tail1 = &x[len - delay..];
    edges = 0.0;
    for (&t, &h) in tail0.iter().zip(head0) {
        edges += t * t - h * h;
    }
    r11 = r00 + edges;
    edges = 0.0;
    for (&t, &h) in tail1.iter().zip(head1) {
        edges += t * t - h * h;
    }
    r22 = r11 + edges;
    edges = 0.0;
    for ((&t0, &t1), (&h0, &h1)) in tail0.iter().zip(tail1).zip(head0.iter().zip(head1)) {
        edges += t0 * t1 - h0 * h1;
    }
    r12 = r01 + edges;
    /* Reverse and sum to get the backward contribution. */
    {
        let R00 = r00 + r22;
        let R01 = r01 + r12;
        let R11 = 2.0 * r11;
        let R02 = 2.0 * r02;
        let R12 = r12 + r01;
        r00 = R00;
        r01 = R01;
        r11 = R11;
        r02 = R02;
        /* R22 = r00 + r22 is never used afterwards */
        let r12 = R12;
        /* Solve A*x=b, where A=[r00, r01; r01, r11] and b=[r02; r12]. */
        let den: f32 = r00 * r11 - r01 * r01;
        if den < 0.001 * (r00 * r11) {
            return 1;
        }
        let num1: f32 = r02 * r11 - r01 * r12;
        if num1 >= den {
            lpc[1] = 1.0;
        } else if num1 <= -den {
            lpc[1] = -1.0;
        } else {
            lpc[1] = num1 / den;
        }
        let num0: f32 = r00 * r12 - r02 * r01;
        if 0.5 * num0 >= den {
            lpc[0] = 1.999999;
        } else if 0.5 * num0 <= -den {
            lpc[0] = -1.999999;
        } else {
            lpc[0] = num0 / den;
        }
    }
    0
}

/// Detects pure of nearly pure tones so we can prevent them from causing problems with the
/// encoder. `in_` holds `CC` channels of `N` samples (`in_[c*N + i]`, the caller passes `N+overlap`).
/// Returns the tone frequency (radians/sample, or -1) and writes `*toneishness`.
pub fn tone_detect(in_: &[f32], CC: i32, N: i32, toneishness: &mut f32) -> f32 {
    let mut delay: i32 = 1;
    let mut lpc = [0.0f32; 2];
    let freq: f32;
    let sg = stack::save();
    let x = sg.alloc::<f32>(N as usize);
    let nu = N as usize;
    /* Shift by SIG_SHIFT+2 (+3 for stereo) to account for HF gain of the preemphasis filter
    (all shifts are the identity in the float build). */
    debug_assert!(in_.len() >= CC as usize * nu);
    if CC == 2 {
        for ((o, &a), &b) in x.iter_mut().zip(in_).zip(in_.iter().skip(nu)) {
            *o = a + b;
        }
    } else {
        for (o, &a) in x.iter_mut().zip(in_) {
            *o = a;
        }
    }
    let mut fail = tone_lpc(x, N, delay, &mut lpc);
    /* If our LPC filter resonates too close to DC, retry the analysis with down-sampling. */
    while delay <= 16 && (fail != 0 || (lpc[0] > 1.0 && lpc[1] < 0.0)) {
        delay *= 2;
        fail = tone_lpc(x, N, delay, &mut lpc);
    }
    /* Check that our filter has complex roots. (3.999999 is a double literal: double arithmetic.) */
    if fail == 0 && (lpc[0] * lpc[0]) as f64 + 3.999999 * (lpc[1] as f64) < 0.0 {
        /* Squared radius of the poles. */
        *toneishness = -lpc[1];
        freq = (acos64((0.5f32 * lpc[0]) as f64) / delay as f64) as f32;
    } else {
        freq = -1.0;
        *toneishness = 0.0;
    }
    freq
}

/// Pitch pre-filter. `in_` is the `in` buffer of celt_encode_with_ec: `CC*(N+overlap)` samples,
/// channel `c` at `c*(N+overlap)`, the `N` new (pre-emphasized) samples at `+overlap`; on return
/// the first `overlap` samples of each channel hold the old `in_mem` and the rest the filtered
/// signal. The C `prefilter_mem` argument is `st->in_mem + CC*overlap`, which is
/// `st.prefilter_mem` here (updated in place). Returns `pf_on`.
/// Dropped parameters: `analysis`, `qext_scale`; `st->arch`.
pub fn run_prefilter(
    st: &mut CELTEncoder,
    in_: &mut [f32],
    CC: i32,
    N: i32,
    prefilter_tapset: i32,
    pitch: &mut i32,
    gain: &mut f32,
    qgain: &mut i32,
    enabled: i32,
    complexity: i32,
    tf_estimate: f32,
    nbAvailableBytes: i32,
    mut tone_freq: f32,
    toneishness: f32,
) -> i32 {
    let sg = stack::save();
    let overlap = modes::overlap;
    let max_period = COMBFILTER_MAXPERIOD;
    let min_period = COMBFILTER_MINPERIOD;
    let mut before = [0.0f32; 2];
    let mut after = [0.0f32; 2];
    let mut cancel_pitch = false;
    let pitch_index: i32;
    let mut gain1: f32;
    let pf_on: i32;
    let qg: i32;

    /* N is 960 (the only frame size); CC is 1 or 2 */
    const NU: usize = 960;
    const OVU: usize = modes::overlap as usize;
    const MP: usize = COMBFILTER_MAXPERIOD as usize;
    debug_assert!(N as usize == NU && (CC == 1 || CC == 2));

    /* If we detect that the signal is dominated by a single tone, don't rely on the standard pitch
    estimator, as it can become unreliable. */
    if enabled != 0 && toneishness > 0.99 {
        let mut multiple: i32 = 1;
        /* Using aliased version of the postfilter above 24 kHz.
        First value is purposely slightly above pi to avoid triggering for Fs=48kHz. */
        if tone_freq >= 3.1416 {
            tone_freq = 3.141593 - tone_freq;
        }
        /* If the pitch is too high for our post-filter, apply pitch doubling until
        we can get something that fits (not ideal, but better than nothing). */
        while tone_freq >= multiple as f32 * 0.39 {
            multiple += 1;
        }
        if tone_freq > 0.006148 {
            /* `.5+2.f*M_PI*multiple/tone_freq` is evaluated in double */
            pitch_index = (floor64(0.5 + 2.0 * PI * multiple as f64 / tone_freq as f64) as i32)
                .min(COMBFILTER_MAXPERIOD - 2);
        } else {
            /* If the pitch is too low, using a very high pitch will actually give us an improvement
            due to the DC component of the filter that will be close to our tone. Again, not ideal,
            but if we only have a single tone, it's better than nothing. */
            pitch_index = COMBFILTER_MINPERIOD;
        }
        gain1 = 0.75;
    } else if enabled != 0 && complexity >= 5 {
        /* freed at the end of this block, before `_pre` is allocated */
        let sga = stack::save();
        let pitch_buf = sga.alloc::<f32>(((max_period + N) >> 1) as usize);
        let mut pi: i32 = 0;

        /* the pitch analysis reads the `_pre` of C ([prefilter_mem | new samples] per channel) in place */
        pitch_downsample_factor2(
            &st.prefilter_mem,
            in_,
            OVU,
            NU + OVU,
            pitch_buf,
            (max_period + N) >> 1,
            CC,
        );
        /* Don't search for the fir last 1.5 octave of the range because
        there's too many false-positives due to short-term correlation */
        pitch_search(
            &pitch_buf[MP >> 1..],
            pitch_buf,
            N,
            max_period - 3 * min_period,
            &mut pi,
        );
        pi = max_period - pi;

        gain1 = remove_doubling(
            pitch_buf,
            max_period,
            min_period,
            N,
            &mut pi,
            st.prefilter_period,
            st.prefilter_gain,
        );
        if pi > max_period - 2 {
            pi = max_period - 2;
        }
        pitch_index = pi;
        gain1 = 0.7 * gain1;
        if st.loss_rate > 2 {
            gain1 = 0.5 * gain1;
        }
        if st.loss_rate > 4 {
            gain1 = 0.5 * gain1;
        }
        if st.loss_rate > 8 {
            gain1 = 0.0;
        }
    } else {
        gain1 = 0.0;
        pitch_index = COMBFILTER_MINPERIOD;
    }
    /* (analysis->valid is always 0) */

    /* Gain threshold for enabling the prefilter/postfilter */
    let mut pf_threshold: f32 = 0.2;

    /* Adjusting the threshold based on rate and continuity */
    if (pitch_index - st.prefilter_period).abs() * 10 > pitch_index {
        pf_threshold += 0.2;
        /* Completely disable the prefilter on strong transients without continuity. */
        if tf_estimate > 0.98 {
            gain1 = 0.0;
        }
    }
    if nbAvailableBytes < 25 {
        pf_threshold += 0.1;
    }
    if nbAvailableBytes < 35 {
        pf_threshold += 0.1;
    }
    if st.prefilter_gain > 0.4 {
        pf_threshold -= 0.1;
    }
    if st.prefilter_gain > 0.55 {
        pf_threshold -= 0.1;
    }

    /* Hard threshold at 0.2 */
    pf_threshold = pf_threshold.max(0.2);
    if gain1 < pf_threshold {
        gain1 = 0.0;
        pf_on = 0;
        qg = 0;
    } else {
        /*This block is not gated by a total bits check only because
        of the nbAvailableBytes check above.*/
        if (gain1 - st.prefilter_gain).abs() < 0.1 {
            gain1 = st.prefilter_gain;
        }

        let q = floor64((0.5f32 + gain1 * 32.0 / 3.0) as f64) as i32 - 1;
        qg = 0i32.max(7i32.min(q));
        gain1 = 0.09375 * (qg + 1) as f32;
        pf_on = 1;
    }

    /* _pre: CC*(N+max_period); pre[c] = _pre + c*(N+max_period); allocated after the pitch
    analysis (which does not need it) to keep the pseudostack peak low */
    let _pre = sg.alloc::<f32>(CC as usize * (NU + MP));

    /* Per-channel views of fixed size: in_[c] = [in_mem copy (overlap) | N samples],
    pre[c] = [prefilter_mem (max_period) | N samples]. */
    let ccu = if CC == 2 { 2 } else { 1 };
    let in_c = &mut in_.as_chunks_mut::<{ NU + OVU }>().0[..ccu];
    let pre_c = &mut _pre.as_chunks_mut::<{ NU + MP }>().0[..ccu];
    let pm_c = st.prefilter_mem.as_chunks_mut::<MP>().0;
    let im_c = st.in_mem.as_chunks_mut::<OVU>().0;

    for c in 0..ccu {
        pre_c[c][..MP].copy_from_slice(&pm_c[c]);
        pre_c[c][MP..].copy_from_slice(&in_c[c][OVU..]);
    }

    /* offset = shortMdctSize - overlap = 0: the first (constant-filter) comb_filter of
    the C code has length 0 */
    st.prefilter_period = st.prefilter_period.max(COMBFILTER_MINPERIOD);
    let abs_sum = |x: &[f32]| x.iter().fold(0.0f32, |s, &v| s + v.abs());
    for c in 0..ccu {
        in_c[c][..OVU].copy_from_slice(&im_c[c]);
        before[c] = abs_sum(&in_c[c][OVU..]);
        unsafe {
            comb_filter(
                in_c[c].as_mut_ptr().add(OVU),
                pre_c[c].as_mut_ptr().add(MP),
                st.prefilter_period,
                pitch_index,
                N,
                -st.prefilter_gain,
                -gain1,
                st.prefilter_tapset,
                prefilter_tapset,
            );
        }
        after[c] = abs_sum(&in_c[c][OVU..]);
    }

    if CC == 2 {
        let thresh0 = (0.25f32 * gain1) * before[0] + 0.01f32 * before[1];
        let thresh1 = (0.25f32 * gain1) * before[1] + 0.01f32 * before[0];
        /* Don't use the filter if one channel gets significantly worse. */
        if after[0] - before[0] > thresh0 || after[1] - before[1] > thresh1 {
            cancel_pitch = true;
        }
        /* Use the filter only if at least one channel gets significantly better. */
        if before[0] - after[0] < thresh0 && before[1] - after[1] < thresh1 {
            cancel_pitch = true;
        }
    } else {
        /* Check that the mono channel actually got better. */
        if after[0] > before[0] {
            cancel_pitch = true;
        }
    }
    /* If needed, revert to a gain of zero. */
    if cancel_pitch {
        for c in 0..ccu {
            in_c[c][OVU..].copy_from_slice(&pre_c[c][MP..]);
            unsafe {
                comb_filter(
                    in_c[c].as_mut_ptr().add(OVU),
                    pre_c[c].as_mut_ptr().add(MP),
                    st.prefilter_period,
                    pitch_index,
                    overlap,
                    -st.prefilter_gain,
                    0.0,
                    st.prefilter_tapset,
                    prefilter_tapset,
                );
            }
        }
        gain1 = 0.0;
    }
    let (pf_on, qg) = if cancel_pitch { (0, 0) } else { (pf_on, qg) };

    for c in 0..ccu {
        im_c[c].copy_from_slice(&in_c[c][NU..]);
        /* N < max_period: shift the memory and append the N new samples */
        pm_c[c].copy_within(NU.., 0);
        pm_c[c][MP - NU..].copy_from_slice(&pre_c[c][MP..]);
    }

    *gain = gain1;
    *pitch = pitch_index;
    *qgain = qg;
    pf_on
}

/// Target rate in 1/8 bits for the frame. Dropped parameters: `analysis` (valid is always 0, so
/// no activity/tonality terms), `pitch_change` (only used with analysis), `lfe`,
/// `has_surround_mask`, `surround_masking` (all 0), `enable_qext`. The CBR second call
/// (`!vbr_rate`, hybrid) does not exist.
pub fn compute_vbr(
    base_target: i32,
    bitrate: i32,
    lastCodedBands: i32,
    C: i32,
    intensity: i32,
    constrained_vbr: i32,
    mut stereo_saving: f32,
    tot_boost: i32,
    tf_estimate: f32,
    maxDepth: f32,
    temporal_vbr: f32,
) -> i32 {
    /* The target rate in 8th bits per frame */
    let mut target: i32;
    let nbEBands = modes::nbEBands;
    let eBands = modes::eBands;

    let coded_bands = if lastCodedBands != 0 {
        lastCodedBands
    } else {
        nbEBands
    };
    /* the band counts are at most nbEBands */
    debug_assert!(coded_bands <= nbEBands && intensity <= nbEBands);
    let coded_stereo_bands = intensity.min(coded_bands);
    /* (clamped to the table so that no check is needed; the counts never exceed it) */
    let edge = |i: i32| (eBands[(i as usize).min(nbEBands as usize)] as i32) << LM;
    let mut coded_bins: i32 = edge(coded_bands);
    if C == 2 {
        coded_bins += edge(coded_stereo_bands);
    }

    target = base_target;

    /* (analysis->valid is always 0: no activity reduction) */
    /* Stereo savings */
    if C == 2 {
        let coded_stereo_dof: i32 = edge(coded_stereo_bands) - coded_stereo_bands;
        /* Maximum fraction of the bits we can save if the signal is mono. */
        let max_frac: f32 = (0.8f32 * coded_stereo_dof as f32) / coded_bins as f32;
        stereo_saving = stereo_saving.min(1.0);
        target -= (max_frac * target as f32)
            .min((stereo_saving - 0.1) * ((coded_stereo_dof << BITRES) as f32))
            as i32;
    }
    /* Boost the rate according to dynalloc (minus the dynalloc average for calibration). */
    target += tot_boost - (19 << LM);
    /* Apply transient boost, compensating for average boost. */
    let tf_calibration: f32 = 0.044;
    target += ((tf_estimate - tf_calibration) * target as f32) as i32;

    /* (analysis->valid and surround masks are always off) */

    {
        let bins: i32 = edge(nbEBands - 2);
        let mut floor_depth: i32 = (((C * bins) << BITRES) as f32 * maxDepth) as i32;
        floor_depth = floor_depth.max(target >> 2);
        target = target.min(floor_depth);
    }

    /* Make VBR less aggressive for constrained VBR because we can't keep a higher bitrate
    for long. Needs tuning. */
    if constrained_vbr != 0 {
        target = base_target + (0.67f32 * (target - base_target) as f32) as i32;
    }

    if tf_estimate < 0.2 {
        let amount: f32 = 0.0000031f32 * 0i32.max(32000i32.min(96000 - bitrate)) as f32;
        let tvbr_factor: f32 = temporal_vbr * amount;
        target += (tvbr_factor * target as f32) as i32;
    }

    /* Don't allow more than doubling the rate */
    (2 * base_target).min(target)
}

/// Encodes one frame (celt_encoder.c).
///
/// Contract: `buf` holds the 960 interleaved samples of each of the `st.channels` channels
/// (float scale +-1 i.e. opus_res) at its start and has room for `st.channels * (960 + overlap)`
/// samples: the caller's static buffer doubles as the `in` buffer of the C code (pre-emphasized
/// signal, `overlap` + 960 samples per channel) and, after the band normalisation, as the
/// normalised spectrum `X`; the PCM is consumed (copied to the pseudostack) first. `enc` is the range coder
/// over the packet bytes (up to 1275: `nbCompressedBytes`), freshly initialised and shrunk by
/// the Opus layer (`ec_tell(enc) == 1`, so `nbFilledBytes == 0`, and `compressed` is not needed).
/// Returns the number of bytes written (`nbCompressedBytes` after ec_enc_shrink in VBR) or a
/// negative Opus error code; on return `st.rng` holds the final range.
///
/// Fixed configuration (see PORTING.md): `upsample == 1`, `LM == 3`, VBR only (`vbr_rate > 0`;
/// the CBR branches are removed), no analysis, no QEXT, no RESYNTH, no LFE, no energy mask
/// (surround), no `signalling`, `start == 0` (so `hybrid == 0` and the hybrid-only code is
/// removed). `pitch_change` is not computed (it only feeds the analysis terms of compute_vbr).
pub fn celt_encode_with_ec(st: &mut CELTEncoder, buf: &mut [f32], enc: &mut ec_enc) -> i32 {
    let mut bits: i32;
    let mut shortBlocks: i32;
    let mut isTransient: i32;
    let CC = st.channels;
    let C = st.stream_channels;
    let mut nbAvailableBytes: i32;
    let mut effectiveBytes: i32;
    let mut pitch_index: i32 = COMBFILTER_MINPERIOD;
    let mut gain1: f32 = 0.0;
    let mut dual_stereo: i32 = 0;
    let vbr_rate: i32;
    let mut total_bits: i32;
    let mut tell: i32;
    let prefilter_tapset: i32;
    let pf_on: i32;
    let mut anti_collapse_on: i32 = 0;
    let silence: i32;
    let mut tf_chan: i32 = 0;
    let mut tf_estimate: f32;
    let mut tot_boost: i32 = 0;
    let mut transient_got_disabled: i32 = 0;
    let mut temporal_vbr: f32;
    let mut toneishness: f32 = 0.0;
    let sg = stack::save();

    let nbEBands = modes::nbEBands;
    let nb = nbEBands as usize;
    let overlap = modes::overlap;
    let eBands = modes::eBands;
    /* start == 0, hybrid == 0 */
    let start = 0;
    let end = st.end;
    debug_assert!(ec_tell(enc) == 1);
    tf_estimate = 0.0;

    /* M = 1 << LM short blocks of shortMdctSize samples */
    let M: i32 = 1 << LM;
    let N: i32 = 960;
    let (Nu, ovu) = (N as usize, overlap as usize);
    let mp = COMBFILTER_MAXPERIOD as usize;
    /* The channel counts and `end` as indices: CC and C are 1 or 2 with C <= CC, end <= nbEBands (the
    clamps only make the bounds visible to the compiler). */
    debug_assert!((CC == 1 || CC == 2) && (C == 1 || C == CC) && end as usize <= nb);
    let ccu: usize = if CC == 2 { 2 } else { 1 };
    let cu: usize = if C == 2 { ccu } else { 1 };
    let endu = (end as usize).min(nb);
    let buf = &mut buf[..ccu * (Nu + ovu)];

    /* prefilter_mem, oldBandE, oldLogE, oldLogE2, energyError are the st fields of those names. */

    tell = 1;
    let nbFilledBytes: i32 = 0;
    /* Can't produce more than 1275 output bytes for the main payload. */
    let mut nbCompressedBytes: i32 = 1275;

    /* VBR only: the CBR branch (vbr_rate = 0) is removed. */
    debug_assert!(st.bitrate > 0);
    {
        /* bitrate_to_bits(bitrate, 48000, 960) */
        vbr_rate = (st.bitrate / 50) << BITRES;
        effectiveBytes = vbr_rate >> (3 + BITRES);
    }
    debug_assert!(vbr_rate > 0);
    nbAvailableBytes = nbCompressedBytes - nbFilledBytes;
    let mut equiv_rate: i32 =
        ((nbCompressedBytes * 8 * 50) << (3 - LM)) - (40 * C + 20) * ((400 >> LM) - 50);
    equiv_rate = equiv_rate.min(st.bitrate - (40 * C + 20) * ((400 >> LM) - 50));

    /* Computes the max bit-rate allowed in VBR mode to avoid violating the
    target rate and buffering.
    We must do this up front so that bust-prevention logic triggers
    correctly if we don't have enough bits. */
    if st.constrained_vbr != 0 {
        /* We could use any multiple of vbr_rate as bound (depending on the
        delay).
        This is clamped to ensure we use at least two bytes if the encoder
        was entirely empty, but to allow 0 in hybrid mode. */
        let vbr_bound: i32 = vbr_rate;
        let max_allowed: i32 = (2i32
            .max((vbr_rate + vbr_bound - st.vbr_reservoir) >> (BITRES + 3)))
        .min(nbAvailableBytes);
        if max_allowed < nbAvailableBytes {
            nbCompressedBytes = nbFilledBytes + max_allowed;
            nbAvailableBytes = max_allowed;
            ec_enc_shrink(enc, nbCompressedBytes as u32);
        }
    }
    total_bits = nbCompressedBytes * 8;

    /* The encoder-wide buffers that outlive the MDCT phase are allocated first: the PCM copy and the
    MDCT output below are scoped, so that the pseudostack peak stays low. */
    let bandE = sg.alloc::<f32>(nb * ccu);
    let bandLogE = sg.alloc::<f32>(nb * ccu);
    let bandLogE2 = sg.alloc::<f32>(cu * nb);

    {
        let sgp = stack::save();
        let pcm = sgp.alloc::<f32>(ccu * Nu);
        pcm.copy_from_slice(&buf[..ccu * Nu]);
        let in_ = &mut *buf;

        let mut sample_max: f32 = st.overlap_max.max(maxabs(pcm, C * (N - overlap)));
        st.overlap_max = maxabs(&pcm[cu * (Nu - ovu)..], C * overlap);
        sample_max = sample_max.max(st.overlap_max);
        silence = (sample_max <= 1.0f32 / (1i32 << st.lsb_depth) as f32) as i32;
        /* tell == 1 */
        ec_enc_bit_logp(enc, silence, 15);
        if silence != 0 {
            /*In VBR mode there is no need to send more than the minimum. */
            {
                nbCompressedBytes = nbCompressedBytes.min(nbFilledBytes + 2);
                effectiveBytes = nbCompressedBytes;
                total_bits = nbCompressedBytes * 8;
                nbAvailableBytes = 2;
                ec_enc_shrink(enc, nbCompressedBytes as u32);
            }
            /* Pretend we've filled all the remaining bits with zeros
            (that's what the initialiser did anyway) */
            tell = nbCompressedBytes * 8;
            enc.nbits_total += tell - ec_tell(enc);
        }
        let need_clip = sample_max > 65536.0;
        /* channel c: [the last `overlap` samples of prefilter_mem[c] | the pre-emphasized input] */
        let pm = st
            .prefilter_mem
            .as_chunks::<{ COMBFILTER_MAXPERIOD as usize }>()
            .0;
        for (c, (x, mem)) in in_
            .as_chunks_mut::<{ 960 + modes::overlap as usize }>()
            .0
            .iter_mut()
            .zip(&mut st.preemph_memE)
            .enumerate()
        {
            celt_preemphasis(pcm, c, &mut x[ovu..], CC, mem, need_clip);
            x[..ovu].copy_from_slice(&pm[c & 1][mp - ovu..]);
        }
    }
    let in_ = &mut *buf;

    let tone_freq: f32 = tone_detect(in_, CC, N + overlap, &mut toneishness);
    isTransient = 0;
    shortBlocks = 0;
    if st.complexity >= 1 {
        isTransient = transient_analysis(
            in_,
            CC,
            &mut tf_estimate,
            &mut tf_chan,
            tone_freq,
            toneishness,
        );
    }
    toneishness = toneishness.min(1.0f32 - tf_estimate);
    /* Find pitch period and gain */
    {
        let mut qg: i32 = 0;
        let enabled: i32 =
            (nbAvailableBytes > 12 * C && silence == 0 && tell + 16 <= total_bits) as i32;

        prefilter_tapset = st.tapset_decision;
        let complexity = st.complexity;
        pf_on = run_prefilter(
            st,
            in_,
            CC,
            N,
            prefilter_tapset,
            &mut pitch_index,
            &mut gain1,
            &mut qg,
            enabled,
            complexity,
            tf_estimate,
            nbAvailableBytes,
            tone_freq,
            toneishness,
        );
        /* (pitch_change is only used by the analysis terms of compute_vbr) */
        if pf_on == 0 {
            if tell + 16 <= total_bits {
                ec_enc_bit_logp(enc, 0, 1);
            }
        } else {
            /*This block is not gated by a total bits check only because
            of the nbAvailableBytes check above.*/
            ec_enc_bit_logp(enc, 1, 1);
            pitch_index += 1;
            let octave: i32 = EC_ILOG(pitch_index as u32) - 5;
            ec_enc_uint(enc, octave as u32, 6);
            ec_enc_bits(
                enc,
                (pitch_index - (16 << octave)) as u32,
                (4 + octave) as u32,
            );
            pitch_index -= 1;
            ec_enc_bits(enc, qg as u32, 3);
            ec_enc_icdf(enc, prefilter_tapset, &tapset_icdf, 2);
        }
    }
    if LM > 0 && ec_tell(enc) + 3 <= total_bits {
        if isTransient != 0 {
            shortBlocks = M;
        }
    } else {
        isTransient = 0;
        transient_got_disabled = 1;
    }

    let sgf = stack::save();
    let freq = sgf.alloc::<f32>((CC * N) as usize); /* Interleaved signal MDCTs */

    let secondMdct = shortBlocks != 0 && st.complexity >= 8;
    if secondMdct {
        compute_mdcts(0, in_, freq, C, CC);
        compute_band_energies(freq, bandE, end, C);
        amp2Log2(end, bandE, bandLogE2, C);
        for row in bandLogE2.as_chunks_mut::<{ modes::nbEBands as usize }>().0 {
            for v in &mut row[..endu] {
                *v += 0.5f32 * LM as f32;
            }
        }
    }

    compute_mdcts(shortBlocks, in_, freq, C, CC);
    /* This should catch any NaN in the CELT input. Since we're not supposed to see any (they're filtered
    at the Opus layer), just abort. */
    debug_assert!(!freq[0].is_nan() && (C == 1 || !freq[Nu].is_nan()));
    if CC == 2 && C == 1 {
        tf_chan = 0;
    }
    compute_band_energies(freq, bandE, end, C);

    /* (LFE band limiting removed) */
    amp2Log2(end, bandE, bandLogE, C);

    /* (surround_dynalloc / surround masking removed: no energy mask) */
    /* Temporal VBR (but not for LFE) */
    {
        let mut follow: f32 = -10.0;
        let mut frame_avg: f32 = 0.0;
        let offset: f32 = if shortBlocks != 0 {
            0.5f32 * LM as f32
        } else {
            0.0
        };
        /* start == 0 */
        let (b0, b1) = bandLogE.split_at(nb);
        let b1 = if C == 2 { b1 } else { b0 };
        for (&x0, &x1) in b0[..endu].iter().zip(b1) {
            follow = (follow - 1.0).max(x0 - offset);
            if C == 2 {
                follow = follow.max(x1 - offset);
            }
            frame_avg += follow;
        }
        frame_avg /= (end - start) as f32;
        temporal_vbr = frame_avg - st.spec_avg;
        temporal_vbr = 3.0f32.min((-1.5f32).max(temporal_vbr));
        st.spec_avg += 0.02f32 * temporal_vbr;
    }

    if !secondMdct {
        bandLogE2.copy_from_slice(&bandLogE[..cu * nb]);
    }

    /* Last chance to catch any transient we might have missed in the
    time-domain analysis */
    if LM > 0 && ec_tell(enc) + 3 <= total_bits && isTransient == 0 && st.complexity >= 5 {
        if patch_transient_decision(bandLogE, &st.oldBandE, nbEBands, start, end, C) != 0 {
            isTransient = 1;
            shortBlocks = M;
            compute_mdcts(shortBlocks, in_, freq, C, CC);
            compute_band_energies(freq, bandE, end, C);
            amp2Log2(end, bandE, bandLogE, C);
            /* Compensate for the scaling of short vs long mdcts */
            for row in bandLogE2.as_chunks_mut::<{ modes::nbEBands as usize }>().0 {
                for v in &mut row[..endu] {
                    *v += 0.5f32 * LM as f32;
                }
            }
            tf_estimate = 0.2;
        }
    }

    if LM > 0 && ec_tell(enc) + 3 <= total_bits {
        ec_enc_bit_logp(enc, isTransient, 3);
    }

    /* Band normalisation: the normalised spectrum X goes to the caller's buffer (`in_` is not
    needed anymore), which frees the MDCT output */
    let X = &mut buf[..cu * Nu];
    normalise_bands(freq, X, bandE, end, C);
    drop(sgf);
    let X = &mut buf[..cu * Nu];

    let enable_tf_analysis =
        effectiveBytes >= 15 * C && st.complexity >= 2 && toneishness < 0.98f32;

    let offsets = sg.alloc::<i32>(nb);
    let importance = sg.alloc::<i32>(nb);
    let spread_weight = sg.alloc::<i32>(nb);

    let maxDepth: f32 = dynalloc_analysis(
        bandLogE,
        bandLogE2,
        nbEBands,
        start,
        end,
        C,
        offsets,
        st.lsb_depth,
        isTransient,
        st.constrained_vbr,
        effectiveBytes,
        &mut tot_boost,
        importance,
        spread_weight,
        tone_freq,
        toneishness,
    );

    let tf_res = sg.alloc::<i32>(nb);
    let tf_select: i32;
    /* Disable variable tf resolution for hybrid and at very low bitrate */
    if enable_tf_analysis {
        let lambda: i32 = 80i32.max(20480 / effectiveBytes + 2);
        tf_select = tf_analysis(
            end,
            isTransient,
            tf_res,
            lambda,
            X,
            N,
            tf_estimate,
            tf_chan,
            importance,
        );
    } else {
        /* (the two hybrid cases are removed) */
        tf_res[..endu].fill(isTransient);
        tf_select = 0;
    }

    let error = sg.alloc::<f32>(cu * nb);
    const NB: usize = modes::nbEBands as usize;
    for ((b, o), e) in bandLogE.as_chunks_mut::<NB>().0[..cu]
        .iter_mut()
        .zip(st.oldBandE.as_chunks::<NB>().0)
        .zip(st.energyError.as_chunks::<NB>().0)
    {
        /* (start == 0) */
        for ((b, &o), &e) in b[..endu].iter_mut().zip(o).zip(e) {
            /* When the energy is stable, slightly bias energy quantization towards
            the previous error to make the gain more stable (a constant offset is
            better than fluctuations). */
            if (*b - o).abs() < 2.0 {
                *b -= 0.25f32 * e;
            }
        }
    }
    quant_coarse_energy(
        end,
        bandLogE,
        &mut st.oldBandE,
        total_bits as u32,
        error,
        enc,
        C,
        nbAvailableBytes,
        &mut st.delayedIntra,
        (st.complexity >= 4) as i32,
        st.loss_rate,
    );

    tf_encode(start, end, isTransient, tf_res, tf_select, enc);

    if ec_tell(enc) + 4 <= total_bits {
        /* (LFE and hybrid cases removed) */
        if shortBlocks != 0 || st.complexity < 3 || nbAvailableBytes < 10 * C {
            if st.complexity == 0 {
                st.spread_decision = SPREAD_NONE;
            } else {
                st.spread_decision = SPREAD_NORMAL;
            }
        } else {
            /* Disable new spreading+tapset estimator until we can show it works
            better than the old one. So far it seems like spreading_decision()
            works best. */
            st.spread_decision = spreading_decision(
                X,
                &mut st.tonal_average,
                st.spread_decision,
                &mut st.hf_average,
                &mut st.tapset_decision,
                (pf_on != 0 && shortBlocks == 0) as i32,
                end,
                C,
                spread_weight,
            );
        }
        ec_enc_icdf(enc, st.spread_decision, &spread_icdf, 5);
    } else {
        st.spread_decision = SPREAD_NORMAL;
    }

    let cap = sg.alloc::<i32>(nb);
    init_caps(cap, C);

    let mut dynalloc_logp: i32 = 6;
    total_bits <<= BITRES;
    let mut total_boost: i32 = 0;
    tell = ec_tell_frac(enc) as i32;
    for i in start as usize..endu {
        let width: i32 = C * (eBands[i + 1] as i32 - eBands[i] as i32) << LM;
        /* quanta is 6 bits, but no more than 1 bit/sample
        and no less than 1/8 bit/sample */
        let quanta: i32 = (width << BITRES).min((6 << BITRES).max(width));
        let mut dynalloc_loop_logp: i32 = dynalloc_logp;
        let mut boost: i32 = 0;
        let mut j: i32 = 0;
        while tell + (dynalloc_loop_logp << BITRES) < total_bits - total_boost && boost < cap[i] {
            let flag: i32 = (j < offsets[i]) as i32;
            ec_enc_bit_logp(enc, flag, dynalloc_loop_logp as u32);
            tell = ec_tell_frac(enc) as i32;
            if flag == 0 {
                break;
            }
            boost += quanta;
            total_boost += quanta;
            dynalloc_loop_logp = 1;
            j += 1;
        }
        /* Making dynalloc more likely */
        if j != 0 {
            dynalloc_logp = 2i32.max(dynalloc_logp - 1);
        }
        offsets[i] = boost;
    }

    if C == 2 {
        static intensity_thresholds: [f32; 21] =
            /* 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19  20  off*/
            [
                1., 2., 3., 4., 5., 6., 7., 8., 16., 24., 36., 44., 50., 56., 62., 67., 72., 79.,
                88., 106., 134.,
            ];
        static intensity_histeresis: [f32; 21] = [
            1., 1., 1., 1., 1., 1., 1., 2., 2., 2., 2., 2., 2., 2., 3., 3., 4., 5., 6., 8., 8.,
        ];

        /* Always use MS for 2.5 ms frames until we can do a better analysis */
        if LM != 0 {
            dual_stereo = stereo_analysis(X, N);
        }

        st.intensity = hysteresis_decision(
            (equiv_rate / 1000) as f32,
            &intensity_thresholds,
            &intensity_histeresis,
            st.intensity,
        );
        st.intensity = end.min(start.max(st.intensity));
    }

    let mut alloc_trim: i32 = 5;
    if tell + (6 << BITRES) <= total_bits - total_boost {
        /* (start > 0 and LFE are impossible here) */
        alloc_trim = alloc_trim_analysis(
            X,
            bandLogE,
            end,
            C,
            N,
            &mut st.stereo_saving,
            tf_estimate,
            st.intensity,
            equiv_rate,
        );
        ec_enc_icdf(enc, alloc_trim, &trim_icdf, 7);
        tell = ec_tell_frac(enc) as i32;
    }

    /* In VBR mode the frame size must not be reduced so much that it would
    result in the encoder running out of bits.
    The margin of 2 bytes ensures that none of the bust-prevention logic
    in the decoder will have triggered so far. */
    let min_allowed: i32 = ((tell + total_boost + (1 << (BITRES + 3)) - 1) >> (BITRES + 3)) + 2;
    /* (the hybrid redundancy margin is removed) */
    /* Variable bitrate */
    {
        let alpha: f32;
        let mut delta: i32;
        /* The target rate in 8th bits per frame */
        let mut target: i32;
        let mut base_target: i32;
        let lm_diff: i32 = modes::maxLM - LM;

        /* Don't attempt to use more than 510 kb/s, even for frames smaller than 20 ms.
        The CELT allocator will just not be able to use more than that anyway. */
        nbCompressedBytes = nbCompressedBytes.min(1275 >> (3 - LM));
        base_target = vbr_rate - ((40 * C + 20) << BITRES);

        if st.constrained_vbr != 0 {
            base_target += st.vbr_offset >> lm_diff;
        }

        target = compute_vbr(
            base_target,
            equiv_rate,
            st.lastCodedBands,
            C,
            st.intensity,
            st.constrained_vbr,
            st.stereo_saving,
            tot_boost,
            tf_estimate,
            maxDepth,
            temporal_vbr,
        );
        /* The current offset is removed from the target and the space used
        so far is added*/
        target = target + tell;

        nbAvailableBytes = (target + (1 << (BITRES + 2))) >> (BITRES + 3);
        nbAvailableBytes = min_allowed.max(nbAvailableBytes);
        nbAvailableBytes = nbCompressedBytes.min(nbAvailableBytes);

        /* By how much did we "miss" the target on that frame */
        delta = target - vbr_rate;

        target = nbAvailableBytes << (BITRES + 3);

        /*If the frame is silent we don't adjust our drift, otherwise
        the encoder will shoot to very high rates after hitting a
        span of silence, but we do allow the bitres to refill.
        This means that we'll undershoot our target in CVBR/VBR modes
        on files with lots of silence. */
        if silence != 0 {
            nbAvailableBytes = 2;
            target = 2 * 8 << BITRES;
            delta = 0;
        }

        if st.vbr_count < 970 {
            st.vbr_count += 1;
            alpha = rcp((st.vbr_count + 20) as f32);
        } else {
            alpha = 0.001;
        }
        /* How many bits have we used in excess of what we're allowed */
        if st.constrained_vbr != 0 {
            st.vbr_reservoir += target - vbr_rate;
        }

        /* Compute the offset we need to apply in order to reach the target */
        if st.constrained_vbr != 0 {
            st.vbr_drift +=
                (alpha * ((delta * (1 << lm_diff)) - st.vbr_offset - st.vbr_drift) as f32) as i32;
            st.vbr_offset = -st.vbr_drift;
        }

        if st.constrained_vbr != 0 && st.vbr_reservoir < 0 {
            /* We're under the min value -- increase rate */
            let adjust: i32 = (-st.vbr_reservoir) / (8 << BITRES);
            /* Unless we're just coding silence */
            nbAvailableBytes += if silence != 0 { 0 } else { adjust };
            st.vbr_reservoir = 0;
        }
        nbCompressedBytes = nbCompressedBytes.min(nbAvailableBytes);
        /* This moves the raw bits to take into account the new compressed size */
        ec_enc_shrink(enc, nbCompressedBytes as u32);
    }

    /* Bit allocation */
    let fine_quant = sg.alloc::<i32>(nb);
    let pulses = sg.alloc::<i32>(nb);
    let fine_priority = sg.alloc::<i32>(nb);

    /* bits =           packet size                    - where we are - safety*/
    bits = ((nbCompressedBytes * 8) << BITRES) - ec_tell_frac(enc) as i32 - 1;
    let anti_collapse_rsv: i32 = if isTransient != 0 && LM >= 2 && bits >= ((LM + 2) << BITRES) {
        1 << BITRES
    } else {
        0
    };
    bits -= anti_collapse_rsv;
    /* (analysis.valid is always 0, LFE is always 0; signalBandwidth == end - 1) */
    let mut balance: i32 = 0;
    let codedBands: i32 = clt_compute_allocation(
        end,
        offsets,
        cap,
        alloc_trim,
        &mut st.intensity,
        &mut dual_stereo,
        bits,
        &mut balance,
        pulses,
        fine_quant,
        fine_priority,
        C,
        enc,
        st.lastCodedBands,
    );
    if st.lastCodedBands != 0 {
        st.lastCodedBands = (st.lastCodedBands + 1).min((st.lastCodedBands - 1).max(codedBands));
    } else {
        st.lastCodedBands = codedBands;
    }

    quant_fine_energy(end, &mut st.oldBandE, error, fine_quant, enc, C);
    st.energyError[..nb * ccu].fill(0.0);

    /* Residual quantisation */
    let collapse_masks = sg.alloc::<u8>(C as usize * nb);
    {
        let (Xl, Xr) = X.split_at_mut(Nu);
        quant_all_bands(
            end,
            Xl,
            if C == 2 { Some(Xr) } else { None },
            collapse_masks,
            bandE,
            pulses,
            shortBlocks,
            st.spread_decision,
            dual_stereo,
            st.intensity,
            tf_res,
            nbCompressedBytes * (8 << BITRES) - anti_collapse_rsv,
            balance,
            enc,
            codedBands,
            &mut st.rng,
            st.complexity,
        );
    }

    if anti_collapse_rsv > 0 {
        anti_collapse_on = (st.consec_transient < 2) as i32;
        ec_enc_bits(enc, anti_collapse_on as u32, 1);
    }
    let _ = anti_collapse_on; /* only used by RESYNTH */
    /* (qext_bytes is always 0) */
    let bits_left = nbCompressedBytes * 8 - ec_tell(enc);
    quant_energy_finalise(
        end,
        &mut st.oldBandE,
        error,
        fine_quant,
        fine_priority,
        bits_left,
        enc,
        C,
    );
    for (ee, er) in st
        .energyError
        .as_chunks_mut::<NB>()
        .0
        .iter_mut()
        .zip(error.as_chunks::<NB>().0)
    {
        /* (start == 0) */
        for (ee, &er) in ee[..endu].iter_mut().zip(er) {
            *ee = (-0.5f32).max(0.5f32.min(er));
        }
    }
    if silence != 0 {
        st.oldBandE[..cu * nb].fill(-28.0);
    }

    /* (RESYNTH removed) */

    st.prefilter_period = pitch_index;
    st.prefilter_gain = gain1;
    st.prefilter_tapset = prefilter_tapset;

    if CC == 2 && C == 1 {
        st.oldBandE.copy_within(0..nb, nb);
    }

    let ccn = ccu * nb;
    if isTransient == 0 {
        st.oldLogE2[..ccn].copy_from_slice(&st.oldLogE[..ccn]);
        st.oldLogE[..ccn].copy_from_slice(&st.oldBandE[..ccn]);
    } else {
        for (l, &b) in st.oldLogE[..ccn].iter_mut().zip(&st.oldBandE) {
            *l = (*l).min(b);
        }
    }
    /* In case start or end were to change (start == 0) */
    for c in 0..ccu {
        let r = c * nb + endu..(c + 1) * nb;
        st.oldBandE[r.clone()].fill(0.0);
        st.oldLogE2[r.clone()].fill(-28.0);
        st.oldLogE[r].fill(-28.0);
    }

    if isTransient != 0 || transient_got_disabled != 0 {
        st.consec_transient += 1;
    } else {
        st.consec_transient = 0;
    }
    st.rng = enc.rng;

    /* If there's any room left (can only happen for very high rates),
    it's already filled with zeros */
    ec_enc_done(enc);

    if ec_get_error(enc) != 0 {
        OPUS_INTERNAL_ERROR
    } else {
        nbCompressedBytes
    }
}
