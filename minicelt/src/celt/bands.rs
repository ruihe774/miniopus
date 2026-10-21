//! Port of celt/bands.c and celt/bands.h (encoder side, float build, no QEXT, upstream
//! libopus 1.6.1 with the RFC 8251 update draft, i.e. DISABLE_UPDATE_DRAFT undefined).
//!
//! `encode` is always 1: the decoder paths are removed and the shared functions lose their
//! `encode` parameter. `denormalise_bands` and `anti_collapse` (decoder only) are not ported.
//!
//! The band quantisation recursion (`quant_band`, `quant_partition`, `quant_band_stereo`,
//! `compute_theta`, ...) works on raw pointers, as in C: X and Y are halves of one buffer, the
//! folding source `lowband` and the output `lowband_out` point into the `norm` buffer, and
//! without resynthesis `lowband_scratch` points into X itself.

use core::ptr;

use super::entcode::{BITRES, EC_ILOG, celt_sudiv, celt_udiv, ec_ctx, ec_tell_frac};
use super::entenc::{ec_enc_bit_logp, ec_enc_bits, ec_enc_uint, ec_encode};
use super::mathops::{Q31ONE, celt_sqrt};
use super::modes;
use super::pitch::celt_inner_prod;
use super::rate::{QTHETA_OFFSET, QTHETA_OFFSET_TWOPHASE, bits2pulses, get_pulses, pulses2bits};
use super::vq::{alg_quant, renormalise_vector, stereo_itheta};
use crate::stack;

pub const SPREAD_NONE: i32 = 0;
pub const SPREAD_LIGHT: i32 = 1;
pub const SPREAD_NORMAL: i32 = 2;
pub const SPREAD_AGGRESSIVE: i32 = 3;

/// FRAC_MUL16 (mathops.h): `(16384+((opus_int32)(opus_int16)(a)*(opus_int16)(b)))>>15`
#[inline(always)]
fn FRAC_MUL16(a: i32, b: i32) -> i32 {
    (16384 + (a as i16 as i32) * (b as i16 as i32)) >> 15
}

#[inline(always)]
unsafe fn slice_mut<'a>(p: *mut f32, n: i32) -> &'a mut [f32] {
    unsafe { core::slice::from_raw_parts_mut(p, n as usize) }
}

#[inline(always)]
unsafe fn slice<'a>(p: *const f32, n: i32) -> &'a [f32] {
    unsafe { core::slice::from_raw_parts(p, n as usize) }
}

/// `N` is the length of the tables; `prev` (the previous decision) is in `0..=N`.
pub fn hysteresis_decision<const N: usize>(
    val: f32,
    thresholds: &[f32; N],
    hysteresis: &[f32; N],
    prev: i32,
) -> i32 {
    let prev = prev as usize;
    let mut i = thresholds.iter().position(|&t| val < t).unwrap_or(N);
    /* i > prev and i < prev imply prev < N and 0 < prev <= N */
    if i > prev && val < thresholds[prev] + hysteresis[prev] {
        i = prev;
    }
    if i < prev && val > thresholds[prev - 1] - hysteresis[prev - 1] {
        i = prev;
    }
    i as i32
}

/// Linear congruential generator (relies on unsigned wraparound).
#[inline]
pub fn celt_lcg_rand(seed: u32) -> u32 {
    1664525u32.wrapping_mul(seed).wrapping_add(1013904223)
}

/* This is a cos() approximation designed to be bit-exact on any platform. Bit exactness
with this approximation is important because it has an impact on the bit allocation */
pub fn bitexact_cos(x: i16) -> i16 {
    let tmp: i32 = (4096 + (x as i32) * (x as i32)) >> 13;
    debug_assert!(tmp <= 32767);
    let mut x2: i16 = tmp as i16;
    x2 = ((32767 - x2 as i32)
        + FRAC_MUL16(
            x2 as i32,
            -7651 + FRAC_MUL16(x2 as i32, 8277 + FRAC_MUL16(-626, x2 as i32)),
        )) as i16;
    debug_assert!(x2 <= 32766);
    (1 + x2 as i32) as i16
}

pub fn bitexact_log2tan(mut isin: i32, mut icos: i32) -> i32 {
    let lc = EC_ILOG(icos as u32);
    let ls = EC_ILOG(isin as u32);
    icos <<= 15 - lc;
    isin <<= 15 - ls;
    (ls - lc) * (1 << 11) + FRAC_MUL16(isin, FRAC_MUL16(isin, -2597) + 7932)
        - FRAC_MUL16(icos, FRAC_MUL16(icos, -2597) + 7932)
}

/// Compute the amplitude (sqrt energy) in each of the bands
pub fn compute_band_energies(X: &[f32], bandE: &mut [f32], end: i32, C: i32) {
    /* LM == 3 */
    let LM = 3;
    let eBands = modes::eBands;
    let N = (modes::shortMdctSize << LM) as usize;
    debug_assert!(
        X.len() >= C as usize * N && bandE.len() >= C as usize * modes::nbEBands as usize
    );
    for (Xc, bandEc) in X
        .chunks_exact(N)
        .zip(bandE.chunks_exact_mut(modes::nbEBands as usize))
        .take(C as usize)
    {
        for (e, b) in bandEc.iter_mut().zip(eBands.windows(2)).take(end as usize) {
            let band = &Xc[(b[0] as usize) << LM..(b[1] as usize) << LM];
            let sum: f32 = 1e-27f32 + celt_inner_prod(band, band, band.len() as i32);
            *e = celt_sqrt(sum);
        }
    }
}

/// Normalise each band such that the energy is one.
pub fn normalise_bands(freq: &[f32], X: &mut [f32], bandE: &[f32], end: i32, C: i32) {
    /* M == 8 */
    let LM = 3;
    let eBands = modes::eBands;
    let N = (modes::shortMdctSize << LM) as usize;
    for c in 0..C as usize {
        let freqc = &freq[c * N..][..N];
        let Xc = &mut X[c * N..][..N];
        let bandEc = &bandE[c * modes::nbEBands as usize..][..end as usize];
        for (&e, b) in bandEc.iter().zip(eBands.windows(2)) {
            let g: f32 = 1.0f32 / (1e-27f32 + e);
            let r = (b[0] as usize) << LM..(b[1] as usize) << LM;
            for (x, &f) in Xc[r.clone()].iter_mut().zip(&freqc[r]) {
                *x = f * g;
            }
        }
    }
}

/* Compute the weights to use for optimizing normalized distortion across
channels. We use the amplitude to weight square distortion, which means
that we use the square root of the value we would have been using if we
wanted to minimize the MSE in the non-normalized domain. This roughly
corresponds to some quick-and-dirty perceptual experiments I ran to
measure inter-aural masking (there doesn't seem to be any published data
on the topic). */
fn compute_channel_weights(mut Ex: f32, mut Ey: f32, w: &mut [f32; 2]) {
    let minE = Ex.min(Ey);
    /* Adjustment to make the weights a bit more conservative. */
    Ex = Ex + minE / 3.0;
    Ey = Ey + minE / 3.0;
    w[0] = Ex;
    w[1] = Ey;
}

fn intensity_stereo(X: &mut [f32], Y: &[f32], bandE: &[f32], bandID: i32, N: i32) {
    let i = bandID as usize;
    let left = bandE[i];
    let right = bandE[i + modes::nbEBands as usize];
    let norm: f32 =
        super::mathops::EPSILON + celt_sqrt(super::mathops::EPSILON + left * left + right * right);
    let a1 = left / norm;
    let a2 = right / norm;
    /* the callers pass X and Y with length N */
    debug_assert!(X.len() == N as usize && Y.len() == N as usize);
    for (x, &y) in X.iter_mut().zip(Y) {
        *x = a1 * *x + a2 * y;
        /* Side is not encoded, no need to calculate */
    }
}

fn stereo_split(X: &mut [f32], Y: &mut [f32], N: i32) {
    for (x, y) in X[..N as usize].iter_mut().zip(&mut Y[..N as usize]) {
        let l = 0.70710678f32 * *x;
        let r = 0.70710678f32 * *y;
        *x = l + r;
        *y = r - l;
    }
}

fn stereo_merge(X: &mut [f32], Y: &mut [f32], mid: f32, N: i32) {
    /* Compute the norm of X+Y and X-Y as |X|^2 + |Y|^2 +/- sum(xy) */
    let mut xp = celt_inner_prod(Y, X, N);
    let side = celt_inner_prod(Y, Y, N);
    /* Compensating for the mid normalization */
    xp = mid * xp;
    /* mid and side are in Q15, not Q14 like X and Y */
    let El = mid * mid + side - 2.0 * xp;
    let Er = mid * mid + side + 2.0 * xp;
    if Er < 6e-4f32 || El < 6e-4f32 {
        Y[..N as usize].copy_from_slice(&X[..N as usize]);
        return;
    }

    let lgain = super::mathops::celt_rsqrt_norm32(El);
    let rgain = super::mathops::celt_rsqrt_norm32(Er);

    for (x, y) in X[..N as usize].iter_mut().zip(&mut Y[..N as usize]) {
        /* Apply mid scaling (side is already scaled) */
        let l = mid * *x;
        let r = *y;
        *x = lgain * (l - r);
        *y = rgain * (l + r);
    }
}

/// Decide whether we should spread the pulses in the current frame
pub fn spreading_decision(
    X: &[f32],
    average: &mut i32,
    last_decision: i32,
    hf_average: &mut i32,
    tapset_decision: &mut i32,
    update_hf: i32,
    end: i32,
    C: i32,
    spread_weight: &[i32],
) -> i32 {
    /* M == 8 */
    let M = 8;
    let mut sum: i32 = 0;
    let mut nbBands: i32 = 0;
    let eBands = modes::eBands;
    let mut hf_sum: i32 = 0;

    debug_assert!(end > 0);

    let N0 = (M * modes::shortMdctSize) as usize;
    let bands = &eBands[..end as usize + 1];
    let spread_weight = &spread_weight[..end as usize];

    if let [.., a, b] = *bands
        && M * (b as i32 - a as i32) <= 8
    {
        return SPREAD_NONE;
    }
    for c in 0..C as usize {
        let Xc = &X[c * N0..][..N0];
        for (i, (b, &w)) in bands.windows(2).zip(spread_weight).enumerate() {
            let mut tcount = [0i32; 3];
            let x = &Xc[M as usize * b[0] as usize..M as usize * b[1] as usize];
            let N = x.len() as i32;
            if N <= 8 {
                continue;
            }
            /* Compute rough CDF of |x[j]| */
            for &xj in x {
                let x2N: f32 = (xj * xj) * N as f32; /* Q13 */
                if x2N < 0.25f32 {
                    tcount[0] += 1;
                }
                if x2N < 0.0625f32 {
                    tcount[1] += 1;
                }
                if x2N < 0.015625f32 {
                    tcount[2] += 1;
                }
            }

            /* Only include four last bands (8 kHz and up) */
            if i as i32 > modes::nbEBands - 4 {
                hf_sum += celt_udiv((32 * (tcount[1] + tcount[0])) as u32, N as u32) as i32;
            }
            let tmp = (2 * tcount[2] >= N) as i32
                + (2 * tcount[1] >= N) as i32
                + (2 * tcount[0] >= N) as i32;
            sum += tmp * w;
            nbBands += w;
        }
    }

    if update_hf != 0 {
        if hf_sum != 0 {
            hf_sum = celt_udiv(hf_sum as u32, (C * (4 - modes::nbEBands + end)) as u32) as i32;
        }
        *hf_average = (*hf_average + hf_sum) >> 1;
        hf_sum = *hf_average;
        if *tapset_decision == 2 {
            hf_sum += 4;
        } else if *tapset_decision == 0 {
            hf_sum -= 4;
        }
        if hf_sum > 22 {
            *tapset_decision = 2;
        } else if hf_sum > 18 {
            *tapset_decision = 1;
        } else {
            *tapset_decision = 0;
        }
    }
    debug_assert!(nbBands > 0); /* end has to be non-zero */
    debug_assert!(sum >= 0);
    sum = celt_udiv((sum << 8) as u32, nbBands as u32) as i32;
    /* Recursive averaging */
    sum = (sum + *average) >> 1;
    *average = sum;
    /* Hysteresis */
    sum = (3 * sum + (((3 - last_decision) << 7) + 64) + 2) >> 2;
    if sum < 80 {
        SPREAD_AGGRESSIVE
    } else if sum < 256 {
        SPREAD_NORMAL
    } else if sum < 384 {
        SPREAD_LIGHT
    } else {
        SPREAD_NONE
    }
}

/* Indexing table for converting from natural Hadamard to ordery Hadamard
This is essentially a bit-reversed Gray, on top of which we've added
an inversion of the order because we want the DC at the end rather than
the beginning. The lines are for N=2, 4, 8, 16 */
static ordery_table: [i32; 30] = [
    1, 0, //
    3, 0, 2, 1, //
    7, 0, 4, 3, 6, 1, 5, 2, //
    15, 0, 8, 7, 12, 3, 11, 4, 14, 1, 9, 6, 13, 2, 10, 5,
];

/// Inverse permutations of the `ordery_table` lines: `ordery_inv[s - 2 + ordery_table[s - 2 + i]] == i`.
static ordery_inv: [u8; 30] = {
    let mut inv = [0u8; 30];
    let mut s = 2;
    while s <= 16 {
        let mut i = 0;
        while i < s {
            inv[s - 2 + ordery_table[s - 2 + i] as usize] = i as u8;
            i += 1;
        }
        s *= 2;
    }
    inv
};

fn deinterleave_hadamard(X: &mut [f32], N0: i32, stride: i32, hadamard: i32) {
    let st = stack::save();
    let N = (N0 * stride) as usize;
    let tmp = st.alloc::<f32>(N);
    debug_assert!(stride > 0);
    let (N0, stride) = (N0 as usize, stride as usize);
    let X = &mut X[..N];
    let inv: &[u8] = if hadamard != 0 {
        &ordery_inv[stride - 2..][..stride]
    } else {
        &[]
    };
    /* row r of tmp is column inv[r] (or r) of X, whose rows have length stride */
    let rows = X.chunks_exact(stride);
    for (r, out) in tmp.chunks_exact_mut(N0).enumerate() {
        let i = if hadamard != 0 { inv[r] as usize } else { r };
        for (t, x) in out.iter_mut().zip(rows.clone()) {
            *t = x[i];
        }
    }
    X.copy_from_slice(tmp);
}

// IX-HARD: the strided `tmp` writes (`j * stride + i`). Tried (as in `deinterleave_hadamard`, looping over
// the rows of X, `i = ordery_inv[r]`): `tmp.chunks_exact_mut(stride).zip(src)` with `t[i] = x`, check-free in
// the loop but a division per row: +0.14%/+0.13% encode time (wasm2, min/median), +0.07% (wasm3), +40 B;
// a `tmp.get_mut(k)`/break form (no division, one compare per sample): +0.21%/+0.17% (wasm2). Not converted.
fn interleave_hadamard(X: &mut [f32], N0: i32, stride: i32, hadamard: i32) {
    let st = stack::save();
    let N = (N0 * stride) as usize;
    let tmp = st.alloc::<f32>(N);
    let (N0, stride) = (N0 as usize, stride as usize);
    // SAFETY: the indices are in range as in the C code (`tmp` has N = N0 * stride entries, X at least N, the
    // `ordery_table` row has `stride` entries); the `debug_assert!`s check them in the `checked` profile.
    unsafe {
        debug_assert!(X.len() >= N);
        if hadamard != 0 {
            debug_assert!(stride - 2 + stride <= ordery_table.len());
            let ordery = ordery_table.get_unchecked(stride - 2..);
            for i in 0..stride {
                let o = *ordery.get_unchecked(i) as usize;
                for j in 0..N0 {
                    debug_assert!(j * stride + i < N && o * N0 + j < X.len());
                    *tmp.get_unchecked_mut(j * stride + i) = *X.get_unchecked(o * N0 + j);
                }
            }
        } else {
            for i in 0..stride {
                for j in 0..N0 {
                    debug_assert!(j * stride + i < N && i * N0 + j < X.len());
                    *tmp.get_unchecked_mut(j * stride + i) = *X.get_unchecked(i * N0 + j);
                }
            }
        }
        X.get_unchecked_mut(..N).copy_from_slice(tmp);
    }
}

// IX-HARD: very sensitive to its loop shape (called per band and level from tf_analysis and quant_band, often
// with few pairs per column, so per-call overhead counts). Check-free forms tried, encode time vs this one
// (wasm2, min/median, sum over bench.ts configs):
// - `chunks_exact_mut(2 * stride)` + `split_at_mut(stride)`, pair by pair: +1.0% (+124 B); with copies
//   for the strides 1/2/4/8: +0.6%/+0.5% (+614 B); `match stride` on 1/2/4/8 to const-generic
//   `as_chunks_mut::<{2 * stride}>` views (no division): +0.6..0.7% (+560..587 B); the same `match` around
//   this unchecked loop costs +0.6% too, so any stride dispatch is out; single form with `#[inline(always)]`:
//   +0.7%/+0.8% (+1.3 KiB).
// - `X[i..].iter_mut().step_by(stride)` taken in pairs: +1.0% (two end tests per pair).
// - one loop over `k += 2 * stride` with `k < n - stride`: the `X[k + stride]` check stays in the loop.
// - a `Cell` view `a = x[..n - stride]`, `b = x[stride..]` indexed by the same `k < a.len()`: check-free,
//   +0.45% (+106 B).
// - column order kept (`for i in 0..stride`, `chunks_exact_mut(2 * stride)`, `c[i]`/`c[stride + i]`): the
//   index checks stay in the inner loop, and +0.4%/+0.55% (+145 B).
pub fn haar1(X: &mut [f32], mut N0: i32, stride: i32) {
    N0 >>= 1;
    let stride = stride as usize;
    for i in 0..stride {
        for j in 0..N0 as usize {
            let (a, b) = (stride * 2 * j + i, stride * (2 * j + 1) + i);
            // SAFETY: the callers pass X with at least `2 * N0 * stride` entries (N0 halved above), as in the
            // C code; the `debug_assert!` checks it in the `checked` profile.
            debug_assert!(a < X.len() && b < X.len());
            unsafe {
                let tmp1 = 0.70710678f32 * *X.get_unchecked(a);
                let tmp2 = 0.70710678f32 * *X.get_unchecked(b);
                *X.get_unchecked_mut(a) = tmp1 + tmp2;
                *X.get_unchecked_mut(b) = tmp1 - tmp2;
            }
        }
    }
}

fn compute_qn(N: i32, b: i32, offset: i32, pulse_cap: i32, stereo: i32) -> i32 {
    static exp2_table8: [i16; 8] = [16384, 17866, 19483, 21247, 23170, 25267, 27554, 30048];
    let mut qn: i32;
    let mut qb: i32;
    let mut N2 = 2 * N - 1;
    if stereo != 0 && N == 2 {
        N2 -= 1;
    }
    /* The upper limit ensures that in a stereo split with itheta==16384, we'll
    always have enough bits left over to code at least one pulse in the
    side; otherwise it would collapse, since it doesn't get folded. */
    qb = celt_sudiv(b + N2 * offset, N2);
    qb = (b - pulse_cap - (4 << BITRES)).min(qb);

    qb = (8 << BITRES).min(qb);

    if qb < (1 << BITRES >> 1) {
        qn = 1;
    } else {
        qn = exp2_table8[(qb & 0x7) as usize] as i32 >> (14 - (qb >> BITRES));
        qn = (qn + 1) >> 1 << 1;
    }
    debug_assert!(qn <= 256);
    qn
}

/// struct band_ctx (`arch` dropped; `encode` is always 1, `disable_inv` always 0). Copied by
/// value to save and restore the state in the theta RDO.
#[derive(Clone, Copy)]
pub struct band_ctx<'a> {
    pub resynth: i32,
    pub i: i32,
    pub intensity: i32,
    pub spread: i32,
    pub tf_change: i32,
    pub ec: *mut ec_ctx,
    pub remaining_bits: i32,
    pub bandE: &'a [f32],
    pub seed: u32,
    pub theta_round: i32,
    pub avoid_split_noise: i32,
}

#[derive(Clone, Copy, Default)]
pub struct split_ctx {
    pub inv: i32,
    pub imid: i32,
    pub iside: i32,
    pub delta: i32,
    pub itheta: i32,
    pub qalloc: i32,
}

/// # Safety
/// `X` and `Y` must each be valid for `N` floats and not overlap; `ctx.ec` must be valid.
unsafe fn compute_theta(
    ctx: &mut band_ctx,
    sctx: &mut split_ctx,
    X: *mut f32,
    Y: *mut f32,
    N: i32,
    b: &mut i32,
    B: i32,
    B0: i32,
    LM: i32,
    stereo: i32,
    fill: &mut i32,
) {
    let mut itheta: i32;
    let delta: i32;
    let imid: i32;
    let iside: i32;
    let mut inv: i32 = 0;

    let i = ctx.i;
    let intensity = ctx.intensity;
    let ec = unsafe { &mut *ctx.ec };
    let bandE = ctx.bandE;

    /* Decide on the resolution to give to the split parameter theta */
    let pulse_cap = modes::logN[i as usize] as i32 + LM * (1 << BITRES);
    let offset = (pulse_cap >> 1)
        - if stereo != 0 && N == 2 {
            QTHETA_OFFSET_TWOPHASE
        } else {
            QTHETA_OFFSET
        };
    let mut qn = compute_qn(N, *b, offset, pulse_cap, stereo);
    if stereo != 0 && i >= intensity {
        qn = 1;
    }
    /* theta is the atan() of the ratio between the (normalized)
    side and mid. With just that parameter, we can re-scale both
    mid and side because we know that 1) they have unit norm and
    2) they are orthogonal. */
    let itheta_q30 = unsafe { stereo_itheta(slice(X, N), slice(Y, N), stereo, N) };
    itheta = itheta_q30 >> 16;
    let tell = ec_tell_frac(ec) as i32;
    if qn != 1 {
        if stereo == 0 || ctx.theta_round == 0 {
            itheta = (itheta * qn + 8192) >> 14;
            if stereo == 0 && ctx.avoid_split_noise != 0 && itheta > 0 && itheta < qn {
                /* Check if the selected value of theta will cause the bit allocation
                to inject noise on one side. If so, make sure the energy of that side
                is zero. */
                let unquantized = celt_udiv((itheta * 16384) as u32, qn as u32) as i32;
                let imid = bitexact_cos(unquantized as i16) as i32;
                let iside = bitexact_cos((16384 - unquantized) as i16) as i32;
                let delta = FRAC_MUL16((N - 1) << 7, bitexact_log2tan(iside, imid));
                if delta > *b {
                    itheta = qn;
                } else if delta < -*b {
                    itheta = 0;
                }
            }
        } else {
            /* Bias quantization towards itheta=0 and itheta=16384. */
            let bias = if itheta > 8192 {
                32767 / qn
            } else {
                -32767 / qn
            };
            let down = (qn - 1).min(0i32.max((itheta * qn + bias) >> 14));
            if ctx.theta_round < 0 {
                itheta = down;
            } else {
                itheta = down + 1;
            }
        }
        /* Entropy coding of the angle. We use a uniform pdf for the
        time split, a step for stereo, and a triangular one for the rest. */
        if stereo != 0 && N > 2 {
            let p0 = 3;
            let x = itheta;
            let x0 = qn / 2;
            let ft = p0 * (x0 + 1) + x0;
            /* Use a probability of p0 up to itheta=8192 and then use 1 after */
            ec_encode(
                ec,
                (if x <= x0 {
                    p0 * x
                } else {
                    (x - 1 - x0) + (x0 + 1) * p0
                }) as u32,
                (if x <= x0 {
                    p0 * (x + 1)
                } else {
                    (x - x0) + (x0 + 1) * p0
                }) as u32,
                ft as u32,
            );
        } else if B0 > 1 || stereo != 0 {
            /* Uniform pdf */
            ec_enc_uint(ec, itheta as u32, (qn + 1) as u32);
        } else {
            let ft = ((qn >> 1) + 1) * ((qn >> 1) + 1);
            let fs = if itheta <= (qn >> 1) {
                itheta + 1
            } else {
                qn + 1 - itheta
            };
            let fl = if itheta <= (qn >> 1) {
                itheta * (itheta + 1) >> 1
            } else {
                ft - ((qn + 1 - itheta) * (qn + 2 - itheta) >> 1)
            };
            ec_encode(ec, fl as u32, (fl + fs) as u32, ft as u32);
        }
        debug_assert!(itheta >= 0);
        itheta = celt_udiv((itheta * 16384) as u32, qn as u32) as i32;
        if stereo != 0 {
            unsafe {
                if itheta == 0 {
                    intensity_stereo(slice_mut(X, N), slice(Y, N), bandE, i, N);
                } else {
                    stereo_split(slice_mut(X, N), slice_mut(Y, N), N);
                }
            }
        }
        /* NOTE: Renormalising X and Y *may* help fixed-point a bit at very high rate.
        Let's do that at higher complexity */
    } else if stereo != 0 {
        inv = (itheta > 8192) as i32;
        if inv != 0 {
            for j in 0..N as usize {
                unsafe { *Y.add(j) = -*Y.add(j) };
            }
        }
        unsafe { intensity_stereo(slice_mut(X, N), slice(Y, N), bandE, i, N) };
        if *b > 2 << BITRES && ctx.remaining_bits > 2 << BITRES {
            ec_enc_bit_logp(ec, inv, 2);
        } else {
            inv = 0;
        }
        itheta = 0;
    }
    let qalloc = ec_tell_frac(ec) as i32 - tell;
    *b -= qalloc;

    if itheta == 0 {
        imid = 32767;
        iside = 0;
        *fill &= (1 << B) - 1;
        delta = -16384;
    } else if itheta == 16384 {
        imid = 0;
        iside = 32767;
        *fill &= ((1 << B) - 1) << B;
        delta = 16384;
    } else {
        imid = bitexact_cos(itheta as i16) as i32;
        iside = bitexact_cos((16384 - itheta) as i16) as i32;
        /* This is the mid vs side allocation that minimizes squared error
        in that band. */
        delta = FRAC_MUL16((N - 1) << 7, bitexact_log2tan(iside, imid));
    }

    sctx.inv = inv;
    sctx.imid = imid;
    sctx.iside = iside;
    sctx.delta = delta;
    sctx.itheta = itheta;
    sctx.qalloc = qalloc;
}

/// # Safety
/// `X`, and `Y` and `lowband_out` when non-null, must be valid for one float; `ctx.ec` valid.
unsafe fn quant_band_n1(
    ctx: &mut band_ctx,
    X: *mut f32,
    Y: *mut f32,
    lowband_out: *mut f32,
) -> u32 {
    let mut x = X;
    let ec = unsafe { &mut *ctx.ec };

    let stereo = !Y.is_null() as i32;
    for _c in 0..1 + stereo {
        let mut sign = 0;
        if ctx.remaining_bits >= 1 << BITRES {
            sign = unsafe { (*x < 0.0) as i32 };
            ec_enc_bits(ec, sign as u32, 1);
            ctx.remaining_bits -= 1 << BITRES;
        }
        if ctx.resynth != 0 {
            unsafe { *x = if sign != 0 { -1.0 } else { 1.0 } };
        }
        x = Y;
    }
    if !lowband_out.is_null() {
        unsafe { *lowband_out = *X };
    }
    1
}

/// The largest pulse cache entry of band `i` at `LM` (0..=3), `cache[cache[0]]` with
/// `cache = &cache_bits[cache_index[(LM + 1) * nbEBands + i]..]`, indexed by `LM * nbEBands + i`.
static split_thresh: [u8; 4 * modes::nbEBands as usize] = {
    let nb = modes::nbEBands as usize;
    let mut t = [0u8; 4 * modes::nbEBands as usize];
    let mut k = 0;
    while k < t.len() {
        let off = modes::cache_index[nb + k];
        assert!(off >= 0);
        let off = off as usize;
        t[k] = modes::cache_bits[off + modes::cache_bits[off] as usize];
        k += 1;
    }
    t
};

/* This function is responsible for encoding and decoding a mono partition.
It can split the band in two and transmit the energy difference with
the two half-bands. It can be called recursively so bands can end up being
split in 8 parts. */
/// # Safety
/// `X` must be valid for `N` floats, `lowband` (if non-null) for `N` floats, not overlapping X.
unsafe fn quant_partition(
    ctx: &mut band_ctx,
    X: *mut f32,
    mut N: i32,
    mut b: i32,
    mut B: i32,
    lowband: *mut f32,
    mut LM: i32,
    gain: f32,
    mut fill: i32,
) -> u32 {
    let B0 = B;
    let mut cm: u32 = 0;

    let i = ctx.i;
    let spread = ctx.spread;

    /* If we need 1.5 more bit than we can produce, split the band in two. */
    if LM != -1 && b > split_thresh[(LM * modes::nbEBands + i) as usize] as i32 + 12 && N > 2 {
        let mut sctx = split_ctx::default();
        let mut next_lowband2: *mut f32 = ptr::null_mut();

        N >>= 1;
        let Y = unsafe { X.add(N as usize) };
        LM -= 1;
        if B == 1 {
            fill = (fill & 1) | (fill << 1);
        }
        B = (B + 1) >> 1;

        unsafe { compute_theta(ctx, &mut sctx, X, Y, N, &mut b, B, B0, LM, 0, &mut fill) };
        let imid = sctx.imid;
        let iside = sctx.iside;
        let mut delta = sctx.delta;
        let itheta = sctx.itheta;
        let qalloc = sctx.qalloc;
        let mid: f32 = (1.0f32 / 32768.0) * imid as f32;
        let side: f32 = (1.0f32 / 32768.0) * iside as f32;

        /* Give more bits to low-energy MDCTs than they would otherwise deserve */
        if B0 > 1 && (itheta & 0x3fff) != 0 {
            if itheta > 8192 {
                /* Rough approximation for pre-echo masking */
                delta -= delta >> (4 - LM);
            } else {
                /* Corresponds to a forward-masking slope of 1.5 dB per 10 ms */
                delta = 0i32.min(delta + (N << BITRES >> (5 - LM)));
            }
        }
        let mut mbits = 0i32.max(b.min((b - delta) / 2));
        let mut sbits = b - mbits;
        ctx.remaining_bits -= qalloc;

        if !lowband.is_null() {
            next_lowband2 = unsafe { lowband.add(N as usize) }; /* >32-bit split case */
        }

        let mut rebalance = ctx.remaining_bits;
        unsafe {
            if mbits >= sbits {
                cm = quant_partition(ctx, X, N, mbits, B, lowband, LM, gain * mid, fill);
                rebalance = mbits - (rebalance - ctx.remaining_bits);
                if rebalance > 3 << BITRES && itheta != 0 {
                    sbits += rebalance - (3 << BITRES);
                }
                cm |= quant_partition(
                    ctx,
                    Y,
                    N,
                    sbits,
                    B,
                    next_lowband2,
                    LM,
                    gain * side,
                    fill >> B,
                ) << (B0 >> 1);
            } else {
                cm = quant_partition(
                    ctx,
                    Y,
                    N,
                    sbits,
                    B,
                    next_lowband2,
                    LM,
                    gain * side,
                    fill >> B,
                ) << (B0 >> 1);
                rebalance = sbits - (rebalance - ctx.remaining_bits);
                if rebalance > 3 << BITRES && itheta != 16384 {
                    mbits += rebalance - (3 << BITRES);
                }
                cm |= quant_partition(ctx, X, N, mbits, B, lowband, LM, gain * mid, fill);
            }
        }
    } else {
        /* This is the basic no-split case */
        let mut q = bits2pulses(i, LM, b);
        let mut curr_bits = pulses2bits(i, LM, q);
        ctx.remaining_bits -= curr_bits;

        /* Ensures we can never bust the budget */
        while ctx.remaining_bits < 0 && q > 0 {
            ctx.remaining_bits += curr_bits;
            q -= 1;
            curr_bits = pulses2bits(i, LM, q);
            ctx.remaining_bits -= curr_bits;
        }

        if q != 0 {
            let K = get_pulses(q);

            /* Finally do the actual quantization */
            cm = unsafe {
                alg_quant(
                    slice_mut(X, N),
                    N,
                    K,
                    spread,
                    B,
                    &mut *ctx.ec,
                    gain,
                    ctx.resynth,
                )
            };
        } else {
            /* If there's no pulse, fill the band anyway */
            if ctx.resynth != 0 {
                /* B can be as large as 16, so this shift might overflow an int on a
                16-bit platform; use a long to get defined behavior.*/
                let cm_mask: u32 = (1u32 << B) - 1;
                fill = (fill as u32 & cm_mask) as i32;
                let x = unsafe { slice_mut(X, N) };
                if fill == 0 {
                    x.fill(0.0);
                } else {
                    if lowband.is_null() {
                        /* Noise */
                        for v in x.iter_mut() {
                            ctx.seed = celt_lcg_rand(ctx.seed);
                            *v = ((ctx.seed as i32) >> 20) as f32;
                        }
                        cm = cm_mask;
                    } else {
                        /* Folded spectrum */
                        let lowband = unsafe { slice(lowband, N) };
                        for (v, &l) in x.iter_mut().zip(lowband) {
                            ctx.seed = celt_lcg_rand(ctx.seed);
                            /* About 48 dB below the "normal" folding level */
                            let mut tmp: f32 = 1.0f32 / 256.0;
                            tmp = if (ctx.seed) & 0x8000 != 0 { tmp } else { -tmp };
                            *v = l + tmp;
                        }
                        cm = fill as u32;
                    }
                    renormalise_vector(x, N, gain);
                }
            }
        }
    }

    cm
}

/* This function is responsible for encoding and decoding a band for the mono case. */
/// # Safety
/// `X` valid for `N` floats; `lowband`, `lowband_out`, `lowband_scratch` (each nullable) valid
/// for `N` floats; `lowband_scratch` must not overlap `X` or `lowband`.
unsafe fn quant_band(
    ctx: &mut band_ctx,
    X: *mut f32,
    N: i32,
    b: i32,
    mut B: i32,
    mut lowband: *mut f32,
    LM: i32,
    lowband_out: *mut f32,
    gain: f32,
    lowband_scratch: *mut f32,
    mut fill: i32,
) -> u32 {
    let N0 = N;
    let mut N_B = N;
    let N_B0: i32;
    let mut B0 = B;
    let mut time_divide = 0;
    let mut recombine = 0;
    let mut cm: u32;

    let mut tf_change = ctx.tf_change;

    let longBlocks = (B0 == 1) as i32;

    N_B = celt_udiv(N_B as u32, B as u32) as i32;

    /* Special case for one sample */
    if N == 1 {
        return unsafe { quant_band_n1(ctx, X, ptr::null_mut(), lowband_out) };
    }

    if tf_change > 0 {
        recombine = tf_change;
    }
    /* Band recombining to increase frequency resolution */

    if !lowband_scratch.is_null()
        && !lowband.is_null()
        && (recombine != 0 || ((N_B & 1) == 0 && tf_change < 0) || B0 > 1)
    {
        unsafe { ptr::copy(lowband, lowband_scratch, N as usize) };
        lowband = lowband_scratch;
    }

    for k in 0..recombine {
        static bit_interleave_table: [u8; 16] = [0, 1, 1, 1, 2, 3, 3, 3, 2, 3, 3, 3, 2, 3, 3, 3];
        unsafe {
            haar1(slice_mut(X, N), N >> k, 1 << k);
            if !lowband.is_null() {
                haar1(slice_mut(lowband, N), N >> k, 1 << k);
            }
        }
        fill = bit_interleave_table[(fill & 0xF) as usize] as i32
            | (bit_interleave_table[(fill >> 4) as usize] as i32) << 2;
    }
    B >>= recombine;
    N_B <<= recombine;

    /* Increasing the time resolution */
    while (N_B & 1) == 0 && tf_change < 0 {
        unsafe {
            haar1(slice_mut(X, N), N_B, B);
            if !lowband.is_null() {
                haar1(slice_mut(lowband, N), N_B, B);
            }
        }
        fill |= fill << B;
        B <<= 1;
        N_B >>= 1;
        time_divide += 1;
        tf_change += 1;
    }
    B0 = B;
    N_B0 = N_B;

    /* Reorganize the samples in time order instead of frequency order */
    if B0 > 1 {
        unsafe {
            deinterleave_hadamard(
                slice_mut(X, N),
                N_B >> recombine,
                B0 << recombine,
                longBlocks,
            );
            if !lowband.is_null() {
                deinterleave_hadamard(
                    slice_mut(lowband, N),
                    N_B >> recombine,
                    B0 << recombine,
                    longBlocks,
                );
            }
        }
    }

    cm = unsafe { quant_partition(ctx, X, N, b, B, lowband, LM, gain, fill) };

    /* This code is used by the decoder and by the resynthesis-enabled encoder */
    if ctx.resynth != 0 {
        let x = unsafe { slice_mut(X, N) };
        /* Undo the sample reorganization going from time order to frequency order */
        if B0 > 1 {
            interleave_hadamard(x, N_B >> recombine, B0 << recombine, longBlocks);
        }

        /* Undo time-freq changes that we did earlier */
        N_B = N_B0;
        B = B0;
        for _k in 0..time_divide {
            B >>= 1;
            N_B <<= 1;
            cm |= cm >> B;
            haar1(x, N_B, B);
        }

        for k in 0..recombine {
            static bit_deinterleave_table: [u8; 16] = [
                0x00, 0x03, 0x0C, 0x0F, 0x30, 0x33, 0x3C, 0x3F, 0xC0, 0xC3, 0xCC, 0xCF, 0xF0, 0xF3,
                0xFC, 0xFF,
            ];
            cm = bit_deinterleave_table[cm as usize] as u32;
            haar1(x, N0 >> k, 1 << k);
        }
        B <<= recombine;

        /* Scale output for later folding */
        if !lowband_out.is_null() {
            let n: f32 = crate::celt::mathops::sqrt64(N0 as f64) as f32;
            let lowband_out = unsafe { slice_mut(lowband_out, N0) };
            for (o, &v) in lowband_out.iter_mut().zip(x.iter()) {
                *o = n * v;
            }
        }
        cm &= ((1 << B) - 1) as u32;
    }
    cm
}

const MIN_STEREO_ENERGY: f32 = 1e-10;

/* This function is responsible for encoding and decoding a band for the stereo case. */
/// # Safety
/// As `quant_band`, plus `Y` valid for `N` floats and disjoint from `X`.
unsafe fn quant_band_stereo(
    ctx: &mut band_ctx,
    X: *mut f32,
    Y: *mut f32,
    N: i32,
    mut b: i32,
    B: i32,
    lowband: *mut f32,
    LM: i32,
    lowband_out: *mut f32,
    lowband_scratch: *mut f32,
    mut fill: i32,
) -> u32 {
    let mut cm: u32;
    let mut mbits: i32;
    let mut sbits: i32;
    let mut sctx = split_ctx::default();

    /* Special case for one sample */
    if N == 1 {
        return unsafe { quant_band_n1(ctx, X, Y, lowband_out) };
    }

    let orig_fill = fill;

    {
        let (bi, bs) = (
            ctx.bandE[ctx.i as usize],
            ctx.bandE[(modes::nbEBands + ctx.i) as usize],
        );
        if bi < MIN_STEREO_ENERGY || bs < MIN_STEREO_ENERGY {
            unsafe {
                if bi > bs {
                    ptr::copy(X, Y, N as usize);
                } else {
                    ptr::copy(Y, X, N as usize);
                }
            }
        }
    }
    unsafe { compute_theta(ctx, &mut sctx, X, Y, N, &mut b, B, B, LM, 1, &mut fill) };
    let inv = sctx.inv;
    let imid = sctx.imid;
    let iside = sctx.iside;
    let delta = sctx.delta;
    let itheta = sctx.itheta;
    let qalloc = sctx.qalloc;
    let mid: f32 = (1.0f32 / 32768.0) * imid as f32;
    let side: f32 = (1.0f32 / 32768.0) * iside as f32;

    /* This is a special case for N=2 that only works for stereo and takes
    advantage of the fact that mid and side are orthogonal to encode
    the side with just one bit. */
    if N == 2 {
        let mut sign = 0;
        mbits = b;
        sbits = 0;
        /* Only need one bit for the side. */
        if itheta != 0 && itheta != 16384 {
            sbits = 1 << BITRES;
        }
        mbits -= sbits;
        let c = itheta > 8192;
        ctx.remaining_bits -= qalloc + sbits;

        let x2 = if c { Y } else { X };
        let y2 = if c { X } else { Y };
        unsafe {
            if sbits != 0 {
                /* Here we only need to encode a sign for the side. */
                /* FIXME: Need to increase fixed-point precision? */
                sign = (*x2 * *y2.add(1) - *x2.add(1) * *y2 < 0.0) as i32;
                ec_enc_bits(&mut *ctx.ec, sign as u32, 1);
            }
            sign = 1 - 2 * sign;
            /* We use orig_fill here because we want to fold the side, but if
            itheta==16384, we'll have cleared the low bits of fill. */
            cm = quant_band(
                ctx,
                x2,
                N,
                mbits,
                B,
                lowband,
                LM,
                lowband_out,
                Q31ONE,
                lowband_scratch,
                orig_fill,
            );
            /* We don't split N=2 bands, so cm is either 1 or 0 (for a fold-collapse),
            and there's no need to worry about mixing with the other channel. */
            *y2 = (-sign) as f32 * *x2.add(1);
            *y2.add(1) = sign as f32 * *x2;
            if ctx.resynth != 0 {
                *X = mid * *X;
                *X.add(1) = mid * *X.add(1);
                *Y = side * *Y;
                *Y.add(1) = side * *Y.add(1);
                let mut tmp = *X;
                *X = tmp - *Y;
                *Y = tmp + *Y;
                tmp = *X.add(1);
                *X.add(1) = tmp - *Y.add(1);
                *Y.add(1) = tmp + *Y.add(1);
            }
        }
    } else {
        /* "Normal" split code */
        mbits = 0i32.max(b.min((b - delta) / 2));
        sbits = b - mbits;
        ctx.remaining_bits -= qalloc;

        let mut rebalance = ctx.remaining_bits;
        let null = ptr::null_mut();
        unsafe {
            if mbits >= sbits {
                /* In stereo mode, we do not apply a scaling to the mid because we need the normalized
                mid for folding later. */
                cm = quant_band(
                    ctx,
                    X,
                    N,
                    mbits,
                    B,
                    lowband,
                    LM,
                    lowband_out,
                    Q31ONE,
                    lowband_scratch,
                    fill,
                );
                rebalance = mbits - (rebalance - ctx.remaining_bits);
                if rebalance > 3 << BITRES && itheta != 0 {
                    sbits += rebalance - (3 << BITRES);
                }
                /* For a stereo split, the high bits of fill are always zero, so no
                folding will be done to the side. */
                cm |= quant_band(ctx, Y, N, sbits, B, null, LM, null, side, null, fill >> B);
            } else {
                /* For a stereo split, the high bits of fill are always zero, so no
                folding will be done to the side. */
                cm = quant_band(ctx, Y, N, sbits, B, null, LM, null, side, null, fill >> B);
                rebalance = sbits - (rebalance - ctx.remaining_bits);
                if rebalance > 3 << BITRES && itheta != 16384 {
                    mbits += rebalance - (3 << BITRES);
                }
                /* In stereo mode, we do not apply a scaling to the mid because we need the normalized
                mid for folding later. */
                cm |= quant_band(
                    ctx,
                    X,
                    N,
                    mbits,
                    B,
                    lowband,
                    LM,
                    lowband_out,
                    Q31ONE,
                    lowband_scratch,
                    fill,
                );
            }
        }
    }

    /* This code is used by the decoder and by the resynthesis-enabled encoder */
    if ctx.resynth != 0 {
        unsafe {
            if N != 2 {
                stereo_merge(slice_mut(X, N), slice_mut(Y, N), mid, N);
            }
            if inv != 0 {
                for j in 0..N as usize {
                    *Y.add(j) = -*Y.add(j);
                }
            }
        }
    }
    cm
}

/// Quantisation/encoding of the residual spectrum (`encode` = 1).
///
/// `X_` and `Y_` are the two channels of the normalised spectrum (in C, `X` and `X+N` of one
/// buffer); each must hold at least `M*eBands[end]` values. `Y_` is `None` for mono.
pub fn quant_all_bands(
    end: i32,
    X_: &mut [f32],
    Y_: Option<&mut [f32]>,
    collapse_masks: &mut [u8],
    bandE: &[f32],
    pulses: &[i32],
    shortBlocks: i32,
    spread: i32,
    mut dual_stereo: i32,
    intensity: i32,
    tf_res: &[i32],
    total_bits: i32,
    mut balance: i32,
    ec: &mut ec_ctx,
    codedBands: i32,
    seed: &mut u32,
    complexity: i32,
) {
    /* start == 0, LM == 3 (so M == 8) */
    let start = 0;
    let LM = 3;
    let eBands = modes::eBands;
    let eb = |k: i32| eBands[k as usize] as i32;
    let nb = modes::nbEBands;
    let mut update_lowband = 1;
    let C: i32 = if Y_.is_some() { 2 } else { 1 };
    let theta_rdo = Y_.is_some() && dual_stereo == 0 && complexity >= 8;
    let resynth = theta_rdo;

    let M = 1 << LM;
    let (pulses, tf_res) = (&pulses[..end as usize], &tf_res[..end as usize]);
    debug_assert!(X_.len() >= (M * eb(end)) as usize);
    debug_assert!(collapse_masks.len() >= (C * end) as usize);
    let X_p: *mut f32 = X_.as_mut_ptr();
    let Y_p: *mut f32 = match Y_ {
        Some(y) => {
            debug_assert!(y.len() >= (M * eb(end)) as usize);
            y.as_mut_ptr()
        }
        None => ptr::null_mut(),
    };
    // Only the raw pointer is used from here on (ctx holds a copy).
    let ec: *mut ec_ctx = ec;

    let st = stack::save();
    let B = if shortBlocks != 0 { M } else { 1 };
    let norm_offset = M * eb(start);
    /* No need to allocate norm for the last band because we don't need an
    output in that band. norm (the folding source) is only read and written by the resynthesis
    (the lowband stays NULL without it, and the N == 1 case that writes lowband_out regardless
    does not occur at LM == 3), and the resynthesis (theta RDO) needs dual_stereo == 0, i.e.
    no norm2: so there is one buffer with resynth and none without (the dual stereo calls get
    `norm` too, which is then never dereferenced). */
    debug_assert!(!resynth || dual_stereo == 0);
    let _norm = st.alloc::<f32>(if resynth {
        (M * eb(nb - 1) - norm_offset) as usize
    } else {
        0
    });
    let norm: *mut f32 = _norm.as_mut_ptr();

    /* For decoding, we can use the last band as scratch space because we don't need that
    scratch space for the last band and we don't care about the data there until we're
    decoding the last band. */
    let resynth_alloc = if resynth {
        (M * (eb(nb) - eb(nb - 1))) as usize
    } else {
        0
    };
    let _lowband_scratch = st.alloc::<f32>(resynth_alloc);
    let mut lowband_scratch: *mut f32 = if resynth {
        _lowband_scratch.as_mut_ptr()
    } else {
        X_p.wrapping_add((M * eb(modes::effEBands - 1)) as usize)
    };
    let X_save = st.alloc::<f32>(resynth_alloc);
    let Y_save = st.alloc::<f32>(resynth_alloc);
    let X_save2 = st.alloc::<f32>(resynth_alloc);
    let Y_save2 = st.alloc::<f32>(resynth_alloc);
    let norm_save2 = st.alloc::<f32>(resynth_alloc);

    let mut lowband_offset = 0;
    let mut ctx = band_ctx {
        resynth: resynth as i32,
        i: 0,
        intensity,
        spread,
        tf_change: 0,
        ec,
        remaining_bits: 0,
        bandE,
        seed: *seed,
        theta_round: 0,
        /* Avoid injecting noise in the first band on transients. */
        avoid_split_noise: (B > 1) as i32,
    };
    let bytes_save = st.alloc::<u8>(if theta_rdo { 1275 } else { 0 });

    for i in start..end {
        let mut effective_lowband: i32 = -1;
        let mut x_cm: u32;
        let mut y_cm: u32;

        ctx.i = i;
        let last = i == end - 1;

        let mut X = X_p.wrapping_add((M * eb(i)) as usize);
        let mut Y = if !Y_p.is_null() {
            Y_p.wrapping_add((M * eb(i)) as usize)
        } else {
            ptr::null_mut()
        };
        let N = M * eb(i + 1) - M * eb(i);
        debug_assert!(N > 0);
        let tell = unsafe { ec_tell_frac(&*ec) } as i32;

        /* Compute how many bits we want to allocate to this band */
        if i != start {
            balance -= tell;
        }
        let remaining_bits = total_bits - tell - 1;
        ctx.remaining_bits = remaining_bits;
        let b = if i <= codedBands - 1 {
            let curr_balance = celt_sudiv(balance, 3i32.min(codedBands - i));
            0i32.max(16383i32.min((remaining_bits + 1).min(pulses[i as usize] + curr_balance)))
        } else {
            0
        };

        if resynth
            && (M * eb(i) - N >= M * eb(start) || i == start + 1)
            && (update_lowband != 0 || lowband_offset == 0)
        {
            lowband_offset = i;
        }
        /* special_hybrid_folding copies no data for start == 0 */

        let tf_change = tf_res[i as usize];
        ctx.tf_change = tf_change;
        if i >= modes::effEBands {
            X = norm;
            if !Y_p.is_null() {
                Y = norm;
            }
            lowband_scratch = ptr::null_mut();
        }
        if last && !theta_rdo {
            lowband_scratch = ptr::null_mut();
        }

        /* Get a conservative estimate of the collapse_mask's for the bands we're
        going to be folding from. */
        if lowband_offset != 0 && (spread != SPREAD_AGGRESSIVE || B > 1 || tf_change < 0) {
            /* This ensures we never repeat spectral content within one band */
            effective_lowband = 0i32.max(M * eb(lowband_offset) - norm_offset - N);
            let mut fold_start = lowband_offset;
            loop {
                fold_start -= 1;
                if !(M * eb(fold_start) > effective_lowband + norm_offset) {
                    break;
                }
            }
            let mut fold_end = lowband_offset - 1;
            loop {
                fold_end += 1;
                if !(fold_end < i && M * eb(fold_end) < effective_lowband + norm_offset + N) {
                    break;
                }
            }
            x_cm = 0;
            y_cm = 0;
            let mut fold_i = fold_start;
            loop {
                x_cm |= collapse_masks[(fold_i * C + 0) as usize] as u32;
                y_cm |= collapse_masks[(fold_i * C + C - 1) as usize] as u32;
                fold_i += 1;
                if fold_i >= fold_end {
                    break;
                }
            }
        }
        /* Otherwise, we'll be using the LCG to fold, so all blocks will (almost
        always) be non-zero. */
        else {
            x_cm = ((1 << B) - 1) as u32;
            y_cm = x_cm;
        }

        if dual_stereo != 0 && i == intensity {
            /* Switch off dual stereo to do intensity. */
            dual_stereo = 0;
            /* (the norm/norm2 average is resynth-only, and resynth has dual_stereo == 0) */
        }
        let lowband = |n: *mut f32| {
            if effective_lowband != -1 {
                n.wrapping_add(effective_lowband as usize)
            } else {
                ptr::null_mut()
            }
        };
        let lowband_out = |n: *mut f32| {
            if last {
                ptr::null_mut()
            } else {
                n.wrapping_add((M * eb(i) - norm_offset) as usize)
            }
        };
        unsafe {
            if dual_stereo != 0 {
                x_cm = quant_band(
                    &mut ctx,
                    X,
                    N,
                    b / 2,
                    B,
                    lowband(norm),
                    LM,
                    lowband_out(norm),
                    Q31ONE,
                    lowband_scratch,
                    x_cm as i32,
                );
                y_cm = quant_band(
                    &mut ctx,
                    Y,
                    N,
                    b / 2,
                    B,
                    lowband(norm),
                    LM,
                    lowband_out(norm),
                    Q31ONE,
                    lowband_scratch,
                    y_cm as i32,
                );
            } else {
                if !Y.is_null() {
                    if theta_rdo && i < intensity {
                        let mut w = [0.0f32; 2];
                        compute_channel_weights(
                            bandE[i as usize],
                            bandE[(i + nb) as usize],
                            &mut w,
                        );
                        /* Make a copy. */
                        let cm = x_cm | y_cm;
                        let ec_save: ec_ctx = *ec;
                        let ctx_save = ctx;
                        let n = N as usize;
                        X_save[..n].copy_from_slice(slice(X, N));
                        Y_save[..n].copy_from_slice(slice(Y, N));
                        /* Encode and round down. */
                        ctx.theta_round = -1;
                        x_cm = quant_band_stereo(
                            &mut ctx,
                            X,
                            Y,
                            N,
                            b,
                            B,
                            lowband(norm),
                            LM,
                            lowband_out(norm),
                            lowband_scratch,
                            cm as i32,
                        );
                        let dist0: f32 = w[0] * celt_inner_prod(X_save, slice(X, N), N)
                            + w[1] * celt_inner_prod(Y_save, slice(Y, N), N);

                        /* Save first result. */
                        let cm2 = x_cm;
                        let ec_save2: ec_ctx = *ec;
                        let ctx_save2 = ctx;
                        X_save2[..n].copy_from_slice(slice(X, N));
                        Y_save2[..n].copy_from_slice(slice(Y, N));
                        if !last {
                            norm_save2[..n].copy_from_slice(slice(lowband_out(norm), N));
                        }
                        let nstart_bytes = ec_save.offs as i32;
                        let nend_bytes = ec_save.storage as i32;
                        let bytes_buf = ec_save.buf.wrapping_add(nstart_bytes as usize);
                        let save_bytes = (nend_bytes - nstart_bytes) as usize;
                        ptr::copy_nonoverlapping(bytes_buf, bytes_save.as_mut_ptr(), save_bytes);
                        /* Restore */
                        *ec = ec_save;
                        ctx = ctx_save;
                        slice_mut(X, N).copy_from_slice(&X_save[..n]);
                        slice_mut(Y, N).copy_from_slice(&Y_save[..n]);
                        /* Encode and round up. */
                        ctx.theta_round = 1;
                        x_cm = quant_band_stereo(
                            &mut ctx,
                            X,
                            Y,
                            N,
                            b,
                            B,
                            lowband(norm),
                            LM,
                            lowband_out(norm),
                            lowband_scratch,
                            cm as i32,
                        );
                        let dist1: f32 = w[0] * celt_inner_prod(X_save, slice(X, N), N)
                            + w[1] * celt_inner_prod(Y_save, slice(Y, N), N);
                        if dist0 >= dist1 {
                            x_cm = cm2;
                            *ec = ec_save2;
                            ctx = ctx_save2;
                            slice_mut(X, N).copy_from_slice(&X_save2[..n]);
                            slice_mut(Y, N).copy_from_slice(&Y_save2[..n]);
                            if !last {
                                slice_mut(lowband_out(norm), N).copy_from_slice(&norm_save2[..n]);
                            }
                            ptr::copy_nonoverlapping(bytes_save.as_ptr(), bytes_buf, save_bytes);
                        }
                    } else {
                        ctx.theta_round = 0;
                        x_cm = quant_band_stereo(
                            &mut ctx,
                            X,
                            Y,
                            N,
                            b,
                            B,
                            lowband(norm),
                            LM,
                            lowband_out(norm),
                            lowband_scratch,
                            (x_cm | y_cm) as i32,
                        );
                    }
                } else {
                    x_cm = quant_band(
                        &mut ctx,
                        X,
                        N,
                        b,
                        B,
                        lowband(norm),
                        LM,
                        lowband_out(norm),
                        Q31ONE,
                        lowband_scratch,
                        (x_cm | y_cm) as i32,
                    );
                }
                y_cm = x_cm;
            }
        }
        collapse_masks[(i * C + 0) as usize] = x_cm as u8;
        collapse_masks[(i * C + C - 1) as usize] = y_cm as u8;
        balance += pulses[i as usize] + tell;

        /* Update the folding position only as long as we have 1 bit/sample depth. */
        update_lowband = (b > (N << BITRES)) as i32;
        /* We only need to avoid noise on a split for the first band. After that, we
        have folding. */
        ctx.avoid_split_noise = 0;
    }
    *seed = ctx.seed;
}
