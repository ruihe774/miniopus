//! Port of celt/vq.c (encoder side, float build, no QEXT) and the fork's celt/wasm/vq_wasm.c
//! (`op_pvq_search_wasm`, which replaces `op_pvq_search_c`).
//!
//! In the float build `celt_inner_prod_norm` and `celt_inner_prod_norm_shift` are
//! `celt_inner_prod`, and `norm_scaleup`/`norm_scaledown` are no-ops.

use core::arch::wasm32::*;

use super::bands::SPREAD_NONE;
use super::cwrs::encode_pulses;
use super::entcode::{celt_udiv, ec_enc};
use super::mathops::{EPSILON, atan2p_norm, cos_norm, floor64, rcp, rsqrt, sqrt};
use super::pitch::celt_inner_prod;
use super::simd::{
    hmax4, hmin_i32, hsum_f32, hsum_i32, laneselect, load4, madd, max4, store4, trunc_i32,
};
use crate::stack;

fn exp_rotation1(X: &mut [f32], len: i32, stride: i32, c: f32, s: f32) {
    let ms = -s;
    let stride = stride as usize;
    // One check up front; the loops below index below `len`.
    let X = &mut X[..len as usize];
    let mut i = 0usize;
    while i + stride < X.len() {
        let x1 = X[i];
        let x2 = X[i + stride];
        X[i + stride] = c * x2 + s * x1;
        X[i] = c * x1 + ms * x2;
        i += 1;
    }
    for p in (0..X.len().saturating_sub(2 * stride)).rev() {
        let x1 = X[p];
        let x2 = X[p + stride];
        X[p + stride] = c * x2 + s * x1;
        X[p] = c * x1 + ms * x2;
    }
}

pub fn exp_rotation(X: &mut [f32], mut len: i32, dir: i32, stride: i32, K: i32, spread: i32) {
    let mut stride2: i32 = 0;

    if 2 * K >= len || spread == SPREAD_NONE {
        return;
    }
    /* SPREAD_LIGHT, SPREAD_NORMAL, SPREAD_AGGRESSIVE */
    let factor = match spread {
        1 => 15,
        2 => 10,
        _ => 5,
    };

    let gain: f32 = (1.0f32 * len as f32) / ((len + factor * K) as f32);
    let theta: f32 = 0.5f32 * (gain * gain);

    let c = cos_norm(theta);
    let s = cos_norm(1.0f32 - theta); /*  sin(theta) */

    if len >= 8 * stride {
        stride2 = 1;
        /* This is just a simple (equivalent) way of computing sqrt(len/stride) with rounding.
        It's basically incrementing long as (stride2+0.5)^2 < len/stride. */
        while (stride2 * stride2 + stride2) * stride + (stride >> 2) < len {
            stride2 += 1;
        }
    }
    /*NOTE: As a minor optimization, we could be passing around log2(B), not B, for both this and for
    extract_collapse_mask().*/
    len = celt_udiv(len as u32, stride as u32) as i32;
    for x in X[..(stride * len) as usize].chunks_exact_mut(len as usize) {
        if dir < 0 {
            if stride2 != 0 {
                exp_rotation1(x, len, stride2, s, c);
            }
            exp_rotation1(x, len, 1, c, s);
        } else {
            exp_rotation1(x, len, 1, c, -s);
            if stride2 != 0 {
                exp_rotation1(x, len, stride2, s, -c);
            }
        }
    }
}

/// Normalizes the decoded integer pvq codeword to unit norm.
fn normalise_residual(iy: &[i32], X: &mut [f32], N: i32, Ryy: f32, gain: f32) {
    let t = Ryy;
    let g = rsqrt(t) * gain;
    for (x, &y) in X[..N as usize].iter_mut().zip(&iy[..N as usize]) {
        *x = y as f32 * g;
    }
}

fn extract_collapse_mask(iy: &[i32], N: i32, B: i32) -> u32 {
    if B <= 1 {
        return 1;
    }
    /*NOTE: As a minor optimization, we could be passing around log2(B), not B, for both this and for
    exp_rotation().*/
    let N0 = celt_udiv(N as u32, B as u32) as usize;
    let mut collapse_mask: u32 = 0;
    for (i, block) in iy[..B as usize * N0].chunks_exact(N0).enumerate() {
        let tmp = block.iter().fold(0u32, |t, &v| t | v as u32);
        collapse_mask |= ((tmp != 0) as u32) << i;
    }
    collapse_mask
}

/// op_pvq_search_wasm. `_X` is not modified (unlike `op_pvq_search_c`). Like the C kernel,
/// `iy` needs room for N+3 entries (whole vectors of 4 are stored).
pub fn op_pvq_search(_X: &[f32], iy: &mut [i32], K: i32, N: i32) -> f32 {
    let n = N as usize;
    debug_assert!(N > 0 && _X.len() >= n && iy.len() >= (n + 3) & !3);
    let st = stack::save();
    /* X and y are padded with (at least) 8 entries for the search loop. */
    let y = st.alloc::<f32>(n + 8);
    let X = st.alloc::<f32>(n + 8);
    let signy = st.alloc::<i32>(n + 3);
    let (py, pX, ps, piy, p_X) = (
        y.as_mut_ptr(),
        X.as_mut_ptr(),
        signy.as_mut_ptr(),
        iy.as_mut_ptr(),
        _X.as_ptr(),
    );
    let mut xy: f32;
    let mut yy: f32;
    let mut pulsesLeft: i32;
    // SAFETY: all vector accesses stay within y[0..N+8], X[0..N+8], signy/iy[0..(N+3)&~3]
    // and _X[0..N], as in the C kernel.
    // UNSAFE-PERF: unchecked pointer loads/stores/indexing instead of bounds-checked slices. Nearly every
    // statement of the kernel touches memory through them, so narrowing the block would wrap each one.
    unsafe {
        /* Copy the last partial vector of _X into a zero-padded X; the full
        vectors are read from _X directly below. */
        let j0 = n & !3;
        store4(pX.add(j0), f32x4_splat(0.0));
        store4(pX.add(j0 + 4), f32x4_splat(0.0));
        for i in j0..n {
            *pX.add(i) = *p_X.add(i);
        }
        let mut sums = f32x4_splat(0.0);
        let mut j = 0usize;
        while j < n {
            let mut x4 = load4(if j + 4 <= n { p_X.add(j) } else { pX.add(j) });
            let s4 = f32x4_lt(x4, f32x4_splat(0.0));
            /* Get rid of the sign */
            x4 = f32x4_abs(x4);
            sums = f32x4_add(sums, x4);
            /* Clear y and iy in case we don't do the projection. */
            store4(py.add(j), f32x4_splat(0.0));
            store4(piy.add(j), i32x4_splat(0));
            store4(pX.add(j), x4);
            store4(ps.add(j), s4);
            j += 4;
        }

        xy = 0.0;
        yy = 0.0;

        pulsesLeft = K;

        /* Do a pre-search by projecting on the pyramid */
        if K > (N >> 1) {
            let mut sum: f32 = hsum_f32(sums);
            /* If X is too small, just replace it with a pulse at 0 */
            /* Prevents infinities and NaNs from causing too many pulses
            to be allocated. 64 is an approximation of infinity here. */
            if !(sum > EPSILON && sum < 64.0) {
                *pX = 1.0;
                let mut j = 1usize;
                loop {
                    *pX.add(j) = 0.0;
                    j += 1;
                    if j >= n {
                        break;
                    }
                }
                sum = 1.0;
            }
            /* Using K+e with e < 1 guarantees we cannot get more than K pulses. */
            let rcp4 = f32x4_splat((K as f32 + 0.8f32) * rcp(sum));
            let mut xy4 = f32x4_splat(0.0);
            let mut yy4 = f32x4_splat(0.0);
            let mut pulses_sum = i32x4_splat(0);
            let mut j = 0usize;
            while j < n {
                let x4 = load4(pX.add(j));
                /* x4 >= 0 here, so truncation is the same as the floor() in the C
                code. x4*rcp4 is finite, not NaN and at most K+.8, so the
                relaxed truncation is exact too. */
                let iy4 = trunc_i32(f32x4_mul(x4, rcp4));
                pulses_sum = i32x4_add(pulses_sum, iy4);
                store4(piy.add(j), iy4);
                let y4 = f32x4_convert_i32x4(iy4);
                xy4 = madd(x4, y4, xy4);
                yy4 = madd(y4, y4, yy4);
                /* double the y[] vector so we don't have to do it in the search loop. */
                store4(py.add(j), f32x4_add(y4, y4));
                j += 4;
            }
            pulsesLeft -= hsum_i32(pulses_sum);
            xy = hsum_f32(xy4);
            yy = hsum_f32(yy4);
        }
        /* The padding lanes get X=0 and a huge y, so their score is either 0
        (never selected, as only scores strictly above 0 are) or far below
        the score of any real position. */
        store4(py.add(n), f32x4_splat(1e15));
        store4(py.add(n + 4), f32x4_splat(1e15));
        debug_assert!(pulsesLeft >= 0);

        /* This should never happen, but just in case it does (e.g. on silence)
        we fill the first bin with pulses. */
        if pulsesLeft > N + 3 {
            let tmp = pulsesLeft as f32;
            yy = yy + tmp * tmp;
            yy = yy + tmp * *py;
            *piy += pulsesLeft;
            pulsesLeft = 0;
        }

        for _ in 0..pulsesLeft {
            /* The squared magnitude term gets added anyway, so we might as well
            add it outside the loop */
            yy = yy + 1.0;
            let xy4 = f32x4_splat(xy);
            let yy4 = f32x4_splat(yy);
            /* Two independent sets of per-lane maxima, to halve the length of the
            loop-carried dependency chain. */
            let mut best0 = f32x4_splat(0.0);
            let mut best1 = best0;
            let mut pos0 = i32x4_splat(0);
            let mut pos1 = pos0;
            let mut count = i32x4(0, 1, 2, 3);
            let mut j = 0usize;
            while j < n {
                /* Temporary sums of the new pulse(s) */
                let rxy0 = f32x4_add(load4(pX.add(j)), xy4);
                let rxy1 = f32x4_add(load4(pX.add(j + 4)), xy4);
                /* We're multiplying y[j] by two so we don't have to do it here */
                let ryy0 = f32x4_add(load4(py.add(j)), yy4);
                let ryy1 = f32x4_add(load4(py.add(j + 4)), yy4);
                /* We maximise Rxy/sqrt(Ryy), i.e. Rxy^2/Ryy since Rxy >= 0.
                WebAssembly has no reciprocal (square root) estimate, and one
                division is cheaper than a square root plus a division. */
                let r0 = f32x4_div(f32x4_mul(rxy0, rxy0), ryy0);
                let r1 = f32x4_div(f32x4_mul(rxy1, rxy1), ryy1);
                /* Keep the first index of each per-lane maximum. */
                pos0 = laneselect(count, pos0, f32x4_gt(r0, best0));
                count = i32x4_add(count, i32x4_splat(4));
                pos1 = laneselect(count, pos1, f32x4_gt(r1, best1));
                count = i32x4_add(count, i32x4_splat(4));
                best0 = max4(best0, r0);
                best1 = max4(best1, r1);
                j += 8;
            }
            /* Merge the two sets, keeping the lowest index on ties. */
            pos0 = laneselect(pos1, pos0, f32x4_gt(best1, best0));
            best0 = max4(best0, best1);
            /* Horizontal max */
            let max = hmax4(best0);
            /* Among the lanes that reach the global max, pick the lowest index,
            like the C code does. */
            pos0 = laneselect(pos0, i32x4_splat(0x7fffffff), f32x4_eq(best0, max));
            let best_id = hmin_i32(pos0) as usize;

            /* Updating the sums of the new pulse(s) */
            xy = xy + *pX.add(best_id);
            /* We're multiplying y[j] by two so we don't have to do it here */
            yy = yy + *py.add(best_id);

            /* Only now that we've made the final choice, update y/iy */
            /* Multiplying y[j] by 2 so we don't have to do it everywhere else */
            *py.add(best_id) += 2.0;
            *piy.add(best_id) += 1;
        }

        /* Put the original sign back */
        let mut j = 0usize;
        while j < n {
            let mut y4 = load4(piy.add(j));
            let s4 = load4(ps.add(j));
            y4 = v128_xor(i32x4_add(y4, s4), s4);
            store4(piy.add(j), y4);
            j += 4;
        }
    }
    yy
}

/// Algebraic pulse-vector quantiser. The signal x is replaced by the sum of
/// the pitch and a combination of pulses such that its norm is still equal
/// to 1. This is the function that will typically require the most CPU.
/// Returns a mask indicating which blocks in the band received pulses.
pub fn alg_quant(
    X: &mut [f32],
    N: i32,
    K: i32,
    spread: i32,
    B: i32,
    enc: &mut ec_enc,
    gain: f32,
    resynth: i32,
) -> u32 {
    let st = stack::save();

    debug_assert!(K > 0, "alg_quant() needs at least one pulse");
    debug_assert!(N > 1, "alg_quant() needs at least two dimensions");

    /* Covers vectorization by up to 4. */
    let iy = st.alloc::<i32>((N + 3) as usize);

    exp_rotation(X, N, 1, B, K, spread);

    let yy = op_pvq_search(X, iy, K, N);
    let collapse_mask = extract_collapse_mask(iy, N, B);
    encode_pulses(&iy[..N as usize], N, K, enc);
    if resynth != 0 {
        normalise_residual(iy, X, N, yy, gain);
    }

    if resynth != 0 {
        exp_rotation(X, N, -1, B, K, spread);
    }

    collapse_mask
}

pub fn renormalise_vector(X: &mut [f32], N: i32, gain: f32) {
    let E: f32 = EPSILON + celt_inner_prod(X, X, N);
    let t = E;
    let g: f32 = rsqrt(t) * gain;

    for x in &mut X[..N as usize] {
        *x = g * *x;
    }
    /*return celt_sqrt(E);*/
}

pub fn stereo_itheta(X: &[f32], Y: &[f32], stereo: i32, N: i32) -> i32 {
    let mut Emid: f32 = 0.0;
    let mut Eside: f32 = 0.0;
    if stereo != 0 {
        for (&x, &y) in X[..N as usize].iter().zip(&Y[..N as usize]) {
            let m = x + y;
            let s = x - y;
            Emid = Emid + m * m;
            Eside = Eside + s * s;
        }
    } else {
        Emid += celt_inner_prod(X, X, N);
        Eside += celt_inner_prod(Y, Y, N);
    }
    let mid = sqrt(Emid);
    let side = sqrt(Eside);
    floor64((0.5f32 + 65536.0f32 * 16384.0 * atan2p_norm(side, mid)) as f64) as i32
}
