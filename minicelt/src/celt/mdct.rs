//! Port of celt/mdct.h and celt/mdct.c (forward MDCT, float build, non-PFA).
//! The twiddles are planar: `t[i]` = cos, `t[N4+i]` = -sin.

use super::kiss_fft::{bitrev, kiss_fft_cpx, kiss_fft_state, opus_fft_impl};
use super::modes;
use super::static_modes_float::{fft_state48000_960_0, fft_state48000_960_3, mdct_twiddles960};
use crate::stack;

/// The overlap (the mode's 120, a multiple of 4), its half and quarter.
const OV: usize = modes::overlap as usize;
const OV2: usize = OV / 2;
const NQ: usize = OV / 4;
const _: () = assert!(OV % 4 == 0);

/// clt_mdct_forward_c for the two transform sizes 20 ms frames use (the `mdct_lookup` of the
/// 48 kHz mode is `n == 1920`, `maxshift == 3`): `shift == 0` for a long block (N = 1920) and
/// `shift == 3` for the 8 short blocks (N = 240). Overlap and window are those of the mode.
/// Forward MDCT (the C comment says it trashes the input array; it does not).
///
/// C passes pointers `in` and `out` into larger buffers; here `in_`/`out` are the whole buffers
/// and `in_off`/`out_off` the offsets of the C pointers, e.g. from compute_mdcts.
/// `out` is written at `out_off + stride*k`; `stride` is 1 for `shift == 0` and the 8 short blocks
/// for `shift == 3`.
///
/// The loops run in `fold_rotate` and `post_rotate`, specialised for the two sizes so that every
/// index has a constant bound (the slices are cut to their constant lengths once, no check in the
/// loops). The C fold writes the folded input to a buffer `f` that the pre-rotation then reads;
/// here each folded pair is rotated at once, so `f` does not exist.
pub fn clt_mdct_forward(
    in_: &[f32],
    in_off: usize,
    out: &mut [f32],
    out_off: usize,
    shift: i32,
    stride: i32,
) {
    let sg = stack::save();
    let bitrev_t = bitrev(shift);
    let x = &in_[in_off..];
    let out = &mut out[out_off..];
    debug_assert!(stride == if shift == 0 { 1 } else { 8 });
    if shift == 0 {
        let st = &fft_state48000_960_0;
        let f2 = sg.alloc::<kiss_fft_cpx>(512);
        fold_rotate::<960, 512>(x, f2, &mdct_twiddles960, bitrev_t, st.scale);
        fft_post::<960, 1>(st, f2, out, &mdct_twiddles960);
    } else {
        let st = &fft_state48000_960_3;
        let f2 = sg.alloc::<kiss_fft_cpx>(64);
        fold_rotate::<120, 64>(x, f2, &mdct_twiddles960[960..], bitrev_t, st.scale);
        fft_post::<120, 8>(st, f2, out, &mdct_twiddles960[960..]);
    }
}

/// Window, shuffle and fold `x[.. N2 + OV]` (four blocks [a, b, c, d]) into `N2 / 2` complex values,
/// pre-rotate them by `trig` and store them, scaled, at their bit-reversed position in `f2`.
/// `P` is the power of two at or above `N2 / 2` (the length of `f2`); the bit-reversal index is
/// masked with `P - 1`, a no-op since the table is a permutation of `0..N2 / 2`.
#[inline(always)]
fn fold_rotate<const N2: usize, const P: usize>(
    x: &[f32],
    f2: &mut [kiss_fft_cpx],
    trig: &[f32],
    bitrev: &[i16],
    scale: f32,
) {
    let n4 = N2 / 2;
    let x = &x[..N2 + OV];
    let w = &modes::window;
    let f2 = &mut f2[..P];
    let bitrev = &bitrev[..n4];
    let (tc, ts) = (&trig[..n4], &trig[n4..2 * n4]);
    let mut put = |i: usize, re: f32, im: f32| {
        let (t0, t1) = (tc[i], ts[i]);
        let yr = re * t0 - im * t1;
        let yi = im * t0 + re * t1;
        debug_assert!((bitrev[i] as usize) < n4);
        /* For floating-point it is best to scale after the rotation (no QEXT). */
        f2[bitrev[i] as usize & (P - 1)] = kiss_fft_cpx {
            r: yr * scale,
            i: yi * scale,
        };
    };
    /* The C xp1 = OV2 + 2i, xp2 = N2 - 1 + OV2 - 2i and, in the first NQ pairs, wp1 = OV2 + 2i and
    wp2 = OV2 - 1 - 2i, in the last NQ pairs wp1 = 2t and wp2 = OV - 1 - 2t (t = i - (n4 - NQ)). */
    for i in 0..NQ {
        let (w1, w2) = (w[OV2 + 2 * i], w[OV2 - 1 - 2 * i]);
        /* Real part arranged as -d-cR, Imag part arranged as -b+aR*/
        let re = x[OV2 + 2 * i + N2] * w2 + x[N2 - 1 + OV2 - 2 * i] * w1;
        let im = x[OV2 + 2 * i] * w1 - x[N2 - 1 + OV2 - 2 * i - N2] * w2;
        put(i, re, im);
    }
    for i in NQ..n4 - NQ {
        /* Real part arranged as a-bR, Imag part arranged as -c-dR */
        put(i, x[N2 - 1 + OV2 - 2 * i], x[OV2 + 2 * i]);
    }
    for i in n4 - NQ..n4 {
        let t = i - (n4 - NQ);
        let (w1, w2) = (w[2 * t], w[OV - 1 - 2 * t]);
        /* Real part arranged as a-bR, Imag part arranged as -c-dR */
        let re = -(x[OV2 + 2 * i - N2] * w1) + x[N2 - 1 + OV2 - 2 * i] * w2;
        let im = x[OV2 + 2 * i] * w2 + x[N2 - 1 + OV2 - 2 * i + N2] * w1;
        put(i, re, im);
    }
}

/// The N/4 complex FFT of `f2[.. N2 / 2]` (does not downscale anymore), then the post-rotation:
/// writes `out[2 * S * i]` and `out[S * (N2 - 1 - 2 * i)]` for `i` in `0..N2 / 2` (output stride `S`).
#[inline(always)]
fn fft_post<const N2: usize, const S: usize>(
    st: &kiss_fft_state,
    f2: &mut [kiss_fft_cpx],
    out: &mut [f32],
    trig: &[f32],
) {
    let n4 = N2 / 2;
    let f2 = &mut f2[..n4];
    opus_fft_impl(st, f2);
    let out = &mut out[..S * (N2 - 1) + 1];
    let (tc, ts) = (&trig[..n4], &trig[n4..2 * n4]);
    for i in 0..n4 {
        let (t0, t1, fp) = (tc[i], ts[i], f2[i]);
        out[2 * S * i] = fp.i * t1 - fp.r * t0;
        out[S * (N2 - 1 - 2 * i)] = fp.r * t1 + fp.i * t0;
    }
}
