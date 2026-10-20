//! Port of celt/mdct.h and celt/mdct.c (forward MDCT, float build, non-PFA).
//! The twiddles are planar: `t[i]` = cos, `t[N4+i]` = -sin.

use super::arch::Ix;
use super::kiss_fft::{bitrev, kiss_fft_cpx, kiss_fft_state, opus_fft_impl};
use super::modes;
use super::static_modes_float::{fft_state48000_960_0, fft_state48000_960_3, mdct_twiddles960};
use crate::stack;

/// clt_mdct_forward_c for the two transform sizes 20 ms frames use (the `mdct_lookup` of the
/// 48 kHz mode is `n == 1920`, `maxshift == 3`): `shift == 0` for a long block (N = 1920) and
/// `shift == 3` for the 8 short blocks (N = 240). Overlap and window are those of the mode.
/// Forward MDCT (the C comment says it trashes the input array; it does not).
///
/// C passes pointers `in` and `out` into larger buffers; here `in_`/`out` are the whole buffers
/// and `in_off`/`out_off` the offsets of the C pointers, e.g. from compute_mdcts.
/// `out` is written at `out_off + stride*k`.
// IX-HARD: the first three loops index `in_` and `window` through the running `xp1`/`xp2`/`wp1`/`wp2`
// pointers (signed counters cast to `usize`), per-sample and with no length to prove them against. The
// `trig`/`f`/`f2`/`bitrev_t`/`out` loops over `0..N4` (a re-slice would do) are left on `Ix` too, since
// the function keeps `Ix` for the first loops; not measured.
pub fn clt_mdct_forward(in_: &[f32], in_off: usize, out: &mut [f32], out_off: usize, shift: i32, stride: i32) {
    let window = &modes::window;
    let overlap = modes::overlap;
    /* trig has the tables of shift 0 (960 values) and shift 3 (120 values) */
    let (st, N, trig): (&kiss_fft_state, i32, &[f32]) = if shift == 0 {
        (&fft_state48000_960_0, 1920, &mdct_twiddles960)
    } else {
        (&fft_state48000_960_3, 240, mdct_twiddles960.rg(960..))
    };
    let scale = st.scale;
    let bitrev_t = bitrev(shift);
    let N2 = (N >> 1) as usize;
    let N4 = (N >> 2) as usize;

    let sg = stack::save();
    let f = sg.alloc::<f32>(N2);
    let f2 = sg.alloc::<kiss_fft_cpx>(N4);

    let ov2 = (overlap >> 1) as usize;
    let nq = ((overlap + 3) >> 2) as usize;

    /* Consider the input to be composed of four blocks: [a, b, c, d] */
    /* Window, shuffle, fold */
    {
        /* xp2 steps downwards by 2 and may end one below the buffer start, so use signed indices. */
        let base = in_off as isize;
        let n2 = N2 as isize;
        let mut xp1: isize = base + ov2 as isize;
        let mut xp2: isize = base + n2 - 1 + ov2 as isize;
        let mut yp = 0usize;
        let mut wp1 = ov2;
        let mut wp2 = ov2 as isize - 1;
        let mut i = 0usize;
        while i < nq {
            /* Real part arranged as -d-cR, Imag part arranged as -b+aR*/
            *f.at_mut(yp) = in_.at((xp1 + n2) as usize) * window.at(wp2 as usize) + in_.at(xp2 as usize) * window.at(wp1);
            *f.at_mut(yp + 1) = in_.at(xp1 as usize) * window.at(wp1) - in_.at((xp2 - n2) as usize) * window.at(wp2 as usize);
            yp += 2;
            xp1 += 2;
            xp2 -= 2;
            wp1 += 2;
            wp2 -= 2;
            i += 1;
        }
        wp1 = 0;
        wp2 = overlap as isize - 1;
        while i < N4 - nq {
            /* Real part arranged as a-bR, Imag part arranged as -c-dR */
            *f.at_mut(yp) = in_.at(xp2 as usize);
            *f.at_mut(yp + 1) = in_.at(xp1 as usize);
            yp += 2;
            xp1 += 2;
            xp2 -= 2;
            i += 1;
        }
        while i < N4 {
            /* Real part arranged as a-bR, Imag part arranged as -c-dR */
            *f.at_mut(yp) = -(in_.at((xp1 - n2) as usize) * window.at(wp1)) + in_.at(xp2 as usize) * window.at(wp2 as usize);
            *f.at_mut(yp + 1) = in_.at(xp1 as usize) * window.at(wp2 as usize) + in_.at((xp2 + n2) as usize) * window.at(wp1);
            yp += 2;
            xp1 += 2;
            xp2 -= 2;
            wp1 += 2;
            wp2 -= 2;
            i += 1;
        }
    }
    /* Pre-rotation */
    for i in 0..N4 {
        let t0 = trig.at(i);
        let t1 = trig.at(N4 + i);
        let re = f.at(2 * i);
        let im = f.at(2 * i + 1);
        let yr = re * t0 - im * t1;
        let yi = im * t0 + re * t1;
        /* For floating-point it is best to scale after the rotation (no QEXT). */
        *f2.at_mut(bitrev_t.at(i) as usize) = kiss_fft_cpx { r: yr * scale, i: yi * scale };
    }

    /* N/4 complex FFT, does not downscale anymore */
    opus_fft_impl(st, f2);

    /* Post-rotate */
    {
        let mut yp1 = out_off;
        let mut yp2 = out_off + stride as usize * (N2 - 1);
        let stride2 = 2 * stride as usize;
        for i in 0..N4 {
            let t0 = trig.at(i);
            let t1 = trig.at(N4 + i);
            let fp = f2.at(i);
            let yr = fp.i * t1 - fp.r * t0;
            let yi = fp.r * t1 + fp.i * t0;
            *out.at_mut(yp1) = yr;
            *out.at_mut(yp2) = yi;
            yp1 += stride2;
            /* the last step may leave yp2 below out_off */
            yp2 = yp2.wrapping_sub(stride2);
        }
    }
}
