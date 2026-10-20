//! Port of celt/celt.h and the encoder-side parts of celt/celt.c.

use super::arch::{COEF_ONE, IMAX};
use super::modes;
use super::pitch::comb_filter_const;

pub static trim_icdf: [u8; 11] = [126, 124, 119, 109, 87, 41, 19, 9, 4, 2, 0];
/* Probs: NONE: 21.875%, LIGHT: 6.25%, NORMAL: 65.625%, AGGRESSIVE: 6.25% */
pub static spread_icdf: [u8; 4] = [25, 23, 2, 0];
pub static tapset_icdf: [u8; 3] = [2, 1, 0];

pub const COMBFILTER_MAXPERIOD: i32 = 1024;
pub const COMBFILTER_MINPERIOD: i32 = 15;

/* celt.c. resampling_factor() is not needed (48 kHz only); comb_filter_const is the WASM
kernel in pitch.rs. */

/// Gains of the three comb filter tapsets (QCONST16(x,15) = x in float).
static gains: [[f32; 3]; 3] = [
    [0.3066406250, 0.2170410156, 0.1296386719],
    [0.4638671875, 0.2680664062, 0.0],
    [0.7998046875, 0.1000976562, 0.0],
];

/// Comb filter `y[i] = x[i] + taps` for i in 0..N, cross-fading from (T0,g0,tapset0) to
/// (T1,g1,tapset1) over `overlap` samples using `window`.
///
/// # Safety
/// `x` must have `max(T0,T1,15)+2` readable floats before it and `N` after it; `y` must have
/// `N` writable floats. `y == x` (in place) is allowed; the regions otherwise must not overlap.
pub unsafe fn comb_filter(
    y: *mut f32,
    x: *const f32,
    mut T0: i32,
    mut T1: i32,
    N: i32,
    g0: f32,
    g1: f32,
    tapset0: i32,
    tapset1: i32,
) {
    unsafe {
        if g0 == 0.0 && g1 == 0.0 {
            /* OPT: Happens to work without the OPUS_MOVE(), but only because the current encoder already copies x to y */
            if x != y as *const f32 {
                core::ptr::copy(x, y, N as usize);
            }
            return;
        }
        /* When the gain is zero, T0 and/or T1 is set to zero. We need
        to have then be at least 2 to avoid processing garbage data. */
        T0 = IMAX(T0, COMBFILTER_MINPERIOD);
        T1 = IMAX(T1, COMBFILTER_MINPERIOD);
        let [t00, t01, t02] = gains[tapset0 as usize];
        let [t10, t11, t12] = gains[tapset1 as usize];
        let (g00, g01, g02) = (g0 * t00, g0 * t01, g0 * t02);
        let (g10, g11, g12) = (g1 * t10, g1 * t11, g1 * t12);
        let (t0, t1) = (T0 as isize, T1 as isize);
        let mut x1 = *x.offset(-t1 + 1);
        let mut x2 = *x.offset(-t1);
        let mut x3 = *x.offset(-t1 - 1);
        let mut x4 = *x.offset(-t1 - 2);
        /* If the filter didn't change, we don't need the overlap */
        let overlap = if g0 == g1 && T0 == T1 && tapset0 == tapset1 { 0 } else { modes::overlap };
        let window = &modes::window;
        let mut i: isize = 0;
        while i < overlap as isize {
            let x0 = *x.offset(i - t1 + 2);
            let w = window[i as usize];
            let f: f32 = w * w;
            *y.offset(i) = *x.offset(i)
                + ((COEF_ONE - f) * g00) * *x.offset(i - t0)
                + ((COEF_ONE - f) * g01) * (*x.offset(i - t0 + 1) + *x.offset(i - t0 - 1))
                + ((COEF_ONE - f) * g02) * (*x.offset(i - t0 + 2) + *x.offset(i - t0 - 2))
                + (f * g10) * x2
                + (f * g11) * (x1 + x3)
                + (f * g12) * (x0 + x4);
            x4 = x3;
            x3 = x2;
            x2 = x1;
            x1 = x0;
            i += 1;
        }
        if g1 == 0.0 {
            /* OPT: Happens to work without the OPUS_MOVE(), but only because the current encoder already copies x to y */
            if x != y as *const f32 {
                core::ptr::copy(x.offset(overlap as isize), y.offset(overlap as isize), (N - overlap) as usize);
            }
            return;
        }

        /* Compute the part with the constant filter. */
        comb_filter_const(y.offset(i), x.offset(i), T1, N - i as i32, g10, g11, g12);
    }
}

/* The tf_select_table second index is 4*isTransient + 2*tf_select + per_band_flag; only the
20 ms (LM == 3) row exists. */
pub static tf_select_table: [i8; 8] = [0, -2, 0, -3, 3, 0, 1, -1];

/// init_caps for LM == 3; `cache_caps` only holds the rows of LM 3, indexed by `(C-1)*nbEBands + i`.
pub fn init_caps(cap: &mut [i32], C: i32) {
    const NB: usize = modes::nbEBands as usize;
    // C is 1 or 2: the mask only tells the compiler that the row is in range.
    let row = &modes::cache_caps[((C - 1) as usize & 1) * NB..][..NB];
    for (i, (cap, &cc)) in cap[..NB].iter_mut().zip(row).enumerate() {
        let N = (modes::eBands[i + 1] as i32 - modes::eBands[i] as i32) << 3;
        *cap = (cc as i32 + 64) * C * N >> 2;
    }
}
