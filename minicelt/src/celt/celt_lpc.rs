//! Port of celt/celt_lpc.c (float build): `_celt_lpc` and `_celt_autocorr` only.

use super::pitch::celt_pitch_xcorr;

/// `_lpc`: out, `[0..p-1]` LPC coefficients; `ac`: in, `[0..p]` autocorrelation values.
pub fn _celt_lpc(_lpc: &mut [f32], ac: &[f32], p: i32) {
    let p = p as usize;
    let lpc = &mut _lpc[..p];
    let ac = &ac[..p + 1];
    let mut error: f32 = ac[0];

    lpc.fill(0.0);
    if ac[0] > 1e-10 {
        for i in 0..p {
            /* Sum up this iteration's reflection coefficient */
            let mut rr: f32 = 0.0;
            for j in 0..i {
                rr += lpc[j] * ac[i - j];
            }
            rr += ac[i + 1];
            let r: f32 = -(rr / error);
            /*  Update LPC coefficients and total error */
            lpc[i] = r;
            for j in 0..(i + 1) >> 1 {
                let tmp1 = lpc[j];
                let tmp2 = lpc[i - 1 - j];
                lpc[j] = tmp1 + r * tmp2;
                lpc[i - 1 - j] = tmp2 + r * tmp1;
            }

            error = error - (r * r) * error;
            /* Bail out once we get 30 dB gain */
            if error <= 0.001 * ac[0] {
                break;
            }
        }
    }
}

/// `x`: in, `[0..n-1]` samples; `ac`: out, `[0..lag]` autocorrelation values (`lag+1` entries).
/// (The only caller, pitch_downsample, has `overlap == 0`: no windowing copy.)
pub fn _celt_autocorr(x: &[f32], ac: &mut [f32], lag: i32, n: i32) {
    let fastN = n - lag;
    debug_assert!(n > 0);
    let (nu, lagu) = (n as usize, lag as usize);
    debug_assert!(x.len() >= nu && ac.len() > lagu);
    celt_pitch_xcorr(x, x, ac, fastN, lag + 1);
    for k in 0..=lagu {
        let mut d: f32 = 0.0;
        for (a, b) in x[k + fastN as usize..nu].iter().zip(&x[fastN as usize..nu - k]) {
            d = d + a * b;
        }
        ac[k] += d;
    }
}
