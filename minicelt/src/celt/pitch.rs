//! Port of celt/pitch.c and the float kernels of celt/wasm/pitch_wasm.c
//! (opus_val16 = opus_val32 = f32).

use crate::celt::mathops::{fmax, imax};
use core::arch::wasm32::*;

use super::celt_lpc::{_celt_autocorr, _celt_lpc};
use super::mathops::sqrt;
use crate::stack;

/* Multiply-add a*b+c. With relaxed SIMD this may be fused (single rounding),
which is fine for these float kernels: like the SSE/NEON versions they do
not match the C summation order anyway. */
#[cfg(target_feature = "relaxed-simd")]
#[inline(always)]
fn WASM_F32X4_MADD(a: v128, b: v128, c: v128) -> v128 {
    f32x4_relaxed_madd(a, b, c)
}
#[cfg(not(target_feature = "relaxed-simd"))]
#[inline(always)]
fn WASM_F32X4_MADD(a: v128, b: v128, c: v128) -> v128 {
    f32x4_add(f32x4_mul(a, b), c)
}

/// Unaligned 4-float load. The caller guarantees `p[0..4]` is readable.
// UNSAFE-MAND: `v128_load` is an unsafe intrinsic (a safe load would need a bounds-checked copy or `f32x4(..)`).
#[inline(always)]
unsafe fn ld(p: *const f32) -> v128 {
    unsafe { v128_load(p as *const v128) }
}

/* Sums the lanes of a, b, c and d: returns {sum(a), sum(b), sum(c), sum(d)}. */
#[inline(always)]
fn wasm_f32x4_hadd4(a: v128, b: v128, c: v128, d: v128) -> v128 {
    let ab = f32x4_add(
        i32x4_shuffle::<0, 4, 1, 5>(a, b),
        i32x4_shuffle::<2, 6, 3, 7>(a, b),
    );
    let cd = f32x4_add(
        i32x4_shuffle::<0, 4, 1, 5>(c, d),
        i32x4_shuffle::<2, 6, 3, 7>(c, d),
    );
    f32x4_add(i64x2_shuffle::<0, 2>(ab, cd), i64x2_shuffle::<1, 3>(ab, cd))
}

#[inline(always)]
fn wasm_f32x4_hsum(a: v128) -> f32 {
    let a = f32x4_add(a, i64x2_shuffle::<1, 0>(a, a));
    let a = f32x4_add(a, i32x4_shuffle::<1, 0, 3, 2>(a, a));
    f32x4_extract_lane::<0>(a)
}

pub fn celt_inner_prod(x: &[f32], y: &[f32], N: i32) -> f32 {
    let n = imax(N, 0) as usize;
    debug_assert!(x.len() >= n && y.len() >= n);
    let (px, py) = (x.as_ptr(), y.as_ptr());
    // SAFETY: every vector load reads 4 floats at an index i with i + 4 <= n.
    let mut i = 0usize;
    let mut sum0 = f32x4_splat(0.0);
    let mut sum1 = f32x4_splat(0.0);
    // UNSAFE-PERF: unchecked pointer loads instead of bounds-checked slices.
    unsafe {
        while i + 7 < n {
            sum0 = WASM_F32X4_MADD(ld(px.add(i)), ld(py.add(i)), sum0);
            sum1 = WASM_F32X4_MADD(ld(px.add(i + 4)), ld(py.add(i + 4)), sum1);
            i += 8;
        }
        if i + 3 < n {
            sum0 = WASM_F32X4_MADD(ld(px.add(i)), ld(py.add(i)), sum0);
            i += 4;
        }
    }
    let mut xy = wasm_f32x4_hsum(f32x4_add(sum0, sum1));
    for (a, b) in x[i..n].iter().zip(&y[i..n]) {
        xy = xy + a * b;
    }
    xy
}

/* Returns {sum_j x[j]*y[j+k], k=0..3}. Reads y[0..len+2]. */
#[inline(always)]
// UNSAFE-PERF: unchecked pointer loads instead of bounds-checked slices.
unsafe fn xcorr4_float(x: *const f32, y: *const f32, len: usize) -> v128 {
    unsafe {
        let mut sum0 = f32x4_splat(0.0);
        let mut sum1 = sum0;
        let mut sum2 = sum0;
        let mut sum3 = sum0;
        /* Same lag-per-lane layout as the C kernel and SSE port: each x[j] is
        broadcast and multiplied with y[j..j+3]. */
        let mut j = 0usize;
        while j + 3 < len {
            let xj = ld(x.add(j));
            sum0 = WASM_F32X4_MADD(i32x4_shuffle::<0, 0, 0, 0>(xj, xj), ld(y.add(j)), sum0);
            sum1 = WASM_F32X4_MADD(i32x4_shuffle::<1, 1, 1, 1>(xj, xj), ld(y.add(j + 1)), sum1);
            sum2 = WASM_F32X4_MADD(i32x4_shuffle::<2, 2, 2, 2>(xj, xj), ld(y.add(j + 2)), sum2);
            sum3 = WASM_F32X4_MADD(i32x4_shuffle::<3, 3, 3, 3>(xj, xj), ld(y.add(j + 3)), sum3);
            j += 4;
        }
        let mut sum = f32x4_add(f32x4_add(sum0, sum1), f32x4_add(sum2, sum3));
        while j < len {
            sum = WASM_F32X4_MADD(f32x4_splat(*x.add(j)), ld(y.add(j)), sum);
            j += 1;
        }
        sum
    }
}

/// `xcorr[i] = sum_j _x[j] * _y[i + j]` for i in 0..max_pitch. Reads `_y[0..len + max_pitch - 1]`.
pub fn celt_pitch_xcorr(_x: &[f32], _y: &[f32], xcorr: &mut [f32], len: i32, max_pitch: i32) {
    debug_assert!(max_pitch > 0);
    let (len, max_pitch) = (imax(len, 0) as usize, imax(max_pitch, 0) as usize);
    let xcorr = &mut xcorr[..max_pitch];
    debug_assert!(_x.len() >= len);
    debug_assert!(_y.len() + 1 >= len + max_pitch);
    let (px, py, po) = (_x.as_ptr(), _y.as_ptr(), xcorr.as_mut_ptr());
    // SAFETY: the C kernel's accesses stay within _x[0..len], _y[0..len+max_pitch-1] and
    // xcorr[0..max_pitch]; checked by the asserts above.
    // UNSAFE-PERF: unchecked pointer loads/stores instead of bounds-checked slices.
    unsafe {
        let mut i = 0usize;
        /* Eight lags at a time, like the AVX2 version: one partial inner product
        per lag (no broadcasts in the inner loop), then a transpose-add. */
        while i + 7 < max_pitch {
            let y = py.add(i);
            let mut sum0 = f32x4_splat(0.0);
            let mut sum1 = sum0;
            let mut sum2 = sum0;
            let mut sum3 = sum0;
            let mut sum4 = sum0;
            let mut sum5 = sum0;
            let mut sum6 = sum0;
            let mut sum7 = sum0;
            let mut j = 0usize;
            while j + 3 < len {
                let xj = ld(px.add(j));
                sum0 = WASM_F32X4_MADD(xj, ld(y.add(j)), sum0);
                sum1 = WASM_F32X4_MADD(xj, ld(y.add(j + 1)), sum1);
                sum2 = WASM_F32X4_MADD(xj, ld(y.add(j + 2)), sum2);
                sum3 = WASM_F32X4_MADD(xj, ld(y.add(j + 3)), sum3);
                sum4 = WASM_F32X4_MADD(xj, ld(y.add(j + 4)), sum4);
                sum5 = WASM_F32X4_MADD(xj, ld(y.add(j + 5)), sum5);
                sum6 = WASM_F32X4_MADD(xj, ld(y.add(j + 6)), sum6);
                sum7 = WASM_F32X4_MADD(xj, ld(y.add(j + 7)), sum7);
                j += 4;
            }
            let mut lo = wasm_f32x4_hadd4(sum0, sum1, sum2, sum3);
            let mut hi = wasm_f32x4_hadd4(sum4, sum5, sum6, sum7);
            while j < len {
                let xj = f32x4_splat(*px.add(j));
                lo = WASM_F32X4_MADD(xj, ld(y.add(j)), lo);
                hi = WASM_F32X4_MADD(xj, ld(y.add(j + 4)), hi);
                j += 1;
            }
            v128_store(po.add(i) as *mut v128, lo);
            v128_store(po.add(i + 4) as *mut v128, hi);
            i += 8;
        }
        if i + 3 < max_pitch {
            v128_store(po.add(i) as *mut v128, xcorr4_float(px, py.add(i), len));
            i += 4;
        }
        /* In case max_pitch isn't a multiple of 4, do non-unrolled version. */
        while i < max_pitch {
            xcorr[i] = celt_inner_prod(_x, &_y[i..], len as i32);
            i += 1;
        }
    }
}

/// `dual_inner_prod_wasm`: `*xy1 = sum x[i]*y01[i]`, `*xy2 = sum x[i]*y02[i]`, i in 0..N.
pub fn dual_inner_prod(x: &[f32], y01: &[f32], y02: &[f32], N: i32, xy1: &mut f32, xy2: &mut f32) {
    let n = imax(N, 0) as usize;
    debug_assert!(x.len() >= n && y01.len() >= n && y02.len() >= n);
    let (px, p1, p2) = (x.as_ptr(), y01.as_ptr(), y02.as_ptr());
    // SAFETY: every vector load reads 4 floats at an index i with i + 4 <= n.
    let mut i = 0usize;
    let mut sum1a = f32x4_splat(0.0);
    let mut sum1b = sum1a;
    let mut sum2a = sum1a;
    let mut sum2b = sum1a;
    // UNSAFE-PERF: unchecked pointer loads instead of bounds-checked slices.
    unsafe {
        while i + 7 < n {
            let xi = ld(px.add(i));
            let xi4 = ld(px.add(i + 4));
            sum1a = WASM_F32X4_MADD(xi, ld(p1.add(i)), sum1a);
            sum2a = WASM_F32X4_MADD(xi, ld(p2.add(i)), sum2a);
            sum1b = WASM_F32X4_MADD(xi4, ld(p1.add(i + 4)), sum1b);
            sum2b = WASM_F32X4_MADD(xi4, ld(p2.add(i + 4)), sum2b);
            i += 8;
        }
        if i + 3 < n {
            let xi = ld(px.add(i));
            sum1a = WASM_F32X4_MADD(xi, ld(p1.add(i)), sum1a);
            sum2a = WASM_F32X4_MADD(xi, ld(p2.add(i)), sum2a);
            i += 4;
        }
    }
    let mut xy01 = wasm_f32x4_hsum(f32x4_add(sum1a, sum1b));
    let mut xy02 = wasm_f32x4_hsum(f32x4_add(sum2a, sum2b));
    let (xt, y1t, y2t) = (&x[i..n], &y01[i..n], &y02[i..n]);
    for j in 0..xt.len() {
        xy01 = xy01 + xt[j] * y1t[j];
        xy02 = xy02 + xt[j] * y2t[j];
    }
    *xy1 = xy01;
    *xy2 = xy02;
}

/// `comb_filter_const_wasm`: y[i] = x[i] + g10*x[i-T] + g11*(x[i-T+1]+x[i-T-1])
/// + g12*(x[i-T+2]+x[i-T-2]) for i in 0..N.
///
/// # Safety
/// `x` must have `T+2` readable floats before it and `N` (plus up to 3 for the vector
/// tail-free loads: reads reach `x[N-T+1]`, `x[N-1]`) after it; `y` must have `N` writable
/// floats. `y == x` (in place) is allowed, with `T >= COMBFILTER_MINPERIOD`, as in C.
// UNSAFE-PERF: raw pointers with negative offsets (and in-place `y == x`) instead of bounds-checked slices.
pub unsafe fn comb_filter_const(
    y: *mut f32,
    x: *const f32,
    T: i32,
    N: i32,
    g10: f32,
    g11: f32,
    g12: f32,
) {
    // UNSAFE-PERF: see the function comment.
    unsafe {
        let T = T as isize;
        let mut i: isize = 0;
        /* T >= COMBFILTER_MINPERIOD, so when filtering in place (x == y) every
        x value read below has either not been written yet or was written by
        an earlier iteration, exactly as in the C code. */
        let g10v = f32x4_splat(g10);
        let g11v = f32x4_splat(g11);
        let g12v = f32x4_splat(g12);
        let mut x0v = ld(x.offset(-T - 2));
        let n = N as isize;
        while i < n - 3 {
            let mut yi = ld(x.offset(i));
            let x4v = ld(x.offset(i - T + 2));
            let x1v = i32x4_shuffle::<1, 2, 3, 4>(x0v, x4v);
            let x2v = i32x4_shuffle::<2, 3, 4, 5>(x0v, x4v);
            let x3v = i32x4_shuffle::<3, 4, 5, 6>(x0v, x4v);
            #[cfg(target_feature = "relaxed-simd")]
            {
                yi = f32x4_relaxed_madd(g10v, x2v, yi);
                yi = f32x4_relaxed_madd(g11v, f32x4_add(x3v, x1v), yi);
                yi = f32x4_relaxed_madd(g12v, f32x4_add(x4v, x0v), yi);
            }
            #[cfg(not(target_feature = "relaxed-simd"))]
            {
                /* Use partial sums, like the SSE version. */
                yi = f32x4_add(yi, f32x4_mul(g10v, x2v));
                yi = f32x4_add(
                    yi,
                    f32x4_add(
                        f32x4_mul(g11v, f32x4_add(x3v, x1v)),
                        f32x4_mul(g12v, f32x4_add(x4v, x0v)),
                    ),
                );
            }
            x0v = x4v;
            v128_store(y.offset(i) as *mut v128, yi);
            i += 4;
        }
        while i < n {
            *y.offset(i) = *x.offset(i)
                + g10 * *x.offset(i - T)
                + g11 * (*x.offset(i - T + 1) + *x.offset(i - T - 1))
                + g12 * (*x.offset(i - T + 2) + *x.offset(i - T - 2));
            i += 1;
        }
    }
}

/// Finds the two best pitch candidates (by xcorr^2 / energy) into `best_pitch`.
fn find_best_pitch(xcorr: &[f32], y: &[f32], len: i32, max_pitch: i32, best_pitch: &mut [i32; 2]) {
    let (len, max_pitch) = (len as usize, max_pitch as usize);
    let xcorr = &xcorr[..max_pitch];
    let y = &y[..max_pitch + len];
    let mut Syy: f32 = 1.0;
    let mut best_num: [f32; 2] = [-1.0, -1.0];
    let mut best_den: [f32; 2] = [0.0, 0.0];
    *best_pitch = [0, 1];
    for j in 0..len {
        Syy = Syy + y[j] * y[j];
    }
    for i in 0..max_pitch {
        if xcorr[i] > 0.0 {
            let mut xcorr16: f32 = xcorr[i];
            /* Considering the range of xcorr16, this should avoid both underflows
            and overflows (inf) when squaring xcorr16 */
            xcorr16 *= 1e-12;
            let num: f32 = xcorr16 * xcorr16;
            if num * best_den[1] > best_num[1] * Syy {
                if num * best_den[0] > best_num[0] * Syy {
                    best_num[1] = best_num[0];
                    best_den[1] = best_den[0];
                    best_pitch[1] = best_pitch[0];
                    best_num[0] = num;
                    best_den[0] = Syy;
                    best_pitch[0] = i as i32;
                } else {
                    best_num[1] = num;
                    best_den[1] = Syy;
                    best_pitch[1] = i as i32;
                }
            }
        }
        Syy += y[i + len] * y[i + len] - y[i] * y[i];
        Syy = fmax(1.0, Syy);
    }
}

fn celt_fir5(x: &mut [f32], num: &[f32; 5], N: i32) {
    let [num0, num1, num2, num3, num4] = *num;
    let (mut mem0, mut mem1, mut mem2, mut mem3, mut mem4) =
        (0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32);
    for v in &mut x[..N as usize] {
        let mut sum: f32 = *v;
        sum = sum + num0 * mem0;
        sum = sum + num1 * mem1;
        sum = sum + num2 * mem2;
        sum = sum + num3 * mem3;
        sum = sum + num4 * mem4;
        mem4 = mem3;
        mem3 = mem2;
        mem2 = mem1;
        mem1 = mem0;
        mem0 = *v;
        *v = sum;
    }
}

/// The `C` channel buffers `x` of the C code are `[prefilter_mem | new samples]` of each channel
/// (`COMBFILTER_MAXPERIOD` history samples, then `len*factor - COMBFILTER_MAXPERIOD` new ones),
/// read in place: `hist[c*COMBFILTER_MAXPERIOD + j]` for `j < COMBFILTER_MAXPERIOD`, else
/// `new[new_off + c*new_stride + j - COMBFILTER_MAXPERIOD]`. `x_lp` receives `len` downsampled,
/// whitened samples. The decimation factor is 2 (the C `factor` argument, fixed for the one caller).
pub fn pitch_downsample_factor2(
    hist: &[f32],
    new: &[f32],
    new_off: usize,
    new_stride: usize,
    x_lp: &mut [f32],
    len: i32,
    C: i32,
) {
    debug_assert!(C == 1 || C == 2);
    const MP: usize = 1024; /* COMBFILTER_MAXPERIOD */
    let mut ac = [0.0f32; 5];
    let mut tmp: f32 = 1.0;
    let mut lpc = [0.0f32; 4];
    let mut lpc2 = [0.0f32; 5];
    let c1: f32 = 0.8;
    let n = len as usize;
    let x_lp = &mut x_lp[..n];
    for c in 0..C as usize {
        /* x = [hist | new] of channel c, read as the pairs (x[2i], x[2i+1]) with x[2i-1] carried along:
        x_lp[i] = .25*x[2i-1] + .25*x[2i+1] + .5*x[2i], x_lp[0] = .25*x[1] + .5*x[0];
        channel 0 sets x_lp, channel 1 adds to it */
        let (h, _) = hist[c * MP..][..MP].as_chunks::<2>();
        let (nw, _) = new[new_off + c * new_stride..][..2 * n - MP].as_chunks::<2>();
        let put = |d: &mut f32, v: f32| if c == 0 { *d = v } else { *d += v };
        let (d0, rest) = x_lp.split_first_mut().unwrap();
        let [a, b] = h[0];
        put(d0, 0.25 * b + 0.5 * a);
        let mut prev = b;
        for (d, &[a, b]) in rest.iter_mut().zip(h[1..].iter().chain(nw)) {
            put(d, 0.25 * prev + 0.25 * b + 0.5 * a);
            prev = b;
        }
    }
    _celt_autocorr(x_lp, &mut ac, 4, len);

    /* Noise floor -40 dB */
    ac[0] *= 1.0001;
    /* Lag windowing */
    for i in 1..=4usize {
        /*ac[i] *= exp(-.5*(2*M_PI*.002*i)*(2*M_PI*.002*i));*/
        ac[i] -= ac[i] * (0.008 * i as f32) * (0.008 * i as f32);
    }

    _celt_lpc(&mut lpc, &ac, 4);
    for i in 0..4 {
        tmp = 0.9 * tmp;
        lpc[i] = lpc[i] * tmp;
    }
    /* Add a zero */
    lpc2[0] = lpc[0] + 0.8;
    lpc2[1] = lpc[1] + c1 * lpc[0];
    lpc2[2] = lpc[2] + c1 * lpc[1];
    lpc2[3] = lpc[3] + c1 * lpc[2];
    lpc2[4] = c1 * lpc[3];
    celt_fir5(x_lp, &lpc2, len);
}

/// Reads `x_lp[0..len/2]`, `y[0..len/2 + max_pitch/2]` (and the decimated copies of them).
pub fn pitch_search(x_lp: &[f32], y: &[f32], len: i32, max_pitch: i32, pitch: &mut i32) {
    let mut best_pitch: [i32; 2] = [0, 0];
    let st = stack::save();

    debug_assert!(len > 0);
    debug_assert!(max_pitch > 0);
    let lag = len + max_pitch;

    let x_lp4 = st.alloc::<f32>((len >> 2) as usize);
    let y_lp4 = st.alloc::<f32>((lag >> 2) as usize);
    let xcorr = st.alloc::<f32>((max_pitch >> 1) as usize);

    /* Downsample by 2 again: the even samples, from pairs (one length check per copy, none in the loops) */
    let (x_pairs, _) = x_lp.as_chunks::<2>();
    for (d, p) in x_lp4.iter_mut().zip(&x_pairs[..(len >> 2) as usize]) {
        *d = p[0];
    }
    let (y_pairs, _) = y.as_chunks::<2>();
    for (d, p) in y_lp4.iter_mut().zip(&y_pairs[..(lag >> 2) as usize]) {
        *d = p[0];
    }

    /* Coarse search with 4x decimation */
    celt_pitch_xcorr(x_lp4, y_lp4, xcorr, len >> 2, max_pitch >> 2);

    find_best_pitch(xcorr, y_lp4, len >> 2, max_pitch >> 2, &mut best_pitch);

    /* Finer search with 2x decimation */
    for (i, xc) in xcorr.iter_mut().enumerate() {
        *xc = 0.0;
        let i = i as i32;
        if (i - 2 * best_pitch[0]).abs() > 2 && (i - 2 * best_pitch[1]).abs() > 2 {
            continue;
        }
        let sum = celt_inner_prod(x_lp, &y[i as usize..], len >> 1);
        *xc = fmax(-1.0, sum);
    }
    find_best_pitch(xcorr, y, len >> 1, max_pitch >> 1, &mut best_pitch);

    /* Refine by pseudo-interpolation */
    let offset: i32;
    if best_pitch[0] > 0 && best_pitch[0] < (max_pitch >> 1) - 1 {
        let b0 = best_pitch[0] as usize;
        let &[a, b, c] = xcorr[b0 - 1..].first_chunk::<3>().unwrap();
        if (c - a) > 0.7 * (b - a) {
            offset = 1;
        } else if (a - c) > 0.7 * (b - c) {
            offset = -1;
        } else {
            offset = 0;
        }
    } else {
        offset = 0;
    }
    *pitch = 2 * best_pitch[0] - offset;
}

fn compute_pitch_gain(xy: f32, xx: f32, yy: f32) -> f32 {
    xy / sqrt(1.0 + xx * yy)
}

static second_check: [i32; 16] = [0, 0, 3, 2, 3, 2, 5, 2, 3, 2, 3, 2, 5, 2, 3, 2];

/// `x` is the start of the buffer; C's `x += maxperiod` (halved) is applied internally, and
/// the function reads `x[0 .. maxperiod/2 + N/2]` (no data before the buffer start).
pub fn remove_doubling(
    x: &[f32],
    mut maxperiod: i32,
    mut minperiod: i32,
    mut N: i32,
    T0_: &mut i32,
    mut prev_period: i32,
    prev_gain: f32,
) -> f32 {
    let mut xx = 0.0f32;
    let mut xy = 0.0f32;
    let mut xy2 = 0.0f32;
    let mut xcorr = [0.0f32; 3];
    let offset: i32;
    let st = stack::save();

    let minperiod0 = minperiod;
    maxperiod /= 2;
    minperiod /= 2;
    *T0_ /= 2;
    prev_period /= 2;
    N /= 2;
    /* x += maxperiod; every access below is x[base + d] with base = maxperiod */
    let base = maxperiod as usize;
    let nn = N as usize;
    if *T0_ >= maxperiod {
        *T0_ = maxperiod - 1;
    }

    let mut T: i32;
    let T0: i32;
    T = *T0_;
    T0 = T;
    let yy_lookup = st.alloc::<f32>(maxperiod as usize + 1);
    let xb = &x[base..];
    dual_inner_prod(xb, xb, &x[base - T0 as usize..], N, &mut xx, &mut xy);
    yy_lookup[0] = xx;
    let mut yy = xx;
    let xw = &x[..base + nn];
    for i in 1..=base {
        yy = yy + xw[base - i] * xw[base - i] - xw[base + nn - i] * xw[base + nn - i];
        yy_lookup[i] = fmax(0.0, yy);
    }
    yy = yy_lookup[T0 as usize];
    let mut best_xy = xy;
    let mut best_yy = yy;
    let mut g = compute_pitch_gain(xy, xx, yy);
    let g0 = g;
    /* Look for any pitch at T/k */
    for k in 2..=15i32 {
        let T1 = (2 * T0 + k) / (2 * k);
        if T1 < minperiod {
            break;
        }
        /* Look for another strong correlation at T1b */
        let T1b: i32;
        if k == 2 {
            if T1 + T0 > maxperiod {
                T1b = T0;
            } else {
                T1b = T0 + T1;
            }
        } else {
            T1b = (2 * second_check[k as usize] * T0 + k) / (2 * k);
        }
        dual_inner_prod(
            xb,
            &x[base - T1 as usize..],
            &x[base - T1b as usize..],
            N,
            &mut xy,
            &mut xy2,
        );
        xy = 0.5 * (xy + xy2);
        yy = 0.5 * (yy_lookup[T1 as usize] + yy_lookup[T1b as usize]);
        let g1 = compute_pitch_gain(xy, xx, yy);
        let cont: f32;
        if (T1 - prev_period).abs() <= 1 {
            cont = prev_gain;
        } else if (T1 - prev_period).abs() <= 2 && 5 * k * k < T0 {
            cont = 0.5 * prev_gain;
        } else {
            cont = 0.0;
        }
        let mut thresh = fmax(0.3, 0.7 * g0 - cont);
        /* Bias against very high pitch (very short period) to avoid false-positives
        due to short-term correlation */
        if T1 < 3 * minperiod {
            thresh = fmax(0.4, 0.85 * g0 - cont);
        } else if T1 < 2 * minperiod {
            thresh = fmax(0.5, 0.9 * g0 - cont);
        }
        if g1 > thresh {
            best_xy = xy;
            best_yy = yy;
            T = T1;
            g = g1;
        }
    }
    best_xy = fmax(0.0, best_xy);
    let mut pg: f32;
    if best_yy <= best_xy {
        pg = 1.0;
    } else {
        pg = best_xy / (best_yy + 1.0);
    }

    for k in 0..3i32 {
        xcorr[k as usize] = celt_inner_prod(&x[base..], &x[base - (T + k - 1) as usize..], N);
    }
    if (xcorr[2] - xcorr[0]) > 0.7 * (xcorr[1] - xcorr[0]) {
        offset = 1;
    } else if (xcorr[0] - xcorr[2]) > 0.7 * (xcorr[1] - xcorr[2]) {
        offset = -1;
    } else {
        offset = 0;
    }
    if pg > g {
        pg = g;
    }
    *T0_ = 2 * T + offset;

    if *T0_ < minperiod0 {
        *T0_ = minperiod0;
    }
    pg
}
