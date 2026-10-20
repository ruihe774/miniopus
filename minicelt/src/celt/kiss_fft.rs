//! Port of celt/kiss_fft.h and celt/kiss_fft.c (forward FFT, float build, static twiddles).
//!
//! The twiddle and bit-reversal tables of the two FFT sizes in use (480 and 60 points) are not
//! stored in full: `init_fft` rebuilds them from the first octant of the twiddles (symmetry,
//! scripts/gen_tables.py checks that this reproduces the table of libopus bit for bit) and with
//! `compute_bitrev_table` of kiss_fft.c. They live in `.bss`.

use super::arch::Ix;
pub const MAXFACTORS: usize = 8;

pub type kiss_fft_scalar = f32;
pub type kiss_twiddle_scalar = f32;

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct kiss_fft_cpx {
    pub r: kiss_fft_scalar,
    pub i: kiss_fft_scalar,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct kiss_twiddle_cpx {
    pub r: kiss_twiddle_scalar,
    pub i: kiss_twiddle_scalar,
}

/// FFT state (static_modes_float.h; `nfft` and `arch_fft` are dropped, the tables are global).
pub struct kiss_fft_state {
    pub scale: f32,
    pub shift: i32,
    pub factors: [i16; 2 * MAXFACTORS],
}

struct FftTables {
    /// `exp(-2 pi i k / 480)`; the 60-point FFT uses every 8th
    twiddles: [kiss_twiddle_cpx; 480],
    /// bit-reversal tables of `fft_state48000_960_0` (480 points) and `_3` (60 points)
    bitrev480: [i16; 480],
    bitrev60: [i16; 60],
}
struct Tables(core::cell::UnsafeCell<FftTables>);
// SAFETY: written only by `init_fft`, single-threaded use only (as the pseudostack).
unsafe impl Sync for Tables {}
static TABLES: Tables = Tables(core::cell::UnsafeCell::new(FftTables {
    twiddles: [kiss_twiddle_cpx { r: 0.0, i: 0.0 }; 480],
    bitrev480: [0; 480],
    bitrev60: [0; 60],
}));

#[inline(always)]
fn twiddles() -> &'static [kiss_twiddle_cpx; 480] {
    // SAFETY: init_fft ran before the first encode; the tables are only read afterwards.
    unsafe { &(*TABLES.0.get()).twiddles }
}

/// Bit-reversal table of the FFT with the given MDCT shift (0: 480 points, 3: 60 points).
#[inline(always)]
pub fn bitrev(shift: i32) -> &'static [i16] {
    // SAFETY: as `twiddles`.
    unsafe {
        if shift == 0 { &(*TABLES.0.get()).bitrev480 } else { &(*TABLES.0.get()).bitrev60 }
    }
}

/// compute_bitrev_table() of kiss_fft.c (`f` indexed from `fi`).
fn compute_bitrev_table(Fout: i32, f: &mut [i16], mut fi: usize, fstride: usize, factors: &[i16]) {
    // IX-HARD: `factors` is a (radix, count) list whose length depends on the mode, and `f`/`fi` advance by
    // computed strides: nothing here lets the compiler bound the indices, so `Ix` stays.
    let p = factors.at(0) as i32;
    let m = factors.at(1) as i32;
    let mut Fout = Fout;
    for _ in 0..p {
        if m == 1 {
            *f.at_mut(fi) = Fout as i16;
            Fout += 1;
        } else {
            compute_bitrev_table(Fout, f, fi, fstride * p as usize, factors.rg(2..));
            Fout += m;
        }
        fi += fstride;
    }
}

/// Builds the twiddle and bit-reversal tables (call once before encoding).
pub fn init_fft() {
    use super::static_modes_float::{fft_state48000_960_0, fft_state48000_960_3, fft_twiddles48000_960 as octant};
    // SAFETY: nothing reads the tables while they are rebuilt (init only).
    let t = unsafe { &mut *TABLES.0.get() };
    let tw = &mut t.twiddles;
    let cpx = |r: f32, i: f32| kiss_twiddle_cpx { r, i };
    tw.rg_mut(..61).copy_from_slice(&octant);
    for k in 61..=120 {
        let o = tw.at(120 - k);
        *tw.at_mut(k) = cpx(-o.i, -o.r);
    }
    for k in 121..240 {
        let o = tw.at(k - 120);
        *tw.at_mut(k) = cpx(o.i, -o.r);
    }
    for k in 240..480 {
        let o = tw.at(k - 240);
        *tw.at_mut(k) = cpx(-o.r, -o.i);
    }
    /* the axis points of the C table are cos/sin of exact multiples of pi/2 in double */
    *tw.at_mut(120) = cpx(6.123234e-17, -1.0);
    *tw.at_mut(240) = cpx(-1.0, -1.2246469e-16);
    *tw.at_mut(360) = cpx(-1.8369701e-16, 1.0);
    compute_bitrev_table(0, &mut t.bitrev480, 0, 1, &fft_state48000_960_0.factors);
    compute_bitrev_table(0, &mut t.bitrev60, 0, 1, &fft_state48000_960_3.factors);
}

// SAFETY: repr(C) pair of f32, no padding, every bit pattern valid.
unsafe impl crate::stack::Pod for kiss_fft_cpx {}

/* The complex-math macros of _kiss_fft_guts.h (float build). Order of operations is kept. */

/// C_MUL(m,a,b): `m.r = a.r*b.r - a.i*b.i; m.i = a.r*b.i + a.i*b.r`
#[inline(always)]
fn C_MUL(a: kiss_fft_cpx, b: kiss_twiddle_cpx) -> kiss_fft_cpx {
    kiss_fft_cpx { r: a.r * b.r - a.i * b.i, i: a.r * b.i + a.i * b.r }
}
/// C_ADD(res,a,b)
#[inline(always)]
fn C_ADD(a: kiss_fft_cpx, b: kiss_fft_cpx) -> kiss_fft_cpx {
    kiss_fft_cpx { r: a.r + b.r, i: a.i + b.i }
}
/// C_SUB(res,a,b)
#[inline(always)]
fn C_SUB(a: kiss_fft_cpx, b: kiss_fft_cpx) -> kiss_fft_cpx {
    kiss_fft_cpx { r: a.r - b.r, i: a.i - b.i }
}
/// C_ADDTO(res,a)
#[inline(always)]
fn C_ADDTO(res: &mut kiss_fft_cpx, a: kiss_fft_cpx) {
    res.r += a.r;
    res.i += a.i;
}
/// HALF_OF(x): `(x)*.5f`
#[inline(always)]
fn HALF_OF(x: f32) -> f32 {
    x * 0.5
}

/// Radix-2 butterfly. `Fout` is the start of the FFT buffer.
fn kf_bfly2(Fout: &mut [kiss_fft_cpx], m: i32, N: i32) {
    // (m==1 is only used by CUSTOM_MODES.)
    let _ = m;
    let tw: f32 = 0.7071067812f32;
    /* We know that m==4 here because the radix-2 is just after a radix-4 */
    debug_assert!(m == 4);
    let mut o = 0usize;
    for _ in 0..N {
        // IX-HARD: provable in principle (`f` has 8 elements, indices 0..8), but only if the optimizer
        // folds `o + 8 - o`; `Fout.len() >= o + 8` is otherwise checked once per butterfly.
        // The butterflies are the hot loop, so the plain-indexing conversion needs a benchmark.
        let f = Fout.rg_mut(o..o + 8);
        let mut t = f.at(4);
        *f.at_mut(4) = C_SUB(f.at(0), t);
        C_ADDTO(f.at_mut(0), t);

        t.r = (f.at(5).r + f.at(5).i) * tw;
        t.i = (f.at(5).i - f.at(5).r) * tw;
        *f.at_mut(5) = C_SUB(f.at(1), t);
        C_ADDTO(f.at_mut(1), t);

        t.r = f.at(6).i;
        t.i = -f.at(6).r;
        *f.at_mut(6) = C_SUB(f.at(2), t);
        C_ADDTO(f.at_mut(2), t);

        t.r = (f.at(7).i - f.at(7).r) * tw;
        t.i = (-(f.at(7).i + f.at(7).r)) * tw;
        *f.at_mut(7) = C_SUB(f.at(3), t);
        C_ADDTO(f.at_mut(3), t);
        o += 8;
    }
}

fn kf_bfly4(Fout: &mut [kiss_fft_cpx], fstride: usize, tw: &[kiss_twiddle_cpx; 480], m: i32, N: i32, mm: i32) {
    if m == 1 {
        /* Degenerate case where all the twiddles are 1. */
        let mut o = 0usize;
        for _ in 0..N {
            let f = Fout.rg_mut(o..o + 4);
            let scratch0 = C_SUB(f.at(0), f.at(2));
            let f2 = f.at(2);
            C_ADDTO(f.at_mut(0), f2);
            let mut scratch1 = C_ADD(f.at(1), f.at(3));
            *f.at_mut(2) = C_SUB(f.at(0), scratch1);
            C_ADDTO(f.at_mut(0), scratch1);
            scratch1 = C_SUB(f.at(1), f.at(3));

            f.at_mut(1).r = scratch0.r + scratch1.i;
            f.at_mut(1).i = scratch0.i - scratch1.r;
            f.at_mut(3).r = scratch0.r - scratch1.i;
            f.at_mut(3).i = scratch0.i + scratch1.r;
            o += 4;
        }
    } else {
        let mut scratch = [kiss_fft_cpx::default(); 6];
        let m = m as usize;
        let m2 = 2 * m;
        let m3 = 3 * m;
        for i in 0..N as usize {
            let mut fo = i * mm as usize;
            let (mut tw1, mut tw2, mut tw3) = (0usize, 0usize, 0usize);
            /* m is guaranteed to be a multiple of 4. */
            for _ in 0..m {
                *scratch.at_mut(0) = C_MUL(Fout.at(fo + m), tw.at(tw1));
                *scratch.at_mut(1) = C_MUL(Fout.at(fo + m2), tw.at(tw2));
                *scratch.at_mut(2) = C_MUL(Fout.at(fo + m3), tw.at(tw3));

                *scratch.at_mut(5) = C_SUB(Fout.at(fo), scratch.at(1));
                C_ADDTO(Fout.at_mut(fo), scratch.at(1));
                *scratch.at_mut(3) = C_ADD(scratch.at(0), scratch.at(2));
                *scratch.at_mut(4) = C_SUB(scratch.at(0), scratch.at(2));
                *Fout.at_mut(fo + m2) = C_SUB(Fout.at(fo), scratch.at(3));
                tw1 += fstride;
                tw2 += fstride * 2;
                tw3 += fstride * 3;
                C_ADDTO(Fout.at_mut(fo), scratch.at(3));

                Fout.at_mut(fo + m).r = scratch.at(5).r + scratch.at(4).i;
                Fout.at_mut(fo + m).i = scratch.at(5).i - scratch.at(4).r;
                Fout.at_mut(fo + m3).r = scratch.at(5).r - scratch.at(4).i;
                Fout.at_mut(fo + m3).i = scratch.at(5).i + scratch.at(4).r;
                fo += 1;
            }
        }
    }
}

fn kf_bfly3(Fout: &mut [kiss_fft_cpx], fstride: usize, tw: &[kiss_twiddle_cpx; 480], m: i32, N: i32, mm: i32) {
    let m = m as usize;
    let m2 = 2 * m;
    let mut scratch = [kiss_fft_cpx::default(); 5];
    let epi3 = tw.at(fstride * m);
    for i in 0..N as usize {
        let mut fo = i * mm as usize;
        let (mut tw1, mut tw2) = (0usize, 0usize);
        /* For non-custom modes, m is guaranteed to be a multiple of 4. */
        let mut k = m;
        loop {
            *scratch.at_mut(1) = C_MUL(Fout.at(fo + m), tw.at(tw1));
            *scratch.at_mut(2) = C_MUL(Fout.at(fo + m2), tw.at(tw2));

            *scratch.at_mut(3) = C_ADD(scratch.at(1), scratch.at(2));
            *scratch.at_mut(0) = C_SUB(scratch.at(1), scratch.at(2));
            tw1 += fstride;
            tw2 += fstride * 2;

            Fout.at_mut(fo + m).r = Fout.at(fo).r - HALF_OF(scratch.at(3).r);
            Fout.at_mut(fo + m).i = Fout.at(fo).i - HALF_OF(scratch.at(3).i);

            /* C_MULBYSCALAR( scratch[0] , epi3.i ) */
            scratch.at_mut(0).r *= epi3.i;
            scratch.at_mut(0).i *= epi3.i;

            C_ADDTO(Fout.at_mut(fo), scratch.at(3));

            Fout.at_mut(fo + m2).r = Fout.at(fo + m).r + scratch.at(0).i;
            Fout.at_mut(fo + m2).i = Fout.at(fo + m).i - scratch.at(0).r;

            Fout.at_mut(fo + m).r = Fout.at(fo + m).r - scratch.at(0).i;
            Fout.at_mut(fo + m).i = Fout.at(fo + m).i + scratch.at(0).r;

            fo += 1;
            k -= 1;
            if k == 0 {
                break;
            }
        }
    }
}

fn kf_bfly5(Fout: &mut [kiss_fft_cpx], fstride: usize, tw: &[kiss_twiddle_cpx; 480], m: i32, N: i32, mm: i32) {
    let m = m as usize;
    let mut scratch = [kiss_fft_cpx::default(); 13];
    let ya = tw.at(fstride * m);
    let yb = tw.at(fstride * 2 * m);

    for i in 0..N as usize {
        let f0 = i * mm as usize;
        let f1 = f0 + m;
        let f2 = f0 + 2 * m;
        let f3 = f0 + 3 * m;
        let f4 = f0 + 4 * m;

        /* For non-custom modes, m is guaranteed to be a multiple of 4. */
        for u in 0..m {
            *scratch.at_mut(0) = Fout.at(f0 + u);

            *scratch.at_mut(1) = C_MUL(Fout.at(f1 + u), tw.at(u * fstride));
            *scratch.at_mut(2) = C_MUL(Fout.at(f2 + u), tw.at(2 * u * fstride));
            *scratch.at_mut(3) = C_MUL(Fout.at(f3 + u), tw.at(3 * u * fstride));
            *scratch.at_mut(4) = C_MUL(Fout.at(f4 + u), tw.at(4 * u * fstride));

            *scratch.at_mut(7) = C_ADD(scratch.at(1), scratch.at(4));
            *scratch.at_mut(10) = C_SUB(scratch.at(1), scratch.at(4));
            *scratch.at_mut(8) = C_ADD(scratch.at(2), scratch.at(3));
            *scratch.at_mut(9) = C_SUB(scratch.at(2), scratch.at(3));

            Fout.at_mut(f0 + u).r = Fout.at(f0 + u).r + (scratch.at(7).r + scratch.at(8).r);
            Fout.at_mut(f0 + u).i = Fout.at(f0 + u).i + (scratch.at(7).i + scratch.at(8).i);

            scratch.at_mut(5).r = scratch.at(0).r + (scratch.at(7).r * ya.r + scratch.at(8).r * yb.r);
            scratch.at_mut(5).i = scratch.at(0).i + (scratch.at(7).i * ya.r + scratch.at(8).i * yb.r);

            scratch.at_mut(6).r = scratch.at(10).i * ya.i + scratch.at(9).i * yb.i;
            scratch.at_mut(6).i = -(scratch.at(10).r * ya.i + scratch.at(9).r * yb.i);

            *Fout.at_mut(f1 + u) = C_SUB(scratch.at(5), scratch.at(6));
            *Fout.at_mut(f4 + u) = C_ADD(scratch.at(5), scratch.at(6));

            scratch.at_mut(11).r = scratch.at(0).r + (scratch.at(7).r * yb.r + scratch.at(8).r * ya.r);
            scratch.at_mut(11).i = scratch.at(0).i + (scratch.at(7).i * yb.r + scratch.at(8).i * ya.r);
            scratch.at_mut(12).r = scratch.at(9).i * ya.i - scratch.at(10).i * yb.i;
            scratch.at_mut(12).i = scratch.at(10).r * yb.i - scratch.at(9).r * ya.i;

            *Fout.at_mut(f2 + u) = C_ADD(scratch.at(11), scratch.at(12));
            *Fout.at_mut(f3 + u) = C_SUB(scratch.at(11), scratch.at(12));
        }
    }
}

/// In-place FFT on bit-reversed input (`fft_downshift` is a no-op in the float build).
pub fn opus_fft_impl(st: &kiss_fft_state, fout: &mut [kiss_fft_cpx]) {
    let mut fstride = [0i32; MAXFACTORS];
    let tw = twiddles();

    /* st->shift can be -1 */
    let shift = if st.shift > 0 { st.shift } else { 0 };

    *fstride.at_mut(0) = 1;
    let mut L = 0usize;
    let mut m;
    loop {
        let p = st.factors.at(2 * L) as i32;
        m = st.factors.at(2 * L + 1) as i32;
        *fstride.at_mut(L + 1) = fstride.at(L) * p;
        L += 1;
        if m == 1 {
            break;
        }
    }
    m = st.factors.at(2 * L - 1) as i32;
    for i in (0..L).rev() {
        let m2 = if i != 0 { st.factors.at(2 * i - 1) as i32 } else { 1 };
        match st.factors.at(2 * i) {
            2 => kf_bfly2(fout, m, fstride.at(i)),
            4 => kf_bfly4(fout, (fstride.at(i) << shift) as usize, tw, m, fstride.at(i), m2),
            3 => kf_bfly3(fout, (fstride.at(i) << shift) as usize, tw, m, fstride.at(i), m2),
            5 => kf_bfly5(fout, (fstride.at(i) << shift) as usize, tw, m, fstride.at(i), m2),
            _ => {}
        }
        m = m2;
    }
}
