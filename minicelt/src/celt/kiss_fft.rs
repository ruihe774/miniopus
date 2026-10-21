//! Port of celt/kiss_fft.h and celt/kiss_fft.c (forward FFT, float build, static twiddles).
//!
//! The twiddle and bit-reversal tables of the two FFT sizes in use (480 and 60 points) are not
//! stored in full: `init_fft` rebuilds them from the first octant of the twiddles (symmetry,
//! scripts/gen_tables.py checks that this reproduces the table of libopus bit for bit) and with
//! `compute_bitrev_table` of kiss_fft.c. They live in `.bss`.

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
// UNSAFE-MAND: `UnsafeCell` statics need a manual `Sync` impl.
unsafe impl Sync for Tables {}
static TABLES: Tables = Tables(core::cell::UnsafeCell::new(FftTables {
    twiddles: [kiss_twiddle_cpx { r: 0.0, i: 0.0 }; 480],
    bitrev480: [0; 480],
    bitrev60: [0; 60],
}));

#[inline(always)]
fn tables() -> &'static FftTables {
    // SAFETY: init_fft ran before the first encode; the tables are only read afterwards.
    // UNSAFE-MAND: reading through the `UnsafeCell` static (the tables are filled at runtime).
    unsafe { &*TABLES.0.get() }
}

#[inline(always)]
fn twiddles() -> &'static [kiss_twiddle_cpx; 480] {
    &tables().twiddles
}

/// Bit-reversal table of the FFT with the given MDCT shift (0: 480 points, 3: 60 points).
#[inline(always)]
pub fn bitrev(shift: i32) -> &'static [i16] {
    if shift == 0 {
        &tables().bitrev480
    } else {
        &tables().bitrev60
    }
}

/// compute_bitrev_table() of kiss_fft.c (`f` starts at the C `f`).
fn compute_bitrev_table(mut Fout: i32, f: &mut [i16], fstride: usize, factors: &[i16]) {
    /* the list always ends with m == 1 before it runs out */
    let &[p, m, ref rest @ ..] = factors else {
        return;
    };
    for k in 0..p as usize {
        let f = &mut f[k * fstride..];
        if m == 1 {
            f[0] = Fout as i16;
            Fout += 1;
        } else {
            compute_bitrev_table(Fout, f, fstride * p as usize, rest);
            Fout += m as i32;
        }
    }
}

/// Builds the twiddle and bit-reversal tables (call once before encoding).
pub fn init_fft() {
    use super::static_modes_float::{
        fft_state48000_960_0, fft_state48000_960_3, fft_twiddles48000_960 as octant,
    };
    // SAFETY: nothing reads the tables while they are rebuilt (init only).
    // UNSAFE-MAND: `&mut` to the `UnsafeCell` static.
    let t = unsafe { &mut *TABLES.0.get() };
    let tw = &mut t.twiddles;
    let cpx = |r: f32, i: f32| kiss_twiddle_cpx { r, i };
    tw[..61].copy_from_slice(&octant);
    for k in 61..=120 {
        let o = tw[120 - k];
        tw[k] = cpx(-o.i, -o.r);
    }
    for k in 121..240 {
        let o = tw[k - 120];
        tw[k] = cpx(o.i, -o.r);
    }
    for k in 240..480 {
        let o = tw[k - 240];
        tw[k] = cpx(-o.r, -o.i);
    }
    /* the axis points of the C table are cos/sin of exact multiples of pi/2 in double */
    tw[120] = cpx(6.123234e-17, -1.0);
    tw[240] = cpx(-1.0, -1.2246469e-16);
    tw[360] = cpx(-1.8369701e-16, 1.0);
    compute_bitrev_table(0, &mut t.bitrev480, 1, &fft_state48000_960_0.factors);
    compute_bitrev_table(0, &mut t.bitrev60, 1, &fft_state48000_960_3.factors);
}

// SAFETY: repr(C) pair of f32, no padding, every bit pattern valid.
// UNSAFE-MAND: implementing the unsafe trait `Pod`.
unsafe impl crate::stack::Pod for kiss_fft_cpx {}

/* The complex-math macros of _kiss_fft_guts.h (float build). Order of operations is kept. */

/// C_MUL(m,a,b): `m.r = a.r*b.r - a.i*b.i; m.i = a.r*b.i + a.i*b.r`
#[inline(always)]
fn C_MUL(a: kiss_fft_cpx, b: kiss_twiddle_cpx) -> kiss_fft_cpx {
    kiss_fft_cpx {
        r: a.r * b.r - a.i * b.i,
        i: a.r * b.i + a.i * b.r,
    }
}
/// C_ADD(res,a,b)
#[inline(always)]
fn C_ADD(a: kiss_fft_cpx, b: kiss_fft_cpx) -> kiss_fft_cpx {
    kiss_fft_cpx {
        r: a.r + b.r,
        i: a.i + b.i,
    }
}
/// C_SUB(res,a,b)
#[inline(always)]
fn C_SUB(a: kiss_fft_cpx, b: kiss_fft_cpx) -> kiss_fft_cpx {
    kiss_fft_cpx {
        r: a.r - b.r,
        i: a.i - b.i,
    }
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
#[inline(always)]
fn kf_bfly2(Fout: &mut [kiss_fft_cpx], m: i32, N: i32) {
    // (m==1 is only used by CUSTOM_MODES.)
    let _ = m;
    let tw: f32 = 0.7071067812f32;
    /* We know that m==4 here because the radix-2 is just after a radix-4 */
    debug_assert!(m == 4);
    for f in &mut Fout.as_chunks_mut::<8>().0[..N as usize] {
        let mut t = f[4];
        f[4] = C_SUB(f[0], t);
        C_ADDTO(&mut f[0], t);

        t.r = (f[5].r + f[5].i) * tw;
        t.i = (f[5].i - f[5].r) * tw;
        f[5] = C_SUB(f[1], t);
        C_ADDTO(&mut f[1], t);

        t.r = f[6].i;
        t.i = -f[6].r;
        f[6] = C_SUB(f[2], t);
        C_ADDTO(&mut f[2], t);

        t.r = (f[7].i - f[7].r) * tw;
        t.i = (-(f[7].i + f[7].r)) * tw;
        f[7] = C_SUB(f[3], t);
        C_ADDTO(&mut f[3], t);
    }
}

#[inline(always)]
fn kf_bfly4(
    Fout: &mut [kiss_fft_cpx],
    fstride: usize,
    tw: &[kiss_twiddle_cpx; 480],
    m: i32,
    N: i32,
    mm: i32,
) {
    if m == 1 {
        /* Degenerate case where all the twiddles are 1. */
        for f in &mut Fout.as_chunks_mut::<4>().0[..N as usize] {
            let scratch0 = C_SUB(f[0], f[2]);
            let f2 = f[2];
            C_ADDTO(&mut f[0], f2);
            let mut scratch1 = C_ADD(f[1], f[3]);
            f[2] = C_SUB(f[0], scratch1);
            C_ADDTO(&mut f[0], scratch1);
            scratch1 = C_SUB(f[1], f[3]);

            f[1].r = scratch0.r + scratch1.i;
            f[1].i = scratch0.i - scratch1.r;
            f[3].r = scratch0.r - scratch1.i;
            f[3].i = scratch0.i + scratch1.r;
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
                scratch[0] = C_MUL(Fout[fo + m], tw[tw1]);
                scratch[1] = C_MUL(Fout[fo + m2], tw[tw2]);
                scratch[2] = C_MUL(Fout[fo + m3], tw[tw3]);

                scratch[5] = C_SUB(Fout[fo], scratch[1]);
                C_ADDTO(&mut Fout[fo], scratch[1]);
                scratch[3] = C_ADD(scratch[0], scratch[2]);
                scratch[4] = C_SUB(scratch[0], scratch[2]);
                Fout[fo + m2] = C_SUB(Fout[fo], scratch[3]);
                tw1 += fstride;
                tw2 += fstride * 2;
                tw3 += fstride * 3;
                C_ADDTO(&mut Fout[fo], scratch[3]);

                Fout[fo + m].r = scratch[5].r + scratch[4].i;
                Fout[fo + m].i = scratch[5].i - scratch[4].r;
                Fout[fo + m3].r = scratch[5].r - scratch[4].i;
                Fout[fo + m3].i = scratch[5].i + scratch[4].r;
                fo += 1;
            }
        }
    }
}

#[inline(always)]
fn kf_bfly3(
    Fout: &mut [kiss_fft_cpx],
    fstride: usize,
    tw: &[kiss_twiddle_cpx; 480],
    m: i32,
    N: i32,
    mm: i32,
) {
    let m = m as usize;
    let m2 = 2 * m;
    let mut scratch = [kiss_fft_cpx::default(); 5];
    let epi3 = tw[fstride * m];
    for i in 0..N as usize {
        let mut fo = i * mm as usize;
        let (mut tw1, mut tw2) = (0usize, 0usize);
        /* For non-custom modes, m is guaranteed to be a multiple of 4. */
        let mut k = m;
        loop {
            scratch[1] = C_MUL(Fout[fo + m], tw[tw1]);
            scratch[2] = C_MUL(Fout[fo + m2], tw[tw2]);

            scratch[3] = C_ADD(scratch[1], scratch[2]);
            scratch[0] = C_SUB(scratch[1], scratch[2]);
            tw1 += fstride;
            tw2 += fstride * 2;

            Fout[fo + m].r = Fout[fo].r - HALF_OF(scratch[3].r);
            Fout[fo + m].i = Fout[fo].i - HALF_OF(scratch[3].i);

            /* C_MULBYSCALAR( scratch[0] , epi3.i ) */
            scratch[0].r *= epi3.i;
            scratch[0].i *= epi3.i;

            C_ADDTO(&mut Fout[fo], scratch[3]);

            Fout[fo + m2].r = Fout[fo + m].r + scratch[0].i;
            Fout[fo + m2].i = Fout[fo + m].i - scratch[0].r;

            Fout[fo + m].r = Fout[fo + m].r - scratch[0].i;
            Fout[fo + m].i = Fout[fo + m].i + scratch[0].r;

            fo += 1;
            k -= 1;
            if k == 0 {
                break;
            }
        }
    }
}

#[inline(always)]
fn kf_bfly5(
    Fout: &mut [kiss_fft_cpx],
    fstride: usize,
    tw: &[kiss_twiddle_cpx; 480],
    m: i32,
    N: i32,
    mm: i32,
) {
    let m = m as usize;
    let mut scratch = [kiss_fft_cpx::default(); 13];
    let ya = tw[fstride * m];
    let yb = tw[fstride * 2 * m];

    for i in 0..N as usize {
        let f0 = i * mm as usize;
        let f1 = f0 + m;
        let f2 = f0 + 2 * m;
        let f3 = f0 + 3 * m;
        let f4 = f0 + 4 * m;

        /* For non-custom modes, m is guaranteed to be a multiple of 4. */
        for u in 0..m {
            scratch[0] = Fout[f0 + u];

            scratch[1] = C_MUL(Fout[f1 + u], tw[u * fstride]);
            scratch[2] = C_MUL(Fout[f2 + u], tw[2 * u * fstride]);
            scratch[3] = C_MUL(Fout[f3 + u], tw[3 * u * fstride]);
            scratch[4] = C_MUL(Fout[f4 + u], tw[4 * u * fstride]);

            scratch[7] = C_ADD(scratch[1], scratch[4]);
            scratch[10] = C_SUB(scratch[1], scratch[4]);
            scratch[8] = C_ADD(scratch[2], scratch[3]);
            scratch[9] = C_SUB(scratch[2], scratch[3]);

            Fout[f0 + u].r = Fout[f0 + u].r + (scratch[7].r + scratch[8].r);
            Fout[f0 + u].i = Fout[f0 + u].i + (scratch[7].i + scratch[8].i);

            scratch[5].r = scratch[0].r + (scratch[7].r * ya.r + scratch[8].r * yb.r);
            scratch[5].i = scratch[0].i + (scratch[7].i * ya.r + scratch[8].i * yb.r);

            scratch[6].r = scratch[10].i * ya.i + scratch[9].i * yb.i;
            scratch[6].i = -(scratch[10].r * ya.i + scratch[9].r * yb.i);

            Fout[f1 + u] = C_SUB(scratch[5], scratch[6]);
            Fout[f4 + u] = C_ADD(scratch[5], scratch[6]);

            scratch[11].r = scratch[0].r + (scratch[7].r * yb.r + scratch[8].r * ya.r);
            scratch[11].i = scratch[0].i + (scratch[7].i * yb.r + scratch[8].i * ya.r);
            scratch[12].r = scratch[9].i * ya.i - scratch[10].i * yb.i;
            scratch[12].i = scratch[10].r * yb.i - scratch[9].r * ya.i;

            Fout[f2 + u] = C_ADD(scratch[11], scratch[12]);
            Fout[f3 + u] = C_SUB(scratch[11], scratch[12]);
        }
    }
}

/// In-place FFT on bit-reversed input (`fft_downshift` is a no-op in the float build).
/// The stage loop of kiss_fft.c unrolled for the two FFTs in use (the factor tables
/// `fft_state48000_960_0` `[5, 96, 3, 32, 4, 8, 2, 4, 4, 1]` and `_3` `[5, 12, 3, 4, 4, 1]`):
/// stages run from the last factor to the first with `kf_bfly*(Fout, fstride << shift, tw, m, N, mm)`.
pub fn opus_fft_impl(st: &kiss_fft_state, fout: &mut [kiss_fft_cpx]) {
    let tw = twiddles();
    if st.shift <= 0 {
        /* 480 points (shift -1, treated as 0) */
        debug_assert!(st.factors[..10] == [5, 96, 3, 32, 4, 8, 2, 4, 4, 1]);
        /* a fixed length lets the constant-stride stages prove their indices */
        let fout: &mut [kiss_fft_cpx; 480] = fout.first_chunk_mut().unwrap();
        kf_bfly4(fout, 120, tw, 1, 120, 4);
        kf_bfly2(fout, 4, 60);
        kf_bfly4(fout, 15, tw, 8, 15, 32);
        kf_bfly3(fout, 5, tw, 32, 5, 96);
        kf_bfly5(fout, 1, tw, 96, 1, 1);
    } else {
        /* 60 points (shift 3) */
        debug_assert!(st.shift == 3 && st.factors[..6] == [5, 12, 3, 4, 4, 1]);
        let fout: &mut [kiss_fft_cpx; 60] = fout.first_chunk_mut().unwrap();
        kf_bfly4(fout, 15 << 3, tw, 1, 15, 4);
        kf_bfly3(fout, 5 << 3, tw, 4, 5, 12);
        kf_bfly5(fout, 1 << 3, tw, 12, 1, 1);
    }
}
