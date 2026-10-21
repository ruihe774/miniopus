//! Port of celt/laplace.c (encoder side).

use super::entcode::ec_enc;
use super::entenc::ec_encode_bin;

/* The minimum probability of an energy delta (out of 32768). */
const LAPLACE_LOG_MINP: i32 = 0;
const LAPLACE_MINP: u32 = 1 << LAPLACE_LOG_MINP;
/* The minimum number of guaranteed representable energy deltas (in one
    direction). */
const LAPLACE_NMIN: u32 = 16;

/* When called, decay is positive and at most 11456. */
pub fn ec_laplace_get_freq1(fs0: u32, decay: i32) -> u32 {
    let ft = 32768 - LAPLACE_MINP * (2 * LAPLACE_NMIN) - fs0;
    /* ft*(opus_int32)(16384-decay): unsigned multiply */
    ft.wrapping_mul((16384 - decay) as u32) >> 15
}

/// Fixed interface (used by quant_bands.rs).
pub fn ec_laplace_encode(enc: &mut ec_enc, value: &mut i32, mut fs: u32, decay: i32) {
    let mut val = *value;
    let mut fl: u32 = 0;
    if val != 0 {
        let s: i32 = -((val < 0) as i32);
        val = (val + s) ^ s;
        fl = fs;
        fs = ec_laplace_get_freq1(fs, decay);
        /* Search the decaying part of the PDF.*/
        let mut i: i32 = 1;
        while fs > 0 && i < val {
            fs *= 2;
            fl += fs + 2 * LAPLACE_MINP;
            fs = fs.wrapping_mul(decay as u32) >> 15;
            i += 1;
        }
        /* Everything beyond that has probability LAPLACE_MINP. */
        if fs == 0 {
            /* (32768-fl+LAPLACE_MINP-1) is unsigned in C, converted to int. */
            let mut ndi_max = (32768u32.wrapping_sub(fl) + LAPLACE_MINP - 1) as i32 >> LAPLACE_LOG_MINP;
            ndi_max = (ndi_max - s) >> 1;
            let di = (val - i).min(ndi_max - 1);
            /* int added to unsigned */
            fl = fl.wrapping_add(((2 * di + 1 + s) * LAPLACE_MINP as i32) as u32);
            /* IMIN(int, unsigned) compares as unsigned */
            let rest = 32768u32.wrapping_sub(fl);
            fs = if LAPLACE_MINP < rest { LAPLACE_MINP } else { rest };
            *value = (i + di + s) ^ s;
        } else {
            fs += LAPLACE_MINP;
            fl += fs & !s as u32;
        }
        debug_assert!(fl + fs <= 32768);
        debug_assert!(fs > 0);
    }
    ec_encode_bin(enc, fl, fl + fs, 15);
}
