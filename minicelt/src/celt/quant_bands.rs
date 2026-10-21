//! Port of celt/quant_bands.c (encoder side, float build).
//!
//! Fixed configuration: `start == 0`, LM == 3 (only the 960-sample rows of the coefficient and
//! probability tables), `effEnd == end`, no forced intra.
//!
//! Dropped: the decoder functions (`unquant_*`), `log2Amp` (decoder/RESYNTH only), and the
//! `lfe` parameter (no LFE). `quant_fine_energy` loses `prev_quant` (always NULL outside
//! ENABLE_QEXT, so `prev == 0`); `quant_energy_finalise` loses the `oldEBands != NULL` check
//! (always non-NULL outside ENABLE_QEXT).

use super::entcode::{ec_enc, ec_get_buffer, ec_range_bytes, ec_tell, ec_tell_frac};
use super::entenc::{ec_enc_bit_logp, ec_enc_bits, ec_enc_icdf};
use super::laplace::ec_laplace_encode;
use super::mathops::{floor64, log2};
use super::modes;
use super::rate::MAX_FINE_BITS;
use crate::celt::mathops::{fmax, fmin, imax, imin};
use crate::stack;

/// Mean energy in each band quantized in Q4 and converted back to float
pub static eMeans: [f32; 21] = [
    6.437500, 6.250000, 5.750000, 5.312500, 5.062500, 4.812500, 4.500000, 4.375000, 4.875000,
    4.687500, 4.562500, 4.437500, 4.875000, 4.625000, 4.312500, 4.500000, 4.375000, 4.625000,
    4.750000, 4.437500, 3.750000,
];

/* prediction coefficients: 0.9, 0.8, 0.65, 0.5 for LM = 0..3; only LM == 3 is used */
pub const pred_coef: f32 = 16384.0 / 32768.0;
pub const beta_coef: f32 = 6554.0 / 32768.0;
pub const beta_intra: f32 = 4915.0 / 32768.0;

/*Parameters of the Laplace-like probability models used for the coarse energy.
There is one pair of parameters for each prediction type (inter/intra) and band number
 (960-sample frames only).
The first number of each pair is the probability of 0, and the second is the
 decay rate, both in Q8 precision.*/
#[rustfmt::skip]
pub static e_prob_model: [[u8; 42]; 2] = [
    // Inter
    [
         42, 121,  96,  66, 108,  43, 111,  40, 117,  44, 123,  32, 120,  36,
        119,  33, 127,  33, 134,  34, 139,  21, 147,  23, 152,  20, 158,  25,
        154,  26, 166,  21, 173,  16, 184,  13, 184,  10, 150,  13, 139,  15,
    ],
    // Intra
    [
         22, 178,  63, 114,  74,  82,  84,  83,  92,  82, 103,  62,  96,  72,
         96,  67, 101,  73, 107,  72, 113,  55, 118,  52, 125,  52, 118,  52,
        117,  55, 135,  49, 137,  39, 157,  32, 145,  29,  97,  33,  77,  40,
    ],
];

pub static small_energy_icdf: [u8; 3] = [2, 1, 0];

pub fn loss_distortion(
    eBands: &[f32],
    oldEBands: &[f32],
    start: i32,
    end: i32,
    len: i32,
    C: i32,
) -> f32 {
    let mut dist: f32 = 0.0;
    for c in 0..C {
        let base = (c * len) as usize;
        let e = &eBands[base + start as usize..base + end as usize];
        let o = &oldEBands[base + start as usize..base + end as usize];
        for (&x, &y) in e.iter().zip(o) {
            let d = x - y;
            dist = dist + d * d;
        }
    }
    fmin(200.0, dist)
}

fn quant_coarse_energy_impl(
    end: i32,
    eBands: &[f32],
    oldEBands: &mut [f32],
    budget: i32,
    mut tell: i32,
    prob_model: &[u8; 42],
    error: &mut [f32],
    enc: &mut ec_enc,
    C: i32,
    intra: i32,
    max_decay: f32,
) -> i32 {
    let start = 0;
    let mut badness: i32 = 0;
    let mut prev: [f32; 2] = [0.0, 0.0];
    let coef: f32;
    let beta: f32;

    if tell + 3 <= budget {
        ec_enc_bit_logp(enc, intra, 3);
    }
    if intra != 0 {
        coef = 0.0;
        beta = beta_intra;
    } else {
        beta = beta_coef;
        coef = pred_coef;
    }

    /* Encode at a fixed coarse resolution */
    for i in start..end {
        let mut c = 0;
        loop {
            let k = (i + c * modes::nbEBands) as usize;
            let x = eBands[k];
            let oldE = fmax(-9.0, oldEBands[k]);
            let f = x - coef * oldE - prev[c as usize & 1];
            /* Rounding to nearest integer here is really important! */
            let mut qi = floor64((0.5f32 + f) as f64) as i32;
            let decay_bound = fmax(-28.0, oldEBands[k]) - max_decay;
            /* Prevent the energy from going down too quickly (e.g. for bands
            that have just one bin) */
            if qi < 0 && x < decay_bound {
                qi += (decay_bound - x) as i32;
                if qi > 0 {
                    qi = 0;
                }
            }
            let qi0 = qi;
            /* If we don't have enough bits to encode all the energy, just assume
            something safe. */
            tell = ec_tell(enc);
            let bits_left = budget - tell - 3 * C * (end - i);
            if i != start && bits_left < 30 {
                if bits_left < 24 {
                    qi = imin(1, qi);
                }
                if bits_left < 16 {
                    qi = imax(-1, qi);
                }
            }
            if budget - tell >= 15 {
                let pi = 2 * (i as usize).min(20);
                ec_laplace_encode(
                    enc,
                    &mut qi,
                    (prob_model[pi] as u32) << 7,
                    (prob_model[pi + 1] as i32) << 6,
                );
            } else if budget - tell >= 2 {
                qi = imax(-1, imin(qi, 1));
                ec_enc_icdf(enc, (2 * qi) ^ -((qi < 0) as i32), &small_energy_icdf, 2);
            } else if budget - tell >= 1 {
                qi = imin(0, qi);
                ec_enc_bit_logp(enc, -qi, 1);
            } else {
                qi = -1;
            }
            error[k] = f - qi as f32;
            badness += (qi0 - qi).abs();
            let q = qi as f32;

            let tmp = coef * oldE + prev[c as usize & 1] + q;
            oldEBands[k] = tmp;
            prev[c as usize & 1] = prev[c as usize & 1] + q - beta * q;
            c += 1;
            if c >= C {
                break;
            }
        }
    }
    badness
}

pub fn quant_coarse_energy(
    end: i32,
    eBands: &[f32],
    oldEBands: &mut [f32],
    budget: u32,
    error: &mut [f32],
    enc: &mut ec_enc,
    C: i32,
    nbAvailableBytes: i32,
    delayedIntra: &mut f32,
    mut two_pass: i32,
    loss_rate: i32,
) {
    let start = 0;
    let mut badness1: i32 = 0;
    let sg = stack::save();

    let mut intra: i32 = (two_pass == 0
        && *delayedIntra > (2 * C * (end - start)) as f32
        && nbAvailableBytes > (end - start) * C) as i32;
    /* budget (unsigned) * float * int / int: the whole expression is float. */
    let intra_bias: i32 =
        (((budget as f32 * *delayedIntra) * loss_rate as f32) / (C * 512) as f32) as i32;
    let new_distortion = loss_distortion(eBands, oldEBands, start, end, modes::nbEBands, C);

    let tell: u32 = ec_tell(enc) as u32;
    if tell.wrapping_add(3) > budget {
        two_pass = 0;
        intra = 0;
    }

    let mut max_decay: f32 = 16.0;
    if end - start > 10 {
        max_decay = fmin(max_decay, 0.125f32 * nbAvailableBytes as f32);
    }
    let enc_start_state: ec_enc = *enc;

    let nb = (C * modes::nbEBands) as usize;
    let oldEBands_intra = sg.alloc::<f32>(nb);
    let error_intra = sg.alloc::<f32>(nb);
    oldEBands_intra.copy_from_slice(&oldEBands[..nb]);

    if two_pass != 0 || intra != 0 {
        badness1 = quant_coarse_energy_impl(
            end,
            eBands,
            oldEBands_intra,
            budget as i32,
            tell as i32,
            &e_prob_model[1],
            error_intra,
            enc,
            C,
            1,
            max_decay,
        );
    }

    if intra == 0 {
        let tell_intra: i32 = ec_tell_frac(enc) as i32;

        let enc_intra_state: ec_enc = *enc;

        let nstart_bytes: u32 = ec_range_bytes(&enc_start_state);
        let nintra_bytes: u32 = ec_range_bytes(&enc_intra_state);
        // SAFETY: the encoder buffer holds at least nintra_bytes bytes.
        let intra_buf: *mut u8 = ec_get_buffer(&enc_intra_state).wrapping_add(nstart_bytes as usize);
        let save_bytes: u32 = nintra_bytes - nstart_bytes;
        // (save_bytes == 0 is ALLOC_NONE in C; a zero-length alloc here.)
        let intra_bits = sg.alloc::<u8>(save_bytes as usize);
        /* Copy bits from intra bit-stream */
        // UNSAFE-MAND: reads the raw `ec_enc` buffer pointer (see `ec_write_byte` in entenc.rs).
        unsafe {
            core::ptr::copy_nonoverlapping(intra_buf, intra_bits.as_mut_ptr(), save_bytes as usize)
        };

        *enc = enc_start_state;

        let badness2 = quant_coarse_energy_impl(
            end,
            eBands,
            oldEBands,
            budget as i32,
            tell as i32,
            &e_prob_model[intra as usize],
            error,
            enc,
            C,
            0,
            max_decay,
        );

        if two_pass != 0
            && (badness1 < badness2
                || (badness1 == badness2 && (ec_tell_frac(enc) as i32) + intra_bias > tell_intra))
        {
            *enc = enc_intra_state;
            /* Copy intra bits to bit-stream */
            // UNSAFE-MAND: writes the raw `ec_enc` buffer pointer (see `ec_write_byte` in entenc.rs).
            unsafe {
                core::ptr::copy_nonoverlapping(intra_bits.as_ptr(), intra_buf, save_bytes as usize)
            };
            oldEBands[..nb].copy_from_slice(oldEBands_intra);
            error[..nb].copy_from_slice(error_intra);
            intra = 1;
        }
    } else {
        oldEBands[..nb].copy_from_slice(oldEBands_intra);
        error[..nb].copy_from_slice(error_intra);
    }

    if intra != 0 {
        *delayedIntra = new_distortion;
    } else {
        *delayedIntra = (pred_coef * pred_coef) * *delayedIntra + new_distortion;
    }
}

pub fn quant_fine_energy(
    end: i32,
    oldEBands: &mut [f32],
    error: &mut [f32],
    extra_quant: &[i32],
    enc: &mut ec_enc,
    C: i32,
) {
    /* prev_quant is always NULL here (it is only used with ENABLE_QEXT): prev == 0 */
    let prev: i32 = 0;
    let start = 0;
    /* Encode finer resolution */
    let extra_quant = &extra_quant[..end as usize];
    for i in start..end {
        let eq = extra_quant[i as usize];
        if eq <= 0 {
            continue;
        }
        let extra: i32 = (1i32 << eq) as i16 as i32;
        if ec_tell(enc) + C * eq > (enc.storage as i32) * 8 {
            continue;
        }
        for c in 0..C {
            let k = (i + c * modes::nbEBands) as usize;
            let mut q2 =
                floor64(((error[k] * (1 << prev) as f32 + 0.5f32) * extra as f32) as f64) as i32;
            if q2 > extra - 1 {
                q2 = extra - 1;
            }
            if q2 < 0 {
                q2 = 0;
            }
            ec_enc_bits(enc, q2 as u32, eq as u32);
            let mut offset: f32 =
                (q2 as f32 + 0.5f32) * (1 << (14 - eq)) as f32 * (1.0f32 / 16384.0) - 0.5f32;
            offset *= (1 << (14 - prev)) as f32 * (1.0f32 / 16384.0);
            oldEBands[k] += offset;
            error[k] -= offset;
        }
    }
}

pub fn quant_energy_finalise(
    end: i32,
    oldEBands: &mut [f32],
    error: &mut [f32],
    fine_quant: &[i32],
    fine_priority: &[i32],
    mut bits_left: i32,
    enc: &mut ec_enc,
    C: i32,
) {
    let start = 0;
    let fine_quant = &fine_quant[..end as usize];
    let fine_priority = &fine_priority[..end as usize];
    /* Use up the remaining bits */
    for prio in 0..2 {
        let mut i = start;
        while i < end && bits_left >= C {
            if fine_quant[i as usize] >= MAX_FINE_BITS || fine_priority[i as usize] != prio {
                i += 1;
                continue;
            }
            for c in 0..C {
                let k = (i + c * modes::nbEBands) as usize;
                let q2: i32 = if error[k] < 0.0 { 0 } else { 1 };
                ec_enc_bits(enc, q2 as u32, 1);
                let offset: f32 = (q2 as f32 - 0.5f32)
                    * (1 << (14 - fine_quant[i as usize] - 1)) as f32
                    * (1.0f32 / 16384.0);
                oldEBands[k] += offset;
                error[k] -= offset;
                bits_left -= 1;
            }
            i += 1;
        }
    }
}

/// (`effEnd == end` here, so the `-14` fill of the bands above effEnd never happens)
pub fn amp2Log2(end: i32, bandE: &[f32], bandLogE: &mut [f32], C: i32) {
    for c in 0..C {
        let base = (c * modes::nbEBands) as usize;
        let out = &mut bandLogE[base..base + end as usize];
        let e = &bandE[base..base + end as usize];
        for ((o, &b), &mean) in out.iter_mut().zip(e).zip(&eMeans[..end as usize]) {
            *o = log2(b) - mean;
        }
    }
}
