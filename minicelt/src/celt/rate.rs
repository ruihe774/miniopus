//! Port of celt/rate.h and celt/rate.c (encoder side; static pulse cache).

use super::entcode::{BITRES, celt_udiv, ec_ctx};
use super::entenc::{ec_enc_bit_logp, ec_enc_uint};
use super::modes;

static LOG2_FRAC_TABLE: [u8; 24] = [
    0, 8, 13, 16, 19, 21, 23, 24, 26, 27, 28, 29, 30, 31, 32, 32, 33, 34, 34, 35, 36, 36, 37, 37,
];

pub const MAX_FINE_BITS: i32 = 8;

pub const FINE_OFFSET: i32 = 21;
pub const QTHETA_OFFSET: i32 = 4;
pub const QTHETA_OFFSET_TWOPHASE: i32 = 16;

/// Fixed interface (used by bands.rs).
#[inline]
pub fn get_pulses(i: i32) -> i32 {
    if i < 8 {
        i
    } else {
        (8 + (i & 7)) << ((i >> 3) - 1)
    }
}

/// Fixed interface (used by bands.rs).
pub fn bits2pulses(band: i32, LM: i32, mut bits: i32) -> i32 {
    let LM = LM + 1;
    /* `cache_bits` is padded so that every entry has a 65-byte window (entries hold at most 40):
    the indices below are at most 64 by construction, so they need no bounds check. */
    let ci = modes::cache_index[(LM * modes::nbEBands + band) as usize] as usize;
    let cache: &[u8; 65] = modes::cache_bits[ci..].first_chunk().unwrap();
    let n = cache[0] as usize;

    bits -= 1;
    /* The binary search of rate.h (LOG_MAX_PSEUDO = 6 steps over 1..=n) as a branchless search:
    cache[1..=n] is non-decreasing, so lo is the number of entries below bits. */
    let step = |lo: usize, half: usize| {
        let k = lo + half;
        if (k <= n) & ((cache[k] as i32) < bits) {
            k
        } else {
            lo
        }
    };
    let lo = step(step(step(step(step(step(0, 32), 16), 8), 4), 2), 1);
    let hi = if lo < n { lo + 1 } else { lo };
    if bits - (if lo == 0 { -1 } else { cache[lo] as i32 }) <= cache[hi] as i32 - bits {
        lo as i32
    } else {
        hi as i32
    }
}

/// Fixed interface (used by bands.rs).
#[inline(always)]
pub fn pulses2bits(band: i32, LM: i32, pulses: i32) -> i32 {
    let LM = LM + 1;
    /* the window of `bits2pulses` (pulses <= 40) */
    let ci = modes::cache_index[(LM * modes::nbEBands + band) as usize] as usize;
    let cache: &[u8; 65] = modes::cache_bits[ci..].first_chunk().unwrap();
    if pulses == 0 {
        0
    } else {
        cache[pulses as usize] as i32 + 1
    }
}

const ALLOC_STEPS: i32 = 6;

#[inline(always)]
fn eb(i: i32) -> i32 {
    modes::eBands[i as usize] as i32
}

fn interp_bits2pulses(
    end: i32,
    skip_start: i32,
    bits1: &[i32],
    bits2: &[i32],
    thresh: &[i32],
    cap: &[i32],
    mut total: i32,
    _balance: &mut i32,
    skip_rsv: i32,
    intensity: &mut i32,
    mut intensity_rsv: i32,
    dual_stereo: &mut i32,
    mut dual_stereo_rsv: i32,
    bits: &mut [i32],
    ebits: &mut [i32],
    fine_priority: &mut [i32],
    C: i32,
    ec: &mut ec_ctx,
    prev: i32,
) -> i32 {
    /* start == 0, LM == 3 */
    let start = 0;
    let LM = 3;
    let eu = end as usize;
    let (bits1, bits2, thresh, cap) = (&bits1[..eu], &bits2[..eu], &thresh[..eu], &cap[..eu]);
    let (bits, ebits, fine_priority) =
        (&mut bits[..eu], &mut ebits[..eu], &mut fine_priority[..eu]);
    /* indexed by the same band counters as `bits` */
    let (log2_frac, logN) = (&LOG2_FRAC_TABLE[..eu], &modes::logN[..eu]);
    let mut psum: i32;
    let mut lo: i32;
    let mut hi: i32;
    let mut j: i32;
    let mut left: i32;
    let mut percoeff: i32;
    let mut done: bool;

    let alloc_floor = C << BITRES;
    let stereo = (C > 1) as i32;

    let logM = LM << BITRES;
    lo = 0;
    hi = 1 << ALLOC_STEPS;
    for _ in 0..ALLOC_STEPS {
        let mid = (lo + hi) >> 1;
        psum = 0;
        done = false;
        j = end;
        while j > start {
            j -= 1;
            let ju = j as usize;
            let tmp = bits1[ju] + (mid * bits2[ju] >> ALLOC_STEPS);
            if tmp >= thresh[ju] || done {
                done = true;
                /* Don't allocate more than we can actually use */
                psum += tmp.min(cap[ju]);
            } else if tmp >= alloc_floor {
                psum += alloc_floor;
            }
        }
        if psum > total {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    psum = 0;
    done = false;
    j = end;
    while j > start {
        j -= 1;
        let ju = j as usize;
        let mut tmp = bits1[ju] + (lo * bits2[ju] >> ALLOC_STEPS);
        if tmp < thresh[ju] && !done {
            if tmp >= alloc_floor {
                tmp = alloc_floor;
            } else {
                tmp = 0;
            }
        } else {
            done = true;
        }
        /* Don't allocate more than we can actually use */
        tmp = tmp.min(cap[ju]);
        bits[ju] = tmp;
        psum += tmp;
    }

    /* Decide which bands to skip, working backwards from the end. */
    let mut codedBands = end;
    loop {
        j = codedBands - 1;
        /* Never skip the first band, nor a band that has been boosted by
         dynalloc.
        In the first case, we'd be coding a bit to signal we're going to waste
         all the other bits.
        In the second case, we'd be coding a bit to redistribute all the bits
         we just signaled should be concentrated in this band. */
        if j <= skip_start {
            /* Give the bit we reserved to end skipping back. */
            total += skip_rsv;
            break;
        }
        let ju = j as usize;
        /*Figure out how many left-over bits we would be adding to this band.
        This can include bits we've stolen back from higher, skipped bands.*/
        left = total - psum;
        /* celt_udiv takes unsigned arguments and returns unsigned */
        percoeff = celt_udiv(left as u32, (eb(codedBands) - eb(start)) as u32) as i32;
        left -= (eb(codedBands) - eb(start)) * percoeff;
        let rem = (left - (eb(j) - eb(start))).max(0);
        let band_width = eb(codedBands) - eb(j);
        let mut band_bits = bits[ju] + percoeff * band_width + rem;
        /*Only code a skip decision if we're above the threshold for this band.
        Otherwise it is force-skipped.
        This ensures that we have enough bits to code the skip flag.*/
        if band_bits >= thresh[ju].max(alloc_floor + (1 << BITRES)) {
            /*This if() block is the only part of the allocation function that
            is not a mandatory part of the bitstream: any bands we choose to
            skip here must be explicitly signaled.*/
            /*We choose a threshold with some hysteresis to keep bands from
            fluctuating in and out, but we try not to fold below a certain point. */
            let depth_threshold = if codedBands > 17 {
                if j < prev { 7 } else { 9 }
            } else {
                0
            };
            if codedBands <= start + 2
                /* (j <= signalBandwidth == end - 1 always holds) */
                || band_bits > (depth_threshold * band_width << LM << BITRES) >> 4
            {
                ec_enc_bit_logp(ec, 1, 1);
                break;
            }
            ec_enc_bit_logp(ec, 0, 1);
            /*We used a bit to skip this band.*/
            psum += 1 << BITRES;
            band_bits -= 1 << BITRES;
        }
        /*Reclaim the bits originally allocated to this band.*/
        psum -= bits[ju] + intensity_rsv;
        if intensity_rsv > 0 {
            intensity_rsv = log2_frac[(j - start) as usize] as i32;
        }
        psum += intensity_rsv;
        if band_bits >= alloc_floor {
            /*If we have enough for a fine energy bit per channel, use it.*/
            psum += alloc_floor;
            bits[ju] = alloc_floor;
        } else {
            /*Otherwise this band gets nothing at all.*/
            bits[ju] = 0;
        }
        codedBands -= 1;
    }

    debug_assert!(codedBands > start);
    /* Code the intensity and dual stereo parameters. */
    if intensity_rsv > 0 {
        *intensity = (*intensity).min(codedBands);
        ec_enc_uint(
            ec,
            (*intensity - start) as u32,
            (codedBands + 1 - start) as u32,
        );
    } else {
        *intensity = 0;
    }
    if *intensity <= start {
        total += dual_stereo_rsv;
        dual_stereo_rsv = 0;
    }
    if dual_stereo_rsv > 0 {
        ec_enc_bit_logp(ec, *dual_stereo, 1);
    } else {
        *dual_stereo = 0;
    }

    /* Allocate the remaining bits */
    left = total - psum;
    percoeff = celt_udiv(left as u32, (eb(codedBands) - eb(start)) as u32) as i32;
    left -= (eb(codedBands) - eb(start)) * percoeff;
    j = start;
    while j < codedBands {
        bits[j as usize] += percoeff * (eb(j + 1) - eb(j));
        j += 1;
    }
    j = start;
    while j < codedBands {
        let tmp = left.min(eb(j + 1) - eb(j));
        bits[j as usize] += tmp;
        left -= tmp;
        j += 1;
    }

    let mut balance: i32 = 0;
    j = start;
    while j < codedBands {
        let ju = j as usize;
        let mut excess: i32;

        debug_assert!(bits[ju] >= 0);
        let N0 = eb(j + 1) - eb(j);
        let N = N0 << LM;
        let bit = bits[ju] + balance;

        if N > 1 {
            excess = (bit - cap[ju]).max(0);
            bits[ju] = bit - excess;

            /* Compensate for the extra DoF in stereo */
            let den = C * N
                + if C == 2 && N > 2 && *dual_stereo == 0 && j < *intensity {
                    1
                } else {
                    0
                };

            let NClogN = den * (logN[ju] as i32 + logM);

            /* Offset for the number of fine bits by log2(N)/2 + FINE_OFFSET
            compared to their "fair share" of total/N */
            let mut offset = (NClogN >> 1) - den * FINE_OFFSET;

            /* N=2 is the only point that doesn't match the curve */
            if N == 2 {
                offset += den << BITRES >> 2;
            }

            /* Changing the offset for allocating the second and third
            fine energy bit */
            if bits[ju] + offset < den * 2 << BITRES {
                offset += NClogN >> 2;
            } else if bits[ju] + offset < den * 3 << BITRES {
                offset += NClogN >> 3;
            }

            /* Divide with rounding */
            ebits[ju] = 0i32.max(bits[ju] + offset + (den << (BITRES - 1)));
            ebits[ju] = (celt_udiv(ebits[ju] as u32, den as u32) >> BITRES) as i32;

            /* Make sure not to bust */
            if C * ebits[ju] > (bits[ju] >> BITRES) {
                ebits[ju] = bits[ju] >> stereo >> BITRES;
            }

            /* More than that is useless because that's about as far as PVQ can go */
            ebits[ju] = ebits[ju].min(MAX_FINE_BITS);

            /* If we rounded down or capped this band, make it a candidate for the
            final fine energy pass */
            fine_priority[ju] = (ebits[ju] * (den << BITRES) >= bits[ju] + offset) as i32;

            /* Remove the allocated fine bits; the rest are assigned to PVQ */
            bits[ju] -= C * ebits[ju] << BITRES;
        } else {
            /* For N=1, all bits go to fine energy except for a single sign bit */
            excess = 0i32.max(bit - (C << BITRES));
            bits[ju] = bit - excess;
            ebits[ju] = 0;
            fine_priority[ju] = 1;
        }

        /* Fine energy can't take advantage of the re-balancing in
         quant_all_bands().
        Instead, do the re-balancing here.*/
        if excess > 0 {
            let extra_fine = (excess >> (stereo + BITRES)).min(MAX_FINE_BITS - ebits[ju]);
            ebits[ju] += extra_fine;
            let extra_bits = extra_fine * C << BITRES;
            fine_priority[ju] = (extra_bits >= excess - balance) as i32;
            excess -= extra_bits;
        }
        balance = excess;

        debug_assert!(bits[ju] >= 0);
        debug_assert!(ebits[ju] >= 0);
        j += 1;
    }
    /* Save any remaining bits over the cap for the rebalancing in
    quant_all_bands(). */
    *_balance = balance;

    /* The skipped bands use all their bits for fine energy. */
    while j < end {
        let ju = j as usize;
        ebits[ju] = bits[ju] >> stereo >> BITRES;
        debug_assert!(C * ebits[ju] << BITRES == bits[ju]);
        bits[ju] = 0;
        fine_priority[ju] = (ebits[ju] < 1) as i32;
        j += 1;
    }
    codedBands
}

/// Compute the pulse allocation, i.e. how many pulses will go in each band
/// (always encoding; `pulses` receives the bits per band).
/// Returns codedBands.
pub fn clt_compute_allocation(
    end: i32,
    offsets: &[i32],
    cap: &[i32],
    alloc_trim: i32,
    intensity: &mut i32,
    dual_stereo: &mut i32,
    mut total: i32,
    balance: &mut i32,
    pulses: &mut [i32],
    ebits: &mut [i32],
    fine_priority: &mut [i32],
    C: i32,
    ec: &mut ec_ctx,
    prev: i32,
) -> i32 {
    /* start == 0, LM == 3 */
    let start = 0;
    let LM = 3;
    let mut lo: i32;
    let mut hi: i32;
    let len = modes::nbEBands;

    total = total.max(0);
    let mut skip_start = start;
    /* Reserve a bit to signal the end of manually skipped bands. */
    let skip_rsv = if total >= 1 << BITRES { 1 << BITRES } else { 0 };
    total -= skip_rsv;
    /* Reserve bits for the intensity and dual stereo parameters. */
    let mut intensity_rsv = 0;
    let mut dual_stereo_rsv = 0;
    if C == 2 {
        intensity_rsv = LOG2_FRAC_TABLE[(end - start) as usize] as i32;
        if intensity_rsv > total {
            intensity_rsv = 0;
        } else {
            total -= intensity_rsv;
            dual_stereo_rsv = if total >= 1 << BITRES { 1 << BITRES } else { 0 };
            total -= dual_stereo_rsv;
        }
    }
    let st = crate::stack::save();
    let bits1 = st.alloc::<i32>(len as usize);
    let bits2 = st.alloc::<i32>(len as usize);
    let thresh = st.alloc::<i32>(len as usize);
    let trim_offset = st.alloc::<i32>(len as usize);
    let eu = end as usize;
    let (offsets, cap) = (&offsets[..eu], &cap[..eu]);
    let (bits1, bits2, thresh, trim_offset) = (
        &mut bits1[..eu],
        &mut bits2[..eu],
        &mut thresh[..eu],
        &mut trim_offset[..eu],
    );
    /* the allocation vectors, one row of nbEBands per quality level, cut to `end` */
    let alloc_rows = modes::allocVectors
        .as_chunks::<{ modes::nbEBands as usize }>()
        .0;
    let alloc = |q: i32| &alloc_rows[q as usize][..eu];

    for j in start..end {
        let ju = j as usize;
        let N = eb(j + 1) - eb(j);
        /* Below this threshold, we're sure not to allocate any PVQ bits */
        thresh[ju] = (C << BITRES).max((3 * N << LM << BITRES) >> 4);
        /* Tilt of the allocation curve */
        trim_offset[ju] = C * N * (alloc_trim - 5 - LM) * (end - j - 1) * (1 << (LM + BITRES)) >> 6;
        /* Giving less resolution to single-coefficient bands because they get
        more benefit from having one coarse value per coefficient*/
        if N << LM == 1 {
            trim_offset[ju] -= C << BITRES;
        }
    }
    lo = 1;
    hi = modes::nbAllocVectors - 1;
    loop {
        let mut done = false;
        let mut psum = 0;
        let mid = (lo + hi) >> 1;
        let alloc_mid = alloc(mid);
        let mut j = end;
        while j > start {
            j -= 1;
            let ju = j as usize;
            let N = eb(j + 1) - eb(j);
            let mut bitsj = C * N * (alloc_mid[ju] as i32) << LM >> 2;
            if bitsj > 0 {
                bitsj = 0i32.max(bitsj + trim_offset[ju]);
            }
            bitsj += offsets[ju];
            if bitsj >= thresh[ju] || done {
                done = true;
                /* Don't allocate more than we can actually use */
                psum += bitsj.min(cap[ju]);
            } else if bitsj >= C << BITRES {
                psum += C << BITRES;
            }
        }
        if psum > total {
            hi = mid - 1;
        } else {
            lo = mid + 1;
        }
        if lo > hi {
            break;
        }
    }
    hi = lo;
    lo -= 1;
    let alloc_lo = alloc(lo);
    /* hi == nbAllocVectors: the cap */
    let alloc_hi = alloc_rows.get(hi as usize).map(|r| &r[..eu]);
    for j in start..end {
        let ju = j as usize;
        let N = eb(j + 1) - eb(j);
        let mut bits1j = C * N * (alloc_lo[ju] as i32) << LM >> 2;
        let mut bits2j = match alloc_hi {
            None => cap[ju],
            Some(alloc_hi) => C * N * (alloc_hi[ju] as i32) << LM >> 2,
        };
        if bits1j > 0 {
            bits1j = 0i32.max(bits1j + trim_offset[ju]);
        }
        if bits2j > 0 {
            bits2j = 0i32.max(bits2j + trim_offset[ju]);
        }
        if lo > 0 {
            bits1j += offsets[ju];
        }
        bits2j += offsets[ju];
        if offsets[ju] > 0 {
            skip_start = j;
        }
        bits2j = 0i32.max(bits2j - bits1j);
        bits1[ju] = bits1j;
        bits2[ju] = bits2j;
    }
    interp_bits2pulses(
        end,
        skip_start,
        bits1,
        bits2,
        thresh,
        cap,
        total,
        balance,
        skip_rsv,
        intensity,
        intensity_rsv,
        dual_stereo,
        dual_stereo_rsv,
        pulses,
        ebits,
        fine_priority,
        C,
        ec,
        prev,
    )
}
