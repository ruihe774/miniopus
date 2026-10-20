//! Port of celt/entenc.c (with the constants of celt/mfrngcod.h).


use crate::celt::entcode::*;

/// The number of bits to output at a time.
pub const EC_SYM_BITS: i32 = 8;
/// The total number of bits in each of the state registers.
pub const EC_CODE_BITS: i32 = 32;
/// The maximum symbol value.
pub const EC_SYM_MAX: u32 = (1u32 << EC_SYM_BITS) - 1;
/// Bits to shift by to move a symbol into the high-order position.
pub const EC_CODE_SHIFT: i32 = EC_CODE_BITS - EC_SYM_BITS - 1;
/// Carry bit of the high-order range symbol.
pub const EC_CODE_TOP: u32 = 1u32 << (EC_CODE_BITS - 1);
/// Low-order bit of the high-order range symbol.
pub const EC_CODE_BOT: u32 = EC_CODE_TOP >> EC_SYM_BITS;

/* A range encoder.
See entdec.c and the references for implementation details. */

fn ec_write_byte(this: &mut ec_enc, value: u32) -> i32 {
    if this.offs >= this.storage {
        return -1;
    }
    unsafe { *this.buf.add(this.offs as usize) = value as u8 };
    this.offs += 1;
    0
}

/* Outputs a symbol, with a carry bit.
If there is a potential to propagate a carry over several symbols, they are
buffered until it can be determined whether or not an actual carry will
occur.
If the counter for the buffered symbols overflows, then the stream becomes
undecodable. */
fn ec_enc_carry_out(this: &mut ec_enc, c: i32) {
    if c as u32 != EC_SYM_MAX {
        /* No further carry propagation possible, flush buffer. */
        let carry = c >> EC_SYM_BITS;
        /* Don't output a byte on the first write. */
        if this.rem >= 0 {
            this.error |= ec_write_byte(this, (this.rem + carry) as u32);
        }
        if this.ext > 0 {
            let sym = (EC_SYM_MAX.wrapping_add(carry as u32)) & EC_SYM_MAX;
            loop {
                this.error |= ec_write_byte(this, sym);
                this.ext -= 1;
                if this.ext == 0 {
                    break;
                }
            }
        }
        this.rem = c & EC_SYM_MAX as i32;
    } else {
        this.ext += 1;
    }
}

#[inline(always)]
fn ec_enc_normalize(this: &mut ec_enc) {
    /* If the range is too small, output some bits and rescale it. */
    while this.rng <= EC_CODE_BOT {
        ec_enc_carry_out(this, (this.val >> EC_CODE_SHIFT) as i32);
        /* Move the next-to-high-order symbol into the high-order position. */
        this.val = (this.val << EC_SYM_BITS) & (EC_CODE_TOP - 1);
        this.rng <<= EC_SYM_BITS;
        this.nbits_total += EC_SYM_BITS;
    }
}

/// Initializes the encoder. `buf` must point to `size` writable bytes (or be null with
/// `size == 0`).
pub fn ec_enc_init(this: &mut ec_enc, buf: *mut u8, size: u32) {
    this.buf = buf;
    /* This is the offset from which ec_tell() will subtract partial bits. */
    this.nbits_total = EC_CODE_BITS + 1;
    this.offs = 0;
    this.rng = EC_CODE_TOP;
    this.rem = -1;
    this.val = 0;
    this.ext = 0;
    this.storage = size;
    this.error = 0;
}

pub fn ec_enc_icdf(this: &mut ec_enc, s: i32, icdf: &[u8], ftb: u32) {
    debug_assert!((s as usize) < icdf.len());
    unsafe { ec_enc_icdf_ptr(this, s, icdf.as_ptr(), ftb) }
}

/// `ec_enc_icdf` taking a raw table pointer (for tables reached through C-style pointers).
///
/// # Safety
/// `icdf` must be valid for reads of indices `s-1` and `s`.
pub unsafe fn ec_enc_icdf_ptr(this: &mut ec_enc, s: i32, icdf: *const u8, ftb: u32) {
    let r = this.rng >> ftb;
    unsafe {
        if s > 0 {
            let a = *icdf.add(s as usize - 1) as u32;
            let b = *icdf.add(s as usize) as u32;
            this.val = this.val.wrapping_add(this.rng.wrapping_sub(r.wrapping_mul(a)));
            this.rng = r.wrapping_mul(a - b);
        } else {
            this.rng = this.rng.wrapping_sub(r.wrapping_mul(*icdf as u32));
        }
    }
    ec_enc_normalize(this);
}

pub fn ec_enc_patch_initial_bits(this: &mut ec_enc, val: u32, nbits: u32) {
    debug_assert!(nbits <= EC_SYM_BITS as u32);
    let shift = EC_SYM_BITS as u32 - nbits;
    let mask: u32 = ((1u32 << nbits) - 1) << shift;
    if this.offs > 0 {
        /* The first byte has been finalized. */
        unsafe { *this.buf = ((*this.buf as u32 & !mask) | (val << shift)) as u8 };
    } else if this.rem >= 0 {
        /* The first byte is still awaiting carry propagation. */
        this.rem = ((this.rem as u32 & !mask) | (val << shift)) as i32;
    } else if this.rng <= (EC_CODE_TOP >> nbits) {
        /* The renormalization loop has never been run. */
        this.val = (this.val & !(mask << EC_CODE_SHIFT)) | (val << (EC_CODE_SHIFT as u32 + shift));
    } else {
        /* The encoder hasn't even encoded _nbits of data yet. */
        this.error = -1;
    }
}

pub fn ec_enc_done(this: &mut ec_enc) {
    /* We output the minimum number of bits that ensures that the symbols encoded
    thus far will be decoded correctly regardless of the bits that follow. */
    let mut l = EC_CODE_BITS - ec_ilog(this.rng);
    let mut msk = (EC_CODE_TOP - 1) >> l;
    let mut end = this.val.wrapping_add(msk) & !msk;
    if (end | msk) >= this.val.wrapping_add(this.rng) {
        l += 1;
        msk >>= 1;
        end = this.val.wrapping_add(msk) & !msk;
    }
    while l > 0 {
        ec_enc_carry_out(this, (end >> EC_CODE_SHIFT) as i32);
        end = (end << EC_SYM_BITS) & (EC_CODE_TOP - 1);
        l -= EC_SYM_BITS;
    }
    /* If we have a buffered byte flush it into the output buffer. */
    if this.rem >= 0 || this.ext > 0 {
        ec_enc_carry_out(this, 0);
    }
    /* Clear any excess space (no raw bits are ever written at the end of the buffer). */
    if this.error == 0 && !this.buf.is_null() {
        unsafe {
            core::ptr::write_bytes(this.buf.add(this.offs as usize), 0, (this.storage - this.offs) as usize);
        }
    }
}
