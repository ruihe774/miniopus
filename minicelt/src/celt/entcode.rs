//! Port of celt/entcode.h and celt/entcode.c (encoder side).

pub type ec_window = u32;

pub const EC_WINDOW_SIZE: i32 = 32;
/// The number of bits to use for the range-coded part of unsigned integers.
pub const EC_UINT_BITS: i32 = 8;
/// The resolution of fractional-precision bit usage measurements, i.e., 3 => 1/8th bits.
pub const BITRES: i32 = 3;

/// The entropy encoder/decoder context. Copied by value (struct assignment) to save and
/// restore the coder state.
#[derive(Clone, Copy)]
pub struct ec_ctx {
    /// Buffered input/output.
    pub buf: *mut u8,
    /// The size of the buffer.
    pub storage: u32,
    /// The offset at which the last byte containing raw bits was read/written.
    pub end_offs: u32,
    /// Bits that will be read from/written at the end.
    pub end_window: ec_window,
    /// Number of valid bits in end_window.
    pub nend_bits: i32,
    /// The total number of whole bits read/written.
    /// This does not include partial bits currently in the range coder.
    pub nbits_total: i32,
    /// The offset at which the next range coder byte will be read/written.
    pub offs: u32,
    /// The number of values in the current range.
    pub rng: u32,
    /// In the encoder: the low end of the current range.
    pub val: u32,
    /// In the encoder: the number of outstanding carry propagating symbols.
    pub ext: u32,
    /// A buffered input/output symbol, awaiting carry propagation.
    pub rem: i32,
    /// Nonzero if an error occurred.
    pub error: i32,
}

impl ec_ctx {
    /// All-zero context (null buffer); `ec_enc_init` sets it up.
    pub const fn new_zeroed() -> Self {
        ec_ctx {
            buf: core::ptr::null_mut(),
            storage: 0,
            end_offs: 0,
            end_window: 0,
            nend_bits: 0,
            nbits_total: 0,
            offs: 0,
            rng: 0,
            val: 0,
            ext: 0,
            rem: 0,
            error: 0,
        }
    }
}

pub type ec_enc = ec_ctx;

/// EC_ILOG: number of bits needed to represent `v` (C: undefined for 0; 0 here).
#[inline(always)]
pub fn ec_ilog(v: u32) -> i32 {
    32 - v.leading_zeros() as i32
}

#[inline(always)]
pub fn EC_ILOG(v: u32) -> i32 {
    ec_ilog(v)
}

#[inline(always)]
pub fn ec_range_bytes(this: &ec_ctx) -> u32 {
    this.offs
}

#[inline(always)]
pub fn ec_get_buffer(this: &ec_ctx) -> *mut u8 {
    this.buf
}

#[inline(always)]
pub fn ec_get_error(this: &ec_ctx) -> i32 {
    this.error
}

/// Returns the number of bits "used" by the encoded or decoded symbols so far.
#[inline(always)]
pub fn ec_tell(this: &ec_ctx) -> i32 {
    this.nbits_total - ec_ilog(this.rng)
}

/// Returns the number of bits used so far, scaled by 2**BITRES.
#[inline(always)]
pub fn ec_tell_frac(this: &ec_ctx) -> u32 {
    /* This is a faster version of ec_tell_frac() that takes advantage
    of the low (1/8 bit) resolution to use just a linear function
    followed by a lookup to determine the exact transition thresholds. */
    static correction: [u32; 8] = [35733, 38967, 42495, 46340, 50535, 55109, 60097, 65535];
    let nbits = (this.nbits_total as u32) << BITRES;
    let mut l = ec_ilog(this.rng);
    let r = this.rng >> (l - 16);
    let mut b = ((r >> 12) - 8) as usize;
    /* r is 16 bits with the top bit set, so b < 8 */
    b += (r > correction[b & 7]) as usize;
    l = (l << 3) + b as i32;
    nbits.wrapping_sub(l as u32)
}

/// celt_udiv (no small-division table on wasm).
#[inline(always)]
pub fn celt_udiv(n: u32, d: u32) -> u32 {
    n / d
}

#[inline(always)]
pub fn celt_sudiv(n: i32, d: i32) -> i32 {
    n / d
}
