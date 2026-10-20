//! Port of celt/entcode.h and celt/entcode.c (encoder side).


/// The entropy encoder/decoder context.
pub struct ec_ctx {
    /// Buffered input/output.
    pub buf: *mut u8,
    /// The size of the buffer.
    pub storage: u32,
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

pub type ec_enc = ec_ctx;

/// EC_ILOG: number of bits needed to represent `v` (C: undefined for 0; 0 here).
#[inline(always)]
pub fn ec_ilog(v: u32) -> i32 {
    32 - v.leading_zeros() as i32
}

/// Returns the number of bits "used" by the encoded or decoded symbols so far.
#[inline(always)]
pub fn ec_tell(this: &ec_ctx) -> i32 {
    this.nbits_total - ec_ilog(this.rng)
}

