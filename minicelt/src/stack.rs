//! Port of the `NONTHREADSAFE_PSEUDOSTACK` mode of celt/stack_alloc.h: one static scratch
//! buffer with a bump pointer.
//!
//! C → Rust:
//! ```text
//! SAVE_STACK;                     let st = stack::save();
//! VARDECL(opus_int32, x);
//! ALLOC(x, n, opus_int32);        let x = st.alloc::<i32>(n);
//! RESTORE_STACK;                  (st dropped at end of scope)
//! ```
//! Allocations borrow the guard, so they cannot outlive the matching RESTORE_STACK. Like C,
//! memory is not cleared; `Pod` restricts allocations to types for which every bit pattern
//! is valid. Not thread-safe: the encoder must only run on one thread at a time.

use core::cell::UnsafeCell;
use core::marker::PhantomData;
use core::mem::{align_of, size_of};
use core::sync::atomic::{AtomicUsize, Ordering::Relaxed};

/// Scratch size in bytes: the measured worst-case peak, 16376 bytes, rounded up to the 16-byte
/// alignment of the buffer.
///
/// The peak is reached in every stereo frame (it does not depend on the signal or the settings) when
/// `run_prefilter` allocates its `CC*(960+1024)` float `_pre` buffer (15872 bytes) on top of the
/// 504 bytes of band energies of `celt_encode_with_ec`; the other phases stay below: the MDCT
/// (15864 for stereo), the pitch analysis, the band quantisation with the stereo theta RDO
/// (about 13 KB), mono encoders (about half). The PCM/`in`/`X` buffer lives in the caller's input
/// buffer instead of here (see `OpusEncoder::encode_float`).
///
/// How it was measured: `minicelt/test/stack_peak.ts` (header: build with the `stack-peak` feature,
/// which tracks the high-water mark of `alloc` and exports it as `minicelt_stack_peak()`, then run
/// the 2592-configuration sweep). It reads 16376 for both variants and both profiles. (Before the
/// trim: 54144 on a similar corpus, 57344 allocated.)
pub const GLOBAL_STACK_SIZE: usize = 16384;

#[repr(C, align(16))]
struct Scratch {
    buf: UnsafeCell<[u8; GLOBAL_STACK_SIZE]>,
    top: AtomicUsize,
    #[cfg(feature = "stack-peak")]
    peak: AtomicUsize,
}

// SAFETY: single-threaded use only, as with NONTHREADSAFE_PSEUDOSTACK in C.
// UNSAFE-MAND: `UnsafeCell` statics need a manual `Sync` impl.
unsafe impl Sync for Scratch {}

static SCRATCH: Scratch = Scratch {
    buf: UnsafeCell::new([0; GLOBAL_STACK_SIZE]),
    top: AtomicUsize::new(0),
    #[cfg(feature = "stack-peak")]
    peak: AtomicUsize::new(0),
};

/// Types for which every bit pattern is a valid value.
///
/// # Safety
/// Implementors must be plain data with no invalid bit patterns and no padding.
// UNSAFE-MAND: an unsafe marker trait is the contract that makes `alloc` sound.
pub unsafe trait Pod: Copy {}

macro_rules! pod {
    // UNSAFE-MAND: implementing the unsafe trait `Pod`.
    ($($t:ty),*) => { $(unsafe impl Pod for $t {})* };
}
pod!(
    i8,
    i16,
    i32,
    i64,
    u8,
    u16,
    u32,
    u64,
    f32,
    f64,
    core::arch::wasm32::v128
);
// UNSAFE-MAND: implementing the unsafe trait `Pod`.
unsafe impl<T: Pod, const N: usize> Pod for [T; N] {}

/// SAVE_STACK; dropping it is RESTORE_STACK.
pub struct StackGuard {
    saved: usize,
    _not_send: PhantomData<*mut u8>,
}

/// SAVE_STACK
#[inline(always)]
pub fn save() -> StackGuard {
    StackGuard {
        saved: SCRATCH.top.load(Relaxed),
        _not_send: PhantomData,
    }
}

impl StackGuard {
    /// ALLOC(var, n, T): `n` uninitialized-in-the-C-sense (stale but valid) elements.
    #[inline(always)]
    #[allow(clippy::mut_from_ref)]
    pub fn alloc<T: Pod>(&self, n: usize) -> &mut [T] {
        let base = SCRATCH.buf.get() as *mut u8;
        let start = (SCRATCH.top.load(Relaxed) + align_of::<T>() - 1) & !(align_of::<T>() - 1);
        let end = start + n * size_of::<T>();
        if end > GLOBAL_STACK_SIZE {
            panic!("pseudostack overflow");
        }
        SCRATCH.top.store(end, Relaxed);
        #[cfg(feature = "stack-peak")]
        SCRATCH.peak.fetch_max(end, Relaxed);
        // UNSAFE-MAND: hands out `&mut` from `&self` into the static buffer; `start..end` is in bounds,
        // aligned for `T`, and disjoint from live allocations (the bump pointer only moves up while a guard lives).
        unsafe { core::slice::from_raw_parts_mut(base.add(start) as *mut T, n) }
    }
}

impl Drop for StackGuard {
    #[inline(always)]
    fn drop(&mut self) {
        SCRATCH.top.store(self.saved, Relaxed);
    }
}

/// High-water mark of the scratch buffer in bytes (`stack-peak` feature only).
#[cfg(feature = "stack-peak")]
pub fn peak() -> usize {
    SCRATCH.peak.load(Relaxed)
}
