//! `Global<T>`: interior mutability for `static`s, for this single-threaded crate.
//!
//! The module is wasm32 without threads and the exports are not re-entrant, so a `static` is never
//! accessed concurrently. `Global` encapsulates that assumption (the only `unsafe` of the crate's
//! statics), so users are safe code. Not for use across threads.

use core::cell::UnsafeCell;

pub struct Global<T>(UnsafeCell<T>);

// UNSAFE-MAND: `UnsafeCell` statics need a manual `Sync` impl.
// SAFETY: single-threaded use only, as with NONTHREADSAFE_PSEUDOSTACK in C.
unsafe impl<T> Sync for Global<T> {}

impl<T> Global<T> {
    pub const fn new(v: T) -> Self {
        Global(UnsafeCell::new(v))
    }

    /// Raw pointer to the value.
    #[inline(always)]
    pub const fn as_ptr(&self) -> *mut T {
        self.0.get()
    }

    /// Shared view. Do not hold it across a `get_mut` of the same global.
    #[inline(always)]
    pub fn get_ref(&self) -> &T {
        // UNSAFE-MAND: reference out of the `UnsafeCell`.
        // SAFETY: single-threaded; by convention no `get_mut` borrow is live (init-only writers).
        unsafe { &*self.0.get() }
    }

    /// Exclusive view. Do not hold it across any other access to the same global.
    #[inline(always)]
    #[allow(clippy::mut_from_ref)]
    pub fn get_mut(&self) -> &mut T {
        // UNSAFE-MAND: `&mut` out of the `UnsafeCell`.
        // SAFETY: single-threaded; by convention no other borrow of this global is live.
        unsafe { &mut *self.0.get() }
    }
}

impl<T: Copy> Global<T> {
    #[inline(always)]
    pub fn get(&self) -> T {
        // UNSAFE-MAND: read through the `UnsafeCell`.
        // SAFETY: single-threaded; no borrow of the value is live across this plain copy.
        unsafe { *self.0.get() }
    }

    #[inline(always)]
    pub fn set(&self, v: T) {
        // UNSAFE-MAND: write through the `UnsafeCell`.
        // SAFETY: as `get`.
        unsafe { *self.0.get() = v }
    }
}
