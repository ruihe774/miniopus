//! C-ABI exports for wasm32-unknown-unknown: one static 48 kHz 20 ms CELT-only encoder.
//!
//! Usage from the host (single-threaded):
//! 1. `minicelt_init(channels, bitrate, complexity, vbr_constraint, dtx)`: (re)initialises the
//!    encoder for 1 or 2 channels; returns 0 or a negative Opus error code. `bitrate` is
//!    bits/s, -1000 (`OPUS_AUTO`) or -1 (`OPUS_BITRATE_MAX`). VBR is always on.
//! 2. Write 960 samples per channel, interleaved, to `minicelt_input_ptr()` (f32, nominal
//!    range +-1) or `minicelt_input_i16_ptr()` (i16), then call `minicelt_encode()` /
//!    `minicelt_encode_i16()`. The return value is the packet length (bytes at
//!    `minicelt_output_ptr()`, capacity 1276) or a negative Opus error code. Both input
//!    pointers are the same buffer (the i16 samples are converted in place) and the encoder
//!    works in it, so it must be rewritten before every call.
//! 3. `minicelt_set_*` mirror the matching `OPUS_SET_*` ctls and return 0 or a negative code.
//!    `minicelt_lookahead()` is `OPUS_GET_LOOKAHEAD` (samples at 48 kHz),
//!    `minicelt_final_range()` `OPUS_GET_FINAL_RANGE` and `minicelt_in_dtx()` `OPUS_GET_IN_DTX`.
#![no_std]

use core::ptr::addr_of_mut;
use minicelt::opus::encoder::OpusEncoder;

const FRAME: usize = 960;
const MAX_CHANNELS: usize = 2;
const OUT_CAP: usize = minicelt::opus::encoder::PACKET_SIZE_CAP;
/// 960 samples plus the 120 of overlap per channel
const IN_BUF_FRAME: usize = minicelt::opus::encoder::IN_BUF_FRAME as usize;

static mut ENCODER: OpusEncoder = OpusEncoder::new_zeroed();
static mut CHANNELS: usize = 0;
/// The input buffer (the first `960 * channels` samples); also the encoder's working memory
/// (room for `120` more samples per channel), and the i16 input is converted in place.
static mut INPUT: [f32; IN_BUF_FRAME * MAX_CHANNELS] = [0.0; IN_BUF_FRAME * MAX_CHANNELS];
static mut OUTPUT: [u8; OUT_CAP] = [0; OUT_CAP];

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    core::arch::wasm32::unreachable()
}

// SAFETY (all accessors): wasm32-unknown-unknown here is single-threaded and the exports are
// not re-entrant, so these statics are never aliased.
fn enc() -> &'static mut OpusEncoder {
    // UNSAFE-MAND: `&mut` to a `static mut`.
    unsafe { &mut *addr_of_mut!(ENCODER) }
}

fn channels() -> usize {
    // UNSAFE-EASY: make CHANNELS an `AtomicUsize` (relaxed), here and in `minicelt_init`.
    unsafe { *addr_of_mut!(CHANNELS) }
}

#[unsafe(no_mangle)] // UNSAFE-MAND: `no_mangle` is an unsafe attribute
pub extern "C" fn minicelt_init(channels: i32, bitrate: i32, complexity: i32, vbr_constraint: i32, dtx: i32) -> i32 {
    // UNSAFE-EASY: `AtomicUsize` store, see `channels`.
    unsafe { *addr_of_mut!(CHANNELS) = 0 };
    let e = enc();
    let mut r = e.init(channels);
    if r >= 0 {
        r = e.set_bitrate(bitrate);
    }
    if r >= 0 {
        r = e.set_complexity(complexity);
    }
    if r >= 0 {
        r = e.set_vbr_constraint(vbr_constraint);
    }
    if r >= 0 {
        r = e.set_dtx(dtx);
    }
    // Leave the encoder unusable unless every setting was accepted.
    if r >= 0 {
        // UNSAFE-EASY: `AtomicUsize` store, see `channels`.
        unsafe { *addr_of_mut!(CHANNELS) = channels as usize };
    }
    r
}

#[unsafe(no_mangle)] // UNSAFE-MAND: `no_mangle` is an unsafe attribute
pub extern "C" fn minicelt_set_bitrate(v: i32) -> i32 {
    enc().set_bitrate(v)
}

#[unsafe(no_mangle)] // UNSAFE-MAND: `no_mangle` is an unsafe attribute
pub extern "C" fn minicelt_set_complexity(v: i32) -> i32 {
    enc().set_complexity(v)
}

#[unsafe(no_mangle)] // UNSAFE-MAND: `no_mangle` is an unsafe attribute
pub extern "C" fn minicelt_set_vbr_constraint(v: i32) -> i32 {
    enc().set_vbr_constraint(v)
}

#[unsafe(no_mangle)] // UNSAFE-MAND: `no_mangle` is an unsafe attribute
pub extern "C" fn minicelt_set_dtx(v: i32) -> i32 {
    enc().set_dtx(v)
}

#[unsafe(no_mangle)] // UNSAFE-MAND: `no_mangle` is an unsafe attribute
pub extern "C" fn minicelt_set_packet_loss_perc(v: i32) -> i32 {
    enc().set_packet_loss_perc(v)
}

#[unsafe(no_mangle)] // UNSAFE-MAND: `no_mangle` is an unsafe attribute
pub extern "C" fn minicelt_set_bandwidth(v: i32) -> i32 {
    enc().set_bandwidth(v)
}

#[unsafe(no_mangle)] // UNSAFE-MAND: `no_mangle` is an unsafe attribute
pub extern "C" fn minicelt_input_ptr() -> *mut f32 {
    addr_of_mut!(INPUT) as *mut f32
}

#[unsafe(no_mangle)] // UNSAFE-MAND: `no_mangle` is an unsafe attribute
pub extern "C" fn minicelt_input_i16_ptr() -> *mut i16 {
    addr_of_mut!(INPUT) as *mut i16
}

#[unsafe(no_mangle)] // UNSAFE-MAND: `no_mangle` is an unsafe attribute
pub extern "C" fn minicelt_output_ptr() -> *const u8 {
    addr_of_mut!(OUTPUT) as *const u8
}

/// Encodes `960 * channels` interleaved samples from `minicelt_input_ptr()`.
#[unsafe(no_mangle)] // UNSAFE-MAND: `no_mangle` is an unsafe attribute
pub extern "C" fn minicelt_encode() -> i32 {
    if channels() == 0 {
        return -1; /* OPUS_BAD_ARG: not initialised */
    }
    // UNSAFE-WIDE-MAND: only the two `&mut *addr_of_mut!(..)` derefs of the statics need `unsafe`.
    unsafe { enc().encode_float(&mut *addr_of_mut!(INPUT), &mut *addr_of_mut!(OUTPUT), 24) }
}

/// Encodes `960 * channels` interleaved samples from `minicelt_input_i16_ptr()` (opus_encode:
/// INT16TORES scaling, 16-bit depth).
#[unsafe(no_mangle)] // UNSAFE-MAND: `no_mangle` is an unsafe attribute
pub extern "C" fn minicelt_encode_i16() -> i32 {
    let n = FRAME * channels();
    if n == 0 {
        return -1; /* OPUS_BAD_ARG: not initialised */
    }
    // UNSAFE-WIDE-MAND: the `&mut *addr_of_mut!(..)` derefs of the statics in the last line need `unsafe`; the in-place
    // i16 -> f32 loop is UNSAFE-EASY (read the i16 out of the f32 word `i / 2` with `to_bits`, going from the end).
    unsafe {
        let f = addr_of_mut!(INPUT) as *mut f32;
        let q = f as *const i16;
        /* in place, from the end: f[i] (4 bytes at 4i) only overwrites i16 samples already read */
        for i in (0..n).rev() {
            *f.add(i) = *q.add(i) as f32 * (1.0f32 / 32768.0f32);
        }
        enc().encode_float(&mut *addr_of_mut!(INPUT), &mut *addr_of_mut!(OUTPUT), 16)
    }
}

/// OPUS_GET_LOOKAHEAD: Fs/400 samples (no delay compensation for RESTRICTED_CELT).
#[unsafe(no_mangle)] // UNSAFE-MAND: `no_mangle` is an unsafe attribute
pub extern "C" fn minicelt_lookahead() -> i32 {
    120
}

#[unsafe(no_mangle)] // UNSAFE-MAND: `no_mangle` is an unsafe attribute
pub extern "C" fn minicelt_final_range() -> u32 {
    enc().get_final_range()
}

#[unsafe(no_mangle)] // UNSAFE-MAND: `no_mangle` is an unsafe attribute
pub extern "C" fn minicelt_in_dtx() -> i32 {
    enc().get_in_dtx()
}

/// Pseudostack high-water mark in bytes (measurement builds only: `--features stack-peak`).
#[cfg(feature = "stack-peak")]
#[unsafe(no_mangle)] // UNSAFE-MAND: `no_mangle` is an unsafe attribute
pub extern "C" fn minicelt_stack_peak() -> u32 {
    minicelt::stack::peak() as u32
}
