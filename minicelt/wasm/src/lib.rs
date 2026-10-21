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

use minicelt::global::Global;
use minicelt::opus::encoder::OpusEncoder;

const FRAME: usize = 960;
const MAX_CHANNELS: usize = 2;
const OUT_CAP: usize = minicelt::opus::encoder::PACKET_SIZE_CAP;
/// 960 samples plus the 120 of overlap per channel
const IN_BUF_FRAME: usize = minicelt::opus::encoder::IN_BUF_FRAME as usize;

static ENCODER: Global<OpusEncoder> = Global::new(OpusEncoder::new_zeroed());
static CHANNELS: Global<usize> = Global::new(0);
/// The input buffer (the first `960 * channels` samples); also the encoder's working memory
/// (room for `120` more samples per channel), and the i16 input is converted in place.
static INPUT: Global<[f32; IN_BUF_FRAME * MAX_CHANNELS]> = Global::new([0.0; IN_BUF_FRAME * MAX_CHANNELS]);
static OUTPUT: Global<[u8; OUT_CAP]> = Global::new([0; OUT_CAP]);

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    core::arch::wasm32::unreachable()
}

// wasm32-unknown-unknown here is single-threaded and the exports are not re-entrant, so these
// statics are never aliased (see `Global`).
fn enc() -> &'static mut OpusEncoder {
    ENCODER.get_mut()
}

fn input() -> &'static mut [f32; IN_BUF_FRAME * MAX_CHANNELS] {
    INPUT.get_mut()
}

fn output() -> &'static mut [u8; OUT_CAP] {
    OUTPUT.get_mut()
}

fn channels() -> usize {
    CHANNELS.get()
}

#[unsafe(no_mangle)] // UNSAFE-MAND: `no_mangle` is an unsafe attribute
pub extern "C" fn minicelt_init(channels: i32, bitrate: i32, complexity: i32, vbr_constraint: i32, dtx: i32) -> i32 {
    CHANNELS.set(0);
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
        CHANNELS.set(channels as usize);
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
    INPUT.as_ptr() as *mut f32
}

#[unsafe(no_mangle)] // UNSAFE-MAND: `no_mangle` is an unsafe attribute
pub extern "C" fn minicelt_input_i16_ptr() -> *mut i16 {
    INPUT.as_ptr() as *mut i16
}

#[unsafe(no_mangle)] // UNSAFE-MAND: `no_mangle` is an unsafe attribute
pub extern "C" fn minicelt_output_ptr() -> *const u8 {
    OUTPUT.as_ptr() as *const u8
}

/// Encodes `960 * channels` interleaved samples from `minicelt_input_ptr()`.
#[unsafe(no_mangle)] // UNSAFE-MAND: `no_mangle` is an unsafe attribute
pub extern "C" fn minicelt_encode() -> i32 {
    if channels() == 0 {
        return -1; /* OPUS_BAD_ARG: not initialised */
    }
    enc().encode_float(input(), output(), 24)
}

/// Encodes `960 * channels` interleaved samples from `minicelt_input_i16_ptr()` (opus_encode:
/// INT16TORES scaling, 16-bit depth).
#[unsafe(no_mangle)] // UNSAFE-MAND: `no_mangle` is an unsafe attribute
pub extern "C" fn minicelt_encode_i16() -> i32 {
    let n = FRAME * channels();
    if n == 0 {
        return -1; /* OPUS_BAD_ARG: not initialised */
    }
    let f = input();
    /* in place, from the end: f[i] (word i) only overwrites i16 samples (in word i / 2, little
    endian) that were already read */
    for i in (0..n).rev() {
        let w = f[i / 2].to_bits();
        let q = if i % 2 == 0 { w as u16 } else { (w >> 16) as u16 };
        f[i] = q as i16 as f32 * (1.0f32 / 32768.0f32);
    }
    enc().encode_float(f, output(), 16)
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

/// Calls `maxabs` `iters` times on `len` samples at `off` of the input buffer and returns the largest
/// result (measurement builds only: `--features bench-maxabs`, see minicelt/test/bench_maxabs.ts).
#[cfg(feature = "bench-maxabs")]
#[unsafe(no_mangle)] // UNSAFE-MAND: `no_mangle` is an unsafe attribute
pub extern "C" fn minicelt_bench_maxabs(off: i32, len: i32, iters: i32) -> f32 {
    let x = &input()[off as usize..][..len as usize];
    let mut m: f32 = 0.0;
    for _ in 0..iters {
        let v = minicelt::maxabs(core::hint::black_box(x));
        m = if v > m { v } else { m };
    }
    m
}
