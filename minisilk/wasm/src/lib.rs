//! C-ABI exports for wasm32-unknown-unknown: one static 16 kHz mono 20 ms encoder.
//!
//! Usage from the host (single-threaded):
//! 1. `minisilk_init(bitrate, complexity, dtx)`: (re)initialises the encoder; returns 0 or a
//!    negative Opus error code. `bitrate` is bits/s, -1000 (`OPUS_AUTO`) or -1
//!    (`OPUS_BITRATE_MAX`).
//! 2. Write 320 samples to `minisilk_input_ptr()` (f32, nominal range +-1) or
//!    `minisilk_input_i16_ptr()` (i16), then call `minisilk_encode()` /
//!    `minisilk_encode_i16()`. The return value is the packet length (bytes at
//!    `minisilk_output_ptr()`, capacity 1276) or a negative Opus error code.
//! 3. `minisilk_set_*` mirror the matching `OPUS_SET_*` ctls and return 0 or a negative code.
//!    `minisilk_lookahead()` is `OPUS_GET_LOOKAHEAD` (samples at 16 kHz),
//!    `minisilk_final_range()` `OPUS_GET_FINAL_RANGE` and `minisilk_in_dtx()` `OPUS_GET_IN_DTX`.
#![no_std]

use core::ptr::addr_of_mut;
use minisilk::opus::encoder::OpusEncoder;

const FRAME: usize = 320;
const OUT_CAP: usize = 1276;

static mut ENCODER: OpusEncoder = OpusEncoder::new_zeroed();
static mut INITIALISED: bool = false;
static mut INPUT_F32: [f32; FRAME] = [0.0; FRAME];
static mut INPUT_I16: [i16; FRAME] = [0; FRAME];
static mut OUTPUT: [u8; OUT_CAP] = [0; OUT_CAP];

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    core::arch::wasm32::unreachable()
}

// SAFETY (all accessors): wasm32-unknown-unknown here is single-threaded and the exports are
// not re-entrant, so these statics are never aliased.
fn enc() -> &'static mut OpusEncoder {
    unsafe { &mut *addr_of_mut!(ENCODER) }
}

fn initialised() -> bool {
    unsafe { *addr_of_mut!(INITIALISED) }
}

#[unsafe(no_mangle)]
pub extern "C" fn minisilk_init(bitrate: i32, complexity: i32, dtx: i32) -> i32 {
    unsafe { *addr_of_mut!(INITIALISED) = false };
    let e = enc();
    e.init();
    let mut r = e.set_bitrate(bitrate);
    if r >= 0 {
        r = e.set_complexity(complexity);
    }
    if r >= 0 {
        r = e.set_dtx(dtx);
    }
    // Leave the encoder unusable unless every setting was accepted.
    if r >= 0 {
        unsafe { *addr_of_mut!(INITIALISED) = true };
    }
    r
}

#[unsafe(no_mangle)]
pub extern "C" fn minisilk_set_bitrate(v: i32) -> i32 {
    enc().set_bitrate(v)
}

#[unsafe(no_mangle)]
pub extern "C" fn minisilk_set_complexity(v: i32) -> i32 {
    enc().set_complexity(v)
}

#[unsafe(no_mangle)]
pub extern "C" fn minisilk_set_dtx(v: i32) -> i32 {
    enc().set_dtx(v)
}

#[unsafe(no_mangle)]
pub extern "C" fn minisilk_set_packet_loss_perc(v: i32) -> i32 {
    enc().set_packet_loss_perc(v)
}

#[unsafe(no_mangle)]
pub extern "C" fn minisilk_input_ptr() -> *mut f32 {
    addr_of_mut!(INPUT_F32) as *mut f32
}

#[unsafe(no_mangle)]
pub extern "C" fn minisilk_input_i16_ptr() -> *mut i16 {
    addr_of_mut!(INPUT_I16) as *mut i16
}

#[unsafe(no_mangle)]
pub extern "C" fn minisilk_output_ptr() -> *const u8 {
    addr_of_mut!(OUTPUT) as *const u8
}

#[unsafe(no_mangle)]
pub extern "C" fn minisilk_encode() -> i32 {
    if !initialised() {
        return -1; /* OPUS_BAD_ARG: not initialised */
    }
    unsafe { enc().encode_float(&*addr_of_mut!(INPUT_F32), &mut *addr_of_mut!(OUTPUT)) }
}

#[unsafe(no_mangle)]
pub extern "C" fn minisilk_encode_i16() -> i32 {
    if !initialised() {
        return -1; /* OPUS_BAD_ARG: not initialised */
    }
    unsafe { enc().encode(&*addr_of_mut!(INPUT_I16), &mut *addr_of_mut!(OUTPUT)) }
}

#[unsafe(no_mangle)]
pub extern "C" fn minisilk_lookahead() -> i32 {
    enc().get_lookahead()
}

#[unsafe(no_mangle)]
pub extern "C" fn minisilk_final_range() -> u32 {
    enc().get_final_range()
}

#[unsafe(no_mangle)]
pub extern "C" fn minisilk_in_dtx() -> i32 {
    enc().get_in_dtx()
}

/// High-water mark of the pseudostack in bytes (`stack-peak` feature only).
#[cfg(feature = "stack-peak")]
#[unsafe(no_mangle)]
pub extern "C" fn minisilk_stack_peak() -> u32 {
    minisilk::stack::peak() as u32
}
