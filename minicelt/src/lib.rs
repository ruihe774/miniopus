//! Minimal Opus encoder: a mechanical Rust port of the libopus CELT-only float encoder
//! (OPUS_APPLICATION_RESTRICTED_CELT) for wasm32, fixed to 48 kHz, mono or stereo, 20 ms
//! frames and (constrained) VBR, with DTX and without the tonality analysis. `no_std`, no
//! allocation; scratch memory is a static pseudostack (see `stack`).

#![no_std]
#![allow(non_snake_case, non_camel_case_types, non_upper_case_globals)]
#![allow(clippy::all)]

#[cfg(not(target_arch = "wasm32"))]
compile_error!("minicelt only supports wasm32 (build with -Ctarget-feature=+simd128)");
#[cfg(not(target_feature = "simd128"))]
compile_error!("minicelt requires -Ctarget-feature=+simd128");

mod celt;
#[cfg(feature = "bench-maxabs")]
pub use celt::mathops::maxabs;
pub mod global;
pub mod opus;
pub mod stack;
