//! Port of src/opus_encoder.c for `OPUS_APPLICATION_RESTRICTED_SILK`, specialised to
//! 16 kHz mono, wideband, 20 ms frames, VBR, DTX supported.
//!
//! What C executes in this configuration is ported as is; the rest is removed:
//! - no CELT/hybrid/redundancy/repacketizer/padding (CBR), no multi-frame packets, no FEC
//!   (`useInBandFEC == 0`, so `decide_fec` always yields `LBRR_coded = 0`);
//! - no analysis (`run_analysis` is never used for RESTRICTED_SILK) and no
//!   mode/bandwidth/channel decisions: they always give SILK-only, WB, mono. Their state
//!   (`peak_signal_energy`, `equiv_rate`, `auto_bandwidth`, `stereoWidth_Q14`, the
//!   `delay_buffer`, `variable_HP_smth2_Q15`, which only feeds the VOIP high-pass) cannot
//!   influence the output and is omitted;
//! - the bandwidth is fixed to WB as with `OPUS_SET_BANDWIDTH(OPUS_BANDWIDTH_WIDEBAND)`, so the
//!   ToC byte of an encoded packet is the constant `TOC_SILK_WB_20MS`;
//! - `silk_EncControlStruct` is gone: its inputs are arguments of `silk_Encode` (the packet loss
//!   percentage, complexity and SILK DTX flag live in this struct).
//!
//! Deviation from C: when `effective_max_rate < 8000` (an output buffer under 20 bytes)
//! C lowers SILK's maximum internal rate to 12/8 kHz, which minisilk cannot do. Such
//! buffers are rejected with `OPUS_BUFFER_TOO_SMALL`. Buffers under 3 bytes still yield
//! the one byte "PLC" (TOC only) packet as in C, since that path does not involve SILK.
//!
//! `OpusEncoder` is statically sizable (`new_zeroed` is `const`, no allocation); call
//! `init` before use.

use crate::celt::entcode::{ec_enc, ec_tell};
use crate::celt::entenc::{ec_enc_done, ec_enc_init};
use crate::celt::pitch::celt_inner_prod;
use crate::opus::defines::*;
use crate::silk::define::{MAX_CONSECUTIVE_DTX, NB_SPEECH_FRAMES_BEFORE_DTX, TYPE_NO_VOICE_ACTIVITY, VAD_NO_DECISION};
use crate::silk::enc_API::{silk_Encode, silk_InitEncoder};
use crate::silk::float::structs_flp::silk_encoder;

/* Fixed configuration. */
const FS: i32 = 16000;
const FRAME_SIZE: i32 = 320;
const FRAME_RATE: i32 = FS / FRAME_SIZE;
/// `st->Fs/400 + st->delay_compensation` with `delay_compensation = Fs/250`.
const LOOKAHEAD: i32 = FS / 400 + FS / 250;
const PACKET_SIZE_CAP: i32 = 1276;

/// `gen_toc(MODE_SILK_ONLY, 50, OPUS_BANDWIDTH_WIDEBAND, 1)`.
const TOC_SILK_WB_20MS: u8 = 0x48;
/// `gen_toc(MODE_HYBRID, 50, OPUS_BANDWIDTH_FULLBAND, 1)`: the ToC of a "PLC" packet emitted
/// before the first real frame, when C's initial mode/bandwidth are still in effect.
const TOC_HYBRID_FB_20MS: u8 = 0x78;

pub struct OpusEncoder {
    silk_enc: silk_encoder,
    /// `silk_mode.complexity`
    complexity: i32,
    /// `silk_mode.packetLossPercentage`
    packetLossPercentage: i32,
    /// `silk_mode.useDTX`
    silk_useDTX: i32,
    use_dtx: i32,
    user_bitrate_bps: i32,
    hp_mem: f32,
    mode: i32,
    prev_mode: i32,
    nb_no_activity_ms_Q1: i32,
    rangeFinal: u32,
}

#[inline]
fn bits_to_bitrate(bits: i32, Fs: i32, frame_size: i32) -> i32 {
    bits * (6 * Fs / frame_size) / 6
}

#[inline]
fn bitrate_to_bits(bitrate: i32, Fs: i32, frame_size: i32) -> i32 {
    bitrate * 6 / (6 * Fs / frame_size)
}

/// Float `dc_reject` (mono, 3 Hz cutoff). `hp_mem` is the filter state.
fn dc_reject(input: &[f32], out: &mut [f32], hp_mem: &mut f32) {
    const VERY_SMALL: f32 = 1e-30;
    let coef: f32 = 6.3f32 * 3 as f32 / FS as f32;
    let coef2: f32 = 1.0 - coef;
    let mut m0 = *hp_mem;
    for i in 0..FRAME_SIZE as usize {
        let x = input[i];
        let y = x - m0;
        m0 = coef * x + VERY_SMALL + coef2 * m0;
        out[i] = y;
    }
    *hp_mem = m0;
}

/// `celt_maxabs16` for floats, including its NaN behaviour (MAX16/MIN16 are `?:` macros).
fn celt_maxabs_res(x: &[f32]) -> f32 {
    let mut maxval = 0.0f32;
    let mut minval = 0.0f32;
    for &v in x {
        maxval = if maxval > v { maxval } else { v };
        minval = if minval < v { minval } else { v };
    }
    if maxval > -minval { maxval } else { -minval }
}

fn is_digital_silence(pcm: &[f32], lsb_depth: i32) -> bool {
    let sample_max = celt_maxabs_res(pcm);
    sample_max <= 1.0f32 / (1i32 << lsb_depth) as f32
}

/// Decides if DTX should be turned on (=1) or off (=0).
fn decide_dtx_mode(activity: i32, nb_no_activity_ms_Q1: &mut i32) -> i32 {
    if activity == 0 {
        /* The number of consecutive DTX frames should be within the allowed bounds.
        The allowed bound is defined in the SILK headers and assumes 20 ms frames. */
        *nb_no_activity_ms_Q1 += 2 * 1000 * FRAME_SIZE / FS;
        if *nb_no_activity_ms_Q1 > NB_SPEECH_FRAMES_BEFORE_DTX * 20 * 2 {
            if *nb_no_activity_ms_Q1 <= (NB_SPEECH_FRAMES_BEFORE_DTX + MAX_CONSECUTIVE_DTX) * 20 * 2 {
                /* Valid frame for DTX! */
                return 1;
            } else {
                *nb_no_activity_ms_Q1 = NB_SPEECH_FRAMES_BEFORE_DTX * 20 * 2;
            }
        }
    } else {
        *nb_no_activity_ms_Q1 = 0;
    }
    0
}

impl OpusEncoder {
    /// All-zero state, usable in a `static`. Call `init` before encoding.
    pub const fn new_zeroed() -> Self {
        // SAFETY: every field is an integer, float or array of those; zero is valid.
        unsafe { core::mem::zeroed() }
    }

    /// `opus_encoder_init(st, 16000, 1, OPUS_APPLICATION_RESTRICTED_SILK)` followed by
    /// `OPUS_SET_BANDWIDTH(WB)`. Defaults as in C: bitrate `OPUS_AUTO`, complexity 9, DTX off.
    pub fn init(&mut self) {
        // SAFETY: see `new_zeroed`.
        unsafe { core::ptr::write_bytes(self as *mut Self, 0, 1) };
        silk_InitEncoder(&mut self.silk_enc);
        self.complexity = 9;
        self.user_bitrate_bps = OPUS_AUTO;
        self.mode = MODE_HYBRID;
    }

    /// `OPUS_SET_BITRATE`: `OPUS_AUTO`, `OPUS_BITRATE_MAX` or a rate in bits/s (clamped to
    /// 500..=750000).
    pub fn set_bitrate(&mut self, value: i32) -> i32 {
        let mut value = value;
        if value != OPUS_AUTO && value != OPUS_BITRATE_MAX {
            if value <= 0 {
                return OPUS_BAD_ARG;
            } else if value <= 500 {
                value = 500;
            } else if value > 750000 {
                value = 750000;
            }
        }
        self.user_bitrate_bps = value;
        OPUS_OK
    }

    /// `OPUS_SET_COMPLEXITY` (0..=10).
    pub fn set_complexity(&mut self, value: i32) -> i32 {
        if value < 0 || value > 10 {
            return OPUS_BAD_ARG;
        }
        self.complexity = value;
        OPUS_OK
    }

    /// `OPUS_SET_DTX` (0 or 1).
    pub fn set_dtx(&mut self, value: i32) -> i32 {
        if value < 0 || value > 1 {
            return OPUS_BAD_ARG;
        }
        self.use_dtx = value;
        OPUS_OK
    }

    /// `OPUS_SET_PACKET_LOSS_PERC` (0..=100).
    pub fn set_packet_loss_perc(&mut self, value: i32) -> i32 {
        if value < 0 || value > 100 {
            return OPUS_BAD_ARG;
        }
        self.packetLossPercentage = value;
        OPUS_OK
    }

    /// `OPUS_GET_LOOKAHEAD` (samples at 16 kHz).
    pub fn get_lookahead(&self) -> i32 {
        LOOKAHEAD
    }

    /// `OPUS_GET_FINAL_RANGE`: range coder state after the last `encode*` call.
    pub fn get_final_range(&self) -> u32 {
        self.rangeFinal
    }

    /// `OPUS_GET_IN_DTX`.
    pub fn get_in_dtx(&self) -> i32 {
        if self.silk_useDTX != 0 && self.prev_mode == MODE_SILK_ONLY {
            /* DTX determined by Silk. */
            (self.silk_enc.state_Fxx.sCmn.noSpeechCounter >= NB_SPEECH_FRAMES_BEFORE_DTX) as i32
        } else if self.use_dtx != 0 {
            /* DTX determined by Opus. */
            (self.nb_no_activity_ms_Q1 >= NB_SPEECH_FRAMES_BEFORE_DTX * 20 * 2) as i32
        } else {
            0
        }
    }

    /// `opus_encode`: 320 samples of 16 kHz mono i16 PCM. Returns the packet length or a
    /// negative Opus error code.
    pub fn encode(&mut self, pcm: &[i16], out: &mut [u8]) -> i32 {
        if pcm.len() != FRAME_SIZE as usize {
            return OPUS_BAD_ARG;
        }
        let mut input = [0f32; FRAME_SIZE as usize];
        for i in 0..FRAME_SIZE as usize {
            /* INT16TORES */
            input[i] = pcm[i] as f32 * (1.0f32 / 32768.0f32);
        }
        self.encode_native(&input, out, 16)
    }

    /// `opus_encode_float`: 320 samples of 16 kHz mono f32 PCM (nominal range +-1).
    pub fn encode_float(&mut self, pcm: &[f32], out: &mut [u8]) -> i32 {
        if pcm.len() != FRAME_SIZE as usize {
            return OPUS_BAD_ARG;
        }
        self.encode_native(pcm, out, 24)
    }

    /// `opus_encode_native`, up to the frame encoder.
    fn encode_native(&mut self, pcm: &[f32], out: &mut [u8], lsb_depth: i32) -> i32 {
        /* Just avoid insane packet sizes here, but the real bounds are applied later on. */
        let max_data_bytes = (PACKET_SIZE_CAP * 6).min(out.len().min(i32::MAX as usize) as i32);

        self.rangeFinal = 0;
        if max_data_bytes <= 0 {
            return OPUS_BAD_ARG;
        }

        let is_silence = is_digital_silence(pcm, lsb_depth);

        let bitrate_bps = self.user_bitrate_to_bitrate(max_data_bytes);

        if max_data_bytes < 3 || bitrate_bps < 3 * FRAME_RATE * 8 {
            /* If the space is too low to do something useful, emit 'PLC' frames. */
            out[0] = if self.mode == MODE_SILK_ONLY { TOC_SILK_WB_20MS } else { TOC_HYBRID_FB_20MS };
            return 1;
        }

        /* Allow SILK DTX if DTX is enabled but the generalized DTX cannot be used,
        i.e. when the frame is not digital silence. */
        self.silk_useDTX = (self.use_dtx != 0 && !is_silence) as i32;

        /* Mode selection: RESTRICTED_SILK is always SILK-only, mono, WB (user bandwidth). */
        self.mode = MODE_SILK_ONLY;

        self.encode_frame_native(pcm, out, max_data_bytes, is_silence, bitrate_bps)
    }

    fn user_bitrate_to_bitrate(&self, max_data_bytes: i32) -> i32 {
        let max_bitrate = bits_to_bitrate(max_data_bytes * 8, FS, FRAME_SIZE);
        let user_bitrate = if self.user_bitrate_bps == OPUS_AUTO {
            60 * FS / FRAME_SIZE + FS
        } else if self.user_bitrate_bps == OPUS_BITRATE_MAX {
            1500000
        } else {
            self.user_bitrate_bps
        };
        user_bitrate.min(max_bitrate)
    }

    /// `opus_encode_frame_native`.
    fn encode_frame_native(&mut self, pcm: &[f32], out: &mut [u8], orig_max_data_bytes: i32, is_silence: bool, bitrate_bps: i32) -> i32 {
        let max_data_bytes = orig_max_data_bytes.min(PACKET_SIZE_CAP);
        self.rangeFinal = 0;

        let mut activity = VAD_NO_DECISION;
        if is_silence {
            activity = 0;
        }

        let effective_max_rate = bits_to_bitrate(max_data_bytes * 8, FS, FRAME_SIZE);
        if effective_max_rate < 8000 {
            /* C would lower SILK's max internal rate to 12/8 kHz here; unsupported. */
            return OPUS_BUFFER_TOO_SMALL;
        }

        let bits_target = (8 * max_data_bytes).min(bitrate_to_bits(bitrate_bps, FS, FRAME_SIZE)) - 8;

        /* DC reject (not the VOIP high-pass); delay_compensation is 0 for RESTRICTED_SILK. */
        let mut pcm_buf = [0f32; FRAME_SIZE as usize];
        dc_reject(pcm, &mut pcm_buf, &mut self.hp_mem);

        /* This should filter out both NaNs and ridiculous signals that could
        cause NaNs further down. */
        let sum = celt_inner_prod(&pcm_buf, &pcm_buf, FRAME_SIZE);
        if !(sum < 1e9f32) || sum != sum {
            pcm_buf = [0f32; FRAME_SIZE as usize];
            self.hp_mem = 0.0;
        }

        /* Call SILK encoder. The range coder writes after the ToC byte. */
        let mut enc: ec_enc = unsafe { core::mem::zeroed() };
        // SAFETY: out.len() >= orig_max_data_bytes >= 3; the coder gets
        // orig_max_data_bytes - 1 bytes starting at out[1].
        unsafe { ec_enc_init(&mut enc, out.as_mut_ptr().add(1), (orig_max_data_bytes - 1) as u32) };
        let nBytes = silk_Encode(
            &mut self.silk_enc,
            &pcm_buf,
            &mut enc,
            /* SILK gets all bits */
            bits_to_bitrate(bits_target, FS, FRAME_SIZE),
            /* Max bits for SILK, counting the ToC. */
            (max_data_bytes - 1) * 8,
            self.complexity,
            self.packetLossPercentage,
            self.silk_useDTX,
            activity,
        );

        if activity == VAD_NO_DECISION {
            activity = (self.silk_enc.state_Fxx.sCmn.indices.signalType as i32 != TYPE_NO_VOICE_ACTIVITY) as i32;
        }
        if nBytes == 0 {
            self.rangeFinal = 0;
            out[0] = TOC_SILK_WB_20MS;
            return 1;
        }

        /* SILK-only: the redundancy flag is inferred from the length (no redundancy). */
        let mut ret = (ec_tell(&enc) + 7) >> 3;
        ec_enc_done(&mut enc);
        self.rangeFinal = enc.rng;

        /* Signalling the mode in the first byte */
        out[0] = TOC_SILK_WB_20MS;

        self.prev_mode = self.mode;

        /* DTX decision */
        if self.use_dtx != 0 && self.silk_useDTX == 0 {
            if decide_dtx_mode(activity, &mut self.nb_no_activity_ms_Q1) != 0 {
                self.rangeFinal = 0;
                return 1;
            }
        } else {
            self.nb_no_activity_ms_Q1 = 0;
        }

        /* In the unlikely case that the SILK encoder busted its target, tell
        the decoder to call the PLC */
        if ec_tell(&enc) > (max_data_bytes - 1) * 8 {
            if max_data_bytes < 2 {
                return OPUS_BUFFER_TOO_SMALL;
            }
            out[1] = 0;
            ret = 1;
            self.rangeFinal = 0;
        } else {
            /* When in LPC only mode it's perfectly reasonable to strip off trailing zero
            bytes as the required range decoder behavior is to fill these in. */
            while ret > 2 && out[ret as usize] == 0 {
                ret -= 1;
            }
        }
        /* Count ToC */
        ret + 1
    }
}
