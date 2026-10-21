//! Port of the CELT-only path of src/opus_encoder.c (OPUS_APPLICATION_RESTRICTED_CELT),
//! specialised to 48 kHz, mono or stereo (chosen at init), 20 ms frames, VBR (constrained or
//! not), DTX supported, float build without the tonality analysis.
//!
//! What C executes in this configuration is ported as is; the rest is removed:
//! - the mode is always MODE_CELT_ONLY (RESTRICTED_CELT), so there is no SILK, hybrid,
//!   redundancy (`prev_mode` is only ever 0 or CELT-only), prefill (`to_celt`, `tmp_prefill`
//!   and the CELT prefill need a mode change or a non-RESTRICTED_CELT application), FEC
//!   (`decide_fec` returns 0 for CELT-only), DRED, QEXT, LFE, surround energy masking;
//! - one 20 ms frame per packet: `frame_size_select` with OPUS_FRAMESIZE_ARG and 960 samples is
//!   just a length check, no repacketizer; VBR only (`use_vbr == 1`): no CBR, no padding; the
//!   output buffer is the 1276-byte maximum Opus packet, so `max_data_bytes` is 1276;
//! - the ctls and settings that the wasm exports do not offer keep their defaults:
//!   `force_channels == OPUS_AUTO`, `signal_type == OPUS_AUTO`, `max_bandwidth ==
//!   OPUS_BANDWIDTH_FULLBAND`, `lsb_depth == 24`, `OPUS_SET_PREDICTION_DISABLED` off
//!   (`reducedDependency == 0`), phase inversion enabled, `OPUS_SET_VOICE_RATIO` unset
//!   (`voice_est == 48` in every decision, so the mono/stereo and bandwidth thresholds are
//!   constants below), `use_vbr == 1`;
//! - no analysis: `analysis_info.valid` is always 0 (`run_analysis` would be called at
//!   complexity >= 7, but the reference replaces it with a stub). Removed with it:
//!   `detected_bandwidth`, the voice_ratio estimate, CELT_SET_ANALYSIS (the CELT state's
//!   `analysis` stays zeroed);
//! - state and computations that cannot influence the output in this configuration:
//!   * `encoder_buffer == 0` and `delay_compensation` unused (RESTRICTED_CELT), so the
//!     `delay_buffer` copies move zero samples;
//!   * `variable_HP_smth2_Q15`/`cutoff_Hz` only feed the VOIP high-pass (`hp_cutoff`);
//!   * `HB_gain == prev_HB_gain == Q15ONE` (hybrid only), so `gain_fade` never runs;
//!   * `compute_stereo_width`/`width_mem`: the width only feeds the AUDIO/VOIP mode decision;
//!   * the activity decision (`peak_signal_energy`, `compute_frame_energy`): the DTX decision
//!     only ever sees `activity == 0` (digital silence), see `encode_frame_native`;
//!   * `max_rate` (hybrid bandwidth cap), `prev_channels`/`prev_framesize`/`silk_bw_switch`/
//!     `nonfinal_frame`/`prev_mode` and the SILK control fields.
//!
//! `OpusEncoder` is statically sizable (`new_zeroed` is `const`, no allocation, all zero so it
//! lives in `.bss`); call `init` before use.
//!
//! The buffer passed to `encode_float` holds the PCM and is the working buffer of the whole
//! encoder: the PCM is high-pass filtered in place (the C code copies it to the stack), the CELT
//! `in` buffer and the normalised spectrum are built in it (see `celt_encode_with_ec`), so its
//! contents are clobbered. It has room for `channels * (960 + overlap)` samples: this keeps
//! those two buffers (8640 bytes for stereo) out of the pseudostack.

use crate::celt::celt_encoder::{CELTEncoder, celt_encode_with_ec, celt_encoder_init};
use crate::celt::entcode::{ec_enc, ec_tell};
use crate::celt::entenc::ec_enc_init;
use crate::celt::mathops::{VERY_SMALL, maxabs};
use crate::celt::mathops::{imax, imin};
use crate::celt::modes;
use crate::celt::pitch::celt_inner_prod;
use crate::opus::defines::*;

/* Fixed configuration. */
const FS: i32 = 48000;
const FRAME_SIZE: i32 = 960;
/// 1276 bytes: the largest Opus frame, the size of the packet buffer.
pub const PACKET_SIZE_CAP: usize = 1276;
/// Samples per channel in the working buffer: the 960 of a frame plus the CELT overlap (the `in`
/// buffer of `celt_encode_with_ec`).
pub const IN_BUF_FRAME: i32 = FRAME_SIZE + modes::overlap;

/* From silk/define.h */
const NB_SPEECH_FRAMES_BEFORE_DTX: i32 = 10; /* eq 200 ms */
const MAX_CONSECUTIVE_DTX: i32 = 20; /* eq 400 ms */

/* Transition tables for the voice and music, interpolated with voice_est == 48 (Q7):
`music + ((voice_est^2 * (voice - music)) >> 14)`. First column is the middle (memoriless)
threshold. The second column is the hysteresis (difference with the middle). They are the same
for mono and stereo. The inputs (voice / music) are
  voice: 9000/700 NB<->MB, 9000/700 MB<->WB, 13500/1000 WB<->SWB, 14000/2000 SWB<->FB
  music: 9000/700 NB<->MB, 9000/700 MB<->WB, 11000/1000 WB<->SWB, 12000/2000 SWB<->FB */
static bandwidth_thresholds: [i32; 8] = [
    9000,
    700, /* NB<->MB */
    9000,
    700, /* MB<->WB */
    11000 + ((48 * 48 * (13500 - 11000)) >> 14),
    1000, /* WB<->SWB */
    12000 + ((48 * 48 * (14000 - 12000)) >> 14),
    2000, /* SWB<->FB */
];
/* Threshold bit-rate for switching between mono and stereo (music + voice interpolation) */
const stereo_threshold: i32 = 17000 + ((48 * 48 * (19000 - 17000)) >> 14);

/// TOC byte of a 20 ms CELT-only frame (the OPUS_BANDWIDTH_* value maps to the CELT bandwidth
/// field, mediumband and narrowband both to 0).
fn celt_toc(bandwidth: i32, channels: i32) -> u8 {
    let tmp = imax(bandwidth - OPUS_BANDWIDTH_MEDIUMBAND, 0);
    (0x80 | tmp << 5 | 3 << 3 | ((channels == 2) as i32) << 2) as u8
}

pub struct OpusEncoder {
    celt_enc: CELTEncoder,
    channels: i32,
    user_bandwidth: i32,
    vbr_constraint: i32,
    user_bitrate_bps: i32,
    /// general DTX for both SILK and CELT
    use_dtx: i32,

    stream_channels: i32,
    hybrid_stereo_width_Q14: i16,
    hp_mem: [f32; 4],
    /// Initially MODE_HYBRID (as C), MODE_CELT_ONLY after the first encoded frame; only the
    /// "PLC" frame emitted at very low bitrates depends on it.
    mode: i32,
    bandwidth: i32,
    /// Bandwidth determined automatically from the rate (before any other adjustment)
    auto_bandwidth: i32,
    first: i32,
    nb_no_activity_ms_Q1: i32,
    rangeFinal: u32,
}

/// Float `dc_reject` (cutoff 3 Hz at 48 kHz), in place: one-pole high-pass, state in
/// `hp_mem[0]` (and `hp_mem[2]` for stereo).
fn dc_reject(buf: &mut [f32], hp_mem: &mut [f32; 4], channels: i32) {
    let coef: f32 = 6.3f32 * 3.0f32 / FS as f32;
    let coef2: f32 = 1.0 - coef;
    if channels == 2 {
        let mut m0 = hp_mem[0];
        let mut m2 = hp_mem[2];
        for f in buf[..2 * FRAME_SIZE as usize].chunks_exact_mut(2) {
            let (x0, x1) = (f[0], f[1]);
            let out0 = x0 - m0;
            let out1 = x1 - m2;
            m0 = coef * x0 + VERY_SMALL + coef2 * m0;
            m2 = coef * x1 + VERY_SMALL + coef2 * m2;
            f[0] = out0;
            f[1] = out1;
        }
        hp_mem[0] = m0;
        hp_mem[2] = m2;
    } else {
        let mut m0 = hp_mem[0];
        for v in &mut buf[..FRAME_SIZE as usize] {
            let x = *v;
            let y = x - m0;
            m0 = coef * x + VERY_SMALL + coef2 * m0;
            *v = y;
        }
        hp_mem[0] = m0;
    }
}

/// Stereo width reduction, in place.
fn stereo_fade(buf: &mut [f32], g1: f32, g2: f32) {
    let overlap = modes::overlap as usize;
    let g1 = 1.0f32 - g1;
    let g2 = 1.0f32 - g2;
    let buf = &mut buf[..2 * FRAME_SIZE as usize];
    let mut i = 0usize;
    while i < overlap {
        let mut w = modes::window[i];
        w = w * w;
        let g = w * g2 + (1.0f32 - w) * g1;
        let mut diff = 0.5f32 * (buf[i * 2] - buf[i * 2 + 1]);
        diff = g * diff;
        buf[i * 2] = buf[i * 2] - diff;
        buf[i * 2 + 1] = buf[i * 2 + 1] + diff;
        i += 1;
    }
    while i < FRAME_SIZE as usize {
        let mut diff = 0.5f32 * (buf[i * 2] - buf[i * 2 + 1]);
        diff = g2 * diff;
        buf[i * 2] = buf[i * 2] - diff;
        buf[i * 2 + 1] = buf[i * 2 + 1] + diff;
        i += 1;
    }
}

impl OpusEncoder {
    /// All-zero state (as after the C OPUS_CLEAR), usable in a `static`. Call `init` before
    /// encoding.
    pub const fn new_zeroed() -> Self {
        OpusEncoder {
            celt_enc: CELTEncoder::new_zeroed(),
            channels: 0,
            user_bandwidth: 0,
            vbr_constraint: 0,
            user_bitrate_bps: 0,
            use_dtx: 0,
            stream_channels: 0,
            hybrid_stereo_width_Q14: 0,
            hp_mem: [0.0; 4],
            mode: 0,
            bandwidth: 0,
            auto_bandwidth: 0,
            first: 0,
            nb_no_activity_ms_Q1: 0,
            rangeFinal: 0,
        }
    }

    /// `opus_encoder_init(st, 48000, channels, OPUS_APPLICATION_RESTRICTED_CELT)`. Defaults as
    /// in C: bitrate `OPUS_AUTO`, complexity 9, constrained VBR, DTX off.
    pub fn init(&mut self, channels: i32) -> i32 {
        if channels != 1 && channels != 2 {
            return OPUS_BAD_ARG;
        }
        *self = Self::new_zeroed();

        self.channels = channels;
        self.stream_channels = channels;

        crate::celt::cwrs::init_cwrs();
        crate::celt::kiss_fft::init_fft();

        /* Initialize CELT encoder (complexity 9 is the opus default) */
        celt_encoder_init(&mut self.celt_enc, channels);
        self.celt_enc.complexity = 9;

        /* Makes constrained VBR the default (safer for real-time use) */
        self.vbr_constraint = 1;
        self.user_bitrate_bps = OPUS_AUTO;
        self.user_bandwidth = OPUS_AUTO;

        self.hybrid_stereo_width_Q14 = 1 << 14;
        self.first = 1;
        self.mode = MODE_HYBRID;
        self.bandwidth = OPUS_BANDWIDTH_FULLBAND;
        OPUS_OK
    }

    /// `OPUS_SET_BITRATE`: `OPUS_AUTO`, `OPUS_BITRATE_MAX` or a rate in bits/s (clamped to
    /// 500..=750000*channels).
    pub fn set_bitrate(&mut self, value: i32) -> i32 {
        let mut value = value;
        if value != OPUS_AUTO && value != OPUS_BITRATE_MAX {
            if value <= 0 {
                return OPUS_BAD_ARG;
            } else if value <= 500 {
                value = 500;
            } else if value > 750000 * self.channels {
                value = 750000 * self.channels;
            }
        }
        self.user_bitrate_bps = value;
        OPUS_OK
    }

    /// `OPUS_SET_BANDWIDTH` (`OPUS_BANDWIDTH_*` or `OPUS_AUTO`).
    pub fn set_bandwidth(&mut self, value: i32) -> i32 {
        if (value < OPUS_BANDWIDTH_NARROWBAND || value > OPUS_BANDWIDTH_FULLBAND)
            && value != OPUS_AUTO
        {
            return OPUS_BAD_ARG;
        }
        self.user_bandwidth = value;
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

    /// `OPUS_SET_COMPLEXITY` (0..=10).
    pub fn set_complexity(&mut self, value: i32) -> i32 {
        if value < 0 || value > 10 {
            return OPUS_BAD_ARG;
        }
        self.celt_enc.complexity = value;
        OPUS_OK
    }

    /// `OPUS_SET_PACKET_LOSS_PERC` (0..=100).
    pub fn set_packet_loss_perc(&mut self, value: i32) -> i32 {
        if value < 0 || value > 100 {
            return OPUS_BAD_ARG;
        }
        self.celt_enc.loss_rate = value;
        OPUS_OK
    }

    /// `OPUS_SET_VBR_CONSTRAINT` (0 or 1).
    pub fn set_vbr_constraint(&mut self, value: i32) -> i32 {
        if value < 0 || value > 1 {
            return OPUS_BAD_ARG;
        }
        self.vbr_constraint = value;
        OPUS_OK
    }

    /// `OPUS_GET_FINAL_RANGE`: range coder state after the last `encode*` call.
    pub fn get_final_range(&self) -> u32 {
        self.rangeFinal
    }

    /// `OPUS_GET_IN_DTX`.
    pub fn get_in_dtx(&self) -> i32 {
        /* The SILK branch (prev_mode SILK-only or hybrid) is unreachable. */
        if self.use_dtx != 0 {
            /* DTX determined by Opus. */
            (self.nb_no_activity_ms_Q1 >= NB_SPEECH_FRAMES_BEFORE_DTX * 20 * 2) as i32
        } else {
            0
        }
    }

    /// `opus_encode_float`: 960 samples per channel of interleaved f32 PCM (nominal range
    /// +-1) at the start of `buf`, which is also the working memory (see the module comment)
    /// and must hold `channels * IN_BUF_FRAME` samples. `lsb_depth` is the depth of the
    /// source (24 for float input; 16 for `opus_encode` of 16-bit input scaled to +-1).
    /// Returns the packet length (1..=1276 bytes at `data`) or a negative Opus error code.
    pub fn encode_float(
        &mut self,
        buf: &mut [f32],
        data: &mut [u8; PACKET_SIZE_CAP],
        lsb_depth: i32,
    ) -> i32 {
        if self.channels == 0 {
            return OPUS_BAD_ARG;
        }
        debug_assert!(buf.len() >= (IN_BUF_FRAME * self.channels) as usize);
        let pcm = &mut buf[..(FRAME_SIZE * self.channels) as usize];
        self.rangeFinal = 0;
        /* "Cannot encode 100 ms in 1 byte" does not apply to 20 ms frames. */

        let is_silence = maxabs(pcm) <= 1.0f32 / (1i32 << lsb_depth) as f32;
        /* analysis_info.valid = 0: run_analysis() (complexity >= 7) is not ported, see the
        module comment; tonality_analysis_reset() has no observable effect either. */

        let complexity = self.celt_enc.complexity;
        /* user_bitrate_to_bitrate(): max_bitrate is bits_to_bitrate(1276 * 8, 48000, 960) */
        let user_bitrate = if self.user_bitrate_bps == OPUS_AUTO {
            3000 + FS * self.channels
        } else if self.user_bitrate_bps == OPUS_BITRATE_MAX {
            1500000
        } else {
            self.user_bitrate_bps
        };
        let bitrate_bps = imin(user_bitrate, PACKET_SIZE_CAP as i32 * 8 * 50);

        /* max_data_bytes (1276) >= 3 and the frame rate is 50 */
        if bitrate_bps < 3 * 50 * 8 {
            /*If the space is too low to do something useful, emit 'PLC' frames.*/
            data[0] = if self.mode == MODE_HYBRID {
                /* hybrid TOC: 0x60, fullband, 20 ms (bandwidth is still fullband here) */
                0x78 | ((self.stream_channels == 2) as u8) << 2
            } else {
                celt_toc(self.bandwidth, self.stream_channels)
            };
            return 1;
        }

        /* Equivalent 20-ms rate for mode/channel/bandwidth decisions (compute_equiv_rate with
        VBR and 20 ms frames; the mode is not known yet for the channel decision) */
        let loss = self.celt_enc.loss_rate;
        let mut equiv_rate = bitrate_bps * (90 + complexity) / 100;

        if self.channels == 2 {
            /* Mode not known yet: half the SILK loss */
            equiv_rate -= equiv_rate * loss / (12 * loss + 20);
            /* Rate-dependent mono-stereo decision */
            let mut threshold = stereo_threshold;
            if self.stream_channels == 2 {
                threshold -= 1000;
            } else {
                threshold += 1000;
            }
            self.stream_channels = if equiv_rate > threshold { 2 } else { 1 };
            equiv_rate = bitrate_bps * (90 + complexity) / 100;
        } else {
            self.stream_channels = 1;
        }

        /* Mode selection: RESTRICTED_CELT is always CELT-only. Since prev_mode is 0 or
        CELT-only, there is no mode transition (no redundancy, to_celt or prefill), and
        toMono stays 0. */
        self.mode = MODE_CELT_ONLY;

        /* Update equivalent rate with mode decision: CELT complexity 0-4 doesn't have the
        pitch filter, which costs about 10%. */
        if complexity < 5 {
            equiv_rate = equiv_rate * 9 / 10;
        }

        /* Automatic (rate-dependent) bandwidth selection (always, since mode is CELT-only) */
        {
            let mut bandwidth = OPUS_BANDWIDTH_FULLBAND;
            loop {
                let mut threshold =
                    bandwidth_thresholds[(2 * (bandwidth - OPUS_BANDWIDTH_MEDIUMBAND)) as usize];
                let hysteresis = bandwidth_thresholds
                    [(2 * (bandwidth - OPUS_BANDWIDTH_MEDIUMBAND) + 1) as usize];
                if self.first == 0 {
                    if self.auto_bandwidth >= bandwidth {
                        threshold -= hysteresis;
                    } else {
                        threshold += hysteresis;
                    }
                }
                if equiv_rate >= threshold {
                    break;
                }
                bandwidth -= 1;
                if bandwidth <= OPUS_BANDWIDTH_NARROWBAND {
                    break;
                }
            }
            /* We don't use mediumband anymore, except when explicitly requested or during
            mode transitions. */
            if bandwidth == OPUS_BANDWIDTH_MEDIUMBAND {
                bandwidth = OPUS_BANDWIDTH_WIDEBAND;
            }
            self.auto_bandwidth = bandwidth;
            self.bandwidth = bandwidth;
        }

        if self.user_bandwidth != OPUS_AUTO {
            self.bandwidth = self.user_bandwidth;
        }

        /* CELT mode doesn't support mediumband, use wideband instead */
        if self.bandwidth == OPUS_BANDWIDTH_MEDIUMBAND {
            self.bandwidth = OPUS_BANDWIDTH_WIDEBAND;
        }

        /* The hybrid max_rate cap, the Nyquist caps (Fs <= 24000), the detected bandwidth
        (no analysis) and decide_fec() (0 for CELT-only) do not apply. frame_size == Fs/50, so no
        multi-frame packet. */
        self.celt_enc.lsb_depth = lsb_depth;
        self.encode_frame_native(buf, data, is_silence, equiv_rate, bitrate_bps)
    }

    /// `opus_encode_frame_native` for CELT-only (redundancy, celt_to_silk, prefill and to_celt
    /// are all 0).
    fn encode_frame_native(
        &mut self,
        buf: &mut [f32],
        data: &mut [u8; PACKET_SIZE_CAP],
        is_silence: bool,
        equiv_rate: i32,
        bitrate_bps: i32,
    ) -> i32 {
        /* delay_compensation = 0 for RESTRICTED_CELT: no delay buffer. The activity decision
        (VAD_NO_DECISION / peak energy vs noise energy) only reaches the DTX logic below, which
        needs digital silence: activity == 0 there. */

        /* data += 1 for the range coder, which keeps a raw pointer into `data`: from here on
        `data` is only accessed through `data_ptr`. The coder has the remaining 1275 bytes
        (`ec_enc_shrink(enc, max_data_bytes - 1)` is then a no-op). */
        let data_ptr = data.as_mut_ptr();
        let mut enc: ec_enc = unsafe { core::mem::zeroed() };
        // SAFETY: data has PACKET_SIZE_CAP bytes.
        unsafe { ec_enc_init(&mut enc, data_ptr.add(1), (PACKET_SIZE_CAP - 1) as u32) };

        /* No delay_buffer prefix: total_buffer == 0. The variable HP smoothing only feeds
        the VOIP high-pass. */
        let pcm = &mut buf[..(FRAME_SIZE * self.channels) as usize];
        dc_reject(pcm, &mut self.hp_mem, self.channels);

        /* float_api */
        {
            let n = FRAME_SIZE * self.channels;
            let sum = celt_inner_prod(pcm, pcm, n);
            /* This should filter out both NaNs and ridiculous signals that could
            cause NaNs further down. */
            if !(sum < 1e9f32) || sum.is_nan() {
                pcm.fill(0.0);
                self.hp_mem = [0.0; 4];
            }
        }

        /* SILK processing: none (HB_gain = Q15ONE). */

        /* CELT processing */
        self.celt_enc.end = match self.bandwidth {
            OPUS_BANDWIDTH_NARROWBAND => 13,
            OPUS_BANDWIDTH_MEDIUMBAND | OPUS_BANDWIDTH_WIDEBAND => 17,
            OPUS_BANDWIDTH_SUPERWIDEBAND => 19,
            _ => 21,
        };
        self.celt_enc.stream_channels = self.stream_channels;
        /* (CELT_SET_PREDICTION(2): prediction enabled; OPUS_SET_VBR(1)) */

        /* tmp_prefill (needs a mode change), the delay_buffer update (encoder_buffer == 0) and
        gain_fade() (HB_gain == prev_HB_gain == Q15ONE) do nothing here. */

        if self.channels == 2 {
            /* mode != MODE_HYBRID */
            let stereoWidth_Q14 = if equiv_rate > 32000 {
                16384
            } else if equiv_rate < 16000 {
                0
            } else {
                16384 - 2048 * (32000 - equiv_rate) / (equiv_rate - 14000)
            };
            /* Apply stereo width reduction (at low bitrates) */
            if (self.hybrid_stereo_width_Q14 as i32) < (1 << 14) || stereoWidth_Q14 < (1 << 14) {
                let mut g1 = self.hybrid_stereo_width_Q14 as f32;
                let mut g2 = stereoWidth_Q14 as f32;
                g1 *= 1.0f32 / 16384.0f32;
                g2 *= 1.0f32 / 16384.0f32;
                stereo_fade(&mut buf[..(2 * FRAME_SIZE) as usize], g1, g2);
                self.hybrid_stereo_width_Q14 = stereoWidth_Q14 as i16;
            }
        }

        // SAFETY (data_ptr accesses below): data_ptr[0..2] lies within `data`.
        unsafe { *data_ptr = 0 };
        self.celt_enc.constrained_vbr = self.vbr_constraint;
        self.celt_enc.bitrate = bitrate_bps;

        /* ec_tell(&enc) <= 8 * (max_data_bytes - 1) holds for the fresh coder */
        let ret = celt_encode_with_ec(&mut self.celt_enc, buf, &mut enc);
        if ret < 0 {
            return OPUS_INTERNAL_ERROR;
        }
        self.rangeFinal = self.celt_enc.rng;

        /* Signalling the mode in the first byte */
        let toc = celt_toc(self.bandwidth, self.stream_channels);
        unsafe { *data_ptr |= toc };

        self.first = 0;

        /* DTX decision (decide_dtx_mode): silk_mode.useDTX == use_dtx && !is_silence, so the
        DTX branch is only reached for digital silence (activity == 0). */
        if self.use_dtx != 0 && is_silence {
            /* frame_size_ms_Q1 = 2 * 1000 * 960 / 48000 */
            self.nb_no_activity_ms_Q1 += 40;
            if self.nb_no_activity_ms_Q1 > NB_SPEECH_FRAMES_BEFORE_DTX * 20 * 2 {
                if self.nb_no_activity_ms_Q1
                    <= (NB_SPEECH_FRAMES_BEFORE_DTX + MAX_CONSECUTIVE_DTX) * 20 * 2
                {
                    /* Valid frame for DTX! */
                    self.rangeFinal = 0;
                    unsafe { *data_ptr = toc };
                    return 1;
                } else {
                    self.nb_no_activity_ms_Q1 = NB_SPEECH_FRAMES_BEFORE_DTX * 20 * 2;
                }
            }
        } else {
            self.nb_no_activity_ms_Q1 = 0;
        }

        /* "In the unlikely case that the SILK encoder busted its target, tell the decoder to
        call the PLC" (ec_tell(&enc) > (max_data_bytes - 1) * 8) cannot happen: CELT never
        spends more than its nbCompressedBytes <= 1275. The trailing-zero stripping is SILK-only.
        Count ToC. */
        debug_assert!(ec_tell(&enc) <= (PACKET_SIZE_CAP as i32 - 1) * 8);
        /* use_vbr: no padding */
        ret + 1
    }
}
