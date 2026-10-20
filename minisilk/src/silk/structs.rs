//! Port of silk/structs.h (encoder side). Only the state the fixed configuration (16 kHz mono,
//! 20 ms frames, no LBRR, no stereo, no resampler) reads or writes is kept.


use crate::silk::define::*;

const MAX_FRAME_LENGTH_: usize = MAX_FRAME_LENGTH as usize;
const MAX_FRAMES_PER_PACKET_: usize = MAX_FRAMES_PER_PACKET as usize;
const MAX_NB_SUBFR_: usize = MAX_NB_SUBFR as usize;
const MAX_LPC_ORDER_: usize = MAX_LPC_ORDER as usize;
const VAD_N_BANDS_: usize = VAD_N_BANDS as usize;
const INPUT_DELAY_: usize = INPUT_DELAY as usize;

/// Noise shaping quantization state
#[derive(Clone, Copy)]
pub struct silk_nsq_state {
    pub xq: [i16; 2 * MAX_FRAME_LENGTH_], /* Buffer for quantized output signal */
    pub sLTP_shp_Q14: [i32; 2 * MAX_FRAME_LENGTH_],
    pub sLPC_Q14: [i32; (MAX_SUB_FRAME_LENGTH + NSQ_LPC_BUF_LENGTH) as usize],
    pub sAR2_Q14: [i32; MAX_SHAPE_LPC_ORDER as usize],
    pub sLF_AR_shp_Q14: i32,
    pub sDiff_shp_Q14: i32,
    pub lagPrev: i32,
    pub sLTP_buf_idx: i32,
    pub sLTP_shp_buf_idx: i32,
    pub rand_seed: i32,
    pub prev_gain_Q16: i32,
    pub rewhite_flag: i32,
}

/// VAD state
#[derive(Clone, Copy)]
pub struct silk_VAD_state {
    pub AnaState: [i32; 2],                     /* Analysis filterbank state: 0-8 kHz */
    pub AnaState1: [i32; 2],                    /* Analysis filterbank state: 0-4 kHz */
    pub AnaState2: [i32; 2],                    /* Analysis filterbank state: 0-2 kHz */
    pub XnrgSubfr: [i32; VAD_N_BANDS_],         /* Subframe energies */
    pub NrgRatioSmth_Q8: [i32; VAD_N_BANDS_],   /* Smoothed energy level in each band */
    pub HPstate: i16,                           /* State of differentiator in the lowest band */
    pub NL: [i32; VAD_N_BANDS_],                /* Noise energy level in each band */
    pub inv_NL: [i32; VAD_N_BANDS_],            /* Inverse noise energy level in each band */
    pub NoiseLevelBias: [i32; VAD_N_BANDS_],    /* Noise level estimator bias/offset */
    pub counter: i32,                           /* Frame counter used in the initial phase */
}

/// Structure containing NLSF codebook
pub struct silk_NLSF_CB_struct {
    pub nVectors: i16,
    pub order: i16,
    pub quantStepSize_Q16: i16,
    pub invQuantStepSize_Q6: i16,
    pub CB1_NLSF_Q8: *const u8,
    pub CB1_Wght_Q9: *const i16,
    pub CB1_iCDF: *const u8,
    pub pred_Q8: *const u8,
    pub ec_sel: *const u8,
    pub ec_iCDF: *const u8,
    pub ec_Rates_Q5: *const u8,
    pub deltaMin_Q15: *const i16,
}

// SAFETY: the codebooks are immutable tables.
unsafe impl Sync for silk_NLSF_CB_struct {}

#[derive(Clone, Copy)]
pub struct SideInfoIndices {
    pub GainsIndices: [i8; MAX_NB_SUBFR_],
    pub LTPIndex: [i8; MAX_NB_SUBFR_],
    pub NLSFIndices: [i8; MAX_LPC_ORDER_ + 1],
    pub lagIndex: i16,
    pub contourIndex: i8,
    pub signalType: i8,
    pub quantOffsetType: i8,
    pub NLSFInterpCoef_Q2: i8,
    pub PERIndex: i8,
    pub LTP_scaleIndex: i8,
    pub Seed: i8,
}

/// Encoder state
#[derive(Clone, Copy)]
pub struct silk_encoder_state {
    pub variable_HP_smth1_Q15: i32,             /* State of first smoother */
    pub sVAD: silk_VAD_state,                   /* Voice activity detector state */
    pub sNSQ: silk_nsq_state,                   /* Noise Shape Quantizer State */
    pub prev_NLSFq_Q15: [i16; MAX_LPC_ORDER_],  /* Previously quantized NLSF vector */
    pub speech_activity_Q8: i32,                /* Speech activity */
    pub prevSignalType: i8,
    pub prevLag: i32,
    pub la_shape: i32,                          /* Look-ahead for noise shape analysis (samples) */
    pub shapeWinLength: i32,                    /* Window length for noise shape analysis (samples) */
    pub PacketLoss_perc: i32,                   /* Packet loss rate measured by farend */
    pub frameCounter: i32,
    pub nStatesDelayedDecision: i32,            /* Number of states in delayed decision quantization */
    pub useInterpolatedNLSFs: i32,              /* Flag for using NLSF interpolation */
    pub shapingLPCOrder: i32,                   /* Filter order for noise shaping filters */
    pub pitchEstimationComplexity: i32,         /* Complexity level for pitch estimator */
    pub pitchEstimationLPCOrder: i32,           /* Whitening filter order for pitch estimator */
    pub pitchEstimationThreshold_Q16: i32,      /* Threshold for pitch estimator */
    pub sum_log_gain_Q7: i32,                   /* Cumulative max prediction gain */
    pub NLSF_MSVQ_Survivors: i32,               /* Number of survivors in NLSF MSVQ */
    pub first_frame_after_reset: i32,           /* Flag for deactivating NLSF interpolation, pitch prediction */
    pub warping_Q16: i32,                       /* Warping parameter for warped noise shaping */
    pub input_quality_bands_Q15: [i32; VAD_N_BANDS_],
    pub input_tilt_Q15: i32,
    pub SNR_dB_Q7: i32,                         /* Quality setting */
    pub VAD_flags: [i8; MAX_FRAMES_PER_PACKET_],
    pub indices: SideInfoIndices,
    pub pulses: [i8; MAX_FRAME_LENGTH_],

    /* Input/output buffering */
    pub inputBuf: [i16; MAX_FRAME_LENGTH_],     /* Buffer containing input signal */
    pub delayBuf: [i16; INPUT_DELAY_],          /* Last INPUT_DELAY input samples of the previous frame (resampler delay line) */

    /* Parameters For LTP scaling Control */

    /* DTX */
    pub useDTX: i32,                            /* Flag to enable DTX */
    pub inDTX: i32,                             /* Flag to signal DTX period */
    pub noSpeechCounter: i32,                   /* Counts concecutive nonactive frames, used by DTX */

    /* Inband Low Bitrate Redundancy (LBRR) data */
}

