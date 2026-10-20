//! Port of silk/float/structs_FLP.h.


use crate::silk::define::*;
use crate::silk::structs::*;

const MAX_NB_SUBFR_: usize = MAX_NB_SUBFR as usize;
const MAX_LPC_ORDER_: usize = MAX_LPC_ORDER as usize;

/// Noise shaping analysis state
#[derive(Clone, Copy)]
pub struct silk_shape_state_FLP {
    pub LastGainIndex: i8,
    pub HarmShapeGain_smth: f32,
    pub Tilt_smth: f32,
}

/// Encoder state FLP
#[derive(Clone, Copy)]
pub struct silk_encoder_state_FLP {
    pub sCmn: silk_encoder_state,    /* Common struct, shared with fixed-point code */
    pub sShape: silk_shape_state_FLP, /* Noise shaping state */

    /* Buffer for find pitch and noise shape analysis */
    pub x_buf: [f32; (2 * MAX_FRAME_LENGTH + LA_SHAPE_MAX) as usize], /* Buffer for find pitch and noise shape analysis */
    pub LTPCorr: f32,                 /* Normalized correlation from pitch lag estimator */
}

/// Encoder control FLP
#[derive(Clone, Copy)]
pub struct silk_encoder_control_FLP {
    /* Prediction and coding parameters */
    pub Gains: [f32; MAX_NB_SUBFR_],
    pub PredCoef: [[f32; MAX_LPC_ORDER_]; 2], /* holds interpolated and final coefficients */
    pub LTPCoef: [f32; (LTP_ORDER * MAX_NB_SUBFR) as usize],
    pub pitchL: [i32; MAX_NB_SUBFR_],

    /* Noise shaping parameters */
    pub AR: [f32; (MAX_NB_SUBFR * MAX_SHAPE_LPC_ORDER) as usize],
    pub LF_MA_shp: [f32; MAX_NB_SUBFR_],
    pub LF_AR_shp: [f32; MAX_NB_SUBFR_],
    pub Tilt: [f32; MAX_NB_SUBFR_],
    pub HarmShapeGain: [f32; MAX_NB_SUBFR_],
    pub Lambda: f32,
    pub input_quality: f32,
    pub coding_quality: f32,

    /* Measures */
    pub predGain: f32,
    pub LTPredCodGain: f32,
    pub ResNrg: [f32; MAX_NB_SUBFR_], /* Residual energy per subframe */

    /* Parameters for CBR mode */
    pub GainsUnq_Q16: [i32; MAX_NB_SUBFR_],
    pub lastGainIndexPrev: i8,
}

/// Encoder Super Struct (mono: a single channel state)
#[derive(Clone, Copy)]
pub struct silk_encoder {
    pub nBitsExceeded: i32,
    pub state_Fxx: silk_encoder_state_FLP,
}
