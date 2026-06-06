//! # BitNet Engine
//!
//! A quantisation-native inference engine for 1-bit and 1.58-bit LLMs.

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(clippy::all, clippy::pedantic)]
#![allow(clippy::module_name_repetitions, clippy::cast_possible_truncation)]

pub mod error;
pub mod quantization;
pub mod ops;
pub mod runtime;
pub mod utils;

#[cfg(feature = "cuda")]
pub mod cuda_ffi;

pub use error::{BitNetError, Result};
pub use runtime::{Engine, EngineConfig, GenerateConfig};
pub use quantization::{BitPacking, QuantMode, TernaryWeight};
