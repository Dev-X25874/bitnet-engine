pub mod pack;
pub mod scale;
pub mod loader;

pub use pack::{BitPacking, PackedMatrix};
pub use scale::compute_absmax_scale;
pub use loader::load_safetensors_weights;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantMode {
    /// Pure 1-bit: w ∈ {-1, +1}
    Binary,
    /// 1.58-bit ternary: w ∈ {-1, 0, +1}
    Ternary,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i8)]
pub enum TernaryWeight {
    NegOne = -1,
    Zero   =  0,
    PosOne =  1,
}

impl TernaryWeight {
    #[inline]
    pub fn from_float(v: f32, threshold: f32) -> Self {
        if v.abs() < threshold {
            TernaryWeight::Zero
        } else if v > 0.0 {
            TernaryWeight::PosOne
        } else {
            TernaryWeight::NegOne
        }
    }
}
