use crate::error::{BitNetError, Result};
use super::{QuantMode, TernaryWeight};

#[derive(Debug, Clone)]
pub struct PackedMatrix {
    pub rows: usize,
    pub cols: usize,
    pub words_per_row: usize,
    pub mode: QuantMode,
    pub scale: f32,
    pub mag: Vec<u64>,
    pub sign: Vec<u64>,
}

impl PackedMatrix {
    #[inline] pub fn mag_bytes(&self) -> usize { self.mag.len() * 8 }
    #[inline] pub fn sign_bytes(&self) -> usize { self.sign.len() * 8 }
}

pub trait BitPacking {
    fn pack_f32(data: &[f32], rows: usize, cols: usize, mode: QuantMode) -> Result<PackedMatrix>;
    fn unpack_f32(packed: &PackedMatrix) -> Vec<f32>;
}

impl BitPacking for PackedMatrix {
    fn pack_f32(data: &[f32], rows: usize, cols: usize, mode: QuantMode) -> Result<PackedMatrix> {
        if data.len() != rows * cols {
            return Err(BitNetError::ShapeMismatch {
                expected: vec![rows, cols],
                got: vec![data.len()],
            });
        }

        let scale = super::scale::compute_absmax_scale(data);
        let words_per_row = (cols + 63) / 64;
        let total_words = rows * words_per_row;

        let mut mag = vec![0u64; total_words];
        let mut sign = vec![0u64; if mode == QuantMode::Ternary { total_words } else { 0 }];

        let threshold = if mode == QuantMode::Ternary {
            super::scale::compute_mean_abs(data)
        } else {
            0.0_f32
        };

        for row in 0..rows {
            let src = &data[row * cols..(row + 1) * cols];
            let mag_row = &mut mag[row * words_per_row..(row + 1) * words_per_row];
            let sign_row = if mode == QuantMode::Ternary {
                Some(&mut sign[row * words_per_row..(row + 1) * words_per_row])
            } else {
                None
            };
            pack_row(src, mag_row, sign_row, mode, threshold);
        }

        Ok(PackedMatrix { rows, cols, words_per_row, mode, scale, mag, sign })
    }

    fn unpack_f32(packed: &PackedMatrix) -> Vec<f32> {
        let mut out = vec![0.0f32; packed.rows * packed.cols];
        for row in 0..packed.rows {
            let mag_row =
                &packed.mag[row * packed.words_per_row..(row + 1) * packed.words_per_row];
            let sign_row = if packed.mode == QuantMode::Ternary {
                &packed.sign[row * packed.words_per_row..(row + 1) * packed.words_per_row]
            } else {
                &[]
            };
            for col in 0..packed.cols {
                let word = col / 64;
                let bit = col % 64;
                let m_bit = (mag_row[word] >> bit) & 1;
                let s_bit = if !sign_row.is_empty() {
                    (sign_row[word] >> bit) & 1
                } else {
                    0
                };
                out[row * packed.cols + col] = match (m_bit, s_bit) {
                    (0, _) => 0.0,
                    (1, 0) => packed.scale,
                    (1, 1) => -packed.scale,
                    _ => unreachable!(),
                };
            }
        }
        out
    }
}

#[inline]
fn pack_row(
    src: &[f32],
    mag_row: &mut [u64],
    mut sign_row: Option<&mut [u64]>,
    mode: QuantMode,
    threshold: f32,
) {
    let cols = src.len();
    for word_idx in 0..mag_row.len() {
        let base = word_idx * 64;
        let limit = (base + 64).min(cols);
        let mut mw = 0u64;
        let mut sw = 0u64;

        for bit in 0..(limit - base) {
            let v = src[base + bit];
            match mode {
                QuantMode::Binary => {
                    if v >= 0.0 {
                        mw |= 1u64 << bit;
                    }
                }
                QuantMode::Ternary => {
                    let tw = TernaryWeight::from_float(v, threshold);
                    if tw != TernaryWeight::Zero {
                        mw |= 1u64 << bit;
                        if tw == TernaryWeight::NegOne {
                            sw |= 1u64 << bit;
                        }
                    }
                }
            }
        }

        mag_row[word_idx] = mw;
        if let Some(ref mut sr) = sign_row {
            sr[word_idx] = sw;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_binary() {
        let data: Vec<f32> =
            (0..128).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
        let packed = PackedMatrix::pack_f32(&data, 2, 64, QuantMode::Binary).unwrap();
        let recovered = PackedMatrix::unpack_f32(&packed);
        for (a, b) in data.iter().zip(recovered.iter()) {
            assert!((a - b).abs() < 1e-3, "a={} b={}", a, b);
        }
    }

    #[test]
    fn roundtrip_ternary() {
        let data: Vec<f32> = vec![
            1.0, -1.0, 0.0, 0.5, -0.5, 0.0, 2.0, -2.0,
            0.1, -0.1, 1.5, -1.5, 0.0, 0.0, 0.8, -0.8,
        ];
        let packed = PackedMatrix::pack_f32(&data, 1, 16, QuantMode::Ternary).unwrap();
        let recovered = PackedMatrix::unpack_f32(&packed);
        for (a, b) in data.iter().zip(recovered.iter()) {
            assert_eq!(a.signum(), b.signum(), "sign mismatch: a={} b={}", a, b);
        }
    }
}
