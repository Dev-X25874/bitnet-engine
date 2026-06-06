use rayon::prelude::*;

#[inline]
pub fn compute_absmax_scale(data: &[f32]) -> f32 {
    data.par_iter()
        .map(|x| x.abs())
        .reduce(|| 0.0f32, f32::max)
        .max(1e-8)
}

#[inline]
pub fn compute_mean_abs(data: &[f32]) -> f32 {
    if data.is_empty() { return 0.0; }
    let sum: f32 = data.par_iter().map(|x| x.abs()).sum();
    (sum / data.len() as f32).max(1e-8)
}

pub fn compute_per_row_scales(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    (0..rows)
        .into_par_iter()
        .map(|r| compute_absmax_scale(&data[r * cols..(r + 1) * cols]))
        .collect()
}

pub fn quantise_activation(x: &[f32]) -> (Vec<u64>, f32) {
    let scale = compute_absmax_scale(x);
    let words = (x.len() + 63) / 64;
    let mut packed = vec![0u64; words];
    for (i, &v) in x.iter().enumerate() {
        if v >= 0.0 {
            packed[i / 64] |= 1u64 << (i % 64);
        }
    }
    (packed, scale)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn absmax_correct() {
        let v = vec![-3.0f32, 1.5, 2.0, -0.5];
        assert!((compute_absmax_scale(&v) - 3.0).abs() < 1e-6);
    }

    #[test]
    fn mean_abs_correct() {
        let v = vec![1.0f32, -1.0, 2.0, -2.0];
        assert!((compute_mean_abs(&v) - 1.5).abs() < 1e-6);
    }
}
