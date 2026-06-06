use crate::quantization::{PackedMatrix, QuantMode};
use rayon::prelude::*;

pub fn bitgemv_cpu(
    w:        &PackedMatrix,
    x_packed: &[u64],
    x_scale:  f32,
    y:        &mut [f32],
) {
    assert_eq!(y.len(), w.rows);
    assert_eq!(x_packed.len(), w.words_per_row,
        "activation word count ({}) != weight words_per_row ({})",
        x_packed.len(), w.words_per_row);

    match w.mode {
        QuantMode::Binary  => binary_gemv(w, x_packed, x_scale, y),
        QuantMode::Ternary => ternary_gemv(w, x_packed, x_scale, y),
    }
}

fn binary_gemv(w: &PackedMatrix, x_packed: &[u64], x_scale: f32, y: &mut [f32]) {
    let k              = w.words_per_row;
    let total_bits     = w.cols as i32;
    let combined_scale = w.scale * x_scale;

    y.par_iter_mut().enumerate().for_each(|(row, out)| {
        let w_row = &w.mag[row * k..(row + 1) * k];
        let dot   = dot_binary(w_row, x_packed);
        *out += combined_scale * (2 * dot - total_bits) as f32;
    });
}

fn ternary_gemv(w: &PackedMatrix, x_packed: &[u64], x_scale: f32, y: &mut [f32]) {
    let k              = w.words_per_row;
    let combined_scale = w.scale * x_scale;

    y.par_iter_mut().enumerate().for_each(|(row, out)| {
        let mag_row  = &w.mag [row * k..(row + 1) * k];
        let sign_row = &w.sign[row * k..(row + 1) * k];
        let dot      = dot_ternary(mag_row, sign_row, x_packed);
        *out += combined_scale * dot as f32;
    });
}

#[inline(always)]
fn dot_binary(w_row: &[u64], x: &[u64]) -> i32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { dot_binary_avx2(w_row, x) };
        }
    }
    dot_binary_scalar(w_row, x)
}

#[inline(always)]
fn dot_ternary(mag: &[u64], sign: &[u64], x: &[u64]) -> i32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { dot_ternary_avx2(mag, sign, x) };
        }
    }
    dot_ternary_scalar(mag, sign, x)
}

fn dot_binary_scalar(w_row: &[u64], x: &[u64]) -> i32 {
    w_row.iter().zip(x.iter())
         .map(|(&ww, &xw)| (!(ww ^ xw)).count_ones() as i32)
         .sum::<i32>()
}

fn dot_ternary_scalar(mag: &[u64], sign: &[u64], x: &[u64]) -> i32 {
    let mut dot = 0i32;
    for ((m, s), xw) in mag.iter().zip(sign.iter()).zip(x.iter()) {
        dot += (m & !(s ^ xw)).count_ones() as i32;
        dot -= (m &  (s ^ xw)).count_ones() as i32;
    }
    dot
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_binary_avx2(w_row: &[u64], x: &[u64]) -> i32 {
    let chunks = w_row.len() / 4;
    let mut total = 0i32;

    for i in 0..chunks {
        total += (!(w_row[i*4]   ^ x[i*4]  )).count_ones() as i32;
        total += (!(w_row[i*4+1] ^ x[i*4+1])).count_ones() as i32;
        total += (!(w_row[i*4+2] ^ x[i*4+2])).count_ones() as i32;
        total += (!(w_row[i*4+3] ^ x[i*4+3])).count_ones() as i32;
    }
    for i in (chunks * 4)..w_row.len() {
        total += (!(w_row[i] ^ x[i])).count_ones() as i32;
    }
    total
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_ternary_avx2(mag: &[u64], sign: &[u64], x: &[u64]) -> i32 {
    let chunks = mag.len() / 4;
    let mut dot = 0i32;

    macro_rules! step {
        ($i:expr) => {{
            let m = mag[$i]; let s = sign[$i]; let xw = x[$i];
            dot += (m & !(s ^ xw)).count_ones() as i32;
            dot -= (m &  (s ^ xw)).count_ones() as i32;
        }};
    }

    for i in 0..chunks {
        step!(i*4); step!(i*4+1); step!(i*4+2); step!(i*4+3);
    }
    for i in (chunks * 4)..mag.len() { step!(i); }
    dot
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binary_dot_all_agree() {
        let w = vec![0xFFFFFFFFFFFFFFFF_u64; 2];
        let x = vec![0xFFFFFFFFFFFFFFFF_u64; 2];
        assert_eq!(dot_binary_scalar(&w, &x), 128);
    }

    #[test]
    fn binary_dot_all_disagree() {
        let w = vec![0xFFFFFFFFFFFFFFFF_u64; 1];
        let x = vec![0x0000000000000000_u64; 1];
        assert_eq!(dot_binary_scalar(&w, &x), 0);
    }
}
