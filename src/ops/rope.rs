pub fn apply_rope(
    x:        &mut [f32],
    seq_len:  usize,
    heads:    usize,
    head_dim: usize,
    theta:    f32,
) {
    for pos in 0..seq_len {
        for h in 0..heads {
            let base = (pos * heads + h) * head_dim;
            for i in 0..head_dim / 2 {
                let freq  = 1.0 / theta.powf(2.0 * i as f32 / head_dim as f32);
                let angle = pos as f32 * freq;
                let (sin, cos) = angle.sin_cos();

                let x0 = x[base + i];
                let x1 = x[base + head_dim / 2 + i];
                x[base + i]                = x0 * cos - x1 * sin;
                x[base + head_dim / 2 + i] = x0 * sin + x1 * cos;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rope_identity_at_pos_zero() {
        // at position 0 all angles are 0 so cos=1 sin=0 meaning identity
        let mut x: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let original = x.clone();
        apply_rope(&mut x, 1, 1, 8, 10_000.0);
        for (a, b) in x.iter().zip(original.iter()) {
            assert!((a - b).abs() < 1e-5, "a={} b={}", a, b);
        }
    }

    #[test]
    fn rope_changes_values_at_nonzero_pos() {
        // at position > 0 values should actually rotate
        let mut x = vec![1.0f32, 0.0, 1.0, 0.0,  1.0, 0.0, 1.0, 0.0];
        let original = x.clone();
        apply_rope(&mut x, 2, 1, 8, 10_000.0);
        // pos=1 slice should differ from original
        let changed = x[8..].iter().zip(original[8..].iter())
            .any(|(a, b)| (a - b).abs() > 1e-5);
        assert!(changed, "RoPE should rotate values at pos > 0");
    }
}
