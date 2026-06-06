use bitnet::quantization::{BitPacking, PackedMatrix, QuantMode};
use bitnet::quantization::scale::quantise_activation;
use bitnet::ops::cpu_gemv::bitgemv_cpu;
use approx::assert_abs_diff_eq;

// ---------------------------------------------------------------------------
// Binary GEMV tests
// ---------------------------------------------------------------------------

#[test]
fn test_binary_gemv_all_positive_weights_and_activations() {
    let n = 4; let k = 64;
    let w_data = vec![1.0f32; n * k];
    let x_data = vec![1.0f32; k];

    let packed          = PackedMatrix::pack_f32(&w_data, n, k, QuantMode::Binary).unwrap();
    let (x_packed, x_scale) = quantise_activation(&x_data);

    let mut y = vec![0.0f32; n];
    bitgemv_cpu(&packed, &x_packed, x_scale, &mut y);

    // all +1 weights dot all +1 activations → every output should be positive
    for &v in &y {
        assert!(v > 0.0, "expected positive output, got {}", v);
    }
}

#[test]
fn test_binary_gemv_sign_preserving() {
    let n = 16; let k = 64;
    // all +1 weights, half positive half negative activations → near zero
    let w_data = vec![1.0f32; n * k];
    let x_data: Vec<f32> = (0..k)
        .map(|i| if i < 32 { 1.0 } else { -1.0 })
        .collect();

    let packed          = PackedMatrix::pack_f32(&w_data, n, k, QuantMode::Binary).unwrap();
    let (x_packed, x_scale) = quantise_activation(&x_data);

    let mut y = vec![0.0f32; n];
    bitgemv_cpu(&packed, &x_packed, x_scale, &mut y);

    for &v in &y {
        assert!(v.abs() < 5.0, "expected near-zero, got {}", v);
    }
}

#[test]
fn test_binary_gemv_opposite_signs_cancel() {
    let n = 2; let k = 64;
    // row 0: all +1   row 1: all -1
    let mut w_data = vec![1.0f32; n * k];
    for i in k..n * k { w_data[i] = -1.0; }
    let x_data = vec![1.0f32; k];

    let packed          = PackedMatrix::pack_f32(&w_data, n, k, QuantMode::Binary).unwrap();
    let (x_packed, x_scale) = quantise_activation(&x_data);

    let mut y = vec![0.0f32; n];
    bitgemv_cpu(&packed, &x_packed, x_scale, &mut y);

    assert!(y[0] > 0.0, "row 0 should be positive: {}", y[0]);
    assert!(y[1] < 0.0, "row 1 should be negative: {}", y[1]);
}

// ---------------------------------------------------------------------------
// Ternary GEMV tests
// ---------------------------------------------------------------------------

#[test]
fn test_ternary_gemv_zeros_contribute_nothing() {
    let n = 4; let k = 64;
    let w_data = vec![0.0f32; n * k];
    let x_data = vec![1.0f32; k];

    let packed          = PackedMatrix::pack_f32(&w_data, n, k, QuantMode::Ternary).unwrap();
    let (x_packed, x_scale) = quantise_activation(&x_data);

    let mut y = vec![0.0f32; n];
    bitgemv_cpu(&packed, &x_packed, x_scale, &mut y);

    for &v in &y {
        assert_abs_diff_eq!(v, 0.0, epsilon = 1e-6);
    }
}

#[test]
fn test_ternary_gemv_sign_correct() {
    let n = 2; let k = 4;
    // row 0: all +1   row 1: all -1
    let w_data = vec![
         1.0,  1.0,  1.0,  1.0,
        -1.0, -1.0, -1.0, -1.0,
    ];
    let x_data = vec![1.0f32; k];

    let packed          = PackedMatrix::pack_f32(&w_data, n, k, QuantMode::Ternary).unwrap();
    let (x_packed, x_scale) = quantise_activation(&x_data);

    let mut y = vec![0.0f32; n];
    bitgemv_cpu(&packed, &x_packed, x_scale, &mut y);

    assert!(y[0] > 0.0, "row 0 should be positive: {}", y[0]);
    assert!(y[1] < 0.0, "row 1 should be negative: {}", y[1]);
}

#[test]
fn test_ternary_gemv_mixed_weights() {
    let n = 1; let k = 6;
    // [+1, -1, 0, +1, -1, 0] dot [1, 1, 1, 1, 1, 1]
    // = 1 - 1 + 0 + 1 - 1 + 0 = 0
    let w_data = vec![1.0f32, -1.0, 0.0, 1.0, -1.0, 0.0];
    let x_data = vec![1.0f32; k];

    let packed          = PackedMatrix::pack_f32(&w_data, n, k, QuantMode::Ternary).unwrap();
    let (x_packed, x_scale) = quantise_activation(&x_data);

    let mut y = vec![0.0f32; n];
    bitgemv_cpu(&packed, &x_packed, x_scale, &mut y);

    assert!(y[0].abs() < 1.0, "expected near zero, got {}", y[0]);
}

// ---------------------------------------------------------------------------
// Scale computation tests
// ---------------------------------------------------------------------------

#[test]
fn test_absmax_scale() {
    let data = vec![-3.0f32, 1.5, 2.0, -0.5];
    let s    = bitnet::quantization::scale::compute_absmax_scale(&data);
    assert_abs_diff_eq!(s, 3.0, epsilon = 1e-6);
}

#[test]
fn test_mean_abs_threshold() {
    let data = vec![2.0f32, -2.0, 2.0, -2.0];
    let t    = bitnet::quantization::scale::compute_mean_abs(&data);
    assert_abs_diff_eq!(t, 2.0, epsilon = 1e-6);
}

#[test]
fn test_quantise_activation_positive_maps_to_one() {
    let x               = vec![1.0f32, 2.0, 3.0, 4.0];
    let (packed, scale) = quantise_activation(&x);
    // all positive → all bits should be 1
    assert_eq!(packed[0] & 0xF, 0xF);
    assert!(scale > 0.0);
}

#[test]
fn test_quantise_activation_negative_maps_to_zero() {
    let x               = vec![-1.0f32, -2.0, -3.0, -4.0];
    let (packed, _scale) = quantise_activation(&x);
    // all negative → all bits should be 0
    assert_eq!(packed[0] & 0xF, 0x0);
}

// ---------------------------------------------------------------------------
// Memory estimate tests
// ---------------------------------------------------------------------------

#[test]
fn test_memory_estimate_3b_ternary() {
    let bytes = bitnet::utils::memory::estimate_model_memory(
        26, 3200, 8640, 32000, QuantMode::Ternary,
    );
    assert!(
        bytes < 2_000_000_000,
        "3B ternary should be < 2 GB, got {} bytes", bytes
    );
}

#[test]
fn test_memory_estimate_binary_less_than_ternary() {
    let binary  = bitnet::utils::memory::estimate_model_memory(
        32, 4096, 11008, 32000, QuantMo
