use crate::error::Result;
use crate::quantization::PackedMatrix;
use crate::quantization::scale::quantise_activation;
use super::cpu_gemv::bitgemv_cpu;

pub struct LinearWeight {
    pub weight: PackedMatrix,
    pub bias:   Option<Vec<f32>>,
}

pub fn linear_forward(lw: &LinearWeight, x: &[f32]) -> Result<Vec<f32>> {
    let n = lw.weight.rows;
    let k = lw.weight.cols;
    assert_eq!(x.len(), k,
        "activation length mismatch: got {}, expected {}", x.len(), k);

    let mut y = if let Some(b) = &lw.bias {
        b.clone()
    } else {
        vec![0.0f32; n]
    };

    let (x_packed, x_scale) = quantise_activation(x);

    #[cfg(feature = "cuda")]
    {
        use crate::cuda_ffi;
        if cuda_ffi::is_available() {
            return cuda_ffi::linear_forward_cuda(lw, &x_packed, x_scale, y);
        }
    }

    bitgemv_cpu(&lw.weight, &x_packed, x_scale, &mut y);
    Ok(y)
}
