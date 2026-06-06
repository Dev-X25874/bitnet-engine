use crate::quantization::QuantMode;

pub fn estimate_model_memory(
    n_layers: usize,
    d_model: usize,
    d_ff: usize,
    vocab_size: usize,
    mode: QuantMode,
) -> usize {
    let bits_per_weight: usize = match mode {
        QuantMode::Binary => 1,
        QuantMode::Ternary => 2,
    };

    let attn_weights = 4 * d_model * d_model;
    let ffn_weights = 3 * d_model * d_ff;

    let weights_per_layer = attn_weights + ffn_weights;
    let total_weight_bits = n_layers * weights_per_layer * bits_per_weight;
    let weight_bytes = total_weight_bits.div_ceil(8);

    let embed_bytes = vocab_size * d_model * 2;
    let norm_bytes = n_layers * 2 * d_model * 2;

    weight_bytes + embed_bytes + norm_bytes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_estimate_7b_ternary() {
        let bytes = estimate_model_memory(32, 4096, 11008, 32000, QuantMode::Ternary);
        let gb = bytes as f64 / 1e9;
        assert!(gb > 1.0 && gb < 10.0, "unexpected size: {:.2} GB", gb);
    }

    #[test]
    fn binary_smaller_than_ternary() {
        let binary = estimate_model_memory(32, 4096, 11008, 32000, QuantMode::Binary);
        let ternary = estimate_model_memory(32, 4096, 11008, 32000, QuantMode::Ternary);
        assert!(binary < ternary, "binary should use less memory than ternary");
    }
}
