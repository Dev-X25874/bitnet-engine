use std::{collections::HashMap, path::Path};
use memmap2::Mmap;
use safetensors::SafeTensors;
use crate::error::{BitNetError, Result};
use super::{PackedMatrix, BitPacking, QuantMode};

pub type WeightMap = HashMap<String, PackedMatrix>;

pub fn load_safetensors_weights(
    path:       &Path,
    mode:       QuantMode,
    quant_keys: &[&str],
) -> Result<(WeightMap, HashMap<String, Vec<f32>>)>
{
    let file    = std::fs::File::open(path)
        .map_err(|e| BitNetError::ModelLoad(e.to_string()))?;
    let mmap    = unsafe { Mmap::map(&file) }
        .map_err(|e| BitNetError::ModelLoad(e.to_string()))?;
    let tensors = SafeTensors::deserialize(&mmap)
        .map_err(|e| BitNetError::ModelLoad(e.to_string()))?;

    let mut packed_map: WeightMap                 = HashMap::new();
    let mut extras: HashMap<String, Vec<f32>>     = HashMap::new();

    for (name, view) in tensors.tensors() {
        let should_pack = quant_keys.iter().any(|k| name.contains(k));
        let f32_data    = bytes_to_f32(view.data(), view.dtype())?;
        let shape       = view.shape();

        if should_pack && shape.len() == 2 {
            let (rows, cols) = (shape[0], shape[1]);
            let packed = PackedMatrix::pack_f32(&f32_data, rows, cols, mode)
                .map_err(|e| BitNetError::ModelLoad(e.to_string()))?;
            packed_map.insert(name.to_string(), packed);
        } else {
            extras.insert(name.to_string(), f32_data);
        }
    }

    log::info!("Loaded {} packed tensors, {} float tensors from {:?}",
               packed_map.len(), extras.len(), path);

    Ok((packed_map, extras))
}

fn bytes_to_f32(data: &[u8], dtype: safetensors::Dtype) -> Result<Vec<f32>> {
    use safetensors::Dtype::*;
    match dtype {
        F32 => Ok(data.chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect()),
        F16 => Ok(data.chunks_exact(2)
            .map(|b| {
                let raw = u16::from_le_bytes(b.try_into().unwrap());
                half::f16::from_bits(raw).to_f32()
            })
            .collect()),
        BF16 => Ok(data.chunks_exact(2)
            .map(|b| {
                let raw = u16::from_le_bytes(b.try_into().unwrap());
                half::bf16::from_bits(raw).to_f32()
            })
            .collect()),
        I8 => Ok(data.iter()
            .map(|&b| if b as i8 >= 0 { 1.0 } else { -1.0 })
            .collect()),
        other => Err(BitNetError::ModelLoad(
            format!("Unsupported dtype {:?}", other)
        )),
    }
}
