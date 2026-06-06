use std::{path::Path, sync::Arc};
use parking_lot::RwLock;
use crate::error::Result;
use crate::quantization::{load_safetensors_weights, QuantMode};
use super::kv_cache::KvCache;
use super::session::{GenerateConfig, Session};

#[derive(Debug, Clone)]
pub struct EngineConfig {
    pub model_path:  std::path::PathBuf,
    pub quant_mode:  QuantMode,
    pub max_seq_len: usize,
    pub max_batch:   usize,
    pub num_threads: usize,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            model_path:  std::path::PathBuf::from("model"),
            quant_mode:  QuantMode::Ternary,
            max_seq_len: 2048,
            max_batch:   8,
            num_threads: num_cpus::get(),
        }
    }
}

pub struct Engine {
    cfg:     EngineConfig,
    kv_pool: Arc<RwLock<KvCache>>,
    _weights: (),
}

impl Engine {
    pub fn from_config(cfg: EngineConfig) -> Result<Self> {
        rayon::ThreadPoolBuilder::new()
            .num_threads(cfg.num_threads)
            .build_global()
            .ok();

        let model_file = cfg.model_path.join("model.safetensors");
        let (_packed, _extras) = load_safetensors_weights(
            &model_file,
            cfg.quant_mode,
            &[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
        )?;

        log::info!(
            "Engine initialised: mode={:?}, threads={}",
            cfg.quant_mode,
            cfg.num_threads
        );

        let kv_pool = Arc::new(RwLock::new(KvCache::new(
            cfg.max_seq_len,
            cfg.max_batch,
            32,   // num_layers
            32,   // num_heads
            128,  // head_dim
        )));

        Ok(Self { cfg, kv_pool, _weights: () })
    }

    pub fn new_session(&self, gen_cfg: GenerateConfig) -> Session {
        Session::new(gen_cfg, Arc::clone(&self.kv_pool))
    }

    pub fn generate(&self, prompt: &str, gen_cfg: GenerateConfig) -> Result<String> {
        let mut session = self.new_session(gen_cfg);
        session.run(prompt)
    }
}
