use std::sync::Arc;
use parking_lot::RwLock;
use crate::error::Result;
use super::kv_cache::KvCache;

#[derive(Debug, Clone)]
pub struct GenerateConfig {
    pub max_new_tokens:     usize,
    pub temperature:        f32,
    pub top_p:              f32,
    pub top_k:              usize,
    pub repetition_penalty: f32,
    pub stop_sequences:     Vec<String>,
}

impl Default for GenerateConfig {
    fn default() -> Self {
        Self {
            max_new_tokens:     256,
            temperature:        0.7,
            top_p:              0.9,
            top_k:              50,
            repetition_penalty: 1.1,
            stop_sequences:     vec![],
        }
    }
}

pub struct Session {
    pub cfg:    GenerateConfig,
    kv_pool:    Arc<RwLock<KvCache>>,
    session_id: Option<usize>,
    token_ids:  Vec<u32>,
    cur_pos:    usize,
}

impl Session {
    pub fn new(cfg: GenerateConfig, kv_pool: Arc<RwLock<KvCache>>) -> Self {
        let session_id = kv_pool.write().alloc_session();
        Self {
            cfg,
            kv_pool,
            session_id,
            token_ids: Vec::new(),
            cur_pos: 0,
        }
    }

    pub fn run(&mut self, _prompt: &str) -> Result<String> {
        log::info!(
            "Session {:?}: starting generation (max_new_tokens={})",
            self.session_id,
            self.cfg.max_new_tokens
        );
        Ok(String::new())
    }

    pub fn sample(&self, logits: &mut [f32]) -> u32 {
        apply_temperature(logits, self.cfg.temperature);
        top_k_filter(logits, self.cfg.top_k);
        top_p_sample(logits, self.cfg.top_p)
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        if let Some(sid) = self.session_id {
            self.kv_pool.write().free_session(sid);
        }
    }
}

fn apply_temperature(logits: &mut [f32], temp: f32) {
    if temp > 0.0 {
        logits.iter_mut().for_each(|l| *l /= temp);
    }
    softmax_inplace(logits);
}

fn softmax_inplace(v: &mut [f32]) {
    let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for x in v.iter_mut() {
        *x = (*x - max).exp();
        sum += *x;
    }
    v.iter_mut().for_each(|x| *x /= sum);
}

fn top_k_filter(probs: &mut [f32], k: usize) {
    if k == 0 || k >= probs.len() { return; }
    let mut indexed: Vec<(usize, f32)> = probs
        .iter()
        .copied()
        .enumerate()
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    for (i, _) in indexed.iter().skip(k) {
        probs[*i] = 0.0;
    }
    let sum: f32 = probs.iter().sum();
    if sum > 0.0 {
        probs.iter_mut().for_each(|p| *p /= sum);
    }
}

fn top_p_sample(probs: &[f32], p: f32) -> u32 {
    let mut indexed: Vec<(usize, f32)> = probs
        .iter()
        .copied()
        .enumerate()
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let mut cumsum   = 0.0f32;
    let threshold    = rand_f32() * p.min(1.0);

    for (idx, prob) in &indexed {
        cumsum += prob;
        if cumsum >= threshold {
            return *idx as u32;
        }
    }
    indexed.last().map(|(i, _)| *i as u32).unwrap_or(0)
}

fn rand_f32() -> f32 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static STATE: AtomicU64 = AtomicU64::new(0x4d595df4d0f33173);
    let s = STATE.fetch_add(0x6c62272e07bb0142, Ordering::Relaxed);
    let s = s ^ (s >> 30);
    let s = s.wrapping_mul(0xbf58476d1ce4e5b9);
    let s = s ^ (s >> 27);
    let s = s.wrapping_mul(0x94d049bb133111eb);
    let s = s ^ (s >> 31);
    (s as f32) / (u64::MAX as f32)
}
