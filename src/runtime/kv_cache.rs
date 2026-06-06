use half::f16;

const BLOCK_SIZE: usize = 16;

#[derive(Clone)]
struct Block {
    data: Vec<f16>,
}

impl Block {
    fn new(num_heads: usize, head_dim: usize) -> Self {
        Self { data: vec![f16::ZERO; BLOCK_SIZE * num_heads * head_dim] }
    }
}

pub struct KvCache {
    max_seq_len: usize,
    max_batch:   usize,
    num_layers:  usize,
    num_heads:   usize,
    head_dim:    usize,
    free_blocks: Vec<usize>,
    blocks_k:    Vec<Vec<Block>>,
    blocks_v:    Vec<Vec<Block>>,
    session_tables: Vec<Option<Vec<usize>>>,
}

impl KvCache {
    pub fn new(
        max_seq_len: usize,
        max_batch:   usize,
        num_layers:  usize,
        num_heads:   usize,
        head_dim:    usize,
    ) -> Self {
        let max_blocks = max_batch * (max_seq_len / BLOCK_SIZE + 1) + 1;

        let blocks_k: Vec<Vec<Block>> = (0..max_blocks)
            .map(|_| (0..num_layers)
                .map(|_| Block::new(num_heads, head_dim))
                .collect())
            .collect();

        let blocks_v: Vec<Vec<Block>> = (0..max_blocks)
            .map(|_| (0..num_layers)
                .map(|_| Block::new(num_heads, head_dim))
                .collect())
            .collect();

        let free_blocks = (0..max_blocks).collect();

        Self {
            max_seq_len,
            max_batch,
            num_layers,
            num_heads,
            head_dim,
            free_blocks,
            blocks_k,
            blocks_v,
            session_tables: vec![None; max_batch],
        }
    }

    pub fn alloc_session(&mut self) -> Option<usize> {
        let sid = self.session_tables.iter().position(|s| s.is_none())?;
        self.session_tables[sid] = Some(Vec::new());
        Some(sid)
    }

    pub fn free_session(&mut self, sid: usize) {
        if let Some(Some(blocks)) = self.session_tables.get_mut(sid) {
            self.free_blocks.extend(blocks.drain(..));
            self.session_tables[sid] = None;
        }
    }

    pub fn extend_session(&mut self, sid: usize) -> Option<usize> {
        let block_id = self.free_blocks.pop()?;
        if let Some(Some(table)) = self.session_tables.get_mut(sid) {
            table.push(block_id);
        }
        Some(block_id)
    }

    pub fn key_block_mut(&mut self, sid: usize, layer: usize, pos: usize) -> &mut [f16] {
        let block_idx = pos / BLOCK_SIZE;
        let table     = self.session_tables[sid].as_ref().unwrap();
        let block_id  = table[block_idx];
        let offset    = (pos % BLOCK_SIZE) * self.num_heads * self.head_dim;
        let end       = offset + self.num_heads * self.head_dim;
        &mut self.blocks_k[block_id][layer].data[offset..end]
    }

    pub fn val_block_mut(&mut self, sid: usize, layer: usize, pos: usize) -> &mut [f16] {
        let block_idx = pos / BLOCK_SIZE;
        let table     = self.session_tables[sid].as_ref().unwrap();
        let block_id  = table[block_idx];
        let offset    = (pos % BLOCK_SIZE) * self.num_heads * self.head_dim;
        let end       = offset + self.num_heads * self.head_dim;
        &mut self.blocks_v[block_id][layer].data[offset..end]
    }

    pub fn block_size() -> usize { BLOCK_SIZE }
}
