use std::collections::VecDeque;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestState {
    Waiting,
    Running,
    Finished,
    Preempted,
}

#[derive(Debug)]
pub struct Request {
    pub id:      u64,
    pub tokens:  Vec<u32>,
    pub max_new: usize,
    pub state:   RequestState,
    pub gen_len: usize,
}

pub struct Scheduler {
    waiting:          VecDeque<Request>,
    running:          Vec<Request>,
    max_batch:        usize,
    free_blocks:      usize,
    blocks_per_token: usize,
}

impl Scheduler {
    pub fn new(
        max_batch:        usize,
        total_blocks:     usize,
        blocks_per_token: usize,
    ) -> Self {
        Self {
            waiting: VecDeque::new(),
            running: Vec::new(),
            max_batch,
            free_blocks: total_blocks,
            blocks_per_token,
        }
    }

    pub fn add(&mut self, req: Request) {
        self.waiting.push_back(req);
    }

    pub fn step(&mut self) -> Vec<u64> {
        while self.running.len() < self.max_batch {
            if let Some(mut req) = self.waiting.pop_front() {
                let blocks_needed = (req.tokens.len() + req.max_new) / 16 + 1;
                if self.free_blocks >= blocks_needed {
                    self.free_blocks -= blocks_needed;
                    req.state = RequestState::Running;
                    self.running.push(req);
                } else {
                    self.waiting.push_front(req);
                    break;
                }
            } else {
                break;
            }
        }
        self.running.iter().map(|r| r.id).collect()
    }

    pub fn finish(&mut self, id: u64) {
        if let Some(pos) = self.running.iter().position(|r| r.id == id) {
            let req = self.running.remove(pos);
            let blocks = (req.tokens.len() + req.gen_len) / 16 + 1;
            self.free_blocks += blocks;
        }
    }

    pub fn preempt(&mut self, id: u64) {
        if let Some(pos) = self.running.iter().position(|r| r.id == id) {
            let mut req = self.running.remove(pos);
            let blocks  = (req.tokens.len() + req.gen_len) / 16 + 1;
            self.free_blocks += blocks;
            req.state = RequestState::Preempted;
            self.waiting.push_front(req);
        }
    }

    pub fn running_count(&self) -> usize { self.running.len() }
    pub fn waiting_count(&self) -> usize { self.waiting.len() }
}
