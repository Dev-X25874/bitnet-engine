use std::time::{Duration, Instant};

pub struct StepTimer {
    label: String,
    start: Option<Instant>,
    total: Duration,
    count: u64,
}

impl StepTimer {
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            start: None,
            total: Duration::ZERO,
            count: 0,
        }
    }

    pub fn start(&mut self) {
        self.start = Some(Instant::now());
    }

    pub fn stop(&mut self) {
        if let Some(s) = self.start.take() {
            self.total += s.elapsed();
            self.count += 1;
        }
    }

    pub fn mean_ms(&self) -> f64 {
        if self.count == 0 { return 0.0; }
        self.total.as_secs_f64() * 1000.0 / self.count as f64
    }

    pub fn report(&self) {
        log::info!(
            "[{}] steps={} mean={:.2}ms total={:.1}ms",
            self.label,
            self.count,
            self.mean_ms(),
            self.total.as_secs_f64() * 1000.0
        );
    }
}
