use thiserror::Error;

pub type Result<T, E = BitNetError> = std::result::Result<T, E>;

#[derive(Debug, Error)]
pub enum BitNetError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Model load error: {0}")]
    ModelLoad(String),

    #[error("Shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch { expected: Vec<usize>, got: Vec<usize> },

    #[error("Unsupported quantisation mode: {0:?}")]
    UnsupportedQuantMode(String),

    #[error("CUDA error: {0}")]
    Cuda(String),

    #[error("Out-of-memory: requested {requested} bytes")]
    OutOfMemory { requested: usize },

    #[error("Tokeniser error: {0}")]
    Tokeniser(String),

    #[error("Invalid configuration: {0}")]
    Config(String),

    #[error(transparent)]
    Other(#[from] anyhow::Error),
}
