//! Error types used across the crate.

use candle_core::Error as CandleError;

/// Errors returned by this crate.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("candle error: {0}")]
    Candle(#[from] CandleError),

    #[error("tokenizer error: {0}")]
    Tokenizer(String),

    #[error("invalid model directory '{dir}': {reason}")]
    InvalidModelDir { dir: String, reason: String },

    #[error("unsupported model: {0}")]
    Unsupported(String),

    #[error("missing tensor '{0}' in the safetensors files")]
    MissingTensor(String),

    #[error("invalid safetensors file '{path}': {reason}")]
    InvalidSafetensors { path: String, reason: String },

    #[error("invalid argument: {0}")]
    InvalidArgument(String),
}

/// Convenient result alias.
pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    pub(crate) fn invalid_arg(msg: impl Into<String>) -> Self {
        Error::InvalidArgument(msg.into())
    }

    pub(crate) fn unsupported(msg: impl Into<String>) -> Self {
        Error::Unsupported(msg.into())
    }
}

impl From<tokenizers::Error> for Error {
    fn from(value: tokenizers::Error) -> Self {
        Error::Tokenizer(value.to_string())
    }
}

impl From<Error> for CandleError {
    fn from(value: Error) -> Self {
        match value {
            Error::Candle(err) => err,
            other => CandleError::Msg(other.to_string()),
        }
    }
}
