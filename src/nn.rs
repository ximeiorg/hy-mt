//! Small neural-network building blocks used by the model.

use candle_core::{Result as CandleResult, Tensor};

/// A dense layer without activation.
///
/// The weight is stored transposed (`[in_features, out_features]`) so that the
/// forward pass is a single `x @ w` matmul.
#[derive(Debug, Clone)]
pub struct Linear {
    w: Tensor,
    b: Option<Tensor>,
}

impl Linear {
    /// Builds a layer from an already transposed weight.
    pub fn new(w: Tensor, b: Option<Tensor>) -> Self {
        Self { w, b }
    }

    /// The `(in_features, out_features)` weight matrix.
    pub fn weight(&self) -> &Tensor {
        &self.w
    }

    /// Applies the layer.
    pub fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        // `broadcast_matmul` folds the leading dimensions into the row dim when
        // the weight is rank-2, i.e. exactly the behaviour we want for
        // `(batch, seq, in) @ (in, out)`.
        let y = x.broadcast_matmul(&self.w)?;
        match &self.b {
            Some(b) => y.broadcast_add(b),
            None => Ok(y),
        }
    }
}

/// Root-mean-square layer normalisation.
#[derive(Debug, Clone)]
pub struct RmsNorm {
    w: Tensor,
    eps: f32,
}

impl RmsNorm {
    /// Builds a normalisation layer from its gain vector.
    pub fn new(w: Tensor, eps: f64) -> Self {
        Self { w, eps: eps as f32 }
    }

    /// Normalises the last dimension of `x`.
    pub fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        candle_nn::ops::rms_norm(x, &self.w, self.eps)
    }

    /// The gain vector.
    pub fn weight(&self) -> &Tensor {
        &self.w
    }
}
