//! SwiGLU feed-forward network (`gate_proj` / `up_proj` / `down_proj`).

use candle_core::{DType, Device, Result as CandleResult, Tensor};

use crate::config::Config;
use crate::error::Result;
use crate::nn::Linear;
use crate::weights::WeightSource;

/// Position-wise feed-forward block.
///
/// `gate_proj` and `up_proj` share the same input, so they are concatenated
/// into one `(in, 2 * intermediate)` matrix at load time.
#[derive(Debug)]
pub struct Mlp {
    gate_up_proj: Linear,
    down_proj: Linear,
    intermediate: usize,
}

impl Mlp {
    /// Loads `layers.{i}.mlp.*` from the checkpoint.
    pub fn load(
        weights: &WeightSource,
        prefix: &str,
        cfg: &Config,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let mlp = format!("{prefix}.mlp");
        let gate = weights.tensor(&format!("{mlp}.gate_proj.weight"), dtype, device)?;
        let up = weights.tensor(&format!("{mlp}.up_proj.weight"), dtype, device)?;
        let gate_up_w = Tensor::cat(&[&gate, &up], 0)?.t()?.contiguous()?;
        let gate_up_b = if cfg.mlp_bias {
            let gb = weights.tensor(&format!("{mlp}.gate_proj.bias"), dtype, device)?;
            let ub = weights.tensor(&format!("{mlp}.up_proj.bias"), dtype, device)?;
            Some(Tensor::cat(&[&gb, &ub], 0)?)
        } else {
            None
        };
        Ok(Self {
            gate_up_proj: Linear::new(gate_up_w, gate_up_b),
            down_proj: weights.linear(&format!("{mlp}.down_proj"), dtype, device, cfg.mlp_bias)?,
            intermediate: cfg.intermediate_size,
        })
    }

    /// `silu(gate(x)) * up(x)`, then `down(...)`.
    pub fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let gate_up = self.gate_up_proj.forward(x)?;
        let gate = gate_up.narrow(2, 0, self.intermediate)?;
        let up = gate_up.narrow(2, self.intermediate, self.intermediate)?;
        let act = candle_nn::ops::silu(&gate)?;
        self.down_proj.forward(&(act * up)?)
    }
}
