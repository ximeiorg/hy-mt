//! Model registry.
//!
//! Everything in this crate is *configuration driven*: the 1.8B and 7B
//! checkpoints share the exact same code path (only `config.json` differs), and
//! the FP8 and bf16 releases differ only in how weights are stored.
//!
//! Architectures live in their own module and implement [`CausalLm`], so a
//! future release with a different structure (e.g. the MoE `Hy-MT3-30B-A3B`)
//! can be added as a new module without touching the tokenizer, sampler,
//! generation loop or the translation API.
//!
//! ```text
//! config.json ──> models::load_model() ──> Box<dyn CausalLm>
//!                                               │
//!                     ┌─────────────────────────┴──────────────────────┐
//!                     │                                                │
//!          models::hunyuan_dense                             (future) models::hunyuan_moe
//!      Hy-MT3-1.8B / -1.8B-FP8 / 7B / 7B-FP8                   Hy-MT3-30B-A3B
//! ```

use std::path::Path;

use candle_core::{DType, Device, Result as CandleResult, Tensor};

use crate::cache::KVCache;
use crate::config::Config;
use crate::error::{Error, Result};

pub mod hunyuan_dense;

/// A decoder-only causal language model.
///
/// Implement this trait to plug a new architecture into [`crate::Translator`],
/// [`crate::generation`] and the sampler. Implementations are `Send` (all candle
/// tensors are) so that a [`crate::Translator`] can be moved between threads or
/// shared behind a mutex.
pub trait CausalLm: Send {
    /// Parsed `config.json`.
    fn config(&self) -> &Config;

    /// Compute dtype of the weights.
    fn dtype(&self) -> DType;

    /// Device the weights live on.
    fn device(&self) -> &Device;

    /// Human readable architecture name (for logs).
    fn architecture(&self) -> &'static str;

    /// Runs the model over `input_ids` of shape `(1, seq_len)` and returns the
    /// logits of the **last** position (`(1, 1, vocab)`) when `last_only` is
    /// set, or of every position (`(1, seq_len, vocab)`).
    ///
    /// The keys/values of the pass are appended to `cache`.
    fn forward(
        &mut self,
        input_ids: &Tensor,
        cache: &mut KVCache,
        last_only: bool,
    ) -> CandleResult<Tensor>;

    /// Allocates a KV cache able to hold `capacity` tokens.
    fn new_cache(&self, capacity: usize) -> Result<KVCache> {
        let cfg = self.config();
        KVCache::new(
            cfg.num_hidden_layers,
            cfg.num_key_value_heads(),
            cfg.head_dim(),
            capacity,
            self.dtype(),
            self.device(),
        )
    }
}

/// How to instantiate a model.
#[derive(Debug, Clone)]
pub struct ModelOptions {
    /// Device to run on.
    pub device: Device,
    /// Weight dtype. `None` picks a sensible default for the device
    /// (f16 on Metal, bf16 on CUDA, f32 on CPU).
    pub dtype: Option<DType>,
    /// Print loading progress.
    pub verbose: bool,
}

impl ModelOptions {
    /// Metal (macOS) when available, else CPU.
    pub fn new() -> Result<Self> {
        Ok(Self {
            device: default_device()?,
            dtype: None,
            verbose: false,
        })
    }

    /// Runs on an explicit device.
    pub fn with_device(device: Device) -> Self {
        Self {
            device,
            dtype: None,
            verbose: false,
        }
    }

    /// Overrides the weight dtype.
    pub fn dtype(mut self, dtype: DType) -> Self {
        self.dtype = Some(dtype);
        self
    }

    /// Enables or disables loading logs.
    pub fn verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Resolved dtype.
    pub fn resolved_dtype(&self) -> DType {
        self.dtype.unwrap_or(match &self.device {
            Device::Metal(_) => DType::F16,
            Device::Cuda(_) => DType::BF16,
            _ => DType::F32,
        })
    }
}

/// Environment variable that overrides backend selection.
///
/// Accepted values: `cpu`, `metal[:N]`, `cuda[:N]`.  It is honoured by
/// [`default_device`], i.e. by `Translator::from_pretrained` and
/// `TranslatorBuilder` unless a device is set explicitly.
///
/// ```bash
/// HYMT_DEVICE=cpu      cargo run --release --features metal --example translate -- ...
/// HYMT_DEVICE=metal:0  cargo run --release --features metal --example translate -- ...
/// ```
pub const DEVICE_ENV: &str = "HYMT_DEVICE";

/// Parses a device specification (`"cpu"`, `"metal"`, `"metal:1"`, `"cuda:0"`).
pub fn device_from_str(spec: &str) -> Result<Device> {
    let spec = spec.trim();
    if spec.is_empty() {
        return Err(Error::invalid_arg("empty device specification"));
    }
    let (kind, ordinal) = match spec.split_once(':') {
        Some((kind, index)) => (
            kind.trim(),
            index
                .trim()
                .parse::<usize>()
                .map_err(|_| Error::invalid_arg(format!("invalid device index in '{spec}'")))?,
        ),
        None => (spec, 0usize),
    };
    match kind.to_ascii_lowercase().as_str() {
        "cpu" => Ok(Device::Cpu),
        "metal" | "mps" | "gpu" => {
            #[cfg(feature = "metal")]
            {
                Device::new_metal(ordinal).map_err(|err| {
                    Error::unsupported(format!("cannot open Metal device {ordinal}: {err}"))
                })
            }
            #[cfg(not(feature = "metal"))]
            {
                let _ = ordinal;
                Err(Error::unsupported(
                    "this build has no `metal` backend; rebuild with --features metal",
                ))
            }
        }
        "cuda" => {
            #[cfg(feature = "cuda")]
            {
                Device::new_cuda(ordinal).map_err(|err| {
                    Error::unsupported(format!("cannot open CUDA device {ordinal}: {err}"))
                })
            }
            #[cfg(not(feature = "cuda"))]
            {
                let _ = ordinal;
                Err(Error::unsupported(
                    "this build has no `cuda` backend; rebuild with --features cuda",
                ))
            }
        }
        other => Err(Error::invalid_arg(format!(
            "unknown device '{other}' (expected cpu, metal[:N] or cuda[:N])"
        ))),
    }
}

/// Picks the fastest compiled-in device, honouring [`DEVICE_ENV`].
///
/// Without the environment variable: Metal (if compiled in and present), then
/// CUDA (same), then the CPU.
pub fn default_device() -> Result<Device> {
    if let Some(spec) = std::env::var_os(DEVICE_ENV) {
        let spec = spec.to_string_lossy();
        if !spec.trim().is_empty() {
            return device_from_str(&spec);
        }
    }
    #[cfg(feature = "metal")]
    {
        if let Ok(device) = Device::new_metal(0) {
            return Ok(device);
        }
    }
    #[cfg(feature = "cuda")]
    {
        if let Ok(device) = Device::new_cuda(0) {
            return Ok(device);
        }
    }
    Ok(Device::Cpu)
}

/// Loads a model directory, dispatching on `config.json`.
///
/// Works for every checkpoint of the family: `Hy-MT3-1.8B`,
/// `Hy-MT3-1.8B-FP8`, `Hy-MT3-7B`, `Hy-MT3-7B-FP8`, ...
pub fn load_model(dir: impl AsRef<Path>, opts: &ModelOptions) -> Result<Box<dyn CausalLm>> {
    let dir = dir.as_ref();
    let cfg = Config::from_dir(dir)?;
    let arch = cfg.architectures.first().cloned().unwrap_or_default();
    let model_type = cfg.model_type.clone().unwrap_or_default();
    match (model_type.as_str(), arch.as_str()) {
        ("hunyuan_v1_dense", _) | (_, "HunYuanDenseV1ForCausalLM") | ("llama", _) => {
            Ok(Box::new(hunyuan_dense::HunyuanDense::load(dir, cfg, opts)?))
        }
        ("hunyuan_v1_moe", _) | (_, "HunYuanMoEV1ForCausalLM") => Err(Error::unsupported(
            "the MoE family (Hy-MT3-30B-A3B) is not implemented yet; \
             the dense Hy-MT3-1.8B / 7B checkpoints are supported. \
             Adding it only requires a new module implementing `CausalLm`.",
        )),
        (mt, arch) => Err(Error::unsupported(format!(
            "unknown architecture (model_type = '{mt}', architectures = '{arch}')"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_cpu_device() {
        assert!(device_from_str("cpu").unwrap().is_cpu());
        assert!(device_from_str("  CPU  ").unwrap().is_cpu());
    }

    #[test]
    fn rejects_unknown_device() {
        assert!(device_from_str("tpu").is_err());
        assert!(device_from_str("").is_err());
    }

    #[test]
    fn rejects_bad_index() {
        assert!(device_from_str("metal:x").is_err());
        assert!(device_from_str("cuda:-1").is_err());
    }

    #[cfg(not(feature = "metal"))]
    #[test]
    fn metal_without_feature_is_an_error() {
        let err = device_from_str("metal:0").unwrap_err().to_string();
        assert!(err.contains("metal"), "unexpected error: {err}");
    }

    #[cfg(not(feature = "cuda"))]
    #[test]
    fn cuda_without_feature_is_an_error() {
        let err = device_from_str("cuda:0").unwrap_err().to_string();
        assert!(err.contains("cuda"), "unexpected error: {err}");
    }
}
