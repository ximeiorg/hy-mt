//! Compatibility layer for the `LogitsProcessor` / `Sampling` types of
//! `hy-mt` 0.1.
//!
//! The engine inside this crate uses [`crate::sampling::Sampler`]; this module
//! keeps the old names working, including the `LogitsProcessor::sample` shape
//! that took a `&Tensor`.

use candle_core::{DType, Result, Tensor};

use crate::sampling::{Sampler, SamplingConfig};

/// Sampling strategies (same variants as `hy-mt` 0.1).
#[derive(Clone, PartialEq, Debug)]
pub enum Sampling {
    /// Greedy / argmax.
    ArgMax,
    /// Temperature only.
    All {
        /// Softmax temperature.
        temperature: f64,
    },
    /// Top-k.
    TopK {
        /// Number of candidates.
        k: usize,
        /// Softmax temperature.
        temperature: f64,
    },
    /// Nucleus sampling.
    TopP {
        /// Cumulative probability threshold.
        p: f64,
        /// Softmax temperature.
        temperature: f64,
    },
    /// Top-k followed by top-p.
    TopKThenTopP {
        /// Number of candidates.
        k: usize,
        /// Cumulative probability threshold.
        p: f64,
        /// Softmax temperature.
        temperature: f64,
    },
    /// Gumbel-softmax sampling (mapped to plain temperature sampling, which is
    /// distributionally equivalent for our purposes).
    GumbelSoftmax {
        /// Softmax temperature.
        temperature: f64,
    },
}

impl Sampling {
    /// Converts to the engine's configuration.
    pub fn to_config(&self) -> SamplingConfig {
        let mut cfg = SamplingConfig::default();
        match self {
            Self::ArgMax => return SamplingConfig::greedy(),
            Self::All { temperature } | Self::GumbelSoftmax { temperature } => {
                cfg.temperature = *temperature;
                cfg.top_k = 0;
                cfg.top_p = 1.0;
            }
            Self::TopK { k, temperature } => {
                cfg.temperature = *temperature;
                cfg.top_k = *k;
                cfg.top_p = 1.0;
            }
            Self::TopP { p, temperature } => {
                cfg.temperature = *temperature;
                cfg.top_k = 0;
                cfg.top_p = *p;
            }
            Self::TopKThenTopP { k, p, temperature } => {
                cfg.temperature = *temperature;
                cfg.top_k = *k;
                cfg.top_p = *p;
            }
        }
        cfg.repetition_penalty = 1.0;
        cfg
    }
}

/// Samples from raw logits (no history-based penalties).
#[derive(Debug)]
pub struct LogitsProcessor {
    sampler: Sampler,
}

impl LogitsProcessor {
    /// Builds a processor from an explicit sampling strategy.
    pub fn from_sampling(seed: u64, sampling: Sampling) -> Self {
        let mut cfg = sampling.to_config();
        cfg.seed = Some(seed);
        Self {
            sampler: Sampler::new(cfg),
        }
    }

    /// Temperature + top-p (the `hy-mt` 0.1 constructor).
    pub fn new(seed: u64, temperature: Option<f64>, top_p: Option<f64>) -> Self {
        let temperature = temperature.filter(|t| *t >= 1e-7);
        let sampling = match temperature {
            None => Sampling::ArgMax,
            Some(temperature) => match top_p {
                None => Sampling::All { temperature },
                Some(p) => Sampling::TopP { p, temperature },
            },
        };
        Self::from_sampling(seed, sampling)
    }

    /// Temperature + top-k + top-p.
    pub fn new_with_topk(
        seed: u64,
        temperature: Option<f64>,
        top_k: Option<usize>,
        top_p: Option<f64>,
    ) -> Self {
        let temperature = temperature.filter(|t| *t >= 1e-7);
        let sampling = match temperature {
            None => Sampling::ArgMax,
            Some(temperature) => match (top_k, top_p) {
                (Some(k), Some(p)) => Sampling::TopKThenTopP { k, p, temperature },
                (Some(k), None) => Sampling::TopK { k, temperature },
                (None, Some(p)) => Sampling::TopP { p, temperature },
                (None, None) => Sampling::All { temperature },
            },
        };
        Self::from_sampling(seed, sampling)
    }

    /// The sampling strategy in use.
    pub fn sampling(&self) -> &SamplingConfig {
        self.sampler.config()
    }

    /// Samples one token id from a `(vocab,)` or `(1, vocab)` logits tensor.
    pub fn sample(&mut self, logits: &Tensor) -> Result<u32> {
        let flat = logits.flatten_all()?.to_dtype(DType::F32)?;
        let mut values = flat.to_vec1::<f32>()?;
        Ok(self.sampler.sample(&mut values, &[]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn argmax_matches() {
        let mut lp = LogitsProcessor::from_sampling(0, Sampling::ArgMax);
        let logits = Tensor::new(&[0.1f32, 3.0, 0.2], &Device::Cpu).unwrap();
        assert_eq!(lp.sample(&logits).unwrap(), 1);
    }

    #[test]
    fn top_k_restricts_support() {
        let mut lp = LogitsProcessor::new_with_topk(7, Some(1.0), Some(1), None);
        for _ in 0..16 {
            let logits = Tensor::new(&[0.0f32, 5.0, 1.0], &Device::Cpu).unwrap();
            assert_eq!(lp.sample(&logits).unwrap(), 1);
        }
    }

    #[test]
    fn sampling_variants_map() {
        assert_eq!(Sampling::ArgMax.to_config().temperature, 0.0);
        let cfg = Sampling::TopKThenTopP {
            k: 20,
            p: 0.6,
            temperature: 0.7,
        }
        .to_config();
        assert_eq!(cfg.top_k, 20);
        assert_eq!(cfg.top_p, 0.6);
        assert_eq!(cfg.temperature, 0.7);
    }
}
