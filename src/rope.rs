//! Rotary position embeddings (RoPE).
//!
//! Hy-MT3 uses the Llama/`rotate_half` convention: the head dimension is split
//! in two halves and the second half is rotated onto the first.
//!
//! The base frequency comes from [`Config::rope_base`]: for the shipped
//! checkpoints (`"dynamic"` + `alpha = 1000`) transformers 5.6 uses the static
//! `ntk_alpha` variant, i.e. a one-off stretch of the base and a cos/sin
//! scaling factor of exactly 1.0. A length-dependent dynamic-NTK update is
//! still implemented for checkpoints without `alpha`.

use candle_core::{DType, Device, Result as CandleResult, Tensor};

use crate::config::{Config, RopeScaling};

/// Pre-computed cos/sin table for rotary embeddings.
#[derive(Debug)]
pub struct RotaryEmb {
    dim: usize,
    base: f64,
    scaling: Option<RopeScaling>,
    max_position_embeddings: usize,
    dtype: DType,
    device: Device,
    /// `(positions, cos[positions, dim/2], sin[positions, dim/2])`, always f32.
    table: Option<(usize, Tensor, Tensor)>,
}

impl RotaryEmb {
    /// Creates the embedding from a model config.
    pub fn new(cfg: &Config, dtype: DType, device: &Device) -> Self {
        Self {
            dim: cfg.head_dim(),
            base: cfg.rope_base(),
            scaling: cfg.rope_scaling.clone(),
            max_position_embeddings: cfg.max_position_embeddings,
            dtype,
            device: device.clone(),
            table: None,
        }
    }

    /// Head dimension the table was built for.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// RoPE base frequency in use (after `ntk_alpha` stretching).
    pub fn base(&self) -> f64 {
        self.base
    }

    /// Number of positions currently cached.
    pub fn cached_positions(&self) -> usize {
        self.table.as_ref().map(|(n, _, _)| *n).unwrap_or(0)
    }

    /// Inverse frequencies (one per rotated pair).
    fn inv_freq(&self, total_len: usize) -> Vec<f32> {
        let mut base = self.base;
        if let Some(rs) = &self.scaling {
            if rs.is_dynamic_ntk() && total_len > self.max_position_embeddings {
                // transformers' `_compute_dynamic_ntk_parameters`:
                // base *= (factor * len / max_pos - (factor - 1)) ** (dim / (dim - 2))
                let factor = rs.factor();
                let exponent = self.dim as f64 / (self.dim as f64 - 2.0);
                let ratio = factor * total_len as f64 / self.max_position_embeddings as f64;
                base *= (ratio - (factor - 1.0)).powf(exponent);
            }
        }
        let half = self.dim / 2;
        (0..half)
            .map(|i| (1.0 / base.powf((2 * i) as f64 / self.dim as f64)) as f32)
            .collect()
    }

    /// Makes sure positions `0..total` are available in the table.
    fn ensure(&mut self, total: usize) -> CandleResult<()> {
        let need = total.max(1);
        if self.cached_positions() >= need {
            return Ok(());
        }
        let mut len = self.cached_positions().max(256);
        while len < need {
            len *= 2;
        }
        let half = self.dim / 2;
        let inv = self.inv_freq(need);
        let mut freqs = Vec::with_capacity(len * half);
        for pos in 0..len {
            let p = pos as f32;
            for f in &inv {
                freqs.push(p * f);
            }
        }
        let freqs = Tensor::from_vec(freqs, (len, half), &self.device)?;
        // `attention_scaling` is 1.0 for every RoPE type used by Hy-MT3.
        // The fused kernel doubles the half-sized table internally
        // (`emb = cat(freqs, freqs)`), exactly like the reference.
        let cos = freqs.cos()?;
        let sin = freqs.sin()?;
        self.table = Some((len, cos, sin));
        Ok(())
    }


    /// cos/sin for positions `start..start + seq_len`, shaped `(seq, dim/2)`.
    pub fn cos_sin(&mut self, start: usize, seq_len: usize) -> CandleResult<(Tensor, Tensor)> {
        self.ensure(start + seq_len)?;
        let (_, cos, sin) = self.table.as_ref().expect("table was just filled");
        let cos = cos.narrow(0, start, seq_len)?;
        let sin = sin.narrow(0, start, seq_len)?;
        let cos = cos.to_dtype(self.dtype)?;
        let sin = sin.to_dtype(self.dtype)?;
        Ok((cos, sin))
    }

    /// Applies RoPE to `x` of shape `(batch, heads, seq, head_dim)`.
    ///
    /// Uses candle's fused kernel (CPU / Metal / CUDA), which is a good deal
    /// cheaper than composing `narrow`/`neg`/`cat`/`mul` in the graph.
    pub fn apply(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> CandleResult<Tensor> {
        let x = if x.is_contiguous() {
            x.clone()
        } else {
            x.contiguous()?
        };
        candle_nn::rotary_emb::rope(&x, cos, sin)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> Config {
        serde_json::from_str(
            r#"{"vocab_size":100,"hidden_size":64,"intermediate_size":128,
                "num_hidden_layers":1,"num_attention_heads":4,"num_key_value_heads":2,
                "head_dim":16,"rope_theta":10000.0,"tie_word_embeddings":true}"#,
        )
        .unwrap()
    }

    /// Hy-MT3's actual RoPE parameters (ntk_alpha, head_dim 128).
    fn hy_cfg() -> Config {
        serde_json::from_str(
            r#"{"vocab_size":120818,"hidden_size":2048,"intermediate_size":6144,
                "num_hidden_layers":32,"num_attention_heads":16,"num_key_value_heads":4,
                "head_dim":128,"rope_theta":10000.0,"tie_word_embeddings":true,
                "max_position_embeddings":262144,
                "rope_scaling":{"alpha":1000.0,"beta_fast":32,"beta_slow":1,
                                "factor":1.0,"mscale":1.0,"mscale_all_dim":1.0,
                                "type":"dynamic"}}"#,
        )
        .unwrap()
    }

    #[test]
    fn ntk_alpha_frequencies_match_transformers() {
        let cfg = hy_cfg();
        let rope = RotaryEmb::new(&cfg, DType::F32, &Device::Cpu);
        assert!(
            (rope.base() - 1.1158839925e7).abs() / 1.1158839925e7 < 1e-9,
            "base = {}",
            rope.base()
        );
        // Reference values computed with the transformers formula.
        let inv = rope.inv_freq(1);
        assert!((inv[0] - 1.0).abs() < 1e-6);
        assert!((inv[1] - 0.77603436305).abs() < 1e-6, "inv[1] = {}", inv[1]);
        assert!((inv[2] - 0.60222933263).abs() < 1e-6, "inv[2] = {}", inv[2]);
        assert!(
            (inv[63] - 1.1547819847e-07).abs() / 1.1547819847e-07 < 1e-5,
            "inv[63] = {}",
            inv[63]
        );
    }

    #[test]
    fn cos_sin_table_matches_frequency() {
        let cfg = hy_cfg();
        let mut rope = RotaryEmb::new(&cfg, DType::F32, &Device::Cpu);
        let (cos, sin) = rope.cos_sin(0, 2).unwrap();
        let c1 = cos
            .narrow(0, 1, 1)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let s1 = sin
            .narrow(0, 1, 1)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        // Position 1, pair 1 -> angle = inv_freq[1] ≈ 0.7760344
        let angle = 0.77603436305f32;
        assert!((c1[1] - angle.cos()).abs() < 1e-6);
        assert!((s1[1] - angle.sin()).abs() < 1e-6);
    }

    #[test]
    fn rotates_pairs_by_position() {
        let cfg = cfg();
        let dev = Device::Cpu;
        let mut rope = RotaryEmb::new(&cfg, DType::F32, &dev);
        let (cos, sin) = rope.cos_sin(0, 3).unwrap();
        assert_eq!(cos.dims(), &[3, 8]);
        // At position 0 the angle is 0: cos = 1, sin = 0.
        let c0 = cos.narrow(0, 0, 1).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let s0 = sin.narrow(0, 0, 1).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(c0.iter().all(|v| (*v - 1.0).abs() < 1e-6));
        assert!(s0.iter().all(|v| v.abs() < 1e-6));

        // Rotating a constant vector must preserve its norm.
        let x = Tensor::ones((1, 2, 3, 16), DType::F32, &dev).unwrap();
        let y = rope.apply(&x, &cos, &sin).unwrap();
        let norm_x = x.sqr().unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap();
        let norm_y = y.sqr().unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap();
        assert!((norm_x - norm_y).abs() < 1e-3);
    }

    #[test]
    fn table_grows() {
        let cfg = cfg();
        let dev = Device::Cpu;
        let mut rope = RotaryEmb::new(&cfg, DType::F32, &dev);
        rope.cos_sin(0, 10).unwrap();
        let first = rope.cached_positions();
        rope.cos_sin(0, 1000).unwrap();
        assert!(rope.cached_positions() > first);
        assert!(rope.cached_positions() >= 1000);
    }
}
