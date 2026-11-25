//! The dense Hunyuan V1 decoder — used by `Hy-MT3-1.8B(-FP8)` and
//! `Hy-MT3-7B(-FP8)`.
//!
//! Layout (matches `HunYuanDenseV1ForCausalLM` in transformers >= 5.6):
//!
//! ```text
//! tokens ─> embed_tokens ─> 32 x [ RMSNorm ─> Attention ─> +residual
//!                                  RMSNorm ─> SwiGLU MLP ─> +residual ]
//!        ─> RMSNorm ─> lm_head (tied to embed_tokens)
//! ```
//!
//! The checkpoint size comes exclusively from `config.json`, so the very same
//! code path runs the 1.8B and 7B models.

mod attention;
mod mlp;

pub use attention::Attention;
pub use mlp::Mlp;

use std::path::Path;

use candle_core::{DType, Device, Result as CandleResult, Tensor};

use super::{CausalLm, ModelOptions};
use crate::cache::KVCache;
use crate::config::Config;
use crate::error::{Error, Result};
use crate::nn::{Linear, RmsNorm};
use crate::rope::RotaryEmb;
use crate::weights::WeightSource;

/// One pre-norm transformer block.
#[derive(Debug)]
pub struct DecoderLayer {
    input_layernorm: RmsNorm,
    self_attn: Attention,
    post_attention_layernorm: RmsNorm,
    mlp: Mlp,
}

impl DecoderLayer {
    fn load(
        weights: &WeightSource,
        prefix: &str,
        cfg: &Config,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        Ok(Self {
            input_layernorm: RmsNorm::new(
                weights.tensor(&format!("{prefix}.input_layernorm.weight"), dtype, device)?,
                cfg.rms_norm_eps,
            ),
            self_attn: Attention::load(weights, prefix, cfg, dtype, device)?,
            post_attention_layernorm: RmsNorm::new(
                weights.tensor(
                    &format!("{prefix}.post_attention_layernorm.weight"),
                    dtype,
                    device,
                )?,
                cfg.rms_norm_eps,
            ),
            mlp: Mlp::load(weights, prefix, cfg, dtype, device)?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        rope: &RotaryEmb,
        mask: Option<&Tensor>,
        cache: &mut KVCache,
        layer: usize,
    ) -> CandleResult<Tensor> {
        let residual = x;
        let h = self.input_layernorm.forward(x)?;
        let h = self
            .self_attn
            .forward(&h, cos, sin, rope, mask, cache, layer)?;
        let x = residual.add(&h)?;

        let residual = &x;
        let h = self.post_attention_layernorm.forward(&x)?;
        let h = self.mlp.forward(&h)?;
        residual.add(&h)
    }
}

/// Dense Hunyuan V1 causal LM.
#[derive(Debug)]
pub struct HunyuanDense {
    cfg: Config,
    embed_tokens: Tensor,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: Linear,
    rope: RotaryEmb,
    dtype: DType,
    device: Device,
}

impl HunyuanDense {
    /// Loads a dense checkpoint directory.
    pub fn load(dir: impl AsRef<Path>, cfg: Config, opts: &ModelOptions) -> Result<Self> {
        let dir = dir.as_ref();
        let dtype = opts.resolved_dtype();
        let device = opts.device.clone();
        let hidden = cfg.hidden_size;

        let log = |msg: String| {
            if opts.verbose {
                eprintln!("[hy_mt] {msg}");
            }
        };
        log(format!(
            "opening weights in {} ({dtype:?} on {:?})",
            dir.display(),
            device
        ));
        let weights = WeightSource::open(dir)?;
        log(format!("{} tensors found", weights.len()));

        let embed_name = "model.embed_tokens.weight";
        let shape = weights.shape(embed_name)?.to_vec();
        if shape != [cfg.vocab_size, hidden] {
            return Err(Error::invalid_arg(format!(
                "unexpected embedding shape {shape:?}, expected [{}, {}]",
                cfg.vocab_size, hidden
            )));
        }
        let embed_tokens = weights.tensor(embed_name, dtype, &device)?;
        log("embedding loaded".into());

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let prefix = format!("model.layers.{i}");
            layers.push(DecoderLayer::load(&weights, &prefix, &cfg, dtype, &device)?);
            log(format!("layer {}/{} loaded", i + 1, cfg.num_hidden_layers));
        }

        let norm = RmsNorm::new(
            weights.tensor("model.norm.weight", dtype, &device)?,
            cfg.rms_norm_eps,
        );

        // `lm_head` is tied to the embedding in every checkpoint of the family
        // (`_tied_weights_keys`). Note that the *bf16* releases (HY-MT1.5-1.8B,
        // ...) still ship a duplicated `lm_head.weight`, while the FP8 ones do
        // not — so trust the config and reuse the embedding, which saves ~0.5 GB
        // for the 1.8B models and also makes the missing-tensor case work.
        let build_tied = || -> Result<Linear> {
            let w = embed_tokens.t()?.contiguous()?;
            Ok(Linear::new(w, None))
        };
        let lm_head = if cfg.tie_word_embeddings {
            log("lm_head tied to the token embedding".into());
            build_tied()?
        } else if weights.contains("lm_head.weight") {
            log("using untied lm_head from the checkpoint".into());
            weights.linear("lm_head", dtype, &device, false)?
        } else {
            log("lm_head.weight absent, falling back to tied embeddings".into());
            build_tied()?
        };

        let rope = RotaryEmb::new(&cfg, dtype, &device);
        log(format!("rope base = {:.6e}", rope.base()));

        Ok(Self {
            cfg,
            embed_tokens,
            layers,
            norm,
            lm_head,
            rope,
            dtype,
            device,
        })
    }

    /// Additive causal mask of shape `(q_len, kv_len)`.
    ///
    /// `offset` is the number of tokens already in the cache, so that chunked
    /// prefill keeps correct absolute positions.
    fn causal_mask(&self, q_len: usize, kv_len: usize, offset: usize) -> Result<Tensor> {
        let mut data = Vec::with_capacity(q_len * kv_len);
        for i in 0..q_len {
            let last_visible = offset + i;
            for j in 0..kv_len {
                data.push(if j <= last_visible {
                    0.0
                } else {
                    f32::NEG_INFINITY
                });
            }
        }
        Ok(Tensor::from_vec(data, (q_len, kv_len), &self.device)?.to_dtype(self.dtype)?)
    }

    /// Number of decoder layers.
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }
}

impl CausalLm for HunyuanDense {
    fn config(&self) -> &Config {
        &self.cfg
    }

    fn dtype(&self) -> DType {
        self.dtype
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn architecture(&self) -> &'static str {
        "hunyuan_v1_dense"
    }

    fn forward(
        &mut self,
        input_ids: &Tensor,
        cache: &mut KVCache,
        last_only: bool,
    ) -> CandleResult<Tensor> {
        let (batch, seq_len) = input_ids.dims2()?;
        if batch != 1 {
            candle_core::bail!("only batch size 1 is supported, got {batch}");
        }
        let offset = cache.len();

        let ids = input_ids.flatten_all()?;
        let mut x = self
            .embed_tokens
            .index_select(&ids, 0)?
            .reshape((batch, seq_len, self.cfg.hidden_size))?;

        let (cos, sin) = self.rope.cos_sin(offset, seq_len)?;
        let mask = if seq_len > 1 {
            Some(self.causal_mask(seq_len, offset + seq_len, offset)?)
        } else {
            None
        };

        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(&x, &cos, &sin, &self.rope, mask.as_ref(), cache, i)?;
        }
        cache.advance(seq_len);

        let x = self.norm.forward(&x)?;
        let x = if last_only {
            x.narrow(1, seq_len - 1, 1)?
        } else {
            x
        };
        self.lm_head.forward(&x)
    }
}
