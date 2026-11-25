//! Model configuration (`config.json`) and generation defaults
//! (`generation_config.json`).
//!
//! The Hy-MT3 1.8B / 7B checkpoints use the `hunyuan_v1_dense` architecture:
//! a Llama-style decoder with QK-RMSNorm, GQA and tied embeddings.

use std::path::Path;

use serde::Deserialize;

use crate::error::{Error, Result};

fn default_eps() -> f64 {
    1e-5
}

fn default_theta() -> f64 {
    10_000.0
}

fn default_max_pos() -> usize {
    32_768
}

fn default_true() -> bool {
    true
}

fn default_act() -> String {
    "silu".to_string()
}

/// `eos_token_id` is a scalar in some checkpoints and a list in others
/// (e.g. HY-MT1.5-7B ships `[127960, 127967]`).
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum EosIds {
    /// Single end-of-sequence token.
    One(u32),
    /// Several end-of-sequence tokens.
    Many(Vec<u32>),
}

impl EosIds {
    /// The ids as a vector.
    pub fn ids(&self) -> Vec<u32> {
        match self {
            Self::One(id) => vec![*id],
            Self::Many(ids) => ids.clone(),
        }
    }
}

impl From<u32> for EosIds {
    fn from(id: u32) -> Self {
        Self::One(id)
    }
}

/// `rope_scaling` (a.k.a. `rope_parameters` in transformers >= 5) block of
/// `config.json`.
///
/// Hy-MT3 ships `{"type": "dynamic", "factor": 1.0, "alpha": 1000, ...}`.
///
/// transformers 5.6 renames `type` to `rope_type` and, in
/// `HunYuanDenseV1RotaryEmbedding`, maps `rope_type == "dynamic"` **with a
/// fixed `alpha`** to a separate `"ntk_alpha"` mode: the base frequency is
/// stretched once and stays static (no dependence on the sequence length):
///
/// ```text
/// base = rope_theta * alpha ** (head_dim / (head_dim - 2))
/// ```
///
/// With the shipped values this gives `base ≈ 1.11588e7` instead of `1e4`,
/// which is exactly what the reference implementation uses.
#[derive(Debug, Clone, Deserialize, Default)]
pub struct RopeScaling {
    #[serde(rename = "type", default)]
    pub rope_type: Option<String>,
    #[serde(default)]
    pub factor: f64,
    #[serde(default)]
    pub alpha: f64,
    #[serde(default)]
    pub beta_fast: f64,
    #[serde(default)]
    pub beta_slow: f64,
    #[serde(default)]
    pub mscale: f64,
    #[serde(default)]
    pub mscale_all_dim: f64,
}

impl RopeScaling {
    /// `rope_type` with the `"default"` fallback used by transformers.
    pub fn rope_type(&self) -> &str {
        self.rope_type.as_deref().unwrap_or("default")
    }

    /// `factor` normalised to at least 1.0.
    pub fn factor(&self) -> f64 {
        if self.factor <= 0.0 {
            1.0
        } else {
            self.factor
        }
    }

    /// `alpha` when the checkpoint provides a usable value.
    pub fn alpha(&self) -> Option<f64> {
        if self.alpha > 0.0 {
            Some(self.alpha)
        } else {
            None
        }
    }

    /// Static NTK-aware stretch (`"dynamic"` + fixed `alpha`), the mode used by
    /// every Hy-MT3 checkpoint.
    pub fn is_ntk_alpha(&self) -> bool {
        self.rope_type() == "dynamic" && self.alpha().is_some()
    }

    /// Length dependent dynamic-NTK scaling (no `alpha`).
    pub fn is_dynamic_ntk(&self) -> bool {
        self.rope_type() == "dynamic" && self.alpha().is_none()
    }
}

/// Parsed `config.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    #[serde(default)]
    pub num_key_value_heads: usize,
    #[serde(default)]
    pub head_dim: Option<usize>,
    #[serde(default)]
    pub attention_head_dim: Option<usize>,
    #[serde(default = "default_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_theta")]
    pub rope_theta: f64,
    #[serde(default)]
    pub rope_scaling: Option<RopeScaling>,
    #[serde(default = "default_max_pos")]
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub mlp_bias: bool,
    #[serde(default = "default_true")]
    pub use_qk_norm: bool,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default = "default_act")]
    pub hidden_act: String,
    #[serde(default)]
    pub model_type: Option<String>,
    #[serde(default)]
    pub architectures: Vec<String>,
    #[serde(default)]
    pub bos_token_id: Option<u32>,
    #[serde(default)]
    pub eos_token_id: Option<EosIds>,
    #[serde(default)]
    pub pad_token_id: Option<u32>,
    /// `dtype` of the fp8 checkpoint ("bfloat16"); older checkpoints spell the
    /// same field `torch_dtype`.
    #[serde(default, alias = "torch_dtype")]
    pub dtype: Option<String>,
    /// Cross-layer attention. Every shipped checkpoint sets this to `false`;
    /// when `true` the K/V of some layers are shared and the architecture
    /// differs, so it is rejected explicitly instead of being ignored.
    #[serde(default)]
    pub use_cla: bool,
}

impl Config {
    /// Reads `config.json` from a model directory.
    pub fn from_dir(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();
        let path = dir.join("config.json");
        if !path.is_file() {
            return Err(Error::InvalidModelDir {
                dir: dir.display().to_string(),
                reason: "config.json not found".into(),
            });
        }
        let raw = std::fs::read_to_string(&path)?;
        let cfg: Config = serde_json::from_str(&raw)?;
        cfg.validate()?;
        Ok(cfg)
    }

    /// Rejects checkpoint families this implementation does not support.
    pub fn validate(&self) -> Result<()> {
        // `load_model` dispatches on the architecture and produces a precise
        // error for known-but-unimplemented families (e.g. the MoE 30B-A3B), so
        // only reject clearly foreign configs here.
        if let Some(mt) = &self.model_type {
            let known = [
                "hunyuan_v1_dense",
                "hunyuan_dense_v1",
                "hunyuan_v1_moe",
                "llama",
            ];
            if !known.contains(&mt.as_str()) {
                return Err(Error::unsupported(format!(
                    "model_type '{mt}' is not supported (expected 'hunyuan_v1_dense')"
                )));
            }
        }
        let act = self.hidden_act.to_ascii_lowercase();
        if act != "silu" && act != "swish" {
            return Err(Error::unsupported(format!(
                "hidden_act '{act}' is not supported (only silu/swish)"
            )));
        }
        if self.num_hidden_layers == 0 || self.num_attention_heads == 0 {
            return Err(Error::unsupported("degenerate config: no layers/heads"));
        }
        if self.head_dim() % 2 != 0 {
            return Err(Error::unsupported("head_dim must be even for RoPE"));
        }
        if self.use_cla {
            return Err(Error::unsupported(
                "this checkpoint enables cross-layer attention (use_cla = true), \
                 whose K/V layout is not implemented yet",
            ));
        }
        Ok(())
    }

    /// Dimension of a single attention head (128 for Hy-MT3).
    pub fn head_dim(&self) -> usize {
        self.head_dim
            .or(self.attention_head_dim)
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    /// Number of key/value heads (GQA).
    pub fn num_key_value_heads(&self) -> usize {
        if self.num_key_value_heads == 0 {
            self.num_attention_heads
        } else {
            self.num_key_value_heads
        }
    }

    /// Number of query heads that share one key/value head.
    pub fn num_key_value_groups(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads()
    }

    /// `q_proj` output dimension.
    pub fn q_dim(&self) -> usize {
        self.num_attention_heads * self.head_dim()
    }

    /// `k_proj` / `v_proj` output dimension.
    pub fn kv_dim(&self) -> usize {
        self.num_key_value_heads() * self.head_dim()
    }

    /// RoPE base frequency, following `HunYuanDenseV1RotaryEmbedding`.
    ///
    /// * `ntk_alpha` (the Hy-MT3 case): `theta * alpha ** (head_dim / (head_dim - 2))`
    /// * otherwise: plain `theta` (dynamic-NTK is applied later, per sequence
    ///   length, in [`crate::rope::RotaryEmb`]).
    pub fn rope_base(&self) -> f64 {
        match &self.rope_scaling {
            Some(rs) if rs.is_ntk_alpha() => {
                let dim = self.head_dim() as f64;
                self.rope_theta * rs.alpha().unwrap_or(1.0).powf(dim / (dim - 2.0))
            }
            _ => self.rope_theta,
        }
    }

    /// Whether the `lm_head` is the transposed embedding matrix.
    pub fn tied_embeddings(&self) -> bool {
        self.tie_word_embeddings
    }

    /// Every end-of-sequence id declared by the checkpoint.
    pub fn eos_ids(&self) -> Vec<u32> {
        self.eos_token_id
            .as_ref()
            .map(|ids| ids.ids())
            .unwrap_or_default()
    }
}

/// The type name used by `hy-mt` 0.1 (`HunYuanDenseV1ForCausalLM`).
pub type HunYuanDenseV1Config = Config;

/// Sampling defaults shipped with the checkpoint (`generation_config.json`).
///
/// Every field is optional so that a missing file simply keeps the library
/// defaults (the ones recommended in the model card).
#[derive(Debug, Clone, Default, Deserialize)]
pub struct GenerationDefaults {
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub top_p: Option<f64>,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub repetition_penalty: Option<f32>,
    #[serde(default)]
    pub eos_token_id: Option<EosIds>,
    #[serde(default)]
    pub bos_token_id: Option<u32>,
    #[serde(default)]
    pub pad_token_id: Option<u32>,
}

impl GenerationDefaults {
    /// Every end-of-sequence id declared in `generation_config.json`.
    pub fn eos_ids(&self) -> Vec<u32> {
        self.eos_token_id
            .as_ref()
            .map(|ids| ids.ids())
            .unwrap_or_default()
    }
}

impl GenerationDefaults {
    /// Loads `generation_config.json`, returning `Default` when absent.
    pub fn from_dir(dir: impl AsRef<Path>) -> Result<Self> {
        let path = dir.as_ref().join("generation_config.json");
        if !path.is_file() {
            return Ok(Self::default());
        }
        let raw = std::fs::read_to_string(&path)?;
        Ok(serde_json::from_str(&raw)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RAW: &str = r#"{
        "vocab_size": 120818,
        "hidden_size": 2048,
        "intermediate_size": 6144,
        "num_hidden_layers": 32,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "rms_norm_eps": 1e-05,
        "rope_theta": 10000.0,
        "rope_scaling": {"alpha": 1000.0, "beta_fast": 32, "beta_slow": 1,
                         "factor": 1.0, "mscale": 1.0, "mscale_all_dim": 1.0,
                         "type": "dynamic"},
        "max_position_embeddings": 262144,
        "tie_word_embeddings": true,
        "use_qk_norm": true,
        "hidden_act": "silu",
        "model_type": "hunyuan_v1_dense",
        "eos_token_id": 120020,
        "bos_token_id": 120000
    }"#;

    #[test]
    fn accepts_torch_dtype_alias_and_eos_list() {
        // HY-MT1.5-7B / Hunyuan-MT-7B style config.
        let cfg: Config = serde_json::from_str(
            r#"{"vocab_size":128167,"hidden_size":4096,"intermediate_size":14336,
                "num_hidden_layers":32,"num_attention_heads":32,"num_key_value_heads":8,
                "head_dim":128,"torch_dtype":"bfloat16","tie_word_embeddings":true,
                "eos_token_id":[127960,127967],"max_position_embeddings":262144,
                "rope_theta":10000.0,
                "rope_scaling":{"alpha":1000.0,"factor":1.0,"mscale":1.0,"type":"dynamic"}}"#,
        )
        .unwrap();
        assert_eq!(cfg.dtype.as_deref(), Some("bfloat16"));
        assert_eq!(cfg.eos_ids(), vec![127960, 127967]);
        assert!(cfg.rope_base() > 1.0e7);
        cfg.validate().unwrap();
    }

    #[test]
    fn rejects_cla_checkpoints() {
        let json = RAW.replace("\"tie_word_embeddings\": true", "\"tie_word_embeddings\": true, \"use_cla\": true");
        let cfg: Config = serde_json::from_str(&json).unwrap();
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn parses_hy_mt_config() {
        let cfg: Config = serde_json::from_str(RAW).unwrap();
        cfg.validate().unwrap();
        assert_eq!(cfg.head_dim(), 128);
        assert_eq!(cfg.q_dim(), 2048);
        assert_eq!(cfg.kv_dim(), 512);
        assert_eq!(cfg.num_key_value_groups(), 4);
        assert!(cfg.tied_embeddings());
        assert_eq!(cfg.eos_ids(), vec![120020]);
        let rs = cfg.rope_scaling.as_ref().unwrap();
        assert!(rs.is_ntk_alpha(), "dynamic + alpha must select ntk_alpha");
        assert!(!rs.is_dynamic_ntk());
        // Reference (transformers 5.6) computes 1e4 * 1000 ** (128 / 126).
        let expected = 1.1158839925e7;
        assert!(
            (cfg.rope_base() - expected).abs() / expected < 1e-9,
            "unexpected rope base: {}",
            cfg.rope_base()
        );
    }
}
