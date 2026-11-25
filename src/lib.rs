//! # hy-mt
//!
//! Rust inference for Tencent Hunyuan's **MT translation model family**, built
//! on [candle](https://github.com/huggingface/candle).
//!
//! One engine covers every dense release, because they all use the same
//! architecture (`hunyuan_v1_dense`):
//!
//! | checkpoint | arch | vocab / chat template | weights |
//! |---|---|---|---|
//! | `HY-MT1.5-1.8B`, `HY-MT1.5-1.8B-FP8` | dense | 120 818 / `Hy18B` | bf16, FP8 |
//! | `Hy-MT2-1.8B`, `Hy-MT2-1.8B-FP8` | dense | 120 818 / `Hy18B` | bf16, FP8 |
//! | `Hy-MT3-1.8B`, `Hy-MT3-1.8B-FP8` | dense | 120 818 / `Hy18B` | bf16, FP8 |
//! | `Hunyuan-MT-7B`, `HY-MT1.5-7B`, `Hy-MT3-7B` (+FP8) | dense | 128k / `Hunyuan7B` | bf16, FP8 |
//! | `Hy-MT3-30B-A3B` (+FP8) | MoE | 128k / `Hunyuan7B` | not implemented yet |
//!
//! What changes between generations is the *checkpoint* (sizes, vocabulary,
//! quantization), never the code path — `config.json` drives everything.
//!
//! ## Two APIs, one engine
//!
//! [`Translator`] is the high-level translation API (prompt templates from the
//! model card, streaming, terminology/style/structured-data constraints):
//!
//! ```no_run
//! use hy_mt::{TranslationRequest, Translator};
//!
//! # fn main() -> hy_mt::Result<()> {
//! let mut translator = Translator::from_pretrained("/models/Hy-MT3-1.8B-FP8")?;
//! println!("{}", translator.translate("今天天气真好。", "en")?.text);
//!
//! let req = TranslationRequest::new("Hunyuan is a translation model.", "zh")
//!     .term("Hunyuan", "混元");
//! println!("{}", translator.translate_with(&req, None)?.text);
//! # Ok(())
//! # }
//! ```
//!
//! [`TextGeneration`] is the original `hy-mt` API, kept source compatible, for
//! raw prompt → text generation:
//!
//! ```no_run
//! use hy_mt::{Device, TextGeneration};
//!
//! # fn main() -> hy_mt::Result<()> {
//! let mut model = TextGeneration::from_pretrained("/models/HY-MT1.5-1.8B", &Device::Cpu)?;
//! let answer = model.generate("将以下文本翻译为英语：你好，世界。", 256)?;
//! println!("{answer}");
//! # Ok(())
//! # }
//! ```
//!
//! For token-level control there is [`Session`] (one token at a time, own KV
//! cache) and [`generate`] (callback based).
//!
//! ## Implemented
//!
//! * FP8 (`compressed-tensors`, per-tensor `weight_scale`) dequantisation at
//!   load time, plus bf16 / f16 / f32 checkpoints and sharded safetensors.
//! * RoPE with the `ntk_alpha` base stretch used by the whole family
//!   (`base = theta * alpha ** (dim / (dim - 2))`) and dynamic-NTK fallback.
//! * GQA attention, per-head QK-RMSNorm **after** RoPE (the reference order),
//!   SwiGLU MLP, tied `lm_head`, causal masking, KV cache.
//! * Both chat template families, BPE tokenizer, incremental streaming decode,
//!   temperature / top-k / top-p / repetition-penalty sampling.
//! * The translation prompt templates documented in the model card
//!   (default, terminology, style, personalisation, delimiters, structured
//!   data, background), in Chinese and English.
//!
//! ## Backends (cargo features)
//!
//! | feature | backend | notes |
//! |---|---|---|
//! | `cpu` *(default)* | CPU | always compiled into candle; the flag makes the choice explicit |
//! | `metal` | Apple GPU | macOS only; f16 recommended |
//! | `accelerate` | CPU | Apple Accelerate BLAS |
//! | `cuda` / `cudnn` | NVIDIA GPU | `cudnn` implies `cuda` |
//! | `mkl` | CPU | Intel MKL |
//!
//! Device selection: explicit `device(..)`, then `HYMT_DEVICE=cpu|metal[:N]|cuda[:N]`,
//! then Metal, CUDA and finally the CPU. The default weight dtype follows the
//! device: f16 on Metal, bf16 on CUDA, f32 on CPU.

#![warn(missing_debug_implementations)]

pub mod cache;
pub mod chat;
pub mod config;
pub mod error;
pub mod fp8;
pub mod generation;
pub mod lang;
pub mod logits_processor;
pub mod models;
pub mod nn;
pub mod prompt;
pub mod rng;
pub mod rope;
pub mod safetensors_file;
pub mod sampling;
pub mod textgen;
pub mod tokenizer;
pub mod translator;
pub mod utils;
pub mod weights;

// candle types appear in the public API (`Device`, `DType`, `Tensor`), so they
// are re-exported: downstream code does not need its own candle dependency to
// stay version-compatible.
pub use candle_core;
pub use candle_core::{DType, Device, Tensor};
pub use candle_nn;

pub use cache::KVCache;
pub use chat::{ChatFormat, Message, Role};
pub use config::{Config, EosIds, GenerationDefaults, HunYuanDenseV1Config, RopeScaling};
pub use error::{Error, Result};
pub use generation::{generate, FinishReason, GenerateConfig, GenerationOutput, Session};
pub use lang::Language;
pub use logits_processor::{LogitsProcessor, Sampling};
pub use models::{default_device, device_from_str, load_model, CausalLm, ModelOptions};
pub use prompt::{PromptLang, Prompter, RawPrompt, TermPair, TranslationRequest};
pub use sampling::SamplingConfig;
pub use textgen::{TextGeneration, TextGenerationStream};
pub use tokenizer::{IncrementalDecoder, SpecialTokens, TokenOutputStream, Tokenizer};
pub use translator::{Translation, Translator, TranslatorBuilder};

/// Environment variable that overrides backend selection (`cpu`, `metal[:N]`,
/// `cuda[:N]`).
pub use models::DEVICE_ENV;
