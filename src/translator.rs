//! High level, ready-to-use translation API.

use std::path::Path;

use candle_core::{DType, Device};

use crate::chat::{ChatFormat, Message};
use crate::config::{Config, GenerationDefaults};
use crate::error::Result;
use crate::generation::{generate, GenerateConfig, GenerationOutput, Session};
use crate::models::{default_device, load_model, CausalLm, ModelOptions};
use crate::prompt::{PromptLang, TranslationRequest};
use crate::sampling::SamplingConfig;
use crate::tokenizer::Tokenizer;

/// Fluent builder for [`Translator`].
#[derive(Debug)]
pub struct TranslatorBuilder {
    dir: std::path::PathBuf,
    device: Option<Device>,
    dtype: Option<DType>,
    verbose: bool,
    prompt_lang: Option<PromptLang>,
    use_checkpoint_generation_config: bool,
    warmup: bool,
}

impl TranslatorBuilder {
    /// Starts a builder for a model directory.
    pub fn new(dir: impl AsRef<Path>) -> Self {
        Self {
            dir: dir.as_ref().to_path_buf(),
            device: None,
            dtype: None,
            verbose: false,
            prompt_lang: None,
            use_checkpoint_generation_config: false,
            warmup: true,
        }
    }

    /// Runs on an explicit device (e.g. `Device::Cpu`).
    ///
    /// When omitted, [`crate::models::default_device`] is used: `HYMT_DEVICE`
    /// if set, else Metal / CUDA when compiled in, else the CPU.
    pub fn device(mut self, device: Device) -> Self {
        self.device = Some(device);
        self
    }

    /// Overrides the weight dtype (f32 / f16 / bf16).
    pub fn dtype(mut self, dtype: DType) -> Self {
        self.dtype = Some(dtype);
        self
    }

    /// Prints loading progress.
    pub fn verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Default instruction language for translation prompts.
    pub fn prompt_lang(mut self, lang: PromptLang) -> Self {
        self.prompt_lang = Some(lang);
        self
    }

    /// Use the sampling defaults of `generation_config.json` instead of the
    /// values recommended in the model card.
    pub fn use_checkpoint_generation_config(mut self, yes: bool) -> Self {
        self.use_checkpoint_generation_config = yes;
        self
    }

    /// Runs a tiny forward pass after loading so that the backend compiles its
    /// kernels up-front (on Metal the first call pays for shader compilation).
    /// Enabled by default.
    pub fn warmup(mut self, warmup: bool) -> Self {
        self.warmup = warmup;
        self
    }

    /// Loads the model and the tokenizer.
    pub fn load(self) -> Result<Translator> {
        let device = match self.device {
            Some(device) => device,
            None => default_device()?,
        };
        let opts = ModelOptions {
            device,
            dtype: self.dtype,
            verbose: self.verbose,
        };
        let tokenizer = Tokenizer::from_dir(&self.dir)?;
        let defaults = GenerationDefaults::from_dir(&self.dir)?;
        let model = load_model(&self.dir, &opts)?;
        let mut translator = Translator {
            model,
            tokenizer,
            defaults,
            prompt_lang: self.prompt_lang,
            use_checkpoint_generation_config: self.use_checkpoint_generation_config,
        };
        translator.sanitise_defaults();
        if self.warmup {
            translator.warm_up()?;
        }
        Ok(translator)
    }
}

/// A loaded Hy-MT3 model plus its tokenizer.
///
/// ```no_run
/// use hy_mt::Translator;
/// # fn main() -> hy_mt::Result<()> {
/// let mut translator = Translator::from_pretrained("/path/to/Hy-MT3-1.8B-FP8")?;
/// let out = translator.translate("今天天气真好。", "en")?;
/// println!("{}", out.text);
/// # Ok(())
/// # }
/// ```
pub struct Translator {
    model: Box<dyn CausalLm>,
    tokenizer: Tokenizer,
    defaults: GenerationDefaults,
    prompt_lang: Option<PromptLang>,
    use_checkpoint_generation_config: bool,
}

impl std::fmt::Debug for Translator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Translator")
            .field("architecture", &self.model.architecture())
            .field("dtype", &self.model.dtype())
            .field("device", self.model.device())
            .finish()
    }
}

/// Result of a translation.
#[derive(Debug, Clone)]
pub struct Translation {
    /// The translated text (trimmed).
    pub text: String,
    /// Target language as given by the caller.
    pub target_lang: String,
    /// The exact prompt that was sent to the model.
    pub prompt: String,
    /// Raw generation details (tokens, timing, ...).
    pub output: GenerationOutput,
}

impl Translation {
    /// Decode throughput in tokens/second.
    pub fn tokens_per_second(&self) -> f64 {
        self.output.tokens_per_second()
    }
}

impl Translator {
    /// Loads a model with default options (Metal when compiled in and
    /// available, f16 weights there, f32 on CPU).
    pub fn from_pretrained(dir: impl AsRef<Path>) -> Result<Self> {
        Self::builder(dir).load()
    }

    /// Starts a configurable load.
    pub fn builder(dir: impl AsRef<Path>) -> TranslatorBuilder {
        TranslatorBuilder::new(dir)
    }

    /// Parsed `config.json`.
    pub fn config(&self) -> &Config {
        self.model.config()
    }

    /// Borrow the underlying model (for custom generation loops).
    pub fn model(&self) -> &dyn CausalLm {
        self.model.as_ref()
    }

    /// Mutably borrow the underlying model.
    pub fn model_mut(&mut self) -> &mut dyn CausalLm {
        self.model.as_mut()
    }

    /// The tokenizer in use.
    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// Architecture name of the loaded model.
    pub fn architecture(&self) -> &'static str {
        self.model.architecture()
    }

    /// Dtype of the loaded weights.
    pub fn dtype(&self) -> DType {
        self.model.dtype()
    }

    /// Device the model runs on.
    pub fn device(&self) -> &Device {
        self.model.device()
    }

    /// The checkpoint's `generation_config.json` (may be all `None`).
    pub fn checkpoint_generation_defaults(&self) -> &GenerationDefaults {
        &self.defaults
    }

    /// Sets the default instruction language (`None` = automatic).
    pub fn set_prompt_lang(&mut self, lang: Option<PromptLang>) {
        self.prompt_lang = lang;
    }

    /// Whether `generation_config.json` overrides the model card defaults.
    pub fn set_use_checkpoint_generation_config(&mut self, yes: bool) {
        self.use_checkpoint_generation_config = yes;
    }

    fn sanitise_defaults(&mut self) {
        // Guard against nonsensical values in a hand edited config file.
        if let Some(t) = self.defaults.temperature {
            if !t.is_finite() || t < 0.0 {
                self.defaults.temperature = None;
            }
        }
        if let Some(p) = self.defaults.top_p {
            if !(p.is_finite() && p > 0.0 && p <= 1.0) {
                self.defaults.top_p = None;
            }
        }
    }

    /// Generation parameters used when the caller does not supply any.
    ///
    /// Defaults to the model card recommendation
    /// (`temperature = 0.7, top_p = 0.6, top_k = 20, repetition_penalty = 1.05`)
    /// unless [`TranslatorBuilder::use_checkpoint_generation_config`] is set, in
    /// which case `generation_config.json` wins.
    pub fn default_generate_config(&self) -> GenerateConfig {
        let mut sampling = SamplingConfig::default();
        if self.use_checkpoint_generation_config {
            if let Some(v) = self.defaults.temperature {
                sampling.temperature = v;
            }
            if let Some(v) = self.defaults.top_p {
                sampling.top_p = v;
            }
            if let Some(v) = self.defaults.top_k {
                sampling.top_k = v;
            }
            if let Some(v) = self.defaults.repetition_penalty {
                sampling.repetition_penalty = v;
            }
        }
        GenerateConfig {
            sampling,
            max_new_tokens: 4096.min(self.config().max_position_embeddings),
            stop_token_ids: Vec::new(),
        }
    }

    /// Generation parameters used by the translation helpers.
    ///
    /// Translation is a deterministic task, so this is **greedy** decoding:
    /// it is reproducible and, on the short sentences typical of translation,
    /// also the highest quality setting (the model card's sampling parameters
    /// occasionally pick a worse continuation, see `examples/try_prompts.rs`).
    /// Use [`Translator::default_generate_config`] or an explicit
    /// [`GenerateConfig`] to sample instead.
    pub fn default_translate_config(&self) -> GenerateConfig {
        GenerateConfig {
            sampling: SamplingConfig::greedy(),
            max_new_tokens: 1024.min(self.config().max_position_embeddings),
            stop_token_ids: Vec::new(),
        }
    }

    /// Compiles the backend kernels by running one tiny generation.
    fn warm_up(&mut self) -> Result<()> {
        let bos = self
            .tokenizer
            .special()
            .bos
            .or(self.config().bos_token_id)
            .unwrap_or(0);
        let ids = vec![bos; 8];
        let cfg = GenerateConfig {
            sampling: SamplingConfig::greedy(),
            max_new_tokens: 1,
            stop_token_ids: Vec::new(),
        };
        generate(self.model.as_mut(), &self.tokenizer, &ids, &cfg, |_| {})?;
        Ok(())
    }

    /// Chat template family detected from the tokenizer.
    pub fn chat_format(&self) -> Option<ChatFormat> {
        self.tokenizer.chat_format()
    }

    /// Starts an incremental generation session over raw prompt ids.
    ///
    /// The session owns its KV cache, so callers can decode token by token and
    /// keep the model borrowed for as long as they need.
    pub fn start_session<'a>(
        &'a mut self,
        prompt_ids: &[u32],
        cfg: &GenerateConfig,
    ) -> Result<Session<'a>> {
        Session::new(self.model.as_mut(), &self.tokenizer, prompt_ids, cfg)
    }

    /// Encodes `messages` with the checkpoint's own chat template.
    pub fn render_chat(&self, messages: &[Message], add_generation_prompt: bool) -> Result<String> {
        let format = self.tokenizer.require_chat_format()?;
        format.render(messages, add_generation_prompt)
    }

    /// Encodes `messages` into token ids using the checkpoint's chat template.
    pub fn encode_chat(&self, messages: &[Message]) -> Result<Vec<u32>> {
        let prompt = self.render_chat(messages, true)?;
        self.tokenizer.encode_with_markers(&prompt)
    }

    /// Starts an incremental session from chat messages.
    pub fn chat_session<'a>(
        &'a mut self,
        messages: &[Message],
        cfg: &GenerateConfig,
    ) -> Result<Session<'a>> {
        let ids = self.encode_chat(messages)?;
        self.start_session(&ids, cfg)
    }

    /// Generates from raw prompt ids (no chat template applied).
    pub fn generate_ids(
        &mut self,
        prompt_ids: &[u32],
        cfg: &GenerateConfig,
    ) -> Result<GenerationOutput> {
        generate(self.model.as_mut(), &self.tokenizer, prompt_ids, cfg, |_| {})
    }

    /// Streaming variant of [`Translator::generate_ids`].
    pub fn generate_ids_stream<F>(
        &mut self,
        prompt_ids: &[u32],
        cfg: &GenerateConfig,
        on_delta: F,
    ) -> Result<GenerationOutput>
    where
        F: FnMut(&str),
    {
        generate(self.model.as_mut(), &self.tokenizer, prompt_ids, cfg, on_delta)
    }

    /// Runs a chat conversation and returns the assistant answer.
    pub fn chat(&mut self, messages: &[Message], cfg: &GenerateConfig) -> Result<GenerationOutput> {
        self.chat_stream(messages, cfg, |_| {})
    }

    /// Streaming variant of [`Translator::chat`].
    pub fn chat_stream<F>(
        &mut self,
        messages: &[Message],
        cfg: &GenerateConfig,
        on_delta: F,
    ) -> Result<GenerationOutput>
    where
        F: FnMut(&str),
    {
        let prompt = self.render_chat(messages, true)?;
        let ids = self.tokenizer.encode_with_markers(&prompt)?;
        generate(self.model.as_mut(), &self.tokenizer, &ids, cfg, on_delta)
    }

    /// Translates `text` into `target_lang` with the default parameters.
    pub fn translate(&mut self, text: &str, target_lang: &str) -> Result<Translation> {
        let req = TranslationRequest::new(text, target_lang);
        self.translate_with(&req, None)
    }

    /// Translates with an explicit request and optional generation config.
    pub fn translate_with(
        &mut self,
        req: &TranslationRequest,
        cfg: Option<&GenerateConfig>,
    ) -> Result<Translation> {
        self.translate_stream(req, cfg, |_| {})
    }

    /// Translates several requests one after another (batch size 1 internally).
    pub fn translate_batch(
        &mut self,
        reqs: &[TranslationRequest],
        cfg: Option<&GenerateConfig>,
    ) -> Result<Vec<Translation>> {
        reqs.iter().map(|req| self.translate_with(req, cfg)).collect()
    }

    /// Streaming translation: `on_delta` receives each decoded increment.
    pub fn translate_stream<F>(
        &mut self,
        req: &TranslationRequest,
        cfg: Option<&GenerateConfig>,
        on_delta: F,
    ) -> Result<Translation>
    where
        F: FnMut(&str),
    {
        let mut req = req.clone();
        if req.prompt_lang.is_none() {
            req.prompt_lang = self.prompt_lang;
        }
        let prompt = req.build_prompt();
        let messages = match &req.system {
            Some(system) => vec![Message::system(system.clone()), Message::user(prompt.clone())],
            None => vec![Message::user(prompt.clone())],
        };

        let mut cfg = cfg
            .cloned()
            .unwrap_or_else(|| self.default_translate_config());
        if let Some(n) = req.max_new_tokens {
            cfg.max_new_tokens = n;
        }
        if let Some(t) = req.temperature {
            cfg.sampling.temperature = t;
        }

        let mut output = self.chat_stream(&messages, &cfg, on_delta)?;
        output.text = clean_translation(&output.text);
        Ok(Translation {
            text: output.text.clone(),
            target_lang: req.target_lang.clone(),
            prompt,
            output,
        })
    }
}

/// Removes the padding models occasionally emit around an answer.
fn clean_translation(raw: &str) -> String {
    let mut text = raw.trim().to_string();
    // Some outputs wrap the answer in straight or curly double quotes when the
    // source was a single sentence; unwrap only when the quotes are balanced.
    for (open, close) in [('"', '"'), ('“', '”'), ('「', '」')] {
        if text.starts_with(open) && text.ends_with(close) && text.chars().count() > 2 {
            let inner: String = text.chars().skip(1).take(text.chars().count() - 2).collect();
            if !inner.contains(open) {
                text = inner.trim().to_string();
            }
            break;
        }
    }
    text
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_wrapping_quotes() {
        assert_eq!(clean_translation("  \"你好\"  "), "你好");
        assert_eq!(clean_translation("“你好”"), "你好");
        assert_eq!(clean_translation("他说\"你好\""), "他说\"你好\"");
        assert_eq!(clean_translation("\n"), "");
    }
}
