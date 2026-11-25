//! The original, minimal `hy-mt` generation API.
//!
//! ```no_run
//! use hy_mt::{Device, TextGeneration};
//!
//! # fn main() -> hy_mt::Result<()> {
//! let mut model = TextGeneration::from_pretrained("/models/HY-MT1.5-1.8B", &Device::Cpu)?;
//!
//! // one shot
//! println!("{}", model.generate("将以下文本翻译为英语：你好。", 256)?);
//!
//! // streaming
//! for delta in model.generate_stream_iter("将以下文本翻译为英语：你好。", 256) {
//!     print!("{delta}");
//! }
//! # Ok(())
//! # }
//! ```
//!
//! Differences to `hy-mt` 0.1 (all of them bug fixes, see `MERGE.md`):
//!
//! * the prompt is terminated with the assistant marker instead of
//!   `<｜hy_place▁holder▁no▁8｜>` ("no generation prompt");
//! * no 512-token context window, no window-relative position ids;
//! * a KV cache instead of re-running the whole prefix for every token;
//! * end-of-turn ids come from the checkpoint, not a hard-coded `120020`;
//! * nothing is printed to stdout;
//! * `max_length` is the number of **new** tokens (kept the name for source
//!   compatibility).
//!
//! Only the `Message`/`&str`/`String` prompt shapes are provided here; the
//! [`Prompter`] trait itself lives in [`crate::prompt`].

use candle_core::Device;

use crate::chat::Message;
use crate::error::Result;
use crate::generation::{GenerateConfig, Session};
use crate::models::CausalLm;
use crate::prompt::Prompter;
use crate::tokenizer::Tokenizer;
use crate::translator::Translator;

/// `TextGeneration`-compatible facade over [`Translator`].
#[derive(Debug)]
pub struct TextGeneration {
    translator: Translator,
    cfg: GenerateConfig,
}

impl TextGeneration {
    /// Loads a checkpoint from `path` on `device`.
    pub fn from_pretrained(path: impl AsRef<std::path::Path>, device: &Device) -> Result<Self> {
        let translator = Translator::builder(path).device(device.clone()).load()?;
        let cfg = translator.default_translate_config();
        Ok(Self { translator, cfg })
    }

    /// Wraps an already loaded [`Translator`] (avoids a second model load).
    pub fn from_translator(translator: Translator) -> Self {
        let cfg = translator.default_translate_config();
        Self { translator, cfg }
    }

    /// Overrides the generation parameters used by [`TextGeneration::generate`].
    pub fn set_generate_config(&mut self, cfg: GenerateConfig) {
        self.cfg = cfg;
    }

    /// Current generation parameters.
    pub fn generate_config(&self) -> &GenerateConfig {
        &self.cfg
    }

    /// Generates up to `max_length` **new** tokens and returns the answer.
    pub fn generate(&mut self, prompt: impl Prompter, max_length: usize) -> Result<String> {
        let mut cfg = self.cfg.clone();
        cfg.max_new_tokens = max_length;
        let text = prompt.to_prompt();
        let ids = self.translator.tokenizer().encode_with_markers(&text)?;
        let mut session = self.translator.start_session(&ids, &cfg)?;
        session.run(|_| {})?;
        Ok(session.into_output()?.text)
    }

    /// Streaming variant: yields text deltas as they are produced.
    pub fn generate_stream_iter(
        &mut self,
        prompt: impl Prompter,
        max_length: usize,
    ) -> TextGenerationStream<'_> {
        let mut cfg = self.cfg.clone();
        cfg.max_new_tokens = max_length;
        let text = prompt.to_prompt();
        let built = self
            .translator
            .tokenizer()
            .encode_with_markers(&text)
            .and_then(|ids| self.translator.start_session(&ids, &cfg));
        TextGenerationStream {
            session: built.ok(),
        }
    }

    /// Chat with a full message list (system / multi-turn supported).
    pub fn chat(&mut self, messages: &[Message], max_length: usize) -> Result<String> {
        let mut cfg = self.cfg.clone();
        cfg.max_new_tokens = max_length;
        let out = self.translator.chat(messages, &cfg)?;
        Ok(out.text)
    }

    /// The underlying model.
    pub fn model(&self) -> &dyn CausalLm {
        self.translator.model()
    }

    /// The tokenizer in use.
    pub fn tokenizer(&self) -> &Tokenizer {
        self.translator.tokenizer()
    }

    /// Device the model runs on.
    pub fn device(&self) -> &Device {
        self.translator.device()
    }

    /// Escape hatch to the full API.
    pub fn translator(&self) -> &Translator {
        &self.translator
    }

    /// Mutable escape hatch to the full API.
    pub fn translator_mut(&mut self) -> &mut Translator {
        &mut self.translator
    }
}

/// Iterator over streamed text deltas.
#[derive(Debug)]
pub struct TextGenerationStream<'a> {
    session: Option<Session<'a>>,
}

impl TextGenerationStream<'_> {
    /// Fallible variant of [`Iterator::next`].
    pub fn try_next(&mut self) -> Result<Option<String>> {
        let Some(session) = self.session.as_mut() else {
            return Ok(None);
        };
        let delta = session.next_delta()?;
        if delta.is_none() {
            self.session = None;
        }
        Ok(delta)
    }
}

impl Iterator for TextGenerationStream<'_> {
    type Item = String;

    fn next(&mut self) -> Option<String> {
        // The old API could not report errors; use `try_next` to see them.
        match self.try_next() {
            Ok(delta) => delta,
            Err(_) => {
                self.session = None;
                None
            }
        }
    }
}
