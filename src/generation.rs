//! Autoregressive generation: an incremental [`Session`] plus the
//! [`generate`] convenience wrapper.

use std::time::{Duration, Instant};

use candle_core::{DType, Result as CandleResult, Tensor};

use crate::cache::KVCache;
use crate::error::{Error, Result};
use crate::models::CausalLm;
use crate::sampling::{Sampler, SamplingConfig};
use crate::tokenizer::{IncrementalDecoder, Tokenizer};

/// Why generation stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    /// The model emitted one of its end-of-turn tokens.
    StopToken,
    /// `max_new_tokens` was reached.
    MaxTokens,
}

impl std::fmt::Display for FinishReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::StopToken => write!(f, "stop_token"),
            Self::MaxTokens => write!(f, "max_tokens"),
        }
    }
}

/// Generation parameters.
#[derive(Debug, Clone)]
pub struct GenerateConfig {
    /// Sampling hyper-parameters.
    pub sampling: SamplingConfig,
    /// Hard limit on the number of new tokens.
    pub max_new_tokens: usize,
    /// Extra stop tokens (on top of the model's own EOS ids).
    pub stop_token_ids: Vec<u32>,
}

impl Default for GenerateConfig {
    fn default() -> Self {
        Self {
            sampling: SamplingConfig::default(),
            max_new_tokens: 1024,
            stop_token_ids: Vec::new(),
        }
    }
}

impl GenerateConfig {
    /// Deterministic greedy decoding.
    pub fn greedy() -> Self {
        Self {
            sampling: SamplingConfig::greedy(),
            ..Default::default()
        }
    }

    /// Sets the maximum number of new tokens.
    pub fn max_new_tokens(mut self, n: usize) -> Self {
        self.max_new_tokens = n;
        self
    }

    /// Sets the temperature (`0` = greedy).
    pub fn temperature(mut self, temperature: f64) -> Self {
        self.sampling.temperature = temperature;
        self
    }

    /// Sets top-p.
    pub fn top_p(mut self, top_p: f64) -> Self {
        self.sampling.top_p = top_p;
        self
    }

    /// Sets top-k.
    pub fn top_k(mut self, top_k: usize) -> Self {
        self.sampling.top_k = top_k;
        self
    }

    /// Sets the repetition penalty.
    pub fn repetition_penalty(mut self, penalty: f32) -> Self {
        self.sampling.repetition_penalty = penalty;
        self
    }

    /// Fixes the RNG seed for reproducible output.
    pub fn seed(mut self, seed: u64) -> Self {
        self.sampling.seed = Some(seed);
        self
    }
}

/// Result of one generation.
#[derive(Debug, Clone)]
pub struct GenerationOutput {
    /// Generated token ids (stop token excluded).
    pub token_ids: Vec<u32>,
    /// Decoded text.
    pub text: String,
    /// Number of prompt tokens.
    pub prompt_tokens: usize,
    /// Number of generated tokens.
    pub generated_tokens: usize,
    /// Why the loop stopped.
    pub finish_reason: FinishReason,
    /// Wall-clock time of the whole call (prefill + decode).
    pub elapsed: Duration,
}

impl GenerationOutput {
    /// Decode throughput in tokens/second.
    pub fn tokens_per_second(&self) -> f64 {
        let secs = self.elapsed.as_secs_f64();
        if secs <= 0.0 {
            return 0.0;
        }
        self.generated_tokens as f64 / secs
    }
}

/// Everything the checkpoint declares as "end of answer".
///
/// Union of the tokenizer's markers and `config.json` / `generation_config.json`
/// ids, because the 7B family ships several (`eos_token_id: [127960, 127967]`).
fn stop_ids(model: &dyn CausalLm, tokenizer: &Tokenizer, extra: &[u32]) -> Vec<u32> {
    let mut ids = tokenizer.special().eos_ids();
    ids.extend(model.config().eos_ids());
    ids.extend_from_slice(extra);
    ids.sort_unstable();
    ids.dedup();
    ids
}

/// An in-progress generation.
///
/// The session owns the KV cache and the sampler, so callers can drive decoding
/// one token at a time (streaming, custom stopping rules, resumable generation).
/// [`generate`] is a thin loop on top of it.
pub struct Session<'a> {
    model: &'a mut dyn CausalLm,
    tokenizer: &'a Tokenizer,
    cache: KVCache,
    sampler: Sampler,
    decoder: IncrementalDecoder<'a>,
    /// Prompt + generated ids (used by the repetition penalty).
    history: Vec<u32>,
    stop: Vec<u32>,
    max_new: usize,
    prompt_len: usize,
    /// Logits of the next position, `(1, 1, vocab)`.
    logits: Tensor,
    finished: Option<FinishReason>,
    started: Instant,
}

impl std::fmt::Debug for Session<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Session")
            .field("prompt_tokens", &self.prompt_len)
            .field("generated_tokens", &self.decoder.len())
            .field("max_new", &self.max_new)
            .field("finished", &self.finished)
            .finish()
    }
}

impl<'a> Session<'a> {
    /// Prefills the prompt and prepares decoding.
    pub fn new(
        model: &'a mut dyn CausalLm,
        tokenizer: &'a Tokenizer,
        prompt_ids: &[u32],
        cfg: &GenerateConfig,
    ) -> Result<Self> {
        if prompt_ids.is_empty() {
            return Err(Error::invalid_arg("generation: empty prompt"));
        }
        let started = Instant::now();
        let max_pos = model.config().max_position_embeddings;
        let max_new = cfg.max_new_tokens.min(max_pos.saturating_sub(prompt_ids.len()));
        let mut cache = model.new_cache(prompt_ids.len() + max_new)?;
        let device = model.device().clone();

        let prompt = Tensor::new(prompt_ids, &device)?.unsqueeze(0)?;
        let logits = model.forward(&prompt, &mut cache, true)?;
        let stop = stop_ids(&*model, tokenizer, &cfg.stop_token_ids);

        Ok(Self {
            model,
            tokenizer,
            cache,
            sampler: Sampler::new(cfg.sampling.clone()),
            decoder: tokenizer.incremental_decoder(),
            history: prompt_ids.to_vec(),
            stop,
            max_new,
            prompt_len: prompt_ids.len(),
            logits,
            finished: None,
            started,
        })
    }

    /// Picks the next token without advancing the model.
    fn sample_token(&mut self) -> CandleResult<u32> {
        let mut logits = self.logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        Ok(self.sampler.sample(&mut logits, &self.history))
    }

    /// Generates one token.
    ///
    /// Returns `Ok(None)` once generation is over, otherwise the token id and the
    /// text delta that is safe to display (it can be `None` when the token only
    /// carries bytes of a multi-byte character).
    pub fn step(&mut self) -> Result<Option<(u32, Option<String>)>> {
        if self.finished.is_some() {
            return Ok(None);
        }
        if self.decoder.len() >= self.max_new {
            self.finished = Some(FinishReason::MaxTokens);
            return Ok(None);
        }
        let token = self.sample_token()?;
        if self.stop.contains(&token) {
            self.finished = Some(FinishReason::StopToken);
            return Ok(None);
        }
        self.history.push(token);
        let delta = self.decoder.push(token)?;

        if self.decoder.len() >= self.max_new {
            self.finished = Some(FinishReason::MaxTokens);
            let _ = self.decoder.finish();
            return Ok(Some((token, delta)));
        }
        let device = self.model.device().clone();
        let input = Tensor::new(&[token], &device)?.unsqueeze(0)?;
        self.logits = self.model.forward(&input, &mut self.cache, true)?;
        Ok(Some((token, delta)))
    }

    /// Runs the session to completion, emitting text deltas to `on_delta`.
    pub fn run<F>(&mut self, mut on_delta: F) -> Result<()>
    where
        F: FnMut(&str),
    {
        while let Some((_, delta)) = self.step()? {
            if let Some(delta) = delta {
                on_delta(&delta);
            }
        }
        if let Some(tail) = self.decoder.finish()? {
            on_delta(&tail);
        }
        Ok(())
    }

    /// Next text delta, or `None` when the session is exhausted.
    pub fn next_delta(&mut self) -> Result<Option<String>> {
        loop {
            match self.step()? {
                None => return self.decoder.finish(),
                Some((_, None)) => continue,
                Some((_, Some(delta))) => return Ok(Some(delta)),
            }
        }
    }

    /// Text emitted so far.
    pub fn text(&self) -> &str {
        self.decoder.text()
    }

    /// Generated token ids so far.
    pub fn token_ids(&self) -> &[u32] {
        self.decoder.ids()
    }

    /// Number of generated tokens.
    pub fn generated_tokens(&self) -> usize {
        self.decoder.len()
    }

    /// Number of prompt tokens.
    pub fn prompt_tokens(&self) -> usize {
        self.prompt_len
    }

    /// Why generation stopped (`None` while it is still running).
    pub fn finish_reason(&self) -> Option<FinishReason> {
        self.finished
    }

    /// Time since the session was created.
    pub fn elapsed(&self) -> Duration {
        self.started.elapsed()
    }

    /// Tokenizer used by the session.
    pub fn tokenizer(&self) -> &Tokenizer {
        self.tokenizer
    }

    /// Turns the finished session into a [`GenerationOutput`].
    pub fn into_output(mut self) -> Result<GenerationOutput> {
        let _ = self.decoder.finish()?;
        Ok(GenerationOutput {
            text: self.decoder.text().to_string(),
            token_ids: self.decoder.ids().to_vec(),
            prompt_tokens: self.prompt_len,
            generated_tokens: self.decoder.len(),
            finish_reason: self.finished.unwrap_or(FinishReason::MaxTokens),
            elapsed: self.started.elapsed(),
        })
    }
}

/// Runs greedy/sampled decoding over `prompt_ids`.
///
/// `on_delta` is called with each decoded increment, which makes streaming
/// trivial (pass `|_| {}` to ignore it).
pub fn generate<F>(
    model: &mut dyn CausalLm,
    tokenizer: &Tokenizer,
    prompt_ids: &[u32],
    cfg: &GenerateConfig,
    on_delta: F,
) -> Result<GenerationOutput>
where
    F: FnMut(&str),
{
    let mut session = Session::new(model, tokenizer, prompt_ids, cfg)?;
    session.run(on_delta)?;
    session.into_output()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finish_reason_display() {
        assert_eq!(FinishReason::StopToken.to_string(), "stop_token");
        assert_eq!(FinishReason::MaxTokens.to_string(), "max_tokens");
    }

    #[test]
    fn config_builders() {
        let cfg = GenerateConfig::greedy()
            .max_new_tokens(7)
            .temperature(0.3)
            .top_p(0.9)
            .top_k(5)
            .repetition_penalty(1.1)
            .seed(42);
        assert_eq!(cfg.max_new_tokens, 7);
        assert_eq!(cfg.sampling.temperature, 0.3);
        assert_eq!(cfg.sampling.top_p, 0.9);
        assert_eq!(cfg.sampling.top_k, 5);
        assert_eq!(cfg.sampling.repetition_penalty, 1.1);
        assert_eq!(cfg.sampling.seed, Some(42));
    }
}
