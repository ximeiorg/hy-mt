use candle_core::{DType, Device, Result, Tensor};
use tokenizers::Tokenizer;

use crate::{
    logits_processor::LogitsProcessor, model::HunYuanDenseV1ForCausalLM, tokenizer::TokenOutputStream, utils::apply_repeat_penalty,
};

pub mod config;
pub mod logits_processor;
pub mod masking_utils;
pub mod model;
pub mod tokenizer;
pub mod utils;

pub struct TextGeneration {
    device: Device,
    model: HunYuanDenseV1ForCausalLM,
    tokenizer: TokenOutputStream,
    logits_processor: LogitsProcessor,
    repetition_penalty: f32,
    repeat_last_n: usize,
}


impl TextGeneration {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        model: HunYuanDenseV1ForCausalLM,
        tokenizer: Tokenizer,
        seed: u64,
        temp: Option<f64>,
        top_p: Option<f64>,
        repetition_penalty: f32,
        repeat_last_n: usize,
        device: &Device,
    ) -> Result<Self> {
        let logits_processor = LogitsProcessor::new(seed, temp, top_p);
        let tokenizer = TokenOutputStream::new(tokenizer);
        Ok(Self {
            device: device.clone(),
            model,
            tokenizer,
            logits_processor,
            repetition_penalty,
            repeat_last_n,
        })
    }
    pub fn new_with_default(model: HunYuanDenseV1ForCausalLM, tokenizer: Tokenizer, device: &Device) -> Result<Self> {
        Self::new(model, tokenizer, 0, Some(0.7), Some(0.6), 1.05, 0, device)
    }

    pub fn generate(&mut self, prompt: &str, max_length: usize) -> Result<String> { 
        use std::io::Write;
        self.tokenizer.clear();
        let mut tokens = self
            .tokenizer
            .tokenizer()
            .encode(prompt, true)
            .unwrap()
            .get_ids()
            .to_vec();
        for &t in tokens.iter() {
            if let Some(t) = self.tokenizer.next_token(t)? {
                print!("{t}")
            }
        }
        std::io::stdout().flush()?;

        let mut generated_tokens = 0usize;

        for index in 0..max_length {
            let context_size = if index > 0 { 1 } else { tokens.len() };
            let start_pos = tokens.len().saturating_sub(context_size);
            let ctxt = &tokens[start_pos..];
            let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;
            let out = self.model.forward(Some(&input), None, None,None,None)?;
            let logits = out.logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
            let logits = if self.repetition_penalty == 1. {
                logits
            } else {
                let start_at = tokens.len().saturating_sub(self.repeat_last_n);
                apply_repeat_penalty(
                    &logits,
                    self.repetition_penalty,
                    &tokens[start_at..],
                )?
            };
            let next_token = self.logits_processor.sample(&logits)?;
            tokens.push(next_token);
            generated_tokens += 1;

            if let Some(t) = self.tokenizer.next_token(next_token)? {
                print!("{t}");
                std::io::stdout().flush()?;
            }
        }

        Ok("()".to_string())
    }
}
