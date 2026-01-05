use std::path::Path;

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

    pub fn from_pretrained<P: AsRef<std::path::Path>>(path: P, device: &Device) -> Result<Self> {
        use std::fs;
        use candle_core::DType;
        
        // 从指定路径加载配置
        let config_path = path.as_ref().join("config.json");
        let config_str = fs::read_to_string(config_path).map_err(candle_core::Error::msg)?;
        let config: config::HunYuanDenseV1Config = serde_json::from_str(&config_str)
            .map_err(candle_core::Error::msg)?;

        // 加载tokenizer
        let tokenizer_path = path.as_ref().join("tokenizer.json");
        let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path)
            .map_err(|e| candle_core::Error::msg(format!("Failed to load tokenizer: {}", e)))?;

        // 加载模型权重
        let weights_path = path.as_ref().join("model.safetensors");

        let vb = unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, device) }?;

        // 加载模型
        let model = model::HunYuanDenseV1ForCausalLM::load(&config, vb)?;

        // 创建TextGeneration实例
        Self::new_with_default(model, tokenizer, device)
    }

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
        // 将 repeat_last_n 设置为合理的值（比如64），避免对空上下文应用惩罚
        Self::new(model, tokenizer, 0, Some(0.7), Some(0.6), 1.05, 64, device)
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
                // 确保上下文不为空，避免对空上下文应用惩罚
                if start_at < tokens.len() {
                    apply_repeat_penalty(
                        &logits,
                        self.repetition_penalty,
                        &tokens[start_at..],
                    )?
                } else {
                    logits
                }
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