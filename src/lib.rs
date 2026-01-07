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

        // 加载模型权重（使用float32精度，因为CPU不支持bfloat16的matmul操作）
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
        // 使用HunYuan模型推荐的生成参数：top_k=20, top_p=0.6, temperature=0.7
        // 降低重复惩罚（从1.05降到1.01），避免过于严格的重复检测
        // 将 repeat_last_n 设置为合理的值（比如64），避免对空上下文应用惩罚
        Self::new_with_topk(model, tokenizer, 0, Some(0.7), Some(20), Some(0.6), 1.01, 64, device)
    }
    
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_topk(
        model: HunYuanDenseV1ForCausalLM,
        tokenizer: Tokenizer,
        seed: u64,
        temp: Option<f64>,
        top_k: Option<usize>,
        top_p: Option<f64>,
        repetition_penalty: f32,
        repeat_last_n: usize,
        device: &Device,
    ) -> Result<Self> {
        let logits_processor = LogitsProcessor::new_with_topk(seed, temp, top_k, top_p);
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

    pub fn generate(&mut self, prompt: &str, max_length: usize) -> Result<String> { 
        use std::io::Write;
        self.tokenizer.clear();
        
        // 构建与Python版本完全一致的聊天模板格式
        // 格式: <｜hy_begin▁of▁sentence｜><｜hy_User｜>Translate the following segment into Chinese without additional explanation.\n{source_text}<｜hy_place▁holder▁no▁8｜>
        let formatted_prompt = format!("<｜hy_begin▁of▁sentence｜><｜hy_User｜>Translate the following segment into Chinese without additional explanation.\n{}<｜hy_place▁holder▁no▁8｜>", prompt);
        
        // 编码格式化后的提示文本
        let mut tokens = self.tokenizer.tokenizer()
            .encode(formatted_prompt, true)
            .map_err(|e| candle_core::Error::msg(format!("Failed to encode prompt: {}", e)))?
            .get_ids()
            .to_vec();
        
        // 编码并打印提示文本
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
            
            // 创建注意力掩码：所有位置都是有效token（非padding）
            let attention_mask = Tensor::ones((1, ctxt.len()), DType::U8, &self.device)?;
            
            // 在生成过程中，我们只需要最后一个位置的logits
            // 根据Python实现，传递None表示不需要特殊处理logits_to_keep
            let out = self.model.forward(Some(&input), Some(&attention_mask), None, None, None)?;
            // 获取logits（形状为[1, seq_len, vocab_size]），然后取最后一个位置的logits
            let logits = out.logits.squeeze(0)?.narrow(0, ctxt.len() - 1, 1)?.squeeze(0)?.to_dtype(DType::F32)?;
            
            // 调试：检查第一个生成的token的logits分布
            if index == 0 {
                println!("\n=== 调试第一个token的logits分布 ===");
                println!("logits形状: {:?}", logits.dims());
                
                // 手动计算softmax概率
                let max_logit = logits.max(0)?;
                // 将max_logit扩展到与logits相同的形状
                let max_logit_expanded = max_logit.broadcast_as(logits.dims())?;
                let shifted_logits = (&logits - &max_logit_expanded)?;
                let exp_logits = shifted_logits.exp()?;
                let sum_exp = exp_logits.sum(0)?;
                // 将sum_exp扩展到与exp_logits相同的形状
                let sum_exp_expanded = sum_exp.broadcast_as(exp_logits.dims())?;
                let probs = (&exp_logits / &sum_exp_expanded)?;
                
                // 手动实现top-k选择
                let k = 10;
                let vocab_size = logits.dims()[0];
                
                // 收集所有概率和索引
                let mut all_probs: Vec<(f32, u32)> = Vec::new();
                for i in 0..vocab_size {
                    let prob: f32 = probs.get(i)?.to_scalar()?;
                    all_probs.push((prob, i as u32));
                }
                
                // 按概率降序排序
                all_probs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
                
                println!("前{}个最可能的token:", k);
                for i in 0..k {
                    let (prob, idx) = all_probs[i];
                    println!("{:2}. Token ID: {:6}, 概率: {:.4}", i+1, idx, prob);
                }
                
                // 检查关键token的概率
                let key_tokens = [2947, 120020, 120007, 120026]; // 2947 = "这是"
                println!("\n关键token检查:");
                for &token_id in &key_tokens {
                    if token_id < vocab_size as u32 {
                        let prob: f32 = probs.get(token_id as usize)?.to_scalar()?;
                        println!("Token ID {}: 概率 = {:.6}", token_id, prob);
                    }
                }
            }
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

            // 检查是否生成了结束标记
            if next_token == 120020 { // eos_token_id
                break;
            }

            if let Some(t) = self.tokenizer.next_token(next_token)? {
                print!("{t}");
                std::io::stdout().flush()?;
            }
        }

        // 解码剩余文本并返回完整结果
        if let Some(rest) = self.tokenizer.decode_rest()? {
            print!("{rest}");
        }
        std::io::stdout().flush()?;
        
        let generated_text = self.tokenizer.decode_all()?;
        Ok(generated_text)
    }
}