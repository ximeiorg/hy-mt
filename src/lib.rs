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
        // 增加重复惩罚到1.1，避免重复生成问题
        // 将 repeat_last_n 设置为合理的值（比如64），避免对空上下文应用惩罚
        Self::new_with_topk(model, tokenizer, 0, Some(0.7), Some(20), Some(0.6), 1.1, 64, device)
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

    pub fn model(&self) -> &HunYuanDenseV1ForCausalLM {
        &self.model
    }

    pub fn tokenizer(&self) -> &TokenOutputStream {
        &self.tokenizer
    }

    pub fn generate(&mut self, prompt: &str, max_length: usize) -> Result<String> { 
        use std::io::Write;
        self.tokenizer.clear();
        
        // 构建与Python版本完全一致的聊天模板格式
        // 格式: <｜hy_begin▁of▁sentence｜><｜hy_User｜>Translate the following segment into Chinese without additional explanation.\n\n{source_text}<｜hy_place▁holder▁no▁8｜>
        let formatted_prompt = format!("<｜hy_begin▁of▁sentence｜><｜hy_User｜>Translate the following segment into Chinese without additional explanation.\n\n{}<｜hy_place▁holder▁no▁8｜>", prompt);
        
        println!("格式化提示文本: {}", TextGeneration::repr(&formatted_prompt));
        println!("原始prompt: {}", prompt);
        
        // 编码格式化后的提示文本
        let mut tokens = self.tokenizer.tokenizer()
            .encode(formatted_prompt, true)
            .map_err(|e| candle_core::Error::msg(format!("Failed to encode prompt: {}", e)))?
            .get_ids()
            .to_vec();
        
        println!("Token IDs: {:?}", tokens);
        
        // 编码并打印提示文本
        for &t in tokens.iter() {
            if let Some(t) = self.tokenizer.next_token(t)? {
                print!("{t}")
            }
        }
        std::io::stdout().flush()?;

        // 使用完整的上下文窗口进行生成，而不是每次只使用1个token
        // 这有助于模型更好地理解上下文
        let context_window = 512; // 使用合理的上下文窗口大小

        for index in 0..max_length {
            // 使用完整的上下文窗口，但不超过tokens长度
            let context_size = tokens.len().min(context_window);
            let start_pos = tokens.len().saturating_sub(context_size);
            let ctxt = &tokens[start_pos..];
            let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;

            
            // 创建注意力掩码：所有位置都是有效token（非padding）
            let attention_mask = Tensor::ones((1, ctxt.len()), DType::U8, &self.device)?;
            
            // 在生成过程中，我们只需要最后一个位置的logits
            // 根据Python实现，传递None表示不需要特殊处理logits_to_keep
            let out = self.model.forward(Some(&input), Some(&attention_mask), None, None, None)?;
            
            // 调试：检查logits形状
            // println!("Logits shape: {:?}", out.logits.shape());
            
            // 获取logits（形状为[1, seq_len, vocab_size]），然后取最后一个位置的logits
            let logits = out.logits.squeeze(0)?.narrow(0, ctxt.len() - 1, 1)?.squeeze(0)?.to_dtype(DType::F32)?;
            
            // 调试：检查logits值
            // println!("Logits shape after processing: {:?}", logits.shape());
            
            // 调试：检查前10个logits值
            // let logits_vec: Vec<f32> = logits.to_vec1()?;
            // let top_5_indices: Vec<usize> = {
            //     let mut indices: Vec<usize> = (0..logits_vec.len()).collect();
            //     indices.sort_by(|&a, &b| logits_vec[b].partial_cmp(&logits_vec[a]).unwrap());
            //     indices.truncate(5);
            //     indices
            // };
            // println!("Top 5 logits indices: {:?}", top_5_indices);
            // println!("Top 5 logits values: {:?}", top_5_indices.iter().map(|&i| logits_vec[i]).collect::<Vec<f32>>());
            
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
            
            // println!("Generated token {}: {} (EOS: {})", index, next_token, next_token == 120020);

            // 检查是否生成了结束标记
            if next_token == 120020 { // eos_token_id
                // println!("检测到EOS token，停止生成");
                break;
            }
            
            // 额外的安全机制：如果连续生成重复的token，强制停止
            if tokens.len() >= 3 {
                let last_three = &tokens[tokens.len()-3..];
                if last_three[0] == last_three[1] && last_three[1] == last_three[2] {
                    // println!("检测到连续重复token，强制停止生成");
                    break;
                }
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
    
    // 辅助函数：显示字符串的repr形式，用于调试
    fn repr(s: &str) -> String {
        s.chars()
            .map(|c| match c {
                '\n' => "\\n".to_string(),
                '\t' => "\\t".to_string(),
                '\r' => "\\r".to_string(),
                '\\' => "\\\\".to_string(),
                '"' => "\\\"".to_string(),
                c if c.is_control() => format!("\\u{{{:04x}}}", c as u32),
                c => c.to_string(),
            })
            .collect()
    }
}