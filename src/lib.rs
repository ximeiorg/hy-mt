use std::path::Path;

use candle_core::{DType, Device, Result, Tensor, Error};
use tokenizers::Tokenizer;

use crate::{
    logits_processor::LogitsProcessor, model::HunYuanDenseV1ForCausalLM, prompt::Prompter, tokenizer::TokenOutputStream, utils::apply_repeat_penalty
};

pub mod config;
pub mod logits_processor;
pub mod masking_utils;
pub mod model;
pub mod tokenizer;
pub mod utils;
pub mod prompt;

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

    pub fn generate(&mut self, prompt: impl Prompter, max_length: usize) -> Result<String> { 
        use std::io::Write;
        self.tokenizer.clear();
        
        // 构建与Python版本完全一致的聊天模板格式
        // 格式: <｜hy_begin of sentence｜><｜hy_User｜>Translate the following segment into Chinese without additional explanation.\n\n{source_text}<｜hy_place holder no 8｜>
        let formatted_prompt = prompt.to_prompt();
        
        // 编码格式化后的提示文本
        let mut tokens = self.tokenizer.tokenizer()
            .encode(formatted_prompt, true)
            .map_err(|e| candle_core::Error::msg(format!("Failed to encode prompt: {}", e)))?
            .get_ids()
            .to_vec();
        
        // println!("Token IDs: {:?}", tokens);
        
        // 编码并打印提示文本
        // for &t in tokens.iter() {
        //     if let Some(t) = self.tokenizer.next_token(t)? {
        //         print!("{t}")
        //     }
        // }
        // std::io::stdout().flush()?;

        // 使用完整的上下文窗口进行生成，而不是每次只使用1个token
        // 这有助于模型更好地理解上下文
        let context_window = 512; // 使用合理的上下文窗口大小

        for _index in 0..max_length {
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

    /// 流式生成文本，返回一个迭代器，可以逐个token地获取生成的文本
    /// 
    /// # 参数
    /// - `prompt`: 输入提示
    /// - `max_length`: 最大生成长度
    /// 
    /// # 示例
    /// ```rust
    /// use hy_mt::TextGeneration;
    /// 
    /// let mut model = TextGeneration::from_pretrained(path, &device)?;
    /// for chunk in model.generate_stream_iter("Hello world", 512) {
    ///     println!("生成片段: {}", chunk);
    /// }
    /// ```
    pub fn generate_stream_iter<'a>(&'a mut self, prompt: impl Prompter, max_length: usize) -> impl Iterator<Item = String> + 'a {
        TextGenerationStream::new(self, prompt, max_length)
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

/// 流式文本生成迭代器
pub struct TextGenerationStream<'a> {
    model: &'a mut TextGeneration,
    max_length: usize,
    tokens: Vec<u32>,
    index: usize,
    finished: bool,
}

impl<'a> TextGenerationStream<'a> {
    fn new(model: &'a mut TextGeneration, prompt: impl Prompter, max_length: usize) -> Self {
        model.tokenizer.clear();
        
        // 构建提示模板
        let formatted_prompt = prompt.to_prompt();
        
        // 编码提示文本
        let tokens = model.tokenizer.tokenizer()
            .encode(formatted_prompt, true)
            .map(|encoding| encoding.get_ids().to_vec())
            .unwrap_or_else(|_| Vec::new());
        
        // 将提示文本的tokens添加到tokenizer中，以便进行增量解码
        for &token in &tokens {
            let _ = model.tokenizer.next_token(token);
        }
        
        Self {
            model,
            max_length,
            tokens,
            index: 0,
            finished: false,
        }
    }
}

impl<'a> Iterator for TextGenerationStream<'a> {
    type Item = String;
    
    fn next(&mut self) -> Option<Self::Item> {
        // 首先检查是否已经结束
        if self.finished {
            return None;
        }
        
        // 检查是否达到最大长度
        if self.index >= self.max_length {
            self.finished = true;
            // 解码剩余文本
            if let Ok(Some(rest)) = self.model.tokenizer.decode_rest() {
                return Some(rest);
            }
            return None;
        }
        
        // 使用完整的上下文窗口
        let context_window = 512;
        let context_size = self.tokens.len().min(context_window);
        let start_pos = self.tokens.len().saturating_sub(context_size);
        let ctxt = &self.tokens[start_pos..];
        
        // 准备输入张量
        let input = match Tensor::new(ctxt, &self.model.device) {
            Ok(t) => match t.unsqueeze(0) {
                Ok(t) => t,
                Err(_) => return None,
            },
            Err(_) => return None,
        };
        
        // 创建注意力掩码
        let attention_mask = match Tensor::ones((1, ctxt.len()), DType::U8, &self.model.device) {
            Ok(mask) => mask,
            Err(_) => return None,
        };
        
        // 前向传播
        let out = match self.model.model.forward(Some(&input), Some(&attention_mask), None, None, None) {
            Ok(out) => out,
            Err(_) => return None,
        };
        
        // 获取logits
        let logits = match out.logits.squeeze(0) {
            Ok(l) => match l.narrow(0, ctxt.len() - 1, 1)
                .and_then(|l| l.squeeze(0))
                .and_then(|l| l.to_dtype(DType::F32)) {
                Ok(logits) => logits,
                Err(_) => return None,
            },
            Err(_) => return None,
        };
        
        // 应用重复惩罚
        let logits = if self.model.repetition_penalty != 1. {
            let start_at = self.tokens.len().saturating_sub(self.model.repeat_last_n);
            if start_at < self.tokens.len() {
                match apply_repeat_penalty(&logits, self.model.repetition_penalty, &self.tokens[start_at..]) {
                    Ok(logits) => logits,
                    Err(_) => return None,
                }
            } else {
                logits
            }
        } else {
            logits
        };
        
        // 采样下一个token
        let next_token = match self.model.logits_processor.sample(&logits) {
            Ok(token) => token,
            Err(_) => return None,
        };
        
        self.tokens.push(next_token);
        self.index += 1;
        
        // 检查结束条件 - 与generate方法保持一致
        // 1. 检查 EOS token (使用与generate方法相同的硬编码值)
        if next_token == 120020 { // eos_token_id
            self.finished = true;
        }
        
        // 2. 检查重复token序列（与generate方法相同的逻辑：连续3个相同token）
        if self.tokens.len() >= 3 {
            let last_three = &self.tokens[self.tokens.len()-3..];
            if last_three[0] == last_three[1] && last_three[1] == last_three[2] {
                self.finished = true;
            }
        }
        
        // 解码token并返回文本片段
        match self.model.tokenizer.next_token(next_token) {
            Ok(Some(chunk)) => {
                // 如果迭代器已经结束，但还有剩余文本，返回它
                if self.finished {
                    // 解码所有剩余文本
                    if let Ok(Some(rest)) = self.model.tokenizer.decode_rest() {
                        return Some(chunk + &rest);
                    }
                }
                Some(chunk)
            },
            Ok(None) => {
                // 如果没有生成文本，但迭代器已结束，返回空字符串
                if self.finished {
                    None
                } else {
                    Some(String::new())
                }
            },
            Err(_) => None,
        }
    }
}

impl<'a> Drop for TextGenerationStream<'a> {
    fn drop(&mut self) {
        // 在迭代器被丢弃时解码剩余文本
        if let Ok(Some(rest)) = self.model.tokenizer.decode_rest() {
            // 这里可以选择将剩余文本作为最后一个元素返回
            // 但由于迭代器已经结束，我们只能忽略它
        }
    }
}