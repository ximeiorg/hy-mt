use candle_core::{D, DType, Device, Result, Tensor};
use crate::config::HunYuanDenseV1Config;

// 定义掩码预处理结果结构
pub struct MaskPreprocessResult {
    pub early_exit: bool,
    pub attention_mask: Option<Tensor>,
    pub kv_length: usize,
    pub kv_offset: usize,
}

// 定义缓存接口（简化版本）
pub trait Cache {
    fn get_mask_sizes(&self, cache_position: &Tensor, layer_idx: Option<usize>) -> (usize, usize);
}

/// 掩码预处理函数，类似于Python中的_preprocess_mask_arguments
pub fn preprocess_mask_arguments(
    config: &HunYuanDenseV1Config,
    input_embeds: &Tensor,
    attention_mask: Option<&Tensor>,
    cache_position: &Tensor,
    past_key_values: Option<&dyn Cache>,
    position_ids: Option<&Tensor>,
    layer_idx: Option<usize>,
) -> Result<MaskPreprocessResult> {
    let device = input_embeds.device();
    
    // 如果掩码已经是4D，直接返回（已经准备好的掩码）
    if let Some(mask) = attention_mask {
        if mask.dims().len() == 4 {
            return Ok(MaskPreprocessResult {
                early_exit: true,
                attention_mask: Some(mask.clone()),
                kv_length: 0,
                kv_offset: 0,
            });
        }
    }
    
    // 检查是否支持掩码注意力实现
    // 这里简化处理，假设总是支持
    let supported_attn_implementation = true;
    if !supported_attn_implementation {
        return Ok(MaskPreprocessResult {
            early_exit: true,
            attention_mask: None,
            kv_length: 0,
            kv_offset: 0,
        });
    }
    
    // 将2D掩码转换为U8类型并移动到正确设备
    let mut processed_mask = attention_mask.cloned();
    if let Some(ref mask) = processed_mask {
        if mask.dims().len() == 2 {
            processed_mask = Some(mask.to_device(device)?.to_dtype(DType::U8)?);
        }
    }
    
    // 确定kv_length和kv_offset
    let (kv_length, kv_offset) = if let Some(cache) = past_key_values {
        cache.get_mask_sizes(cache_position, layer_idx)
    } else {
        // 根据Python实现，如果没有缓存，kv_length就是输入序列长度，kv_offset为0
        (input_embeds.dims()[1], 0)
    };
    
    // 这里简化处理，不实现packed_sequence_mask
    
    Ok(MaskPreprocessResult {
        early_exit: false,
        attention_mask: processed_mask,
        kv_length,
        kv_offset,
    })
}

fn causal_mask_function(q_idx: usize, kv_idx: usize)->bool {
    kv_idx <= q_idx
}

pub fn sdpa_mask(
    batch_size: usize,
    q_length: usize,
    kv_length: usize,
    is_causal: bool,
    device: &Device,
)-> Result<Tensor> {
    // 如果不需要因果掩码，返回全True掩码
    if !is_causal {
        return Tensor::ones((batch_size, 1, q_length, kv_length), DType::U8, device);
    }

    // 创建kv位置的索引 [0, 1, 2, ..., kv_length-1]
    let kv_indices = Tensor::arange(0u32, kv_length as u32, device)?;
    
    // 创建q位置的索引 [0, 1, 2, ..., q_length-1]
    let q_indices = Tensor::arange(0u32, q_length as u32, device)?;

    // 扩展维度以便广播
    // q_indices: [q_length] -> [q_length, 1]
    // kv_indices: [kv_length] -> [1, kv_length]
    let q_indices = q_indices.unsqueeze(D::Minus1)?; // [q_length, 1]
    let kv_indices = kv_indices.unsqueeze(0)?; // [1, kv_length]

    // 生成因果掩码：q_position >= kv_position
    // 对于因果掩码，每个位置只能看到前面和自身的位置
    // mask[q, k] = True if q >= k else False
    let mask = q_indices.broadcast_ge(&kv_indices)?; // [q_length, kv_length]
    
    // 扩展为4D: [1, 1, q_length, kv_length]
    let mask = mask.unsqueeze(0)?.unsqueeze(0)?; 
    
    // 扩展到batch维度: [batch_size, 1, q_length, kv_length]
    mask.expand((batch_size, 1, q_length, kv_length))
}



pub fn create_causal_mask(
    config: &HunYuanDenseV1Config,
    input_embeds: &Tensor,
    attention_mask: Option<&Tensor>,
    cache_position: &Tensor,
    past_key_values: Option<&dyn Cache>,
    position_ids: Option<&Tensor>,
) -> Result<Option<Tensor>> {
    // 确定layer_idx：如果有混合缓存结构，找到第一个非滑动层的索引
    let layer_idx = {
        // 这里简化处理，假设没有混合缓存结构，使用默认值0
        // 在实际实现中，需要检查past_key_values是否有is_sliding属性
        0
    };
    
    // 预处理掩码参数
    let preprocess_result = preprocess_mask_arguments(
        config,
        input_embeds,
        attention_mask,
        cache_position,
        past_key_values,
        position_ids,
        Some(layer_idx), // layer_idx
    )?;
    
    // 如果需要提前退出，直接返回处理后的掩码
    if preprocess_result.early_exit {
        return Ok(preprocess_result.attention_mask);
    }
    
    // 获取输入的形状信息
    let batch_size = input_embeds.dims()[0];
    let q_length = input_embeds.dims()[1];
    let device = input_embeds.device();
    
    // 创建因果掩码 - 使用sdpa_mask作为mask_interface
    let causal_mask = sdpa_mask(
        batch_size,
        q_length,
        preprocess_result.kv_length,
        true, // is_causal
        device,
    )?;
    
    // 如果有2D注意力掩码，需要将其与因果掩码结合
    if let Some(mask_2d) = preprocess_result.attention_mask {
        if mask_2d.dims().len() == 2 {
            // 将2D掩码扩展为4D
            // 原始形状: [batch_size, kv_length]
            // 目标形状: [batch_size, 1, q_length, kv_length]
            // 正确的方式：将2D掩码扩展为4D，然后与因果掩码进行逻辑与操作
            
            // 首先将2D掩码扩展为4D: [batch_size, 1, 1, kv_length]
            let mask_4d = mask_2d
                .unsqueeze(1)?  // [batch_size, 1, kv_length]
                .unsqueeze(2)?; // [batch_size, 1, 1, kv_length]
            
            // 然后扩展到完整的4D形状: [batch_size, 1, q_length, kv_length]
            let mask_4d = mask_4d.expand((batch_size, 1, q_length, preprocess_result.kv_length))?;
            
            // 将因果掩码和2D掩码进行逻辑与操作
            // 正确的逻辑：使用where_cond方法
            // 当mask_4d不为0时，取causal_mask的值；当mask_4d为0时，取0
            let zeros = causal_mask.zeros_like()?;
            let combined_mask = mask_4d.where_cond(&causal_mask, &zeros)?;
            return Ok(Some(combined_mask));
        }
    }
    
    Ok(Some(causal_mask))
}



mod tests {
    use super::*;
    use candle_core::{*};


    #[test]
    fn test_causal_mask() -> Result<()> {
        let device = Device::Cpu;
        let batch_size = 1;
        let q_length = 5;
        let kv_length = 5;
        
        let mask = sdpa_mask(batch_size, q_length, kv_length, true, &device)?;
        
        // 检查形状
        assert_eq!(mask.dims(), &[batch_size, 1, q_length, kv_length]);
        
        // 检查对角线以上应该是True（包括对角线），以下应该是False
        let mask_slice = mask.flatten_all()?.to_vec1::<u8>()?;
        
        // 预期的因果掩码模式
        // ■ ⬚ ⬚ ⬚ ⬚  = 1 0 0 0 0
        // ■ ■ ⬚ ⬚ ⬚  = 1 1 0 0 0  
        // ■ ■ ■ ⬚ ⬚  = 1 1 1 0 0
        // ■ ■ ■ ■ ⬚  = 1 1 1 1 0
        // ■ ■ ■ ■ ■  = 1 1 1 1 1
        
        // 验证第一行
        assert_eq!(mask_slice[0..5], [1, 0, 0, 0, 0]);
        // 验证最后一行  
        assert_eq!(mask_slice[20..25], [1, 1, 1, 1, 1]);
        
        Ok(())
    }

    #[test]
    fn test_create_causal_mask_basic() -> Result<()> {
        let device = Device::Cpu;
        let config = HunYuanDenseV1Config::default();
        
        // 创建输入嵌入
        let batch_size = 2;
        let seq_len = 3;
        let hidden_size = 4;
        let input_embeds = Tensor::randn(0f32, 1.0, (batch_size, seq_len, hidden_size), &device)?;
        
        // 创建缓存位置
        let cache_position = Tensor::arange(0u32, seq_len as u32, &device)?;
        
        // 测试基本因果掩码创建
        let mask = create_causal_mask(
            &config,
            &input_embeds,
            None, // 无注意力掩码
            &cache_position,
            None, // 无缓存
            None, // 无位置ID
        )?;
        
        assert!(mask.is_some());
        let mask = mask.unwrap();
        
        // 检查掩码形状
        assert_eq!(mask.dims(), &[batch_size, 1, seq_len, seq_len]);
        
        // 验证因果掩码模式
        let mask_slice = mask.flatten_all()?.to_vec1::<u8>()?;
        
        // 对于每个batch，检查因果掩码
        for b in 0..batch_size {
            let start = b * seq_len * seq_len;
            // 第一行: [1, 0, 0]
            assert_eq!(mask_slice[start..start+3], [1, 0, 0]);
            // 第二行: [1, 1, 0]  
            assert_eq!(mask_slice[start+3..start+6], [1, 1, 0]);
            // 第三行: [1, 1, 1]
            assert_eq!(mask_slice[start+6..start+9], [1, 1, 1]);
        }
        
        Ok(())
    }

    #[test]
    fn test_create_causal_mask_with_2d_attention_mask() -> Result<()> {
        let device = Device::Cpu;
        let config = HunYuanDenseV1Config::default();
        
        // 创建输入嵌入
        let batch_size = 1;
        let seq_len = 4;
        let hidden_size = 4;
        let input_embeds = Tensor::randn(0f32, 1.0, (batch_size, seq_len, hidden_size), &device)?;
        
        // 创建2D注意力掩码（模拟padding掩码）
        // 假设序列长度为3，最后一个位置是padding
        let attention_mask = Tensor::new(&[1u8, 1u8, 1u8, 0u8], &device)?.unsqueeze(0)?; // [1, 4]
        
        // 创建缓存位置
        let cache_position = Tensor::arange(0u32, seq_len as u32, &device)?;
        
        // 测试带2D掩码的因果掩码创建
        let mask = create_causal_mask(
            &config,
            &input_embeds,
            Some(&attention_mask),
            &cache_position,
            None,
            None,
        )?;
        
        assert!(mask.is_some());
        let mask = mask.unwrap();
        
        // 检查掩码形状
        assert_eq!(mask.dims(), &[batch_size, 1, seq_len, seq_len]);
        
        // 验证掩码模式：因果掩码与2D掩码的结合
        let mask_slice = mask.flatten_all()?.to_vec1::<u8>()?;
        
        // 预期模式：
        // 因果掩码:       2D掩码:       结果:
        // 1 0 0 0        1 1 1 0       1 0 0 0
        // 1 1 0 0    &   1 1 1 0   =   1 1 0 0  
        // 1 1 1 0        1 1 1 0       1 1 1 0
        // 1 1 1 1        1 1 1 0       1 1 1 0
        
        // 注意：where_cond的行为是：mask_4d不为0时取causal_mask，为0时取0
        // 所以对于第四行，mask_4d为[1,1,1,0]，因此第四行是[1,1,1,0]
        assert_eq!(mask_slice[0..4], [1, 0, 0, 0]);   // 第一行
        assert_eq!(mask_slice[4..8], [1, 1, 0, 0]);   // 第二行
        assert_eq!(mask_slice[8..12], [1, 1, 1, 0]);  // 第三行
        assert_eq!(mask_slice[12..16], [1, 1, 1, 0]); // 第四行（padding位置）
        
        Ok(())
    }



    #[test]
    fn test_create_causal_mask_with_4d_mask() -> Result<()> {
        let device = Device::Cpu;
        let config = HunYuanDenseV1Config::default();
        
        // 创建输入嵌入
        let batch_size = 1;
        let seq_len = 3;
        let hidden_size = 4;
        let input_embeds = Tensor::randn(0f32, 1.0, (batch_size, seq_len, hidden_size), &device)?;
        
        // 创建4D掩码（直接准备好的掩码）
        let attention_mask = Tensor::ones((batch_size, 1, seq_len, seq_len), DType::U8, &device)?;
        
        // 创建缓存位置
        let cache_position = Tensor::arange(0u32, seq_len as u32, &device)?;
        
        // 测试带4D掩码的情况（应该提前退出）
        let mask = create_causal_mask(
            &config,
            &input_embeds,
            Some(&attention_mask),
            &cache_position,
            None,
            None,
        )?;
        
        assert!(mask.is_some());
        let mask = mask.unwrap();
        
        // 检查返回的掩码就是原始的4D掩码
        assert_eq!(mask.dims(), &[batch_size, 1, seq_len, seq_len]);
        
        // 验证所有元素都是1（全True掩码）- 使用u8类型
        let mask_sum = mask.sum_all()?.to_scalar::<u8>()?;
        assert_eq!(mask_sum, (batch_size * seq_len * seq_len) as u8);
        
        Ok(())
    }

    #[test]
    fn test_preprocess_mask_arguments() -> Result<()> {
        let device = Device::Cpu;
        let config = HunYuanDenseV1Config::default();
        
        // 创建输入嵌入
        let batch_size = 1;
        let seq_len = 3;
        let hidden_size = 4;
        let input_embeds = Tensor::randn(0f32, 1.0, (batch_size, seq_len, hidden_size), &device)?;
        
        // 创建缓存位置
        let cache_position = Tensor::arange(0u32, seq_len as u32, &device)?;
        
        // 测试预处理函数
        let result = preprocess_mask_arguments(
            &config,
            &input_embeds,
            None, // 无注意力掩码
            &cache_position,
            None, // 无缓存
            None, // 无位置ID
            Some(0), // layer_idx
        )?;
        
        // 验证预处理结果
        assert!(!result.early_exit);
        assert!(result.attention_mask.is_none());
        assert_eq!(result.kv_length, seq_len);
        assert_eq!(result.kv_offset, 0);
        
        Ok(())
    }

    // 模拟Cache trait的实现用于测试
    struct TestCache {
        kv_length: usize,
        kv_offset: usize,
    }
    
    impl Cache for TestCache {
        fn get_mask_sizes(&self, _cache_position: &Tensor, _layer_idx: Option<usize>) -> (usize, usize) {
            (self.kv_length, self.kv_offset)
        }
    }

    #[test]
    fn debug_where_cond_behavior() -> Result<()> {
        let device = Device::Cpu;
        
        // 创建因果掩码（4x4下三角矩阵）
        let causal_mask = Tensor::new(&[
            1u8, 0u8, 0u8, 0u8,
            1u8, 1u8, 0u8, 0u8,
            1u8, 1u8, 1u8, 0u8,
            1u8, 1u8, 1u8, 1u8
        ], &device)?.reshape((1, 1, 4, 4))?;
        
        println!("Causal mask shape: {:?}", causal_mask.dims());
        println!("Causal mask values:");
        let causal_values = causal_mask.flatten_all()?.to_vec1::<u8>()?;
        for i in 0..4 {
            let start = i * 4;
            let row = &causal_values[start..start + 4];
            println!("Row {}: {:?}", i, row);
        }
        
        // 创建4D掩码（模拟扩展后的2D掩码）
        let mask_4d = Tensor::new(&[
            1u8, 1u8, 1u8, 0u8,
            1u8, 1u8, 1u8, 0u8,
            1u8, 1u8, 1u8, 0u8,
            1u8, 1u8, 1u8, 0u8
        ], &device)?.reshape((1, 1, 4, 4))?;
        
        println!("\n4D mask shape: {:?}", mask_4d.dims());
        println!("4D mask values:");
        let mask_values = mask_4d.flatten_all()?.to_vec1::<u8>()?;
        for i in 0..4 {
            let start = i * 4;
            let row = &mask_values[start..start + 4];
            println!("Row {}: {:?}", i, row);
        }
        
        // 测试where_cond行为
        let zeros = causal_mask.zeros_like()?;
        let result = mask_4d.where_cond(&causal_mask, &zeros)?;
        
        println!("\nResult of where_cond:");
        let result_values = result.flatten_all()?.to_vec1::<u8>()?;
        for i in 0..4 {
            let start = i * 4;
            let row = &result_values[start..start + 4];
            println!("Row {}: {:?}", i, row);
        }
        
        Ok(())
    }

    #[test]
    fn debug_mask_expansion() -> Result<()> {
        let device = Device::Cpu;
        
        // 创建2D掩码 [1, 1, 1, 0]
        let mask_2d = Tensor::new(&[1u8, 1u8, 1u8, 0u8], &device)?.unsqueeze(0)?; // [1, 4]
        
        println!("Original 2D mask shape: {:?}", mask_2d.dims());
        println!("Original 2D mask values: {:?}", mask_2d.flatten_all()?.to_vec1::<u8>()?);
        
        // 测试当前的扩展方式
        let batch_size = 1;
        let q_length = 4;
        let kv_length = 4;
        
        let mask_4d = mask_2d
            .unsqueeze(1)?  // [1, 1, 4]
            .unsqueeze(2)?  // [1, 1, 4, 1]
            .expand((batch_size, 1, q_length, kv_length))?;
        
        println!("Expanded 4D mask shape: {:?}", mask_4d.dims());
        println!("Expanded 4D mask values:");
        
        let mask_values = mask_4d.flatten_all()?.to_vec1::<u8>()?;
        for i in 0..q_length {
            let start = i * kv_length;
            let row = &mask_values[start..start + kv_length];
            println!("Row {}: {:?}", i, row);
        }
        
        Ok(())
    }

    #[test]
    fn test_create_causal_mask_with_cache() -> Result<()> {
        let device = Device::Cpu;
        let config = HunYuanDenseV1Config::default();
        
        // 创建输入嵌入
        let batch_size = 1;
        let seq_len = 3;
        let hidden_size = 4;
        let input_embeds = Tensor::randn(0f32, 1.0, (batch_size, seq_len, hidden_size), &device)?;
        
        // 创建缓存位置
        let cache_position = Tensor::arange(0u32, seq_len as u32, &device)?;
        
        // 创建模拟缓存
        let test_cache = TestCache {
            kv_length: 5,  // 缓存中有5个token
            kv_offset: 2,  // 偏移量为2
        };
        
        // 测试带缓存的因果掩码创建
        let mask = create_causal_mask(
            &config,
            &input_embeds,
            None,
            &cache_position,
            Some(&test_cache),
            None,
        )?;
        
        assert!(mask.is_some());
        let mask = mask.unwrap();
        
        // 检查掩码形状：q_length=3, kv_length=5
        assert_eq!(mask.dims(), &[batch_size, 1, seq_len, 5]);
        
        Ok(())
    }
}