use candle_core::{D, DType, Device, Result, Tensor};



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
    input_embeds: &Tensor,
    attention_mask: Option<&Tensor>,
    position_ids: Option<&Tensor>,
) ->Result<()> {


    sdpa_mask(batch_size, q_length, kv_length, is_causal, device)
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
}