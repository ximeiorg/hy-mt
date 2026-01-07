use candle_core::{D, DType, Result, Tensor};
use candle_nn::ops::{dropout, softmax_last_dim};

pub(crate) fn rotate_half(xs: &Tensor) -> Result<Tensor> {
    let last_dim = xs.dim(D::Minus1)?;
    let xs1 = xs.narrow(D::Minus1, 0, last_dim / 2)?;
    let xs2 = xs.narrow(D::Minus1, last_dim / 2, last_dim - last_dim / 2)?;
    Tensor::cat(&[&xs2.neg()?, &xs1], D::Minus1)
}

pub(crate) fn apply_rotary_pos_emb(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    unsqueeze_dim: Option<usize>, // default 1
) -> Result<(Tensor, Tensor)> {
    let cos = if let Some(dim) = unsqueeze_dim {
        cos.unsqueeze(dim)?
    } else {
        cos.clone()
    };

    let sin = if let Some(dim) = unsqueeze_dim {
        sin.unsqueeze(dim)?
    } else {
        sin.clone()
    };

    let q_embed = (q.broadcast_mul(&cos)? + rotate_half(q)?.broadcast_mul(&sin)?)?;
    let k_embed = (k.broadcast_mul(&cos)? + rotate_half(k)?.broadcast_mul(&sin)?)?;
    Ok((q_embed, k_embed))
}

pub(crate) fn repeat_kv(hidden_states: &Tensor, n_rep: usize) -> Result<Tensor> {
    let (batch, num_key_value_heads, seq_len, head_dim) = hidden_states.dims4()?;

    if n_rep == 1 {
        return Ok(hidden_states.clone());
    }

    // hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)

    let hidden_states = hidden_states.unsqueeze(2)?.expand((
        batch,
        num_key_value_heads,
        n_rep,
        seq_len,
        head_dim,
    ))?;

    hidden_states.reshape((batch, num_key_value_heads * n_rep, seq_len, head_dim))
}

pub fn apply_repeat_penalty(logits: &Tensor, penalty: f32, context: &[u32]) -> Result<Tensor> {
    let device = logits.device();
    let mut logits = logits.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    let mut already_seen = std::collections::HashSet::new();
    for token_id in context {
        if already_seen.contains(token_id) {
            continue;
        }
        already_seen.insert(token_id);
        if let Some(logit) = logits.get_mut(*token_id as usize) {
            if *logit >= 0. {
                *logit /= penalty
            } else {
                *logit *= penalty
            }
        }
    }
    let logits_len = logits.len();
    Tensor::from_vec(logits, logits_len, device)
}

fn apply_triangular(xs: &Tensor, diagonal: isize, upper: bool) -> Result<Tensor> {
        let device = xs.device();
        let (l, s) = xs.dims2()?;
        let mut xs_tri = vec![];
        for i in 0..l as isize {
            for j in 0..s as isize {
                let cond = if upper {
                    i + diagonal > j
                } else {
                    i + diagonal < j
                };
                xs_tri.push(if cond { 0u8 } else { 1u8 });
            }
        }
        xs * Tensor::from_vec(xs_tri, (l, s), device)?.to_dtype(xs.dtype())?
}

pub trait Triangular {
    fn tril(&self, diagonal: isize) -> Result<Tensor>;
}
impl Triangular for Tensor {
    fn tril(&self, diagonal: isize) -> Result<Tensor> {
        apply_triangular(self, diagonal, false)
    }
}

pub trait MaskedFill {
    fn masked_fill(&self, mask: &Tensor, value: f32) -> Result<Tensor>;
}
impl MaskedFill for Tensor {
    fn masked_fill(&self, mask: &Tensor, value: f32) -> Result<Tensor> {
        masked_fill(self, mask, value)
    }
}



fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: f32) -> Result<Tensor> {
    let shape = mask.shape();
    // 修复：使用正确的方法创建包含特定值的张量
    let on_true_tensor = Tensor::new(on_true, on_false.device())?.to_dtype(on_false.dtype())?.broadcast_as(shape.dims())?;
    let m = mask.where_cond(&on_true_tensor, on_false)?;
    Ok(m)
}

pub fn scaled_dot_product_attention(
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        attn_mask: Option<&Tensor>,
        dropout_p: Option<f32>,
        is_causal: Option<bool>,
        scale: Option<f64>,
    ) -> Result<Tensor> {
        let device = query.device();
        let l = query.dim(D::Minus2)?;
        let s = key.dim(D::Minus2)?;
        let dim = query.dim(D::Minus1)?;

        let scale_factor = if let Some(scale) = scale {
            scale
        } else {
            1.0 / (dim as f64).sqrt()
        };

        let mut attn_bias = Tensor::zeros((l, s), query.dtype(), device)?;

        if matches!(is_causal, Some(true)) {
            assert!(attn_mask.is_none(), "scaled_dot_product_attention: Explicit attn_mask should not be set when is_causal=True");
            let mask = Tensor::ones((l, s), DType::U8, device)?.tril(0)?;
            
            // 直接使用mask.eq(0)来找到上三角位置（需要被屏蔽的位置）
            let bool_mask = mask.eq(&Tensor::zeros((l, s), DType::U8, device)?)?;
            attn_bias = attn_bias.masked_fill(&bool_mask, f32::NEG_INFINITY)?;
        }

        if let Some(attn_mask) = attn_mask {
            if attn_mask.rank() > attn_bias.rank() {
                attn_bias = attn_bias.broadcast_as(attn_mask.shape())?;
            }
            if attn_mask.dtype() == DType::U8 {
                // bool
                // 修复：将U8掩码转换为布尔掩码
                let bool_mask = attn_mask.eq(&Tensor::ones(attn_mask.shape(), DType::U8, device)?)?;
                attn_bias = attn_bias.masked_fill(&bool_mask, f32::NEG_INFINITY)?;
            } else {
                attn_bias = (&attn_bias
                    + attn_mask
                        .to_dtype(attn_bias.dtype())?
                        .broadcast_as(attn_bias.shape())?)?;
            }
        }

        let mut attn_weights =
            (query.matmul(&key.transpose(D::Minus2, D::Minus1)?.contiguous()?)? * scale_factor)?;

        attn_weights = (&attn_weights + attn_bias.broadcast_as(attn_weights.shape())?)?;
        attn_weights = softmax_last_dim(&attn_weights)?;

        if let Some(drop_p) = dropout_p {
            attn_weights = dropout(&attn_weights, drop_p)?;
        }
        attn_weights.matmul(value)
    }
mod tests {

    use candle_core::Device;

    use super::*;

    #[test]
    fn test_repeat_kv() {
        let array = [1.].repeat(18);
        let hidden_states = Tensor::from_slice(&array, (2, 3, 3), &Device::Cpu).unwrap();
        let hidden_states = hidden_states.unsqueeze(0).unwrap();
        println!("hidden_states: {:?}", hidden_states);
        println!("{hidden_states}");

        let hidden_states = repeat_kv(&hidden_states, 2).unwrap();

        //结果是[1,4,3,3]
        println!("res: {hidden_states}");
        assert_eq!(hidden_states.dims(), [1, 4, 3, 3]);
    }

    #[test]
    fn test_scaled_dot_product_attention_shape_compatibility() {
        let device = Device::Cpu;
        
        // 模拟HunYuanDenseV1Attention中的形状：query有16个头，key/value有4个头
        // 经过repeat_kv后，key/value应该变成16个头
        let batch_size = 1;
        let seq_len = 18;
        let num_query_heads = 16;
        let num_kv_heads = 4;
        let head_dim = 128;
        
        // 创建query张量 [1, 16, 18, 128]
        let query_shape = (batch_size, num_query_heads, seq_len, head_dim);
        let query = Tensor::randn(0.0, 1.0, query_shape, &device).unwrap();
        
        // 创建key张量 [1, 4, 18, 128]，然后重复为16个头
        let key_original_shape = (batch_size, num_kv_heads, seq_len, head_dim);
        let key_original = Tensor::randn(0.0, 1.0, key_original_shape, &device).unwrap();
        let key = repeat_kv(&key_original, num_query_heads / num_kv_heads).unwrap();
        
        // 创建value张量 [1, 4, 18, 128]，然后重复为16个头
        let value_original_shape = (batch_size, num_kv_heads, seq_len, head_dim);
        let value_original = Tensor::randn(0.0, 1.0, value_original_shape, &device).unwrap();
        let value = repeat_kv(&value_original, num_query_heads / num_kv_heads).unwrap();
        
        println!("query shape: {:?}", query.dims());
        println!("key shape: {:?}", key.dims());
        println!("value shape: {:?}", value.dims());
        
        // 验证形状匹配
        assert_eq!(query.dims(), key.dims());
        assert_eq!(query.dims(), value.dims());
        
        // 测试scaled_dot_product_attention函数
        let result = scaled_dot_product_attention(
            &query,
            &key,
            &value,
            None,
            Some(0.0),
            Some(true),
            None,
        );
        
        match &result {
            Ok(_) => println!("scaled_dot_product_attention succeeded"),
            Err(e) => println!("scaled_dot_product_attention failed: {}", e),
        }
        
        assert!(result.is_ok(), "scaled_dot_product_attention should succeed with compatible shapes");
        
        let output = result.unwrap();
        println!("output shape: {:?}", output.dims());
        
        // 输出形状应该是 [1, 16, 18, 128]
        assert_eq!(output.dims(), &[query_shape.0, query_shape.1, query_shape.2, query_shape.3]);
    }

    #[test]
    fn test_scaled_dot_product_attention_shape_mismatch() {
        let device = Device::Cpu;
        
        // 模拟不匹配的形状：query有16个头，key只有4个头（没有重复）
        let batch_size = 1;
        let seq_len = 18;
        let num_query_heads = 16;
        let num_kv_heads = 4;
        let head_dim = 128;
        
        // 创建query张量 [1, 16, 18, 128]
        let query_shape = (batch_size, num_query_heads, seq_len, head_dim);
        let query = Tensor::randn(0.0, 1.0, query_shape, &device).unwrap();
        
        // 创建key张量 [1, 4, 18, 128]（没有重复，不匹配）
        let key_shape = (batch_size, num_kv_heads, seq_len, head_dim);
        let key = Tensor::randn(0.0, 1.0, key_shape, &device).unwrap();
        
        // 创建value张量 [1, 4, 18, 128]（没有重复，不匹配）
        let value_shape = (batch_size, num_kv_heads, seq_len, head_dim);
        let value = Tensor::randn(0.0, 1.0, value_shape, &device).unwrap();
        
        println!("query shape: {:?}", query.dims());
        println!("key shape: {:?}", key.dims());
        println!("value shape: {:?}", value.dims());
        
        // 测试scaled_dot_product_attention函数，应该失败
        let result = scaled_dot_product_attention(
            &query,
            &key,
            &value,
            None,
            Some(0.0),
            Some(true),
            None,
        );
        
        // 这个测试应该失败，因为形状不匹配
        assert!(result.is_err(), "scaled_dot_product_attention should fail with incompatible shapes");
    }
}