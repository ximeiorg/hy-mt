//! Small tensor helpers kept from `hy-mt` 0.1 (they are handy when writing a
//! custom model on top of this crate).
//!
//! The engine itself no longer uses most of these: RoPE goes through candle's
//! fused kernel and attention uses the stack-based key/value expansion, both of
//! which are considerably faster.

use candle_core::{D, DType, Result, Tensor};

/// `rotate_half` of the Llama/Hunyuan RoPE.
pub fn rotate_half(xs: &Tensor) -> Result<Tensor> {
    let last_dim = xs.dim(D::Minus1)?;
    let xs1 = xs.narrow(D::Minus1, 0, last_dim / 2)?;
    let xs2 = xs.narrow(D::Minus1, last_dim / 2, last_dim - last_dim / 2)?;
    Tensor::cat(&[&xs2.neg()?, &xs1], D::Minus1)
}

/// Applies RoPE to queries and keys.
///
/// `cos`/`sin` are shaped `(batch, seq, dim)`; `unsqueeze_dim` says where the
/// head dimension sits (`Some(1)` for `(batch, heads, seq, dim)`).
pub fn apply_rotary_pos_emb(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    unsqueeze_dim: Option<usize>,
) -> Result<(Tensor, Tensor)> {
    let (cos, sin) = match unsqueeze_dim {
        Some(dim) => (cos.unsqueeze(dim)?, sin.unsqueeze(dim)?),
        None => (cos.clone(), sin.clone()),
    };
    let q_embed = (q.broadcast_mul(&cos)? + rotate_half(q)?.broadcast_mul(&sin)?)?;
    let k_embed = (k.broadcast_mul(&cos)? + rotate_half(k)?.broadcast_mul(&sin)?)?;
    Ok((q_embed, k_embed))
}

/// Expands `num_kv_heads` heads to `num_heads` in HuggingFace's `repeat_kv`
/// order.
///
/// Uses `stack` (a handful of contiguous copies) instead of
/// `broadcast_as(..).contiguous()`: on Metal the broadcast path runs at about
/// 1 GB/s and dominates a long-context decode step.
pub fn repeat_kv(hidden_states: &Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(hidden_states.clone());
    }
    let (batch, n_kv_heads, seq_len, head_dim) = hidden_states.dims4()?;
    let copies: Vec<&Tensor> = vec![hidden_states; n_rep];
    let stacked = Tensor::stack(&copies, 2)?; // (b, kv, n_rep, seq, dim)
    stacked.reshape((batch, n_kv_heads * n_rep, seq_len, head_dim))
}

/// Applies the multiplicative repetition penalty to `logits`.
pub fn apply_repeat_penalty(logits: &Tensor, penalty: f32, context: &[u32]) -> Result<Tensor> {
    let device = logits.device();
    let mut logits = logits.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let mut seen = vec![false; logits.len()];
    for &token in context {
        let idx = token as usize;
        if idx < logits.len() && !seen[idx] {
            seen[idx] = true;
            let v = logits[idx];
            logits[idx] = if v > 0.0 { v / penalty } else { v * penalty };
        }
    }
    let len = logits.len();
    Tensor::from_vec(logits, len, device)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn repeat_kv_order_matches_huggingface() {
        // one head with values 1,2,3 repeated 2x -> [1,1,2,2] per position
        let x = Tensor::new(&[[[[1.0f32], [2.0], [3.0]]]], &Device::Cpu).unwrap();
        let out = repeat_kv(&x, 2).unwrap();
        assert_eq!(out.dims(), &[1, 2, 3, 1]);
        let v = out.squeeze(3).unwrap().squeeze(0).unwrap();
        assert_eq!(
            v.to_vec2::<f32>().unwrap(),
            vec![vec![1.0, 2.0, 3.0], vec![1.0, 2.0, 3.0]]
        );
    }

    #[test]
    fn repeat_kv_preserves_head_groups() {
        let x = Tensor::new(
            &[[[[1.0f32]], [[2.0]]]], // 1 batch, 2 kv heads, 1 pos, 1 dim
            &Device::Cpu,
        )
        .unwrap();
        let out = repeat_kv(&x, 2).unwrap();
        let v = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(v, vec![1.0, 1.0, 2.0, 2.0]);
    }

    #[test]
    fn penalty_pushes_seen_tokens() {
        let logits = Tensor::new(&[2.0f32, 1.9], &Device::Cpu).unwrap();
        let out = apply_repeat_penalty(&logits, 4.0, &[0]).unwrap();
        let v = out.to_vec1::<f32>().unwrap();
        assert!(v[1] > v[0]);
    }
}
