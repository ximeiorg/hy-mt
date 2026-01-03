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
    let on_true = Tensor::new(on_true, on_false.device())?.broadcast_as(shape.dims())?;
    let m = mask.where_cond(&on_true, on_false)?;
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
            // attn_bias = attn_bias.masked_fill(&mask.logical_not()?, f32::NEG_INFINITY)?;
            attn_bias = attn_bias.masked_fill(&mask.where_cond(&mask.zeros_like()?, &mask.ones_like()?)?, f32::NEG_INFINITY)?;
        }

        if let Some(attn_mask) = attn_mask {
            if attn_mask.rank() > attn_bias.rank() {
                attn_bias = attn_bias.broadcast_as(attn_mask.shape())?;
            }
            if attn_mask.dtype() == DType::U8 {
                // bool
                attn_bias = attn_bias.masked_fill(&attn_mask.where_cond(&attn_mask.zeros_like()?, &attn_mask.ones_like()?)?, f32::NEG_INFINITY)?;
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
}
