use candle_core::*;
use candle_nn::{Embedding, Linear, Module, VarBuilder, embedding, linear, linear_no_bias};

use crate::{
    config::HunYuanDenseV1Config,
    utils::{apply_rotary_pos_emb, repeat_kv, scaled_dot_product_attention},
};

#[derive(Debug, Clone)]
struct T5LayerNorm {
    weight: Tensor,
    variance_epsilon: f64,
    span: tracing::Span,
}

impl T5LayerNorm {
    fn load(hidden_size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(hidden_size, "weight")?;
        Ok(Self {
            weight,
            variance_epsilon: eps,
            span: tracing::span!(tracing::Level::TRACE, "layer-norm"),
        })
    }
}

impl Module for T5LayerNorm {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let dtype = xs.dtype();
        let xs_f32 = xs.to_dtype(DType::F32)?;
        // variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        let variance = xs_f32.sqr()?.mean_keepdim(D::Minus1)?;
        let xs = xs_f32.broadcast_div(&(variance + self.variance_epsilon)?.sqrt()?)?;
        let xs = xs.to_dtype(dtype)?;
        let xs = xs.broadcast_mul(&self.weight.to_dtype(dtype)?)?;
        Ok(xs)
    }
}

pub type HunYuanDenseV1RMSNorm = T5LayerNorm;

pub struct HunYuanDenseV1MLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl HunYuanDenseV1MLP {
    pub fn load(
        config: &HunYuanDenseV1Config,
        layer_idx: usize,
        is_share_mlp: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let gate_proj = linear_no_bias(
            config.hidden_size,
            config.intermediate_size,
            vb.pp("gate_proj"),
        )?;
        let up_proj = linear_no_bias(
            config.hidden_size,
            config.intermediate_size,
            vb.pp("up_proj"),
        )?;

        let down_proj = linear_no_bias(
            config.intermediate_size,
            config.hidden_size,
            vb.pp("down_proj"),
        )?;

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }
}

impl Module for HunYuanDenseV1MLP {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate_output = self.gate_proj.forward(xs)?.silu()?;
        let up_output = self.up_proj.forward(xs)?;
        let ys = gate_output.broadcast_mul(&up_output)?;

        self.down_proj.forward(&ys)
    }
}

fn eager_attention_forward(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    attention_mask: Option<&Tensor>,
    head_mask: Option<&Tensor>,
    output_attentions: bool,
    scaling: f64,
    dropout: f64, //,推理默认为0.0
) -> Result<Tensor> {
    // let key_states = repeat_kv(key, n_rep)
    unimplemented!("不走这条路径")
}

pub struct HunYuanDenseV1Attention {
    head_dim: usize,
    config: HunYuanDenseV1Config,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    query_layernorm: HunYuanDenseV1RMSNorm,
    key_layernorm: HunYuanDenseV1RMSNorm,
    scaling: f64,
    is_causal: bool,
}

impl HunYuanDenseV1Attention {
    pub fn load(config: &HunYuanDenseV1Config, layer_idx: usize, vb: VarBuilder) -> Result<Self> {
        let head_dim = config
            .head_dim
            .unwrap_or(config.hidden_size / config.num_key_value_heads);

        let num_key_value_groups = config.num_attention_heads / config.num_key_value_heads;

        // head_dim**-0.5
        let scaling = 1.0 / f64::sqrt(head_dim as f64);
        let is_causal = true;

        //attention_bias 默认直接写死为 false,看情况
        let q_proj = linear_no_bias(
            config.hidden_size,
            config.num_attention_heads * head_dim,
            vb.pp("q_proj"),
        )?;
        let k_proj = linear_no_bias(
            config.hidden_size,
            config.num_attention_heads * head_dim,
            vb.pp("k_proj"),
        )?;
        let v_proj = linear_no_bias(
            config.hidden_size,
            config.num_attention_heads * head_dim,
            vb.pp("v_proj"),
        )?;
        let o_proj = linear_no_bias(
            config.num_attention_heads * head_dim,
            config.hidden_size,
            vb.pp("o_proj"),
        )?;

        let query_layernorm =
            HunYuanDenseV1RMSNorm::load(head_dim, config.rms_norm_eps, vb.pp("query_layernorm"))?;
        let key_layernorm =
            HunYuanDenseV1RMSNorm::load(head_dim, config.rms_norm_eps, vb.pp("key_layernorm"))?;

        Ok(Self {
            head_dim,
            config: config.clone(),
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            query_layernorm,
            key_layernorm,
            scaling,
            is_causal,
        })
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        position_embeddings: (&Tensor, &Tensor),
        attention_mask: Option<&Tensor>,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let input_shape = hidden_states.dims();
        let hidden_shape = (
            input_shape[0],
            input_shape[1],
            self.config.num_attention_heads,
            self.head_dim,
        ); // (1, 1, -1, 128)

        let query_states = self
            .q_proj
            .forward(hidden_states)?
            .reshape(hidden_shape)?
            .transpose(1, 2)?;

        let key_states = self
            .k_proj
            .forward(hidden_states)?
            .reshape(hidden_shape)?
            .transpose(1, 2)?;

        let value_states = self
            .v_proj
            .forward(hidden_states)?
            .reshape(hidden_shape)?
            .transpose(1, 2)?;

        let (sin, cos) = position_embeddings;

        let (query_states, key_states) =
            apply_rotary_pos_emb(&query_states, &key_states, cos, sin, Some(1))?;

        let query_states = self.query_layernorm.forward(&query_states)?;
        let key_states = self.key_layernorm.forward(&key_states)?;

        let attn_output = scaled_dot_product_attention(
            &query_states,
            &key_states,
            &value_states,
            attention_mask,
            Some(0.0),
            Some(self.is_causal),
            Some(self.scaling),
        )?;
        // attn_weights is None

        let attn_output = attn_output.reshape(input_shape)?.contiguous()?;
        let attn_output = self.o_proj.forward(&attn_output)?;

        Ok((attn_output, None))
    }
}

pub struct HunYuanDenseV1DecoderLayer {
    self_attn: HunYuanDenseV1Attention,
    mlp: HunYuanDenseV1MLP,
    input_layernorm: HunYuanDenseV1RMSNorm,
    post_attention_layernorm: HunYuanDenseV1RMSNorm,
    layer_idx: usize,
}

impl HunYuanDenseV1DecoderLayer {
    pub fn load(config: &HunYuanDenseV1Config, layer_idx: usize, vb: VarBuilder) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let self_attn = HunYuanDenseV1Attention::load(config, layer_idx, vb.pp("self_attn"))?;
        let mlp = HunYuanDenseV1MLP::load(config, layer_idx, false, vb.pp("mlp"))?;
        let input_layernorm = HunYuanDenseV1RMSNorm::load(
            hidden_size,
            config.rms_norm_eps,
            vb.pp("input_layernorm"),
        )?;
        let post_attention_layernorm = HunYuanDenseV1RMSNorm::load(
            hidden_size,
            config.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
            layer_idx,
        })
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        position_embeddings: (&Tensor, &Tensor),
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let residual = hidden_states;
        let hidden_states = self.input_layernorm.forward(hidden_states)?;
        // self attention
        let (hidden_states, _) =
            self.self_attn
                .forward(&hidden_states, position_embeddings, attention_mask)?;

        let hidden_states = residual.broadcast_add(&hidden_states)?;
        // full connect
        let residual = &hidden_states;
        let hidden_states = self.post_attention_layernorm.forward(&hidden_states)?;
        let hidden_states = self.mlp.forward(&hidden_states)?;
        let hidden_states = residual.broadcast_add(&hidden_states)?;
        Ok(hidden_states)
    }
}

pub struct HunYuanDenseV1RotaryEmbedding {
    inv_freq: Tensor,
    attention_scaling: f32,
}

impl HunYuanDenseV1RotaryEmbedding {
    pub fn load(config: &HunYuanDenseV1Config) -> Result<Self> {
        let rope_theta = config.rope_theta;
        let alpha = config.rope_scaling.alpha;
        let base = rope_theta * alpha.powf((config.head_dim as f64) / ((config.head_dim as f64) - 2.0));
        let inv_freq = {
            let mut inv_freq_data = Vec::new();
            for i in (0..config.head_dim).step_by(2) {
                inv_freq_data.push(1.0 / (base.powf(i as f64 / config.head_dim as f64)));
            }
            Tensor::from_vec(inv_freq_data, (config.head_dim / 2,), &Device::Cpu)?
        };
        Ok(Self {
            inv_freq,
            attention_scaling: 1.0,
        })
    }

    pub fn forward(
        &self,
        xs: &Tensor,
        position_ids: &Tensor,
    ) ->Result<(Tensor, Tensor)> {
        //inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        let inv_freq_expanded = {
            let inv_freq = &self.inv_freq;
            let inv_freq_3d = if inv_freq.rank() == 1 {
                // [d] -> [1, d, 1]
                inv_freq.unsqueeze(0)?.unsqueeze(2)?
            } else if inv_freq.rank() == 2 {
                // [1, d] -> [1, d, 1]
                inv_freq.unsqueeze(2)?
            } else {
                inv_freq.clone()
            };
            // 转换为float32（确保与PyTorch一致）
            let inv_freq_f32 = inv_freq_3d.to_dtype(DType::F32)?;

            // 使用expand扩展到目标形状: [batch_size, rotary_dim/2, 1]
            let expanded =
                inv_freq_f32.expand((position_ids.dims()[0], self.inv_freq.dims()[1] / 2, 1))?;

            // 移动到与x相同的设备（通常不需要，但保持一致性）
            expanded.to_device(xs.device())
        }?;
        let position_ids_expanded = position_ids.unsqueeze(1)?.to_device(xs.device())?;

        let freqs = inv_freq_expanded
            .matmul(&position_ids_expanded.to_dtype(DType::F32)?)?
            .transpose(1, 2)?;
        let emb = Tensor::cat(&[&freqs, &freqs], D::Minus1)?;
        // cos = emb.cos() * self.attention_scaling
        let cos = emb
            .cos()?
            .broadcast_mul(&Tensor::new(&[self.attention_scaling], emb.device())?)?;
        let sin = emb
            .sin()?
            .broadcast_mul(&Tensor::new(&[self.attention_scaling], emb.device())?)?;

        Ok((cos, sin))
    }

}


pub struct HunYuanDenseV1Model {
    embed_tokens: Embedding,
    layers: Vec<HunYuanDenseV1DecoderLayer>,
    norm: HunYuanDenseV1RMSNorm,
    rotary_emb: HunYuanDenseV1RotaryEmbedding,
}

impl HunYuanDenseV1Model {
    pub fn load(config: &HunYuanDenseV1Config, vb: VarBuilder) -> Result<Self> {
        let embed_tokens = embedding(config.vocab_size, config.hidden_size, vb.pp("embed_tokens"))?; 

        let layers = {
            let mut layers = Vec::with_capacity(config.num_hidden_layers);
            for layer_idx in 0..config.num_hidden_layers {
                layers.push(HunYuanDenseV1DecoderLayer::load(
                    config,
                    layer_idx,
                    vb.pp(&format!("layers.{}", layer_idx)),
                )?);

            }
            layers
        };

        let norm = HunYuanDenseV1RMSNorm::load(
            config.hidden_size,
            config.rms_norm_eps,
            vb.pp("norm"),
        )?;
        let rotary_emb = HunYuanDenseV1RotaryEmbedding::load(config)?;
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            rotary_emb,
        })
    }

    pub fn forward(&self,input_ids: &Tensor,attention_mask: &Tensor,position_ids:&Tensor) -> Result<Tensor> {

        // candle_nn::rotary_emb::rope(xs, cos, sin)

        Ok(())
    }
}