//! Multi-head / grouped-query attention with QK-RMSNorm and RoPE.

use candle_core::{DType, Device, Result as CandleResult, Tensor};

use crate::cache::KVCache;
use crate::config::Config;
use crate::error::Result;
use crate::nn::{Linear, RmsNorm};
use crate::rope::RotaryEmb;
use crate::weights::WeightSource;

/// Self-attention block of one decoder layer.
///
/// `q_proj` / `k_proj` / `v_proj` are fused into a single `(in, q + 2kv)` matrix
/// at load time: one bigger GEMM per token is measurably faster than three
/// narrow ones (fewer dispatches, better memory access), and it does not change
/// the math.
#[derive(Debug)]
pub struct Attention {
    qkv_proj: Linear,
    o_proj: Linear,
    q_norm: Option<RmsNorm>,
    k_norm: Option<RmsNorm>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    q_dim: usize,
    kv_dim: usize,
    scale: f64,
}

impl Attention {
    /// Loads `layers.{i}.self_attn.*` from the checkpoint.
    pub fn load(
        weights: &WeightSource,
        prefix: &str,
        cfg: &Config,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let attn = format!("{prefix}.self_attn");
        let q = weights.tensor(&format!("{attn}.q_proj.weight"), dtype, device)?;
        let k = weights.tensor(&format!("{attn}.k_proj.weight"), dtype, device)?;
        let v = weights.tensor(&format!("{attn}.v_proj.weight"), dtype, device)?;
        // (out, in) each -> cat rows -> (in, q + kv + kv) for a fused matmul.
        let qkv_w = Tensor::cat(&[&q, &k, &v], 0)?.t()?.contiguous()?;
        let qkv_b = if cfg.attention_bias {
            let qb = weights.tensor(&format!("{attn}.q_proj.bias"), dtype, device)?;
            let kb = weights.tensor(&format!("{attn}.k_proj.bias"), dtype, device)?;
            let vb = weights.tensor(&format!("{attn}.v_proj.bias"), dtype, device)?;
            Some(Tensor::cat(&[&qb, &kb, &vb], 0)?)
        } else {
            None
        };
        let qkv_proj = Linear::new(qkv_w, qkv_b);
        let o_proj = weights.linear(&format!("{attn}.o_proj"), dtype, device, cfg.attention_bias)?;
        let (q_norm, k_norm) = if cfg.use_qk_norm {
            let q = weights.tensor(
                &format!("{attn}.query_layernorm.weight"),
                dtype,
                device,
            )?;
            let k = weights.tensor(&format!("{attn}.key_layernorm.weight"), dtype, device)?;
            (
                Some(RmsNorm::new(q, cfg.rms_norm_eps)),
                Some(RmsNorm::new(k, cfg.rms_norm_eps)),
            )
        } else {
            (None, None)
        };
        Ok(Self {
            qkv_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads(),
            head_dim: cfg.head_dim(),
            q_dim: cfg.q_dim(),
            kv_dim: cfg.kv_dim(),
            scale: 1.0 / (cfg.head_dim() as f64).sqrt(),
        })
    }

    /// `x`: `(batch, seq, hidden)`; `cos`/`sin`: `(1, 1, seq, head_dim)`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        rope: &RotaryEmb,
        mask: Option<&Tensor>,
        cache: &mut KVCache,
        layer: usize,
    ) -> CandleResult<Tensor> {
        let (batch, seq_len, _) = x.dims3()?;
        let nh = self.num_heads;
        let nkv = self.num_kv_heads;
        let hd = self.head_dim;

        let qkv = self.qkv_proj.forward(x)?;
        let project = |offset: usize, heads: usize| -> CandleResult<Tensor> {
            qkv.narrow(2, offset, heads * hd)?
                .contiguous()?
                .reshape((batch, seq_len, heads, hd))?
                .transpose(1, 2)?
                .contiguous()
        };
        let q = project(0, nh)?;
        let k = project(self.q_dim, nkv)?;
        let v = project(self.q_dim + self.kv_dim, nkv)?;

        // RoPE first, then the per-head QK normalisation: this is the order
        // used by `HunYuanDenseV1Attention.forward`
        // (`apply_rotary_pos_emb` happens *before* `query_layernorm`).
        let q = rope.apply(&q, cos, sin)?;
        let k = rope.apply(&k, cos, sin)?;
        let q = match &self.q_norm {
            Some(norm) => norm.forward(&q)?,
            None => q,
        };
        let k = match &self.k_norm {
            Some(norm) => norm.forward(&k)?,
            None => k,
        };


        let (k, v) = cache.append(layer, &k, &v)?;

        let (k, v) = (repeat_kv(&k, nh / nkv)?, repeat_kv(&v, nh / nkv)?);

        // candle's matmul handles a strided (transposed) rhs directly, and
        // materialising it with `contiguous()` costs far more than the matmul.
        let attn = q.matmul(&k.transpose(2, 3)?)?;
        let attn = (attn * self.scale)?;
        let attn = match mask {
            Some(mask) => attn.broadcast_add(mask)?,
            None => attn,
        };
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        let out = attn.matmul(&v)?;
        let out = out
            .transpose(1, 2)?
            .contiguous()?
            .reshape((batch, seq_len, nh * hd))?;
        let out = self.o_proj.forward(&out)?;
        Ok(out)
    }
}

/// Expands `num_kv_heads` key/value heads to `num_heads` by repeating each
/// group, matching HuggingFace's `repeat_kv`.
///
/// Implemented with `stack` (= N contiguous copies) rather than
/// `broadcast_as(..).contiguous()`: on Metal the broadcast-then-contiguous path
/// runs at ~1 GB/s, which dominates a long-context decode step, while plain
/// copies run at memory speed.
fn repeat_kv(x: &Tensor, n_rep: usize) -> CandleResult<Tensor> {
    if n_rep == 1 {
        return Ok(x.clone());
    }
    let (b, nkv, t, hd) = x.dims4()?;
    let copies: Vec<&Tensor> = vec![x; n_rep];
    let stacked = Tensor::stack(&copies, 2)?; // (b, nkv, n_rep, t, hd)
    stacked.reshape((b, nkv * n_rep, t, hd))
}
