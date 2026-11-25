//! Key/value cache for autoregressive decoding.
//!
//! Each layer keeps its valid prefix as a `(1, kv_heads, len, head_dim)` tensor
//! and appends with `cat`. That is one dispatch per tensor per step instead of
//! the dozen custom ops that a pre-allocated buffer + `slice_assign` needs, and
//! it keeps memory proportional to the real context instead of to `capacity`
//! (copying the prefix costs `len * head_dim * 2` bytes per token and layer,
//! i.e. a few MB for the whole of a 4k-token generation).

use candle_core::{DType, Device, Tensor};

use crate::error::{Error, Result};

/// Per-layer key/value cache.
#[derive(Debug)]
pub struct KVCache {
    k: Vec<Option<Tensor>>,
    v: Vec<Option<Tensor>>,
    len: usize,
    capacity: usize,
    num_kv_heads: usize,
    head_dim: usize,
    dtype: DType,
    device: Device,
}

impl KVCache {
    /// Creates an empty cache that can hold up to `capacity` tokens.
    pub fn new(
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        capacity: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        if capacity == 0 {
            return Err(Error::invalid_arg("kv cache capacity must be > 0"));
        }
        Ok(Self {
            k: vec![None; num_layers],
            v: vec![None; num_layers],
            len: 0,
            capacity,
            num_kv_heads,
            head_dim,
            dtype,
            device: device.clone(),
        })
    }

    /// Number of cached tokens.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Maximum number of tokens the cache accepts.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Free slots left.
    pub fn remaining(&self) -> usize {
        self.capacity.saturating_sub(self.len)
    }

    /// Element type of the buffers.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Device of the buffers.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Forgets every cached token.
    pub fn reset(&mut self) {
        for slot in self.k.iter_mut().chain(self.v.iter_mut()) {
            *slot = None;
        }
        self.len = 0;
    }

    /// Appends the keys/values of one layer and returns the whole prefix
    /// `(1, kv_heads, len + t, head_dim)`.
    ///
    /// `k` and `v` must be shaped `(1, kv_heads, t, head_dim)`.
    pub(crate) fn append(
        &mut self,
        layer: usize,
        k: &Tensor,
        v: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let t = k.dim(2)?;
        if self.len + t > self.capacity {
            return Err(Error::invalid_arg(format!(
                "kv cache overflow: {} cached + {} new > capacity {}",
                self.len, t, self.capacity
            )));
        }
        let (k_all, v_all) = match (&self.k[layer], &self.v[layer]) {
            (Some(prev_k), Some(prev_v)) => (
                Tensor::cat(&[prev_k, k], 2)?,
                Tensor::cat(&[prev_v, v], 2)?,
            ),
            _ => (k.clone(), v.clone()),
        };
        self.k[layer] = Some(k_all.clone());
        self.v[layer] = Some(v_all.clone());
        Ok((k_all, v_all))
    }

    /// Marks `t` more tokens as cached (called once per forward pass, after
    /// every layer appended its keys/values).
    pub(crate) fn advance(&mut self, t: usize) {
        self.len += t;
    }

    /// Number of heads per key/value.
    pub fn num_kv_heads(&self) -> usize {
        self.num_kv_heads
    }

    /// Dimension of one head.
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Bytes currently held by the cache.
    pub fn allocated_bytes(&self) -> usize {
        let row = self.num_kv_heads * self.head_dim * self.dtype.size_in_bytes();
        self.len * row * 2 * self.k.len()
    }
}
