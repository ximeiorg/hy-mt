//! Loading of `.safetensors` weights, including FP8 dequantisation.
//!
//! The Hy-MT3 `-FP8` checkpoints store each `Linear` weight as
//! `float8_e4m3fn` with a per-tensor `weight_scale` (and an `input_scale`
//! which is only relevant for the W8A8 reference implementation).  We
//! dequantise to f16/bf16/f32 at load time:
//!
//! ```text
//! w_fp32 = e4m3_table[byte] * weight_scale
//! ```
//!
//! This is the "weight-only" variant of the reference kernel. It is slightly
//! *more* accurate than the W8A8 reference (activations stay in f16) and it
//! makes every forward pass use plain dense matmuls.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use candle_core::{DType, Device, Tensor};
use half::{bf16, f16};

use crate::error::{Error, Result};
use crate::fp8::e4m3_table;
use crate::nn::Linear;
use crate::safetensors_file::{Entry, SafetensorsFile, SrcDtype};

/// Element types we produce on the CPU before moving to the device.
trait Cast: candle_core::WithDType + Copy {
    fn cast(v: f32) -> Self;
}

impl Cast for f32 {
    #[inline]
    fn cast(v: f32) -> Self {
        v
    }
}

impl Cast for f16 {
    #[inline]
    fn cast(v: f32) -> Self {
        f16::from_f32(v)
    }
}

impl Cast for bf16 {
    #[inline]
    fn cast(v: f32) -> Self {
        bf16::from_f32(v)
    }
}

/// Dequantisation factors attached to a weight tensor.
#[derive(Debug, Clone, Default)]
struct Scales {
    scalar: f32,
    rows: Option<Vec<f32>>,
    row_len: usize,
}

impl Scales {
    fn unit() -> Self {
        Self {
            scalar: 1.0,
            rows: None,
            row_len: 0,
        }
    }

    fn new(values: Vec<f32>, numel: usize) -> Self {
        if values.is_empty() {
            return Self::unit();
        }
        if values.len() == 1 {
            return Self {
                scalar: values[0],
                rows: None,
                row_len: 0,
            };
        }
        let row_len = (numel / values.len()).max(1);
        Self {
            scalar: 1.0,
            rows: Some(values),
            row_len,
        }
    }
}

#[inline]
fn read_e5m2(b: u8) -> f32 {
    // E5M2: 1 sign, 5 exp (bias 15), 2 mantissa, inf/nan supported.
    let sign = if b & 0x80 != 0 { -1.0f32 } else { 1.0 };
    let exp = ((b >> 2) & 0x1F) as i32;
    let mant = (b & 0x03) as f32;
    if exp == 0 {
        sign * (mant / 4.0) * 2f32.powi(-14)
    } else if exp == 0x1F {
        if mant == 0.0 {
            sign * f32::INFINITY
        } else {
            f32::NAN
        }
    } else {
        sign * (1.0 + mant / 4.0) * 2f32.powi(exp - 15)
    }
}

#[inline]
fn read_be(bytes: &[u8], i: usize, width: usize) -> u64 {
    let mut acc = 0u64;
    for k in 0..width {
        acc |= (bytes[i * width + k] as u64) << (8 * k);
    }
    acc
}

/// Converts `n` elements of `src` to `D`, multiplying by a single factor.
fn decode_scaled<D: Cast>(bytes: &[u8], src: SrcDtype, n: usize, scale: f32) -> Result<Vec<D>> {
    let needed = n * src.size_in_bytes();
    if bytes.len() < needed {
        return Err(Error::invalid_arg(format!(
            "tensor needs {needed} bytes but only {} are available",
            bytes.len()
        )));
    }
    let mut out = Vec::with_capacity(n);
    match src {
        SrcDtype::F32 => {
            for c in bytes.chunks_exact(4).take(n) {
                let v = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                out.push(D::cast(v * scale));
            }
        }
        SrcDtype::F16 => {
            for c in bytes.chunks_exact(2).take(n) {
                let v = f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32();
                out.push(D::cast(v * scale));
            }
        }
        SrcDtype::BF16 => {
            for c in bytes.chunks_exact(2).take(n) {
                let v = bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32();
                out.push(D::cast(v * scale));
            }
        }
        SrcDtype::F8E4M3 => {
            let lut = e4m3_table();
            for &b in bytes.iter().take(n) {
                out.push(D::cast(lut[b as usize] * scale));
            }
        }
        other => {
            // Rare paths (f64 / e5m2 / integers): go through f32.
            for i in 0..n {
                let v = match other {
                    SrcDtype::F64 => f64::from_bits(read_be(bytes, i, 8)) as f32,
                    SrcDtype::F8E5M2 => read_e5m2(bytes[i]),
                    SrcDtype::U8 => bytes[i] as f32,
                    SrcDtype::I8 => bytes[i] as i8 as f32,
                    SrcDtype::U16 => read_be(bytes, i, 2) as u16 as f32,
                    SrcDtype::I16 => read_be(bytes, i, 2) as u16 as i16 as f32,
                    SrcDtype::U32 => read_be(bytes, i, 4) as u32 as f32,
                    SrcDtype::I32 => read_be(bytes, i, 4) as u32 as i32 as f32,
                    SrcDtype::U64 => read_be(bytes, i, 8) as f32,
                    SrcDtype::I64 => read_be(bytes, i, 8) as i64 as f32,
                    SrcDtype::Bool => (bytes[i] != 0) as u8 as f32,
                    _ => unreachable!("handled above"),
                };
                out.push(D::cast(v * scale));
            }
        }
    }
    Ok(out)
}

/// Converts `n` elements applying one factor per contiguous row.
fn decode_per_row<D: Cast>(
    bytes: &[u8],
    src: SrcDtype,
    n: usize,
    scales: &Scales,
) -> Result<Vec<D>> {
    let flat: Vec<f32> = decode_scaled(bytes, src, n, 1.0)?;
    let rows = scales.rows.as_ref().expect("per-row scales required");
    let mut out = Vec::with_capacity(n);
    for (i, v) in flat.into_iter().enumerate() {
        let s = rows[(i / scales.row_len).min(rows.len() - 1)];
        out.push(D::cast(v * s));
    }
    Ok(out)
}

fn build<D: Cast>(
    bytes: &[u8],
    src: SrcDtype,
    n: usize,
    scales: &Scales,
    shape: &[usize],
) -> Result<Tensor> {
    let data = match &scales.rows {
        None => decode_scaled::<D>(bytes, src, n, scales.scalar)?,
        Some(_) => decode_per_row::<D>(bytes, src, n, scales)?,
    };
    Ok(Tensor::from_vec(data, shape.to_vec(), &Device::Cpu)?)
}

/// All safetensors files of a model directory, addressable by tensor name.
#[derive(Debug)]
pub struct WeightSource {
    dir: PathBuf,
    files: Vec<SafetensorsFile>,
    index: HashMap<String, usize>,
}

impl WeightSource {
    /// Opens every `*.safetensors` file of `dir` (sharded checkpoints work).
    pub fn open(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();
        let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.is_file()
                    && p.extension().and_then(|s| s.to_str()) == Some("safetensors")
            })
            .collect();
        paths.sort();
        if paths.is_empty() {
            return Err(Error::InvalidModelDir {
                dir: dir.display().to_string(),
                reason: "no *.safetensors file found".into(),
            });
        }
        let mut files: Vec<SafetensorsFile> = Vec::with_capacity(paths.len());
        let mut index = HashMap::new();
        for path in paths {
            let file = SafetensorsFile::open(&path)?;
            let file_idx = files.len();
            for name in file.names() {
                if let Some(prev) = index.insert(name.clone(), file_idx) {
                    return Err(Error::invalid_arg(format!(
                        "tensor '{name}' appears in both '{}' and '{}'",
                        files[prev].path().display(),
                        path.display()
                    )));
                }
            }
            files.push(file);
        }
        Ok(Self { dir: dir.to_path_buf(), files, index })
    }

    /// Model directory.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Whether the checkpoint contains `name`.
    pub fn contains(&self, name: &str) -> bool {
        self.index.contains_key(name)
    }

    /// Number of tensors.
    pub fn len(&self) -> usize {
        self.index.len()
    }

    /// Whether no tensor was loaded.
    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    fn locate(&self, name: &str) -> Option<(&SafetensorsFile, &Entry)> {
        let file = &self.files[*self.index.get(name)?];
        Some((file, file.entry(name)?))
    }

    /// Reads a tensor as `dst` dtype, applying `{name}_scale` when present.
    pub fn tensor(&self, name: &str, dst: DType, device: &Device) -> Result<Tensor> {
        let (file, entry) = self
            .locate(name)
            .ok_or_else(|| Error::MissingTensor(name.to_string()))?;
        let bytes = file.raw(entry)?;
        let n = entry.numel();
        let scales = match self.locate(&format!("{name}_scale")) {
            Some((f, e)) => {
                let values = decode_scaled::<f32>(f.raw(e)?, e.dtype, e.numel(), 1.0)?;
                Scales::new(values, n)
            }
            None => Scales::unit(),
        };
        let tensor = match dst {
            DType::F32 => build::<f32>(bytes, entry.dtype, n, &scales, &entry.shape)?,
            DType::F16 => build::<f16>(bytes, entry.dtype, n, &scales, &entry.shape)?,
            DType::BF16 => build::<bf16>(bytes, entry.dtype, n, &scales, &entry.shape)?,
            other => {
                return Err(Error::unsupported(format!(
                    "cannot load weights as {other:?}; use f32, f16 or bf16"
                )))
            }
        };
        Ok(tensor.to_device(device)?)
    }

    /// Reads a tensor without applying any `_scale` companion.
    pub fn tensor_raw(&self, name: &str, dst: DType, device: &Device) -> Result<Tensor> {
        let (file, entry) = self
            .locate(name)
            .ok_or_else(|| Error::MissingTensor(name.to_string()))?;
        let bytes = file.raw(entry)?;
        let n = entry.numel();
        let tensor = match dst {
            DType::F32 => build::<f32>(bytes, entry.dtype, n, &Scales::unit(), &entry.shape)?,
            DType::F16 => build::<f16>(bytes, entry.dtype, n, &Scales::unit(), &entry.shape)?,
            DType::BF16 => build::<bf16>(bytes, entry.dtype, n, &Scales::unit(), &entry.shape)?,
            other => {
                return Err(Error::unsupported(format!(
                    "cannot load weights as {other:?}; use f32, f16 or bf16"
                )))
            }
        };
        Ok(tensor.to_device(device)?)
    }

    /// Reads a scalar stored as any small tensor.
    pub fn scalar(&self, name: &str) -> Result<f32> {
        let values = self.vector(name)?;
        values
            .first()
            .copied()
            .ok_or_else(|| Error::invalid_arg(format!("tensor '{name}' is empty")))
    }

    /// Reads a tensor as a `Vec<f32>` on the host.
    pub fn vector(&self, name: &str) -> Result<Vec<f32>> {
        let (file, entry) = self
            .locate(name)
            .ok_or_else(|| Error::MissingTensor(name.to_string()))?;
        decode_scaled::<f32>(file.raw(entry)?, entry.dtype, entry.numel(), 1.0)
    }

    /// Shape of a tensor as declared in the safetensors header.
    pub fn shape(&self, name: &str) -> Result<&[usize]> {
        self.locate(name)
            .map(|(_, e)| e.shape.as_slice())
            .ok_or_else(|| Error::MissingTensor(name.to_string()))
    }

    /// Builds a bias-free (or biased) linear layer, transposing the weight so
    /// that the forward pass is a single `x @ w` matmul.
    pub fn linear(
        &self,
        prefix: &str,
        dst: DType,
        device: &Device,
        with_bias: bool,
    ) -> Result<Linear> {
        let weight_name = format!("{prefix}.weight");
        let shape = self.shape(&weight_name)?.to_vec();
        if shape.len() != 2 {
            return Err(Error::invalid_arg(format!(
                "'{weight_name}' should be 2-D, got {shape:?}"
            )));
        }
        let weight = self.tensor(&weight_name, dst, device)?;
        // (out, in) -> (in, out) so that forward is `x.matmul(w)`.
        let weight = weight.t()?.contiguous()?;
        let bias = if with_bias && self.contains(&format!("{prefix}.bias")) {
            Some(self.tensor(&format!("{prefix}.bias"), dst, device)?)
        } else {
            None
        };
        Ok(Linear::new(weight, bias))
    }
}
