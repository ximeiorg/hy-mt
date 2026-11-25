//! A tiny memory-mapped `.safetensors` reader.
//!
//! candle can load safetensors files itself, but the FP8 variants of Hy-MT3
//! need a custom dequantisation step (`weight` * `weight_scale`), and reading
//! raw bytes also avoids materialising a 2 GB `HashMap<String, Tensor>` on the
//! device. The file is memory-mapped, so pages are only touched for tensors
//! that are actually loaded and can be reclaimed by the OS afterwards.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use memmap2::Mmap;

use crate::error::{Error, Result};

/// On-disk element type of a tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SrcDtype {
    F64,
    F32,
    F16,
    BF16,
    F8E4M3,
    F8E5M2,
    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
    U64,
    I64,
    Bool,
}

impl SrcDtype {
    /// Parses the `dtype` string used by the safetensors header.
    pub fn parse(s: &str) -> Result<Self> {
        Ok(match s {
            "F64" => Self::F64,
            "F32" => Self::F32,
            "F16" => Self::F16,
            "BF16" => Self::BF16,
            "F8_E4M3" | "F8E4M3" => Self::F8E4M3,
            "F8_E5M2" | "F8E5M2" => Self::F8E5M2,
            "U8" => Self::U8,
            "I8" => Self::I8,
            "U16" => Self::U16,
            "I16" => Self::I16,
            "U32" => Self::U32,
            "I32" => Self::I32,
            "U64" => Self::U64,
            "I64" => Self::I64,
            "BOOL" => Self::Bool,
            other => {
                return Err(Error::unsupported(format!(
                    "unsupported safetensors dtype '{other}'"
                )))
            }
        })
    }

    /// Size of one element in bytes.
    pub fn size_in_bytes(&self) -> usize {
        match self {
            Self::F64 | Self::U64 | Self::I64 => 8,
            Self::F32 | Self::U32 | Self::I32 => 4,
            Self::F16 | Self::BF16 | Self::U16 | Self::I16 => 2,
            Self::F8E4M3 | Self::F8E5M2 | Self::U8 | Self::I8 | Self::Bool => 1,
        }
    }
}

/// Location of one tensor inside its file.
#[derive(Debug, Clone)]
pub struct Entry {
    pub dtype: SrcDtype,
    pub shape: Vec<usize>,
    pub begin: usize,
    pub end: usize,
}

impl Entry {
    /// Number of elements.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

/// One memory-mapped safetensors file.
pub struct SafetensorsFile {
    path: PathBuf,
    mmap: Mmap,
    data_start: usize,
    entries: HashMap<String, Entry>,
}

impl std::fmt::Debug for SafetensorsFile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SafetensorsFile")
            .field("path", &self.path)
            .field("tensors", &self.entries.len())
            .finish()
    }
}

impl SafetensorsFile {
    /// Opens (and memory-maps) a `.safetensors` file.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = std::fs::File::open(&path)?;
        // SAFETY: the mapping is read-only and the file is not mutated by us.
        let mmap = unsafe { Mmap::map(&file)? };
        let bad = |reason: String| Error::InvalidSafetensors {
            path: path.display().to_string(),
            reason,
        };
        if mmap.len() < 8 {
            return Err(bad("file is smaller than the header length prefix".into()));
        }
        let header_len = u64::from_le_bytes(mmap[0..8].try_into().unwrap()) as usize;
        let data_start = 8 + header_len;
        if data_start > mmap.len() {
            return Err(bad(format!(
                "declared header length {header_len} exceeds the file size {}",
                mmap.len()
            )));
        }
        let header: serde_json::Value = serde_json::from_slice(&mmap[8..data_start])
            .map_err(|e| bad(format!("invalid JSON header: {e}")))?;
        let obj = header
            .as_object()
            .ok_or_else(|| bad("header is not a JSON object".into()))?;

        let mut entries = HashMap::with_capacity(obj.len());
        for (name, info) in obj {
            if name == "__metadata__" {
                continue;
            }
            let dtype = info
                .get("dtype")
                .and_then(|v| v.as_str())
                .ok_or_else(|| bad(format!("tensor '{name}' has no dtype")))?;
            let shape = info
                .get("shape")
                .and_then(|v| v.as_array())
                .ok_or_else(|| bad(format!("tensor '{name}' has no shape")))?;
            let shape: Vec<usize> = shape
                .iter()
                .map(|v| v.as_u64().unwrap_or(0) as usize)
                .collect();
            let offsets = info
                .get("data_offsets")
                .and_then(|v| v.as_array())
                .ok_or_else(|| bad(format!("tensor '{name}' has no data_offsets")))?;
            if offsets.len() != 2 {
                return Err(bad(format!("tensor '{name}' has malformed data_offsets")));
            }
            let begin = offsets[0].as_u64().unwrap_or(0) as usize;
            let end = offsets[1].as_u64().unwrap_or(0) as usize;
            entries.insert(
                name.clone(),
                Entry {
                    dtype: SrcDtype::parse(dtype)?,
                    shape,
                    begin,
                    end,
                },
            );
        }
        Ok(Self {
            path,
            mmap,
            data_start,
            entries,
        })
    }

    /// Path of the file.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// All tensor names in this file.
    pub fn names(&self) -> impl Iterator<Item = &String> {
        self.entries.keys()
    }

    /// Header entry for `name`.
    pub fn entry(&self, name: &str) -> Option<&Entry> {
        self.entries.get(name)
    }

    /// Raw little-endian bytes of `name`.
    pub fn raw(&self, entry: &Entry) -> Result<&[u8]> {
        let start = self.data_start + entry.begin;
        let end = self.data_start + entry.end;
        if end > self.mmap.len() || start > end {
            return Err(Error::InvalidSafetensors {
                path: self.path.display().to_string(),
                reason: format!("data range {start}..{end} out of bounds"),
            });
        }
        Ok(&self.mmap[start..end])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dtype_sizes() {
        assert_eq!(SrcDtype::parse("F8_E4M3").unwrap().size_in_bytes(), 1);
        assert_eq!(SrcDtype::parse("BF16").unwrap().size_in_bytes(), 2);
        assert_eq!(SrcDtype::parse("F32").unwrap().size_in_bytes(), 4);
        assert!(SrcDtype::parse("nope").is_err());
    }
}
