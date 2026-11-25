//! Minimal FP8 (E4M3) support.
//!
//! The Hy-MT3 `*-FP8` checkpoints are stored with `compressed-tensors`
//! (`quant_method: "compressed-tensors"`, `format: "naive-quantized"`): every
//! `Linear` weight is a per-tensor scaled `float8_e4m3fn` tensor plus a
//! `weight_scale` f32 scalar (and an unused `input_scale`).
//!
//! We dequantise once at load time with the table below, which is also what
//! lets the rest of the model run on plain f16/bf16/f32 kernels.

use std::sync::OnceLock;

/// E4M3FN: 1 sign bit, 4 exponent bits (bias 7), 3 mantissa bits, no inf,
/// NaN at `0x7F`/`0xFF`, max finite value 448.
fn build_table() -> [f32; 256] {
    let mut table = [0f32; 256];
    for (i, slot) in table.iter_mut().enumerate() {
        let b = i as u8;
        let sign = if b & 0x80 != 0 { -1.0f32 } else { 1.0 };
        let exp = ((b >> 3) & 0x0F) as i32;
        let mant = (b & 0x07) as f32;
        let value = if exp == 0 {
            // subnormal: 2^-6 * mant/8
            sign * (mant / 8.0) * 2f32.powi(-6)
        } else if exp == 0x0F && (b & 0x07) == 0x07 {
            f32::NAN
        } else {
            sign * (1.0 + mant / 8.0) * 2f32.powi(exp - 7)
        };
        *slot = value;
    }
    table
}

/// Lookup table mapping every E4M3FN byte to its f32 value.
pub fn e4m3_table() -> &'static [f32; 256] {
    static TABLE: OnceLock<[f32; 256]> = OnceLock::new();
    TABLE.get_or_init(build_table)
}

/// Converts a single E4M3FN byte to `f32`.
#[inline]
pub fn e4m3_to_f32(byte: u8) -> f32 {
    e4m3_table()[byte as usize]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_values() {
        assert_eq!(e4m3_to_f32(0x00), 0.0);
        assert_eq!(e4m3_to_f32(0x38), 1.0);
        assert_eq!(e4m3_to_f32(0xB8), -1.0);
        assert_eq!(e4m3_to_f32(0x3C), 1.5);
        assert_eq!(e4m3_to_f32(0x40), 2.0);
        assert_eq!(e4m3_to_f32(0x01), 2f32.powi(-9)); // smallest subnormal
        assert_eq!(e4m3_to_f32(0x7E), 448.0); // largest finite
        assert!(e4m3_to_f32(0x7F).is_nan());
    }

    #[test]
    fn monotonic_in_positive_range() {
        let t = e4m3_table();
        for i in 1..0x7E {
            assert!(t[i] < t[i + 1], "not monotonic at {i:#x}");
        }
    }
}
