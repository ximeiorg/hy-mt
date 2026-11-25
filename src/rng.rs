//! A small, dependency-free PRNG used for token sampling.
//!
//! Sampling does not need cryptographic quality; xorshift64* is fast and its
//! statistical properties are far beyond what token sampling needs.

use std::time::{SystemTime, UNIX_EPOCH};

/// Deterministic (optionally seeded) random number generator.
#[derive(Debug, Clone)]
pub struct Rng {
    state: u64,
}

impl Rng {
    /// Creates a generator from an explicit seed.
    pub fn new(seed: u64) -> Self {
        // 0 is a fixed point of xorshift, never use it as state.
        Self {
            state: if seed == 0 { 0x9E37_79B9_7F4A_7C15 } else { seed },
        }
    }

    /// Creates a generator seeded from the wall clock.
    pub fn from_entropy() -> Self {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0x1234_5678_9ABC_DEF0);
        let tid = &nanos as *const u64 as u64;
        Self::new(nanos ^ tid.rotate_left(17))
    }

    /// Raw 64-bit xorshift64* step.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform sample in `[0, 1)` with 24 bits of precision.
    #[inline]
    pub fn next_f32(&mut self) -> f32 {
        let v = (self.next_u64() >> 40) as u32; // 24 bits
        v as f32 / (1u32 << 24) as f32
    }

    /// Uniform sample in `[0, n)`.
    #[inline]
    pub fn next_below(&mut self, n: usize) -> usize {
        if n <= 1 {
            0
        } else {
            (self.next_u64() % n as u64) as usize
        }
    }
}

impl Default for Rng {
    fn default() -> Self {
        Self::from_entropy()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_with_seed() {
        let mut a = Rng::new(42);
        let mut b = Rng::new(42);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn uniform_range() {
        let mut rng = Rng::new(7);
        let mut buckets = [0usize; 10];
        for _ in 0..20_000 {
            let v = rng.next_f32();
            assert!((0.0..1.0).contains(&v));
            buckets[(v * 10.0) as usize % 10] += 1;
        }
        for (i, c) in buckets.iter().enumerate() {
            assert!(*c > 1500, "bucket {i} too small: {c}");
        }
    }
}
