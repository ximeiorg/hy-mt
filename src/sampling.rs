//! Token sampling (temperature, top-k, top-p, repetition penalty).

use crate::rng::Rng;

/// Sampler hyper-parameters.
///
/// The defaults follow the model card recommendation for Hy-MT3-1.8B/7B:
/// `temperature = 0.7, top_p = 0.6, top_k = 20, repetition_penalty = 1.05`.
#[derive(Debug, Clone)]
pub struct SamplingConfig {
    /// Softmax temperature. `<= 0` means greedy decoding.
    pub temperature: f64,
    /// Nucleus sampling threshold (`1.0` disables it).
    pub top_p: f64,
    /// Keep the `top_k` most likely tokens (`0` disables it).
    pub top_k: usize,
    /// Multiplicative penalty for tokens already in the context (`1.0` = off).
    pub repetition_penalty: f32,
    /// How many recent tokens the penalty looks at (`0` = whole history).
    pub repetition_window: usize,
    /// Fixed seed for reproducible output.
    pub seed: Option<u64>,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.6,
            top_k: 20,
            repetition_penalty: 1.05,
            repetition_window: 0,
            seed: None,
        }
    }
}

impl SamplingConfig {
    /// Greedy (deterministic) decoding.
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            repetition_penalty: 1.0,
            repetition_window: 0,
            seed: None,
        }
    }
}

/// Stateful sampler: keeps the RNG so that successive calls advance it.
#[derive(Debug)]
pub struct Sampler {
    cfg: SamplingConfig,
    rng: Rng,
}

impl Sampler {
    /// Creates a sampler, seeding the RNG from the config or the clock.
    pub fn new(cfg: SamplingConfig) -> Self {
        let rng = match cfg.seed {
            Some(seed) => Rng::new(seed),
            None => Rng::from_entropy(),
        };
        Self { cfg, rng }
    }

    /// The configuration in use.
    pub fn config(&self) -> &SamplingConfig {
        &self.cfg
    }

    /// Applies the repetition penalty in place.
    fn penalise(&self, logits: &mut [f32], history: &[u32]) {
        let penalty = self.cfg.repetition_penalty;
        if (penalty - 1.0).abs() < f32::EPSILON || history.is_empty() {
            return;
        }
        let start = if self.cfg.repetition_window == 0 {
            0
        } else {
            history.len().saturating_sub(self.cfg.repetition_window)
        };
        // `history` can be long; use a bitset-like marker to penalise once.
        let mut seen = vec![false; logits.len()];
        for &id in &history[start..] {
            let idx = id as usize;
            if idx < logits.len() && !seen[idx] {
                seen[idx] = true;
                let v = logits[idx];
                logits[idx] = if v > 0.0 { v / penalty } else { v * penalty };
            }
        }
    }

    /// Samples one token id from `logits`.
    ///
    /// `logits` is consumed (modified in place) and `history` is the list of
    /// token ids that should be penalised (prompt + generated).
    pub fn sample(&mut self, logits: &mut [f32], history: &[u32]) -> u32 {
        self.penalise(logits, history);

        if self.cfg.temperature <= 0.0 {
            return argmax(logits).unwrap_or(0) as u32;
        }

        let temperature = self.cfg.temperature as f32;
        let mut best = f32::NEG_INFINITY;
        for v in logits.iter() {
            if *v > best {
                best = *v;
            }
        }
        let mut probs: Vec<f32> = Vec::with_capacity(logits.len());
        let mut sum = 0f64;
        for v in logits.iter() {
            let p = ((v - best) / temperature).exp();
            sum += p as f64;
            probs.push(p);
        }
        if !(sum.is_finite() && sum > 0.0) {
            return argmax(logits).unwrap_or(0) as u32;
        }
        let inv = 1.0 / sum as f32;
        for p in probs.iter_mut() {
            *p *= inv;
        }

        // ---- top-k -------------------------------------------------------
        let mut candidates: Vec<usize> = (0..probs.len()).collect();
        if self.cfg.top_k > 0 && self.cfg.top_k < candidates.len() {
            let k = self.cfg.top_k;
            candidates.select_nth_unstable_by(k - 1, |&a, &b| {
                probs[b].partial_cmp(&probs[a]).unwrap_or(std::cmp::Ordering::Equal)
            });
            candidates.truncate(k);
        }
        candidates.retain(|&i| probs[i] > 0.0);
        if candidates.is_empty() {
            return argmax(logits).unwrap_or(0) as u32;
        }
        candidates.sort_unstable_by(|&a, &b| {
            probs[b].partial_cmp(&probs[a]).unwrap_or(std::cmp::Ordering::Equal)
        });

        // ---- top-p -------------------------------------------------------
        let mut keep = candidates;
        if self.cfg.top_p > 0.0 && self.cfg.top_p < 1.0 {
            let mut cum = 0f64;
            let mut cutoff = keep.len();
            for (pos, &idx) in keep.iter().enumerate() {
                cum += probs[idx] as f64;
                if cum >= self.cfg.top_p {
                    cutoff = pos + 1;
                    break;
                }
            }
            keep.truncate(cutoff.max(1));
        }

        // ---- draw --------------------------------------------------------
        let total: f64 = keep.iter().map(|&i| probs[i] as f64).sum();
        if !(total.is_finite() && total > 0.0) {
            return argmax(logits).unwrap_or(0) as u32;
        }
        let u = (self.rng.next_f32() as f64) * total;
        let mut acc = 0f64;
        for &idx in &keep {
            acc += probs[idx] as f64;
            if acc >= u {
                return idx as u32;
            }
        }
        *keep.last().unwrap() as u32
    }
}

/// Index of the largest value.
pub fn argmax(values: &[f32]) -> Option<usize> {
    let mut best: Option<(usize, f32)> = None;
    for (i, v) in values.iter().enumerate() {
        if v.is_nan() {
            continue;
        }
        match best {
            Some((_, b)) if *v <= b => {}
            _ => best = Some((i, *v)),
        }
    }
    best.map(|(i, _)| i)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_is_argmax() {
        let mut s = Sampler::new(SamplingConfig::greedy());
        let mut logits = vec![0.1, 3.0, 0.2, 2.9];
        assert_eq!(s.sample(&mut logits, &[]), 1);
    }

    #[test]
    fn top_k_restricts_support() {
        let cfg = SamplingConfig {
            temperature: 1.0,
            top_k: 1,
            top_p: 1.0,
            repetition_penalty: 1.0,
            ..Default::default()
        };
        let mut s = Sampler::new(cfg);
        for _ in 0..32 {
            let mut logits = vec![0.0, 5.0, 1.0, -2.0];
            assert_eq!(s.sample(&mut logits, &[]), 1);
        }
    }

    #[test]
    fn repetition_penalty_pushes_down_seen_tokens() {
        let cfg = SamplingConfig {
            temperature: 0.0,
            repetition_penalty: 4.0,
            ..SamplingConfig::greedy()
        };
        let mut s = Sampler::new(cfg);
        let mut logits = vec![2.0, 1.9];
        // Without history token 0 wins; after seeing it, token 1 should win.
        assert_eq!(s.sample(&mut logits.clone(), &[]), 0);
        assert_eq!(s.sample(&mut logits, &[0]), 1);
    }

    #[test]
    fn sampling_respects_distribution() {
        let cfg = SamplingConfig {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
            seed: Some(11),
            repetition_window: 0,
        };
        let mut s = Sampler::new(cfg);
        let mut counts = [0usize; 2];
        for _ in 0..4000 {
            let mut logits = vec![0.0, 1.0]; // p ≈ [0.269, 0.731]
            counts[s.sample(&mut logits, &[]) as usize] += 1;
        }
        let p0 = counts[0] as f64 / 4000.0;
        assert!((p0 - 0.269).abs() < 0.05, "p0 = {p0}");
    }
}
