//! Raw throughput benchmark: measures prefill and steady-state decode without
//! stopping at EOS.
//!
//! ```bash
//! cargo run --release --features metal --example bench -- <model-dir> [steps] [prompt-len]
//! ```

use std::time::Instant;

use candle_core::{DType, Tensor};
use hy_mt::Tokenizer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = std::env::args()
        .nth(1)
        .ok_or("usage: bench <model-dir> [steps] [extra-context-tokens]")?;
    let steps: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(32);
    let extra: usize = std::env::args()
        .nth(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);

    let t0 = Instant::now();
    let mut translator = hy_mt::Translator::builder(&dir).warmup(false).load()?;
    println!("load: {:.2}s", t0.elapsed().as_secs_f64());
    println!(
        "arch {} / {:?} / {:?}",
        translator.architecture(),
        translator.dtype(),
        translator.device()
    );

    let tokenizer = Tokenizer::from_dir(&dir)?;
    let prompt = "<｜hy_begin▁of▁sentence｜><｜hy_User｜>将以下文本翻译为 英语：今天天气真好。<｜hy_Assistant｜>";
    let mut ids = tokenizer.encode_with_markers(prompt)?;
    // Pad the context so that the attention part of the cost is visible.
    let pad = tokenizer.encode("The weather is nice. ", false)?;
    while ids.len() < 14 + extra {
        ids.extend_from_slice(&pad);
    }
    println!("prompt tokens: {}", ids.len());

    let device = translator.device().clone();
    let mut cache = translator.model().new_cache(ids.len() + steps + 8)?;

    // Warm-up (Metal shader compilation happens on first dispatch).
    let warm = Tensor::new(ids.as_slice(), &device)?.unsqueeze(0)?;
    let t0 = Instant::now();
    let logits = translator.model_mut().forward(&warm, &mut cache, true)?;
    println!("prefill ({} tok): {:.3}s", ids.len(), t0.elapsed().as_secs_f64());

    let mut last = logits
        .squeeze(0)?
        .squeeze(0)?
        .to_dtype(DType::F32)?
        .argmax(0)?
        .to_scalar::<u32>()?;

    let t0 = Instant::now();
    for i in 0..steps {
        let input = Tensor::new(&[last], &device)?.unsqueeze(0)?;
        let logits = translator.model_mut().forward(&input, &mut cache, true)?;
        if i + 1 < steps {
            last = logits
                .squeeze(0)?
                .squeeze(0)?
                .to_dtype(DType::F32)?
                .argmax(0)?
                .to_scalar::<u32>()?;
        }
    }
    let secs = t0.elapsed().as_secs_f64();
    println!(
        "decode: {} tokens in {:.2}s -> {:.2} tok/s ({:.1} ms/token), cache {:.1} MB",
        steps,
        secs,
        steps as f64 / secs,
        secs * 1000.0 / steps as f64,
        cache.allocated_bytes() as f64 / 1e6
    );
    Ok(())
}
