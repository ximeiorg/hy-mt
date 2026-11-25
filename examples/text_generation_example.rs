//! Original `hy-mt` style usage: load a checkpoint, generate text.
//!
//! ```bash
//! cargo run --release --features metal --example text_generation_example -- \
//!     /path/to/HY-MT1.5-1.8B-FP8
//! ```

use std::io::Write;

use hy_mt::{Device, TextGeneration};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = std::env::args()
        .nth(1)
        .ok_or("usage: text_generation_example <model-dir>")?;

    // Pick the fastest compiled-in backend (Metal → CUDA → CPU), or override
    // with HYMT_DEVICE=cpu|metal[:N]|cuda[:N].
    let device = match std::env::var(hy_mt::DEVICE_ENV) {
        Ok(_) => hy_mt::default_device()?,
        Err(_) => hy_mt::default_device()?,
    };
    println!("device: {device:?}");

    let mut model = TextGeneration::from_pretrained(&model_path, &device)?;
    println!(
        "architecture: {} | dtype: {:?} | chat format: {:?}",
        model.model().architecture(),
        model.model().dtype(),
        model.tokenizer().chat_format()
    );

    let prompts = [
        "将以下文本翻译为英语，注意只需要输出翻译后的结果，不要额外解释：\n\n今天天气真好。",
        "Translate the following text into Chinese without additional explanation:\n\nThe quick brown fox jumps over the lazy dog.",
    ];
    for prompt in prompts {
        print!("\n{}{prompt}\n\n=> ", "-".repeat(20));
        let answer = model.generate(prompt, 128)?;
        println!("{}", answer.trim());
        std::io::stdout().flush()?;
    }
    Ok(())
}

/// Silence the unused-import warning for `Device` in builds without a GPU.
#[allow(dead_code)]
fn _device_hint(d: Device) -> Device {
    d
}
