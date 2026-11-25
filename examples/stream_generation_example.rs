//! Streaming output with the original `hy-mt` iterator API.
//!
//! ```bash
//! cargo run --release --features metal --example stream_generation_example -- \
//!     /path/to/Hy-MT3-1.8B-FP8
//! ```

use std::io::Write;

use hy_mt::TextGeneration;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = std::env::args()
        .nth(1)
        .ok_or("usage: stream_generation_example <model-dir>")?;
    let device = hy_mt::default_device()?;
    let mut model = TextGeneration::from_pretrained(&model_path, &device)?;

    let prompt = "将以下文本翻译为英语，注意只需要输出翻译后的结果，不要额外解释：\n\n人工智能正在改变我们的工作方式。";
    print!("{prompt}\n\n=> ");
    std::io::stdout().flush()?;

    for delta in model.generate_stream_iter(prompt, 128) {
        print!("{delta}");
        std::io::stdout().flush()?;
    }
    println!();
    Ok(())
}
