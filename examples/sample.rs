//! Minimal end-to-end sample: load, translate one sentence, print it.
//!
//! ```bash
//! cargo run --release --features metal --example sample -- <model-dir>
//! ```

use hy_mt::{Device, TextGeneration};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .ok_or("usage: sample <model-dir>")?;
    let device = hy_mt::default_device()?;
    println!("device: {device:?}");

    let mut model = TextGeneration::from_pretrained(std::path::Path::new(&path), &device)?;

    // Plain text is treated as user content and wrapped in the chat template.
    let prompt = "Translate the following segment into Chinese without additional explanation.\n\nIt's on the house.";
    let result = model.generate(prompt, 100)?;
    println!("结果: {}", result.trim());
    Ok(())
}

#[allow(dead_code)]
fn _explicit_cpu() -> hy_mt::Result<Device> {
    Ok(Device::Cpu)
}
