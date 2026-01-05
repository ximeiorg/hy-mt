use candle_core::{Device, Result};
use hy_mt::
    TextGeneration
;

fn main() -> Result<()> {
    // 设置设备
    let device = Device::Cpu; // 在CPU上运行，如果需要GPU，可以改为Device::Cuda(0)等
    // let path = env!("CARGO_MANIFEST_DIR") / "tencent/HY-MT1.5-1.8B";
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tencent/HY-MT1.5-1.8B");

    let mut model = TextGeneration::from_pretrained(path, &device)?;

    // 生成文本
    let prompt = "Translate the following segment into Chinese, without additional explanation.\n\nIt’s on the house.\n";
    println!("输入提示: {}", prompt);
    println!("生成结果: ");
    println!();
    let _ = model.generate(prompt, 100)?; // 生成最多100个token
    println!();

    Ok(())
}