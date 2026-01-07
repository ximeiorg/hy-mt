use candle_core::{Device, Result};
use hy_mt::TextGeneration;

fn main() -> Result<()> {
    
    // 设置设备
    let device = Device::cuda_if_available(0)?; // 在CPU上运行，如果需要GPU，可以改为Device::Cuda(0)等
    println!("设备: {:?}", &device);
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tencent/HY-MT1.5-1.8B");

    let mut model = TextGeneration::from_pretrained(path, &device)?;

    // 生成文本 - 使用与Python版本相同的聊天模板格式
    // 只需要提供源文本，翻译指令会在generate函数中自动添加
    let prompt = "This pull request updates documentation to improve clarity and add support for new features, particularly around audio extraction for video datasets and the model quickstart matrix. It also synchronizes documentation with recent code changes, such as updated line numbers in webhook references.";
    println!("输入提示: {}", prompt);
    println!("生成结果: ");
    println!();
    let _ = model.generate(prompt, 2048)?; // 生成最多2048个token，与Python版本对齐
    println!();

    Ok(())
}