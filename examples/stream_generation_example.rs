//! 流式文本生成示例
//! 
//! 这个示例展示了如何使用 hy-mt 库的迭代器流式生成功能

use std::io::Write as _;

use candle_core::{Device, Result};
use hy_mt::TextGeneration;

fn main() -> Result<()> {
    // 设置设备
    let device = Device::cuda_if_available(0)?;
    println!("设备: {:?}", &device);
    
    // 模型路径（需要提前下载模型权重）
    // 注意：这里需要替换为实际的模型路径
    let model_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tencent/HY-MT1.5-1.8B");
        
    
    // 加载模型
    let mut model = TextGeneration::from_pretrained(&model_path, &device)?;
    
    println!("\n=== 流式文本生成示例 ===");
    
    // 使用迭代器的流式生成
    // let prompt = "Are you sure you want to proceed?";
    let prompt = "Translate the following segment into Chinese without additional explanation.\n\nAre you sure you want to proceed?";
    println!("输入提示: {}", prompt);
    println!("生成结果: ");
    
    let mut full_result = String::new();
    for chunk in model.generate_stream_iter(prompt, 256) {
        print!("{}", chunk);
        std::io::stdout().flush()?;
        full_result.push_str(&chunk);
    }
    
    println!("\n\n完整结果: {}", full_result);
    
    Ok(())
}