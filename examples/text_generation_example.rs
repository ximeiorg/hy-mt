use candle_core::{Device, Result};
use hy_mt::
    TextGeneration
;

fn main() -> Result<()> {
    println!("=== 调试前向传播 ===");
    debug_forward_pass()?;
    
    println!("\n=== 生成测试 ===");
    // 设置设备
    let device = Device::Cpu; // 在CPU上运行，如果需要GPU，可以改为Device::Cuda(0)等
    // let path = env!("CARGO_MANIFEST_DIR") / "tencent/HY-MT1.5-1.8B";
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tencent/HY-MT1.5-1.8B");

    let mut model = TextGeneration::from_pretrained(path, &device)?;

    // 生成文本 - 使用简单的英文文本，让模型自动应用翻译模板
    let prompt = "It's on the house.";
    println!("输入提示: {}", prompt);
    println!("生成结果: ");
    println!();
    let _ = model.generate(prompt, 100)?; // 生成最多100个token
    println!();

    Ok(())
}

// 调试函数：检查前向传播输出
fn debug_forward_pass() -> Result<()> {
    println!("由于model字段是私有的，无法直接访问前向传播。");
    println!("将使用generate方法进行测试。");
    
    let device = Device::Cpu;
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tencent/HY-MT1.5-1.8B");

    let mut model = TextGeneration::from_pretrained(path, &device)?;

    // 使用与Python完全一致的输入tokens
    let rust_tokens = vec![120000, 120006, 26403, 252, 1873, 11538, 1079, 8251, 11, 2553, 5682, 12509, 412, 2171, 665, 377, 252, 5205, 13, 120026];
    
    println!("=== 调试前向传播 ===");
    println!("输入token IDs: {:?}", rust_tokens);
    println!("token数量: {}", rust_tokens.len());
    
    // 由于无法直接访问model，我们只能通过generate方法测试
    println!("\n=== 单步生成测试 ===");
    
    // 测试单步生成
    let prompt = "It's on the house.";
    println!("输入提示: {}", prompt);
    
    // 生成单个token来检查输出
    let result = model.generate(prompt, 1)?;
    println!("单步生成结果: {}", result);
    
    Ok(())
}