# hy-mt

`hy-mt` 是一个基于 Rust 实现的轻量级机器翻译模型推理库，专门用于加载和运行腾讯混元（Hunyuan）MT系列模型。

>  ⚠️⚠️⚠️ 只做了模型的基本实现，精度方面还未对齐，所以目前的翻译效果尚未达到预期。

## 功能特点

- 使用 Rust 编写，确保高性能和内存安全性
- 基于 Candle 深度学习框架，支持 CPU 和 GPU 推理
- 支持腾讯混元 MT 系列模型（如 HY-MT1.5-1.8B 及其 FP8 量化版本）
- 提供简洁的文本生成接口

## 依赖

- Rust (edition 2024)
- Candle 框架 (`candle-core`, `candle-nn`)
- CUDA (可选，用于 GPU 加速)

## 安装

```bash
# 克隆项目
git clone https://github.com/kingzcheung/hy-mt
cd hy-mt

# 构建项目
cargo build --release
```

## 使用

### 示例代码

项目包含一个文本生成示例：

```rust
use candle_core::{Device, Result};
use hy_mt::TextGeneration;

fn main() -> Result<()> {
    // 设置设备
    let device = Device::cuda_if_available(0)?;
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tencent/HY-MT1.5-1.8B");

    let mut model = TextGeneration::from_pretrained(path, &device)?;

    // 生成文本
    let prompt = "Candle is a minimalist ML framework for Rust...";
    println!("输入提示: {}", prompt);
    println!("生成结果: ");
    let _ = model.generate(prompt, 2048)?;

    Ok(())
}
```

### 运行示例

```bash
cargo run --example text_generation_example
```

## 配置

为了运行模型，你需要下载腾讯混元 MT 系列模型权重文件并放置在正确的路径。

## 许可证

MIT