# hy-mt

[![Crates.io](https://img.shields.io/crates/v/hy-mt.svg)](https://crates.io/crates/hy-mt) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Rust](https://img.shields.io/badge/rust-2024-orange.svg)](https://www.rust-lang.org)

`hy-mt` 是一个基于 Rust 实现的轻量级机器翻译模型推理库，专门用于加载和运行腾讯混元（Hunyuan）MT系列模型。

> ⚠️ **注意**: 当前版本为实验性实现，模型精度尚未完全对齐，翻译效果和python 版本有一些小差异。

## 功能特点

- 🚀 **高性能**: 使用 Rust 编写，确保内存安全和高性能推理
- 🔥 **多设备支持**: 基于 Candle 框架，支持 CPU 和 CUDA GPU 推理
- 🤖 **模型兼容**: 支持腾讯混元 MT 系列模型（HY-MT1.5-1.8B 及其 FP8 量化版本）
- 📦 **简洁 API**: 提供易于使用的文本生成接口
- 🧩 **模块化设计**: 清晰的模块结构，易于扩展和维护

## 安装

### 作为依赖添加

在 `Cargo.toml` 中添加：

```toml
[dependencies]
hy-mt = "0.1.0"
```

### 从源码构建

```bash
# 克隆项目
git clone https://github.com/ximeiorg/hy-mt
cd hy-mt

# 构建项目
cargo build --release

# 运行测试
cargo test
```

## 快速开始

### 基本使用

```rust
use candle_core::{Device, Result};
use hy_mt::TextGeneration;

fn main() -> Result<()> {
    // 设置设备（优先使用GPU）
    let device = Device::cuda_if_available(0)?;
    
    // 加载预训练模型（需要提前下载模型权重）
    let model_path = std::path::Path::new("path/to/hunyuan/model");
    let mut model = TextGeneration::from_pretrained(model_path, &device)?;

    // 生成文本
    let prompt = "Translate the following English text to Chinese: Hello, world!";
    println!("输入提示: {}", prompt);
    println!("生成结果: ");
    
    let result = model.generate(prompt, 512)?;
    println!("\n完整结果: {}", result);

    Ok(())
}
```

### 高级配置

```rust
use hy_mt::{TextGeneration, model::HunYuanDenseV1ForCausalLM};

// 自定义生成参数
let mut model = TextGeneration::new_with_topk(
    model,           // 加载的模型
    tokenizer,       // tokenizer
    42,              // 随机种子
    Some(0.7),       // temperature
    Some(20),        // top-k
    Some(0.6),       // top-p
    1.1,             // 重复惩罚系数
    64,              // 重复惩罚窗口大小
    &device,
)?;
```

## 项目结构

```
src/
├── lib.rs          # 主库文件，导出公共API
├── config.rs       # 模型配置结构
├── model.rs        # 模型架构实现
├── tokenizer.rs    # Tokenizer封装
├── prompt.rs       # 提示模板处理
├── logits_processor.rs # 生成策略实现
├── masking_utils.rs    # 注意力掩码工具
└── utils.rs        # 工具函数
```

## 示例

项目包含完整的文本生成示例：

```bash
# 运行文本生成示例
cargo run --example text_generation_example

# 查看CUDA支持示例
cargo run --example cuda

# 运行简单示例
cargo run --example sample
```

## 模型准备

要使用此库，你需要：

1. 下载腾讯混元 MT 系列模型权重文件
2. 将模型文件放置在项目目录下的 `tencent/HY-MT1.5-1.8B/` 路径中
3. 确保包含以下文件：
   - `config.json` - 模型配置文件
   - `tokenizer.json` - Tokenizer文件
   - `model.safetensors` - 模型权重文件

## 依赖

- **Rust**: edition 2024
- **Candle**: 深度学习框架 (`candle-core`, `candle-nn`)
- **CUDA**: 可选，用于 GPU 加速
- **tokenizers**: Tokenizer处理库

## 开发状态

- ✅ 基础模型加载和推理
- ✅ CPU/GPU 多设备支持
- ✅ 文本生成功能
- 🔄 模型精度对齐（进行中）
- 🔄 性能优化（进行中）
- 🔄 更多模型支持（计划中）

## 贡献

欢迎提交 Issue 和 Pull Request！

## 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。