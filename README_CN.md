# hy-mt

[![Crates.io](https://img.shields.io/crates/v/hy-mt.svg)](https://crates.io/crates/hy-mt) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Rust](https://img.shields.io/badge/rust-1.82%2B-orange.svg)](https://www.rust-lang.org)

[English](README.md) | **中文**

基于 [candle](https://github.com/huggingface/candle) 的**腾讯混元 MT 翻译模型** Rust 推理库。
一套引擎覆盖整个 dense 家族（HY-MT1.5 / Hy-MT2 / Hy-MT3，1.8B / 7B，bf16 / FP8），
无需 Python、PyTorch、transformers。

* 直接加载官方 `*-FP8`（compressed-tensors）与 bf16 权重，加载时反量化到 f16/bf16/f32
* Apple Metal、CUDA、CPU（Accelerate / MKL）后端，feature 可叠加、运行时选设备
* 两套 chat 模板族自适应：1.8B 的 `<｜hy_…｜>` 与 7B 的 `<|startoftext|>` / `<|extra_N|>`
* KV cache + 增量 detokenizer，流式输出不会切断多字节字符
* 模型卡给出的 7 类翻译 prompt（术语、风格、个性化、分隔符、结构化数据、背景信息…）中英双语
* 保留 0.1 的 `TextGeneration` API，老代码只需升级版本号

## 支持的模型

| checkpoint | 架构 | 词表 / 模板族 | 状态 |
|---|---|---|---|
| `HY-MT1.5-1.8B` / `-FP8` | dense | 120 818 / `Hy18B` | ✅ 端到端实测通过 |
| `Hy-MT2-1.8B` / `-FP8` | dense | 120 818 / `Hy18B` | ✅ 端到端实测通过 |
| `Hy-MT3-1.8B` / `-FP8` | dense | 120 818 / `Hy18B` | ✅ 端到端实测通过 |
| `Hunyuan-MT-7B`、`HY-MT1.5-7B`、`Hy-MT3-7B`(+FP8) | dense | 128k / `Hunyuan7B` | ✅ 代码路径就绪 |
| `Hy-MT3-30B-A3B`(+FP8) | **MoE** | 128k / `Hunyuan7B` | ⛔ 未实现 |

一切由 `config.json` 驱动（`hunyuan_v1_dense`），所以同架构的版本**不需要改代码**，
换模型目录即可。1.8B 与 7B 只差规模、词表与模板族；FP8 与 bf16 只差权重的存储方式；
`Hy-MT2-1.8B` 与前后两代只差一个配置字段名（`torch_dtype`，已用 serde 别名兼容），
`chat_template.jinja` 逐字节相同。

同一引擎跑三代 1.8B（贪心解码，同一批句子）：

| 源文 | HY-MT1.5 | Hy-MT2 | Hy-MT3 |
|---|---|---|---|
| 今天天气真好，我们一起去公园散步吧。(→en) | It's a beautiful day today. Let's go for a walk in the park together. | The weather is really nice today. Let's go for a walk in the park together. | The weather is really nice today; let's go for a walk in the park together. |
| The committee approved the budget after a lengthy debate. (→zh) | 在长时间的讨论之后，委员会最终批准了这份预算方案。 | 经过长时间的讨论，委员会通过了预算。 | 委员会经过长时间辩论后批准了预算。 |
| 混元大模型支持 102 种语言的互译。(→en) | The Hunyuan Large Model supports translation between 102 languages. | Hunyuan large model supports translation between 102 languages. | The Hunyuan large model supports mutual translation in 102 languages. |

## 安装

```toml
[dependencies]
hy-mt = "0.2"

# 按需选一个后端（feature 可叠加，设备在运行时挑）
# hy-mt = { version = "0.2", features = ["metal"] }
# hy-mt = { version = "0.2", features = ["cuda"] }
```

构建本仓库：

```bash
cargo build --release --features metal      # Apple GPU（推荐，f16）
cargo build --release                       # CPU（f32）
cargo build --release --features "metal,accelerate"
```

## 快速开始

### 1. 翻译 API（推荐）

```rust
use hy_mt::{TranslationRequest, Translator};

let mut translator = Translator::from_pretrained("/models/Hy-MT3-1.8B-FP8")?;

// 默认模板
println!("{}", translator.translate("今天天气真好。", "en")?.text);

// 带术语 / 风格约束
let req = TranslationRequest::new("The Hunyuan-MT3 model runs on Apple silicon.", "zh")
    .term("Hunyuan-MT3", "混元 MT3")
    .term("Apple silicon", "苹果芯片")
    .style("正式");
println!("{}", translator.translate_with(&req, None)?.text);

// 流式
translator.translate_stream(&req, None, |delta| print!("{delta}"))?;
```

设备与精度：

```rust
use hy_mt::{DType, Device, Translator};

let t = Translator::builder(dir)
    .device(Device::Cpu)          // 或 Device::new_metal(0)? / Device::new_cuda(0)?
    .dtype(DType::F16)
    .verbose(true)
    .load()?;
```

选择顺序：显式 `device(..)` > `HYMT_DEVICE=cpu|metal[:N]|cuda[:N]` > 自动（Metal → CUDA → CPU）。
默认权重 dtype 随设备：Metal→f16、CUDA→bf16、CPU→f32。

### 2. 原始生成 API（0.1 兼容）

```rust
use hy_mt::{Device, TextGeneration};

let mut model = TextGeneration::from_pretrained("/models/HY-MT1.5-1.8B-FP8", &Device::Cpu)?;

// 一次性
let answer = model.generate("将以下文本翻译为英语：你好，世界。", 256)?;
println!("{answer}");

// 流式
for delta in model.generate_stream_iter("将以下文本翻译为英语：你好，世界。", 256) {
    print!("{delta}");
}
```

纯文本会被当作**用户内容**并按 chat 模板包装（与 0.1 一致）。如果你手上已经是渲染好的完整
prompt，用 `hy_mt::prompt::RawPrompt`。

### 3. 逐 token 控制

```rust
use hy_mt::{GenerateConfig, Message, SamplingConfig};

let cfg = GenerateConfig {
    sampling: SamplingConfig::greedy(),
    max_new_tokens: 512,
    stop_token_ids: vec![],
};
let ids = translator.encode_chat(&[Message::user("将以下文本翻译为英语：…")])?;
let mut session = translator.start_session(&ids, &cfg)?;
while let Some(delta) = session.next_delta()? {
    print!("{delta}");
}
let out = session.into_output()?;
println!("\n{} tokens, {:?}", out.generated_tokens, out.finish_reason);
```

## 翻译 Prompt

`TranslationRequest` 覆盖模型卡列出的 7 类场景，约束越具体优先级越高：

1. `format_type` — 结构化数据（JSON/YAML/HTML…，保护 key 与 `{{var}}` 占位符）
2. `background` — 背景信息
3. `preferences` — 个性化任务列表
4. `terminology` — 术语表
5. `style` — 风格约束
6. `preserve_delimiters` — 分隔符保持
7. 默认模板

指令语言自动选择（目标语言是中文 → 中文指令，否则英文），也可用
`prompt_lang(PromptLang::Zh)` 指定。语言参数支持代码（`"en"`、`"zh-Hans"`）或全称
（`"English"`、`"中文"`），内部按模型卡要求展开成全称。

## 后端 features

每个后端 feature 都必须同时转发给两个 candle crate：

```toml
metal = ["candle-core/metal", "candle-nn/metal"]
```

`candle-core` 管存储与矩阵乘，而本库调用的融合算子（`rms_norm`、`rotary_emb::rope`、
`softmax`）的 GPU kernel 在 `candle-nn` 里，且由 `#[cfg(feature = "metal")]` /
`#[cfg(feature = "cuda")]` 门控。只开一边能编译通过，但运行时会在 GPU 上报
`no metal implementation for op …`。

| feature | 后端 | 说明 |
|---|---|---|
| `cpu`（默认） | CPU | candle 始终编译 CPU 后端，flag 只是让选择显式化 |
| `metal` | Apple GPU | 仅 macOS，建议 f16 |
| `metal-debug-labels` | Apple GPU | command encoder 加 label，调试用 |
| `accelerate` | CPU | Apple Accelerate BLAS（macOS/iOS） |
| `cuda` / `cudnn` | NVIDIA GPU | `cudnn` 隐含 `cuda` |
| `mkl` | CPU | Intel MKL（构建时下载） |

## 性能

Apple M1（8 核 GPU / 16 GB），f16 + Metal，`cargo build --release`：

| 模型 | 短上下文（14 token） | 长上下文（518 token） | 加载 |
|---|---|---|---|
| `HY-MT1.5-1.8B-FP8` | 14.3 tok/s（70 ms/token） | 14.0 tok/s（71 ms/token） | ~6 s |
| `Hy-MT2-1.8B-FP8` | 13.0 tok/s（77 ms/token） | 12.5 tok/s（80 ms/token） | ~6 s |
| `Hy-MT3-1.8B-FP8` | 15.0 tok/s（67 ms/token） | 12.0 tok/s（84 ms/token） | ~6 s |

* 内存：f16 权重 ≈ 4.1 GB；KV cache 64 KB/token
* 预填充：518 token ≈ 0.12 s（Metal 内核已缓存时；首次运行有一次性的 shader 编译，
  `Translator` 默认在 `load()` 阶段预热）
* CPU（f16）：≈ 4.4 tok/s；CPU f32 更快，但 1.8B 需要 7.2 GB 内存

```bash
cargo run --release --features metal --example bench -- <model-dir> 40        # 短上下文
cargo run --release --features metal --example bench -- <model-dir> 20 500    # 长上下文
```

## 示例

```bash
cargo run --release --features metal --example text_generation_example -- <model-dir>
cargo run --release --features metal --example stream_generation_example -- <model-dir>
cargo run --release --features metal --example translate -- --model <model-dir> --target en --text "今天天气真好。"
cargo run --release --features metal --example quality -- <model-dir>   # 7 类 prompt 场景
cargo run --release --features metal --example bench -- <model-dir> 40
cargo run --example inspect_tokens -- <model-dir>                       # 分词 / 模板检查
```

## 测试

```bash
cargo test                       # 单元测试（不需要模型）
cargo test --features metal

# 端到端（会加载真实权重；建议串行，避免同时驻留多份模型）
HYMT_MODEL_DIRS=/path/HY-MT1.5-1.8B-FP8:/path/Hy-MT2-1.8B-FP8:/path/Hy-MT3-1.8B-FP8 \
  cargo test --release --features metal --test model -- --test-threads=1 --nocapture
```

## 从 0.1 升级

0.1 存在若干问题（FP8 权重无法加载、512 token 上下文截断、硬编码 EOS、无 KV cache、
candle 依赖 CUDA fork 等），详见 [MERGE_CN.md](MERGE_CN.md)：包含逐条 `file:line`、
实测证据、以及新旧 API 对照表。（[English](MERGE.md)）

公开 API 基本保持兼容：

```diff
- use hy_mt::TextGeneration;
+ use hy_mt::TextGeneration;                    // 仍在，行为已修复
+ use hy_mt::{Translator, TranslationRequest};  // 新的翻译 API
```

* `TextGeneration::generate` 不再向 stdout 打印，返回 `Result<String>`
* `TextGeneration::tokenizer()` 返回 `&Tokenizer`（需要原始 HF 句柄用 `tokenizer().inner()`）
* `hy_mt::model::*` 与 `hy_mt::masking_utils::*` 已移除，由 `hy_mt::models::CausalLm` 取代

## 许可

MIT。模型权重与 tokenizer 遵循混元官方许可（见各 checkpoint 内 `License.txt`）。
合并来源与归属见 [MERGE_CN.md](MERGE_CN.md#5-许可与归属)。
