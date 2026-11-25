# hy-mt

[![Crates.io](https://img.shields.io/crates/v/hy-mt.svg)](https://crates.io/crates/hy-mt) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Rust](https://img.shields.io/badge/rust-1.82%2B-orange.svg)](https://www.rust-lang.org)

**English** | [中文](README_CN.md)

Rust inference for Tencent Hunyuan's **MT translation models**, built on
[candle](https://github.com/huggingface/candle). One engine covers the whole dense
family (HY-MT1.5 / Hy-MT2 / Hy-MT3, 1.8B / 7B, bf16 / FP8) — no Python, PyTorch or
transformers required.

* Loads the official `*-FP8` (compressed-tensors) and bf16 checkpoints, dequantising
  to f16/bf16/f32 at load time
* Apple Metal, CUDA and CPU (Accelerate / MKL) backends; features are additive and
  the device is chosen at runtime
* Both chat template families, auto-detected: `<｜hy_…｜>` for 1.8B, `<|startoftext|>` /
  `<|extra_N|>` for 7B
* KV cache and an incremental detokeniser, so streaming never splits a multi-byte
  character
* The seven translation prompt templates documented in the model card (terminology,
  style, personalisation, delimiters, structured data, background…), in Chinese and
  English
* Keeps the `TextGeneration` API of 0.1 — existing code only needs a version bump

## Supported checkpoints

| checkpoint | arch | vocab / template | status |
|---|---|---|---|
| `HY-MT1.5-1.8B` / `-FP8` | dense | 120 818 / `Hy18B` | ✅ verified end to end |
| `Hy-MT2-1.8B` / `-FP8` | dense | 120 818 / `Hy18B` | ✅ verified end to end |
| `Hy-MT3-1.8B` / `-FP8` | dense | 120 818 / `Hy18B` | ✅ verified end to end |
| `Hunyuan-MT-7B`, `HY-MT1.5-7B`, `Hy-MT3-7B` (+FP8) | dense | 128k / `Hunyuan7B` | ✅ code path in place |
| `Hy-MT3-30B-A3B` (+FP8) | **MoE** | 128k / `Hunyuan7B` | ⛔ not implemented |

Everything is driven by `config.json` (`hunyuan_v1_dense`), so a checkpoint of the
same architecture needs **no code change** — just point it at another directory.
1.8B vs 7B only differs in size, vocabulary and template family; FP8 vs bf16 only
differs in how weights are stored. `Hy-MT2-1.8B` and its neighbours differ by a
single config field name (`torch_dtype`, accepted through a serde alias), and their
`chat_template.jinja` files are byte-identical.

The same engine on three generations (greedy decoding, identical prompts):

| source | HY-MT1.5 | Hy-MT2 | Hy-MT3 |
|---|---|---|---|
| 今天天气真好，我们一起去公园散步吧。(→en) | It's a beautiful day today. Let's go for a walk in the park together. | The weather is really nice today. Let's go for a walk in the park together. | The weather is really nice today; let's go for a walk in the park together. |
| The committee approved the budget after a lengthy debate. (→zh) | 在长时间的讨论之后，委员会最终批准了这份预算方案。 | 经过长时间的讨论，委员会通过了预算。 | 委员会经过长时间辩论后批准了预算。 |
| 混元大模型支持 102 种语言的互译。(→en) | The Hunyuan Large Model supports translation between 102 languages. | Hunyuan large model supports translation between 102 languages. | The Hunyuan large model supports mutual translation in 102 languages. |

## Install

```toml
[dependencies]
hy-mt = "0.2"

# pick one backend (features are additive, the device is chosen at runtime)
# hy-mt = { version = "0.2", features = ["metal"] }
# hy-mt = { version = "0.2", features = ["cuda"] }
```

Building the repository:

```bash
cargo build --release --features metal      # Apple GPU (recommended, f16)
cargo build --release                       # CPU (f32)
cargo build --release --features "metal,accelerate"
```

## Quick start

### 1. Translation API (recommended)

```rust
use hy_mt::{TranslationRequest, Translator};

let mut translator = Translator::from_pretrained("/models/Hy-MT3-1.8B-FP8")?;

// default template
println!("{}", translator.translate("今天天气真好。", "en")?.text);

// with terminology / style constraints
let req = TranslationRequest::new("The Hunyuan-MT3 model runs on Apple silicon.", "zh")
    .term("Hunyuan-MT3", "混元 MT3")
    .term("Apple silicon", "苹果芯片")
    .style("正式");
println!("{}", translator.translate_with(&req, None)?.text);

// streaming
translator.translate_stream(&req, None, |delta| print!("{delta}"))?;
```

Device and precision:

```rust
use hy_mt::{DType, Device, Translator};

let t = Translator::builder(dir)
    .device(Device::Cpu)          // or Device::new_metal(0)? / Device::new_cuda(0)?
    .dtype(DType::F16)
    .verbose(true)
    .load()?;
```

Selection order: explicit `device(..)`, then `HYMT_DEVICE=cpu|metal[:N]|cuda[:N]`,
then Metal, CUDA and finally the CPU. The default weight dtype follows the device:
f16 on Metal, bf16 on CUDA, f32 on CPU.

### 2. Raw generation API (0.1 compatible)

```rust
use hy_mt::{Device, TextGeneration};

let mut model = TextGeneration::from_pretrained("/models/HY-MT1.5-1.8B-FP8", &Device::Cpu)?;

// one shot
let answer = model.generate("将以下文本翻译为英语：你好，世界。", 256)?;
println!("{answer}");

// streaming
for delta in model.generate_stream_iter("将以下文本翻译为英语：你好，世界。", 256) {
    print!("{delta}");
}
```

Plain strings are treated as **user content** and wrapped in the chat template (as in
0.1). Use `hy_mt::prompt::RawPrompt` when you already have a rendered prompt.

### 3. Token-level control

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

## Translation prompts

`TranslationRequest` covers the seven scenarios of the model card; the most specific
constraint wins:

1. `format_type` — structured data (JSON/YAML/HTML…, keys and `{{var}}` placeholders preserved)
2. `background` — background information
3. `preferences` — personalisation task list
4. `terminology` — glossary
5. `style` — style constraint
6. `preserve_delimiters` — keep delimiters verbatim
7. default template

The instruction language is picked automatically (Chinese target → Chinese
instruction, otherwise English); force it with `prompt_lang(PromptLang::Zh)`.
Languages may be given as codes (`"en"`, `"zh-Hans"`) or full names (`"English"`,
`"中文"`); they are expanded to full names as the model card requires.

## Backend features

Every backend feature must be forwarded to **both** candle crates:

```toml
metal = ["candle-core/metal", "candle-nn/metal"]
```

`candle-core` owns storage and matmuls, while `candle-nn` provides the GPU kernels of
the fused ops this crate calls (`rms_norm`, `rotary_emb::rope`, `softmax`), and those
are gated behind `#[cfg(feature = "metal")]` / `#[cfg(feature = "cuda")]`. Enabling
only one side compiles but fails at runtime with `no metal implementation for op …`.

| feature | backend | notes |
|---|---|---|
| `cpu` (default) | CPU | candle always compiles the CPU backend; the flag makes the choice explicit |
| `metal` | Apple GPU | macOS only, f16 recommended |
| `metal-debug-labels` | Apple GPU | command-encoder labels, for debugging |
| `accelerate` | CPU | Apple Accelerate BLAS (macOS/iOS) |
| `cuda` / `cudnn` | NVIDIA GPU | `cudnn` implies `cuda` |
| `mkl` | CPU | Intel MKL (downloaded at build time) |

## Performance

Apple M1 (8-core GPU, 16 GB), f16 + Metal, `cargo build --release`:

| model | short context (14 tokens) | long context (518 tokens) | load |
|---|---|---|---|
| `HY-MT1.5-1.8B-FP8` | 14.3 tok/s (70 ms/token) | 14.0 tok/s (71 ms/token) | ~6 s |
| `Hy-MT2-1.8B-FP8` | 13.0 tok/s (77 ms/token) | 12.5 tok/s (80 ms/token) | ~6 s |
| `Hy-MT3-1.8B-FP8` | 15.0 tok/s (67 ms/token) | 12.0 tok/s (84 ms/token) | ~6 s |

* Memory: ~4.1 GB of f16 weights; KV cache 64 KB/token
* Prefill: 518 tokens in ~0.12 s (with warm Metal kernels; the first run pays a one-off
  shader compilation, and `Translator` warms up during `load()` by default)
* CPU (f16): ~4.4 tok/s; CPU f32 is faster but needs 7.2 GB for the 1.8B model

```bash
cargo run --release --features metal --example bench -- <model-dir> 40        # short context
cargo run --release --features metal --example bench -- <model-dir> 20 500    # long context
```

## Examples

```bash
cargo run --release --features metal --example text_generation_example -- <model-dir>
cargo run --release --features metal --example stream_generation_example -- <model-dir>
cargo run --release --features metal --example translate -- --model <model-dir> --target en --text "今天天气真好。"
cargo run --release --features metal --example quality -- <model-dir>   # the seven prompt scenarios
cargo run --release --features metal --example bench -- <model-dir> 40
cargo run --release --features metal --example micro_bench               # operator-level timings
cargo run --example inspect_tokens -- <model-dir>                       # tokenizer / template inspection
```

## Testing

```bash
cargo test                       # unit tests (no checkpoint needed)
cargo test --features metal

# end to end (loads real weights; run serially so only one model is resident)
HYMT_MODEL_DIRS=/path/HY-MT1.5-1.8B-FP8:/path/Hy-MT2-1.8B-FP8:/path/Hy-MT3-1.8B-FP8 \
  cargo test --release --features metal --test model -- --test-threads=1 --nocapture
```

## Upgrading from 0.1

0.1 had several problems (FP8 checkpoints could not be loaded at all, a 512-token
context window truncated inputs, a hard-coded EOS id, no KV cache, a CUDA-only candle
fork…). See [MERGE.md](MERGE.md) for the full report with `file:line` references,
measurements and an old→new API table. ([中文版](MERGE_CN.md))

The public API is largely source compatible:

```diff
- use hy_mt::TextGeneration;
+ use hy_mt::TextGeneration;                    // still there, bugs fixed
+ use hy_mt::{Translator, TranslationRequest};  // new translation API
```

* `TextGeneration::generate` no longer prints to stdout and returns `Result<String>`
* `TextGeneration::tokenizer()` returns `&Tokenizer` (`tokenizer().inner()` for the raw
  HuggingFace handle)
* `hy_mt::model::*` and `hy_mt::masking_utils::*` were removed in favour of
  `hy_mt::models::CausalLm`

## License

MIT. Model weights and tokenizers follow Tencent Hunyuan's own license (see
`License.txt` inside each checkpoint). Provenance of the merged code is documented in
[MERGE.md](MERGE.md#5-license-and-provenance).
