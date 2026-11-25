# 合并报告：`hy-mt` 0.1 → 0.2（统一引擎）

[English](MERGE.md) | **中文**

本文件记录把 [hymt3](https://github.com/kingzcheung/hymt3) 的引擎并入 `hy-mt` 的过程：
为什么可以合并、旧实现的问题清单（带 file:line 与实测）、API 兼容性、以及验证方式。

---

## 1. 结论：一套引擎覆盖整个 dense 家族

合并的依据不是"看起来像"，而是 checkpoint 级别的同一性。下面两个目录的 `config.json`
只差 `transformers_version` 一个字段，`chat_template.jinja` 逐字节相同：

| 对比项 | `HY-MT1.5-1.8B-FP8` | `Hy-MT3-1.8B-FP8` |
|---|---|---|
| `architectures` | `HunYuanDenseV1ForCausalLM` | 同 |
| `model_type` | `hunyuan_v1_dense` | 同 |
| vocab / hidden / layers | 120818 / 2048 / 32 | 同 |
| heads / kv heads / head_dim | 16 / 4 / 128 | 同 |
| `rope_scaling` | `alpha=1000, factor=1, type=dynamic` | 同 |
| chat template | `chat_template.jinja` | **逐字节相同** |
| 权重布局 | FP8 E4M3 + `weight_scale`，**无 `lm_head`**（tie） | 同 |
| `transformers_version` | 4.57.1 | 4.57.6 |
| 权重内容 | — | **不同**（`cmp` 在第 88587 字节起不同，是两次训练） |

即：**架构不变，变的是规模、词表、量化与训练**。所以"二合一"落点是"一套引擎 + 两套
chat/tokenizer 族"，而不是两套模型代码：

```
hy_mt
├── models::CausalLm              ← 架构抽象
│   └── models::hunyuan_dense     ← dense 家族（1.8B / 7B，bf16 / FP8）
├── chat::ChatFormat              ← 模板族：Hy18B | Hunyuan7B
├── tokenizer                     ← 120818(<｜hy_…｜>) 与 128k(<|…|>) 两族自适应
└── Translator / TextGeneration   ← 上层 API（新 / 旧）
```

### 版本矩阵

| checkpoint | 架构 | 词表 / 模板族 | 状态 |
|---|---|---|---|
| `HY-MT1.5-1.8B(-FP8)` | dense | 120818 / `Hy18B` | ✅ 实测通过 |
| `Hy-MT2-1.8B(-FP8)` | dense | 120818 / `Hy18B` | ✅ 实测通过（config 只差 `torch_dtype` 字段名） |
| `Hy-MT3-1.8B(-FP8)` | dense | 120818 / `Hy18B` | ✅ 实测通过 |
| `Hunyuan-MT-7B`、`HY-MT1.5-7B`、`Hy-MT3-7B`(+FP8) | dense | 128k / `Hunyuan7B` | ✅ 代码路径就绪（`torch_dtype`、数组 eos、ckpt 特有点位都已适配） |
| `Hy-MT3-30B-A3B(-FP8)` | **MoE** | 128k / `Hunyuan7B` | ⛔ 未实现（唯一真正"结构大变"的版本） |

三代 1.8B 的 `chat_template.jinja` **逐字节相同**，`config.json` 的差异只有
`dtype` → `torch_dtype` 这一处字段改名（0.2 用 `#[serde(alias = "torch_dtype")]` 兼容）。
同一引擎、同一份 prompt，三代给出的译文各不相同（`cmp` 也确认权重不同）：

| 源文 | HY-MT1.5 | Hy-MT2 | Hy-MT3 |
|---|---|---|---|
| 今天天气真好，我们一起去公园散步吧。(→en) | It's a beautiful day today. Let's go for a walk in the park together. | The weather is really nice today. Let's go for a walk in the park together. | The weather is really nice today; let's go for a walk in the park together. |
| The committee approved the budget after a lengthy debate. (→zh) | 在长时间的讨论之后，委员会最终批准了这份预算方案。 | 经过长时间的讨论，委员会通过了预算。 | 委员会经过长时间辩论后批准了预算。 |
| 混元大模型支持 102 种语言的互译。(→en) | The Hunyuan Large Model supports translation between 102 languages. | Hunyuan large model supports translation between 102 languages. | The Hunyuan large model supports mutual translation in 102 languages. |

> 7B 族与 1.8B 族的模板**完全不同**（不是措辞差异）：7B 用 `<|startoftext|>` 开头、
> `<|extra_0|>` 结束用户轮、`<|extra_4|>` 结束 system、`<|eos|>` 结束 assistant 轮，
> 且模板**忽略 `add_generation_prompt`**。旧实现只硬编码了 1.8B 族，所以 7B 一律不可用。

---

## 2. 旧实现的问题（按影响排序）

行号对应 `main` 分支（`git show main:src/…`）。

### A. 阻断级：FP8 checkpoint 根本加载不了

| # | 位置 | 问题 |
|---|---|---|
| A1 | `Cargo.toml:18-19` | 两个 candle 依赖**硬编码 `features = ["cuda"]`** → 无 GPU 的机器/CI 无法构建，也没有 metal/accelerate 选择 |
| A2 | `Cargo.toml:27-28` | `[patch.crates-io] cudarc = git tag` + candle 来自个人 fork（0.9.2-alpha.2）→ 下游不可复现 |
| A3 | `lib.rs:48` | `VarBuilder::from_mmaped_safetensors(..., DType::F32, ...)` → 只支持 bf16/f32 权重；FP8 的 `F8_E4M3 + weight_scale` 无法读 |
| A4 | `model.rs:525` | 无条件 `vb.pp("lm_head")` → 两个 FP8 checkpoint **都没有 `lm_head.weight`**（`config.tie_word_embeddings=true`），直接报找不到张量 |

**验证**：读两个 checkpoint 的 safetensors header（不下载整模型）：

```
Hy-MT1.5-1.8B-FP8: 802 tensors, lm_head present: False, dtypes ['BF16','F32','F8_E4M3']
Hy-MT3-1.8B-FP8  : 802 tensors, lm_head present: False, dtypes ['BF16','F32','F8_E4M3']
```

也就是说：旧实现**只能**跑 0.1 时代的 bf16 单文件发布（那份确实带重复的 `lm_head.weight`），
现在机器上的两个 FP8 模型它一个也打不开。新版按 config 的 `tie_word_embeddings` 决定 tie，
并按 `_scale` 反量化 FP8。

### B. 正确性

| # | 位置 | 问题 | 影响 |
|---|---|---|---|
| B1 | `lib.rs:145` | `context_window = 512`，只取最后 512 token 作为上下文 | 长文翻译**静默丢掉输入开头**，译文出现漏译；输出超过窗口也会自截断 |
| B2 | `lib.rs:145-150` + `model.rs` | 窗口滑动后 `position_ids` 仍从 0 重算（`Tensor::arange(0, seq_len)`） | 超过 512 token 时绝对位置全错，注意力位置信息失真 |
| B3 | `lib.rs:204` | `if next_token == 120020` 硬编码 EOS | 7B 族（`<|eos|>`=127960）、`Hunyuan-MT-7B`（127960）永远停不下来；只能靠 B5 的兜底 |
| B4 | `prompt.rs:24` | 用户轮以 `<｜hy_place▁holder▁no▁8｜>`（120026，模板里 `add_generation_prompt=false` 分支）结尾 | 与官方模板不一致（应为 `<｜hy_Assistant｜>`=120007）。实测影响有限但确实偏离 |
| B5 | `lib.rs:211-212` | "连续 3 个相同 token 就停" | 会切断合法输出（`---`、`。。。`、连续重复词），且掩盖了 B3 |
| B6 | `prompt.rs:47-56` | `Vec<Message>` = 每条消息各自生成 `<BOS><USER>…<NO_GEN>` 再 `\n` 拼接 | 多轮对话出现多个 BOS、没有 assistant 轮标记，结构错误 |
| B7 | `config.rs:14,38` | 配置字段全部必填（无 serde 默认） | `HY-MT1.5-7B`/`Hunyuan-MT-7B` 用 `torch_dtype` 而非 `dtype` → 解析直接失败；`Hy-MT3-*` 新增 `quantization_config` 同样炸 |
| B8 | `lib.rs:128` | `encode(prompt, true)` 未关闭 `tokenizer.json` 里的 truncation/padding | `Hy-MT3-1.8B` 的 tokenizer 声明 `padding: Fixed(8192)`，一旦命中会把 prompt 补齐到 8192 |

**B4 实测**（同一 prompt，仅替换结尾标记，greedy，`HY-MT1.5-1.8B-FP8`）：

| 源文 | 旧（NO_GEN） | 新（ASSISTANT） |
|---|---|---|
| 今天天气真好。 | `It's a beautiful day today.` | `It's a beautiful day today.` |
| The quick brown fox… | `那只快速的棕色狐狸，跳过了那只懒惰的狗。` | `那只快速的棕色狐狸跳过了那只懒惰的狗。` |
| 我们明天上午十点开会。 | `We will have a meeting tomorrow at 10 a.m.` | 同 |

→ 3 条里 1 条出现多余逗号。所以 B4 是**确实偏离模板、但影响有限**，不能当作"翻译效果差异"的唯一解释；
真正拖后腿的是 A（FP8 打不开）、B1/B2（长文截断）、B3（7B 停不下来）。

### B9. `Prompter for &str` 语义（迁移时的新发现）

旧实现里 `&str`/`String` 会被**包成** `<BOS><USER>{content}<NO_GEN>`；迁移时一度写成"原样透传"，
结果模型看到裸文本就顺着续写（实测出现 128 token 复读 `The quick brown fox jumps over the lazy dog.`）。
0.2 恢复"纯文本 = 一条 user 消息"的语义（终标记改为 `<ASSISTANT>`），并新增
`prompt::RawPrompt(&str)` 给"已经是完整 prompt"的调用方。这条也解释了为什么 0.1 的用户代码
`generate("Translate … Hello, world!", 512)` 是能工作的。

### C. 性能

| # | 位置 | 问题 | 新版 |
|---|---|---|---|
| C1 | `lib.rs:145-190` | **没有 KV cache**，每生成 1 个 token 重算整个前缀 | `KVCache` + 增量 decode |
| C2 | `utils.rs:44` | `expand().reshape()` 做 GQA repeat；Metal 上该路径 ~1 GB/s | `stack` + reshape（M1 实测 1838 µs → 28 µs @512 ctx） |
| C3 | `model.rs:494` | 每步都重算 RoPE 的 cos/sin（含 matmul + 三角函数） | 位置表按需增长并缓存 |
| C4 | `utils.rs:20-39` / `150` / `161` | RoPE 用 narrow+neg+cat 组合、mask 用 `broadcast_as` 新建整型张量 | candle 融合 `rotary_emb::rope` 内核；mask 直接预算 |
| C5 | `lib.rs:48` | 全模型 f32（1.8B ≈ 7.2 GB） | 按设备选 f16/bf16/f32；1.8B f16 ≈ 4.1 GB（含 tie 后的 lm_head） |
| C6 | `model.rs:256` | `is_causal=true` 时 `create_causal_mask` 算出的 mask 被丢弃，SDPA 内部再造一次三角 mask | 只构造真正需要的 mask |

### D. 工程/API

| # | 问题 |
|---|---|
| D1 | `lib.rs:180` 等处在库里 `print!` 输出，调用方无法捕获、并发会串行化 |
| D2 | 所有错误都压成 `candle_core::Error::msg`，丢失分类 |
| D3 | `generate(prompt, max_length)` 的 `max_length` 实际是"新 token 数"，命名误导 |
| D4 | 无 stop token、无 seed/采样参数的单次覆盖、无 Session 级控制 |
| D5 | `tokenizers 0.22` 默认特性会拉 oniguruma（C 依赖） |

---

## 3. 新版做了什么

### 引擎（来自 hymt3，Apache-2.0，同一作者）

* 配置驱动：`config.json` 决定层数/维度/头数/RoPE/tie；`torch_dtype` 别名、数组 `eos_token_id`、
  `use_cla` 显式拒绝。
* 权重：mmap safetensors（支持分片）、FP8 E4M3 查表反量化、`tie_word_embeddings` 优先 tie
  （省 0.5 GB）、f16/bf16/f32。
* 数学：`ntk_alpha` RoPE（`base = theta * alpha^(dim/(dim-2))`，全家族一致）、QK-RMSNorm 在
  RoPE **之后**、GQA、SwiGLU、tied lm_head、因果掩码、KV cache。
* 运行：`Session`（逐 token）、`generate`（回调）、增量 detokenizer（多字节不截断）。
* 模板：`ChatFormat::{Hy18B, Hunyuan7B}` 两族，按词表自动识别，7B 的怪癖（system 作为 head、
  空首条消息、忽略 `add_generation_prompt`）逐行复刻并有单测。
* 采样：temperature / top-k / top-p / repetition penalty（含窗口），自带 PRNG。
* 翻译：模型卡里的 7 类 prompt（默认/术语/风格/个性化/分隔符/结构化/背景）× 中英双语，
  语言代码→全称映射。

### 兼容层

| 0.1 | 0.2 |
|---|---|
| `hy_mt::TextGeneration::from_pretrained(path, &device)` | 保留 |
| `TextGeneration::generate(prompt, max_length)` | 保留（不再 print，返回 `Result<String>`） |
| `TextGeneration::generate_stream_iter(..)` | 保留（新增 `try_next()` 可看错误） |
| `prompt::{Prompter, Message}` | 保留（`Message` 从 `chat` 重导出） |
| `logits_processor::{Sampling, LogitsProcessor}` | 保留（映射到新采样器） |
| `tokenizer::TokenOutputStream` | 保留（改为 O(新 token) 增量解码） |
| `config::HunYuanDenseV1Config` | 保留为 `Config` 的类型别名 |
| `model::{HunyuanDenseV1ForCausalLM, …}`、`masking_utils::*` | 移除（由 `models::CausalLm` + `hunyuan_dense` 取代，见下） |
| `TextGeneration::tokenizer()` 返回 `&TokenOutputStream` | 改为返回 `&Tokenizer`；需要原始 HF 句柄用 `tokenizer().inner()` |

新增（推荐）API：`Translator::{translate, translate_with, translate_stream, translate_batch,
chat, chat_stream, start_session, encode_chat}`、`Session`、`TranslationRequest`、
`GenerateConfig`、`ModelOptions::device/dtype`、`HYMT_DEVICE`。

---

## 4. 实测（Apple M1, 16 GB, f16, Metal）

| 模型 | 短上下文（14 tok） | 长上下文（518 tok） | 加载 |
|---|---|---|---|
| `HY-MT1.5-1.8B-FP8` | 14.3 tok/s（70 ms/token） | 14.0 tok/s（71 ms/token） | ~6 s |
| `Hy-MT2-1.8B-FP8` | 13.0 tok/s（77 ms/token） | 12.5 tok/s（80 ms/token） | ~6 s |
| `Hy-MT3-1.8B-FP8` | 15.0 tok/s（67 ms/token） | 12.0 tok/s（84 ms/token） | ~6 s |

内存：f16 权重 ≈ 4.1 GB；KV cache 64 KB/token；预填充 518 token ≈ 0.12 s（Metal 内核已缓存时）。

端到端测试（两个 checkpoint 都跑）：

```bash
HYMT_MODEL_DIRS=…/Hy-MT1.5-1.8B-FP8:…/Hy-MT2-1.8B-FP8:…/Hy-MT3-1.8B-FP8 \
  cargo test --release --features metal --test model -- --test-threads=1 --nocapture
# 9 passed（三代 checkpoint 全部通过）
```

---

## 5. 许可与归属

* `hy-mt` 0.1（本项目原有代码）：MIT，版权归 kingzcheung。
* 并入的引擎源自同一作者的 `hymt3`（Apache-2.0）；两处作者相同，本仓库整体继续以 **MIT** 发布，
  保留本文件的来源说明。
* 兼容层里 `TokenOutputStream` 的思路可追溯到
  [text-generation-inference](https://github.com/huggingface/text-generation-inference)（Apache-2.0）
  与 candle 示例。
* 模型权重与 tokenizer 仍遵循混元官方许可（见各 checkpoint 内 `License.txt`）。
