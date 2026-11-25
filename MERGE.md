# Merge report: `hy-mt` 0.1 → 0.2 (unified engine)

**English** | [中文](MERGE_CN.md)

This document records how the engine of the `hymt3` prototype crate was merged into
`hy-mt`: why the two can be one, the defect list of the old implementation
(with `file:line` and measurements), API compatibility, and how the result was verified.

---

## 1. Conclusion: one engine for the whole dense family

The argument is not "they look alike" but checkpoint-level identity. The `config.json` of
the two directories below differs in a single field, and their `chat_template.jinja` files
are byte-identical:

| | `HY-MT1.5-1.8B-FP8` | `Hy-MT3-1.8B-FP8` |
|---|---|---|
| `architectures` | `HunYuanDenseV1ForCausalLM` | same |
| `model_type` | `hunyuan_v1_dense` | same |
| vocab / hidden / layers | 120 818 / 2048 / 32 | same |
| heads / kv heads / head_dim | 16 / 4 / 128 | same |
| `rope_scaling` | `alpha=1000, factor=1, type=dynamic` | same |
| chat template | `chat_template.jinja` | **byte-identical** |
| weight layout | FP8 E4M3 + `weight_scale`, **no `lm_head`** (tied) | same |
| `transformers_version` | 4.57.1 | 4.57.6 |
| weight bytes | — | **different** (`cmp` diverges at byte 88587: three distinct trainings) |

`Hy-MT2-1.8B` fits the same picture: identical sizes and template, with only
`dtype` spelled `torch_dtype` (accepted through a serde alias).

So the architecture never changed across generations — scale, vocabulary, quantisation and
training did. "Merging" therefore means **one engine + two chat/tokenizer families**, not
two model implementations:

```
hy_mt
├── models::CausalLm              ← architecture abstraction
│   └── models::hunyuan_dense     ← dense family (1.8B / 7B, bf16 / FP8)
├── chat::ChatFormat              ← template families: Hy18B | Hunyuan7B
├── tokenizer                     ← detects 120818 (<｜hy_…｜>) vs 128k (<|…|>) vocabularies
└── Translator / TextGeneration   ← public API (new / 0.1-compatible)
```

### Version matrix

| checkpoint | arch | vocab / template | status |
|---|---|---|---|
| `HY-MT1.5-1.8B(-FP8)` | dense | 120 818 / `Hy18B` | ✅ verified end to end |
| `Hy-MT2-1.8B(-FP8)` | dense | 120 818 / `Hy18B` | ✅ verified end to end (config only renames `dtype`) |
| `Hy-MT3-1.8B(-FP8)` | dense | 120 818 / `Hy18B` | ✅ verified end to end |
| `Hunyuan-MT-7B`, `HY-MT1.5-7B`, `Hy-MT3-7B` (+FP8) | dense | 128k / `Hunyuan7B` | ✅ code path in place (`torch_dtype`, list-valued eos and the checkpoint specifics are handled) |
| `Hy-MT3-30B-A3B(-FP8)` | **MoE** | 128k / `Hunyuan7B` | ⛔ not implemented (the only real structural change) |

The same engine, same prompts, three generations (greedy decoding; weights are genuinely
different, as `cmp` confirms):

| source | HY-MT1.5 | Hy-MT2 | Hy-MT3 |
|---|---|---|---|
| 今天天气真好，我们一起去公园散步吧。(→en) | It's a beautiful day today. Let's go for a walk in the park together. | The weather is really nice today. Let's go for a walk in the park together. | The weather is really nice today; let's go for a walk in the park together. |
| The committee approved the budget after a lengthy debate. (→zh) | 在长时间的讨论之后，委员会最终批准了这份预算方案。 | 经过长时间的讨论，委员会通过了预算。 | 委员会经过长时间辩论后批准了预算。 |
| 混元大模型支持 102 种语言的互译。(→en) | The Hunyuan Large Model supports translation between 102 languages. | Hunyuan large model supports translation between 102 languages. | The Hunyuan large model supports mutual translation in 102 languages. |

> The 7B family uses a **completely different** template, not a rephrasing: `<|startoftext|>`
> opens a turn, `<|extra_0|>` closes the user turn, `<|extra_4|>` closes a system message,
> `<|eos|>` closes an assistant turn, and the template **ignores `add_generation_prompt`**.
> The old implementation hard-coded the 1.8B family only, so 7B never worked.

---

## 2. Defects of the old implementation (ordered by impact)

Line numbers refer to `main` (`git show main:src/…`).

### A. Blocker: FP8 checkpoints cannot be loaded at all

| # | location | problem |
|---|---|---|
| A1 | `Cargo.toml:18-19` | both candle dependencies **hard-code `features = ["cuda"]`** → no way to build on a machine without a GPU, and no metal/accelerate choice |
| A2 | `Cargo.toml:27-28` | `[patch.crates-io] cudarc = git tag` plus a personal candle fork (0.9.2-alpha.2) → not reproducible downstream |
| A3 | `lib.rs:48` | `VarBuilder::from_mmaped_safetensors(..., DType::F32, ...)` → only bf16/f32 weights; FP8 `F8_E4M3 + weight_scale` cannot be read |
| A4 | `model.rs:525` | unconditional `vb.pp("lm_head")` → the FP8 checkpoints **have no `lm_head.weight`** (`tie_word_embeddings=true`) and fail with a missing-tensor error |

Evidence — reading the safetensors headers (no full download needed):

```
Hy-MT1.5-1.8B-FP8: 802 tensors, lm_head present: False, dtypes ['BF16','F32','F8_E4M3']
Hy-MT2-1.8B-FP8  : 802 tensors, lm_head present: False, dtypes ['BF16','F32','F8_E4M3']
Hy-MT3-1.8B-FP8  : 802 tensors, lm_head present: False, dtypes ['BF16','F32','F8_E4M3']
```

In other words the old implementation could only run the bf16 single-file release of its own
era (which does ship a duplicated `lm_head.weight`); it cannot open any of the FP8
checkpoints now on disk. 0.2 ties the head according to `tie_word_embeddings` and
dequantises FP8 through the per-tensor `weight_scale`.

### B. Correctness

| # | location | problem | impact |
|---|---|---|---|
| B1 | `lib.rs:145` | `context_window = 512`; only the last 512 tokens are used | long documents silently **lose their beginning** (omissions in the translation); output beyond the window is cut too |
| B2 | `lib.rs:145-150` + `model.rs` | after the window slides, `position_ids` are still rebuilt from 0 (`Tensor::arange(0, seq_len)`) | absolute positions are wrong beyond 512 tokens, so positional information is distorted |
| B3 | `lib.rs:204` | `if next_token == 120020`, hard-coded EOS | the 7B family (`<|eos|>` = 127960) and `Hunyuan-MT-7B` never stop; only B5's fallback ends them |
| B4 | `prompt.rs:24` | the user turn ends with `<｜hy_place▁holder▁no▁8｜>` (120026, the `add_generation_prompt=false` branch) | deviates from the official template (it should be `<｜hy_Assistant｜>` = 120007); measured impact is small but real |
| B5 | `lib.rs:211-212` | "stop after 3 identical consecutive tokens" | truncates legitimate output (`---`, `。。。`, repeated words) and masks B3 |
| B6 | `prompt.rs:47-56` | `Vec<Message>` renders each message as its own `<BOS><USER>…<NO_GEN>` block and joins with `\n` | multi-turn prompts contain several BOS tokens and no assistant turn markers |
| B7 | `config.rs:14,38` | every config field is required (no serde defaults) | `HY-MT1.5-7B`/`Hunyuan-MT-7B` use `torch_dtype` instead of `dtype` → parsing fails; `Hy-MT3-*` adds `quantization_config` → same |
| B8 | `lib.rs:128` | `encode(prompt, true)` without disabling the truncation/padding of `tokenizer.json` | `Hy-MT3-1.8B`'s tokenizer declares `padding: Fixed(8192)`, which would pad a prompt to 8192 tokens |
| B9 | `prompt.rs:23-24` | `Prompter for &str` semantics: 0.1 **wraps** plain text | while merging, a "pass-through" variant made the model continue raw text; it emitted the source sentence 128 times. 0.2 restores wrapping (with the assistant terminator) and adds `prompt::RawPrompt` for already-rendered prompts |

**B4 measurement** (same prompt, only the final marker differs, greedy, `HY-MT1.5-1.8B-FP8`):

| source | old (`NO_GEN`) | fixed (`ASSISTANT`) |
|---|---|---|
| 今天天气真好。 | `It's a beautiful day today.` | `It's a beautiful day today.` |
| The quick brown fox… | `那只快速的棕色狐狸，跳过了那只懒惰的狗。` | `那只快速的棕色狐狸跳过了那只懒惰的狗。` |
| 我们明天上午十点开会。 | `We will have a meeting tomorrow at 10 a.m.` | same |

→ 1 of 3 sentences gained a spurious comma. B4 is a genuine deviation from the template but
a *minor* one; it is not the explanation for the quality gap. What actually hurts is
A (FP8 cannot load), B1/B2 (long-input truncation) and B3 (7B never stops).

### C. Performance

| # | location | problem | 0.2 |
|---|---|---|---|
| C1 | `lib.rs:145-190` | **no KV cache**: the entire prefix is recomputed for every generated token | `KVCache` + incremental decoding |
| C2 | `utils.rs:44` | GQA repeat via `expand().reshape()`; that path runs at ~1 GB/s on Metal | `stack` + reshape (M1, 512 ctx: 1838 µs → 28 µs) |
| C3 | `model.rs:494` | RoPE cos/sin recomputed on every step (matmul + trigonometry) | position table grown on demand and cached |
| C4 | `utils.rs:20-39`, `150`, `161` | RoPE assembled from narrow+neg+cat, masks rebuilt with `broadcast_as` | candle's fused `rotary_emb::rope` kernel; masks precomputed once |
| C5 | `lib.rs:48` | the whole model in f32 (1.8B ≈ 7.2 GB) | dtype chosen per device; 1.8B f16 ≈ 4.1 GB (including the tied head) |
| C6 | `model.rs:256` | with `is_causal=true` the `create_causal_mask` result is discarded and SDPA builds another triangular mask | only the masks that are actually used are built |

### D. Engineering / API

| # | problem |
|---|---|
| D1 | `print!` calls inside library functions (`lib.rs:180` etc.): callers cannot capture output, and concurrent use serialises on stdout |
| D2 | every error is flattened into `candle_core::Error::msg`, losing classification |
| D3 | `generate(prompt, max_length)` — `max_length` is really "number of new tokens", which is misleading |
| D4 | no stop tokens, no per-call seed/sampling override, no session-level control |
| D5 | `tokenizers 0.22` with default features pulls in oniguruma (a C dependency) |

---

## 3. What 0.2 does instead

### Engine (from hymt3, Apache-2.0, same author)

* Configuration driven: `config.json` determines layers/dims/heads/RoPE/tying;
  `torch_dtype` alias, list-valued `eos_token_id`, explicit rejection of `use_cla`.
* Weights: mmapped safetensors (sharded files supported), FP8 E4M3 table dequantisation,
  `tie_word_embeddings` honoured (saves 0.5 GB), f16/bf16/f32.
* Maths: `ntk_alpha` RoPE (`base = theta * alpha^(dim/(dim-2))`, identical across the
  family), QK-RMSNorm **after** RoPE, GQA, SwiGLU, tied `lm_head`, causal masking, KV cache.
* Runtime: `Session` (token by token), `generate` (callback), incremental detokeniser that
  never splits multi-byte characters.
* Templates: `ChatFormat::{Hy18B, Hunyuan7B}`, auto-detected from the vocabulary; the 7B
  quirks (system message as head, empty first message, ignored `add_generation_prompt`) are
  reproduced line by line and covered by unit tests.
* Sampling: temperature / top-k / top-p / repetition penalty (with window), own PRNG.
* Translation: the seven prompt templates of the model card (default, terminology, style,
  personalisation, delimiters, structured data, background) in Chinese and English, plus
  code→full-name language mapping.

### Compatibility layer

| 0.1 | 0.2 |
|---|---|
| `hy_mt::TextGeneration::from_pretrained(path, &device)` | kept |
| `TextGeneration::generate(prompt, max_length)` | kept (no longer prints; returns `Result<String>`) |
| `TextGeneration::generate_stream_iter(..)` | kept (new `try_next()` exposes errors) |
| `prompt::{Prompter, Message}` | kept (`Message` re-exported from `chat`) |
| `logits_processor::{Sampling, LogitsProcessor}` | kept (mapped onto the new sampler) |
| `tokenizer::TokenOutputStream` | kept (now decodes only the new tail, O(new tokens)) |
| `config::HunYuanDenseV1Config` | kept as a type alias of `Config` |
| `model::{HunyuanDenseV1ForCausalLM, …}`, `masking_utils::*` | removed (replaced by `models::CausalLm` + `hunyuan_dense`) |
| `TextGeneration::tokenizer()` returning `&TokenOutputStream` | now returns `&Tokenizer`; use `tokenizer().inner()` for the raw HF handle |

New (recommended) API: `Translator::{translate, translate_with, translate_stream,
translate_batch, chat, chat_stream, start_session, encode_chat}`, `Session`,
`TranslationRequest`, `GenerateConfig`, `ModelOptions::device/dtype`, `HYMT_DEVICE`.

---

## 4. Measurements (Apple M1, 16 GB, f16, Metal)

| model | short context (14 tokens) | long context (518 tokens) | load |
|---|---|---|---|
| `HY-MT1.5-1.8B-FP8` | 14.3 tok/s (70 ms/token) | 14.0 tok/s (71 ms/token) | ~6 s |
| `Hy-MT2-1.8B-FP8` | 13.0 tok/s (77 ms/token) | 12.5 tok/s (80 ms/token) | ~6 s |
| `Hy-MT3-1.8B-FP8` | 15.0 tok/s (67 ms/token) | 12.0 tok/s (84 ms/token) | ~6 s |

Memory: ~4.1 GB of f16 weights; KV cache 64 KB/token; prefill of 518 tokens ≈ 0.12 s
(with warm Metal kernels).

End-to-end tests (all generations):

```bash
HYMT_MODEL_DIRS=…/Hy-MT1.5-1.8B-FP8:…/Hy-MT2-1.8B-FP8:…/Hy-MT3-1.8B-FP8 \
  cargo test --release --features metal --test model -- --test-threads=1 --nocapture
# 9 passed
```

---

## 5. License and provenance

* `hy-mt` 0.1 (the original code of this repository): MIT, copyright kingzcheung.
* The merged engine comes from the same author's `hymt3` prototype (Apache-2.0). Both are authored by
  the same person, and this repository continues to be released under **MIT**; the
  provenance note is kept here.
* The `TokenOutputStream` idea in the compatibility layer traces back to
  [text-generation-inference](https://github.com/huggingface/text-generation-inference)
  (Apache-2.0) and the candle examples.
* Model weights and tokenizers remain under Tencent Hunyuan's license (see `License.txt`
  inside each checkpoint).
