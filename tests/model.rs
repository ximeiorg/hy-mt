//! End-to-end tests against real checkpoints.
//!
//! Skipped unless `HYMT_MODEL_DIR` (or `HYMT_MODEL_DIRS`, colon separated) points
//! at one or more model directories:
//!
//! ```bash
//! HYMT_MODEL_DIRS=/models/HY-MT1.5-1.8B-FP8:/models/Hy-MT3-1.8B-FP8 \
//!     cargo test --release --features metal --test model -- --test-threads=1 --nocapture
//! ```
//!
//! The tests share one loaded model (`OnceLock<Mutex<Translator>>`); run them
//! serially so that only one extra checkpoint is alive at a time.

use std::sync::{Mutex, MutexGuard, OnceLock};
use std::time::Instant;

use hy_mt::{
    ChatFormat, GenerateConfig, Message, PromptLang, SamplingConfig, TextGeneration,
    TranslationRequest, Translator,
};

/// Every checkpoint directory given on the environment.
fn model_dirs() -> Vec<String> {
    if let Ok(list) = std::env::var("HYMT_MODEL_DIRS") {
        let dirs: Vec<String> = list
            .split(':')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty() && std::path::Path::new(s).is_dir())
            .collect();
        if !dirs.is_empty() {
            return dirs;
        }
    }
    if let Ok(dir) = std::env::var("HYMT_MODEL_DIR") {
        if std::path::Path::new(&dir).is_dir() {
            return vec![dir];
        }
    }
    Vec::new()
}

macro_rules! require_model {
    () => {
        match model_dirs().first().cloned() {
            Some(dir) => dir,
            None => {
                eprintln!("skipping: set HYMT_MODEL_DIR or HYMT_MODEL_DIRS");
                return;
            }
        }
    };
}

static TRANSLATOR: OnceLock<Mutex<Translator>> = OnceLock::new();

fn translator(dir: &str) -> MutexGuard<'static, Translator> {
    let cell = TRANSLATOR.get_or_init(|| Mutex::new(Translator::from_pretrained(dir).unwrap()));
    cell.lock().unwrap_or_else(|e| e.into_inner())
}

fn greedy(max_new_tokens: usize) -> GenerateConfig {
    GenerateConfig {
        sampling: SamplingConfig::greedy(),
        max_new_tokens,
        stop_token_ids: Vec::new(),
    }
}

#[test]
fn loads_config_and_tokenizer() {
    let dir = require_model!();
    let tokenizer = hy_mt::Tokenizer::from_dir(&dir).unwrap();
    assert!(tokenizer.vocab_size() >= 120_000);
    let format = tokenizer.chat_format().expect("chat format detected");
    match format {
        ChatFormat::Hy18B => {
            assert_eq!(tokenizer.special().bos, Some(120_000));
            assert_eq!(tokenizer.special().eos, Some(120_020));
            assert_eq!(tokenizer.special().user, Some(120_006));
            assert_eq!(tokenizer.special().assistant, Some(120_007));
        }
        ChatFormat::Hunyuan7B => {
            assert!(tokenizer.special().bos.is_some());
            assert!(tokenizer.special().eos.is_some());
        }
    }
    let cfg = hy_mt::Config::from_dir(&dir).unwrap();
    assert_eq!(cfg.head_dim(), 128);
    assert_eq!(cfg.num_key_value_groups(), 4);
    // The ntk_alpha base stretch of the whole family.
    assert!(cfg.rope_base() > 1.0e7, "rope base: {}", cfg.rope_base());
}

/// Regression guard for the `hy-mt` 0.1 bug: the prompt must end with the
/// *assistant* marker, never with the "no generation" marker.
#[test]
fn chat_prompt_opens_the_assistant_turn() {
    let dir = require_model!();
    let translator = translator(&dir);
    let prompt = translator
        .render_chat(&[Message::user("你好")], true)
        .unwrap();
    match translator.chat_format().unwrap() {
        ChatFormat::Hy18B => {
            assert!(prompt.ends_with(hy_mt::chat::HY_ASSISTANT), "{prompt:?}");
            assert!(!prompt.contains(hy_mt::chat::HY_NO_GEN), "{prompt:?}");
        }
        ChatFormat::Hunyuan7B => {
            assert!(prompt.ends_with(hy_mt::chat::H7_ASSISTANT), "{prompt:?}");
        }
    }
}

#[test]
fn translates_zh_to_en() {
    let dir = require_model!();
    let mut translator = translator(&dir);
    let req = TranslationRequest::new("今天天气真好。", "en");
    let t0 = Instant::now();
    let out = translator.translate_with(&req, Some(&greedy(64))).unwrap();
    println!(
        "zh->en: {:?} ({} tok, {:.2}s, {:.1} tok/s)",
        out.text,
        out.output.generated_tokens,
        t0.elapsed().as_secs_f64(),
        out.tokens_per_second()
    );
    let lowered = out.text.to_lowercase();
    assert!(
        lowered.contains("weather") || lowered.contains("nice") || lowered.contains("good"),
        "unexpected translation: {}",
        out.text
    );
}

#[test]
fn translates_en_to_zh_with_terminology() {
    let dir = require_model!();
    let mut translator = translator(&dir);
    let req = TranslationRequest::new("Hunyuan is a translation model.", "zh")
        .prompt_lang(PromptLang::Zh)
        .term("Hunyuan", "混元");
    let out = translator
        .translate_with(&req, Some(&greedy(96)))
        .unwrap();
    println!("en->zh: {:?}", out.text);
    assert!(
        out.text.contains("混元"),
        "terminology not applied: {}",
        out.text
    );
}

#[test]
fn streaming_matches_full_decode() {
    let dir = require_model!();
    let mut translator = translator(&dir);
    let cfg = greedy(32);
    let req = TranslationRequest::new("Good morning.", "zh").prompt_lang(PromptLang::Zh);

    let full = translator.translate_with(&req, Some(&cfg)).unwrap();
    let mut streamed = String::new();
    let streamed_out = translator
        .translate_stream(&req, Some(&cfg), |delta| streamed.push_str(delta))
        .unwrap();
    assert_eq!(full.output.token_ids, streamed_out.output.token_ids);
    assert_eq!(streamed, streamed_out.text);
}

#[test]
fn multi_turn_chat() {
    let dir = require_model!();
    let mut translator = translator(&dir);
    let messages = [
        Message::user("将以下文本翻译为英语：你好，世界。"),
        Message::assistant("Hello, world."),
        Message::user("将以下文本翻译为日语：你好，世界。"),
    ];
    let out = translator.chat(&messages, &greedy(48)).unwrap();
    println!("multi-turn: {:?}", out.text);
    assert!(!out.text.trim().is_empty());
}

#[test]
fn session_api_matches_generate() {
    let dir = require_model!();
    let mut translator = translator(&dir);
    let cfg = greedy(24);
    let messages = [Message::user("将以下文本翻译为英语：今天下雨。")];
    let ids = translator.encode_chat(&messages).unwrap();

    let full = translator.generate_ids(&ids, &cfg).unwrap();
    let mut session = translator.start_session(&ids, &cfg).unwrap();
    let mut deltas = String::new();
    while let Some(delta) = session.next_delta().unwrap() {
        deltas.push_str(&delta);
    }
    let out = session.into_output().unwrap();
    assert_eq!(full.token_ids, out.token_ids);
    assert_eq!(deltas, full.text);
}

/// The `hy-mt` 0.1 API (`TextGeneration`) must keep working.
#[test]
fn text_generation_compat_api() {
    let dir = require_model!();
    let device = hy_mt::default_device().unwrap();
    let mut model = TextGeneration::from_pretrained(&dir, &device).unwrap();
    let answer = model
        .generate("将以下文本翻译为英语：这个库支持多个模型版本。", 64)
        .unwrap();
    println!("compat: {answer:?}");
    assert!(!answer.trim().is_empty());

    let streamed: String = model
        .generate_stream_iter("将以下文本翻译为英语：流式输出。", 48)
        .collect();
    println!("compat stream: {streamed:?}");
    assert!(!streamed.trim().is_empty());
}

/// Runs the same sentence through every checkpoint given in `HYMT_MODEL_DIRS`,
/// which is how the cross-generation claim is verified (HY-MT1.5 vs Hy-MT3).
#[test]
fn every_generation_translates() {
    let dirs = model_dirs();
    if dirs.len() < 2 {
        eprintln!("skipping: set HYMT_MODEL_DIRS to at least two checkpoints");
        return;
    }
    for dir in dirs {
        let mut translator = Translator::from_pretrained(&dir).unwrap();
        let out = translator
            .translate_with(
                &TranslationRequest::new("Artificial intelligence is transforming the way we work.", "zh"),
                Some(&greedy(64)),
            )
            .unwrap();
        println!("{dir} -> {}", out.text);
        assert!(out.text.contains('人') || out.text.contains("智能"), "{dir}: {}", out.text);
        drop(translator);
    }
}
