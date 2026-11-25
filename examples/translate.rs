//! Minimal CLI / usage example.
//!
//! ```bash
//! cargo run --release --features metal --example translate -- \
//!     --model /path/to/Hy-MT3-1.8B-FP8 --target en --text "今天天气真好。"
//! ```
//!
//! Without `--text` a couple of demo sentences are translated.

use std::time::Instant;

use hy_mt::{GenerateConfig, SamplingConfig, TranslationRequest, Translator};

struct Args {
    model: String,
    target: String,
    text: Option<String>,
    greedy: bool,
    max_new_tokens: usize,
    cpu: bool,
    prompt_lang_zh: bool,
    sample: bool,
    dtype: Option<String>,
}

fn parse_args() -> Result<Args, String> {
    let mut args = Args {
        model: String::new(),
        target: "en".into(),
        text: None,
        greedy: false,
        max_new_tokens: 512,
        cpu: false,
        prompt_lang_zh: false,
        sample: false,
        dtype: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let mut value = |name: &str| -> Result<String, String> {
            it.next().ok_or_else(|| format!("{name} requires a value"))
        };
        match arg.as_str() {
            "--model" | "-m" => args.model = value("--model")?,
            "--target" | "-t" => args.target = value("--target")?,
            "--text" | "-s" => args.text = Some(value("--text")?),
            "--max-new-tokens" | "-n" => {
                args.max_new_tokens = value("--max-new-tokens")?
                    .parse()
                    .map_err(|_| "--max-new-tokens expects an integer".to_string())?
            }
            "--greedy" => args.greedy = true,
            "--cpu" => args.cpu = true,
            "--prompt-zh" => args.prompt_lang_zh = true,
            "--sample" => args.sample = true,
            "--dtype" => args.dtype = Some(value("--dtype")?),
            "--help" | "-h" => return Err("help".into()),
            other => return Err(format!("unknown argument '{other}'")),
        }
    }
    if args.model.is_empty() {
        return Err("missing --model <dir>".into());
    }
    Ok(args)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = match parse_args() {
        Ok(args) => args,
        Err(msg) => {
            if msg == "help" {
                println!(
                    "usage: translate --model <dir> [--target <lang>] [--text <s>]\n\
                     \x20                [--sample] [--cpu] [--dtype f32|f16|bf16] [--prompt-zh]\n\
                     \x20                [--max-new-tokens <n>]"
                );
                return Ok(());
            }
            eprintln!("error: {msg}");
            std::process::exit(2);
        }
    };

    let mut builder = Translator::builder(&args.model).verbose(true);
    if args.cpu {
        builder = builder.device(candle_device_cpu());
    }
    if args.prompt_lang_zh {
        builder = builder.prompt_lang(hy_mt::PromptLang::Zh);
    }
    if let Some(dtype) = &args.dtype {
        builder = builder.dtype(match dtype.as_str() {
            "f32" => candle_core::DType::F32,
            "f16" => candle_core::DType::F16,
            "bf16" => candle_core::DType::BF16,
            other => return Err(format!("unknown --dtype '{other}'").into()),
        });
    }

    let t0 = Instant::now();
    let mut translator = builder.load()?;
    println!(
        "[hy_mt] loaded {} in {:.1}s (dtype {:?}, device {:?})",
        translator.architecture(),
        t0.elapsed().as_secs_f64(),
        translator.dtype(),
        translator.device()
    );

    let mut cfg: GenerateConfig = if args.sample {
        translator.default_generate_config()
    } else {
        translator.default_translate_config()
    };
    cfg.max_new_tokens = args.max_new_tokens;
    if args.greedy {
        cfg.sampling = SamplingConfig::greedy();
    }

    let texts: Vec<String> = match args.text {
        Some(text) => vec![text],
        None => vec![
            "今天天气真好，我们一起去公园散步吧。".to_string(),
            "The quick brown fox jumps over the lazy dog.".to_string(),
            "Artificial intelligence is transforming the way we work.".to_string(),
        ],
    };

    for text in texts {
        let req = TranslationRequest::new(text.clone(), args.target.clone());
        let t0 = Instant::now();
        let out = translator.translate_with(&req, Some(&cfg))?;
        let elapsed = t0.elapsed().as_secs_f64();
        println!("\n── source ─────────────────────────────");
        println!("{text}");
        println!("── {} ({:.2}s, {} tok, {:.1} tok/s) ──", args.target, elapsed, out.output.generated_tokens, out.tokens_per_second());
        println!("{}", out.text);
    }
    Ok(())
}

fn candle_device_cpu() -> candle_core::Device {
    candle_core::Device::Cpu
}
