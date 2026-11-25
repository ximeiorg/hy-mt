//! Runs the documented translation scenarios on a checkpoint (dev/eval tool).
//!
//! ```bash
//! cargo run --release --features metal --example quality -- <model-dir>
//! ```

use hy_mt::{GenerateConfig, PromptLang, TranslationRequest, Translator};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = std::env::args()
        .nth(1)
        .ok_or("usage: quality <model-dir>")?;
    let mut t = Translator::builder(&dir).warmup(true).load()?;
    let cfg = GenerateConfig::greedy().max_new_tokens(512);

    let cases: Vec<(&str, TranslationRequest)> = vec![
        (
            "default zh->en (paragraph)",
            TranslationRequest::new(
                "混元大模型是腾讯自研的通用大语言模型，支持多语言翻译。\
                 它在游戏本地化、电商描述和图片翻译等场景中表现稳定，\
                 并且可以通过术语表约束来保证品牌名和专有名词的一致性。",
                "en",
            ),
        ),
        (
            "terminology en->zh",
            TranslationRequest::new(
                "The Hunyuan-MT3 model runs on Apple silicon thanks to the candle framework.",
                "zh",
            )
            .prompt_lang(PromptLang::Zh)
            .term("Hunyuan-MT3", "混元 MT3")
            .term("Apple silicon", "苹果芯片")
            .term("candle", "Candle 框架"),
        ),
        (
            "style zh->ja",
            TranslationRequest::new("请尽快回复，我们明天开会。", "ja")
                .prompt_lang(PromptLang::Zh)
                .style("正式、商务"),
        ),
        (
            "structured JSON en->zh",
            TranslationRequest::new(
                r#"{"title": "Getting Started", "button": {"label": "Download", "count": 12}, "note": "Version {{version}} is ready"}"#,
                "zh",
            )
            .prompt_lang(PromptLang::En)
            .format_type("JSON"),
        ),
        (
            "background en->zh",
            TranslationRequest::new("It runs three times faster than the previous version.", "zh")
                .prompt_lang(PromptLang::En)
                .background("The text is from a press release about the new Hy-MT3 translation model."),
        ),
        (
            "delimiters zh->en",
            TranslationRequest::new("【重要】请在 ||24 小时|| 内完成 #审核#。", "en")
                .prompt_lang(PromptLang::Zh)
                .preserve_delimiters(true),
        ),
    ];

    for (name, req) in cases {
        let t0 = std::time::Instant::now();
        let out = t.translate_with(&req, Some(&cfg))?;
        println!("\n=== {name} ===");
        println!("src: {}", req.text.replace('\n', "\\n"));
        println!(
            "out ({:.2}s, {} tok): {}",
            t0.elapsed().as_secs_f64(),
            out.output.generated_tokens,
            out.text
        );
        if out.text.trim().is_empty() {
            println!("!! EMPTY OUTPUT");
        }
    }
    Ok(())
}
