//! Prints how the chat prompt is tokenised — useful to sanity check the
//! tokenizer / template against the reference implementation.
//!
//! ```bash
//! cargo run --example inspect_tokens -- /path/to/Hy-MT3-1.8B-FP8
//! ```

use hy_mt::chat::{apply_chat_template, Message};
use hy_mt::Tokenizer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = std::env::args()
        .nth(1)
        .ok_or("usage: inspect_tokens <model-dir>")?;
    let tokenizer = Tokenizer::from_dir(&dir)?;
    println!("vocab size: {}", tokenizer.vocab_size());
    println!("special tokens: {:?}", tokenizer.special());

    let messages = [Message::user(
        "将以下文本翻译为 英语，注意只需要输出翻译后的结果，不要额外解释：\n\n今天天气真好。",
    )];
    let prompt = apply_chat_template(&messages, true)?;
    println!("\n--- prompt ---\n{prompt}\n--------------");
    let ids = tokenizer.encode_with_markers(&prompt)?;
    println!("{} tokens: {ids:?}", ids.len());
    let plain = tokenizer.encode(&prompt, false)?;
    println!("plain encode {} tokens: {plain:?}", plain.len());
    println!("decoded back: {:?}", tokenizer.decode(&ids, false)?);
    println!("decoded (skip special): {:?}", tokenizer.decode(&ids, true)?);
    Ok(())
}
