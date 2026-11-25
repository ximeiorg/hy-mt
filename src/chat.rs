//! Chat templates for the whole Hunyuan-MT family.
//!
//! The checkpoints ship one of two template families. They are **not**
//! interchangeable, and neither is a cosmetic variation of the other:
//!
//! ### `Hy18B` — the 1.8B family
//!
//! `HY-MT1.5-1.8B`, `Hy-MT3-1.8B` (and their FP8 releases) share the same
//! architecture (`hunyuan_v1_dense`) *and* the same template:
//!
//! ```jinja
//! {% if messages[0].role == 'system' %}<BOS>{{ system }}<SYS_END>
//! {% else %}<BOS>{% endif %}
//! {% for m in loop_messages %}
//!   {% if m.role == 'user' %}<USER>{{ m.content }}
//!   {% elif m.role == 'assistant' %}<ASSISTANT>{{ m.content }}<EOS>{% endif %}
//! {% endfor %}
//! {% if add_generation_prompt %}<ASSISTANT>{% else %}<NO_GEN>{% endif %}
//! ```
//!
//! ### `Hunyuan7B` — the 7B family
//!
//! `Hunyuan-MT-7B`, `HY-MT1.5-7B`, `Hy-MT3-7B` use a different vocabulary
//! (128k, `<|startoftext|>` / `<|extra_N|>`) and a different template:
//!
//! ```jinja
//! {% for m in messages %}
//!   {% if loop.first and m.content == '' %}has_head = false
//!   {% elif loop.first and m.role == 'system' %}m.content = '<|startoftext|>' + m.content + '<|extra_4|>'
//!   {% endif %}
//!   {% if m.role == 'user' %}
//!     {% if loop.index0 == 1 and has_head %}m.content = m.content + '<|extra_0|>'
//!     {% else %}m.content = '<|startoftext|>' + m.content + '<|extra_0|>'{% endif %}
//!   {% elif m.role == 'assistant' %}m.content = m.content + '<|eos|>'{% endif %}
//!   {{ m.content }}
//! {% endfor %}
//! ```
//!
//! Note that the 7B template **ignores `add_generation_prompt`**: the assistant
//! turn is opened by the `<|extra_0|>` appended to the user turn.

use crate::error::{Error, Result};

/// Speaker of a message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    /// System / instruction message, only valid as the first message.
    System,
    /// End-user message.
    User,
    /// Model answer (used for multi-turn context).
    Assistant,
}

impl Role {
    /// Parses `"system"`, `"user"`, `"assistant"`.
    pub fn parse(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "system" => Ok(Self::System),
            "user" | "human" => Ok(Self::User),
            "assistant" | "model" | "gpt" => Ok(Self::Assistant),
            other => Err(Error::invalid_arg(format!("unknown role '{other}'"))),
        }
    }

    /// Lowercase name of the role.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
        }
    }
}

/// One chat message.
#[derive(Debug, Clone)]
pub struct Message {
    /// Who is speaking.
    pub role: Role,
    /// Raw content (no special tokens).
    pub content: String,
}

impl Message {
    /// Builds a message.
    pub fn new(role: Role, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
        }
    }

    /// System message helper.
    pub fn system(content: impl Into<String>) -> Self {
        Self::new(Role::System, content)
    }

    /// User message helper.
    pub fn user(content: impl Into<String>) -> Self {
        Self::new(Role::User, content)
    }

    /// Assistant message helper.
    pub fn assistant(content: impl Into<String>) -> Self {
        Self::new(Role::Assistant, content)
    }
}

impl From<&str> for Message {
    fn from(s: &str) -> Self {
        Self::user(s)
    }
}

impl From<String> for Message {
    fn from(s: String) -> Self {
        Self::user(s)
    }
}

// ---------------------------------------------------------------------------
// Hy18B markers (`<｜hy_...｜>`, vocab 120818)
// ---------------------------------------------------------------------------

/// `<｜hy_begin▁of▁sentence｜>` — id 120000.
pub const HY_BOS: &str = "<｜hy_begin▁of▁sentence｜>";
/// `<｜hy_User｜>` — id 120006.
pub const HY_USER: &str = "<｜hy_User｜>";
/// `<｜hy_Assistant｜>` — id 120007.
pub const HY_ASSISTANT: &str = "<｜hy_Assistant｜>";
/// `<｜hy_place▁holder▁no▁2｜>` — id 120020, end of turn (EOS).
pub const HY_EOS: &str = "<｜hy_place▁holder▁no▁2｜>";
/// `<｜hy_place▁holder▁no▁3｜>` — id 120021, closes a system message.
pub const HY_SYS_END: &str = "<｜hy_place▁holder▁no▁3｜>";
/// `<｜hy_place▁holder▁no▁8｜>` — id 120026, closes a prompt without generation.
pub const HY_NO_GEN: &str = "<｜hy_place▁holder▁no▁8｜>";
/// `<｜hy_▁pad▁｜>` — id 120002.
pub const HY_PAD: &str = "<｜hy_▁pad▁｜>";

// ---------------------------------------------------------------------------
// Hunyuan7B markers (`<|...|>`, vocab 128k)
// ---------------------------------------------------------------------------

/// `<|startoftext|>` — beginning of a turn.
pub const H7_BOS: &str = "<|startoftext|>";
/// `<|endoftext|>`.
pub const H7_EOT: &str = "<|endoftext|>";
/// `<|eos|>` — end of an assistant turn.
pub const H7_EOS: &str = "<|eos|>";
/// `<|pad|>`.
pub const H7_PAD: &str = "<|pad|>";
/// `<|extra_0|>` — opens the assistant answer (appended to the user turn).
pub const H7_ASSISTANT: &str = "<|extra_0|>";
/// `<|extra_4|>` — closes a system message.
pub const H7_SYS_END: &str = "<|extra_4|>";

/// Which chat template a checkpoint uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ChatFormat {
    /// 1.8B family: `<｜hy_begin▁of▁sentence｜>`, `<｜hy_User｜>`, `<｜hy_Assistant｜>`.
    Hy18B,
    /// 7B family: `<|startoftext|>`, `<|extra_0|>`, `<|extra_4|>`.
    Hunyuan7B,
}

/// Every special string either template can emit.
///
/// Used by [`crate::tokenizer::Tokenizer::encode_with_markers`] so that template
/// markers always become single tokens, whatever the vocabulary.
pub const MARKER_CANDIDATES: &[&str] = &[
    HY_BOS,
    HY_USER,
    HY_ASSISTANT,
    HY_EOS,
    HY_SYS_END,
    HY_NO_GEN,
    HY_PAD,
    H7_BOS,
    H7_EOT,
    H7_EOS,
    H7_PAD,
    H7_ASSISTANT,
    H7_SYS_END,
    "<｜hy_end▁of▁sentence｜>",
    "<｜hy_EOT｜>",
];

impl ChatFormat {
    /// Human readable name.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Hy18B => "hy18b",
            Self::Hunyuan7B => "hunyuan7b",
        }
    }

    /// Detects the family from a tokenizer's vocabulary.
    pub fn from_vocab(vocab: impl Fn(&str) -> Option<u32>) -> Option<Self> {
        let hy = [HY_BOS, HY_USER, HY_ASSISTANT, HY_EOS]
            .iter()
            .all(|t| vocab(t).is_some());
        if hy {
            return Some(Self::Hy18B);
        }
        let h7 = [H7_BOS, H7_ASSISTANT, H7_EOS]
            .iter()
            .all(|t| vocab(t).is_some());
        if h7 {
            return Some(Self::Hunyuan7B);
        }
        None
    }

    /// Renders the template for `messages`.
    ///
    /// `add_generation_prompt` is honoured by [`ChatFormat::Hy18B`] only; the
    /// 7B template has no such branch (the user turn already opens the answer).
    pub fn render(&self, messages: &[Message], add_generation_prompt: bool) -> Result<String> {
        match self {
            Self::Hy18B => render_hy18b(messages, add_generation_prompt),
            Self::Hunyuan7B => render_hunyuan7b(messages, add_generation_prompt),
        }
    }
}

fn render_hy18b(messages: &[Message], add_generation_prompt: bool) -> Result<String> {
    if messages.is_empty() {
        return Err(Error::invalid_arg("chat: at least one message is required"));
    }
    for (i, m) in messages.iter().enumerate() {
        if m.role == Role::System && i > 0 {
            return Err(Error::invalid_arg(
                "chat: a system message is only allowed as the first message",
            ));
        }
    }
    let mut out =
        String::with_capacity(messages.iter().map(|m| m.content.len() + 8).sum::<usize>() + 32);
    out.push_str(HY_BOS);
    let rest = if messages[0].role == Role::System {
        out.push_str(&messages[0].content);
        out.push_str(HY_SYS_END);
        &messages[1..]
    } else {
        messages
    };
    for m in rest {
        match m.role {
            Role::User => {
                out.push_str(HY_USER);
                out.push_str(&m.content);
            }
            Role::Assistant => {
                out.push_str(HY_ASSISTANT);
                out.push_str(&m.content);
                out.push_str(HY_EOS);
            }
            // Non-leading system messages are ignored, like the Jinja template.
            Role::System => {}
        }
    }
    if add_generation_prompt {
        out.push_str(HY_ASSISTANT);
    } else {
        out.push_str(HY_NO_GEN);
    }
    Ok(out)
}

/// Faithful re-implementation of the 7B Jinja template, including its quirks:
///
/// * a leading empty message disables the "head" so that the first user turn
///   carries its own `<|startoftext|>`;
/// * a leading system message *is* the head, so the first user turn is appended
///   without a second `<|startoftext|>`.
fn render_hunyuan7b(messages: &[Message], _add_generation_prompt: bool) -> Result<String> {
    if messages.is_empty() {
        return Err(Error::invalid_arg("chat: at least one message is required"));
    }
    for (i, m) in messages.iter().enumerate() {
        if m.role == Role::System && i > 0 {
            return Err(Error::invalid_arg(
                "chat: a system message is only allowed as the first message",
            ));
        }
    }

    let mut has_head = true;
    let mut out = String::new();
    for (i, m) in messages.iter().enumerate() {
        let mut content = m.content.clone();
        if i == 0 {
            if content.is_empty() {
                has_head = false;
            } else if m.role == Role::System {
                content = format!("{H7_BOS}{content}{H7_SYS_END}");
            }
        }
        match m.role {
            Role::User => {
                if i == 1 && has_head {
                    content.push_str(H7_ASSISTANT);
                } else {
                    content = format!("{H7_BOS}{content}{H7_ASSISTANT}");
                }
            }
            Role::Assistant => content.push_str(H7_EOS),
            Role::System => {}
        }
        out.push_str(&content);
    }
    Ok(out)
}

/// Renders the `Hy18B` template (convenience wrapper kept from the first
/// version of the crate).
///
/// Prefer [`ChatFormat::render`] with the format detected by
/// [`crate::tokenizer::Tokenizer::chat_format`].
pub fn apply_chat_template(messages: &[Message], add_generation_prompt: bool) -> Result<String> {
    ChatFormat::Hy18B.render(messages, add_generation_prompt)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hy18b_single_user_turn() {
        let msgs = [Message::user("你好")];
        let s = ChatFormat::Hy18B.render(&msgs, true).unwrap();
        assert_eq!(s, format!("{HY_BOS}{HY_USER}你好{HY_ASSISTANT}"));
        // The "no generation" branch of the Jinja template.
        let s = ChatFormat::Hy18B.render(&msgs, false).unwrap();
        assert_eq!(s, format!("{HY_BOS}{HY_USER}你好{HY_NO_GEN}"));
    }

    #[test]
    fn hy18b_system_then_user() {
        let msgs = [Message::system("sys"), Message::user("hi")];
        let s = ChatFormat::Hy18B.render(&msgs, true).unwrap();
        assert_eq!(s, format!("{HY_BOS}sys{HY_SYS_END}{HY_USER}hi{HY_ASSISTANT}"));
    }

    #[test]
    fn hy18b_multi_turn_closes_assistant_turns() {
        let msgs = [
            Message::user("a"),
            Message::assistant("b"),
            Message::user("c"),
        ];
        let s = ChatFormat::Hy18B.render(&msgs, false).unwrap();
        assert_eq!(
            s,
            format!("{HY_BOS}{HY_USER}a{HY_ASSISTANT}b{HY_EOS}{HY_USER}c{HY_NO_GEN}")
        );
    }

    #[test]
    fn hunyuan7b_single_user_turn() {
        let msgs = [Message::user("hello")];
        let s = ChatFormat::Hunyuan7B.render(&msgs, true).unwrap();
        assert_eq!(s, "<|startoftext|>hello<|extra_0|>");
        // add_generation_prompt is ignored by this template.
        let s = ChatFormat::Hunyuan7B.render(&msgs, false).unwrap();
        assert_eq!(s, "<|startoftext|>hello<|extra_0|>");
    }

    #[test]
    fn hunyuan7b_multi_turn() {
        let msgs = [
            Message::user("a"),
            Message::assistant("b"),
            Message::user("c"),
        ];
        let s = ChatFormat::Hunyuan7B.render(&msgs, true).unwrap();
        assert_eq!(
            s,
            "<|startoftext|>a<|extra_0|>b<|eos|><|startoftext|>c<|extra_0|>"
        );
    }

    #[test]
    fn hunyuan7b_system_head_quirk() {
        // A leading system message becomes the head, so the first user turn
        // does not get its own <|startoftext|> (index 0 == 1 && has_head).
        let msgs = [Message::system("sys"), Message::user("hi")];
        let s = ChatFormat::Hunyuan7B.render(&msgs, true).unwrap();
        assert_eq!(s, "<|startoftext|>sys<|extra_4|>hi<|extra_0|>");
    }

    #[test]
    fn hunyuan7b_empty_first_message_disables_head() {
        // Jinja: an empty first message clears `has_head`, and the first user
        // turn still goes through the `else` branch (index 0 != 1).
        let msgs = [Message::user(""), Message::user("hi")];
        let s = ChatFormat::Hunyuan7B.render(&msgs, true).unwrap();
        assert_eq!(s, "<|startoftext|><|extra_0|><|startoftext|>hi<|extra_0|>");
    }

    #[test]
    fn detects_family_from_vocab() {
        let hy = |t: &str| {
            [HY_BOS, HY_USER, HY_ASSISTANT, HY_EOS]
                .contains(&t)
                .then_some(1)
        };
        assert_eq!(ChatFormat::from_vocab(hy), Some(ChatFormat::Hy18B));

        let h7 = |t: &str| {
            [H7_BOS, H7_ASSISTANT, H7_EOS]
                .contains(&t)
                .then_some(1)
        };
        assert_eq!(ChatFormat::from_vocab(h7), Some(ChatFormat::Hunyuan7B));

        assert_eq!(ChatFormat::from_vocab(|_| None), None);
    }

    #[test]
    fn rejects_late_system_message() {
        let msgs = [Message::user("a"), Message::system("b")];
        assert!(ChatFormat::Hy18B.render(&msgs, true).is_err());
        assert!(ChatFormat::Hunyuan7B.render(&msgs, true).is_err());
    }
}
