//! Prompt building.
//!
//! Two things live here:
//!
//! * [`Prompter`] — the small trait from `hy-mt` 0.1 that turns prompts into the
//!   string a model should see;
//! * [`TranslationRequest`] — the structured translation request of the model
//!   card (target language, terminology, style, background, structured data…).
//!
//! The model card is explicit that language names must be spelled out
//! (`英语`/`English` rather than `en`), which [`crate::lang`] takes care of.

use crate::chat::Message;

pub use crate::chat::Message as PromptMessage;

/// Anything that can be turned into a prompt string (the original `hy-mt`
/// trait, kept for source compatibility).
///
/// Implementations are provided for `&str`, `String`, [`Message`], `Vec<Message>`
/// and `Vec<String>`.
pub trait Prompter {
    /// Renders the prompt.
    fn to_prompt(&self) -> String;
}

/// Plain text is treated as **user content** and wrapped in the chat template,
/// which is what `hy-mt` 0.1 did (it wrapped with `<BOS><USER>…<NO_GEN>`; we
/// wrap with the correct `<BOS><USER>…<ASSISTANT>`). Use [`RawPrompt`] when the
/// string is already a fully rendered prompt.
impl Prompter for &str {
    fn to_prompt(&self) -> String {
        Message::user(*self).to_prompt()
    }
}

impl Prompter for String {
    fn to_prompt(&self) -> String {
        Message::user(self.as_str()).to_prompt()
    }
}

impl Prompter for &String {
    fn to_prompt(&self) -> String {
        Message::user(self.as_str()).to_prompt()
    }
}

/// A prompt that must be used verbatim (no chat template applied).
///
/// ```rust
/// use hy_mt::{prompt::RawPrompt, Prompter};
/// // Already contains the template markers.
/// let raw = RawPrompt("<｜hy_begin▁of▁sentence｜>hi<｜hy_Assistant｜>");
/// assert_eq!(raw.to_prompt(), "<｜hy_begin▁of▁sentence｜>hi<｜hy_Assistant｜>");
/// ```
#[derive(Debug, Clone, Copy)]
pub struct RawPrompt<'a>(pub &'a str);

impl Prompter for RawPrompt<'_> {
    fn to_prompt(&self) -> String {
        self.0.to_string()
    }
}

impl Prompter for &&str {
    fn to_prompt(&self) -> String {
        Message::user(**self).to_prompt()
    }
}

impl Prompter for Message {
    /// A single message becomes a one-turn `Hy18B` chat prompt, terminated by
    /// the **assistant** marker (the old code emitted `<｜hy_place▁holder▁no▁8｜>`,
    /// the "no generation prompt" branch, which is why answers drifted).
    fn to_prompt(&self) -> String {
        crate::chat::ChatFormat::Hy18B
            .render(std::slice::from_ref(self), true)
            .unwrap_or_else(|_| self.content.clone())
    }
}

impl Prompter for Vec<Message> {
    /// A whole conversation is rendered with the chat template (one BOS, closing
    /// assistant turns) instead of concatenating one full prompt per message.
    fn to_prompt(&self) -> String {
        crate::chat::ChatFormat::Hy18B
            .render(self, true)
            .unwrap_or_else(|_| {
                self.iter()
                    .map(|m| m.content.clone())
                    .collect::<Vec<_>>()
                    .join("\n")
            })
    }
}

impl Prompter for Vec<String> {
    fn to_prompt(&self) -> String {
        let messages: Vec<Message> = self.iter().cloned().map(Message::from).collect();
        messages.to_prompt()
    }
}

use crate::lang;

/// Language the instruction is written in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromptLang {
    /// Chinese instructions (the default for Chinese targets).
    Zh,
    /// English instructions.
    En,
}

/// One terminology constraint: `source` must be translated as `target`.
#[derive(Debug, Clone)]
pub struct TermPair {
    /// Term in the source text.
    pub source: String,
    /// Required translation.
    pub target: String,
}

impl TermPair {
    /// Builds a term pair.
    pub fn new(source: impl Into<String>, target: impl Into<String>) -> Self {
        Self {
            source: source.into(),
            target: target.into(),
        }
    }
}

/// A translation job.
///
/// `text` and `target_lang` are the only required fields; everything else
/// selects one of the documented prompt templates.
#[derive(Debug, Clone, Default)]
pub struct TranslationRequest {
    /// Source text.
    pub text: String,
    /// Target language, as an ISO code or a full name.
    pub target_lang: String,
    /// Optional hint about the source language (the model detects it anyway).
    pub source_lang: Option<String>,
    /// Translation style, e.g. `"正式"` / `"口语化"`.
    pub style: Option<String>,
    /// Terminology constraints.
    pub terminology: Vec<TermPair>,
    /// Background information that should be taken into account.
    pub background: Option<String>,
    /// Free-form user preferences (personalisation template).
    pub preferences: Vec<String>,
    /// When set, treat the text as structured data of this format
    /// (e.g. `"JSON"`, `"YAML"`, `"HTML"`).
    pub format_type: Option<String>,
    /// Ask the model to preserve delimiters verbatim.
    pub preserve_delimiters: bool,
    /// Language of the instruction itself. `None` picks it automatically.
    pub prompt_lang: Option<PromptLang>,
    /// Optional system message.
    pub system: Option<String>,
    /// Per-request override of `max_new_tokens`.
    pub max_new_tokens: Option<usize>,
    /// Per-request override of the sampling temperature.
    pub temperature: Option<f64>,
}

impl TranslationRequest {
    /// Minimal request: `text` -> `target_lang`.
    pub fn new(text: impl Into<String>, target_lang: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            target_lang: target_lang.into(),
            ..Default::default()
        }
    }

    /// Sets the source language hint.
    pub fn source_lang(mut self, lang: impl Into<String>) -> Self {
        self.source_lang = Some(lang.into());
        self
    }

    /// Sets the translation style.
    pub fn style(mut self, style: impl Into<String>) -> Self {
        self.style = Some(style.into());
        self
    }

    /// Adds a terminology constraint.
    pub fn term(mut self, source: impl Into<String>, target: impl Into<String>) -> Self {
        self.terminology.push(TermPair::new(source, target));
        self
    }

    /// Adds background information.
    pub fn background(mut self, background: impl Into<String>) -> Self {
        self.background = Some(background.into());
        self
    }

    /// Adds a user preference.
    pub fn preference(mut self, preference: impl Into<String>) -> Self {
        self.preferences.push(preference.into());
        self
    }

    /// Marks the text as structured data of `format`.
    pub fn format_type(mut self, format: impl Into<String>) -> Self {
        self.format_type = Some(format.into());
        self
    }

    /// Asks for delimiters to be preserved.
    pub fn preserve_delimiters(mut self, yes: bool) -> Self {
        self.preserve_delimiters = yes;
        self
    }

    /// Forces the instruction language.
    pub fn prompt_lang(mut self, lang: PromptLang) -> Self {
        self.prompt_lang = Some(lang);
        self
    }

    /// Adds a system message.
    pub fn system(mut self, system: impl Into<String>) -> Self {
        self.system = Some(system.into());
        self
    }

    /// The instruction language that will be used.
    pub fn resolved_prompt_lang(&self) -> PromptLang {
        self.prompt_lang.unwrap_or_else(|| {
            if lang::is_chinese(&self.target_lang) {
                PromptLang::Zh
            } else {
                PromptLang::En
            }
        })
    }

    /// The target language as it will appear in the instruction.
    pub fn resolved_target_lang(&self) -> String {
        lang::resolve(&self.target_lang, self.resolved_prompt_lang())
    }

    /// Renders the user message of the translation request.
    pub fn build_prompt(&self) -> String {
        let plang = self.resolved_prompt_lang();
        let target = lang::resolve(&self.target_lang, plang);
        let text = &self.text;

        // Priority follows the documentation table: the more specific
        // constraint wins.
        if let Some(format) = &self.format_type {
            return match plang {
                PromptLang::Zh => format!(
                    "# 任务目标\n\
                     将下方 `{format}` 格式数据翻译为 {target}。\n\n\
                     # 严格约束\n\
                     1. **结构锁定**：绝对保持原有的 `{format}` 数据结构、缩进和层级完全不变。\n\
                     2. **选择性翻译**：仅翻译面向用户展示的可见文本内容。\n\
                     3. **禁止修改**：严禁翻译或更改任何代码标签、键名 (Key)、变量占位符（如 {{{{var}}}}、${{{{var}}}}、%s、%d 等）或代码属性。\n\n\
                     # 数据输入\n{text}"
                ),
                PromptLang::En => format!(
                    "### Task\n\
                     Translate the user-facing text within the following `{format}` data into {target}.\n\n\
                     ### Strict Rules\n\
                     1. **Structure Preservation:** You MUST preserve the original `{format}` data structure, nesting, hierarchy, and indentation exactly as they are.\n\
                     2. **Selective Translation:** Translate ONLY the visible, user-facing text content/values.\n\
                     3. **Strict Non-Translation:** NEVER translate or alter code tags, keys, properties, object names, or variable placeholders (such as {{{{var}}}}, ${{{{var}}}}, %s, %d).\n\n\
                     ### Source Data\n{text}"
                ),
            };
        }

        if let Some(background) = &self.background {
            return match plang {
                PromptLang::Zh => format!(
                    "【背景信息】\n{background}\n\n\
                     请结合背景信息将以下文本翻译为 {target}。\n\n\
                     【待翻译文本】\n{text}"
                ),
                PromptLang::En => format!(
                    "[Background Information]\n{background}\n\n\
                     Please translate the following text into {target}, taking the provided background information into consideration.\n\n\
                     [Source Text]\n{text}"
                ),
            };
        }

        if !self.preferences.is_empty() {
            let mut tasks = String::new();
            for (i, pref) in self.preferences.iter().enumerate() {
                match plang {
                    PromptLang::Zh => tasks.push_str(&format!("{}、{}\n", i + 1, pref)),
                    PromptLang::En => tasks.push_str(&format!("{}. {}\n", i + 1, pref)),
                }
            }
            let last = self.preferences.len() + 1;
            return match plang {
                PromptLang::Zh => format!(
                    "【待翻译文本】\n{text}\n\n【翻译任务】\n{tasks}{last}、将【待翻译文本】翻译为 {target}。"
                ),
                PromptLang::En => format!(
                    "[Source Text]\n{text}\n\n[Translation Tasks]\n{tasks}{last}. Translate the [Source Text] into {target}."
                ),
            };
        }

        if !self.terminology.is_empty() {
            let mut refs = String::new();
            for pair in &self.terminology {
                match plang {
                    PromptLang::Zh => {
                        refs.push_str(&format!("{} 翻译成 {}\n", pair.source, pair.target))
                    }
                    PromptLang::En => {
                        refs.push_str(&format!("{} translates to {}\n", pair.source, pair.target))
                    }
                }
            }
            return match plang {
                PromptLang::Zh => format!(
                    "参考下面的翻译：\n{refs}\
                     将以下文本翻译为 {target}，注意只需要输出翻译后的结果，不要额外解释：\n\n{text}"
                ),
                PromptLang::En => format!(
                    "Reference the following translations:\n{refs}\n\
                     Translate the following text into {target}. Note that you must ONLY output the translated result without any additional explanation:\n\n{text}"
                ),
            };
        }

        if let Some(style) = &self.style {
            return match plang {
                PromptLang::Zh => format!(
                    "请将以下文本翻译为 {target}。\n\
                     注意翻译的风格要严格符合【{style}】\n\n{text}"
                ),
                PromptLang::En => format!(
                    "Please translate the following text into {target}. \
                     Note that the translation style must strictly conform to [{style}]:\n\n{text}"
                ),
            };
        }

        if self.preserve_delimiters {
            return match plang {
                PromptLang::Zh => format!(
                    "请将以下文本准确翻译为 {target}。\n\
                     你必须在译文中保留等量的分隔符，绝对不可遗漏、转义或翻译该符号，并注意分隔符的位置。\n\n{text}"
                ),
                PromptLang::En => format!(
                    "Please accurately translate the following text into {target}.\n\
                     You must retain the exact same number of delimiters in the translation. \
                     Strictly do not omit, escape, or translate these symbols, and pay close attention to their placement.\n\n{text}"
                ),
            };
        }

        match plang {
            PromptLang::Zh => format!(
                "将以下文本翻译为 {target}，注意只需要输出翻译后的结果，不要额外解释：\n\n{text}"
            ),
            PromptLang::En => format!(
                "Translate the following text into {target}. Note that you should only output the translated result without any additional explanation:\n\n{text}"
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_chinese_prompt() {
        let req = TranslationRequest::new("今天天气真好。", "en");
        assert_eq!(req.resolved_prompt_lang(), PromptLang::En);
        assert_eq!(
            req.build_prompt(),
            "Translate the following text into English. Note that you should only output the translated result without any additional explanation:\n\n今天天气真好。"
        );

        let req = TranslationRequest::new("Hello.", "zh").prompt_lang(PromptLang::Zh);
        assert_eq!(
            req.build_prompt(),
            "将以下文本翻译为 中文，注意只需要输出翻译后的结果，不要额外解释：\n\nHello."
        );
    }

    #[test]
    fn terminology_prompt_lists_pairs() {
        let req = TranslationRequest::new("The cat sat.", "zh")
            .prompt_lang(PromptLang::Zh)
            .term("cat", "猫")
            .term("sat", "坐");
        let prompt = req.build_prompt();
        assert!(prompt.contains("cat 翻译成 猫"));
        assert!(prompt.contains("sat 翻译成 坐"));
        assert!(prompt.contains("将以下文本翻译为 中文"));
    }

    #[test]
    fn style_and_structured_variants() {
        let req = TranslationRequest::new("hi", "ja")
            .prompt_lang(PromptLang::Zh)
            .style("正式");
        assert!(req.build_prompt().contains("风格要严格符合【正式】"));

        let req = TranslationRequest::new("{\"a\": \"你好\"}", "en")
            .prompt_lang(PromptLang::En)
            .format_type("JSON");
        let prompt = req.build_prompt();
        assert!(prompt.contains("`JSON`"));
        assert!(prompt.contains("{{var}}"));
    }

    #[test]
    fn personalisation_numbers_tasks() {
        let req = TranslationRequest::new("text", "zh")
            .prompt_lang(PromptLang::Zh)
            .preference("使用敬语")
            .preference("不要翻译人名");
        let prompt = req.build_prompt();
        assert!(prompt.contains("1、使用敬语"));
        assert!(prompt.contains("3、将【待翻译文本】翻译为 中文。"));
    }

    #[test]
    fn plain_text_is_wrapped_as_a_user_turn() {
        // hy-mt 0.1 wrapped `&str` too; the difference is the terminator.
        assert_eq!("hi".to_prompt(), format!("{}{}hi{}", crate::chat::HY_BOS, crate::chat::HY_USER, crate::chat::HY_ASSISTANT));
        assert_eq!(String::from("hi").to_prompt(), "hi".to_prompt());
        assert!(!"hi".to_prompt().contains(crate::chat::HY_NO_GEN));
    }

    #[test]
    fn raw_prompt_is_verbatim() {
        let raw = RawPrompt("<｜hy_begin▁of▁sentence｜>x<｜hy_Assistant｜>");
        assert_eq!(raw.to_prompt(), "<｜hy_begin▁of▁sentence｜>x<｜hy_Assistant｜>");
    }

    #[test]
    fn messages_use_one_bos_and_close_assistant_turns() {
        let msgs = vec![Message::user("a"), Message::assistant("b")];
        let out = msgs.to_prompt();
        assert_eq!(out.matches(crate::chat::HY_BOS).count(), 1);
        assert!(out.ends_with(crate::chat::HY_ASSISTANT));
        assert!(out.contains(crate::chat::HY_EOS));
    }

    #[test]
    fn vec_string_is_a_conversation() {
        let out = vec!["a".to_string(), "b".to_string()].to_prompt();
        assert_eq!(out.matches(crate::chat::HY_BOS).count(), 1);
        assert_eq!(out.matches(crate::chat::HY_USER).count(), 2);
    }
}
