//! Language names.
//!
//! The model is instructed with **full language names** (the model card is
//! explicit about this: “source_lang and target_lang use the full language
//! name, Chinese in Chinese, English in English”).  Users can pass either an
//! ISO code (`"en"`, `"zh-Hans"`) or a full name (`"English"`, `"中文"`); this
//! module maps codes to names and leaves anything it does not know untouched.

use crate::prompt::PromptLang;

/// One language of the 100+ supported by Hy-MT3.
#[derive(Debug, Clone, Copy)]
pub struct Language {
    /// ISO 639-1 (or 639-3) code.
    pub code: &'static str,
    /// Chinese name used in Chinese prompts.
    pub zh: &'static str,
    /// English name used in English prompts.
    pub en: &'static str,
}

const fn lang(code: &'static str, zh: &'static str, en: &'static str) -> Language {
    Language { code, zh, en }
}

/// Known languages, ordered roughly by frequency.
pub const LANGUAGES: &[Language] = &[
    lang("zh", "中文", "Chinese"),
    lang("en", "英语", "English"),
    lang("ja", "日语", "Japanese"),
    lang("ko", "韩语", "Korean"),
    lang("fr", "法语", "French"),
    lang("de", "德语", "German"),
    lang("es", "西班牙语", "Spanish"),
    lang("pt", "葡萄牙语", "Portuguese"),
    lang("ru", "俄语", "Russian"),
    lang("ar", "阿拉伯语", "Arabic"),
    lang("it", "意大利语", "Italian"),
    lang("nl", "荷兰语", "Dutch"),
    lang("pl", "波兰语", "Polish"),
    lang("tr", "土耳其语", "Turkish"),
    lang("vi", "越南语", "Vietnamese"),
    lang("th", "泰语", "Thai"),
    lang("id", "印尼语", "Indonesian"),
    lang("ms", "马来语", "Malay"),
    lang("hi", "印地语", "Hindi"),
    lang("bn", "孟加拉语", "Bengali"),
    lang("ur", "乌尔都语", "Urdu"),
    lang("fa", "波斯语", "Persian"),
    lang("he", "希伯来语", "Hebrew"),
    lang("uk", "乌克兰语", "Ukrainian"),
    lang("cs", "捷克语", "Czech"),
    lang("sk", "斯洛伐克语", "Slovak"),
    lang("sv", "瑞典语", "Swedish"),
    lang("da", "丹麦语", "Danish"),
    lang("no", "挪威语", "Norwegian"),
    lang("fi", "芬兰语", "Finnish"),
    lang("el", "希腊语", "Greek"),
    lang("hu", "匈牙利语", "Hungarian"),
    lang("ro", "罗马尼亚语", "Romanian"),
    lang("bg", "保加利亚语", "Bulgarian"),
    lang("sl", "斯洛文尼亚语", "Slovenian"),
    lang("hr", "克罗地亚语", "Croatian"),
    lang("sr", "塞尔维亚语", "Serbian"),
    lang("et", "爱沙尼亚语", "Estonian"),
    lang("lv", "拉脱维亚语", "Latvian"),
    lang("lt", "立陶宛语", "Lithuanian"),
    lang("ca", "加泰罗尼亚语", "Catalan"),
    lang("gl", "加利西亚语", "Galician"),
    lang("eu", "巴斯克语", "Basque"),
    lang("af", "南非荷兰语", "Afrikaans"),
    lang("sw", "斯瓦希里语", "Swahili"),
    lang("am", "阿姆哈拉语", "Amharic"),
    lang("yo", "约鲁巴语", "Yoruba"),
    lang("ig", "伊博语", "Igbo"),
    lang("ha", "豪萨语", "Hausa"),
    lang("zu", "祖鲁语", "Zulu"),
    lang("km", "高棉语", "Khmer"),
    lang("lo", "老挝语", "Lao"),
    lang("my", "缅甸语", "Burmese"),
    lang("si", "僧伽罗语", "Sinhala"),
    lang("ne", "尼泊尔语", "Nepali"),
    lang("ta", "泰米尔语", "Tamil"),
    lang("te", "泰卢固语", "Telugu"),
    lang("kn", "卡纳达语", "Kannada"),
    lang("ml", "马拉雅拉姆语", "Malayalam"),
    lang("mr", "马拉地语", "Marathi"),
    lang("gu", "古吉拉特语", "Gujarati"),
    lang("pa", "旁遮普语", "Punjabi"),
    lang("or", "奥里亚语", "Odia"),
    lang("as", "阿萨姆语", "Assamese"),
    lang("bo", "藏语", "Tibetan"),
    lang("mn", "蒙古语", "Mongolian"),
    lang("kk", "哈萨克语", "Kazakh"),
    lang("ky", "吉尔吉斯语", "Kyrgyz"),
    lang("uz", "乌兹别克语", "Uzbek"),
    lang("tg", "塔吉克语", "Tajik"),
    lang("az", "阿塞拜疆语", "Azerbaijani"),
    lang("hy", "亚美尼亚语", "Armenian"),
    lang("ka", "格鲁吉亚语", "Georgian"),
    lang("sq", "阿尔巴尼亚语", "Albanian"),
    lang("mk", "马其顿语", "Macedonian"),
    lang("bs", "波斯尼亚语", "Bosnian"),
    lang("be", "白俄罗斯语", "Belarusian"),
    lang("is", "冰岛语", "Icelandic"),
    lang("ga", "爱尔兰语", "Irish"),
    lang("cy", "威尔士语", "Welsh"),
    lang("mt", "马耳他语", "Maltese"),
    lang("lb", "卢森堡语", "Luxembourgish"),
    lang("fy", "西弗里西亚语", "Western Frisian"),
    lang("eo", "世界语", "Esperanto"),
    lang("fil", "菲律宾语", "Filipino"),
    lang("jv", "爪哇语", "Javanese"),
    lang("su", "巽他语", "Sundanese"),
    lang("ceb", "宿务语", "Cebuano"),
    lang("hmn", "苗语", "Hmong"),
    lang("ku", "库尔德语", "Kurdish"),
    lang("ps", "普什图语", "Pashto"),
    lang("sd", "信德语", "Sindhi"),
    lang("so", "索马里语", "Somali"),
    lang("rw", "卢旺达语", "Kinyarwanda"),
    lang("ny", "齐切瓦语", "Chichewa"),
    lang("sn", "绍纳语", "Shona"),
    lang("st", "塞索托语", "Sesotho"),
    lang("tn", "茨瓦纳语", "Tswana"),
    lang("xh", "科萨语", "Xhosa"),
];

/// Looks a language up by code or by (lowercase) English/Chinese name.
pub fn find(input: &str) -> Option<&'static Language> {
    let needle = input.trim();
    if needle.is_empty() {
        return None;
    }
    let lower = needle.to_ascii_lowercase();
    LANGUAGES.iter().find(|l| {
        l.code == lower
            || lower.split(['-', '_']).next() == Some(l.code)
            || l.en.to_ascii_lowercase() == lower
            || l.zh == needle
    })
}

/// Resolves a user supplied language into the name a prompt should use.
///
/// Unknown values are returned unchanged (the model also understands language
/// names that are not in this table).
pub fn resolve(input: &str, prompt_lang: PromptLang) -> String {
    match find(input) {
        Some(lang) => match prompt_lang {
            PromptLang::Zh => lang.zh.to_string(),
            PromptLang::En => lang.en.to_string(),
        },
        None => input.trim().to_string(),
    }
}

/// Whether the language is Chinese (used to pick a prompt language).
pub fn is_chinese(input: &str) -> bool {
    let lower = input.trim().to_ascii_lowercase();
    lower == "zh"
        || lower.starts_with("zh-")
        || lower.starts_with("zh_")
        || lower == "chinese"
        || lower == "中文"
        || lower == "汉语"
        || lower == "简体中文"
}

/// Heuristic: does the text contain CJK characters?
pub fn contains_cjk(text: &str) -> bool {
    text.chars().any(|c| {
        matches!(c as u32,
            0x4E00..=0x9FFF | 0x3400..=0x4DBF | 0x3040..=0x30FF | 0xAC00..=0xD7AF
            | 0x0E00..=0x0E7F | 0x1000..=0x109F)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_codes_and_names() {
        assert_eq!(resolve("en", PromptLang::Zh), "英语");
        assert_eq!(resolve("EN", PromptLang::En), "English");
        assert_eq!(resolve("English", PromptLang::Zh), "英语");
        assert_eq!(resolve("中文", PromptLang::En), "Chinese");
        assert_eq!(resolve("zh-Hans", PromptLang::En), "Chinese");
        // Unknown values pass through untouched.
        assert_eq!(resolve("Klingon", PromptLang::Zh), "Klingon");
    }

    #[test]
    fn detects_chinese() {
        assert!(is_chinese("zh"));
        assert!(is_chinese("中文"));
        assert!(is_chinese("Chinese"));
        assert!(!is_chinese("en"));
        assert!(contains_cjk("你好世界"));
        assert!(!contains_cjk("hello"));
    }
}
