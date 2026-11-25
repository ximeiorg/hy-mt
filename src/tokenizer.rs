//! Tokenizer wrapper.
//!
//! The family ships two vocabularies (and two chat templates, see
//! [`crate::chat`]):
//!
//! * 1.8B family — 120818 tokens, `<｜hy_...｜>` markers;
//! * 7B family — ~128k tokens, `<|startoftext|>` / `<|extra_N|>` markers.
//!
//! Both are fast BPE tokenizers described by `tokenizer.json`. We use the
//! pure-Rust `tokenizers` crate (fancy-regex backend, no oniguruma/C
//! dependency) and drive the chat template ourselves.

use std::path::Path;

use tokenizers::Tokenizer as HfTokenizer;

use crate::chat::{self, ChatFormat, MARKER_CANDIDATES};
use crate::error::{Error, Result};

pub use crate::chat::{
    HY_ASSISTANT as ASSISTANT, HY_BOS as BOS, HY_EOS as EOS, HY_NO_GEN as NO_GEN,
    HY_PAD as PAD, HY_SYS_END as SYS_END, HY_USER as USER,
};
pub use crate::chat::{H7_EOS as H7_EOS_TOKEN, H7_BOS as H7_BOS_TOKEN};

/// Ids of the tokens that matter for chat handling.
///
/// Every field is optional because the two template families expose different
/// markers: the 7B family has no separate `<user>` / `<assistant>` token.
#[derive(Debug, Clone, Copy)]
pub struct SpecialTokens {
    /// Detected chat template family, if any.
    pub family: Option<ChatFormat>,
    /// Beginning of sequence (`<｜hy_begin▁of▁sentence｜>` or `<|startoftext|>`).
    pub bos: Option<u32>,
    /// End of turn (EOS).
    pub eos: Option<u32>,
    /// Padding token.
    pub pad: Option<u32>,
    /// User-turn marker (1.8B family only).
    pub user: Option<u32>,
    /// Assistant-turn marker (1.8B family only).
    pub assistant: Option<u32>,
    /// System-message terminator.
    pub sys_end: Option<u32>,
    /// "No generation" terminator (1.8B family only).
    pub no_gen: Option<u32>,
}

impl SpecialTokens {
    /// All end-of-sequence ids known to the tokenizer.
    pub fn eos_ids(&self) -> Vec<u32> {
        self.eos.into_iter().collect()
    }
}

/// Thin wrapper around the HuggingFace BPE tokenizer.
#[derive(Debug)]
pub struct Tokenizer {
    inner: HfTokenizer,
    special: SpecialTokens,
}

impl Tokenizer {
    /// Loads `tokenizer.json` from a model directory.
    pub fn from_dir(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();
        let path = dir.join("tokenizer.json");
        if !path.is_file() {
            return Err(Error::InvalidModelDir {
                dir: dir.display().to_string(),
                reason: "tokenizer.json not found".into(),
            });
        }
        Self::from_file(&path)
    }

    /// Loads a `tokenizer.json` file.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let mut inner = HfTokenizer::from_file(path.as_ref())
            .map_err(|e| Error::Tokenizer(format!("cannot load tokenizer.json: {e}")))?;
        // The checkpoints ship truncation/padding settings intended for batched
        // training (Hy-MT3-1.8B declares `padding: Fixed(8192)`); for inference
        // we want full control over the sequence, so both are disabled.
        let _ = inner.with_truncation(None);
        let _ = inner.with_padding(None);

        let id = |name: &str| inner.token_to_id(name);
        let family = ChatFormat::from_vocab(id);
        let special = match family {
            Some(ChatFormat::Hy18B) => SpecialTokens {
                family,
                bos: id(chat::HY_BOS),
                eos: id(chat::HY_EOS),
                pad: id(chat::HY_PAD),
                user: id(chat::HY_USER),
                assistant: id(chat::HY_ASSISTANT),
                sys_end: id(chat::HY_SYS_END),
                no_gen: id(chat::HY_NO_GEN),
            },
            Some(ChatFormat::Hunyuan7B) => SpecialTokens {
                family,
                bos: id(chat::H7_BOS),
                eos: id(chat::H7_EOS),
                pad: id(chat::H7_PAD),
                user: None,
                assistant: id(chat::H7_ASSISTANT),
                sys_end: id(chat::H7_SYS_END),
                no_gen: None,
            },
            None => SpecialTokens {
                family: None,
                bos: id(chat::HY_BOS).or_else(|| id(chat::H7_BOS)),
                eos: id(chat::HY_EOS).or_else(|| id(chat::H7_EOS)),
                pad: id(chat::HY_PAD).or_else(|| id(chat::H7_PAD)),
                user: id(chat::HY_USER),
                assistant: id(chat::HY_ASSISTANT).or_else(|| id(chat::H7_ASSISTANT)),
                sys_end: id(chat::HY_SYS_END).or_else(|| id(chat::H7_SYS_END)),
                no_gen: id(chat::HY_NO_GEN),
            },
        };
        Ok(Self { inner, special })
    }

    /// The special token ids of this model.
    pub fn special(&self) -> &SpecialTokens {
        &self.special
    }

    /// Chat template family of this vocabulary, if recognised.
    pub fn chat_format(&self) -> Option<ChatFormat> {
        self.special.family
    }

    /// Like [`Tokenizer::chat_format`] but errors when the family is unknown.
    pub fn require_chat_format(&self) -> Result<ChatFormat> {
        self.special.family.ok_or_else(|| {
            Error::Tokenizer(
                "unrecognised vocabulary: neither the 1.8B (`<｜hy_User｜>`) nor the 7B \
                 (`<|startoftext|>`) chat markers are present"
                    .into(),
            )
        })
    }

    /// Number of tokens in the vocabulary.
    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    /// Encodes plain text.
    ///
    /// Added tokens appearing verbatim in `text` are kept atomic, exactly like
    /// a HuggingFace fast tokenizer with `split_special_tokens=False`.
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        let encoding = self
            .inner
            .encode(text, add_special_tokens)
            .map_err(|e| Error::Tokenizer(e.to_string()))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Encodes a chat prompt, treating template markers as single tokens.
    ///
    /// This does not rely on the tokenizer's own added-token matching, which
    /// makes the chat path independent of `tokenizer.json` specifics. Markers of
    /// both families are handled, whatever the vocabulary contains.
    pub fn encode_with_markers(&self, text: &str) -> Result<Vec<u32>> {
        let mut ids = Vec::new();
        let mut rest = text;
        loop {
            let mut best: Option<(usize, &str)> = None;
            for marker in MARKER_CANDIDATES {
                if self.inner.token_to_id(marker).is_none() {
                    continue;
                }
                if let Some(pos) = rest.find(marker) {
                    if best.map(|(p, _)| pos < p).unwrap_or(true) {
                        best = Some((pos, marker));
                    }
                }
            }
            match best {
                Some((pos, marker)) => {
                    if pos > 0 {
                        ids.extend(self.encode(&rest[..pos], false)?);
                    }
                    let id = self
                        .inner
                        .token_to_id(marker)
                        .ok_or_else(|| Error::Tokenizer(format!("unknown marker '{marker}'")))?;
                    ids.push(id);
                    rest = &rest[pos + marker.len()..];
                }
                None => {
                    if !rest.is_empty() {
                        ids.extend(self.encode(rest, false)?);
                    }
                    break;
                }
            }
        }
        Ok(ids)
    }

    /// Decodes token ids to text.
    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        self.inner
            .decode(ids, skip_special_tokens)
            .map_err(|e| Error::Tokenizer(e.to_string()))
    }

    /// Maps a token string to its id.
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }

    /// Maps an id to its token string.
    pub fn id_to_token(&self, id: u32) -> Option<String> {
        self.inner.id_to_token(id)
    }

    /// Borrows the underlying HuggingFace tokenizer.
    pub fn inner(&self) -> &HfTokenizer {
        &self.inner
    }

    /// Incremental decoder for streaming generation.
    pub fn incremental_decoder(&self) -> IncrementalDecoder<'_> {
        IncrementalDecoder {
            tokenizer: self,
            ids: Vec::new(),
            pending_start: 0,
            text: String::new(),
        }
    }
}

/// Turns generated tokens into text deltas.
///
/// Decoding only the pending tail (instead of the whole sequence on every step)
/// keeps streaming output linear in the number of generated tokens. A tail that
/// ends in U+FFFD is held back until the remaining bytes of the multi-byte
/// character arrive, so emoji/CJK are never split mid-character.
#[derive(Debug)]
pub struct IncrementalDecoder<'t> {
    tokenizer: &'t Tokenizer,
    ids: Vec<u32>,
    pending_start: usize,
    text: String,
}

impl IncrementalDecoder<'_> {
    /// All text emitted so far.
    pub fn text(&self) -> &str {
        &self.text
    }

    /// Number of tokens consumed.
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// Whether nothing was decoded yet.
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// Generated token ids so far.
    pub fn ids(&self) -> &[u32] {
        &self.ids
    }

    /// The tokenizer behind the decoder.
    pub fn tokenizer(&self) -> &Tokenizer {
        self.tokenizer
    }

    /// Forgets every token decoded so far.
    pub fn reset(&mut self) {
        self.ids.clear();
        self.pending_start = 0;
        self.text.clear();
    }

    /// Pushes one generated token, returning a delta when it is safe to show.
    pub fn push(&mut self, id: u32) -> Result<Option<String>> {
        self.ids.push(id);
        let tail = self
            .tokenizer
            .decode(&self.ids[self.pending_start..], true)?;
        if tail.is_empty() || tail.ends_with('\u{FFFD}') {
            return Ok(None);
        }
        self.pending_start = self.ids.len();
        self.text.push_str(&tail);
        Ok(Some(tail))
    }

    /// Flushes anything still pending (end of generation).
    pub fn finish(&mut self) -> Result<Option<String>> {
        if self.pending_start >= self.ids.len() {
            return Ok(None);
        }
        let tail = self
            .tokenizer
            .decode(&self.ids[self.pending_start..], true)?;
        self.pending_start = self.ids.len();
        if tail.is_empty() {
            return Ok(None);
        }
        self.text.push_str(&tail);
        Ok(Some(tail))
    }
}

/// Compatibility wrapper with the `TokenOutputStream` API of `hy-mt` 0.1.
///
/// Unlike the old type it only decodes the pending token tail, so streaming
/// cost is linear in the number of generated tokens instead of quadratic.
#[derive(Debug)]
pub struct TokenOutputStream<'t> {
    inner: IncrementalDecoder<'t>,
}

impl<'t> TokenOutputStream<'t> {
    /// Wraps a tokenizer.
    pub fn new(tokenizer: &'t Tokenizer) -> Self {
        Self {
            inner: tokenizer.incremental_decoder(),
        }
    }

    /// The wrapped tokenizer.
    pub fn tokenizer(&self) -> &Tokenizer {
        self.inner.tokenizer()
    }

    /// Forgets everything decoded so far.
    pub fn clear(&mut self) {
        self.inner.reset();
    }

    /// Consumes one generated token, returning the delta that is safe to show.
    pub fn next_token(&mut self, token: u32) -> Result<Option<String>> {
        self.inner.push(token)
    }

    /// Flushes the pending tail.
    pub fn decode_rest(&mut self) -> Result<Option<String>> {
        self.inner.finish()
    }

    /// Everything decoded so far, including any pending tail.
    pub fn decode_all(&self) -> Result<String> {
        self.inner.tokenizer().decode(self.inner.ids(), true)
    }
}
