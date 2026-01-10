use serde::{Deserialize, Serialize};

pub trait Prompter {
    fn to_prompt(&self) -> String;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    role: String,
    content: String,
}

impl Default for Message {
    fn default() -> Self {
        Self {
            role: "user".to_string(),
            content: "".to_string(),
        }
    }
}

impl Prompter for Message {
    fn to_prompt(&self) -> String {
        format!("<｜hy_begin▁of▁sentence｜><｜hy_User｜>{}<｜hy_place▁holder▁no▁8｜>", self.content)
    }
}

impl From<String> for Message {
    fn from(s: String) -> Self {
        Self {
            role: "user".to_string(),
            content: s,
        }
    }
}

impl From<&str> for Message {
    fn from(s: &str) -> Self {
        Self {
            role: "user".to_string(),
            content: s.to_string(),
        }
    }
}

impl Prompter for Vec<Message> {
    fn to_prompt(&self) -> String {
        self.iter()
            .map(|msg| msg.to_prompt())
            .collect::<Vec<_>>()
            .join("\n")
    }
}

impl Prompter for String {
    fn to_prompt(&self) -> String {
        Message::from(self.clone()).to_prompt()
    }
}

impl Prompter for Vec<String> {
    fn to_prompt(&self) -> String {
        self.iter().map(|s| s.to_prompt()).collect::<Vec<_>>().join("\n")
    }
}

impl Prompter for &str {
    fn to_prompt(&self) -> String {
        Message::from(self.to_string()).to_prompt()
    }
}


mod tests {
     use super::*;

     #[test]
     fn test_to_prompt () { 
          let message ="Hello, how are you?";
          assert_eq!(message.to_prompt(), "<｜hy_begin▁of▁sentence｜><｜hy_User｜>Hello, how are you?<｜hy_place▁holder▁no▁8｜>");
     }
}