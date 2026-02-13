use anyhow::Context;
use reqwest::Client;
use serde::{Deserialize, Serialize};

use crate::config::Config;

// ── Groq request/response shapes ────────────────────────────────────

#[derive(Serialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct Message {
    pub role: String,
    pub content: String,
}

#[derive(Deserialize)]
struct ChatCompletionResponse {
    choices: Vec<Choice>,
    model: Option<String>,
    usage: Option<ApiUsage>,
}

#[derive(Deserialize)]
struct Choice {
    message: ChoiceMessage,
}

#[derive(Deserialize)]
struct ChoiceMessage {
    content: Option<String>,
}

#[derive(Deserialize)]
pub struct ApiUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

// ── Public result type ──────────────────────────────────────────────

pub struct ChatResult {
    pub model: String,
    pub content: String,
    pub usage: Option<ApiUsage>,
}

// ── Client ──────────────────────────────────────────────────────────

pub struct LlmClient {
    http: Client,
    config: Config,
}

impl LlmClient {
    pub fn new(config: Config) -> Self {
        Self {
            http: Client::new(),
            config,
        }
    }

    /// Send a chat completion request via Groq.
    pub async fn chat(
        &self,
        message: &str,
        model: Option<&str>,
        max_tokens: Option<u32>,
    ) -> anyhow::Result<ChatResult> {
        self.chat_with_messages(
            &[Message {
                role: "user".into(),
                content: message.into(),
            }],
            model,
            max_tokens,
        )
        .await
    }

    /// Chat with a system prompt (used for RAG).
    pub async fn chat_with_system(
        &self,
        system_prompt: &str,
        user_message: &str,
        model: Option<&str>,
        max_tokens: Option<u32>,
    ) -> anyhow::Result<ChatResult> {
        self.chat_with_messages(
            &[
                Message {
                    role: "system".into(),
                    content: system_prompt.into(),
                },
                Message {
                    role: "user".into(),
                    content: user_message.into(),
                },
            ],
            model,
            max_tokens,
        )
        .await
    }

    /// Stream a chat completion — returns the raw SSE response from Groq.
    pub async fn chat_stream(&self, message: &str) -> anyhow::Result<reqwest::Response> {
        self.stream_messages(&[Message {
            role: "user".into(),
            content: message.into(),
        }])
        .await
    }

    /// Stream a chat completion with a system prompt (used for RAG streaming).
    pub async fn chat_stream_with_system(
        &self,
        system_prompt: &str,
        user_message: &str,
    ) -> anyhow::Result<reqwest::Response> {
        self.stream_messages(&[
            Message {
                role: "system".into(),
                content: system_prompt.into(),
            },
            Message {
                role: "user".into(),
                content: user_message.into(),
            },
        ])
        .await
    }

    /// Internal: stream arbitrary messages, returning the raw SSE response.
    async fn stream_messages(&self, messages: &[Message]) -> anyhow::Result<reqwest::Response> {
        let body = ChatCompletionRequest {
            model: self.config.llm_model.clone(),
            messages: messages.to_vec(),
            max_tokens: None,
            stream: Some(true),
        };

        let res = self
            .http
            .post("https://api.groq.com/openai/v1/chat/completions")
            .bearer_auth(&self.config.groq_api_key)
            .json(&body)
            .send()
            .await
            .context("Failed to reach Groq")?;

        let status = res.status();
        if !status.is_success() {
            let text = res.text().await.unwrap_or_default();
            anyhow::bail!("Groq returned {status}: {text}");
        }

        Ok(res)
    }

    /// Internal: send arbitrary messages to the LLM.
    async fn chat_with_messages(
        &self,
        messages: &[Message],
        model: Option<&str>,
        max_tokens: Option<u32>,
    ) -> anyhow::Result<ChatResult> {
        let model_id: String = model
            .map(|m| m.to_string())
            .unwrap_or_else(|| self.config.llm_model.clone());

        let body = ChatCompletionRequest {
            model: model_id.clone(),
            messages: messages.to_vec(),
            max_tokens,
            stream: None,
        };

        let res = self
            .http
            .post("https://api.groq.com/openai/v1/chat/completions")
            .bearer_auth(&self.config.groq_api_key)
            .json(&body)
            .send()
            .await
            .context("Failed to reach Groq")?;

        let status = res.status();
        if !status.is_success() {
            let text = res.text().await.unwrap_or_default();
            anyhow::bail!("Groq returned {status}: {text}");
        }

        let data: ChatCompletionResponse = res
            .json()
            .await
            .context("Failed to parse Groq response")?;

        let content = data
            .choices
            .first()
            .and_then(|c| c.message.content.clone())
            .unwrap_or_default();

        Ok(ChatResult {
            model: data.model.unwrap_or(model_id),
            content,
            usage: data.usage,
        })
    }
}
