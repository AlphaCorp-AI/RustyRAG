use anyhow::Context;
use reqwest::Client;
use serde::{Deserialize, Serialize};

use crate::config::Config;

const DEFAULT_MODEL: &str = "openai/gpt-oss-20b";

// ── OpenRouter request/response shapes ──────────────────────────────

#[derive(Serialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    provider: Option<ProviderPreferences>,
}

#[derive(Serialize)]
struct ProviderPreferences {
    order: Vec<String>,
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

    /// Send a chat completion request via OpenRouter.
    /// Defaults to `openai/gpt-oss-20b` routed through Groq.
    pub async fn chat(
        &self,
        message: &str,
        model: Option<&str>,
        max_tokens: Option<u32>,
    ) -> anyhow::Result<ChatResult> {
        let model_id: String = model.unwrap_or(DEFAULT_MODEL).to_string();

        let body = ChatCompletionRequest {
            model: model_id.clone(),
            messages: vec![Message {
                role: "user".into(),
                content: message.into(),
            }],
            max_tokens,
            provider: Some(ProviderPreferences {
                order: vec!["Groq".into()],
            }),
        };

        let res = self
            .http
            .post("https://openrouter.ai/api/v1/chat/completions")
            .bearer_auth(&self.config.openrouter_api_key)
            .header("HTTP-Referer", "https://alpharust.local")
            .header("X-Title", "alpharust")
            .json(&body)
            .send()
            .await
            .context("Failed to reach OpenRouter")?;

        let status = res.status();
        if !status.is_success() {
            let text = res.text().await.unwrap_or_default();
            anyhow::bail!("OpenRouter returned {status}: {text}");
        }

        let data: ChatCompletionResponse = res
            .json()
            .await
            .context("Failed to parse OpenRouter response")?;

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
