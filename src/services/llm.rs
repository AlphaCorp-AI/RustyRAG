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
    temperature: Option<f32>,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Provider {
    Groq,
    Cerebras,
}

const GROQ_MODELS: &[&str] = &[
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
];

const CEREBRAS_MODELS: &[&str] = &[
    "llama3.1-8b",
    "gpt-oss-120b",
    "qwen-3-235b-a22b-instruct-2507",
    "zai-glm-4.7",
];

impl LlmClient {
    pub fn new(config: Config) -> Self {
        Self {
            http: Client::builder()
                .connect_timeout(std::time::Duration::from_secs(2))
                .pool_idle_timeout(std::time::Duration::from_secs(90))
                .pool_max_idle_per_host(32)
                .build()
                .expect("Failed to build reqwest client for LLM"),
            config,
        }
    }

    fn provider_from_str(raw: &str) -> anyhow::Result<Provider> {
        let provider = raw.trim().to_lowercase();
        match provider.as_str() {
            "groq" => Ok(Provider::Groq),
            "cerebras" => Ok(Provider::Cerebras),
            _ => anyhow::bail!(
                "Unsupported LLM provider '{provider}' (expected 'groq' or 'cerebras')"
            ),
        }
    }

    fn provider_auth_and_url(&self, provider: Provider) -> anyhow::Result<(&str, &str)> {
        match provider {
            Provider::Groq => {
                if self.config.groq_api_key.is_empty() {
                    anyhow::bail!("GROQ_API_KEY is not configured");
                }
                Ok((
                    "https://api.groq.com/openai/v1/chat/completions",
                    self.config.groq_api_key.as_str(),
                ))
            }
            Provider::Cerebras => {
                if self.config.cerebras_api_key.is_empty() {
                    anyhow::bail!("CEREBRAS_API_KEY is not configured");
                }
                Ok((
                    "https://api.cerebras.ai/v1/chat/completions",
                    self.config.cerebras_api_key.as_str(),
                ))
            }
        }
    }

    fn validate_model_for_provider(&self, provider: Provider, model: &str) -> anyhow::Result<()> {
        let allowed = match provider {
            Provider::Groq => GROQ_MODELS,
            Provider::Cerebras => CEREBRAS_MODELS,
        };
        if allowed.contains(&model) {
            return Ok(());
        }
        anyhow::bail!(
            "Model '{model}' is not supported for provider '{}'. Allowed models: {}",
            match provider {
                Provider::Groq => "groq",
                Provider::Cerebras => "cerebras",
            },
            allowed.join(", ")
        );
    }

    pub fn supported_models() -> Vec<(&'static str, &'static str)> {
        let mut out = Vec::with_capacity(GROQ_MODELS.len() + CEREBRAS_MODELS.len());
        out.extend(GROQ_MODELS.iter().map(|m| ("groq", *m)));
        out.extend(CEREBRAS_MODELS.iter().map(|m| ("cerebras", *m)));
        out
    }

    /// Send a chat completion request via the selected provider.
    pub async fn chat(
        &self,
        message: &str,
        model: &str,
        provider: &str,
        max_tokens: Option<u32>,
    ) -> anyhow::Result<ChatResult> {
        self.chat_with_messages(
            &[Message {
                role: "user".into(),
                content: message.into(),
            }],
            model,
            provider,
            max_tokens,
        )
        .await
    }

    /// Chat with a system prompt (used for RAG).
    pub async fn chat_with_system(
        &self,
        system_prompt: &str,
        user_message: &str,
        model: &str,
        provider: &str,
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
            provider,
            max_tokens,
        )
        .await
    }

    /// Stream a chat completion — returns the raw SSE response from provider.
    pub async fn chat_stream(
        &self,
        message: &str,
        model: &str,
        provider: &str,
    ) -> anyhow::Result<reqwest::Response> {
        self.stream_messages(
            &[Message {
                role: "user".into(),
                content: message.into(),
            }],
            model,
            provider,
            None,
            None,
        )
        .await
    }

    /// Stream a chat completion with a system prompt (used for RAG streaming).
    pub async fn chat_stream_with_system(
        &self,
        system_prompt: &str,
        user_message: &str,
        model: &str,
        provider: &str,
    ) -> anyhow::Result<reqwest::Response> {
        self.stream_messages(
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
            provider,
            None,
            None,
        )
        .await
    }

    /// Stream with explicit temperature and max_tokens control (used for competition).
    pub async fn chat_stream_with_options(
        &self,
        system_prompt: &str,
        user_message: &str,
        model: &str,
        provider: &str,
        temperature: Option<f32>,
        max_tokens: Option<u32>,
    ) -> anyhow::Result<reqwest::Response> {
        self.stream_messages(
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
            provider,
            temperature,
            max_tokens,
        )
        .await
    }

    /// Internal: stream arbitrary messages, returning the raw SSE response.
    async fn stream_messages(
        &self,
        messages: &[Message],
        model: &str,
        provider: &str,
        temperature: Option<f32>,
        max_tokens: Option<u32>,
    ) -> anyhow::Result<reqwest::Response> {
        let resolved_provider = Self::provider_from_str(provider)?;
        let (url, api_key) = self.provider_auth_and_url(resolved_provider)?;
        let model_id: String = model.to_string();
        self.validate_model_for_provider(resolved_provider, &model_id)?;

        let body = ChatCompletionRequest {
            model: model_id,
            messages: messages.to_vec(),
            max_tokens,
            temperature,
            stream: Some(true),
        };

        let res = self
            .http
            .post(url)
            .bearer_auth(api_key)
            .json(&body)
            .send()
            .await
            .context("Failed to reach LLM provider")?;

        let status = res.status();
        if !status.is_success() {
            let text = res.text().await.unwrap_or_default();
            anyhow::bail!("LLM provider returned {status}: {text}");
        }

        Ok(res)
    }

    /// Internal: send arbitrary messages to the LLM.
    async fn chat_with_messages(
        &self,
        messages: &[Message],
        model: &str,
        provider: &str,
        max_tokens: Option<u32>,
    ) -> anyhow::Result<ChatResult> {
        let resolved_provider = Self::provider_from_str(provider)?;
        let (url, api_key) = self.provider_auth_and_url(resolved_provider)?;
        let model_id: String = model.to_string();
        self.validate_model_for_provider(resolved_provider, &model_id)?;

        let body = ChatCompletionRequest {
            model: model_id.clone(),
            messages: messages.to_vec(),
            max_tokens,
            temperature: None,
            stream: None,
        };

        let res = self
            .http
            .post(url)
            .bearer_auth(api_key)
            .json(&body)
            .send()
            .await
            .context("Failed to reach LLM provider")?;

        let status = res.status();
        if !status.is_success() {
            let text = res.text().await.unwrap_or_default();
            anyhow::bail!("LLM provider returned {status}: {text}");
        }

        let data: ChatCompletionResponse = res
            .json()
            .await
            .context("Failed to parse LLM provider response")?;

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
