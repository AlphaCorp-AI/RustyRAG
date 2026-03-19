use anyhow::Context;
use reqwest::Client;
use serde::{Deserialize, Serialize};

// ── Request/response shapes (OpenAI-compatible) ─────────────────────

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

pub struct ChatResult {
    pub model: String,
    pub content: String,
    pub usage: Option<ApiUsage>,
}

// ── Provider registry ───────────────────────────────────────────────

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

impl Provider {
    fn from_str(raw: &str) -> anyhow::Result<Self> {
        match raw.trim().to_lowercase().as_str() {
            "groq" => Ok(Self::Groq),
            "cerebras" => Ok(Self::Cerebras),
            other => anyhow::bail!("Unsupported LLM provider '{other}' (expected 'groq' or 'cerebras')"),
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Groq => "groq",
            Self::Cerebras => "cerebras",
        }
    }

    fn url(self) -> &'static str {
        match self {
            Self::Groq => "https://api.groq.com/openai/v1/chat/completions",
            Self::Cerebras => "https://api.cerebras.ai/v1/chat/completions",
        }
    }

    fn models(self) -> &'static [&'static str] {
        match self {
            Self::Groq => GROQ_MODELS,
            Self::Cerebras => CEREBRAS_MODELS,
        }
    }
}

// ── Client ──────────────────────────────────────────────────────────

pub struct LlmClient {
    http: Client,
    groq_api_key: String,
    cerebras_api_key: String,
}

/// Resolved provider + API key + validated model, ready for a request.
struct ResolvedRequest<'a> {
    url: &'static str,
    api_key: &'a str,
    model: String,
}

impl LlmClient {
    pub fn new(config: crate::config::Config) -> Self {
        Self {
            http: Client::builder()
                .connect_timeout(std::time::Duration::from_secs(2))
                .pool_idle_timeout(std::time::Duration::from_secs(90))
                .pool_max_idle_per_host(32)
                .build()
                .expect("Failed to build reqwest client for LLM"),
            groq_api_key: config.groq_api_key,
            cerebras_api_key: config.cerebras_api_key,
        }
    }

    /// Resolve provider, validate API key and model in one step.
    fn resolve(&self, provider: &str, model: &str) -> anyhow::Result<ResolvedRequest<'_>> {
        let p = Provider::from_str(provider)?;

        let api_key = match p {
            Provider::Groq => &self.groq_api_key,
            Provider::Cerebras => &self.cerebras_api_key,
        };
        if api_key.is_empty() {
            anyhow::bail!("{}_API_KEY is not configured", p.label().to_uppercase());
        }

        let allowed = p.models();
        if !allowed.contains(&model) {
            anyhow::bail!(
                "Model '{model}' is not supported for provider '{}'. Allowed: {}",
                p.label(),
                allowed.join(", ")
            );
        }

        Ok(ResolvedRequest {
            url: p.url(),
            api_key,
            model: model.to_string(),
        })
    }

    pub fn supported_models() -> Vec<(&'static str, &'static str)> {
        let mut out = Vec::with_capacity(GROQ_MODELS.len() + CEREBRAS_MODELS.len());
        out.extend(GROQ_MODELS.iter().map(|m| ("groq", *m)));
        out.extend(CEREBRAS_MODELS.iter().map(|m| ("cerebras", *m)));
        out
    }

    // ── Public API ──────────────────────────────────────────────────

    pub async fn chat(
        &self,
        message: &str,
        model: &str,
        provider: &str,
        max_tokens: Option<u32>,
    ) -> anyhow::Result<ChatResult> {
        let messages = vec![Message {
            role: "user".into(),
            content: message.into(),
        }];
        self.send_chat(&messages, model, provider, max_tokens).await
    }

    pub async fn chat_with_system(
        &self,
        system_prompt: &str,
        user_message: &str,
        model: &str,
        provider: &str,
        max_tokens: Option<u32>,
    ) -> anyhow::Result<ChatResult> {
        let messages = vec![
            Message { role: "system".into(), content: system_prompt.into() },
            Message { role: "user".into(), content: user_message.into() },
        ];
        self.send_chat(&messages, model, provider, max_tokens).await
    }

    pub async fn chat_stream(
        &self,
        message: &str,
        model: &str,
        provider: &str,
    ) -> anyhow::Result<reqwest::Response> {
        let messages = vec![Message {
            role: "user".into(),
            content: message.into(),
        }];
        self.send_stream(&messages, model, provider).await
    }

    pub async fn chat_stream_with_system(
        &self,
        system_prompt: &str,
        user_message: &str,
        model: &str,
        provider: &str,
    ) -> anyhow::Result<reqwest::Response> {
        let messages = vec![
            Message { role: "system".into(), content: system_prompt.into() },
            Message { role: "user".into(), content: user_message.into() },
        ];
        self.send_stream(&messages, model, provider).await
    }

    // ── Internal ────────────────────────────────────────────────────

    async fn send_chat(
        &self,
        messages: &[Message],
        model: &str,
        provider: &str,
        max_tokens: Option<u32>,
    ) -> anyhow::Result<ChatResult> {
        let r = self.resolve(provider, model)?;

        let body = ChatCompletionRequest {
            model: r.model.clone(),
            messages: messages.to_vec(),
            max_tokens,
            stream: None,
        };

        let res = self
            .http
            .post(r.url)
            .bearer_auth(r.api_key)
            .json(&body)
            .send()
            .await
            .context("Failed to reach LLM provider")?;

        if !res.status().is_success() {
            let text = res.text().await.unwrap_or_default();
            anyhow::bail!("LLM provider returned error: {text}");
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
            model: data.model.unwrap_or(r.model),
            content,
            usage: data.usage,
        })
    }

    async fn send_stream(
        &self,
        messages: &[Message],
        model: &str,
        provider: &str,
    ) -> anyhow::Result<reqwest::Response> {
        let r = self.resolve(provider, model)?;

        let body = ChatCompletionRequest {
            model: r.model,
            messages: messages.to_vec(),
            max_tokens: None,
            stream: Some(true),
        };

        let res = self
            .http
            .post(r.url)
            .bearer_auth(r.api_key)
            .json(&body)
            .send()
            .await
            .context("Failed to reach LLM provider")?;

        if !res.status().is_success() {
            let text = res.text().await.unwrap_or_default();
            anyhow::bail!("LLM provider returned error: {text}");
        }

        Ok(res)
    }
}
