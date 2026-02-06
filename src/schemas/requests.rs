use serde::Deserialize;
use utoipa::ToSchema;
use validator::Validate;

/// POST /api/v1/chat
#[derive(Debug, Deserialize, Validate, ToSchema)]
pub struct ChatRequest {
    /// The user message to send to the LLM
    #[validate(length(min = 1, max = 10_000))]
    pub message: String,

    /// Override the default model (defaults to openai/gpt-oss-20b)
    #[schema(example = "openai/gpt-oss-20b")]
    pub model: Option<String>,

    /// Max tokens for the completion
    #[schema(example = 2048)]
    pub max_tokens: Option<u32>,
}
