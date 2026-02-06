use serde::Serialize;
use utoipa::ToSchema;

#[derive(Debug, Serialize, ToSchema)]
pub struct ChatResponse {
    /// The model that generated the response
    #[schema(example = "openai/gpt-oss-20b")]
    pub model: String,
    /// The LLM's reply
    pub message: String,
    /// Token usage breakdown (if available)
    pub usage: Option<Usage>,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct HealthResponse {
    #[schema(example = "ok")]
    pub status: String,
    #[schema(example = "0.1.0")]
    pub version: String,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct ErrorResponse {
    /// Error description
    pub error: String,
}
