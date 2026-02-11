use serde::Deserialize;
use utoipa::ToSchema;
use validator::Validate;

// ── Chat ────────────────────────────────────────────────────────────

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

// ── Users ───────────────────────────────────────────────────────────

/// POST /api/v1/users/register
#[derive(Debug, Deserialize, Validate, ToSchema)]
pub struct CreateUserRequest {
    /// Display name
    #[validate(length(min = 1, max = 255))]
    #[schema(example = "John Doe")]
    pub name: String,

    /// Unique email address
    #[validate(email)]
    #[schema(example = "john@example.com")]
    pub email: String,

    /// Plain-text password (hashed server-side)
    #[validate(length(min = 8, max = 128))]
    #[schema(example = "supersecret123")]
    pub password: String,
}

/// POST /api/v1/users/login
#[derive(Debug, Deserialize, Validate, ToSchema)]
pub struct LoginRequest {
    #[validate(email)]
    #[schema(example = "john@example.com")]
    pub email: String,

    #[validate(length(min = 1))]
    #[schema(example = "supersecret123")]
    pub password: String,
}
