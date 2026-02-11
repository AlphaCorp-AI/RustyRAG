use chrono::{DateTime, Utc};
use serde::Serialize;
use utoipa::ToSchema;
use uuid::Uuid;

// ── Chat ────────────────────────────────────────────────────────────

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

// ── Health ──────────────────────────────────────────────────────────

#[derive(Debug, Serialize, ToSchema)]
pub struct HealthResponse {
    #[schema(example = "ok")]
    pub status: String,
    #[schema(example = "0.1.0")]
    pub version: String,
}

// ── Users ───────────────────────────────────────────────────────────

#[derive(Debug, Serialize, ToSchema)]
pub struct UserResponse {
    pub id: Uuid,
    pub name: String,
    pub email: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct TokenResponse {
    /// Bearer JWT token
    pub access_token: String,
    pub token_type: String,
}

// ── Generic ─────────────────────────────────────────────────────────

#[derive(Debug, Serialize, ToSchema)]
pub struct ErrorResponse {
    /// Error description
    pub error: String,
}
