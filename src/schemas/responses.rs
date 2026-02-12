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

// ── Documents ──────────────────────────────────────────────────────

#[derive(Debug, Serialize, ToSchema)]
pub struct DocumentUploadResponse {
    pub message: String,
    /// Collection the chunks were inserted into
    pub collection: String,
    /// Number of text chunks created and embedded
    pub total_chunks: usize,
    /// Words per chunk that was used
    pub chunk_size: usize,
    /// Overlap words between chunks
    pub chunk_overlap: usize,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct DocumentSearchResponse {
    pub query: String,
    pub collection: String,
    pub results: Vec<DocumentSearchHit>,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct DocumentSearchHit {
    pub text: String,
    pub source_file: String,
    pub chunk_index: i64,
    /// Cosine similarity score (higher = more relevant)
    pub score: f32,
}

// ── Chat RAG ────────────────────────────────────────────────────────

#[derive(Debug, Serialize, ToSchema)]
pub struct ChatRagResponse {
    /// The model that generated the response
    #[schema(example = "openai/gpt-oss-20b")]
    pub model: String,
    /// The LLM's answer grounded in the retrieved context
    pub message: String,
    /// Source chunks that were used as context
    pub sources: Vec<RagSource>,
    /// Token usage breakdown (if available)
    pub usage: Option<Usage>,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct RagSource {
    /// Text content of the source chunk
    pub text: String,
    /// Original file the chunk came from
    pub source_file: String,
    /// Position of the chunk within the file
    pub chunk_index: i64,
    /// Cosine similarity score (higher = more relevant)
    pub score: f32,
}

// ── Generic ─────────────────────────────────────────────────────────

#[derive(Debug, Serialize, ToSchema)]
pub struct ErrorResponse {
    /// Error description
    pub error: String,
}
