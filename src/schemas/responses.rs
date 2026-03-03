use serde::Serialize;
use utoipa::ToSchema;

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

// ── Documents ──────────────────────────────────────────────────────

#[derive(Debug, Serialize, ToSchema)]
pub struct DocumentUploadResponse {
    pub message: String,
    /// Collection the chunks were inserted into
    pub collection: String,
    /// Number of text chunks created and embedded
    pub total_chunks: usize,
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
    /// 1-based page number (0 if unknown, e.g. for .txt files)
    pub page_number: i64,
    /// LLM-generated context prefix (empty if contextual retrieval was not used)
    #[serde(skip_serializing_if = "String::is_empty")]
    pub context_prefix: String,
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
    /// 1-based page number (0 if unknown)
    pub page_number: i64,
    /// Cosine similarity score (higher = more relevant)
    pub score: f32,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct LlmModelEntry {
    pub provider: String,
    pub model: String,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct LlmModelsResponse {
    pub models: Vec<LlmModelEntry>,
}

// ── Generic ─────────────────────────────────────────────────────────

#[derive(Debug, Serialize, ToSchema)]
pub struct ErrorResponse {
    /// Error description
    pub error: String,
}
