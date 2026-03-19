use serde::Serialize;
use utoipa::ToSchema;

use crate::services::llm::ApiUsage;
use crate::services::milvus::SearchResult;

// ── Chat ────────────────────────────────────────────────────────────

#[derive(Debug, Serialize, ToSchema)]
pub struct ChatResponse {
    #[schema(example = "openai/gpt-oss-20b")]
    pub model: String,
    pub message: String,
    pub usage: Option<Usage>,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

impl From<ApiUsage> for Usage {
    fn from(u: ApiUsage) -> Self {
        Self {
            prompt_tokens: u.prompt_tokens,
            completion_tokens: u.completion_tokens,
            total_tokens: u.total_tokens,
        }
    }
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
    pub collection: String,
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
    pub id: i64,
    pub text: String,
    pub file_name: String,
    pub file_size: i64,
    pub chunk_index: i64,
    /// 1-based page number (0 if unknown)
    pub page_number: i64,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub context_prefix: String,
    pub score: f32,
}

impl From<SearchResult> for DocumentSearchHit {
    fn from(h: SearchResult) -> Self {
        Self {
            id: h.id,
            text: h.text,
            file_name: h.file_name,
            file_size: h.file_size,
            chunk_index: h.chunk_index,
            page_number: h.page_number,
            context_prefix: h.context_prefix,
            score: h.score,
        }
    }
}

// ── Chat RAG ────────────────────────────────────────────────────────

#[derive(Debug, Serialize, ToSchema)]
pub struct ChatRagResponse {
    #[schema(example = "openai/gpt-oss-20b")]
    pub model: String,
    pub message: String,
    pub sources: Vec<RagSource>,
    pub usage: Option<Usage>,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct RagSource {
    pub id: i64,
    pub text: String,
    pub file_name: String,
    pub file_size: i64,
    pub chunk_index: i64,
    pub page_number: i64,
    pub score: f32,
}

impl From<&SearchResult> for RagSource {
    fn from(h: &SearchResult) -> Self {
        Self {
            id: h.id,
            text: h.text.clone(),
            file_name: h.file_name.clone(),
            file_size: h.file_size,
            chunk_index: h.chunk_index,
            page_number: h.page_number,
            score: h.score,
        }
    }
}

// ── LLM Models ──────────────────────────────────────────────────────

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
    pub error: String,
}
