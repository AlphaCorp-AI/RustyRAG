use serde::Deserialize;
use utoipa::ToSchema;
use validator::Validate;

// Re-export for utoipa multipart body schema
pub use self::multipart::DocumentUploadBody;

// ── Documents ──────────────────────────────────────────────────────

/// Query parameters for POST /api/v1/documents/upload
#[derive(Debug, Deserialize, ToSchema)]
pub struct DocumentUploadQuery {
    /// Target Milvus collection (defaults to "documents")
    #[schema(example = "documents")]
    pub collection_name: Option<String>,
}

/// POST /api/v1/documents/search
#[derive(Debug, Deserialize, Validate, ToSchema)]
pub struct DocumentSearchRequest {
    /// Natural-language search query
    #[validate(length(min = 1, max = 10_000))]
    pub query: String,

    /// Collection to search (defaults to "documents")
    #[schema(example = "documents")]
    pub collection_name: Option<String>,

    /// Max results to return (defaults to 10)
    #[schema(example = 10)]
    pub limit: Option<i64>,

    /// Optional embedding type sent to the embedding server (e.g. "float")
    #[schema(example = "float")]
    pub embedding_type: Option<String>,

    /// Optional override for Milvus search `ef` parameter
    #[schema(example = 64)]
    pub milvus_search_ef: Option<i64>,
}

// ── Chat RAG ────────────────────────────────────────────────────────

/// POST /api/v1/chat-rag
#[derive(Debug, Deserialize, Validate, ToSchema)]
pub struct ChatRagRequest {
    /// The user's question
    #[validate(length(min = 1, max = 10_000))]
    pub message: String,

    /// Milvus collection to search for context (defaults to "documents")
    #[schema(example = "documents")]
    pub collection_name: Option<String>,

    /// Number of source chunks to include as context (defaults to 5)
    #[schema(example = 5)]
    pub limit: Option<i64>,

    /// Model to use for this request
    #[validate(length(min = 1))]
    #[schema(example = "llama-3.1-8b")]
    pub model: String,

    /// Provider to use for this request ("groq" or "cerebras")
    #[validate(length(min = 1))]
    #[schema(example = "cerebras")]
    pub provider: String,

    /// Optional embedding type sent to the embedding server (e.g. "float")
    #[schema(example = "float")]
    pub embedding_type: Option<String>,

    /// Optional override for Milvus search `ef` parameter
    #[schema(example = 64)]
    pub milvus_search_ef: Option<i64>,
}

/// Schema-only struct so Swagger renders a file picker for the upload endpoint.
mod multipart {
    use utoipa::ToSchema;

    #[derive(Debug, ToSchema)]
    #[allow(dead_code)]
    pub struct DocumentUploadBody {
        /// The file to upload (.txt, .pdf, or .zip)
        #[schema(format = "binary")]
        pub file: String,
    }
}
