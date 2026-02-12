use serde::Deserialize;
use utoipa::ToSchema;
use validator::Validate;

// Re-export for utoipa multipart body schema
pub use self::multipart::DocumentUploadBody;

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

// ── Documents ──────────────────────────────────────────────────────

/// Query parameters for POST /api/v1/documents/upload
#[derive(Debug, Deserialize, ToSchema)]
pub struct DocumentUploadQuery {
    /// Target Milvus collection (defaults to "documents")
    #[schema(example = "documents")]
    pub collection_name: Option<String>,

    /// Words per chunk (overrides server default)
    #[schema(example = 500)]
    pub chunk_size: Option<usize>,

    /// Overlap words between consecutive chunks
    #[schema(example = 50)]
    pub chunk_overlap: Option<usize>,
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
}

// ── Chat RAG ────────────────────────────────────────────────────────

/// POST /api/v1/chat-rag
#[derive(Debug, Deserialize, Validate, ToSchema)]
pub struct ChatRagRequest {
    /// The user's question
    #[validate(length(min = 1, max = 10_000))]
    pub message: String,

    /// Milvus collection to search for context
    #[schema(example = "documents")]
    pub collection_name: String,

    /// Number of source chunks to include as context (defaults to 5)
    #[schema(example = 5)]
    pub limit: Option<i64>,
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
