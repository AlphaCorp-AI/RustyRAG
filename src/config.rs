use serde::Deserialize;

#[derive(Debug, Deserialize, Clone)]
pub struct Config {
    pub database_url: String,
    pub jwt_secret: String,
    pub openrouter_api_key: String,
    pub host: String,
    pub port: u16,

    // ── Milvus / Vector DB ─────────────────────────────────────────
    #[serde(default = "default_milvus_url")]
    pub milvus_url: String,

    // ── Embedding model (OpenAI-compatible API) ────────────────────
    /// e.g. "text-embedding-3-small", "BAAI/bge-small-en-v1.5", etc.
    #[serde(default)]
    pub embedding_model: String,
    /// Full URL for the embeddings endpoint, e.g. https://api.openai.com/v1/embeddings
    #[serde(default)]
    pub embedding_api_url: String,
    /// Bearer token for the embedding API
    #[serde(default)]
    pub embedding_api_key: String,
    /// Dimensionality of the embedding vectors (must match the model)
    #[serde(default = "default_embedding_dimension")]
    pub embedding_dimension: i64,

    // ── Chunking defaults ──────────────────────────────────────────
    /// Number of words per chunk (overridable per request)
    #[serde(default = "default_chunk_size")]
    pub chunk_size: usize,
    /// Number of overlapping words between consecutive chunks
    #[serde(default = "default_chunk_overlap")]
    pub chunk_overlap: usize,
}

fn default_milvus_url() -> String {
    "http://localhost:19530".to_string()
}
fn default_embedding_dimension() -> i64 {
    3072
}
fn default_chunk_size() -> usize {
    500
}
fn default_chunk_overlap() -> usize {
    50
}

impl Config {
    pub fn from_env() -> Result<Self, envy::Error> {
        dotenvy::dotenv().ok();
        envy::from_env::<Config>()
    }
}
