use serde::Deserialize;

#[derive(Debug, Deserialize, Clone)]
pub struct Config {
    pub database_url: String,
    pub jwt_secret: String,
    pub groq_api_key: String,
    pub host: String,
    pub port: u16,

    // ── Milvus / Vector DB ─────────────────────────────────────────
    #[serde(default = "default_milvus_url")]
    pub milvus_url: String,

    // ── LLM model ──────────────────────────────────────────────────
    /// Groq model to use for chat completions
    #[serde(default = "default_llm_model")]
    pub llm_model: String,

    // ── Cohere embeddings ──────────────────────────────────────────
    /// Cohere API key for embeddings
    #[serde(default)]
    pub cohere_api_key: String,
    /// Cohere embedding model name
    #[serde(default = "default_embedding_model")]
    pub embedding_model: String,
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
fn default_llm_model() -> String {
    "llama-3.3-70b-versatile".to_string()
}
fn default_embedding_model() -> String {
    "embed-english-light-v3.0".to_string()
}
fn default_embedding_dimension() -> i64 {
    384
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
