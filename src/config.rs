use serde::Deserialize;

#[derive(Debug, Deserialize, Clone)]
pub struct Config {
    #[serde(default)]
    pub groq_api_key: String,
    #[serde(default)]
    pub cerebras_api_key: String,
    #[serde(default = "default_host")]
    pub host: String,
    #[serde(default = "default_port")]
    pub port: u16,

    // ── Milvus / Vector DB ─────────────────────────────────────────
    #[serde(default = "default_milvus_url")]
    pub milvus_url: String,

    // ── Local/OpenAI-compatible embeddings ─────────────────────────
    #[serde(default)]
    pub embedding_api_key: String,
    #[serde(default = "default_embedding_api_url")]
    pub embedding_api_url: String,
    #[serde(default = "default_embedding_model")]
    pub embedding_model: String,
    #[serde(default = "default_embedding_type")]
    pub embedding_type: String,
    #[serde(default = "default_embedding_task_document")]
    pub embedding_task_document: String,
    #[serde(default = "default_embedding_task_query")]
    pub embedding_task_query: String,
    #[serde(default = "default_embedding_dimension")]
    pub embedding_dimension: i64,
    #[serde(default = "default_embedding_max_batch_size")]
    pub embedding_max_batch_size: usize,

    // ── Milvus index/search defaults ───────────────────────────────
    #[serde(default = "default_milvus_metric_type")]
    pub milvus_metric_type: String,
    #[serde(default = "default_milvus_index_type")]
    pub milvus_index_type: String,
    #[serde(default = "default_milvus_hnsw_m")]
    pub milvus_hnsw_m: i64,
    #[serde(default = "default_milvus_hnsw_ef_construction")]
    pub milvus_hnsw_ef_construction: i64,
    #[serde(default = "default_milvus_search_ef")]
    pub milvus_search_ef: i64,

    // ── Chunking defaults (character-based, semantic splitting) ────
    /// Maximum characters per chunk (overridable per request)
    #[serde(default = "default_chunk_size")]
    pub chunk_size: usize,
    /// Overlap characters prepended from the previous chunk
    #[serde(default = "default_chunk_overlap")]
    pub chunk_overlap: usize,

    // ── Contextual retrieval (opt-in) ──────────────────────────────
    /// Enable LLM-generated context prefixes before embedding
    #[serde(default)]
    pub contextual_retrieval_enabled: bool,
    /// LLM provider for context generation ("groq" or "cerebras")
    #[serde(default = "default_contextual_provider")]
    pub contextual_retrieval_provider: String,
    /// LLM model for context generation
    #[serde(default = "default_contextual_model")]
    pub contextual_retrieval_model: String,
    /// Max concurrent LLM calls for context generation
    #[serde(default = "default_contextual_concurrency")]
    pub contextual_retrieval_concurrency: usize,
    /// Max characters of document text included in each contextual prompt
    #[serde(default = "default_contextual_max_doc_chars")]
    pub contextual_retrieval_max_doc_chars: usize,
}

fn default_host() -> String {
    "127.0.0.1".to_string()
}
fn default_port() -> u16 {
    8080
}
fn default_milvus_url() -> String {
    "http://localhost:19530".to_string()
}
fn default_embedding_api_url() -> String {
    "http://localhost:7997/v1/embeddings".to_string()
}
fn default_embedding_model() -> String {
    "jinaai/jina-embeddings-v5-text-nano-retrieval".to_string()
}
fn default_embedding_type() -> String {
    String::new()
}
fn default_embedding_task_document() -> String {
    String::new()
}
fn default_embedding_task_query() -> String {
    String::new()
}
fn default_embedding_dimension() -> i64 {
    768
}
fn default_embedding_max_batch_size() -> usize {
    8
}
fn default_milvus_metric_type() -> String {
    "COSINE".to_string()
}
fn default_milvus_index_type() -> String {
    "HNSW".to_string()
}
fn default_milvus_hnsw_m() -> i64 {
    16
}
fn default_milvus_hnsw_ef_construction() -> i64 {
    256
}
fn default_milvus_search_ef() -> i64 {
    64
}
fn default_chunk_size() -> usize {
    2000
}
fn default_chunk_overlap() -> usize {
    200
}
fn default_contextual_provider() -> String {
    "groq".to_string()
}
fn default_contextual_model() -> String {
    "llama-3.1-8b-instant".to_string()
}
fn default_contextual_concurrency() -> usize {
    24
}
fn default_contextual_max_doc_chars() -> usize {
    6_000
}

impl Config {
    pub fn from_env() -> Result<Self, envy::Error> {
        dotenvy::dotenv().ok();
        envy::from_env::<Config>()
    }
}
