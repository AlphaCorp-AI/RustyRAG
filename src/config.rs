use serde::Deserialize;

#[derive(Debug, Deserialize, Clone)]
pub struct Config {
    // ── Required ─────────────────────────────────────────────────────
    pub host: String,
    pub port: u16,
    #[serde(default)]
    pub groq_api_key: String,
    #[serde(default)]
    pub cerebras_api_key: String,

    // ── Optional — services (sensible defaults for local docker) ─────
    #[serde(default = "defaults::milvus_url")]
    pub milvus_url: String,
    #[serde(default = "defaults::embedding_api_url")]
    pub embedding_api_url: String,
    #[serde(default = "defaults::embedding_model")]
    pub embedding_model: String,
    #[serde(default)]
    pub embedding_api_key: String,
    #[serde(default)]
    pub reranker_api_url: String,
    #[serde(default = "defaults::docling_url")]
    pub docling_url: String,

    // Vision model for describing images found in documents
    #[serde(default = "defaults::vision_model")]
    pub vision_model: String,

    // ── Optional — tuning (rarely need to change) ────────────────────
    #[serde(default)]
    pub cors_allowed_origins: String,
    #[serde(default = "defaults::embedding_dimension")]
    pub embedding_dimension: i64,
    #[serde(default = "defaults::embedding_max_batch_size")]
    pub embedding_max_batch_size: usize,
    #[serde(default)]
    pub embedding_type: String,
    #[serde(default)]
    pub embedding_task_document: String,
    #[serde(default)]
    pub embedding_task_query: String,

    // Retrieval pipeline
    #[serde(default = "defaults::retrieval_limit")]
    pub retrieval_limit: i64,
    #[serde(default = "defaults::rerank_top_n")]
    pub rerank_top_n: usize,
    #[serde(default = "defaults::chunk_size")]
    pub chunk_size: usize,
    #[serde(default = "defaults::chunk_overlap")]
    pub chunk_overlap: usize,

    // Milvus index tuning
    #[serde(default = "defaults::milvus_metric_type")]
    pub milvus_metric_type: String,
    #[serde(default = "defaults::milvus_index_type")]
    pub milvus_index_type: String,
    #[serde(default = "defaults::milvus_hnsw_m")]
    pub milvus_hnsw_m: i64,
    #[serde(default = "defaults::milvus_hnsw_ef_construction")]
    pub milvus_hnsw_ef_construction: i64,
    #[serde(default = "defaults::milvus_search_ef")]
    pub milvus_search_ef: i64,

    // Contextual retrieval (opt-in)
    #[serde(default)]
    pub contextual_retrieval_enabled: bool,
    #[serde(default = "defaults::contextual_provider")]
    pub contextual_retrieval_provider: String,
    #[serde(default = "defaults::contextual_model")]
    pub contextual_retrieval_model: String,
    #[serde(default = "defaults::contextual_concurrency")]
    pub contextual_retrieval_concurrency: usize,
    #[serde(default = "defaults::contextual_max_doc_chars")]
    pub contextual_retrieval_max_doc_chars: usize,
}

impl Config {
    pub fn from_env() -> Result<Self, envy::Error> {
        dotenvy::dotenv().ok();
        envy::from_env::<Config>()
    }
}

mod defaults {
    // Services
    pub fn milvus_url() -> String { "http://localhost:19530".into() }
    pub fn embedding_api_url() -> String { "http://localhost:7997/v1/embeddings".into() }
    pub fn embedding_model() -> String { "jinaai/jina-embeddings-v5-text-small-retrieval".into() }
    pub fn docling_url() -> String { "http://localhost:5001".into() }
    pub fn vision_model() -> String { "meta-llama/llama-4-scout-17b-16e-instruct".into() }

    // Embedding tuning
    pub fn embedding_dimension() -> i64 { 1024 }
    pub fn embedding_max_batch_size() -> usize { 8 }

    // Retrieval pipeline
    pub fn retrieval_limit() -> i64 { 20 }
    pub fn rerank_top_n() -> usize { 3 }
    pub fn chunk_size() -> usize { 2000 }
    pub fn chunk_overlap() -> usize { 200 }

    // Milvus index
    pub fn milvus_metric_type() -> String { "COSINE".into() }
    pub fn milvus_index_type() -> String { "HNSW".into() }
    pub fn milvus_hnsw_m() -> i64 { 16 }
    pub fn milvus_hnsw_ef_construction() -> i64 { 256 }
    pub fn milvus_search_ef() -> i64 { 64 }

    // Contextual retrieval
    pub fn contextual_provider() -> String { "groq".into() }
    pub fn contextual_model() -> String { "llama-3.1-8b-instant".into() }
    pub fn contextual_concurrency() -> usize { 24 }
    pub fn contextual_max_doc_chars() -> usize { 6_000 }
}
