use anyhow::Context;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::json;

pub const DEFAULT_COLLECTION: &str = "documents";

// ── Milvus REST API v2 client ──────────────────────────────────────

#[derive(Clone)]
pub struct MilvusClient {
    http: Client,
    base_url: String,
    dimension: i64,
    metric_type: String,
    index_type: String,
    hnsw_m: i64,
    hnsw_ef_construction: i64,
    default_search_ef: i64,
}

/// A single document chunk ready to be inserted into Milvus.
#[derive(Debug, Clone, Serialize)]
pub struct DocumentChunk {
    pub text: String,
    pub source_file: String,
    pub chunk_index: i64,
    pub page_number: i64,
    pub context_prefix: String,
    pub embedding: Vec<f32>,
}

/// A single search hit returned by Milvus.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub text: String,
    pub source_file: String,
    pub chunk_index: i64,
    pub page_number: i64,
    pub context_prefix: String,
    pub score: f32,
}

#[derive(Debug, Clone)]
pub struct SearchOptions {
    pub ef: Option<i64>,
}

// ── Milvus REST API response envelope ──────────────────────────────

#[derive(Debug, Deserialize)]
struct MilvusResponse {
    code: i64,
    #[serde(default)]
    message: String,
    #[serde(default)]
    data: serde_json::Value,
}

impl MilvusClient {
    pub fn new(
        base_url: &str,
        dimension: i64,
        metric_type: &str,
        index_type: &str,
        hnsw_m: i64,
        hnsw_ef_construction: i64,
        default_search_ef: i64,
    ) -> Self {
        Self {
            http: Client::builder()
                .connect_timeout(std::time::Duration::from_secs(2))
                .pool_idle_timeout(std::time::Duration::from_secs(90))
                .pool_max_idle_per_host(32)
                .build()
                .expect("Failed to build reqwest client for Milvus"),
            base_url: base_url.trim_end_matches('/').to_string(),
            dimension,
            metric_type: metric_type.to_string(),
            index_type: index_type.to_string(),
            hnsw_m,
            hnsw_ef_construction,
            default_search_ef,
        }
    }

    // ── Collection management ──────────────────────────────────────

    /// Ensure a collection exists (create if missing).
    pub async fn ensure_collection(&self, collection_name: &str) -> anyhow::Result<()> {
        if !self.has_collection(collection_name).await? {
            self.create_collection(collection_name).await?;
            tracing::info!("Created Milvus collection '{collection_name}'");
        }
        Ok(())
    }

    async fn has_collection(&self, name: &str) -> anyhow::Result<bool> {
        let url = format!("{}/v2/vectordb/collections/has", self.base_url);
        let body = json!({ "collectionName": name });

        let resp: MilvusResponse = self
            .http
            .post(&url)
            .json(&body)
            .send()
            .await
            .context("Milvus: failed to check collection")?
            .json()
            .await
            .context("Milvus: bad response from has_collection")?;

        if resp.code != 0 {
            anyhow::bail!("Milvus has_collection error: {}", resp.message);
        }

        Ok(resp.data["has"].as_bool().unwrap_or(false))
    }

    async fn create_collection(&self, name: &str) -> anyhow::Result<()> {
        let url = format!("{}/v2/vectordb/collections/create", self.base_url);

        let body = json!({
            "collectionName": name,
            "schema": {
                "autoId": true,
                "enableDynamicField": false,
                "fields": [
                    {
                        "fieldName": "id",
                        "dataType": "Int64",
                        "isPrimary": true
                    },
                    {
                        "fieldName": "text",
                        "dataType": "VarChar",
                        "elementTypeParams": { "max_length": "65535" }
                    },
                    {
                        "fieldName": "source_file",
                        "dataType": "VarChar",
                        "elementTypeParams": { "max_length": "1024" }
                    },
                    {
                        "fieldName": "chunk_index",
                        "dataType": "Int64"
                    },
                    {
                        "fieldName": "page_number",
                        "dataType": "Int64"
                    },
                    {
                        "fieldName": "context_prefix",
                        "dataType": "VarChar",
                        "elementTypeParams": { "max_length": "2048" }
                    },
                    {
                        "fieldName": "embedding",
                        "dataType": "FloatVector",
                        "elementTypeParams": { "dim": self.dimension.to_string() }
                    }
                ]
            },
            "indexParams": [
                {
                    "fieldName": "embedding",
                    "indexName": "embedding_idx",
                    "metricType": self.metric_type,
                    "params": {
                        "index_type": self.index_type,
                        "M": self.hnsw_m,
                        "efConstruction": self.hnsw_ef_construction
                    }
                }
            ]
        });

        let resp: MilvusResponse = self
            .http
            .post(&url)
            .json(&body)
            .send()
            .await
            .context("Milvus: failed to create collection")?
            .json()
            .await
            .context("Milvus: bad response from create_collection")?;

        if resp.code != 0 {
            anyhow::bail!("Milvus create_collection error: {}", resp.message);
        }

        Ok(())
    }

    // ── Insert ─────────────────────────────────────────────────────

    pub async fn insert(
        &self,
        collection_name: &str,
        chunks: Vec<DocumentChunk>,
    ) -> anyhow::Result<usize> {
        let url = format!("{}/v2/vectordb/entities/insert", self.base_url);

        let data: Vec<serde_json::Value> = chunks
            .iter()
            .map(|c| {
                json!({
                    "text": c.text,
                    "source_file": c.source_file,
                    "chunk_index": c.chunk_index,
                    "page_number": c.page_number,
                    "context_prefix": c.context_prefix,
                    "embedding": c.embedding,
                })
            })
            .collect();

        let count = data.len();

        let body = json!({
            "collectionName": collection_name,
            "data": data,
        });

        let resp: MilvusResponse = self
            .http
            .post(&url)
            .json(&body)
            .send()
            .await
            .context("Milvus: failed to insert")?
            .json()
            .await
            .context("Milvus: bad response from insert")?;

        if resp.code != 0 {
            anyhow::bail!("Milvus insert error: {}", resp.message);
        }

        Ok(count)
    }

    // ── Search ─────────────────────────────────────────────────────

    pub async fn search(
        &self,
        collection_name: &str,
        embedding: Vec<f32>,
        limit: i64,
        options: Option<SearchOptions>,
    ) -> anyhow::Result<Vec<SearchResult>> {
        let url = format!("{}/v2/vectordb/entities/search", self.base_url);
        let ef = options.and_then(|v| v.ef).unwrap_or(self.default_search_ef);

        let body = json!({
            "collectionName": collection_name,
            "data": [embedding],
            "limit": limit,
            "outputFields": ["text", "source_file", "chunk_index", "page_number", "context_prefix"],
            "searchParams": {
                "params": { "ef": ef }
            },
        });

        let resp: MilvusResponse = self
            .http
            .post(&url)
            .json(&body)
            .send()
            .await
            .context("Milvus: failed to search")?
            .json()
            .await
            .context("Milvus: bad response from search")?;

        if resp.code != 0 {
            anyhow::bail!("Milvus search error: {}", resp.message);
        }

        let results = resp
            .data
            .as_array()
            .map(|arr| {
                arr.iter()
                    .map(|item| SearchResult {
                        text: item["text"].as_str().unwrap_or("").to_string(),
                        source_file: item["source_file"].as_str().unwrap_or("").to_string(),
                        chunk_index: item["chunk_index"].as_i64().unwrap_or(0),
                        page_number: item["page_number"].as_i64().unwrap_or(0),
                        context_prefix: item["context_prefix"].as_str().unwrap_or("").to_string(),
                        score: item["distance"].as_f64().unwrap_or(0.0) as f32,
                    })
                    .collect()
            })
            .unwrap_or_default();

        Ok(results)
    }
}
