use anyhow::Context;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::json;

pub const DEFAULT_COLLECTION: &str = "documents";

const OUTPUT_FIELDS: &[&str] = &[
    "id",
    "text",
    "file_name",
    "file_size",
    "chunk_index",
    "page_number",
    "context_prefix",
];

// ── Public types ────────────────────────────────────────────────────

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

/// A document chunk ready for insertion.
#[derive(Debug, Clone, Serialize)]
pub struct DocumentChunk {
    pub text: String,
    pub file_name: String,
    pub file_size: i64,
    pub chunk_index: i64,
    pub page_number: i64,
    pub context_prefix: String,
    pub embedding: Vec<f32>,
}

/// A search hit from Milvus.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub id: i64,
    pub text: String,
    pub file_name: String,
    pub file_size: i64,
    pub chunk_index: i64,
    pub page_number: i64,
    pub context_prefix: String,
    pub score: f32,
}

#[derive(Debug, Clone)]
pub struct SearchOptions {
    pub ef: Option<i64>,
}

// ── Milvus response envelope ────────────────────────────────────────

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

    // ── Collection management ───────────────────────────────────────

    pub async fn ensure_collection(&self, name: &str) -> anyhow::Result<()> {
        if !self.has_collection(name).await? {
            self.create_collection(name).await?;
            tracing::info!("Created Milvus collection '{name}'");
        }
        Ok(())
    }

    async fn has_collection(&self, name: &str) -> anyhow::Result<bool> {
        let resp = self
            .post("collections/has", json!({ "collectionName": name }))
            .await
            .context("Milvus: failed to check collection")?;
        Ok(resp.data["has"].as_bool().unwrap_or(false))
    }

    async fn create_collection(&self, name: &str) -> anyhow::Result<()> {
        let body = json!({
            "collectionName": name,
            "schema": {
                "autoId": true,
                "enableDynamicField": false,
                "fields": [
                    { "fieldName": "id", "dataType": "Int64", "isPrimary": true },
                    {
                        "fieldName": "text",
                        "dataType": "VarChar",
                        "elementTypeParams": {
                            "max_length": "65535",
                            "enable_analyzer": "true",
                            "analyzer_params": "{\"type\":\"standard\"}"
                        }
                    },
                    {
                        "fieldName": "file_name",
                        "dataType": "VarChar",
                        "elementTypeParams": { "max_length": "1024" }
                    },
                    { "fieldName": "file_size", "dataType": "Int64" },
                    { "fieldName": "chunk_index", "dataType": "Int64" },
                    { "fieldName": "page_number", "dataType": "Int64" },
                    {
                        "fieldName": "context_prefix",
                        "dataType": "VarChar",
                        "elementTypeParams": { "max_length": "2048" }
                    },
                    {
                        "fieldName": "embedding",
                        "dataType": "FloatVector",
                        "elementTypeParams": { "dim": self.dimension.to_string() }
                    },
                    { "fieldName": "sparse_embedding", "dataType": "SparseFloatVector" }
                ],
                "functions": [{
                    "name": "bm25_fn",
                    "type": "BM25",
                    "inputFieldNames": ["text"],
                    "outputFieldNames": ["sparse_embedding"]
                }]
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
                },
                {
                    "fieldName": "sparse_embedding",
                    "indexName": "sparse_idx",
                    "metricType": "BM25",
                    "params": { "index_type": "AUTOINDEX" }
                }
            ]
        });

        self.post("collections/create", body)
            .await
            .context("Milvus: failed to create collection")?;
        Ok(())
    }

    // ── Insert ──────────────────────────────────────────────────────

    pub async fn insert(
        &self,
        collection_name: &str,
        chunks: Vec<DocumentChunk>,
    ) -> anyhow::Result<usize> {
        let count = chunks.len();

        let data: Vec<serde_json::Value> = chunks
            .into_iter()
            .map(|c| {
                json!({
                    "text": c.text,
                    "file_name": c.file_name,
                    "file_size": c.file_size,
                    "chunk_index": c.chunk_index,
                    "page_number": c.page_number,
                    "context_prefix": c.context_prefix,
                    "embedding": c.embedding,
                })
            })
            .collect();

        self.post(
            "entities/insert",
            json!({ "collectionName": collection_name, "data": data }),
        )
        .await
        .context("Milvus: failed to insert")?;

        Ok(count)
    }

    // ── Search (dense only) ─────────────────────────────────────────

    pub async fn search(
        &self,
        collection_name: &str,
        embedding: Vec<f32>,
        limit: i64,
        options: Option<SearchOptions>,
    ) -> anyhow::Result<Vec<SearchResult>> {
        let ef = options.and_then(|v| v.ef).unwrap_or(self.default_search_ef);

        let body = json!({
            "collectionName": collection_name,
            "data": [embedding],
            "limit": limit,
            "outputFields": OUTPUT_FIELDS,
            "searchParams": { "params": { "ef": ef } },
        });

        let resp = self
            .post("entities/search", body)
            .await
            .context("Milvus: failed to search")?;

        Ok(Self::parse_results(&resp.data))
    }

    // ── BM25-only text search (sparse) ─────────────────────────────

    /// Search using only the BM25 sparse index — does NOT need an embedding
    /// vector, so it can run concurrently with embedding generation.
    pub async fn text_search(
        &self,
        collection_name: &str,
        query_text: &str,
        limit: i64,
    ) -> anyhow::Result<Vec<SearchResult>> {
        let body = json!({
            "collectionName": collection_name,
            "data": [query_text],
            "annsField": "sparse_embedding",
            "limit": limit,
            "outputFields": OUTPUT_FIELDS,
        });

        let resp = self
            .post("entities/search", body)
            .await
            .context("Milvus: failed to text_search")?;

        Ok(Self::parse_results(&resp.data))
    }

    // ── RRF merge ────────────────────────────────────────────────────

    /// Reciprocal Rank Fusion: merge two ranked lists into one.
    /// `k` controls how much lower ranks are dampened (lower k = sharper).
    pub fn rrf_merge(
        dense: Vec<SearchResult>,
        sparse: Vec<SearchResult>,
        k: f32,
        limit: usize,
    ) -> Vec<SearchResult> {
        use std::collections::HashMap;

        // Map id → (best SearchResult, cumulative RRF score)
        let mut scores: HashMap<i64, (SearchResult, f32)> = HashMap::new();

        for (rank, hit) in dense.into_iter().enumerate() {
            let rrf = 1.0 / (k + rank as f32 + 1.0);
            scores
                .entry(hit.id)
                .and_modify(|(_, s)| *s += rrf)
                .or_insert((hit, rrf));
        }

        for (rank, hit) in sparse.into_iter().enumerate() {
            let rrf = 1.0 / (k + rank as f32 + 1.0);
            scores
                .entry(hit.id)
                .and_modify(|(_, s)| *s += rrf)
                .or_insert((hit, rrf));
        }

        let mut merged: Vec<SearchResult> = scores
            .into_values()
            .map(|(mut hit, score)| {
                hit.score = score;
                hit
            })
            .collect();

        merged.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        merged.truncate(limit);
        merged
    }

    // ── Export (query all rows) ────────────────────────────────────

    /// Fetch every row from a collection using cursor-based pagination
    /// (keyset pagination on `id`) to avoid Milvus offset limits.
    /// Returns all fields including embeddings.
    pub async fn query_all(
        &self,
        collection_name: &str,
    ) -> anyhow::Result<Vec<serde_json::Value>> {
        const PAGE_SIZE: i64 = 1000;
        let output_fields: Vec<&str> = OUTPUT_FIELDS.iter().copied().chain(["embedding"]).collect();

        let mut all_rows: Vec<serde_json::Value> = Vec::new();
        let mut last_id: i64 = 0;

        loop {
            let body = json!({
                "collectionName": collection_name,
                "filter": format!("id > {last_id}"),
                "outputFields": output_fields,
                "limit": PAGE_SIZE,
            });

            let resp = self
                .post("entities/query", body)
                .await
                .context("Milvus: failed to query for backup")?;

            let rows = resp.data.as_array().map(|a| a.len()).unwrap_or(0);
            if rows == 0 {
                break;
            }

            if let Some(arr) = resp.data.as_array() {
                // Track the max id for the next page cursor
                for item in arr {
                    if let Some(id) = item["id"].as_i64() {
                        if id > last_id {
                            last_id = id;
                        }
                    }
                }
                all_rows.extend(arr.iter().cloned());
            }

            tracing::info!(
                "Backup: fetched {} rows so far from '{collection_name}'…",
                all_rows.len()
            );

            if rows < PAGE_SIZE as usize {
                break;
            }
        }

        Ok(all_rows)
    }

    // ── Internal helpers ────────────────────────────────────────────

    /// POST to Milvus REST API v2 and check for errors.
    async fn post(&self, path: &str, body: serde_json::Value) -> anyhow::Result<MilvusResponse> {
        let url = format!("{}/v2/vectordb/{}", self.base_url, path);

        let resp: MilvusResponse = self
            .http
            .post(&url)
            .json(&body)
            .send()
            .await?
            .json()
            .await?;

        if resp.code != 0 {
            anyhow::bail!("Milvus {path} error: {}", resp.message);
        }

        Ok(resp)
    }

    fn parse_results(data: &serde_json::Value) -> Vec<SearchResult> {
        data.as_array()
            .map(|arr| {
                arr.iter()
                    .map(|item| SearchResult {
                        id: item["id"].as_i64().unwrap_or(0),
                        text: item["text"].as_str().unwrap_or("").to_string(),
                        file_name: item["file_name"].as_str().unwrap_or("").to_string(),
                        file_size: item["file_size"].as_i64().unwrap_or(0),
                        chunk_index: item["chunk_index"].as_i64().unwrap_or(0),
                        page_number: item["page_number"].as_i64().unwrap_or(0),
                        context_prefix: item["context_prefix"]
                            .as_str()
                            .unwrap_or("")
                            .to_string(),
                        score: item["distance"].as_f64().unwrap_or(0.0) as f32,
                    })
                    .collect()
            })
            .unwrap_or_default()
    }
}
