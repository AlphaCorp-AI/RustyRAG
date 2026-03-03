use anyhow::Context;
use reqwest::Client;
use serde::{Deserialize, Serialize};

// ── OpenAI-compatible embedding client ─────────────────────────────

/// Controls how the embedding model encodes the text.
///
/// Use `SearchDocument` when indexing/storing and `SearchQuery` when
/// embedding a user query for retrieval.
#[derive(Debug, Clone, Copy)]
pub enum InputType {
    SearchDocument,
    SearchQuery,
}

#[derive(Clone)]
pub struct EmbeddingClient {
    http: Client,
    api_url: String,
    api_key: String,
    model: String,
    default_embedding_type: String,
    task_document: String,
    task_query: String,
}

#[derive(Serialize)]
struct EmbeddingRequest {
    model: String,
    input: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    embedding_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    task: Option<String>,
}

#[derive(Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
}

#[derive(Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
}

impl EmbeddingClient {
    pub fn new(
        api_url: &str,
        api_key: &str,
        model: &str,
        default_embedding_type: &str,
        task_document: &str,
        task_query: &str,
    ) -> Self {
        Self {
            http: Client::builder()
                .connect_timeout(std::time::Duration::from_secs(2))
                .pool_idle_timeout(std::time::Duration::from_secs(90))
                .pool_max_idle_per_host(32)
                .build()
                .expect("Failed to build reqwest client for embeddings"),
            api_url: api_url.trim_end_matches('/').to_string(),
            api_key: api_key.to_string(),
            model: model.to_string(),
            default_embedding_type: default_embedding_type.to_string(),
            task_document: task_document.to_string(),
            task_query: task_query.to_string(),
        }
    }

    /// Returns `true` when the client has been configured (all fields non-empty).
    pub fn is_configured(&self) -> bool {
        !self.api_url.is_empty() && !self.model.is_empty()
    }

    fn task_for(&self, input_type: InputType) -> String {
        match input_type {
            InputType::SearchDocument => self.task_document.clone(),
            InputType::SearchQuery => self.task_query.clone(),
        }
    }

    fn is_v5_retrieval_model(&self) -> bool {
        self.model.contains("jina-embeddings-v5-text-") && self.model.contains("-retrieval")
    }

    fn apply_retrieval_prefixes(&self, texts: &[String], input_type: InputType) -> Vec<String> {
        if !self.is_v5_retrieval_model() {
            return texts.to_vec();
        }

        let prefix = match input_type {
            InputType::SearchDocument => "Document: ",
            InputType::SearchQuery => "Query: ",
        };

        texts
            .iter()
            .map(|t| {
                if t.starts_with("Query: ") || t.starts_with("Document: ") {
                    t.clone()
                } else {
                    format!("{prefix}{t}")
                }
            })
            .collect()
    }

    pub async fn embed_with_options(
        &self,
        texts: &[String],
        input_type: InputType,
        embedding_type: Option<&str>,
        task_override: Option<&str>,
    ) -> anyhow::Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        let task = task_override.map(|v| v.to_string()).or_else(|| {
            let t = self.task_for(input_type);
            if t.is_empty() {
                None
            } else {
                Some(t)
            }
        });

        let requested_embedding_type = embedding_type.map(|v| v.to_string()).or_else(|| {
            if self.default_embedding_type.is_empty() {
                None
            } else {
                Some(self.default_embedding_type.clone())
            }
        });

        let body = EmbeddingRequest {
            model: self.model.clone(),
            input: self.apply_retrieval_prefixes(texts, input_type),
            embedding_type: requested_embedding_type,
            task,
        };

        let mut req = self.http.post(&self.api_url).json(&body);
        if !self.api_key.is_empty() {
            req = req.bearer_auth(&self.api_key);
        }
        let res = req.send().await.context("Failed to call embedding API")?;

        let status = res.status();
        if !status.is_success() {
            let text = res.text().await.unwrap_or_default();
            anyhow::bail!("Embedding API returned {status}: {text}");
        }

        let data: EmbeddingResponse = res
            .json()
            .await
            .context("Failed to parse embedding response")?;

        Ok(data.data.into_iter().map(|v| v.embedding).collect())
    }
}
