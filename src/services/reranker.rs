use anyhow::Context;
use reqwest::Client;
use serde::{Deserialize, Serialize};

/// Reranker client compatible with TEI /rerank endpoint (local) or Jina API.
pub struct RerankerClient {
    http: Client,
    api_url: String,
    api_key: String,
    model: String,
}

#[derive(Serialize)]
struct RerankRequest {
    query: String,
    texts: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    truncate: Option<bool>,
}

#[derive(Deserialize)]
struct RerankResult {
    index: usize,
    score: f64,
}

/// Result of reranking: original index + new relevance score.
#[derive(Debug, Clone)]
pub struct RerankedItem {
    pub original_index: usize,
    pub relevance_score: f64,
}

impl RerankerClient {
    pub fn new(api_url: &str, api_key: &str, model: &str) -> Self {
        Self {
            http: Client::builder()
                .connect_timeout(std::time::Duration::from_secs(5))
                .timeout(std::time::Duration::from_secs(30))
                .build()
                .expect("Failed to build reqwest client for reranker"),
            api_url: api_url.trim_end_matches('/').to_string(),
            api_key: api_key.to_string(),
            model: model.to_string(),
        }
    }

    pub fn is_configured(&self) -> bool {
        !self.api_url.is_empty()
    }

    /// Rerank documents by relevance to the query.
    /// Returns items sorted by relevance (highest first).
    pub async fn rerank(
        &self,
        query: &str,
        documents: &[String],
        top_n: usize,
    ) -> anyhow::Result<Vec<RerankedItem>> {
        if documents.is_empty() {
            return Ok(vec![]);
        }

        let body = RerankRequest {
            query: query.to_string(),
            texts: documents.to_vec(),
            truncate: Some(true),
        };

        let mut req = self.http.post(&format!("{}/rerank", self.api_url)).json(&body);
        if !self.api_key.is_empty() {
            req = req.bearer_auth(&self.api_key);
        }

        let resp = req
            .send()
            .await
            .context("Failed to call reranker API")?;

        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            anyhow::bail!("Reranker returned {status}: {text}");
        }

        let mut results: Vec<RerankResult> = resp
            .json()
            .await
            .context("Failed to parse reranker response")?;

        // Sort by score descending
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

        // Take top_n
        Ok(results
            .into_iter()
            .take(top_n)
            .map(|r| RerankedItem {
                original_index: r.index,
                relevance_score: r.score,
            })
            .collect())
    }
}
