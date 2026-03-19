use anyhow::Context;
use reqwest::Client;
use serde::{Deserialize, Serialize};

#[derive(Clone)]
pub struct RerankerClient {
    http: Client,
    api_url: String,
}

#[derive(Serialize)]
struct RerankRequest {
    query: String,
    texts: Vec<String>,
    raw_scores: bool,
}

#[derive(Debug, Deserialize)]
pub struct RerankResult {
    pub index: usize,
    pub score: f32,
}

impl RerankerClient {
    pub fn new(api_url: &str) -> Self {
        Self {
            http: Client::builder()
                .connect_timeout(std::time::Duration::from_secs(5))
                .timeout(std::time::Duration::from_secs(30))
                .pool_idle_timeout(std::time::Duration::from_secs(90))
                .pool_max_idle_per_host(8)
                .build()
                .expect("Failed to build reqwest client for reranker"),
            api_url: api_url.trim_end_matches('/').to_string(),
        }
    }

    pub fn is_configured(&self) -> bool {
        !self.api_url.is_empty()
    }

    /// Rerank documents by relevance to the query, returning the top_n results
    /// sorted by score descending.
    pub async fn rerank(
        &self,
        query: &str,
        texts: &[String],
        top_n: usize,
    ) -> anyhow::Result<Vec<RerankResult>> {
        let url = format!("{}/rerank", self.api_url);

        let body = RerankRequest {
            query: query.to_string(),
            texts: texts.to_vec(),
            raw_scores: false,
        };

        let resp: Vec<RerankResult> = self
            .http
            .post(&url)
            .json(&body)
            .send()
            .await
            .context("Reranker: failed to send request")?
            .error_for_status()
            .context("Reranker: server returned error")?
            .json()
            .await
            .context("Reranker: failed to parse response")?;

        // TEI returns results sorted by score desc already, just truncate
        let mut results = resp;
        results.truncate(top_n);

        Ok(results)
    }
}