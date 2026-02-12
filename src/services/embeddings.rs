use anyhow::Context;
use reqwest::Client;
use serde::{Deserialize, Serialize};

// ── OpenAI-compatible embedding client ─────────────────────────────

#[derive(Clone)]
pub struct EmbeddingClient {
    http: Client,
    api_url: String,
    api_key: String,
    model: String,
}

#[derive(Serialize)]
struct EmbeddingRequest {
    model: String,
    input: Vec<String>,
}

#[derive(Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
}

#[derive(Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
    index: usize,
}

impl EmbeddingClient {
    pub fn new(api_url: &str, api_key: &str, model: &str) -> Self {
        Self {
            http: Client::new(),
            api_url: api_url.to_string(),
            api_key: api_key.to_string(),
            model: model.to_string(),
        }
    }

    /// Returns `true` when the client has been configured (all fields non-empty).
    pub fn is_configured(&self) -> bool {
        !self.api_url.is_empty() && !self.api_key.is_empty() && !self.model.is_empty()
    }

    /// Embed a batch of texts. Returns one `Vec<f32>` per input, in order.
    pub async fn embed(&self, texts: &[String]) -> anyhow::Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        let body = EmbeddingRequest {
            model: self.model.clone(),
            input: texts.to_vec(),
        };

        let res = self
            .http
            .post(&self.api_url)
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await
            .context("Failed to call embedding API")?;

        let status = res.status();
        if !status.is_success() {
            let text = res.text().await.unwrap_or_default();
            anyhow::bail!("Embedding API returned {status}: {text}");
        }

        let data: EmbeddingResponse = res
            .json()
            .await
            .context("Failed to parse embedding API response")?;

        // Sort by index to guarantee order matches input
        let mut pairs: Vec<(usize, Vec<f32>)> =
            data.data.into_iter().map(|d| (d.index, d.embedding)).collect();
        pairs.sort_by_key(|(idx, _)| *idx);

        Ok(pairs.into_iter().map(|(_, emb)| emb).collect())
    }
}
