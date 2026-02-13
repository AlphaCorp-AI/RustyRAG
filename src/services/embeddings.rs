use anyhow::Context;
use reqwest::Client;
use serde::{Deserialize, Serialize};

// ── Cohere Embed v2 client ─────────────────────────────────────────

const COHERE_EMBED_URL: &str = "https://api.cohere.com/v2/embed";

/// Controls how Cohere encodes the text.
///
/// Use `SearchDocument` when indexing/storing and `SearchQuery` when
/// embedding a user query for retrieval.
#[derive(Debug, Clone, Copy)]
pub enum InputType {
    SearchDocument,
    SearchQuery,
}

impl InputType {
    fn as_str(self) -> &'static str {
        match self {
            InputType::SearchDocument => "search_document",
            InputType::SearchQuery => "search_query",
        }
    }
}

#[derive(Clone)]
pub struct EmbeddingClient {
    http: Client,
    api_key: String,
    model: String,
}

#[derive(Serialize)]
struct CohereEmbedRequest {
    model: String,
    texts: Vec<String>,
    input_type: String,
    embedding_types: Vec<String>,
}

#[derive(Deserialize)]
struct CohereEmbedResponse {
    embeddings: CohereEmbeddings,
}

#[derive(Deserialize)]
struct CohereEmbeddings {
    float: Vec<Vec<f32>>,
}

impl EmbeddingClient {
    pub fn new(api_key: &str, model: &str) -> Self {
        Self {
            http: Client::new(),
            api_key: api_key.to_string(),
            model: model.to_string(),
        }
    }

    /// Returns `true` when the client has been configured (all fields non-empty).
    pub fn is_configured(&self) -> bool {
        !self.api_key.is_empty() && !self.model.is_empty()
    }

    /// Embed a batch of texts via the Cohere v2 API.
    ///
    /// Returns one `Vec<f32>` per input, in order.
    pub async fn embed(
        &self,
        texts: &[String],
        input_type: InputType,
    ) -> anyhow::Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        let body = CohereEmbedRequest {
            model: self.model.clone(),
            texts: texts.to_vec(),
            input_type: input_type.as_str().to_string(),
            embedding_types: vec!["float".to_string()],
        };

        let res = self
            .http
            .post(COHERE_EMBED_URL)
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await
            .context("Failed to call Cohere embedding API")?;

        let status = res.status();
        if !status.is_success() {
            let text = res.text().await.unwrap_or_default();
            anyhow::bail!("Cohere embedding API returned {status}: {text}");
        }

        let data: CohereEmbedResponse = res
            .json()
            .await
            .context("Failed to parse Cohere embedding response")?;

        Ok(data.embeddings.float)
    }
}
