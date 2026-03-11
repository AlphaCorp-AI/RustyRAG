use std::path::Path;

use anyhow::Context;
use reqwest::Client;

use super::submission::Question;

/// Thin client for the Agentic RAG Legal Challenge evaluation API.
pub struct EvalClient {
    http: Client,
    base_url: String,
    api_key: String,
}

impl EvalClient {
    pub fn new(base_url: &str, api_key: &str) -> Self {
        Self {
            http: Client::builder()
                .timeout(std::time::Duration::from_secs(120))
                .build()
                .expect("Failed to build eval HTTP client"),
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key: api_key.to_string(),
        }
    }

    /// Download competition questions.
    pub async fn download_questions(&self) -> anyhow::Result<Vec<Question>> {
        let url = format!("{}/questions", self.base_url);
        let resp = self
            .http
            .get(&url)
            .header("X-API-Key", &self.api_key)
            .send()
            .await
            .context("Failed to reach eval API for questions")?;

        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            anyhow::bail!("GET /questions returned {status}: {text}");
        }

        let questions: Vec<Question> = resp.json().await.context("Failed to parse questions")?;
        Ok(questions)
    }

    /// Download and extract the document corpus ZIP.
    pub async fn download_documents(&self, target_dir: &str) -> anyhow::Result<()> {
        let url = format!("{}/documents", self.base_url);
        tracing::info!("Downloading documents from {url}...");

        let resp = self
            .http
            .get(&url)
            .header("X-API-Key", &self.api_key)
            .send()
            .await
            .context("Failed to reach eval API for documents")?;

        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            anyhow::bail!("GET /documents returned {status}: {text}");
        }

        let bytes = resp.bytes().await.context("Failed to read document ZIP")?;

        std::fs::create_dir_all(target_dir)?;
        let zip_path = Path::new(target_dir).join("documents.zip");
        std::fs::write(&zip_path, &bytes)?;

        // Extract ZIP
        let file = std::fs::File::open(&zip_path)?;
        let mut archive = zip::ZipArchive::new(file)?;
        archive.extract(target_dir)?;

        tracing::info!("Extracted documents to {target_dir}");
        Ok(())
    }

    /// Submit submission.json + code archive.
    pub async fn submit(
        &self,
        submission_path: &str,
        code_archive_path: &str,
    ) -> anyhow::Result<serde_json::Value> {
        let url = format!("{}/submissions", self.base_url);

        let submission_bytes = std::fs::read(submission_path)
            .context(format!("Failed to read {submission_path}"))?;
        let archive_bytes = std::fs::read(code_archive_path)
            .context(format!("Failed to read {code_archive_path}"))?;

        let submission_part = reqwest::multipart::Part::bytes(submission_bytes)
            .file_name("submission.json")
            .mime_str("application/json")?;
        let archive_part = reqwest::multipart::Part::bytes(archive_bytes)
            .file_name("code_archive.zip")
            .mime_str("application/zip")?;

        let form = reqwest::multipart::Form::new()
            .part("file", submission_part)
            .part("code_archive", archive_part);

        let resp = self
            .http
            .post(&url)
            .header("X-API-Key", &self.api_key)
            .multipart(form)
            .send()
            .await
            .context("Failed to submit")?;

        let status = resp.status();
        let body: serde_json::Value = resp.json().await.unwrap_or_default();

        if !status.is_success() {
            anyhow::bail!("POST /submissions returned {status}: {body}");
        }

        Ok(body)
    }

    /// Check submission status.
    pub async fn get_status(&self, uuid: &str) -> anyhow::Result<serde_json::Value> {
        let url = format!("{}/submissions/{uuid}/status", self.base_url);

        let resp = self
            .http
            .get(&url)
            .header("X-API-Key", &self.api_key)
            .send()
            .await
            .context("Failed to get submission status")?;

        let status = resp.status();
        let body: serde_json::Value = resp.json().await.unwrap_or_default();

        if !status.is_success() {
            anyhow::bail!("GET /submissions/{uuid}/status returned {status}: {body}");
        }

        Ok(body)
    }
}
