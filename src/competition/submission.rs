use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

// ── Question (downloaded from eval API) ─────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Question {
    pub id: String,
    pub question: String,
    pub answer_type: String,
}

// ── Submission JSON format ──────────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct Submission {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub architecture_summary: Option<String>,
    pub answers: Vec<SubmissionAnswer>,
}

#[derive(Debug, Serialize)]
pub struct SubmissionAnswer {
    pub question_id: String,
    pub answer: Value,
    pub telemetry: SubmissionTelemetry,
}

#[derive(Debug, Serialize)]
pub struct SubmissionTelemetry {
    pub timing: TimingMetrics,
    pub retrieval: RetrievalData,
    pub usage: UsageMetrics,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_name: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct TimingMetrics {
    pub ttft_ms: u64,
    pub tpot_ms: u64,
    pub total_time_ms: u64,
}

#[derive(Debug, Serialize)]
pub struct RetrievalData {
    pub retrieved_chunk_pages: Vec<RetrievalRef>,
}

#[derive(Debug, Clone, Serialize)]
pub struct RetrievalRef {
    pub doc_id: String,
    pub page_numbers: Vec<u32>,
}

#[derive(Debug, Serialize)]
pub struct UsageMetrics {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

// ── Builder ─────────────────────────────────────────────────────────

pub struct SubmissionBuilder {
    architecture_summary: Option<String>,
    answers: Vec<SubmissionAnswer>,
}

impl SubmissionBuilder {
    pub fn new(summary: &str) -> Self {
        Self {
            architecture_summary: Some(summary.to_string()),
            answers: Vec::new(),
        }
    }

    pub fn add_answer(&mut self, answer: SubmissionAnswer) {
        self.answers.push(answer);
    }

    pub fn save(&self, path: &str) -> anyhow::Result<()> {
        let payload = serde_json::json!({
            "architecture_summary": self.architecture_summary,
            "answers": self.answers,
        });
        let json = serde_json::to_string_pretty(&payload)?;
        std::fs::write(path, json)?;
        tracing::info!("Saved submission to {path} ({} answers)", self.answers.len());
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.answers.len()
    }
}

// ── Helpers ─────────────────────────────────────────────────────────

/// Build retrieval refs from Milvus search results.
/// Merges page numbers per doc_id and strips .pdf extension.
pub fn build_retrieval_refs(
    hits: &[crate::services::milvus::SearchResult],
) -> Vec<RetrievalRef> {
    let mut by_doc: BTreeMap<String, Vec<u32>> = BTreeMap::new();

    for hit in hits {
        if hit.source_file.is_empty() {
            continue;
        }
        // doc_id = filename without .pdf extension
        let doc_id = hit
            .source_file
            .strip_suffix(".pdf")
            .unwrap_or(&hit.source_file)
            .to_string();
        let page = hit.page_number as u32;
        if page > 0 {
            by_doc.entry(doc_id).or_default().push(page);
        }
    }

    by_doc
        .into_iter()
        .map(|(doc_id, mut pages)| {
            pages.sort();
            pages.dedup();
            RetrievalRef { doc_id, page_numbers: pages }
        })
        .collect()
}
