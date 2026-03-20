use actix_web::{post, web, HttpResponse};
use serde::{Deserialize, Serialize};
use std::time::Instant;
use utoipa::ToSchema;

use crate::config::Config;
use crate::errors::AppError;
use crate::schemas::requests::ChatRagRequest;
use crate::services::embeddings::EmbeddingClient;
use crate::services::llm::LlmClient;
use crate::services::milvus::MilvusClient;
use crate::services::reranker::RerankerClient;

use super::chat::build_rag_context;

const EVAL_CSV: &str = include_str!("../../docs/eval_data.csv");
const MAX_RETRIES: u32 = 3;
const RETRY_DELAY_MS: u64 = 1000;

// ── CSV row ─────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct EvalRow {
    query: String,
    answer: String,
}

// ── Request / Response ──────────────────────────────────────────────

#[derive(Debug, Deserialize, ToSchema)]
pub struct RunEvalsRequest {
    /// Model to use
    #[schema(example = "qwen-3-235b-a22b-instruct-2507")]
    pub model: Option<String>,

    /// Provider ("groq" or "cerebras")
    #[schema(example = "cerebras")]
    pub provider: Option<String>,

    /// Milvus collection to search
    pub collection_name: Option<String>,

    /// Max source chunks per question
    pub limit: Option<i64>,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct EvalResult {
    pub question: String,
    pub expected_answer: String,
    pub rustyrag_answer: String,
    pub ttft_ms: u64,
    pub total_ms: u64,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct RunEvalsResponse {
    pub total_questions: usize,
    pub completed: usize,
    pub failed: usize,
    pub total_duration_ms: u64,
    pub results: Vec<EvalResult>,
}

// ── Endpoint ────────────────────────────────────────────────────────

#[utoipa::path(
    post,
    path = "/evals/run",
    request_body = RunEvalsRequest,
    responses(
        (status = 200, description = "Eval results as downloadable JSON", content_type = "application/json"),
        (status = 400, description = "Bad request"),
        (status = 502, description = "Upstream service error"),
    ),
    tag = "Evals"
)]
#[post("/evals/run")]
pub async fn run_evals(
    body: web::Json<RunEvalsRequest>,
    llm: web::Data<LlmClient>,
    milvus: web::Data<MilvusClient>,
    embeddings: web::Data<EmbeddingClient>,
    reranker: web::Data<RerankerClient>,
    config: web::Data<Config>,
) -> Result<HttpResponse, AppError> {
    let overall_start = Instant::now();

    // Parse CSV
    let mut reader = csv::Reader::from_reader(EVAL_CSV.as_bytes());
    let rows: Vec<EvalRow> = reader.deserialize().filter_map(|r| r.ok()).collect();

    if rows.is_empty() {
        return Err(AppError::BadRequest(
            "No eval questions found in CSV".into(),
        ));
    }

    let model = body
        .model
        .clone()
        .unwrap_or_else(|| "qwen-3-235b-a22b-instruct-2507".into());
    let provider = body
        .provider
        .clone()
        .unwrap_or_else(|| "cerebras".into());

    tracing::info!(
        "Running evals: {} questions (model={model}, provider={provider})",
        rows.len()
    );

    let mut results = Vec::with_capacity(rows.len());
    let mut failed = 0usize;

    for (i, row) in rows.iter().enumerate() {
        let question_start = Instant::now();

        let rag_request = ChatRagRequest {
            message: row.query.clone(),
            collection_name: body.collection_name.clone(),
            limit: body.limit,
            model: model.clone(),
            provider: provider.clone(),
            embedding_type: None,
            milvus_search_ef: None,
        };

        let mut last_err = String::new();
        let mut succeeded = false;

        for attempt in 1..=MAX_RETRIES {
            match run_single_question(&rag_request, &llm, &milvus, &embeddings, &reranker, &config)
                .await
            {
                Ok((content, ttft_ms)) => {
                    let total_ms = question_start.elapsed().as_millis() as u64;
                    if attempt > 1 {
                        tracing::info!(
                            "  [{}/{}] succeeded on attempt {attempt} in {total_ms}ms",
                            i + 1,
                            rows.len(),
                        );
                    } else {
                        tracing::info!(
                            "  [{}/{}] done in {total_ms}ms",
                            i + 1,
                            rows.len(),
                        );
                    }
                    results.push(EvalResult {
                        question: row.query.clone(),
                        expected_answer: row.answer.clone(),
                        rustyrag_answer: content,
                        ttft_ms,
                        total_ms,
                    });
                    succeeded = true;
                    break;
                }
                Err(e) => {
                    last_err = e.to_string();
                    if attempt < MAX_RETRIES {
                        tracing::warn!(
                            "  [{}/{}] attempt {attempt}/{MAX_RETRIES} failed: {last_err}, retrying…",
                            i + 1,
                            rows.len(),
                        );
                        tokio::time::sleep(std::time::Duration::from_millis(
                            RETRY_DELAY_MS * attempt as u64,
                        ))
                        .await;
                    }
                }
            }
        }

        if !succeeded {
            let total_ms = question_start.elapsed().as_millis() as u64;
            tracing::warn!(
                "  [{}/{}] FAILED after {MAX_RETRIES} attempts: {last_err}",
                i + 1,
                rows.len(),
            );
            failed += 1;
            results.push(EvalResult {
                question: row.query.clone(),
                expected_answer: row.answer.clone(),
                rustyrag_answer: format!("ERROR: {last_err}"),
                ttft_ms: 0,
                total_ms,
            });
        }
    }

    let total_duration_ms = overall_start.elapsed().as_millis() as u64;
    let completed = results.len() - failed;

    tracing::info!(
        "Evals complete: {completed}/{} succeeded, {failed} failed, {total_duration_ms}ms total",
        results.len(),
    );

    let response = RunEvalsResponse {
        total_questions: results.len(),
        completed,
        failed,
        total_duration_ms,
        results,
    };

    let body = serde_json::to_vec_pretty(&response)
        .map_err(|e| AppError::Internal(anyhow::anyhow!("JSON serialization error: {e}")))?;

    Ok(HttpResponse::Ok()
        .content_type("application/json")
        .insert_header((
            "Content-Disposition",
            "attachment; filename=\"eval_results.json\"",
        ))
        .body(body))
}

async fn run_single_question(
    request: &ChatRagRequest,
    llm: &LlmClient,
    milvus: &MilvusClient,
    embeddings: &EmbeddingClient,
    reranker: &RerankerClient,
    config: &Config,
) -> Result<(String, u64), AppError> {
    let ctx = build_rag_context(request, embeddings, milvus, reranker, config)
        .await?
        .ok_or_else(|| AppError::BadRequest("No documents found for query".into()))?;

    let llm_start = Instant::now();
    let result = llm
        .chat_with_system(
            &ctx.system_prompt,
            &request.message,
            &request.model,
            &request.provider,
            Some(1024),
        )
        .await
        .map_err(|e| AppError::LlmError(e.to_string()))?;

    let ttft_ms = llm_start.elapsed().as_millis() as u64;
    Ok((result.content, ttft_ms))
}
