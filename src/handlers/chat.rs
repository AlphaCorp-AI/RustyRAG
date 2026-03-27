use actix_web::{get, post, web, HttpResponse};
use futures_util::StreamExt;
use std::time::Instant;
use validator::Validate;

use crate::config::Config;
use crate::errors::AppError;
use crate::prompts::build_rag_system_prompt;
use crate::schemas::requests::ChatRagRequest;
use crate::schemas::responses::{ChatRagResponse, ErrorResponse, RagSource, Timing, Usage};
use crate::schemas::responses::{LlmModelEntry, LlmModelsResponse};
use crate::services::embeddings::{EmbeddingClient, InputType};
use crate::services::llm::LlmClient;
use crate::services::milvus::{MilvusClient, SearchOptions, SearchResult, DEFAULT_COLLECTION};
use crate::services::reranker::RerankerClient;

// ── Validation helpers ─────────────────────────────────────────────

pub(crate) fn validate_collection_name(name: &str) -> Result<(), AppError> {
    if name.is_empty()
        || name.len() > 128
        || !name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
    {
        return Err(AppError::BadRequest(
            "Invalid collection name: use only alphanumeric, underscore, or hyphen (max 128 chars)"
                .into(),
        ));
    }
    Ok(())
}

fn validated_top_n(limit: Option<i64>, default: usize) -> Result<usize, AppError> {
    match limit {
        Some(n) if n >= 1 => Ok(n as usize),
        Some(_) => Err(AppError::BadRequest(
            "limit must be a positive integer".into(),
        )),
        None => Ok(default),
    }
}

// ── Shared RAG pipeline ────────────────────────────────────────────

/// Holds the result of the retrieval + reranking pipeline, ready for
/// the LLM call.
pub(crate) struct RagContext {
    pub sources: Vec<RagSource>,
    pub system_prompt: String,
}

/// Embed the query → parallel hybrid search Milvus → rerank → build LLM context.
///
/// Fires BM25 text search concurrently with embedding generation to reduce TTFT.
/// Returns `Ok(None)` when the collection has no matching documents.
pub(crate) async fn build_rag_context(
    body: &ChatRagRequest,
    embeddings: &EmbeddingClient,
    milvus: &MilvusClient,
    reranker: &RerankerClient,
    config: &Config,
) -> Result<Option<RagContext>, AppError> {
    let collection = body
        .collection_name
        .as_deref()
        .unwrap_or(DEFAULT_COLLECTION);
    validate_collection_name(collection)?;

    let top_n = validated_top_n(body.limit, config.rerank_top_n)?;

    // 1. Fire embedding + BM25 text search in parallel
    //    BM25 doesn't need the vector, so we don't wait for embedding.
    let (embed_result, sparse_result) = tokio::join!(
        embed_query(embeddings, &body.message, body.embedding_type.as_deref()),
        async {
            milvus
                .text_search(collection, &body.message, config.retrieval_limit)
                .await
                .map_err(|e| AppError::MilvusError(e.to_string()))
        }
    );

    let query_embedding = embed_result?;
    let sparse_hits = sparse_result?;

    // 2. Dense search (needs the embedding, so runs after embed completes)
    let dense_hits = milvus
        .search(
            collection,
            query_embedding,
            config.retrieval_limit,
            Some(SearchOptions {
                ef: body.milvus_search_ef,
            }),
        )
        .await
        .map_err(|e| AppError::MilvusError(e.to_string()))?;

    // 3. RRF merge (k=20 for sharper discrimination)
    let hits = MilvusClient::rrf_merge(
        dense_hits,
        sparse_hits,
        20.0,
        config.retrieval_limit as usize,
    );

    if hits.is_empty() {
        return Ok(None);
    }

    // 4. Rerank
    let hits = rerank_hits(reranker, &body.message, hits, top_n).await?;

    // 5. Build context
    let sources: Vec<RagSource> = hits.iter().map(RagSource::from).collect();

    let context = hits
        .iter()
        .enumerate()
        .map(|(i, h)| format!("[Source {} — {}]\n{}", i + 1, h.file_name, h.text))
        .collect::<Vec<_>>()
        .join("\n\n");

    Ok(Some(RagContext {
        sources,
        system_prompt: build_rag_system_prompt(&context),
    }))
}

/// Embed a single query string, returning its vector.
async fn embed_query(
    embeddings: &EmbeddingClient,
    message: &str,
    embedding_type: Option<&str>,
) -> Result<Vec<f32>, AppError> {
    if !embeddings.is_configured() {
        return Err(AppError::BadRequest(
            "Embedding API is not configured. Set EMBEDDING_API_URL and EMBEDDING_MODEL.".into(),
        ));
    }

    embeddings
        .embed_with_options(
            &[message.to_string()],
            InputType::SearchQuery,
            embedding_type,
            None,
        )
        .await
        .map_err(|e| AppError::EmbeddingError(e.to_string()))?
        .into_iter()
        .next()
        .ok_or_else(|| AppError::EmbeddingError("No embedding returned for query".into()))
}

/// If the reranker is configured, rerank `hits` by relevance to `query` and
/// return the top `top_n` results with reranker scores. Falls back to
/// truncating the original Milvus results when the reranker is unavailable.
async fn rerank_hits(
    reranker: &RerankerClient,
    query: &str,
    hits: Vec<SearchResult>,
    top_n: usize,
) -> Result<Vec<SearchResult>, AppError> {
    if !reranker.is_configured() {
        let mut truncated = hits;
        truncated.truncate(top_n);
        return Ok(truncated);
    }

    let texts: Vec<String> = hits.iter().map(|h| h.text.clone()).collect();

    let ranked = reranker
        .rerank(query, &texts, top_n)
        .await
        .map_err(|e| AppError::LlmError(format!("Reranker error: {e}")))?;

    Ok(ranked
        .into_iter()
        .filter_map(|r| {
            hits.get(r.index).map(|h| SearchResult {
                score: r.score,
                ..h.clone()
            })
        })
        .collect())
}

// ── Endpoints ──────────────────────────────────────────────────────

#[utoipa::path(
    get,
    path = "/llms",
    responses(
        (status = 200, description = "Supported LLMs grouped by provider", body = LlmModelsResponse),
    ),
    tag = "Chat"
)]
#[get("/llms")]
pub async fn list_llms() -> HttpResponse {
    let models = LlmClient::supported_models()
        .into_iter()
        .map(|(provider, model)| LlmModelEntry {
            provider: provider.to_string(),
            model: model.to_string(),
        })
        .collect();
    HttpResponse::Ok().json(LlmModelsResponse { models })
}

/// Ask a question grounded in documents from a Milvus collection.
#[utoipa::path(
    post,
    path = "/chat-rag",
    request_body = ChatRagRequest,
    responses(
        (status = 200, description = "RAG answer with sources", body = ChatRagResponse),
        (status = 400, description = "Validation error", body = ErrorResponse),
        (status = 502, description = "Upstream service error", body = ErrorResponse),
    ),
    tag = "Chat"
)]
#[post("/chat-rag")]
pub async fn chat_rag(
    body: web::Json<ChatRagRequest>,
    llm: web::Data<LlmClient>,
    milvus: web::Data<MilvusClient>,
    embeddings: web::Data<EmbeddingClient>,
    reranker: web::Data<RerankerClient>,
    config: web::Data<Config>,
) -> Result<HttpResponse, AppError> {
    let request_start = Instant::now();

    body.validate()
        .map_err(|e| AppError::BadRequest(e.to_string()))?;

    let ctx = match build_rag_context(&body, &embeddings, &milvus, &reranker, &config).await? {
        Some(ctx) => ctx,
        None => {
            let total_ms = request_start.elapsed().as_millis() as u64;
            let collection = body.collection_name.as_deref().unwrap_or(DEFAULT_COLLECTION);
            return Ok(HttpResponse::Ok().json(ChatRagResponse {
                model: body.model.clone(),
                message: format!("No documents found in collection '{collection}'."),
                sources: vec![],
                usage: None,
                timing: Timing {
                    ttft_ms: total_ms,
                    total_ms,
                },
            }));
        }
    };

    // Use streaming to measure real TTFT (time to first token)
    let llm_start = Instant::now();
    let response = llm
        .chat_stream_with_system(
            &ctx.system_prompt,
            &body.message,
            &body.model,
            &body.provider,
            Some(1024),
        )
        .await
        .map_err(|e| AppError::LlmError(e.to_string()))?;

    let mut stream = response.bytes_stream();
    let mut ttft_ms: Option<u64> = None;
    let mut full_content = String::new();
    let mut model_name = body.model.clone();

    while let Some(chunk) = stream.next().await {
        let bytes = chunk.map_err(|e| AppError::LlmError(format!("Stream error: {e}")))?;
        if ttft_ms.is_none() {
            ttft_ms = Some(llm_start.elapsed().as_millis() as u64);
        }
        let text = String::from_utf8_lossy(&bytes);
        for line in text.lines() {
            if let Some(data) = line.strip_prefix("data: ") {
                if data == "[DONE]" {
                    continue;
                }
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(data) {
                    if let Some(delta) = v["choices"][0]["delta"]["content"].as_str() {
                        full_content.push_str(delta);
                    }
                    if let Some(m) = v["model"].as_str() {
                        model_name = m.to_string();
                    }
                }
            }
        }
    }

    let ttft = ttft_ms.unwrap_or_else(|| llm_start.elapsed().as_millis() as u64);
    let total_ms = request_start.elapsed().as_millis() as u64;

    Ok(HttpResponse::Ok().json(ChatRagResponse {
        model: model_name,
        message: full_content,
        sources: ctx.sources,
        usage: None,
        timing: Timing { ttft_ms: ttft, total_ms },
    }))
}

/// Stream a RAG answer via Server-Sent Events.
#[utoipa::path(
    post,
    path = "/chat-rag/stream",
    request_body = ChatRagRequest,
    responses(
        (status = 200, description = "SSE stream: sources event then LLM tokens", content_type = "text/event-stream"),
        (status = 400, description = "Validation error", body = ErrorResponse),
        (status = 502, description = "Upstream service error", body = ErrorResponse),
    ),
    tag = "Chat"
)]
#[post("/chat-rag/stream")]
pub async fn chat_rag_stream(
    body: web::Json<ChatRagRequest>,
    llm: web::Data<LlmClient>,
    milvus: web::Data<MilvusClient>,
    embeddings: web::Data<EmbeddingClient>,
    reranker: web::Data<RerankerClient>,
    config: web::Data<Config>,
) -> Result<HttpResponse, AppError> {
    let request_start = Instant::now();

    body.validate()
        .map_err(|e| AppError::BadRequest(e.to_string()))?;

    let ctx = build_rag_context(&body, &embeddings, &milvus, &reranker, &config)
        .await?
        .ok_or_else(|| {
            let collection = body.collection_name.as_deref().unwrap_or(DEFAULT_COLLECTION);
            AppError::BadRequest(format!("No documents found in collection '{collection}'"))
        })?;

    let upstream = llm
        .chat_stream_with_system(&ctx.system_prompt, &body.message, &body.model, &body.provider, Some(1024))
        .await
        .map_err(|e| AppError::LlmError(e.to_string()))?;

    // Emit sources as a typed SSE event, then stream LLM tokens
    let sources_json = serde_json::to_string(&ctx.sources).unwrap_or_default();
    let sources_event = format!("event: sources\ndata: {sources_json}\n\n");

    let sources_stream = futures_util::stream::once(async move {
        Ok::<_, actix_web::error::Error>(actix_web::web::Bytes::from(sources_event))
    });

    // Wrap the LLM stream to track TTFT (first chunk) and emit timing at the end
    let llm_byte_stream = upstream.bytes_stream();
    let timed_stream = async_stream::stream! {
        let mut ttft_ms: Option<u64> = None;
        futures_util::pin_mut!(llm_byte_stream);
        while let Some(chunk) = llm_byte_stream.next().await {
            if ttft_ms.is_none() {
                ttft_ms = Some(request_start.elapsed().as_millis() as u64);
            }
            match chunk {
                Ok(bytes) => yield Ok::<_, actix_web::error::Error>(actix_web::web::Bytes::from(bytes.to_vec())),
                Err(e) => {
                    yield Err(actix_web::error::ErrorBadGateway(format!("Stream error: {e}")));
                    return;
                }
            }
        }
        let total_ms = request_start.elapsed().as_millis() as u64;
        let ttft = ttft_ms.unwrap_or(total_ms);
        let timing_event = format!(
            "event: timing\ndata: {{\"ttft_ms\":{ttft},\"total_ms\":{total_ms}}}\n\n"
        );
        yield Ok(actix_web::web::Bytes::from(timing_event));
    };

    Ok(sse_response_from_stream(sources_stream.chain(timed_stream)))
}

// ── SSE helpers ────────────────────────────────────────────────────

fn sse_response_from_stream<S>(stream: S) -> HttpResponse
where
    S: futures_util::Stream<Item = Result<actix_web::web::Bytes, actix_web::error::Error>> + 'static,
{
    HttpResponse::Ok()
        .content_type("text/event-stream")
        .insert_header(("Cache-Control", "no-cache"))
        .insert_header(("X-Accel-Buffering", "no"))
        .streaming(stream)
}
