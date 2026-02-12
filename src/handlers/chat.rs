use actix_web::{post, web, HttpResponse};
use futures_util::StreamExt;
use validator::Validate;

use crate::errors::AppError;
use crate::schemas::requests::{ChatRagRequest, ChatRequest};
use crate::schemas::responses::{
    ChatRagResponse, ChatResponse, ErrorResponse, RagSource, Usage,
};
use crate::services::embeddings::EmbeddingClient;
use crate::services::llm::LlmClient;
use crate::services::milvus::MilvusClient;

/// Send a message to the LLM via OpenRouter
#[utoipa::path(
    post,
    path = "/chat",
    request_body = ChatRequest,
    responses(
        (status = 200, description = "Chat completion", body = ChatResponse),
        (status = 400, description = "Validation error", body = ErrorResponse),
        (status = 502, description = "LLM provider error", body = ErrorResponse),
    ),
    tag = "chat"
)]
#[post("/chat")]
pub async fn chat(
    llm: web::Data<LlmClient>,
    body: web::Json<ChatRequest>,
) -> Result<HttpResponse, AppError> {
    body.validate()
        .map_err(|e| AppError::BadRequest(e.to_string()))?;

    let result = llm
        .chat(&body.message, body.model.as_deref(), body.max_tokens)
        .await
        .map_err(|e| AppError::LlmError(e.to_string()))?;

    let usage = result.usage.map(|u| Usage {
        prompt_tokens: u.prompt_tokens,
        completion_tokens: u.completion_tokens,
        total_tokens: u.total_tokens,
    });

    Ok(HttpResponse::Ok().json(ChatResponse {
        model: result.model,
        message: result.content,
        usage,
    }))
}

// ── Streaming endpoint ──────────────────────────────────────────────

/// Stream a chat completion via Server-Sent Events.
///
/// Proxies the SSE stream from OpenRouter directly to the client.
#[utoipa::path(
    post,
    path = "/chat/stream",
    request_body = ChatRequest,
    responses(
        (status = 200, description = "SSE stream of chat tokens", content_type = "text/event-stream"),
        (status = 400, description = "Validation error", body = ErrorResponse),
        (status = 502, description = "LLM provider error", body = ErrorResponse),
    ),
    tag = "chat"
)]
#[post("/chat/stream")]
pub async fn chat_stream(
    llm: web::Data<LlmClient>,
    body: web::Json<ChatRequest>,
) -> Result<HttpResponse, AppError> {
    body.validate()
        .map_err(|e| AppError::BadRequest(e.to_string()))?;

    let upstream = llm
        .chat_stream(&body.message)
        .await
        .map_err(|e| AppError::LlmError(e.to_string()))?;

    // Forward the upstream SSE byte stream to the client
    let byte_stream = upstream.bytes_stream().map(|chunk| {
        chunk
            .map(|b| actix_web::web::Bytes::from(b.to_vec()))
            .map_err(|e| {
                actix_web::error::ErrorBadGateway(format!("Stream error: {e}"))
            })
    });

    Ok(HttpResponse::Ok()
        .content_type("text/event-stream")
        .insert_header(("Cache-Control", "no-cache"))
        .insert_header(("X-Accel-Buffering", "no"))
        .streaming(byte_stream))
}

// ── RAG endpoint ────────────────────────────────────────────────────

/// Ask a question grounded in documents from a Milvus collection.
///
/// 1. Embeds the question
/// 2. Retrieves the most similar chunks from Milvus
/// 3. Sends them as context to the LLM
/// 4. Returns the answer together with the source chunks
#[utoipa::path(
    post,
    path = "/chat-rag",
    request_body = ChatRagRequest,
    responses(
        (status = 200, description = "RAG answer with sources", body = ChatRagResponse),
        (status = 400, description = "Validation error", body = ErrorResponse),
        (status = 502, description = "Upstream service error", body = ErrorResponse),
    ),
    tag = "chat"
)]
#[post("/chat-rag")]
pub async fn chat_rag(
    body: web::Json<ChatRagRequest>,
    llm: web::Data<LlmClient>,
    milvus: web::Data<MilvusClient>,
    embeddings: web::Data<EmbeddingClient>,
) -> Result<HttpResponse, AppError> {
    body.validate()
        .map_err(|e| AppError::BadRequest(e.to_string()))?;

    if !embeddings.is_configured() {
        return Err(AppError::BadRequest(
            "Embedding API is not configured. \
             Set EMBEDDING_MODEL, EMBEDDING_API_URL, and EMBEDDING_API_KEY."
                .into(),
        ));
    }

    let limit = body.limit.unwrap_or(5);

    // ── 1. Embed the user's question ────────────────────────────
    let embs = embeddings
        .embed(&[body.message.clone()])
        .await
        .map_err(|e| AppError::EmbeddingError(e.to_string()))?;

    let query_embedding = embs
        .into_iter()
        .next()
        .ok_or_else(|| AppError::EmbeddingError("No embedding returned for query".into()))?;

    // ── 2. Retrieve similar chunks from Milvus ──────────────────
    let hits = milvus
        .search(&body.collection_name, query_embedding, limit)
        .await
        .map_err(|e| AppError::MilvusError(e.to_string()))?;

    if hits.is_empty() {
        return Err(AppError::BadRequest(format!(
            "No documents found in collection '{}'",
            body.collection_name
        )));
    }

    let sources: Vec<RagSource> = hits
        .iter()
        .map(|h| RagSource {
            text: h.text.clone(),
            source_file: h.source_file.clone(),
            chunk_index: h.chunk_index,
            score: h.score,
        })
        .collect();

    // ── 3. Build context and call the LLM ───────────────────────
    let context = hits
        .iter()
        .enumerate()
        .map(|(i, h)| {
            format!(
                "[Source {} — {}]\n{}",
                i + 1,
                h.source_file,
                h.text
            )
        })
        .collect::<Vec<_>>()
        .join("\n\n");

    let system_prompt = format!(
        "Answer the question directly and concisely using only the context below. \
         Do not mention or cite sources in your answer.\n\n\
         ---\n{context}\n---"
    );

    let result = llm
        .chat_with_system(
            &system_prompt,
            &body.message,
            None,
            None,
        )
        .await
        .map_err(|e| AppError::LlmError(e.to_string()))?;

    let usage = result.usage.map(|u| Usage {
        prompt_tokens: u.prompt_tokens,
        completion_tokens: u.completion_tokens,
        total_tokens: u.total_tokens,
    });

    Ok(HttpResponse::Ok().json(ChatRagResponse {
        model: result.model,
        message: result.content,
        sources,
        usage,
    }))
}

// ── Streaming RAG endpoint ──────────────────────────────────────────

/// Stream a RAG answer via Server-Sent Events.
///
/// 1. Embeds the question
/// 2. Retrieves similar chunks from Milvus
/// 3. Emits sources as `event: sources` SSE event
/// 4. Streams the LLM answer tokens as standard SSE `data:` lines
#[utoipa::path(
    post,
    path = "/chat-rag/stream",
    request_body = ChatRagRequest,
    responses(
        (status = 200, description = "SSE stream: sources event then LLM tokens", content_type = "text/event-stream"),
        (status = 400, description = "Validation error", body = ErrorResponse),
        (status = 502, description = "Upstream service error", body = ErrorResponse),
    ),
    tag = "chat"
)]
#[post("/chat-rag/stream")]
pub async fn chat_rag_stream(
    body: web::Json<ChatRagRequest>,
    llm: web::Data<LlmClient>,
    milvus: web::Data<MilvusClient>,
    embeddings: web::Data<EmbeddingClient>,
) -> Result<HttpResponse, AppError> {
    body.validate()
        .map_err(|e| AppError::BadRequest(e.to_string()))?;

    if !embeddings.is_configured() {
        return Err(AppError::BadRequest(
            "Embedding API is not configured. \
             Set EMBEDDING_MODEL, EMBEDDING_API_URL, and EMBEDDING_API_KEY."
                .into(),
        ));
    }

    let limit = body.limit.unwrap_or(5);

    // ── 1. Embed the user's question ────────────────────────────
    let embs = embeddings
        .embed(&[body.message.clone()])
        .await
        .map_err(|e| AppError::EmbeddingError(e.to_string()))?;

    let query_embedding = embs
        .into_iter()
        .next()
        .ok_or_else(|| AppError::EmbeddingError("No embedding returned for query".into()))?;

    // ── 2. Retrieve similar chunks from Milvus ──────────────────
    let hits = milvus
        .search(&body.collection_name, query_embedding, limit)
        .await
        .map_err(|e| AppError::MilvusError(e.to_string()))?;

    if hits.is_empty() {
        return Err(AppError::BadRequest(format!(
            "No documents found in collection '{}'",
            body.collection_name
        )));
    }

    let sources: Vec<RagSource> = hits
        .iter()
        .map(|h| RagSource {
            text: h.text.clone(),
            source_file: h.source_file.clone(),
            chunk_index: h.chunk_index,
            score: h.score,
        })
        .collect();

    // ── 3. Build context ────────────────────────────────────────
    let context = hits
        .iter()
        .enumerate()
        .map(|(i, h)| {
            format!(
                "[Source {} — {}]\n{}",
                i + 1,
                h.source_file,
                h.text
            )
        })
        .collect::<Vec<_>>()
        .join("\n\n");

    let system_prompt = format!(
        "Answer the question directly and concisely using only the context below. \
         Do not mention or cite sources in your answer.\n\n\
         ---\n{context}\n---"
    );

    // ── 4. Start streaming LLM response ─────────────────────────
    let upstream = llm
        .chat_stream_with_system(&system_prompt, &body.message)
        .await
        .map_err(|e| AppError::LlmError(e.to_string()))?;

    // Build a stream that first emits the sources event, then proxies LLM SSE
    let sources_json = serde_json::to_string(&sources).unwrap_or_default();
    let sources_event = format!("event: sources\ndata: {sources_json}\n\n");

    let sources_stream =
        futures_util::stream::once(async move {
            Ok::<_, actix_web::error::Error>(actix_web::web::Bytes::from(sources_event))
        });

    let llm_stream = upstream.bytes_stream().map(|chunk| {
        chunk
            .map(|b| actix_web::web::Bytes::from(b.to_vec()))
            .map_err(|e| actix_web::error::ErrorBadGateway(format!("Stream error: {e}")))
    });

    let combined = sources_stream.chain(llm_stream);

    Ok(HttpResponse::Ok()
        .content_type("text/event-stream")
        .insert_header(("Cache-Control", "no-cache"))
        .insert_header(("X-Accel-Buffering", "no"))
        .streaming(combined))
}
