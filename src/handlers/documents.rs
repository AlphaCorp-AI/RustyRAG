use std::io::Write as _;

use actix_multipart::Multipart;
use actix_web::{post, web, HttpResponse};
use futures_util::StreamExt;
use tempfile::NamedTempFile;

use crate::config::Config;
use crate::errors::AppError;
use crate::prompts::build_contextual_prompt;
use crate::schemas::requests::{DocumentSearchRequest, DocumentUploadBody, DocumentUploadQuery};
use crate::schemas::responses::{
    DocumentSearchHit, DocumentSearchResponse, DocumentUploadResponse,
};
use crate::services::document::{chunk_pages, extract_pages_from_path, unpack_zip_entries, TextChunk};
use crate::services::embeddings::{EmbeddingClient, InputType};
use crate::services::llm::LlmClient;
use crate::services::milvus::{DocumentChunk, MilvusClient, SearchOptions, DEFAULT_COLLECTION};

/// Batch size when inserting into Milvus
const INSERT_BATCH_SIZE: usize = 50;

/// Max number of files from a ZIP processed concurrently
const CONCURRENT_FILES: usize = 8;

// ── Upload endpoint ────────────────────────────────────────────────

/// Upload .txt / .pdf / .zip files, chunk them, embed, and store in Milvus.
///
/// Uses semantic (sentence-boundary-aware) chunking, per-page PDF extraction,
/// and contextual retrieval (LLM-generated context prefix per chunk).
/// All processing parameters are read from the server configuration.
#[utoipa::path(
    post,
    path = "/documents/upload",
    request_body(content = DocumentUploadBody, content_type = "multipart/form-data"),
    params(
        ("collection_name" = Option<String>, Query, description = "Target collection (default: documents)"),
    ),
    responses(
        (status = 200, description = "Documents processed successfully", body = DocumentUploadResponse),
        (status = 400, description = "Bad request"),
        (status = 502, description = "Upstream service error"),
    ),
    tag = "documents"
)]
#[post("/documents/upload")]
pub async fn upload_document(
    mut payload: Multipart,
    query: web::Query<DocumentUploadQuery>,
    config: web::Data<Config>,
    milvus: web::Data<MilvusClient>,
    embeddings: web::Data<EmbeddingClient>,
    llm: web::Data<LlmClient>,
) -> Result<HttpResponse, AppError> {
    if !embeddings.is_configured() {
        return Err(AppError::BadRequest(
            "Embedding API is not configured. \
             Set EMBEDDING_API_URL and EMBEDDING_MODEL."
                .into(),
        ));
    }

    let collection_name = query
        .collection_name
        .as_deref()
        .unwrap_or(DEFAULT_COLLECTION);

    milvus
        .ensure_collection(collection_name)
        .await
        .map_err(|e| AppError::MilvusError(e.to_string()))?;

    // ── Process uploaded files ──────────────────────────────────
    let mut total_chunks = 0usize;
    let mut total_files = 0usize;

    while let Some(field) = payload.next().await {
        let mut field = field.map_err(|e| AppError::BadRequest(format!("Multipart error: {e}")))?;

        let filename = field
            .content_disposition()
            .and_then(|cd| cd.get_filename().map(|s| s.to_string()))
            .unwrap_or_else(|| "unknown".to_string());

        let mut tmp = NamedTempFile::new()
            .map_err(|e| AppError::DocumentError(format!("Failed to create temp file: {e}")))?;

        while let Some(chunk) = field.next().await {
            let chunk =
                chunk.map_err(|e| AppError::BadRequest(format!("Error reading upload: {e}")))?;
            tmp.write_all(&chunk)
                .map_err(|e| AppError::DocumentError(format!("Failed to write temp file: {e}")))?;
        }
        tmp.flush()
            .map_err(|e| AppError::DocumentError(format!("Failed to flush temp file: {e}")))?;

        let ext = filename.rsplit('.').next().unwrap_or("").to_lowercase();

        match ext.as_str() {
            "txt" | "pdf" => {
                let path = tmp.path().to_path_buf();
                let fname = filename.clone();
                let pages =
                    tokio::task::spawn_blocking(move || extract_pages_from_path(&path, &fname))
                        .await
                        .map_err(|e| AppError::DocumentError(format!("Task join error: {e}")))?
                        .map_err(|e| AppError::DocumentError(format!("{filename}: {e}")))?;

                let n = embed_and_insert_chunks(
                    &filename,
                    pages,
                    &config,
                    &embeddings,
                    &milvus,
                    &llm,
                    collection_name,
                )
                .await?;
                total_chunks += n;
                total_files += 1;
            }

            "zip" => {
                let path = tmp.path().to_path_buf();
                let entries = tokio::task::spawn_blocking(move || unpack_zip_entries(&path))
                    .await
                    .map_err(|e| AppError::DocumentError(format!("Task join error: {e}")))?
                    .map_err(|e| AppError::DocumentError(format!("{filename}: {e}")))?;

                let entries_count = entries.len();
                tracing::info!(
                    "ZIP '{}' → {} supported files, processing {} at a time…",
                    filename,
                    entries_count,
                    CONCURRENT_FILES,
                );

                let results: Vec<Result<usize, AppError>> = futures_util::stream::iter(entries)
                    .map(|(name, tmp_file)| {
                        let emb = embeddings.clone();
                        let mil = milvus.clone();
                        let llm_ref = llm.clone();
                        let cfg = config.clone();
                        let coll = collection_name.to_string();
                        async move {
                            let fname = name.clone();
                            let tmp_path = tmp_file.path().to_path_buf();
                            let pages_result = tokio::task::spawn_blocking(move || {
                                extract_pages_from_path(&tmp_path, &fname)
                            })
                            .await;

                            drop(tmp_file);

                            let pages = pages_result
                                .map_err(|e| {
                                    AppError::DocumentError(format!("{name}: task join error: {e}"))
                                })?
                                .map_err(|e| AppError::DocumentError(format!("{name}: {e}")))?;

                            let all_empty = pages.iter().all(|p| p.text.trim().is_empty());
                            if all_empty {
                                tracing::debug!("{name}: empty text, skipping");
                                return Ok(0);
                            }

                            embed_and_insert_chunks(
                                &name,
                                pages,
                                &cfg,
                                &emb,
                                &mil,
                                &llm_ref,
                                &coll,
                            )
                            .await
                        }
                    })
                    .buffer_unordered(CONCURRENT_FILES)
                    .collect()
                    .await;

                let mut skipped = 0usize;
                for res in results {
                    match res {
                        Ok(n) if n > 0 => {
                            total_chunks += n;
                            total_files += 1;
                        }
                        Ok(_) => {}
                        Err(e) => {
                            tracing::warn!("Skipping file: {e}");
                            skipped += 1;
                        }
                    }
                }

                if skipped > 0 {
                    tracing::warn!(
                        "ZIP '{}': {skipped}/{entries_count} files skipped due to errors",
                        filename,
                    );
                }
            }

            _ => {
                return Err(AppError::BadRequest(format!(
                    "Unsupported file type: .{ext}  (allowed: .txt, .pdf, .zip)"
                )));
            }
        }
    }

    if total_files == 0 {
        return Err(AppError::BadRequest(
            "No files uploaded or no text could be extracted".into(),
        ));
    }

    Ok(HttpResponse::Ok().json(DocumentUploadResponse {
        message: "Documents uploaded and embedded successfully".into(),
        collection: collection_name.to_string(),
        total_chunks,
    }))
}

// ── Per-file pipeline helper ───────────────────────────────────────

/// Semantic-chunk pages, generate contextual prefixes, embed, and insert into
/// Milvus.  All parameters are read from `config`.
async fn embed_and_insert_chunks(
    source_file: &str,
    pages: Vec<crate::services::document::PageText>,
    config: &Config,
    embeddings: &EmbeddingClient,
    milvus: &MilvusClient,
    llm: &LlmClient,
    collection_name: &str,
) -> Result<usize, AppError> {
    let chunks = chunk_pages(&pages, config.chunk_size, config.chunk_overlap);

    if chunks.is_empty() {
        return Ok(0);
    }

    tracing::info!("  {} → {} chunks, embedding…", source_file, chunks.len());

    let context_prefixes =
        generate_context_prefixes(&pages, &chunks, config, llm).await?;
    drop(pages);

    // ── Embed and insert in batches ─────────────────────────────
    let mut total = 0usize;
    let embed_batch_size = config.embedding_max_batch_size.max(1);
    let embedding_type = if config.embedding_type.is_empty() {
        None
    } else {
        Some(config.embedding_type.as_str())
    };

    for batch_start in (0..chunks.len()).step_by(embed_batch_size) {
        let batch_end = (batch_start + embed_batch_size).min(chunks.len());
        let batch_chunks = &chunks[batch_start..batch_end];
        let batch_prefixes = &context_prefixes[batch_start..batch_end];

        let texts_to_embed: Vec<String> = batch_chunks
            .iter()
            .zip(batch_prefixes.iter())
            .map(|(chunk, prefix)| {
                if prefix.is_empty() {
                    chunk.text.clone()
                } else {
                    format!("{prefix}\n\n{}", chunk.text)
                }
            })
            .collect();

        let embs = embeddings
            .embed_with_options(&texts_to_embed, InputType::SearchDocument, embedding_type, None)
            .await
            .map_err(|e| AppError::EmbeddingError(e.to_string()))?;

        let doc_chunks: Vec<DocumentChunk> = batch_chunks
            .iter()
            .zip(batch_prefixes.iter())
            .zip(embs)
            .map(|((chunk, prefix), embedding)| DocumentChunk {
                text: chunk.text.clone(),
                source_file: source_file.to_string(),
                chunk_index: chunk.chunk_index as i64,
                page_number: chunk.page_number.map(|p| p as i64).unwrap_or(0),
                context_prefix: prefix.clone(),
                embedding,
            })
            .collect();

        for insert_batch in doc_chunks.chunks(INSERT_BATCH_SIZE) {
            milvus
                .insert(collection_name, insert_batch.to_vec())
                .await
                .map_err(|e| AppError::MilvusError(e.to_string()))?;
        }

        total += batch_chunks.len();
    }

    Ok(total)
}

// ── Contextual retrieval ───────────────────────────────────────────

const MAX_RETRIES: u32 = 6;
const INITIAL_BACKOFF_MS: u64 = 500;
const PAGE_WINDOW: u32 = 2;
const DOC_OVERVIEW_CHARS: usize = 2000;

/// Build a truncated document context for a chunk: a short document overview
/// (first ~2K chars) plus a sliding window of pages around the chunk's location.
/// Capped at `max_chars` total to keep token usage manageable.
fn build_doc_window(
    pages: &[crate::services::document::PageText],
    chunk_page: Option<u32>,
    max_chars: usize,
) -> String {
    let full_text: String = pages.iter().map(|p| p.text.as_str()).collect::<Vec<_>>().join("\n\n");

    if full_text.len() <= max_chars {
        return full_text;
    }

    let overview: String = full_text.chars().take(DOC_OVERVIEW_CHARS).collect();

    let page_num = match chunk_page {
        Some(p) => p,
        None => return full_text.chars().take(max_chars).collect(),
    };

    let lo = page_num.saturating_sub(PAGE_WINDOW);
    let hi = page_num.saturating_add(PAGE_WINDOW);

    let mut window = String::new();
    for p in pages {
        if let Some(pn) = p.page_number {
            if pn >= lo && pn <= hi {
                if !window.is_empty() {
                    window.push_str("\n\n");
                }
                window.push_str(&p.text);
            }
        }
    }

    let budget = max_chars.saturating_sub(overview.len() + 30);
    let window_truncated: String = window.chars().take(budget).collect();

    format!("{overview}\n\n[…]\n\n{window_truncated}")
}

/// Call the LLM in parallel to generate a 1-2 sentence context prefix for each
/// chunk.  A truncated document window (overview + surrounding pages) is included
/// in the prompt so the LLM can situate each chunk (Anthropic's Contextual
/// Retrieval approach).  Retries on 429 rate-limit errors with exponential
/// backoff.  Falls back to an empty string on persistent failures.
async fn generate_context_prefixes(
    pages: &[crate::services::document::PageText],
    chunks: &[TextChunk],
    config: &Config,
    llm: &LlmClient,
) -> Result<Vec<String>, AppError> {
    let concurrency = config.contextual_retrieval_concurrency.max(1);
    let provider = &config.contextual_retrieval_provider;
    let model = &config.contextual_retrieval_model;
    let max_doc_chars = config.contextual_retrieval_max_doc_chars;

    tracing::info!(
        "Generating contextual prefixes for {} chunks (model={}, concurrency={}, max_doc_chars={})…",
        chunks.len(),
        model,
        concurrency,
        max_doc_chars,
    );

    let doc_windows: Vec<String> = chunks
        .iter()
        .map(|chunk| build_doc_window(pages, chunk.page_number, max_doc_chars))
        .collect();

    let results: Vec<String> = futures_util::stream::iter(chunks.iter().zip(doc_windows).enumerate())
        .map(|(idx, (chunk, doc_window))| {
            let prompt = build_contextual_prompt(&doc_window, &chunk.text);
            let model = model.to_string();
            let provider = provider.to_string();
            async move {
                let mut backoff_ms = INITIAL_BACKOFF_MS;
                for attempt in 0..=MAX_RETRIES {
                    match llm.chat(&prompt, &model, &provider, Some(128)).await {
                        Ok(result) => return result.content,
                        Err(e) => {
                            let err_str = e.to_string();
                            let is_rate_limit = err_str.contains("429");
                            if is_rate_limit && attempt < MAX_RETRIES {
                                tracing::debug!(
                                    "Chunk {idx}: rate limited, retrying in {backoff_ms}ms (attempt {}/{})",
                                    attempt + 1,
                                    MAX_RETRIES,
                                );
                                tokio::time::sleep(std::time::Duration::from_millis(backoff_ms)).await;
                                backoff_ms *= 2;
                                continue;
                            }
                            tracing::warn!(
                                "Contextual retrieval failed for chunk {idx}: {e}, using empty prefix"
                            );
                            return String::new();
                        }
                    }
                }
                String::new()
            }
        })
        .buffered(concurrency)
        .collect()
        .await;

    Ok(results)
}

// ── Search endpoint ────────────────────────────────────────────────

/// Semantic search across embedded documents.
#[utoipa::path(
    post,
    path = "/documents/search",
    request_body = DocumentSearchRequest,
    responses(
        (status = 200, description = "Search results", body = DocumentSearchResponse),
        (status = 400, description = "Bad request"),
        (status = 502, description = "Upstream service error"),
    ),
    tag = "documents"
)]
#[post("/documents/search")]
pub async fn search_documents(
    body: web::Json<DocumentSearchRequest>,
    milvus: web::Data<MilvusClient>,
    embeddings: web::Data<EmbeddingClient>,
) -> Result<HttpResponse, AppError> {
    if !embeddings.is_configured() {
        return Err(AppError::BadRequest(
            "Embedding API is not configured. \
             Set EMBEDDING_API_URL and EMBEDDING_MODEL."
                .into(),
        ));
    }

    let collection_name = body
        .collection_name
        .as_deref()
        .unwrap_or(DEFAULT_COLLECTION);
    let limit = body.limit.unwrap_or(10);

    let embs = embeddings
        .embed_with_options(
            &[body.query.clone()],
            InputType::SearchQuery,
            body.embedding_type.as_deref(),
            None,
        )
        .await
        .map_err(|e| AppError::EmbeddingError(e.to_string()))?;

    let query_embedding = embs
        .into_iter()
        .next()
        .ok_or_else(|| AppError::EmbeddingError("No embedding returned for query".into()))?;

    let hits = milvus
        .search(
            collection_name,
            query_embedding,
            limit,
            Some(SearchOptions {
                ef: body.milvus_search_ef,
            }),
        )
        .await
        .map_err(|e| AppError::MilvusError(e.to_string()))?;

    let results: Vec<DocumentSearchHit> = hits
        .into_iter()
        .map(|h| DocumentSearchHit {
            text: h.text,
            source_file: h.source_file,
            chunk_index: h.chunk_index,
            page_number: h.page_number,
            context_prefix: h.context_prefix,
            score: h.score,
        })
        .collect();

    Ok(HttpResponse::Ok().json(DocumentSearchResponse {
        query: body.query.clone(),
        collection: collection_name.to_string(),
        results,
    }))
}
