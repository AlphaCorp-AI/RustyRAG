use std::io::Write as _;

use actix_multipart::Multipart;
use actix_web::{get, post, web, HttpResponse};
use flate2::write::GzEncoder;
use flate2::Compression;
use futures_util::StreamExt;
use tempfile::NamedTempFile;

use crate::config::Config;
use crate::errors::AppError;
use crate::prompts::build_contextual_prompt;
use crate::schemas::requests::{DocumentSearchRequest, DocumentUploadBody, DocumentUploadQuery};
use crate::schemas::responses::{
    DocumentSearchHit, DocumentSearchResponse, DocumentUploadResponse,
};
use crate::services::docling::DoclingClient;
use crate::services::document::{
    chunk_pages, extract_document, is_supported_extension, unpack_zip_entries, TextChunk,
};
use crate::services::embeddings::{EmbeddingClient, InputType};
use crate::services::llm::LlmClient;
use crate::services::milvus::{DocumentChunk, MilvusClient, SearchOptions, DEFAULT_COLLECTION};

use super::chat::validate_collection_name;

const INSERT_BATCH_SIZE: usize = 50;
const CONCURRENT_FILES: usize = 2;

// ── Upload endpoint ────────────────────────────────────────────────

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
    tag = "Documents"
)]
#[post("/documents/upload")]
pub async fn upload_document(
    mut payload: Multipart,
    query: web::Query<DocumentUploadQuery>,
    config: web::Data<Config>,
    milvus: web::Data<MilvusClient>,
    embeddings: web::Data<EmbeddingClient>,
    llm: web::Data<LlmClient>,
    docling: web::Data<DoclingClient>,
) -> Result<HttpResponse, AppError> {
    if !embeddings.is_configured() {
        return Err(AppError::BadRequest(
            "Embedding API is not configured. Set EMBEDDING_API_URL and EMBEDDING_MODEL.".into(),
        ));
    }

    let collection_name = query
        .collection_name
        .as_deref()
        .unwrap_or(DEFAULT_COLLECTION);
    validate_collection_name(collection_name)?;

    milvus
        .ensure_collection(collection_name)
        .await
        .map_err(|e| AppError::MilvusError(e.to_string()))?;

    let mut total_chunks = 0usize;
    let mut total_files = 0usize;

    while let Some(field) = payload.next().await {
        let mut field = field.map_err(|e| AppError::BadRequest(format!("Multipart error: {e}")))?;

        let filename = field
            .content_disposition()
            .and_then(|cd| cd.get_filename().map(|s| s.to_string()))
            .unwrap_or_else(|| "unknown".to_string());

        let mut tmp = NamedTempFile::new()
            .map_err(|e| AppError::DocumentError(format!("Temp file: {e}")))?;

        while let Some(chunk) = field.next().await {
            let chunk =
                chunk.map_err(|e| AppError::BadRequest(format!("Error reading upload: {e}")))?;
            tmp.write_all(&chunk)
                .map_err(|e| AppError::DocumentError(format!("Write error: {e}")))?;
        }
        tmp.flush()
            .map_err(|e| AppError::DocumentError(format!("Flush error: {e}")))?;

        let file_size = tmp
            .as_file()
            .metadata()
            .map(|m| m.len() as i64)
            .unwrap_or(0);

        let ext = filename.rsplit('.').next().unwrap_or("").to_lowercase();

        match ext.as_str() {
            _ if ext == "zip" => {
                let path = tmp.path().to_path_buf();
                let entries = tokio::task::spawn_blocking(move || unpack_zip_entries(&path))
                    .await
                    .map_err(|e| AppError::DocumentError(format!("Task join error: {e}")))?
                    .map_err(|e| AppError::DocumentError(format!("{filename}: {e}")))?;

                let entries_count = entries.len();
                tracing::info!(
                    "ZIP '{filename}' → {entries_count} files, processing {CONCURRENT_FILES} at a time…"
                );

                let results: Vec<Result<usize, AppError>> = futures_util::stream::iter(entries)
                    .map(|(name, tmp_file)| {
                        let emb = embeddings.clone();
                        let mil = milvus.clone();
                        let llm_ref = llm.clone();
                        let cfg = config.clone();
                        let doc = docling.clone();
                        let coll = collection_name.to_string();
                        async move {
                            let entry_size = tmp_file
                                .as_file()
                                .metadata()
                                .map(|m| m.len() as i64)
                                .unwrap_or(0);

                            let pages = extract_document(tmp_file.path(), &name, &doc)
                                .await
                                .map_err(|e| AppError::DocumentError(format!("{name}: {e}")))?;

                            drop(tmp_file);

                            if pages.iter().all(|p| p.text.trim().is_empty()) {
                                tracing::warn!("{name}: empty text after extraction, skipping");
                                return Ok(0);
                            }

                            embed_and_insert_chunks(
                                &name, entry_size, pages, &cfg, &emb, &mil, &llm_ref, &coll,
                            )
                            .await
                        }
                    })
                    .buffer_unordered(CONCURRENT_FILES)
                    .collect()
                    .await;

                let mut skipped_errors = 0usize;
                let mut skipped_empty = 0usize;
                for res in results {
                    match res {
                        Ok(n) if n > 0 => {
                            total_chunks += n;
                            total_files += 1;
                        }
                        Ok(_) => {
                            skipped_empty += 1;
                        }
                        Err(e) => {
                            tracing::warn!("Skipping file: {e}");
                            skipped_errors += 1;
                        }
                    }
                }
                let skipped_total = skipped_errors + skipped_empty;
                if skipped_total > 0 {
                    tracing::warn!(
                        "ZIP '{filename}': {skipped_total}/{entries_count} files skipped \
                         ({skipped_errors} errors, {skipped_empty} empty)"
                    );
                }
            }

            _ if is_supported_extension(&ext) => {
                let pages = extract_document(tmp.path(), &filename, &docling)
                    .await
                    .map_err(|e| AppError::DocumentError(format!("{filename}: {e}")))?;

                let n = embed_and_insert_chunks(
                    &filename,
                    file_size,
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

            _ => {
                return Err(AppError::BadRequest(format!(
                    "Unsupported file type: .{ext} (allowed: .txt, .pdf, .docx, .pptx, .xlsx, .html, .zip)"
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

// ── Per-file pipeline ──────────────────────────────────────────────

async fn embed_and_insert_chunks(
    file_name: &str,
    file_size: i64,
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

    tracing::info!("  {file_name} → {} chunks, embedding…", chunks.len());

    let context_prefixes = if config.contextual_retrieval_enabled {
        generate_context_prefixes(&pages, &chunks, config, llm).await?
    } else {
        vec![String::new(); chunks.len()]
    };
    drop(pages);

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
                file_name: file_name.to_string(),
                file_size,
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

fn build_doc_window(
    pages: &[crate::services::document::PageText],
    chunk_page: Option<u32>,
    max_chars: usize,
) -> String {
    let total_len: usize = pages.iter().map(|p| p.text.len()).sum::<usize>()
        + pages.len().saturating_sub(1) * 2;

    if total_len <= max_chars {
        return pages
            .iter()
            .map(|p| p.text.as_str())
            .collect::<Vec<_>>()
            .join("\n\n");
    }

    let mut overview = String::with_capacity(DOC_OVERVIEW_CHARS);
    for (i, p) in pages.iter().enumerate() {
        if overview.len() >= DOC_OVERVIEW_CHARS {
            break;
        }
        if i > 0 {
            overview.push_str("\n\n");
        }
        let remaining = DOC_OVERVIEW_CHARS - overview.len();
        if p.text.len() <= remaining {
            overview.push_str(&p.text);
        } else {
            overview.extend(p.text.chars().take(remaining));
        }
    }

    let page_num = match chunk_page {
        Some(p) => p,
        None => {
            overview.truncate(max_chars);
            return overview;
        }
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
        "Generating contextual prefixes for {} chunks (model={model}, concurrency={concurrency})…",
        chunks.len(),
    );

    let doc_windows: Vec<String> = chunks
        .iter()
        .map(|chunk| build_doc_window(pages, chunk.page_number, max_doc_chars))
        .collect();

    let results: Vec<String> =
        futures_util::stream::iter(chunks.iter().zip(doc_windows).enumerate())
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
                                if e.to_string().contains("429") && attempt < MAX_RETRIES {
                                    tracing::debug!(
                                        "Chunk {idx}: rate limited, retry in {backoff_ms}ms ({}/{})",
                                        attempt + 1,
                                        MAX_RETRIES,
                                    );
                                    tokio::time::sleep(std::time::Duration::from_millis(backoff_ms))
                                        .await;
                                    backoff_ms *= 2;
                                    continue;
                                }
                                tracing::warn!("Context prefix failed for chunk {idx}: {e}");
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

#[utoipa::path(
    post,
    path = "/documents/search",
    request_body = DocumentSearchRequest,
    responses(
        (status = 200, description = "Search results", body = DocumentSearchResponse),
        (status = 400, description = "Bad request"),
        (status = 502, description = "Upstream service error"),
    ),
    tag = "Documents"
)]
#[post("/documents/search")]
pub async fn search_documents(
    body: web::Json<DocumentSearchRequest>,
    milvus: web::Data<MilvusClient>,
    embeddings: web::Data<EmbeddingClient>,
) -> Result<HttpResponse, AppError> {
    if !embeddings.is_configured() {
        return Err(AppError::BadRequest(
            "Embedding API is not configured. Set EMBEDDING_API_URL and EMBEDDING_MODEL.".into(),
        ));
    }

    let collection_name = body
        .collection_name
        .as_deref()
        .unwrap_or(DEFAULT_COLLECTION);
    validate_collection_name(collection_name)?;
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

    let results: Vec<DocumentSearchHit> = hits.into_iter().map(DocumentSearchHit::from).collect();

    Ok(HttpResponse::Ok().json(DocumentSearchResponse {
        query: body.query.clone(),
        collection: collection_name.to_string(),
        results,
    }))
}

// ── Backup endpoint ─────────────────────────────────────────────

#[utoipa::path(
    get,
    path = "/documents/backup",
    params(
        ("collection_name" = Option<String>, Query, description = "Collection to back up (default: documents)"),
    ),
    responses(
        (status = 200, description = "JSON backup file download", content_type = "application/json"),
        (status = 400, description = "Bad request"),
        (status = 502, description = "Upstream service error"),
    ),
    tag = "Documents"
)]
#[get("/documents/backup")]
pub async fn backup_collection(
    query: web::Query<DocumentUploadQuery>,
    milvus: web::Data<MilvusClient>,
) -> Result<HttpResponse, AppError> {
    let collection_name = query
        .collection_name
        .as_deref()
        .unwrap_or(DEFAULT_COLLECTION);
    validate_collection_name(collection_name)?;

    tracing::info!("Starting backup of collection '{collection_name}'…");

    let rows = milvus
        .query_all(collection_name)
        .await
        .map_err(|e| AppError::MilvusError(e.to_string()))?;

    tracing::info!(
        "Backup of '{collection_name}' complete: {} rows",
        rows.len()
    );

    let backup = serde_json::json!({
        "collection": collection_name,
        "count": rows.len(),
        "data": rows,
    });

    let json_bytes = serde_json::to_vec(&backup)
        .map_err(|e| AppError::Internal(anyhow::anyhow!("JSON serialization error: {e}")))?;

    let mut encoder = GzEncoder::new(Vec::new(), Compression::fast());
    encoder
        .write_all(&json_bytes)
        .map_err(|e| AppError::Internal(anyhow::anyhow!("Gzip compression error: {e}")))?;
    let compressed = encoder
        .finish()
        .map_err(|e| AppError::Internal(anyhow::anyhow!("Gzip finish error: {e}")))?;

    tracing::info!(
        "Backup compressed: {}MB → {}MB",
        json_bytes.len() / (1024 * 1024),
        compressed.len() / (1024 * 1024),
    );

    let filename = format!("{collection_name}_backup.json.gz");

    Ok(HttpResponse::Ok()
        .content_type("application/gzip")
        .insert_header((
            "Content-Disposition",
            format!("attachment; filename=\"{filename}\""),
        ))
        .body(compressed))
}
