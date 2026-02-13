use std::io::Write as _;

use actix_multipart::Multipart;
use actix_web::{post, web, HttpResponse};
use futures_util::StreamExt;
use tempfile::NamedTempFile;

use crate::config::Config;
use crate::errors::AppError;
use crate::schemas::requests::{DocumentSearchRequest, DocumentUploadBody, DocumentUploadQuery};
use crate::schemas::responses::{
    DocumentSearchHit, DocumentSearchResponse, DocumentUploadResponse,
};
use crate::services::document::{chunk_text, extract_text_from_path, unpack_zip_entries};
use crate::services::embeddings::{EmbeddingClient, InputType};
use crate::services::milvus::{DocumentChunk, MilvusClient, DEFAULT_COLLECTION};

/// Batch size when calling the embedding API
const EMBED_BATCH_SIZE: usize = 100;

/// Batch size when inserting into Milvus
const INSERT_BATCH_SIZE: usize = 50;

/// Max number of files from a ZIP processed concurrently
/// (extraction + embedding + insertion pipeline per file).
const CONCURRENT_FILES: usize = 8;

// ── Upload endpoint ────────────────────────────────────────────────

/// Upload .txt / .pdf / .zip files, chunk them, embed, and store in Milvus.
///
/// Files are streamed to disk — arbitrarily large uploads (1 GB+) are
/// handled without loading the whole payload into memory.
#[utoipa::path(
    post,
    path = "/documents/upload",
    request_body(content = DocumentUploadBody, content_type = "multipart/form-data"),
    params(
        ("collection_name" = Option<String>, Query, description = "Target collection (default: documents)"),
        ("chunk_size" = Option<usize>, Query, description = "Words per chunk"),
        ("chunk_overlap" = Option<usize>, Query, description = "Overlap words between chunks"),
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
) -> Result<HttpResponse, AppError> {
    // Validate embedding client is configured
    if !embeddings.is_configured() {
        return Err(AppError::BadRequest(
            "Embedding API is not configured. \
             Set COHERE_API_KEY and EMBEDDING_MODEL."
                .into(),
        ));
    }

    let collection_name = query
        .collection_name
        .as_deref()
        .unwrap_or(DEFAULT_COLLECTION);
    let chunk_size = query.chunk_size.unwrap_or(config.chunk_size);
    let chunk_overlap = query.chunk_overlap.unwrap_or(config.chunk_overlap);

    if chunk_size == 0 {
        return Err(AppError::BadRequest("chunk_size must be > 0".into()));
    }
    if chunk_overlap >= chunk_size {
        return Err(AppError::BadRequest(
            "chunk_overlap must be less than chunk_size".into(),
        ));
    }

    // Ensure the Milvus collection exists
    milvus
        .ensure_collection(collection_name)
        .await
        .map_err(|e| AppError::MilvusError(e.to_string()))?;

    // ── Process uploaded files ──────────────────────────────────
    let mut total_chunks = 0usize;
    let mut total_files = 0usize;

    while let Some(field) = payload.next().await {
        let mut field =
            field.map_err(|e| AppError::BadRequest(format!("Multipart error: {e}")))?;

        let filename = field
            .content_disposition()
            .and_then(|cd| cd.get_filename().map(|s| s.to_string()))
            .unwrap_or_else(|| "unknown".to_string());

        // ── Stream upload bytes directly to a temp file ─────────
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

        let ext = filename
            .rsplit('.')
            .next()
            .unwrap_or("")
            .to_lowercase();

        match ext.as_str() {
            // ── Single .txt / .pdf file ─────────────────────────
            "txt" | "pdf" => {
                let path = tmp.path().to_path_buf();
                let fname = filename.clone();
                let text = tokio::task::spawn_blocking(move || {
                    extract_text_from_path(&path, &fname)
                })
                .await
                .map_err(|e| AppError::DocumentError(format!("Task join error: {e}")))?
                .map_err(|e| AppError::DocumentError(format!("{filename}: {e}")))?;

                let n = embed_and_insert_text(
                    &filename,
                    text,
                    chunk_size,
                    chunk_overlap,
                    &embeddings,
                    &milvus,
                    collection_name,
                )
                .await?;
                total_chunks += n;
                total_files += 1;
            }

            // ── ZIP archive — unpack then process entries concurrently ──
            "zip" => {
                let path = tmp.path().to_path_buf();
                let entries = tokio::task::spawn_blocking(move || {
                    unpack_zip_entries(&path)
                })
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

                // Process entries concurrently:
                //   extract text (blocking) → chunk → embed → insert
                let results: Vec<Result<usize, AppError>> =
                    futures_util::stream::iter(entries)
                        .map(|(name, tmp_file)| {
                            let emb = embeddings.clone();
                            let mil = milvus.clone();
                            let coll = collection_name.to_string();
                            async move {
                                // Extract text on the blocking thread pool
                                let fname = name.clone();
                                let tmp_path = tmp_file.path().to_path_buf();
                                let text_result =
                                    tokio::task::spawn_blocking(move || {
                                        extract_text_from_path(&tmp_path, &fname)
                                    })
                                    .await;

                                // Release the temp file ASAP
                                drop(tmp_file);

                                let text = text_result
                                    .map_err(|e| {
                                        AppError::DocumentError(format!(
                                            "{name}: task join error: {e}"
                                        ))
                                    })?
                                    .map_err(|e| {
                                        AppError::DocumentError(format!(
                                            "{name}: {e}"
                                        ))
                                    })?;

                                if text.trim().is_empty() {
                                    tracing::debug!("{name}: empty text, skipping");
                                    return Ok(0);
                                }

                                embed_and_insert_text(
                                    &name,
                                    text,
                                    chunk_size,
                                    chunk_overlap,
                                    &emb,
                                    &mil,
                                    &coll,
                                )
                                .await
                            }
                        })
                        .buffer_unordered(CONCURRENT_FILES)
                        .collect()
                        .await;

                // Aggregate — log failures but keep going
                let mut skipped = 0usize;
                for res in results {
                    match res {
                        Ok(n) if n > 0 => {
                            total_chunks += n;
                            total_files += 1;
                        }
                        Ok(_) => { /* empty text, already logged */ }
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
        // `tmp` dropped here → temp file deleted from disk
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
        chunk_size,
        chunk_overlap,
    }))
}

// ── Per-file pipeline helper ───────────────────────────────────────

/// Chunk a single file's text, embed in batches, and insert into Milvus.
///
/// Memory usage stays proportional to one embed batch (~100 chunks ×
/// dim × 4 bytes) rather than all chunks for the whole upload.
async fn embed_and_insert_text(
    source_file: &str,
    text: String,
    chunk_size: usize,
    chunk_overlap: usize,
    embeddings: &EmbeddingClient,
    milvus: &MilvusClient,
    collection_name: &str,
) -> Result<usize, AppError> {
    let chunks = chunk_text(&text, chunk_size, chunk_overlap);
    // `text` is moved into chunk_text and freed after splitting
    drop(text);

    if chunks.is_empty() {
        return Ok(0);
    }

    tracing::info!(
        "  {} → {} chunks, embedding…",
        source_file,
        chunks.len(),
    );

    let mut total = 0usize;

    for batch_start in (0..chunks.len()).step_by(EMBED_BATCH_SIZE) {
        let batch_end = (batch_start + EMBED_BATCH_SIZE).min(chunks.len());
        let batch = &chunks[batch_start..batch_end];

        // Embed this batch
        let texts: Vec<String> = batch.to_vec();
        let embs = embeddings
            .embed(&texts, InputType::SearchDocument)
            .await
            .map_err(|e| AppError::EmbeddingError(e.to_string()))?;

        // Build DocumentChunks with correct global chunk indices
        let doc_chunks: Vec<DocumentChunk> = batch
            .iter()
            .enumerate()
            .zip(embs)
            .map(|((idx, text), embedding)| DocumentChunk {
                text: text.clone(),
                source_file: source_file.to_string(),
                chunk_index: (batch_start + idx) as i64,
                embedding,
            })
            .collect();

        // Insert into Milvus in sub-batches
        for insert_batch in doc_chunks.chunks(INSERT_BATCH_SIZE) {
            milvus
                .insert(collection_name, insert_batch.to_vec())
                .await
                .map_err(|e| AppError::MilvusError(e.to_string()))?;
        }

        total += batch.len();
    }

    Ok(total)
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
             Set COHERE_API_KEY and EMBEDDING_MODEL."
                .into(),
        ));
    }

    let collection_name = body
        .collection_name
        .as_deref()
        .unwrap_or(DEFAULT_COLLECTION);
    let limit = body.limit.unwrap_or(10);

    // Embed the query
    let embs = embeddings
        .embed(&[body.query.clone()], InputType::SearchQuery)
        .await
        .map_err(|e| AppError::EmbeddingError(e.to_string()))?;

    let query_embedding = embs
        .into_iter()
        .next()
        .ok_or_else(|| AppError::EmbeddingError("No embedding returned for query".into()))?;

    // Search Milvus
    let hits = milvus
        .search(collection_name, query_embedding, limit)
        .await
        .map_err(|e| AppError::MilvusError(e.to_string()))?;

    let results: Vec<DocumentSearchHit> = hits
        .into_iter()
        .map(|h| DocumentSearchHit {
            text: h.text,
            source_file: h.source_file,
            chunk_index: h.chunk_index,
            score: h.score,
        })
        .collect();

    Ok(HttpResponse::Ok().json(DocumentSearchResponse {
        query: body.query.clone(),
        collection: collection_name.to_string(),
        results,
    }))
}
