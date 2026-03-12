use std::path::Path;
use std::time::Instant;

use anyhow::Context;
use futures_util::StreamExt;
use serde::Deserialize;

use crate::config::Config;
use crate::prompts::build_contextual_prompt;
use crate::services::document::{chunk_pages, extract_pages_from_path, PageText, TextChunk};
use crate::services::embeddings::{EmbeddingClient, InputType};
use crate::services::llm::LlmClient;
use crate::services::milvus::{DocumentChunk, MilvusClient, SearchOptions};
use crate::services::reranker::RerankerClient;

use super::eval_client::EvalClient;
use super::prompts::{build_competition_system_prompt, parse_answer, strip_sources_line};
use super::submission::*;

// ── Competition configuration ───────────────────────────────────────

#[derive(Debug, Deserialize, Clone)]
pub struct CompetitionConfig {
    #[serde(default)]
    pub eval_api_key: String,
    #[serde(default = "default_eval_base_url")]
    pub eval_base_url: String,
    #[serde(default = "default_docs_dir")]
    pub docs_dir: String,
    #[serde(default = "default_questions_path")]
    pub questions_path: String,
    #[serde(default = "default_submission_path")]
    pub submission_path: String,
    #[serde(default = "default_collection_name")]
    pub collection_name: String,
    #[serde(default = "default_competition_model")]
    pub competition_model: String,
    #[serde(default = "default_competition_provider")]
    pub competition_provider: String,
    #[serde(default = "default_competition_top_k")]
    pub competition_top_k: i64,
}

fn default_eval_base_url() -> String {
    "https://platform.agentic-challenge.ai/api/v1".to_string()
}
fn default_docs_dir() -> String {
    "docs_corpus".to_string()
}
fn default_questions_path() -> String {
    "questions.json".to_string()
}
fn default_submission_path() -> String {
    "submission.json".to_string()
}
fn default_collection_name() -> String {
    "competition".to_string()
}
fn default_competition_model() -> String {
    "qwen-3-235b-a22b-instruct-2507".to_string()
}
fn default_competition_provider() -> String {
    "cerebras".to_string()
}
fn default_competition_top_k() -> i64 {
    20
}

impl CompetitionConfig {
    pub fn from_env() -> Result<Self, envy::Error> {
        envy::from_env::<Self>()
    }
}

// ── Streaming result ────────────────────────────────────────────────

struct StreamResult {
    content: String,
    ttft_ms: u64,
    tpot_ms: u64,
    total_time_ms: u64,
    input_tokens: u32,
    output_tokens: u32,
}

// ── Main pipeline commands ──────────────────────────────────────────

/// Download questions and documents from the eval API.
pub async fn download(comp: &CompetitionConfig) -> anyhow::Result<()> {
    let client = EvalClient::new(&comp.eval_base_url, &comp.eval_api_key);

    // Download questions
    tracing::info!("Downloading questions...");
    let questions = client.download_questions().await?;
    let json = serde_json::to_string_pretty(&questions)?;
    std::fs::write(&comp.questions_path, json)?;
    tracing::info!("Saved {} questions to {}", questions.len(), comp.questions_path);

    // Download documents
    tracing::info!("Downloading documents...");
    client.download_documents(&comp.docs_dir).await?;

    Ok(())
}

/// Ingest all PDF documents into Milvus.
pub async fn ingest(config: &Config, comp: &CompetitionConfig) -> anyhow::Result<()> {
    let milvus = create_milvus(config);
    let embeddings = create_embeddings(config);
    let llm = LlmClient::new(config.clone());

    milvus.ensure_collection(&comp.collection_name).await?;

    let docs_dir = Path::new(&comp.docs_dir);
    if !docs_dir.exists() {
        anyhow::bail!("Documents directory '{}' does not exist", comp.docs_dir);
    }

    // Find all PDFs
    let mut pdf_paths: Vec<_> = std::fs::read_dir(docs_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .map(|ext| ext.eq_ignore_ascii_case("pdf"))
                .unwrap_or(false)
        })
        .map(|e| e.path())
        .collect();

    // Also check subdirectories (one level deep)
    if let Ok(entries) = std::fs::read_dir(docs_dir) {
        for entry in entries.filter_map(|e| e.ok()) {
            if entry.path().is_dir() {
                if let Ok(sub_entries) = std::fs::read_dir(entry.path()) {
                    for sub_entry in sub_entries.filter_map(|e| e.ok()) {
                        if sub_entry
                            .path()
                            .extension()
                            .map(|ext| ext.eq_ignore_ascii_case("pdf"))
                            .unwrap_or(false)
                        {
                            pdf_paths.push(sub_entry.path());
                        }
                    }
                }
            }
        }
    }

    pdf_paths.sort();
    pdf_paths.dedup();
    tracing::info!("Found {} PDF files to ingest", pdf_paths.len());

    let embed_batch_size = config.embedding_max_batch_size.max(1);
    let embedding_type = if config.embedding_type.is_empty() {
        None
    } else {
        Some(config.embedding_type.as_str())
    };

    let mut total_chunks = 0usize;

    for (file_idx, pdf_path) in pdf_paths.iter().enumerate() {
        let filename = pdf_path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();

        tracing::info!(
            "[{}/{}] Processing {}",
            file_idx + 1,
            pdf_paths.len(),
            filename
        );

        // Extract pages (with OCR fallback for scanned docs)
        let path_clone = pdf_path.clone();
        let fname_clone = filename.clone();
        let pages = match tokio::task::spawn_blocking(move || {
            extract_pages_from_path(&path_clone, &fname_clone)
        })
        .await?
        {
            Ok(pages) => pages,
            Err(e) => {
                tracing::warn!("  Skipping {filename}: {e}");
                continue;
            }
        };

        if pages.iter().all(|p| p.text.trim().is_empty()) {
            tracing::warn!("  Skipping {filename}: no extractable text");
            continue;
        }

        // Chunk
        let chunks = chunk_pages(&pages, config.chunk_size, config.chunk_overlap);
        if chunks.is_empty() {
            continue;
        }

        tracing::info!("  {} → {} chunks, embedding...", filename, chunks.len());

        // Optional contextual retrieval
        let context_prefixes = if config.contextual_retrieval_enabled {
            generate_context_prefixes(&pages, &chunks, config, &llm).await?
        } else {
            vec![String::new(); chunks.len()]
        };

        // Embed and insert in batches
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
                .context("Embedding failed")?;

            let doc_chunks: Vec<DocumentChunk> = batch_chunks
                .iter()
                .zip(batch_prefixes.iter())
                .zip(embs)
                .map(|((chunk, prefix), embedding)| DocumentChunk {
                    text: chunk.text.clone(),
                    source_file: filename.clone(),
                    chunk_index: chunk.chunk_index as i64,
                    page_number: chunk.page_number.map(|p| p as i64).unwrap_or(0),
                    context_prefix: prefix.clone(),
                    embedding,
                })
                .collect();

            for insert_batch in doc_chunks.chunks(50) {
                milvus
                    .insert(&comp.collection_name, insert_batch.to_vec())
                    .await
                    .context("Milvus insert failed")?;
            }

            total_chunks += batch_chunks.len();
        }
    }

    tracing::info!(
        "Ingestion complete: {} total chunks in collection '{}'",
        total_chunks,
        comp.collection_name
    );
    Ok(())
}

/// Answer all questions and generate submission.json.
pub async fn answer(config: &Config, comp: &CompetitionConfig) -> anyhow::Result<()> {
    let milvus = create_milvus(config);
    let embeddings = create_embeddings(config);
    let llm = LlmClient::new(config.clone());
    let reranker = RerankerClient::new(&config.reranker_api_url, &config.jina_api_key, &config.reranker_model);

    if reranker.is_configured() {
        tracing::info!("Reranker enabled at {}", config.reranker_api_url);
    } else {
        tracing::warn!("Reranker not configured — skipping reranking step");
    }

    // Load questions
    let questions = load_questions(&comp.questions_path)?;
    tracing::info!("Loaded {} questions", questions.len());

    let embedding_type = if config.embedding_type.is_empty() {
        None
    } else {
        Some(config.embedding_type.as_str())
    };

    let mut builder = SubmissionBuilder::new(
        "RustyRAG: GPU-accelerated Rust RAG with Jina v5 embeddings, Jina reranker v3, \
         Milvus GPU vector search, and Cerebras LLM inference",
    );

    for (idx, question) in questions.iter().enumerate() {
        tracing::info!(
            "[{}/{}] {} (type={})",
            idx + 1,
            questions.len(),
            &question.id[..8.min(question.id.len())],
            question.answer_type
        );

        match answer_question(&milvus, &embeddings, &llm, &reranker, question, comp, config, embedding_type)
            .await
        {
            Ok(submission_answer) => {
                let answer_preview = format!("{}", submission_answer.answer);
                let ttft = submission_answer.telemetry.timing.ttft_ms;
                tracing::info!("  → {} (ttft={}ms)", &answer_preview[..80.min(answer_preview.len())], ttft);
                builder.add_answer(submission_answer);
            }
            Err(e) => {
                tracing::error!("  Failed: {e}");
                // Add a null answer with minimal telemetry to avoid missing questions
                builder.add_answer(SubmissionAnswer {
                    question_id: question.id.clone(),
                    answer: serde_json::Value::Null,
                    telemetry: SubmissionTelemetry {
                        timing: TimingMetrics {
                            ttft_ms: 0,
                            tpot_ms: 0,
                            total_time_ms: 0,
                        },
                        retrieval: RetrievalData {
                            retrieved_chunk_pages: vec![],
                        },
                        usage: UsageMetrics {
                            input_tokens: 0,
                            output_tokens: 0,
                        },
                        model_name: Some(comp.competition_model.clone()),
                    },
                });
            }
        }
    }

    builder.save(&comp.submission_path)?;
    tracing::info!(
        "Submission complete: {} answers saved to {}",
        builder.len(),
        comp.submission_path
    );
    Ok(())
}

/// Submit submission.json to the eval API.
pub async fn submit(comp: &CompetitionConfig) -> anyhow::Result<()> {
    let client = EvalClient::new(&comp.eval_base_url, &comp.eval_api_key);

    // Create a minimal code archive
    let archive_path = "code_archive.zip";
    create_code_archive(archive_path)?;

    tracing::info!("Submitting {} ...", comp.submission_path);
    let result = client.submit(&comp.submission_path, archive_path).await?;
    tracing::info!("Submission response: {}", serde_json::to_string_pretty(&result)?);

    if let Some(uuid) = result["uuid"].as_str() {
        tracing::info!("Submission UUID: {uuid}");
        tracing::info!("Check status: compete status {uuid}");
    }

    Ok(())
}

/// Check submission status.
pub async fn status(comp: &CompetitionConfig, uuid: &str) -> anyhow::Result<()> {
    let client = EvalClient::new(&comp.eval_base_url, &comp.eval_api_key);
    let result = client.get_status(uuid).await?;
    // Use stderr (tracing) since stdout is redirected to /dev/null
    tracing::info!("Status: {}", serde_json::to_string_pretty(&result)?);
    Ok(())
}

/// Full pipeline: ingest → answer.
pub async fn run(config: &Config, comp: &CompetitionConfig) -> anyhow::Result<()> {
    ingest(config, comp).await?;
    answer(config, comp).await?;
    Ok(())
}

// ── Internal helpers ────────────────────────────────────────────────

fn create_milvus(config: &Config) -> MilvusClient {
    MilvusClient::new(
        &config.milvus_url,
        config.embedding_dimension,
        &config.milvus_metric_type,
        &config.milvus_index_type,
        config.milvus_hnsw_m,
        config.milvus_hnsw_ef_construction,
        config.milvus_search_ef,
    )
}

fn create_embeddings(config: &Config) -> EmbeddingClient {
    EmbeddingClient::new(
        &config.embedding_api_url,
        &config.embedding_api_key,
        &config.embedding_model,
        &config.embedding_type,
        &config.embedding_task_document,
        &config.embedding_task_query,
    )
}

fn load_questions(path: &str) -> anyhow::Result<Vec<Question>> {
    let data = std::fs::read_to_string(path)
        .context(format!("Failed to read questions from {path}"))?;
    let questions: Vec<Question> =
        serde_json::from_str(&data).context("Failed to parse questions JSON")?;
    Ok(questions)
}

async fn answer_question(
    milvus: &MilvusClient,
    embeddings: &EmbeddingClient,
    llm: &LlmClient,
    reranker: &RerankerClient,
    question: &Question,
    comp: &CompetitionConfig,
    config: &Config,
    embedding_type: Option<&str>,
) -> anyhow::Result<SubmissionAnswer> {
    // 1. Embed the question
    let query_embs = embeddings
        .embed_with_options(
            &[question.question.clone()],
            InputType::SearchQuery,
            embedding_type,
            None,
        )
        .await
        .context("Failed to embed question")?;

    let query_embedding = query_embs
        .into_iter()
        .next()
        .ok_or_else(|| anyhow::anyhow!("No embedding returned"))?;

    // 2. Search Milvus with high ef for quality
    let search_ef = config.milvus_search_ef.max(256);
    let hits = milvus
        .search(
            &comp.collection_name,
            query_embedding,
            comp.competition_top_k,
            Some(SearchOptions { ef: Some(search_ef) }),
        )
        .await
        .context("Milvus search failed")?;

    // 3. Rerank hits with cross-encoder for better context ordering.
    // IMPORTANT: Use reranked order for LLM context (better answers),
    // but use ORIGINAL Milvus embedding order for grounding pages
    // (embedding similarity better matches gold pages than cross-encoder).
    let context_hits = if reranker.is_configured() && !hits.is_empty() {
        let docs: Vec<String> = hits.iter().map(|h| h.text.clone()).collect();
        match reranker.rerank(&question.question, &docs, hits.len()).await {
            Ok(reranked) => {
                reranked
                    .iter()
                    .filter_map(|r| hits.get(r.original_index).cloned())
                    .collect()
            }
            Err(e) => {
                tracing::warn!("Reranker failed, using original order: {e}");
                hits.clone()
            }
        }
    } else {
        hits.clone()
    };
    // Grounding: use original Milvus top hits (embedding similarity order)
    let grounding_hits: Vec<_> = hits.iter().take(10).cloned().collect();

    // 4. Build context from (reranked) chunks
    let context = context_hits
        .iter()
        .enumerate()
        .map(|(i, h)| {
            let prefix = if h.context_prefix.is_empty() {
                String::new()
            } else {
                format!("{}\n", h.context_prefix)
            };
            format!("--- Excerpt {} ---\n{}{}", i + 1, prefix, h.text)
        })
        .collect::<Vec<_>>()
        .join("\n\n");

    let system_prompt = build_competition_system_prompt(&context, &question.answer_type);

    // 5. Generate answer via streaming (for accurate TTFT)
    let (temperature, max_tokens) = match question.answer_type.as_str() {
        "number" | "boolean" | "date" => (0.0_f32, Some(64_u32)),
        "name" => (0.0, Some(64)),
        "names" => (0.0, Some(256)),
        "free_text" => (0.1, Some(256)),
        _ => (0.0, Some(128)),
    };

    let stream_result = stream_llm_answer(
        llm,
        &system_prompt,
        &question.question,
        &comp.competition_model,
        &comp.competition_provider,
        Some(temperature),
        max_tokens,
    )
    .await?;

    // 6. Strip any SOURCES line the LLM may have added, then parse answer
    let clean_content = strip_sources_line(&stream_result.content);
    let answer = parse_answer(&clean_content, &question.answer_type);

    // 7. Build retrieval refs from original Milvus top hits (not reranked)
    let retrieval_refs = build_retrieval_refs(&grounding_hits);

    // For unanswerable questions, set empty retrieval refs.
    let is_unanswerable = answer.is_null()
        || (question.answer_type == "free_text"
            && answer
                .as_str()
                .map(|s| {
                    s.to_lowercase().contains("no information")
                        || s.to_lowercase().contains("cannot be found")
                        || s.to_lowercase().contains("not available")
                })
                .unwrap_or(false));

    let final_refs = if is_unanswerable {
        vec![]
    } else if retrieval_refs.is_empty() {
        build_retrieval_refs(&hits.iter().take(1).cloned().collect::<Vec<_>>())
    } else {
        retrieval_refs
    };

    Ok(SubmissionAnswer {
        question_id: question.id.clone(),
        answer,
        telemetry: SubmissionTelemetry {
            timing: TimingMetrics {
                ttft_ms: stream_result.ttft_ms,
                tpot_ms: stream_result.tpot_ms,
                total_time_ms: stream_result.total_time_ms,
            },
            retrieval: RetrievalData {
                retrieved_chunk_pages: final_refs,
            },
            usage: UsageMetrics {
                input_tokens: stream_result.input_tokens,
                output_tokens: stream_result.output_tokens,
            },
            model_name: Some(comp.competition_model.clone()),
        },
    })
}

/// Stream LLM completion and track TTFT/TPOT/usage.
async fn stream_llm_answer(
    llm: &LlmClient,
    system_prompt: &str,
    question: &str,
    model: &str,
    provider: &str,
    temperature: Option<f32>,
    max_tokens: Option<u32>,
) -> anyhow::Result<StreamResult> {
    let response = llm
        .chat_stream_with_options(system_prompt, question, model, provider, temperature, max_tokens)
        .await
        .context("Failed to start LLM stream")?;

    consume_sse_stream(response).await
}

/// Parse SSE stream, collect content, track timing metrics.
async fn consume_sse_stream(response: reqwest::Response) -> anyhow::Result<StreamResult> {
    let start = Instant::now();
    let mut first_token_time: Option<std::time::Duration> = None;
    let mut content = String::new();
    let mut token_times: Vec<std::time::Duration> = Vec::new();
    let mut input_tokens: u32 = 0;
    let mut output_tokens: u32 = 0;

    let mut stream = response.bytes_stream();
    let mut buffer = String::new();

    while let Some(chunk) = stream.next().await {
        let bytes = chunk.context("Stream read error")?;
        buffer.push_str(&String::from_utf8_lossy(&bytes));

        while let Some(newline_pos) = buffer.find('\n') {
            let line = buffer[..newline_pos].trim_end_matches('\r').to_string();
            buffer = buffer[newline_pos + 1..].to_string();

            if !line.starts_with("data: ") {
                continue;
            }

            let data = &line[6..];
            if data == "[DONE]" {
                continue;
            }

            if let Ok(json) = serde_json::from_str::<serde_json::Value>(data) {
                // Extract content delta
                if let Some(delta_content) = json["choices"][0]["delta"]["content"].as_str() {
                    if !delta_content.is_empty() {
                        let now = start.elapsed();
                        if first_token_time.is_none() {
                            first_token_time = Some(now);
                        }
                        token_times.push(now);
                        content.push_str(delta_content);
                    }
                }

                // Extract usage from final event
                if let Some(u) = json.get("usage") {
                    input_tokens = u["prompt_tokens"].as_u64().unwrap_or(0) as u32;
                    output_tokens = u["completion_tokens"].as_u64().unwrap_or(0) as u32;
                }
            }
        }
    }

    let total_time = start.elapsed();
    let ttft_ms = first_token_time
        .map(|t| t.as_millis() as u64)
        .unwrap_or(total_time.as_millis() as u64);

    let tpot_ms = if token_times.len() >= 2 {
        let diffs: Vec<u64> = token_times
            .windows(2)
            .map(|w| (w[1] - w[0]).as_millis() as u64)
            .collect();
        diffs.iter().sum::<u64>() / diffs.len() as u64
    } else if output_tokens > 1 {
        // Estimate tpot from generation time / output tokens when streaming is too fast
        let gen_time_ms = total_time.as_millis() as u64 - ttft_ms;
        gen_time_ms / (output_tokens as u64 - 1).max(1)
    } else {
        0
    };

    Ok(StreamResult {
        content,
        ttft_ms,
        tpot_ms,
        total_time_ms: total_time.as_millis() as u64,
        input_tokens,
        output_tokens,
    })
}

/// Generate contextual prefixes for chunks (Anthropic's Contextual Retrieval approach).
async fn generate_context_prefixes(
    pages: &[PageText],
    chunks: &[TextChunk],
    config: &Config,
    llm: &LlmClient,
) -> anyhow::Result<Vec<String>> {
    let concurrency = config.contextual_retrieval_concurrency.max(1);
    let provider = &config.contextual_retrieval_provider;
    let model = &config.contextual_retrieval_model;
    let max_doc_chars = config.contextual_retrieval_max_doc_chars;

    tracing::info!(
        "Generating contextual prefixes for {} chunks (model={model}, concurrency={concurrency})",
        chunks.len(),
    );

    let full_text: String = pages
        .iter()
        .map(|p| p.text.as_str())
        .collect::<Vec<_>>()
        .join("\n\n");

    let results: Vec<String> = futures_util::stream::iter(chunks.iter().enumerate())
        .map(|(idx, chunk)| {
            let doc_window = build_doc_window_text(&full_text, pages, chunk.page_number, max_doc_chars);
            let prompt = build_contextual_prompt(&doc_window, &chunk.text);
            let model = model.to_string();
            let provider = provider.to_string();
            async move {
                let mut backoff_ms = 500u64;
                for attempt in 0..=6 {
                    match llm.chat(&prompt, &model, &provider, Some(128)).await {
                        Ok(result) => return result.content,
                        Err(e) => {
                            let err_str = e.to_string();
                            if err_str.contains("429") && attempt < 6 {
                                tracing::debug!(
                                    "Chunk {idx}: rate limited, retrying in {backoff_ms}ms"
                                );
                                tokio::time::sleep(std::time::Duration::from_millis(backoff_ms))
                                    .await;
                                backoff_ms *= 2;
                                continue;
                            }
                            tracing::warn!("Contextual retrieval failed for chunk {idx}: {e}");
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

fn build_doc_window_text(
    full_text: &str,
    pages: &[PageText],
    chunk_page: Option<u32>,
    max_chars: usize,
) -> String {
    if full_text.len() <= max_chars {
        return full_text.to_string();
    }

    let overview: String = full_text.chars().take(2000).collect();

    let page_num = match chunk_page {
        Some(p) => p,
        None => return full_text.chars().take(max_chars).collect(),
    };

    let lo = page_num.saturating_sub(2);
    let hi = page_num.saturating_add(2);

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

fn create_code_archive(archive_path: &str) -> anyhow::Result<()> {
    use std::io::Write as _;

    let file = std::fs::File::create(archive_path)?;
    let mut zip = zip::ZipWriter::new(file);
    let options = zip::write::SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Deflated);

    // Include key source files
    let root = std::env::current_dir()?;

    fn add_dir_to_zip(
        zip: &mut zip::ZipWriter<std::fs::File>,
        dir: &Path,
        root: &Path,
        options: zip::write::SimpleFileOptions,
    ) -> anyhow::Result<()> {
        if !dir.exists() {
            return Ok(());
        }
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                let dirname = path.strip_prefix(root)?;
                if dirname.to_string_lossy().contains("target") {
                    continue;
                }
                add_dir_to_zip(zip, &path, root, options)?;
            } else if path.is_file() {
                let name = path.strip_prefix(root)?.to_string_lossy().to_string();
                if name.contains("target/") || name.starts_with(".git/") {
                    continue;
                }
                zip.start_file(&name, options)?;
                let content = std::fs::read(&path)?;
                zip.write_all(&content)?;
            }
        }
        Ok(())
    }

    // Add key files
    for filename in ["Cargo.toml", "Cargo.lock", "SETUP.md", ".env.gpu.example"] {
        let path = root.join(filename);
        if path.exists() {
            zip.start_file(filename, options)?;
            let content = std::fs::read(&path)?;
            zip.write_all(&content)?;
        }
    }

    // Add src/
    add_dir_to_zip(&mut zip, &root.join("src"), &root, options)?;

    zip.finish()?;
    tracing::info!("Created code archive: {archive_path}");
    Ok(())
}
