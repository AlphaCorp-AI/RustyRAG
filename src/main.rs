mod config;
mod errors;
mod handlers;
mod prompts;
mod routes;
mod schemas;
mod services;

use actix_web::{web, App, HttpServer};
use tracing_actix_web::TracingLogger;
use utoipa::openapi::info::{Contact, Info};
use utoipa::openapi::tag::Tag;
use utoipa_actix_web::{scope, AppExt};
use utoipa_swagger_ui::SwaggerUi;

use crate::config::Config;
use crate::services::embeddings::EmbeddingClient;
use crate::services::llm::LlmClient;
use crate::services::docling::DoclingClient;
use crate::services::milvus::MilvusClient;
use crate::services::reranker::RerankerClient;

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    // Tracing goes to stderr; pdf-extract's println! noise goes to stdout
    // and is harmless (stdout is not used for application output).
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .init();

    let config = Config::from_env().expect("Failed to load config");

    // Milvus vector DB client (REST API v2)
    let milvus_client = MilvusClient::new(
        &config.milvus_url,
        config.embedding_dimension,
        &config.milvus_metric_type,
        &config.milvus_index_type,
        config.milvus_hnsw_m,
        config.milvus_hnsw_ef_construction,
        config.milvus_search_ef,
    );
    tracing::info!("Milvus client configured → {}", config.milvus_url);

    // Local/OpenAI-compatible embedding client
    let embedding_client = EmbeddingClient::new(
        &config.embedding_api_url,
        &config.embedding_api_key,
        &config.embedding_model,
        &config.embedding_type,
        &config.embedding_task_document,
        &config.embedding_task_query,
    );
    if embedding_client.is_configured() {
        tracing::info!(
            "Embedding client configured → url={}, model={}",
            config.embedding_api_url,
            config.embedding_model,
        );
    } else {
        tracing::warn!(
            "Embedding client NOT configured – set EMBEDDING_API_URL and EMBEDDING_MODEL"
        );
    }

    // Reranker client (optional — set RERANKER_API_URL to enable)
    let reranker_client = RerankerClient::new(&config.reranker_api_url);
    if reranker_client.is_configured() {
        tracing::info!(
            "Reranker configured → {} (retrieve {} → rerank top {})",
            config.reranker_api_url,
            config.retrieval_limit,
            config.rerank_top_n,
        );
    } else {
        tracing::warn!("Reranker NOT configured – set RERANKER_API_URL to enable reranking");
    }

    // Docling document extraction service
    let docling_client = DoclingClient::new(
        &config.docling_url,
        &config.groq_api_key,
        &config.vision_model,
    );
    if docling_client.is_configured() {
        tracing::info!(
            "Docling configured → {} (vision: {})",
            config.docling_url,
            config.vision_model,
        );
    } else {
        tracing::warn!("Docling NOT configured – PDF/DOCX uploads will fail");
    }

    let config_data = web::Data::new(config.clone());
    let llm_client = web::Data::new(LlmClient::new(config.clone()));
    let milvus_data = web::Data::new(milvus_client);
    let embedding_data = web::Data::new(embedding_client);
    let reranker_data = web::Data::new(reranker_client);
    let docling_data = web::Data::new(docling_client);

    let host = config.host.clone();
    let port = config.port;

    tracing::info!("Starting server at http://{host}:{port}");
    tracing::info!("Swagger UI at http://{host}:{port}/swagger-ui/");

    HttpServer::new(move || {
        let app = App::new()
            .into_utoipa_app()
            .map(|app| {
                let cors = build_cors(&config.cors_allowed_origins);

                app.wrap(TracingLogger::default())
                    .wrap(cors)
                    .app_data(web::PayloadConfig::new(2 * 1024 * 1024 * 1024))
                    .app_data(web::JsonConfig::default().limit(1_048_576))
                    .app_data(config_data.clone())
                    .app_data(llm_client.clone())
                    .app_data(milvus_data.clone())
                    .app_data(embedding_data.clone())
                    .app_data(reranker_data.clone())
                    .app_data(docling_data.clone())
            })
            .service(scope::scope("/api/v1").configure(routes::configure))
            .openapi_service(|mut api| {
                api.info = Info::builder()
                    .title("RustyRAG API")
                    .version("0.3.0")
                    .description(Some(
                        "Production-grade RAG in a single Rust binary. \
                         Hybrid search (dense + BM25), cross-encoder reranking, \
                         Docling document extraction, and vision model support."
                    ))
                    .contact(Some(
                        Contact::builder()
                            .name(Some("Ignas Vaitukaitis"))
                            .email(Some("ignas@alphacorp.ai"))
                            .url(Some("https://github.com/AlphaCorp-AI/RustyRAG"))
                            .build(),
                    ))
                    .build();

                let tag = |name: &str, desc: &str| {
                    let mut t = Tag::new(name);
                    t.description = Some(desc.into());
                    t
                };
                api.tags = Some(vec![
                    tag("Chat", "RAG chat completions with hybrid search and cross-encoder reranking"),
                    tag("Documents", "Document upload (PDF, DOCX, PPTX, XLSX, HTML, TXT), embedding & semantic search"),
                    tag("Health", "Health & readiness checks"),
                ]);

                SwaggerUi::new("/swagger-ui/{_:.*}").url("/api-docs/openapi.json", api)
            })
            .into_app()
            .service(actix_files::Files::new("/static", "static").show_files_listing());
        app
    })
    .bind(format!("{host}:{port}"))?
    .run()
    .await
}

/// Build CORS middleware. Permissive when `origins` is empty (dev mode),
/// explicit allowlist otherwise.
fn build_cors(origins: &str) -> actix_cors::Cors {
    if origins.is_empty() {
        return actix_cors::Cors::permissive();
    }

    let mut cors = actix_cors::Cors::default()
        .allowed_methods(vec!["GET", "POST", "OPTIONS"])
        .allowed_headers(vec![
            actix_web::http::header::CONTENT_TYPE,
            actix_web::http::header::AUTHORIZATION,
        ])
        .max_age(3600);

    for origin in origins.split(',') {
        let origin = origin.trim();
        if !origin.is_empty() {
            cors = cors.allowed_origin(origin);
        }
    }

    cors
}
