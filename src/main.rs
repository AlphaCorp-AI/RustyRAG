mod config;
mod errors;
mod handlers;
mod prompts;
mod routes;
mod schemas;
mod services;

use actix_web::{web, App, HttpServer};
use tracing_actix_web::TracingLogger;
use utoipa::openapi::tag::Tag;
use utoipa_actix_web::{scope, AppExt};
use utoipa_swagger_ui::SwaggerUi;

use crate::config::Config;
use crate::services::embeddings::EmbeddingClient;
use crate::services::llm::LlmClient;
use crate::services::milvus::MilvusClient;

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    // Force tracing output to stderr so it survives the stdout redirect below.
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .init();

    // Redirect stdout → /dev/null once at startup.
    // pdf-extract uses println! for noisy debug output; tracing now goes to
    // stderr, so only the pdf-extract noise is silenced.
    {
        use std::os::unix::io::AsRawFd;
        if let Ok(devnull) = std::fs::File::open("/dev/null") {
            unsafe {
                libc::dup2(devnull.as_raw_fd(), std::io::stdout().as_raw_fd());
            }
        }
    }

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

    let config_data = web::Data::new(config.clone());
    let llm_client = web::Data::new(LlmClient::new(config.clone()));
    let milvus_data = web::Data::new(milvus_client);
    let embedding_data = web::Data::new(embedding_client);

    let host = config.host.clone();
    let port = config.port;

    tracing::info!("Starting server at http://{host}:{port}");
    tracing::info!("Swagger UI at http://{host}:{port}/swagger-ui/");

    HttpServer::new(move || {
        let app = App::new()
            .into_utoipa_app()
            .map(|app| {
                app.wrap(TracingLogger::default())
                    .wrap(actix_cors::Cors::permissive())
                    // Allow uploads up to 2 GB (default is 256 KB)
                    .app_data(web::PayloadConfig::new(2 * 1024 * 1024 * 1024))
                    .app_data(config_data.clone())
                    .app_data(llm_client.clone())
                    .app_data(milvus_data.clone())
                    .app_data(embedding_data.clone())
            })
            .service(scope::scope("/api/v1").configure(routes::configure))
            .openapi_service(|mut api| {
                // Tag order & descriptions (Swagger UI renders them in this order)
                let tag = |name: &str, desc: &str| {
                    let mut t = Tag::new(name);
                    t.description = Some(desc.into());
                    t
                };
                api.tags = Some(vec![
                    tag("chat", "LLM chat completions"),
                    tag("documents", "Document upload, embedding & semantic search"),
                    tag("health", "Health & readiness checks"),
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
