mod config;
mod db;
mod errors;
mod handlers;
mod middleware;
mod routes;
mod schemas;
mod services;

use actix_web::{web, App, HttpServer};
use tracing_actix_web::TracingLogger;
use utoipa::openapi::security::{HttpAuthScheme, HttpBuilder, SecurityScheme};
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

    // Database
    let pool = db::init_pool(&config.database_url)
        .await
        .expect("Failed to connect to database");

    // Milvus vector DB client (REST API v2)
    let milvus_client = MilvusClient::new(&config.milvus_url, config.embedding_dimension);
    tracing::info!("Milvus client configured → {}", config.milvus_url);

    // Cohere embedding client
    let embedding_client = EmbeddingClient::new(&config.cohere_api_key, &config.embedding_model);
    if embedding_client.is_configured() {
        tracing::info!(
            "Cohere embedding client configured → model={}",
            config.embedding_model,
        );
    } else {
        tracing::warn!(
            "Cohere embedding client NOT configured – set COHERE_API_KEY to enable document upload"
        );
    }

    let pool_data = web::Data::new(pool);
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
                    .app_data(pool_data.clone())
                    .app_data(config_data.clone())
                    .app_data(llm_client.clone())
                    .app_data(milvus_data.clone())
                    .app_data(embedding_data.clone())
            })
            .service(
                scope::scope("/api/v1").configure(routes::configure),
            )
            .openapi_service(|mut api| {
                // Register the Bearer security scheme in Swagger
                let mut components = api.components.unwrap_or_default();
                components.add_security_scheme(
                    "bearer_auth",
                    SecurityScheme::Http(
                        HttpBuilder::new()
                            .scheme(HttpAuthScheme::Bearer)
                            .bearer_format("JWT")
                            .build(),
                    ),
                );
                api.components = Some(components);

                // Tag order & descriptions (Swagger UI renders them in this order)
                let tag = |name: &str, desc: &str| {
                    let mut t = Tag::new(name);
                    t.description = Some(desc.into());
                    t
                };
                api.tags = Some(vec![
                    tag("users", "User registration, login & profile"),
                    tag("chat", "LLM chat completions"),
                    tag("documents", "Document upload, embedding & semantic search"),
                    tag("health", "Health & readiness checks"),
                ]);

                SwaggerUi::new("/swagger-ui/{_:.*}")
                    .url("/api-docs/openapi.json", api)
            })
            .into_app()
            .service(actix_files::Files::new("/static", "static").show_files_listing());
        app
    })
    .bind(format!("{host}:{port}"))?
    .run()
    .await
}
