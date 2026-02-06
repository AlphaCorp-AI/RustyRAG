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
use utoipa_actix_web::{scope, AppExt};
use utoipa_swagger_ui::SwaggerUi;

use crate::config::Config;
use crate::services::llm::LlmClient;

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    tracing_subscriber::fmt::init();

    let config = Config::from_env().expect("Failed to load config");

    let llm_client = web::Data::new(LlmClient::new(config.clone()));

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
                    .app_data(llm_client.clone())
            })
            .service(
                scope::scope("/api/v1").configure(routes::configure),
            )
            .openapi_service(|api| {
                SwaggerUi::new("/swagger-ui/{_:.*}")
                    .url("/api-docs/openapi.json", api)
            })
            .into_app();
        app
    })
    .bind(format!("{host}:{port}"))?
    .run()
    .await
}
