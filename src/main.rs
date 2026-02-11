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
use utoipa_actix_web::{scope, AppExt};
use utoipa_swagger_ui::SwaggerUi;

use crate::config::Config;
use crate::services::llm::LlmClient;

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    tracing_subscriber::fmt::init();

    let config = Config::from_env().expect("Failed to load config");

    // Database
    let pool = db::init_pool(&config.database_url)
        .await
        .expect("Failed to connect to database");

    let pool_data = web::Data::new(pool);
    let config_data = web::Data::new(config.clone());
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
                    .app_data(pool_data.clone())
                    .app_data(config_data.clone())
                    .app_data(llm_client.clone())
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
