use utoipa_actix_web::service_config::ServiceConfig;

use crate::handlers;

pub fn configure(cfg: &mut ServiceConfig) {
    // Public routes
    cfg.service(handlers::health::health)
        .service(handlers::chat::list_llms)
        .service(handlers::chat::chat_rag)
        .service(handlers::chat::chat_rag_stream)
        .service(handlers::documents::upload_document)
        .service(handlers::documents::search_documents)
        .service(handlers::documents::backup_collection)
        .service(handlers::evals::run_evals);
}
