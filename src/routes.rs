use utoipa_actix_web::service_config::ServiceConfig;

use crate::handlers;

pub fn configure(cfg: &mut ServiceConfig) {
    cfg.service(handlers::health::health)
        .service(handlers::chat::chat);
}
