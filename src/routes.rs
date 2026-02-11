use actix_web_httpauth::middleware::HttpAuthentication;
use utoipa_actix_web::scope;
use utoipa_actix_web::service_config::ServiceConfig;

use crate::handlers;
use crate::middleware::auth;

pub fn configure(cfg: &mut ServiceConfig) {
    // Public routes
    cfg.service(handlers::health::health)
        .service(handlers::chat::chat)
        .service(handlers::users::register)
        .service(handlers::users::login);

    // Protected routes (require Bearer JWT)
    let bearer_middleware = HttpAuthentication::bearer(auth::validator);
    cfg.service(
        scope::scope("")
            .wrap(bearer_middleware)
            .service(handlers::users::get_me),
    );
}
