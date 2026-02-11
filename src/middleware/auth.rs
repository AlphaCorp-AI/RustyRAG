use actix_web::{dev::ServiceRequest, Error, HttpMessage};
use actix_web_httpauth::extractors::bearer::BearerAuth;
use jsonwebtoken::{decode, DecodingKey, Validation};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::errors::AppError;

// ── JWT Claims ──────────────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Claims {
    /// Subject – the user's UUID
    pub sub: Uuid,
    /// Expiration (epoch seconds)
    pub exp: usize,
}

// ── Token helpers ───────────────────────────────────────────────────

pub fn create_token(user_id: Uuid, secret: &str) -> Result<String, AppError> {
    let exp = chrono::Utc::now()
        .checked_add_signed(chrono::Duration::hours(24))
        .expect("valid timestamp")
        .timestamp() as usize;

    let claims = Claims { sub: user_id, exp };

    jsonwebtoken::encode(
        &jsonwebtoken::Header::default(),
        &claims,
        &jsonwebtoken::EncodingKey::from_secret(secret.as_bytes()),
    )
    .map_err(|e| AppError::Internal(e.into()))
}

// ── Actix-web-httpauth validator ────────────────────────────────────

/// Called by `HttpAuthentication::bearer(validator)` on every protected
/// request. Decodes the JWT and inserts `Claims` into request extensions
/// so handlers can extract the current user id.
pub async fn validator(
    req: ServiceRequest,
    credentials: BearerAuth,
) -> Result<ServiceRequest, (Error, ServiceRequest)> {
    let secret = req
        .app_data::<actix_web::web::Data<crate::config::Config>>()
        .map(|c| c.jwt_secret.clone())
        .unwrap_or_default();

    let token_result = decode::<Claims>(
        credentials.token(),
        &DecodingKey::from_secret(secret.as_bytes()),
        &Validation::default(),
    );

    match token_result {
        Ok(token_data) => {
            req.extensions_mut().insert(token_data.claims);
            Ok(req)
        }
        Err(_) => {
            let err = AppError::Unauthorized("Invalid or expired token".into());
            Err((err.into(), req))
        }
    }
}
