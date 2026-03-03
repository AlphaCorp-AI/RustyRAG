use actix_web::{HttpResponse, ResponseError};

#[derive(Debug, thiserror::Error)]
pub enum AppError {
    #[error("Not found: {0}")]
    NotFound(String),

    #[error("Bad request: {0}")]
    BadRequest(String),

    #[error("LLM error: {0}")]
    LlmError(String),

    #[error("Milvus error: {0}")]
    MilvusError(String),

    #[error("Embedding error: {0}")]
    EmbeddingError(String),

    #[error("Document processing error: {0}")]
    DocumentError(String),

    #[error(transparent)]
    Internal(#[from] anyhow::Error),
}

impl ResponseError for AppError {
    fn error_response(&self) -> HttpResponse {
        let (status, msg) = match self {
            AppError::NotFound(msg) => (actix_web::http::StatusCode::NOT_FOUND, msg.clone()),
            AppError::BadRequest(msg) => (actix_web::http::StatusCode::BAD_REQUEST, msg.clone()),
            AppError::LlmError(msg) => (actix_web::http::StatusCode::BAD_GATEWAY, msg.clone()),
            AppError::MilvusError(msg) => {
                tracing::error!("Milvus error: {msg}");
                (
                    actix_web::http::StatusCode::BAD_GATEWAY,
                    format!("Vector DB error: {msg}"),
                )
            }
            AppError::EmbeddingError(msg) => {
                tracing::error!("Embedding error: {msg}");
                (
                    actix_web::http::StatusCode::BAD_GATEWAY,
                    format!("Embedding error: {msg}"),
                )
            }
            AppError::DocumentError(msg) => (
                actix_web::http::StatusCode::UNPROCESSABLE_ENTITY,
                msg.clone(),
            ),
            AppError::Internal(e) => {
                tracing::error!("Internal error: {e:?}");
                (
                    actix_web::http::StatusCode::INTERNAL_SERVER_ERROR,
                    "Internal server error".into(),
                )
            }
        };

        HttpResponse::build(status).json(serde_json::json!({ "error": msg }))
    }
}
