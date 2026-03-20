use actix_web::{HttpResponse, ResponseError};

#[derive(Debug, thiserror::Error)]
pub enum AppError {
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
            AppError::BadRequest(msg) => (actix_web::http::StatusCode::BAD_REQUEST, msg.clone()),
            AppError::LlmError(msg) => {
                tracing::error!("LLM error: {msg}");
                (
                    actix_web::http::StatusCode::BAD_GATEWAY,
                    "LLM provider error".into(),
                )
            }
            AppError::MilvusError(msg) => {
                tracing::error!("Milvus error: {msg}");
                (
                    actix_web::http::StatusCode::BAD_GATEWAY,
                    "Vector database error".into(),
                )
            }
            AppError::EmbeddingError(msg) => {
                tracing::error!("Embedding error: {msg}");
                (
                    actix_web::http::StatusCode::BAD_GATEWAY,
                    "Embedding service error".into(),
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
