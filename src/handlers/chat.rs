use actix_web::{post, web, HttpResponse};
use validator::Validate;

use crate::errors::AppError;
use crate::schemas::requests::ChatRequest;
use crate::schemas::responses::{ChatResponse, ErrorResponse, Usage};
use crate::services::llm::LlmClient;

/// Send a message to the LLM via OpenRouter
#[utoipa::path(
    post,
    path = "/chat",
    request_body = ChatRequest,
    responses(
        (status = 200, description = "Chat completion", body = ChatResponse),
        (status = 400, description = "Validation error", body = ErrorResponse),
        (status = 502, description = "LLM provider error", body = ErrorResponse),
    ),
    tag = "chat"
)]
#[post("/chat")]
pub async fn chat(
    llm: web::Data<LlmClient>,
    body: web::Json<ChatRequest>,
) -> Result<HttpResponse, AppError> {
    body.validate()
        .map_err(|e| AppError::BadRequest(e.to_string()))?;

    let result = llm
        .chat(
            &body.message,
            body.model.as_deref(),
            body.max_tokens,
        )
        .await
        .map_err(|e| AppError::LlmError(e.to_string()))?;

    let usage = result.usage.map(|u| Usage {
        prompt_tokens: u.prompt_tokens,
        completion_tokens: u.completion_tokens,
        total_tokens: u.total_tokens,
    });

    Ok(HttpResponse::Ok().json(ChatResponse {
        model: result.model,
        message: result.content,
        usage,
    }))
}
