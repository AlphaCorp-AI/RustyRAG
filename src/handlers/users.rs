use actix_web::{get, post, web, HttpMessage, HttpRequest, HttpResponse};
use sqlx::PgPool;
use validator::Validate;

use crate::config::Config;
use crate::db::repositories::users as user_repo;
use crate::errors::AppError;
use crate::middleware::auth::{create_token, Claims};
use crate::schemas::requests::{CreateUserRequest, LoginRequest};
use crate::schemas::responses::{ErrorResponse, TokenResponse, UserResponse};
use crate::services::password;

// ── Register ────────────────────────────────────────────────────────

/// Create a new user account
#[utoipa::path(
    post,
    path = "/users/register",
    request_body = CreateUserRequest,
    responses(
        (status = 201, description = "User created", body = UserResponse),
        (status = 400, description = "Validation error", body = ErrorResponse),
        (status = 409, description = "Email already taken", body = ErrorResponse),
    ),
    tag = "users"
)]
#[post("/users/register")]
pub async fn register(
    pool: web::Data<PgPool>,
    body: web::Json<CreateUserRequest>,
) -> Result<HttpResponse, AppError> {
    body.validate()
        .map_err(|e| AppError::BadRequest(e.to_string()))?;

    // Check for existing email
    if user_repo::find_by_email(&pool, &body.email).await?.is_some() {
        return Err(AppError::Conflict("Email already registered".into()));
    }

    let password_hash = password::hash(&body.password)?;

    let user = user_repo::create_user(&pool, &body.name, &body.email, &password_hash).await?;

    Ok(HttpResponse::Created().json(UserResponse {
        id: user.id,
        name: user.name,
        email: user.email,
        created_at: user.created_at,
        updated_at: user.updated_at,
    }))
}

// ── Login ───────────────────────────────────────────────────────────

/// Login and receive a JWT token
#[utoipa::path(
    post,
    path = "/users/login",
    request_body = LoginRequest,
    responses(
        (status = 200, description = "Login successful", body = TokenResponse),
        (status = 401, description = "Invalid credentials", body = ErrorResponse),
    ),
    tag = "users"
)]
#[post("/users/login")]
pub async fn login(
    pool: web::Data<PgPool>,
    config: web::Data<Config>,
    body: web::Json<LoginRequest>,
) -> Result<HttpResponse, AppError> {
    body.validate()
        .map_err(|e| AppError::BadRequest(e.to_string()))?;

    let user = user_repo::find_by_email(&pool, &body.email)
        .await?
        .ok_or_else(|| AppError::Unauthorized("Invalid email or password".into()))?;

    let valid = password::verify(&body.password, &user.password)?;
    if !valid {
        return Err(AppError::Unauthorized("Invalid email or password".into()));
    }

    let token = create_token(user.id, &config.jwt_secret)?;

    Ok(HttpResponse::Ok().json(TokenResponse {
        access_token: token,
        token_type: "Bearer".into(),
    }))
}

// ── Get Me (protected) ──────────────────────────────────────────────

/// Get the currently authenticated user
#[utoipa::path(
    get,
    path = "/users/me",
    responses(
        (status = 200, description = "Current user", body = UserResponse),
        (status = 401, description = "Not authenticated", body = ErrorResponse),
    ),
    security(
        ("bearer_auth" = [])
    ),
    tag = "users"
)]
#[get("/users/me")]
pub async fn get_me(
    pool: web::Data<PgPool>,
    req: HttpRequest,
) -> Result<HttpResponse, AppError> {
    let claims = req
        .extensions()
        .get::<Claims>()
        .cloned()
        .ok_or_else(|| AppError::Unauthorized("Missing token".into()))?;

    let user = user_repo::find_by_id(&pool, claims.sub)
        .await?
        .ok_or_else(|| AppError::NotFound("User not found".into()))?;

    Ok(HttpResponse::Ok().json(UserResponse {
        id: user.id,
        name: user.name,
        email: user.email,
        created_at: user.created_at,
        updated_at: user.updated_at,
    }))
}
