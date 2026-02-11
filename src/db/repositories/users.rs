use sqlx::PgPool;
use uuid::Uuid;

use crate::db::models::User;

/// Insert a new user and return the full row.
pub async fn create_user(
    pool: &PgPool,
    name: &str,
    email: &str,
    password_hash: &str,
) -> sqlx::Result<User> {
    sqlx::query_as::<_, User>(
        r#"
        INSERT INTO users (name, email, password)
        VALUES ($1, $2, $3)
        RETURNING *
        "#,
    )
    .bind(name)
    .bind(email)
    .bind(password_hash)
    .fetch_one(pool)
    .await
}

/// Find a user by email (for login).
pub async fn find_by_email(pool: &PgPool, email: &str) -> sqlx::Result<Option<User>> {
    sqlx::query_as::<_, User>("SELECT * FROM users WHERE email = $1")
        .bind(email)
        .fetch_optional(pool)
        .await
}

/// Find a user by ID (for get_me / token lookups).
pub async fn find_by_id(pool: &PgPool, id: Uuid) -> sqlx::Result<Option<User>> {
    sqlx::query_as::<_, User>("SELECT * FROM users WHERE id = $1")
        .bind(id)
        .fetch_optional(pool)
        .await
}
