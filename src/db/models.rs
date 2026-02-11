use chrono::{DateTime, Utc};
use serde::Serialize;
use sqlx::FromRow;
use uuid::Uuid;

#[derive(Debug, FromRow, Serialize, Clone)]
pub struct User {
    pub id: Uuid,
    pub name: String,
    pub email: String,
    #[serde(skip_serializing)]
    pub password: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}
