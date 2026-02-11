pub mod models;
pub mod repositories;

use sqlx::postgres::PgPoolOptions;
use sqlx::PgPool;

/// Create a connection pool and run migrations.
pub async fn init_pool(database_url: &str) -> anyhow::Result<PgPool> {
    let pool = PgPoolOptions::new()
        .max_connections(10)
        .connect(database_url)
        .await?;

    sqlx::migrate!("./migrations").run(&pool).await?;

    tracing::info!("Database connected & migrations applied");

    Ok(pool)
}
