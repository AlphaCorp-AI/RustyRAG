use serde::Deserialize;

#[derive(Debug, Deserialize, Clone)]
pub struct Config {
    pub database_url: String,
    pub jwt_secret: String,
    pub openrouter_api_key: String,
    pub host: String,
    pub port: u16,
}

impl Config {
    pub fn from_env() -> Result<Self, envy::Error> {
        dotenvy::dotenv().ok();
        envy::from_env::<Config>()
    }
}
