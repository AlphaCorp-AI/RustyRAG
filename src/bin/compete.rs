use rustyrag::competition::runner::{self, CompetitionConfig};
use rustyrag::config::Config;

fn print_usage() {
    eprintln!(
        "Usage: compete <command>\n\n\
         Commands:\n\
         \x20 download   Download questions and documents from the eval API\n\
         \x20 ingest     Ingest all PDF documents into Milvus\n\
         \x20 answer     Answer all questions and generate submission.json\n\
         \x20 run        Full pipeline: ingest + answer\n\
         \x20 submit     Submit submission.json to the eval API\n\
         \x20 status <uuid>  Check submission status\n\n\
         Environment:\n\
         \x20 All settings are loaded from .env (see .env.gpu.example)\n\
         \x20 Competition-specific:\n\
         \x20   EVAL_API_KEY          Evaluation API key\n\
         \x20   EVAL_BASE_URL         API base URL (default: production)\n\
         \x20   DOCS_DIR              Path to PDF corpus (default: docs_corpus)\n\
         \x20   QUESTIONS_PATH        Path to questions JSON (default: questions.json)\n\
         \x20   SUBMISSION_PATH       Output path (default: submission.json)\n\
         \x20   COLLECTION_NAME       Milvus collection (default: competition)\n\
         \x20   COMPETITION_MODEL     LLM model (default: qwen-3-235b-a22b-instruct-2507)\n\
         \x20   COMPETITION_PROVIDER  LLM provider (default: cerebras)\n\
         \x20   COMPETITION_TOP_K     Chunks to retrieve per question (default: 15)"
    );
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Init tracing
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .init();

    // Suppress pdf-extract stdout noise
    {
        use std::os::unix::io::AsRawFd;
        if let Ok(devnull) = std::fs::File::open("/dev/null") {
            unsafe {
                libc::dup2(devnull.as_raw_fd(), std::io::stdout().as_raw_fd());
            }
        }
    }

    dotenvy::dotenv().ok();

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        print_usage();
        std::process::exit(1);
    }

    let command = args[1].as_str();

    // Load configs
    let config = Config::from_env().map_err(|e| anyhow::anyhow!("Config error: {e}"))?;
    let comp = CompetitionConfig::from_env().map_err(|e| anyhow::anyhow!("Competition config error: {e}"))?;

    match command {
        "download" => {
            if comp.eval_api_key.is_empty() {
                anyhow::bail!("EVAL_API_KEY is required for download");
            }
            runner::download(&comp).await?;
        }

        "ingest" => {
            runner::ingest(&config, &comp).await?;
        }

        "answer" => {
            runner::answer(&config, &comp).await?;
        }

        "run" => {
            runner::run(&config, &comp).await?;
        }

        "submit" => {
            if comp.eval_api_key.is_empty() {
                anyhow::bail!("EVAL_API_KEY is required for submit");
            }
            runner::submit(&comp).await?;
        }

        "status" => {
            let uuid = args
                .get(2)
                .ok_or_else(|| anyhow::anyhow!("Usage: compete status <uuid>"))?;
            if comp.eval_api_key.is_empty() {
                anyhow::bail!("EVAL_API_KEY is required for status");
            }
            runner::status(&comp, uuid).await?;
        }

        _ => {
            eprintln!("Unknown command: {command}");
            print_usage();
            std::process::exit(1);
        }
    }

    Ok(())
}
