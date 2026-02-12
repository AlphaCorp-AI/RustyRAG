<div align="center">

# ⚡ AlphaRustyRAG

**Sub-second RAG retrieval and generation, built in Rust.**

A high-performance Retrieval-Augmented Generation API that chunks, embeds, stores, and queries documents — then streams LLM answers grounded in your data.

Built by **[Ignas Vaitukaitis](https://www.linkedin.com/in/ignas-vaitukaitis/)** · CEO @ **[AlphaCorp AI](https://alphacorp.ai)**

<br/>

<img src="https://img.shields.io/badge/Rust-000000?style=for-the-badge&logo=rust&logoColor=white" alt="Rust"/>
<img src="https://img.shields.io/badge/Actix--web-000000?style=for-the-badge&logo=rust&logoColor=white" alt="Actix-web"/>
<img src="https://img.shields.io/badge/OpenRouter-6366F1?style=for-the-badge&logo=openai&logoColor=white" alt="OpenRouter"/>
<img src="https://img.shields.io/badge/Milvus-00A1EA?style=for-the-badge&logo=apachekafka&logoColor=white" alt="Milvus"/>
<img src="https://img.shields.io/badge/PostgreSQL-4169E1?style=for-the-badge&logo=postgresql&logoColor=white" alt="PostgreSQL"/>
<img src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker"/>
<img src="https://img.shields.io/badge/Swagger_UI-85EA2D?style=for-the-badge&logo=swagger&logoColor=black" alt="Swagger"/>
<img src="https://img.shields.io/badge/Groq-F55036?style=for-the-badge&logo=groq&logoColor=white" alt="Groq"/>
<img src="https://img.shields.io/badge/Tokio-463E52?style=for-the-badge&logo=rust&logoColor=white" alt="Tokio"/>

</div>

---

## Why AlphaRust?

Most RAG stacks glue together Python microservices with high per-request overhead. AlphaRust collapses the entire pipeline — document ingestion, vector search, and LLM streaming — into a **single async Rust binary**. The result: **sub-1-second time-to-first-token** on retrieval-augmented queries, even with large document collections.

### Key Features

- **Full RAG pipeline in one binary** — upload, chunk, embed, store, search, and generate
- **Only one API key needed** — [OpenRouter](https://openrouter.ai) handles both LLM completions *and* embeddings out of the box
- **Real-time SSE streaming** — tokens stream to the client as they're generated, with sources delivered as a leading SSE event
- **Default model: `openai/gpt-oss-safeguard-20b`** — routed through **[Groq](https://groq.com)** via OpenRouter for maximum inference speed
- **Concurrent document ingestion** — ZIP archives are unpacked and processed across 8 parallel workers with batched embedding calls
- **Pluggable embedding provider** — OpenAI, Ollama, vLLM, LiteLLM, HuggingFace TEI, or just use OpenRouter
- **Milvus HNSW vector search** — cosine similarity with tunable `ef` and `M` parameters
- **JWT auth + Argon2id password hashing** — production-ready user management
- **Interactive Swagger UI** — every endpoint documented with OpenAPI 3.0
- **Built-in chat frontend** — a minimal SSE-powered UI at `/static/chat.html` for testing RAG and plain chat

---

## Architecture

```
┌──────────────┐       ┌──────────────────────────────────────────────────┐
│              │  SSE  │                   AlphaRust                      │
│  Client /    │◄─────►│                                                  │
│  Frontend    │       │  ┌──────────┐  ┌────────────┐  ┌──────────────┐  │
│              │       │  │ Actix-web│  │ Embedding  │  │  OpenRouter  │  │
└──────────────┘       │  │  Router  │─►│  Client    │  │  LLM Client  │  │
                       │  └──────────┘  └──────┬─────┘  └──────┬───────┘  │
                       │                       │               │          │
                       │       ┌───────────────▼───────────────┘          │
                       │       │                                          │
                       │  ┌────▼────────┐  ┌──────────────┐               │
                       │  │   Milvus    │  │  PostgreSQL  │               │
                       │  │ HNSW Index  │  │  Users/Auth  │               │
                       │  └─────────────┘  └──────────────┘               │
                       └──────────────────────────────────────────────────┘
                                          │
                              ┌────────────────────────┐
                              │    OpenRouter API      │
                              │  LLM + Embeddings      │
                              │  (one key, 200+ models)│
                              └────────────────────────┘
```

---

## Quick Start

### Prerequisites

- **Rust 1.70+** — install via [rustup](https://rustup.rs/)
- **Docker & Docker Compose** — for PostgreSQL and Milvus
- **OpenRouter API key** — get one at [openrouter.ai](https://openrouter.ai/) ← **this is all you need**

### 1. Clone and configure

```bash
git clone https://github.com/your-org/alpharust.git
cd alpharust
cp .env.example .env
```

Edit `.env` — the only required external credential is your OpenRouter key:

```env
# ── Required ─────────────────────────────────────────────
OPENROUTER_API_KEY=sk-or-your-key-here

# ── Infrastructure (defaults work with docker-compose) ───
DATABASE_URL=postgres://alpharust:alpharust@localhost:5432/alpharust
JWT_SECRET=change-me-to-a-long-random-string
MILVUS_URL=http://localhost:19530

# ── Embedding (optional — falls back to OpenRouter) ──────
# Uncomment only if you want a dedicated provider like OpenAI:
# EMBEDDING_MODEL=text-embedding-3-small
# EMBEDDING_API_URL=https://api.openai.com/v1/embeddings
# EMBEDDING_API_KEY=sk-...
# EMBEDDING_DIMENSION=1536
```

> **How it works:** When `EMBEDDING_API_URL` and `EMBEDDING_API_KEY` are left empty, AlphaRust automatically routes embedding requests through `https://openrouter.ai/api/v1/embeddings` using your same `OPENROUTER_API_KEY`. Zero extra setup — one key powers the entire stack.

### 2. Start infrastructure

```bash
docker compose up -d
```

This spins up **PostgreSQL 16** and **Milvus 2.4** (standalone mode with embedded etcd).

### 3. Build and run

```bash
cargo build --release
cargo run --release
```

The server starts at `http://127.0.0.1:8080`. Database migrations run automatically on first boot.

### 4. Try it out

- **Chat UI** → [http://localhost:8080/static/chat.html](http://localhost:8080/static/chat.html)
- **Swagger UI** → [http://localhost:8080/swagger-ui/](http://localhost:8080/swagger-ui/)

Upload a document, type a question with a collection name, and watch tokens stream back in under a second.

---

## Default LLM: gpt-oss-safeguard-20b via Groq

AlphaRust ships with **`openai/gpt-oss-safeguard-20b`** as the default model, routed through **[Groq](https://groq.com)** on OpenRouter via provider preferences. This gives you:

- **Extremely low latency** — Groq's LPU inference hardware delivers tokens faster than GPU-based providers
- **Free or low-cost** — available on OpenRouter's free tier
- **No config needed** — works immediately with just your OpenRouter key

You can override the model per request by passing `"model": "anthropic/claude-sonnet-4"` (or any other [OpenRouter model](https://openrouter.ai/models)) in the `/chat` request body.

---

## API Reference

All endpoints live under `/api/v1`.

### Documents

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/documents/upload` | Upload `.txt`, `.pdf`, or `.zip` — chunks, embeds, stores in Milvus |
| `POST` | `/documents/search` | Semantic search across embedded documents |

### Chat

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/chat` | Single-turn LLM completion |
| `POST` | `/chat/stream` | SSE-streamed LLM completion |
| `POST` | `/chat-rag` | RAG: retrieve context → generate answer |
| `POST` | `/chat-rag/stream` | SSE-streamed RAG (sources event + LLM tokens) |

### Users

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/users/register` | Create account (Argon2id-hashed passwords) |
| `POST` | `/users/login` | Get a JWT bearer token |
| `GET`  | `/users/me` | Current user profile (🔒 requires Bearer token) |

### Health

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/health` | Liveness check |

> For full request/response schemas with examples, see the interactive **[Swagger UI](http://localhost:8080/swagger-ui/)**.

---

## RAG Pipeline

### Upload flow

```
File upload (up to 2 GB, streamed to disk — never held in memory)
  → Text extraction (PDF via pdf-extract / TXT via UTF-8)
  → ZIP? Unpack → process entries concurrently (8 workers)
  → Word-level chunking (configurable size + overlap)
  → Batch embedding (100 chunks per API call)
  → Batch insert into Milvus (50 chunks per insert)
```

### Query flow

```
User question
  → Embed the query
  → Milvus HNSW search (top-K, cosine similarity)
  → Inject retrieved chunks as system prompt context
  → Stream LLM answer via SSE
  → Sources emitted as leading "event: sources" SSE event
```

---

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENROUTER_API_KEY` | **Only required API key** — powers LLM + embeddings | *required* |
| `DATABASE_URL` | PostgreSQL connection string | *required (local)* |
| `JWT_SECRET` | Secret for signing JWT tokens | *required (local)* |
| `MILVUS_URL` | Milvus REST API endpoint | `http://localhost:19530` |
| `EMBEDDING_MODEL` | Model name (e.g. `text-embedding-3-small`) | — |
| `EMBEDDING_API_URL` | Dedicated embeddings endpoint | Falls back to OpenRouter |
| `EMBEDDING_API_KEY` | Dedicated embeddings key | Falls back to `OPENROUTER_API_KEY` |
| `EMBEDDING_DIMENSION` | Vector dimensionality (must match model) | `3072` |
| `CHUNK_SIZE` | Default words per chunk | `500` |
| `CHUNK_OVERLAP` | Overlap words between consecutive chunks | `50` |
| `HOST` | Server bind address | `127.0.0.1` |
| `PORT` | Server port | `8080` |
| `RUST_LOG` | Log level | `info` |

---

## Project Structure

```
src/
├── main.rs                 # Entry point, server bootstrap, pdf-extract stdout silencing
├── config.rs               # Env-based config via serde + envy
├── routes.rs               # Route registration (public + JWT-protected)
├── errors.rs               # Unified AppError → HTTP response mapping
├── handlers/
│   ├── chat.rs             # /chat, /chat/stream, /chat-rag, /chat-rag/stream
│   ├── documents.rs        # /documents/upload, /documents/search
│   ├── users.rs            # /users/register, /users/login, /users/me
│   └── health.rs           # /health
├── schemas/
│   ├── requests.rs         # Validated request DTOs (serde + validator)
│   └── responses.rs        # Response DTOs with utoipa OpenAPI schemas
├── services/
│   ├── llm.rs              # OpenRouter client (sync + SSE streaming, Groq routing)
│   ├── embeddings.rs       # OpenAI-compatible embedding client
│   ├── milvus.rs           # Milvus v2 REST client (collections, insert, search)
│   ├── document.rs         # Text extraction, ZIP unpacking, word-level chunking
│   └── password.rs         # Argon2id hashing & verification
├── middleware/
│   └── auth.rs             # JWT decode + Claims injection
└── db/
    ├── models.rs           # SQLx row types
    └── repositories/
        └── users.rs        # User CRUD queries

migrations/                 # Auto-applied SQL migrations
static/
└── chat.html               # Built-in SSE chat + RAG frontend
docker-compose.yml          # PostgreSQL 16 + Milvus 2.4
```

---

## Tech Stack

| Layer | Technology | Role |
|-------|-----------|------|
| 🦀 **Runtime** | **Rust** + **Tokio** + **Actix-web 4** | Async web server with zero-cost abstractions |
| 🤖 **LLM** | **OpenRouter** → **Groq** (`gpt-oss-safeguard-20b`) | Chat completions + SSE streaming at LPU speed |
| 📐 **Embeddings** | **OpenRouter** (or any OpenAI-compatible API) | Document + query vectorization |
| 🔍 **Vector DB** | **Milvus 2.4** (HNSW, cosine similarity) | Sub-millisecond approximate nearest neighbor search |
| 🗄️ **Database** | **PostgreSQL 16** + **SQLx** | Users, auth, migrations, compile-time checked queries |
| 🔐 **Auth** | **JWT** + **Argon2id** | Stateless authentication with secure password storage |
| 📄 **Docs** | **utoipa** → OpenAPI 3.0 → **Swagger UI** | Auto-generated interactive API docs |
| 📦 **Ingestion** | **pdf-extract** + **zip** crate | PDF text extraction, ZIP archive processing |
| 🐳 **Infra** | **Docker Compose** | One-command PostgreSQL + Milvus setup |

---

## Development

```bash
# Development mode
cargo run

# Debug logging
RUST_LOG=debug cargo run

# Run tests
cargo test

# Production build
cargo build --release
./target/release/alpharust
```

---

## License

MIT

---

<div align="center">
  <br/>
  Built with 🦀 by <a href="https://alphacorp.ai"><strong>AlphaCorp AI</strong></a>
  <br/><br/>
</div>
