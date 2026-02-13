<div align="center">

# ⚡ AlphaRustyRAG

**Sub-second RAG retrieval and generation, built in Rust.**

A high-performance Retrieval-Augmented Generation API that chunks, embeds, stores, and queries documents — then streams LLM answers grounded in your data.

Built by **[Ignas Vaitukaitis](https://www.linkedin.com/in/ignas-vaitukaitis/)** · AI Agent Engineer

<br/>

<img src="https://img.shields.io/badge/Rust-000000?style=for-the-badge&logo=rust&logoColor=white" alt="Rust"/>
<img src="https://img.shields.io/badge/Actix--web-000000?style=for-the-badge&logo=rust&logoColor=white" alt="Actix-web"/>
<img src="https://img.shields.io/badge/Groq-F55036?style=for-the-badge&logo=groq&logoColor=white" alt="Groq"/>
<img src="https://img.shields.io/badge/Cohere-39594D?style=for-the-badge&logo=cohere&logoColor=white" alt="Cohere"/>
<img src="https://img.shields.io/badge/Milvus-00A1EA?style=for-the-badge&logo=apachekafka&logoColor=white" alt="Milvus"/>
<img src="https://img.shields.io/badge/PostgreSQL-4169E1?style=for-the-badge&logo=postgresql&logoColor=white" alt="PostgreSQL"/>
<img src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker"/>
<img src="https://img.shields.io/badge/Swagger_UI-85EA2D?style=for-the-badge&logo=swagger&logoColor=black" alt="Swagger"/>
<img src="https://img.shields.io/badge/Tokio-463E52?style=for-the-badge&logo=rust&logoColor=white" alt="Tokio"/>

</div>

---

## 🏁 Benchmarks

> Tested on **Azure Standard F4s v2** (4 vCPUs, 8 GiB memory) in **West US**, using **Groq** (`openai/gpt-oss-20b`) for LLM completions and **Cohere** (`embed-english-light-v3.0`) for embeddings.
>
> The vector database contains the full **1,000 PDFs** from the [Open RAG Bench dataset](https://www.reddit.com/r/Rag/comments/1nkad09/open_rag_bench_dataset_1000_pdfs_3000_queries/) ([Google Drive](https://drive.google.com/drive/u/1/folders/18q_zokgsrMsL-Xfx4OcYST1DLb8TNzYY)) — totaling **10,250 + 16,968 chunks** (500-word chunk size, 50-word overlap). All answers include **3 sources**.

### Results Summary

| Metric | USA Server (curl localhost) | Brazil Browser → USA Server |
|--------|----------------------------|-----------------------------|
| **Time to First Token (TTFT)** | **< 160 ms** | **< 900 ms** |
| **Total Completion** | **< 700 ms** | **< 1 second** |

### Query 1 — *"Is there an iterative process involved in refining code during collaboration among LLM agents?"*

<table>
<tr>
<td><strong>🇺🇸 USA Server (curl)</strong></td>
<td><strong>🇧🇷 Brazil Browser → USA Server</strong></td>
</tr>
<tr>
<td><img src="docs/benchmarks/1_usa_server.png" alt="Query 1 — USA server curl results" width="500"/></td>
<td><img src="docs/benchmarks/1_brazil_browser.png" alt="Query 1 — Brazil browser results" width="500"/></td>
</tr>
<tr>
<td>TTFT: 0.150–0.155s · Total: 0.459–0.567s</td>
<td>TTFT: 0.64–0.78s · Total: 0.67–0.88s</td>
</tr>
</table>

### Query 2 — *"Are higher values better for SI-SDR, SDR, PESQ, and STOI metrics?"*

<table>
<tr>
<td><strong>🇺🇸 USA Server (curl)</strong></td>
<td><strong>🇧🇷 Brazil Browser → USA Server</strong></td>
</tr>
<tr>
<td><img src="docs/benchmarks/2_usa_server.png" alt="Query 2 — USA server curl results" width="500"/></td>
<td><img src="docs/benchmarks/2_brazil_browser.png" alt="Query 2 — Brazil browser results" width="500"/></td>
</tr>
<tr>
<td>TTFT: 0.149–0.150s · Total: 0.506–0.632s</td>
<td>TTFT: 0.65–0.80s · Total: 0.80–0.84s</td>
</tr>
</table>

### Query 3 — *"What is the accuracy of the base NLP model for market return direction?"*

<table>
<tr>
<td><strong>🇺🇸 USA Server (curl)</strong></td>
<td><strong>🇧🇷 Brazil Browser → USA Server</strong></td>
</tr>
<tr>
<td><img src="docs/benchmarks/3_usa_server.png" alt="Query 3 — USA server curl results" width="500"/></td>
<td><img src="docs/benchmarks/3_brazil_browser.png" alt="Query 3 — Brazil browser results" width="500"/></td>
</tr>
<tr>
<td>TTFT: 0.146–0.149s · Total: 0.601–0.654s</td>
<td>TTFT: 0.54–0.64s · Total: 0.64–0.81s</td>
</tr>
</table>

### 🗺️ Roadmap

- Further optimize TTFT and total completion latency
- Improve answer accuracy and source relevance
- Expand benchmark coverage with more query types

---

## Why AlphaRust?

Most RAG stacks glue together Python microservices with high per-request overhead. AlphaRust collapses the entire pipeline — document ingestion, vector search, and LLM streaming — into a **single async Rust binary**. The result: **sub-1-second time-to-first-token** on retrieval-augmented queries, even with large document collections.

### Key Features

- **Full RAG pipeline in one binary** — upload, chunk, embed, store, search, and generate
- **Two API keys, maximum speed** — [Groq](https://groq.com) for LLM completions, [Cohere](https://cohere.com) for embeddings
- **Real-time SSE streaming** — tokens stream to the client as they're generated, with sources delivered as a leading SSE event
- **Default model: `openai/gpt-oss-20b`** — running on **Groq's LPU** inference hardware for maximum speed
- **Cohere Embed v3** — `embed-english-light-v3.0` at 384 dimensions for fast, high-quality embeddings with asymmetric search (separate query vs document encoding)
- **Concurrent document ingestion** — ZIP archives are unpacked and processed across 8 parallel workers with batched embedding calls
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
│              │       │  │ Actix-web│  │  Cohere    │  │    Groq      │  │
└──────────────┘       │  │  Router  │─►│ Embeddings │  │  LLM Client  │  │
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
                       ┌──────────────────┴──────────────────┐
                       │                                     │
              ┌────────────────┐                 ┌───────────────────┐
              │   Groq API     │                 │   Cohere API      │
              │  LLM on LPU    │                 │   Embed v3        │
              └────────────────┘                 └───────────────────┘
```

---

## Quick Start

### Prerequisites

- **Rust 1.70+** — install via [rustup](https://rustup.rs/)
- **Docker & Docker Compose** — for PostgreSQL and Milvus
- **Groq API key** — get one at [console.groq.com](https://console.groq.com/) — for LLM completions
- **Cohere API key** — get one at [dashboard.cohere.com](https://dashboard.cohere.com/) — for embeddings

### 1. Clone and configure

```bash
git clone https://github.com/AlphaCorp-AI/AlphaRustyRAG
cd alpharust
cp .env.example .env
```

Edit `.env` with your API keys:

```env
# ── Required ─────────────────────────────────────────────
GROQ_API_KEY=gsk_your-groq-key-here
COHERE_API_KEY=your-cohere-key-here

# ── Infrastructure (defaults work with docker-compose) ───
DATABASE_URL=postgres://alpharust:alpharust@localhost:5432/alpharust
JWT_SECRET=change-me-to-a-long-random-string
MILVUS_URL=http://localhost:19530

# ── Embedding model ─────────────────────────────────────
EMBEDDING_MODEL=embed-english-light-v3.0
EMBEDDING_DIMENSION=384
```

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

## Default LLM: openai/gpt-oss-20b via Groq

AlphaRust ships with **`openai/gpt-oss-20b`** as the default model, running on **[Groq](https://groq.com)**'s LPU inference hardware. This gives you:

- **Extremely low latency** — Groq's LPU delivers tokens faster than GPU-based providers
- **High quality** — gpt-oss-20b provides strong reasoning and instruction following
- **Simple setup** — just set your `GROQ_API_KEY`

You can override the model per request by passing `"model": "openai/gpt-oss-20b"` (or any other [Groq model](https://console.groq.com/docs/models)) in the `/chat` request body.

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
  → Batch embedding via Cohere (100 chunks per API call, input_type=search_document)
  → Batch insert into Milvus (50 chunks per insert)
```

### Query flow

```
User question
  → Embed the query via Cohere (input_type=search_query)
  → Milvus HNSW search (top-K, cosine similarity)
  → Inject retrieved chunks as system prompt context
  → Stream LLM answer via SSE (Groq)
  → Sources emitted as leading "event: sources" SSE event
```

---

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `GROQ_API_KEY` | Groq API key for LLM chat completions | *required* |
| `COHERE_API_KEY` | Cohere API key for embeddings | *required* |
| `DATABASE_URL` | PostgreSQL connection string | *required (local)* |
| `JWT_SECRET` | Secret for signing JWT tokens | *required (local)* |
| `MILVUS_URL` | Milvus REST API endpoint | `http://localhost:19530` |
| `LLM_MODEL` | Groq model for chat completions | `openai/gpt-oss-20b` |
| `EMBEDDING_MODEL` | Cohere embedding model name | `embed-english-light-v3.0` |
| `EMBEDDING_DIMENSION` | Vector dimensionality (must match model) | `384` |
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
│   ├── llm.rs              # Groq LLM client (sync + SSE streaming)
│   ├── embeddings.rs       # Cohere Embed v3 client (asymmetric search support)
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
| 🤖 **LLM** | **Groq** (`openai/gpt-oss-20b`) | Chat completions + SSE streaming at LPU speed |
| 📐 **Embeddings** | **Cohere** Embed v3 (`embed-english-light-v3.0`) | Fast asymmetric document + query vectorization |
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