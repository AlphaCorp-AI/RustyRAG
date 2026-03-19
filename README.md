<div align="center">

# RustyRAG

### Production-grade RAG in a single Rust binary

Sub-200ms responses on localhost. Sub-600ms to a browser across continents. No GPU required.

<br/>

<a href="https://cerebras.ai"><img src="https://img.shields.io/badge/Cerebras-FF6B00?style=for-the-badge" alt="Cerebras"/></a>
<a href="https://groq.com"><img src="https://img.shields.io/badge/Groq-F55036?style=for-the-badge&logo=groq&logoColor=white" alt="Groq"/></a>
<a href="https://jina.ai"><img src="https://img.shields.io/badge/Jina_AI-111827?style=for-the-badge" alt="Jina AI"/></a>
<a href="https://milvus.io"><img src="https://img.shields.io/badge/Milvus-00A1EA?style=for-the-badge" alt="Milvus"/></a>
<a href="https://huggingface.co"><img src="https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="HuggingFace"/></a>
<a href="https://ds4sd.github.io/docling/"><img src="https://img.shields.io/badge/Docling-4B0082?style=for-the-badge" alt="Docling"/></a>

<br/>

<img src="https://img.shields.io/badge/Rust-000000?style=flat-square&logo=rust&logoColor=white" alt="Rust"/>
<img src="https://img.shields.io/badge/Actix--web-000000?style=flat-square&logo=rust&logoColor=white" alt="Actix-web"/>
<img src="https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white" alt="Docker"/>
<img src="https://img.shields.io/badge/Swagger_UI-85EA2D?style=flat-square&logo=swagger&logoColor=black" alt="Swagger"/>

<br/>

Built by **Ignas Vaitukaitis** &nbsp; <a href="https://www.linkedin.com/in/ignas-vaitukaitis/"><img src="https://img.shields.io/badge/LinkedIn-0A66C2?style=flat-square&logo=linkedin&logoColor=white" alt="LinkedIn"/></a> <a href="https://x.com/zer0tokens"><img src="https://img.shields.io/badge/X-000000?style=flat-square&logo=x&logoColor=white" alt="X"/></a>

</div>

---

## Benchmarks

| Metric | Value |
|--------|-------|
| Localhost response time | **< 200ms** |
| Remote response time (Azure North Central US → Rio de Janeiro) | **< 600ms** |
| Instance | Azure F4s_v2 (no GPU) |
| Sources per response | 3 |
| PDFs ingested | 977 |
| Chunks in Milvus | 56,114 |

### Localhost (curl → localhost:8080)

RAG streaming responses with 3 sources, Cerebras `qwen-3-235b-a22b-instruct-2507`. TTFT = time to first token.

<p>
<img src="docs/localhost_time_1.png" alt="Localhost benchmark 1 — TTFT: 159ms, Total: 199ms" width="100%"/>
</p>
<p>
<img src="docs/localhost_time_2.png" alt="Localhost benchmark 2 — TTFT: 165ms, Total: 187ms" width="100%"/>
</p>
<p>
<img src="docs/localhost_time_3.png" alt="Localhost benchmark 3 — TTFT: 169ms, Total: 208ms" width="100%"/>
</p>

### Browser (Azure North Central US → Rio de Janeiro)

Same 977-PDF corpus, same model. Chat UI showing TTFT and total response time per query.

<p>
<img src="docs/browser_time_1.png" alt="Browser benchmark 1" width="700"/>
</p>
<p>
<img src="docs/browser_time_2.png" alt="Browser benchmark 2" width="700"/>
</p>
<p>
<img src="docs/browser_time_3.png" alt="Browser benchmark 3" width="700"/>
</p>

---

## What's New in v0.3

**Document extraction** — Replaced `pdf-extract` with [Docling](https://ds4sd.github.io/docling/) (IBM). Tables are extracted as markdown tables, multi-column layouts are linearized correctly, scanned pages get OCR, and embedded images are described by a vision model.

**Hybrid search** — Dense vector search (HNSW) + BM25 keyword search combined via Reciprocal Rank Fusion. Catches exact terms and acronyms that pure semantic search misses.

**Reranker** — Retrieved chunks are reranked by a cross-encoder (jina-reranker-v2) before being sent to the LLM. Retrieves 20 candidates, keeps the top 3.

**Vision model** — Images found inside PDFs are described by Llama 4 Scout (17B) via Groq and included in the chunk text so they're searchable.

**More file formats** — PDF, DOCX, PPTX, XLSX, HTML, TXT, and ZIP archives containing any of these.

**Production metadata** — Every search result now returns `id`, `file_name`, `file_size`, `page_number`, and `chunk_index`.

---

## Supported File Formats

| Format | Extension | Extraction Method |
|--------|-----------|-------------------|
| Plain text | `.txt` | Direct UTF-8 read |
| PDF | `.pdf` | Docling (layout-aware, tables, OCR) |
| Word | `.docx` | Docling |
| PowerPoint | `.pptx` | Docling |
| Excel | `.xlsx` | Docling |
| HTML | `.html` | Docling |
| ZIP archive | `.zip` | Unpacked, each entry processed individually |

Docling provides:
- **Table extraction** — 97.9% accuracy on complex tables via TableFormer
- **Layout analysis** — multi-column, headers/footers handled correctly
- **OCR** — automatic for scanned pages (skipped for digital text = fast)
- **Image descriptions** — embedded images described by Llama 4 Scout vision model

---

## Architecture

```
                                RustyRAG
┌──────────┐         ┌──────────────────────────────────────────────────┐
│          │   SSE   │                                                  │
│  Client  │◄───────►│  Actix-web Router                               │
│          │         │       │                                         │
└──────────┘         │       ├── /documents/upload                     │
                     │       │     ├─ Extract (Docling API) ──────────►│── Docling
                     │       │     ├─ Describe images ─────────────────►│── Groq (Llama 4 Scout)
                     │       │     ├─ Semantic chunking                │
                     │       │     ├─ Contextual prefix gen ──────────►│── LLM API
                     │       │     ├─ Embed (prefix + chunk) ─────────►│── Jina TEI
                     │       │     └─ Insert vectors ─────────────────►│── Milvus
                     │       │                                         │
                     │       ├── /chat-rag/stream                      │
                     │       │     ├─ Embed query ────────────────────►│── Jina TEI
                     │       │     ├─ Hybrid search (dense + BM25) ───►│── Milvus
                     │       │     ├─ Rerank (cross-encoder) ─────────►│── Jina Reranker
                     │       │     └─ Stream answer ──────────────────►│── LLM API
                     │       │                                         │
                     │       └── /chat/stream                          │
                     │             └─ Direct LLM streaming ───────────►│── LLM API
                     └──────────────────────────────────────────────────┘
```

---

## Quick Start

### Prerequisites

- **Rust 1.70+** — install via [rustup](https://rustup.rs/)
- **Docker & Docker Compose** — for Docling, embeddings, reranker, and Milvus
- **Groq API key** — get one at [console.groq.com](https://console.groq.com/) (also powers vision model)
- **Cerebras API key** — get one at [cloud.cerebras.ai](https://cloud.cerebras.ai/)

### 1. Clone and configure

```bash
git clone https://github.com/AlphaCorp-AI/RustyRAG
cd rustyrag
cp .env.example .env
```

Edit `.env` with your API keys:

```env
GROQ_API_KEY=gsk_your-groq-key-here
CEREBRAS_API_KEY=csk_your-cerebras-key-here
```

### 2. Start infrastructure

```bash
docker compose up -d
```

This starts **Docling** (document extraction), **Jina embeddings** (TEI), **Jina reranker** (TEI), and **Milvus 2.5** locally.

### 3. Build and run

```bash
cargo build --release
cargo run --release
```

The server starts at `http://127.0.0.1:8080`.

### 4. Try it out

- **Chat UI** — [http://localhost:8080/static/chat.html](http://localhost:8080/static/chat.html)
- **Swagger UI** — [http://localhost:8080/swagger-ui/](http://localhost:8080/swagger-ui/)
- **Docling UI** — [http://localhost:5001/ui](http://localhost:5001/ui)

Upload a PDF (with tables, images, multi-column), ask a question, and watch tokens stream back with source citations.

---

## RAG Pipeline

### Upload flow

```
File upload (.txt, .pdf, .docx, .pptx, .xlsx, .html, or .zip)
  → Docling extraction (tables as markdown, OCR, layout analysis)
  → Image description via Llama 4 Scout vision model
  → Semantic chunking (sentence-boundary-aware, configurable size + overlap)
  → Contextual prefix generation per chunk via LLM (opt-in)
  → Embed (prefix + chunk text) via local Jina TEI
  → Batch insert into Milvus (dense vectors + BM25 sparse vectors)
```

### Query flow

```
User question
  → Embed query via Jina TEI
  → Hybrid search: dense HNSW + BM25 sparse (RRF fusion, top 20)
  → Rerank via cross-encoder (jina-reranker-v2, top 3)
  → Inject top chunks as system prompt context
  → Stream LLM answer via SSE (Groq or Cerebras)
  → Sources emitted as leading "event: sources" SSE event
```

### Contextual retrieval

Each chunk can be enriched with an LLM-generated context prefix before embedding. The LLM receives a document overview plus surrounding pages and produces a 1-2 sentence description. This prefix is concatenated with the chunk text before embedding, so vectors encode both content and document context.

---

## API Reference

All endpoints live under `/api/v1`. Interactive docs at [`/swagger-ui/`](http://localhost:8080/swagger-ui/).

<p>
<img src="docs/swagger-ui.png" alt="Swagger UI" width="700"/>
</p>

### Documents

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/documents/upload` | Upload files — extract, chunk, embed, store in Milvus |
| `POST` | `/documents/search` | Semantic search across embedded documents |

### Chat

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/llms` | List supported models with provider names |
| `POST` | `/chat` | Single-turn LLM completion |
| `POST` | `/chat/stream` | SSE-streamed LLM completion |
| `POST` | `/chat-rag` | RAG: hybrid search → rerank → generate answer |
| `POST` | `/chat-rag/stream` | SSE-streamed RAG (sources event + LLM tokens) |

### Health

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/health` | Liveness check |

### Search result fields

Every search hit and RAG source returns:

| Field | Type | Description |
|-------|------|-------------|
| `id` | i64 | Milvus document ID |
| `file_name` | string | Original uploaded filename |
| `file_size` | i64 | File size in bytes |
| `chunk_index` | i64 | Position of chunk within the file |
| `page_number` | i64 | Page the chunk came from (0 if unknown) |
| `score` | f32 | Relevance score (from reranker or cosine similarity) |
| `text` | string | Chunk content |

---

## LLM Providers

RustyRAG uses **Groq** and **Cerebras** for their low-latency inference hardware. Both expose OpenAI-compatible APIs.

### Groq
- `llama-3.1-8b-instant`
- `llama-3.3-70b-versatile`
- `openai/gpt-oss-120b`
- `openai/gpt-oss-20b`

### Cerebras
- `llama3.1-8b`
- `gpt-oss-120b`
- `qwen-3-235b-a22b-instruct-2507`
- `zai-glm-4.7`

### Vision
- `meta-llama/llama-4-scout-17b-16e-instruct` (Groq) — describes images found inside documents

---

## Configuration

### Required

| Variable | Description |
|----------|-------------|
| `HOST` | Server bind address |
| `PORT` | Server port |
| `GROQ_API_KEY` | Groq API key (also used for vision model) |
| `CEREBRAS_API_KEY` | Cerebras API key |

### Optional — services

| Variable | Default | Description |
|----------|---------|-------------|
| `MILVUS_URL` | `http://localhost:19530` | Milvus REST API |
| `EMBEDDING_API_URL` | `http://localhost:7997/v1/embeddings` | Embedding endpoint |
| `EMBEDDING_MODEL` | `jinaai/jina-embeddings-v5-text-nano-retrieval` | Embedding model |
| `RERANKER_API_URL` | *(empty = disabled)* | TEI reranker endpoint |
| `DOCLING_URL` | `http://localhost:5001` | Docling extraction API |
| `VISION_MODEL` | `meta-llama/llama-4-scout-17b-16e-instruct` | Vision model for images |

### Optional — tuning

| Variable | Default | Description |
|----------|---------|-------------|
| `RETRIEVAL_LIMIT` | `20` | Chunks fetched from Milvus before reranking |
| `RERANK_TOP_N` | `3` | Chunks kept after reranking |
| `CHUNK_SIZE` | `2000` | Max characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `EMBEDDING_DIMENSION` | `768` | Vector dimensionality |
| `CORS_ALLOWED_ORIGINS` | *(empty = permissive)* | Comma-separated allowed origins |

See `.env.example` for the full list including Milvus index tuning and contextual retrieval options.

---

## Project Structure

```
src/
├── main.rs                 # Entry point, server bootstrap
├── config.rs               # Env-based config (required/optional/tuning)
├── routes.rs               # Route registration
├── errors.rs               # AppError → HTTP response mapping
├── handlers/
│   ├── chat.rs             # /llms, /chat, /chat-rag + streaming variants
│   ├── documents.rs        # /documents/upload, /documents/search
│   └── health.rs           # /health
├── schemas/
│   ├── requests.rs         # Request DTOs (serde + validator)
│   └── responses.rs        # Response DTOs (utoipa OpenAPI schemas)
├── services/
│   ├── docling.rs          # Docling API client + vision image descriptions
│   ├── document.rs         # Document extraction routing, semantic chunking
│   ├── embeddings.rs       # Jina/OpenAI-compatible embedding client
│   ├── llm.rs              # Groq + Cerebras LLM client
│   ├── milvus.rs           # Milvus v2 REST client (hybrid search, BM25)
│   └── reranker.rs         # TEI cross-encoder reranker client
├── prompts/
│   ├── mod.rs              # Prompt builder functions
│   ├── rag_system_prompt.txt
│   └── contextual_prompt.txt
static/
└── chat.html               # Built-in SSE chat + RAG frontend
docker-compose.yml          # Docling + Jina TEI + Reranker + Milvus 2.5
```

---

## Tech Stack

| Layer | Technology | Role |
|-------|-----------|------|
| **Runtime** | Rust + Tokio + Actix-web 4 | Async web server |
| **LLM** | Groq (LPU) + Cerebras (wafer-scale) | Low-latency chat completions + SSE streaming |
| **Vision** | Llama 4 Scout 17B (Groq) | Image descriptions inside documents |
| **Embeddings** | Jina v5 text nano retrieval (TEI) | Local vectorization, 768-dim |
| **Reranker** | jina-reranker-v2-base-multilingual (TEI) | Cross-encoder reranking |
| **Document extraction** | Docling (IBM) | PDF/DOCX/PPTX tables, OCR, layout analysis |
| **Vector DB** | Milvus 2.5 (HNSW + BM25) | Hybrid dense + sparse search |
| **Chunking** | text-splitter crate | Semantic sentence-boundary-aware splitting |
| **Docs** | utoipa + Swagger UI | Auto-generated interactive API documentation |
| **Infra** | Docker Compose | One-command local stack |

---

## Development

```bash
# Development mode
cargo run

# Debug logging
RUST_LOG=debug cargo run

# Production build
cargo build --release
./target/release/rustyrag
```

---

## License

MIT

---

<div align="center">
  <br/>
  Built by <strong>Ignas Vaitukaitis</strong> &nbsp; <a href="https://www.linkedin.com/in/ignas-vaitukaitis/"><img src="https://img.shields.io/badge/LinkedIn-0A66C2?style=flat-square&logo=linkedin&logoColor=white" alt="LinkedIn"/></a> <a href="https://x.com/zer0tokens"><img src="https://img.shields.io/badge/X-000000?style=flat-square&logo=x&logoColor=white" alt="X"/></a>
  <br/><br/>
</div>
