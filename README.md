<div align="center">

<img src="docs/rustyrag_logo.png" alt="RustyRAG Logo" width="200"/>

<br/>

### The fastest open-source RAG API on GitHub

Sub-200ms time-to-first-token. Hybrid search + cross-encoder reranking + streaming answers from the fastest LLM providers on the planet. Written in Rust.

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

## Performance

Benchmarked on **3,045 questions** from [Open RAG Bench](https://github.com/vectara/open-rag-bench) (academic papers, 1,000 PDFs, 57,347 chunks). LLM-as-judge evaluation using `qwen-3-235b`. Infrastructure running on **VastAI RTX 4090** (USA region).

### Benchmark Results

```
                With Reranker                         Without Reranker
  ┌───────────────────────────────────┐    ┌───────────────────────────────────┐
  │                                   │    │                                   │
  │   Pass Rate:  94.5%               │    │   Pass Rate:  91.6%               │
  │   ████████████████████████░░░     │    │   ██████████████████████░░░░░     │
  │   2,857 / 3,024 judged            │    │   2,767 / 3,020 judged            │
  │                                   │    │                                   │
  └───────────────────────────────────┘    └───────────────────────────────────┘
```

### Latency Comparison

```
  Time to First Token (TTFT)
  ──────────────────────────────────────────────────────────
  With Reranker     ████████████████████████████  279ms
  Without Reranker  ████████████████████          181ms

  Total Response Time
  ──────────────────────────────────────────────────────────
  With Reranker     ████████████████████████████████████████████████████████████  883ms
  Without Reranker  ██████████████████████████████████████                       511ms
```

### Summary

| Metric | With Reranker | Without Reranker |
|--------|:---:|:---:|
| **Pass rate** | **94.5%** | 91.6% |
| **Avg TTFT** | 279ms | **181ms** |
| **Avg total response** | 883ms | **511ms** |
| Failed | 167 / 3,024 | 253 / 3,020 |
| LLM | Cerebras `qwen-3-235b` | Cerebras `qwen-3-235b` |
| Embeddings | Jina v5 nano (768-dim) | Jina v5 nano (768-dim) |
| Search | Hybrid HNSW + BM25 + RRF | Hybrid HNSW + BM25 + RRF |
| Reranker | Jina Reranker v3 | Disabled |
| Retrieval | 20 → top 3 after reranking | Top 3 from Milvus |
| Infra | VastAI RTX 4090 (USA) | VastAI RTX 4090 (USA) |

> Reranking adds ~100ms TTFT and ~370ms total but improves accuracy by **+2.9 percentage points** and cuts failures by **34%**. Without reranking, you get sub-200ms TTFT — choose your tradeoff.

---

## Features

- **Sub-200ms TTFT** — first tokens stream back before most APIs even respond
- **Hybrid search** — dense HNSW vectors + BM25 sparse keywords fused with Reciprocal Rank Fusion
- **Cross-encoder reranking** — Jina Reranker v3 scores each candidate against the query for precise relevance
- **SSE streaming** — real-time token streaming with source citations and server-side timing
- **Multi-format ingestion** — PDF, DOCX, PPTX, XLSX, HTML, TXT, ZIP via Docling (tables, OCR, layout-aware)
- **Image understanding** — embedded images described by Llama 4 Scout vision model
- **Contextual retrieval** — optional LLM-generated chunk prefixes for better search recall
- **Multiple LLM providers** — Groq (LPU) and Cerebras (wafer-scale) with OpenAI-compatible APIs
- **Built-in eval framework** — 3,045-question benchmark with LLM-as-judge scoring
- **Single binary** — no Python, no Node.js, no runtime dependencies on the server
- **Swagger UI** — auto-generated interactive API docs
- **Collection management** — multiple document collections with backup/export

---

## Use Cases

RustyRAG is built for scenarios where both **speed and accuracy** matter:

- **Voice AI / Conversational Agents** — sub-200ms TTFT means RAG-grounded answers start streaming before the user notices a pause, critical for natural voice interactions
- **AI Agents** — give your agents a fast, reliable knowledge backend — agents can call RustyRAG as a tool to ground their reasoning in real documents instead of hallucinating
- **Legal & Compliance** — search across contracts, filings, and policy documents with table-aware extraction and precise reranking
- **Research & Academic** — query thousands of papers with hybrid search that catches both semantic meaning and exact terminology
- **Internal Knowledge Bases** — drop in PDFs, docs, and spreadsheets — get an instant Q&A system with no training required
- **Real-time Copilots** — embed RustyRAG behind IDE plugins, dashboards, or Slack bots where latency kills adoption

> If your RAG pipeline adds 2-3 seconds of latency, users won't use it. RustyRAG keeps the full pipeline — retrieval, reranking, and generation — under 1 second.

---

## How It Works

### Query Flow

```
  User Question
       │
       ▼
  ┌──────────┐     ┌─────────────────────────────────────┐     ┌──────────────┐
  │  Embed   │────▶│  Milvus Hybrid Search               │────▶│   Reranker   │
  │ (Jina v5)│     │  Dense HNSW + BM25 Sparse + RRF     │     │  (Jina v3)   │
  └──────────┘     │  → 20 candidates                    │     │   → Top 3    │
                   └─────────────────────────────────────┘     └──────┬───────┘
                                                                      │
                                                                      ▼
                                                               ┌──────────────┐
                                                               │  LLM (SSE)   │
                                                               │ Groq/Cerebras│
                                                               └──────┬───────┘
                                                                      │
                                                                      ▼
                                                               Sources + Tokens
                                                                + Timing (ms)
```

### Upload Flow

```
  File (.pdf, .docx, .pptx, .xlsx, .html, .txt, .zip)
       │
       ├─ 1. Extract ──────────────────► Docling (layout-aware, tables, OCR)
       │
       ├─ 2. Describe images ──────────► Groq Llama 4 Scout (vision, 17B)
       │
       ├─ 3. Chunk ────────────────────► Sentence-boundary-aware splitting
       │
       ├─ 4. Contextual prefix (opt-in) ► LLM generates 1-2 sentence context
       │
       ├─ 5. Embed ────────────────────► Jina v5 nano (768-dim vectors)
       │
       └─ 6. Insert ───────────────────► Milvus (dense HNSW + BM25 index)
```

### Why reranking matters

Hybrid search retrieves 20 candidates using two complementary signals — dense vectors (semantic meaning) and BM25 (exact keyword matching). But vector similarity and keyword overlap are rough proxies. The cross-encoder reranker reads each candidate alongside the query as a single sequence and produces a precise relevance score. This promotes the most relevant chunks and demotes false positives, improving answer quality by **+2.9%** without increasing LLM context size.

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

This starts **Docling** (document extraction), **Jina embeddings** (TEI), **cross-encoder reranker** (TEI), and **Milvus 2.5** locally.

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

Upload a PDF, ask a question, and watch tokens stream back with source citations.

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

## API Reference

All endpoints live under `/api/v1`. Interactive docs at [`/swagger-ui/`](http://localhost:8080/swagger-ui/).

### Documents

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/documents/upload` | Upload files — extract, chunk, embed, store in Milvus |
| `POST` | `/documents/search` | Semantic search across embedded documents |
| `GET`  | `/documents/backup` | Download full collection as gzip-compressed JSON |

### Chat

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/llms` | List supported models with provider names |
| `POST` | `/chat-rag` | RAG: hybrid search, rerank, generate answer |
| `POST` | `/chat-rag/stream` | SSE-streamed RAG (sources event + LLM tokens + timing) |

### Evals

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/evals/run` | Run evaluation benchmark (3,045 questions) and download results as JSON |

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

## Infrastructure

All services run via Docker Compose:

| Service | Image | Port | Role |
|---------|-------|------|------|
| **Embeddings** | `text-embeddings-inference:cuda-1.9` | 7997 | Jina v5 nano (768-dim) |
| **Reranker** | `text-embeddings-inference:cuda-1.9` | 7998 | Jina Reranker v3 |
| **Docling** | `docling-serve-cu128:v1.14.3` | 5001 | Document extraction (PDF/DOCX/PPTX) |
| **Milvus** | `milvus:v2.5.27` | 19530 | Vector DB (HNSW + BM25) |

GPU is used by embeddings, reranker, and Docling. The Rust server itself runs on CPU only.

---

## Project Structure

```
src/
├── main.rs                 # Entry point, server bootstrap
├── config.rs               # Env-based config (required/optional/tuning)
├── routes.rs               # Route registration
├── errors.rs               # AppError → HTTP response mapping
├── handlers/
│   ├── chat.rs             # /llms, /chat-rag + streaming variants
│   ├── documents.rs        # /documents/upload, /documents/search, /documents/backup
│   ├── evals.rs            # /evals/run (benchmark runner)
│   └── health.rs           # /health
├── schemas/
│   ├── requests.rs         # Request DTOs (serde + validator)
│   └── responses.rs        # Response DTOs (utoipa OpenAPI schemas)
├── services/
│   ├── docling.rs          # Docling API client + vision image descriptions
│   ├── document.rs         # Document extraction routing, semantic chunking
│   ├── embeddings.rs       # Jina/OpenAI-compatible embedding client
│   ├── llm.rs              # Groq + Cerebras LLM client
│   ├── milvus.rs           # Milvus v2 REST client (hybrid search, BM25, backup)
│   └── reranker.rs         # TEI cross-encoder reranker client
├── prompts/
│   ├── mod.rs              # Prompt builder functions
│   ├── rag_system_prompt.txt
│   └── contextual_prompt.txt
scripts/
└── judge_evals.py          # LLM-as-judge evaluation script
static/
└── chat.html               # Built-in SSE chat + RAG frontend
docs/
├── eval_data.csv           # 3,045 evaluation questions with expected answers
├── eval_judged_reranker.json    # Benchmark results (with reranker)
└── eval_judged_no_rerank.json   # Benchmark results (without reranker)
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
| **Reranker** | Jina Reranker v3 (TEI) | Cross-encoder reranking |
| **Document extraction** | Docling (IBM) | PDF/DOCX/PPTX tables, OCR, layout analysis |
| **Vector DB** | Milvus 2.5 (HNSW + BM25) | Hybrid dense + sparse search with RRF |
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

# Run evaluation benchmark
curl -X POST http://localhost:8080/api/v1/evals/run -H 'Content-Type: application/json' -d '{}' -o eval_results.json

# Judge evaluation results
python3 scripts/judge_evals.py docs/eval_results.json
```

---

## Roadmap

- **95%+ accuracy without reranker** — improve retrieval quality so the fast path (sub-200ms TTFT) matches reranker-level accuracy
- **99%+ accuracy with reranker** — push the reranker path toward near-perfect on Open RAG Bench through better chunking, contextual retrieval, and prompt tuning
- **Hosted version** — deploy a managed RustyRAG instance so you can get started with a single API key, no infrastructure required

---

## License

[Elastic License 2.0 (ELv2)](LICENSE)

---

<div align="center">
  <br/>
  Built by <strong>Ignas Vaitukaitis</strong> &nbsp; <a href="https://www.linkedin.com/in/ignas-vaitukaitis/"><img src="https://img.shields.io/badge/LinkedIn-0A66C2?style=flat-square&logo=linkedin&logoColor=white" alt="LinkedIn"/></a> <a href="https://x.com/zer0tokens"><img src="https://img.shields.io/badge/X-000000?style=flat-square&logo=x&logoColor=white" alt="X"/></a>
  <br/><br/>
</div>
