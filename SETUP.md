# RustyRAG - GPU Server Setup Guide

Step-by-step setup for a fresh Ubuntu instance with an NVIDIA GPU (e.g. VastAI, RunPod, bare metal).

## Prerequisites

- Ubuntu 22.04+ with NVIDIA GPU (tested on RTX 4090)
- NVIDIA drivers installed (`nvidia-smi` should work)
- API keys for [Groq](https://console.groq.com/) and/or [Cerebras](https://cloud.cerebras.ai/)

## 1. Install system dependencies

```bash
# Essential build tools
sudo apt update && sudo apt install -y build-essential pkg-config libssl-dev curl git

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

# Install Docker + Docker Compose
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker "$USER"
newgrp docker

# Install NVIDIA Container Toolkit (required for GPU containers)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update && sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Verify GPU access from Docker:

```bash
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

## 2. Clone and configure

```bash
git clone https://github.com/AlphaCorp-AI/RustyRAG.git
cd RustyRAG
cp .env.example .env
```

Edit `.env` and add your API keys:

```bash
# Required - at least one LLM provider
GROQ_API_KEY=gsk_your_key_here
CEREBRAS_API_KEY=csk_your_key_here
```

All other settings have sensible defaults for the docker-compose stack. See `.env.example` for the full list.

## 3. Start infrastructure services

```bash
docker compose up -d
```

This starts four services:

| Service | Port | What it does |
|---------|------|-------------|
| **embeddings** (TEI + Jina v5 small) | 7997 | Generates 1024-dim embeddings on GPU |
| **reranker** (TEI + Jina Reranker v3) | 7998 | Cross-encoder reranking on GPU |
| **docling** | 5001 | Extracts text from PDF/DOCX/PPTX/XLSX |
| **milvus** | 19530 | Vector database (hybrid dense + BM25 search) |

Wait for all services to become healthy (first run downloads models, can take a few minutes):

```bash
# Watch health status
docker compose ps

# Or poll until all healthy
until docker compose ps --format json | python3 -c "
import sys, json
services = [json.loads(l) for l in sys.stdin]
healthy = all('healthy' in s.get('Health','') for s in services)
print('All healthy' if healthy else 'Waiting...')
sys.exit(0 if healthy else 1)
"; do sleep 10; done
```

You can also check individual services:

```bash
curl -s http://localhost:7997/health   # embeddings
curl -s http://localhost:7998/health   # reranker
curl -s http://localhost:5001/health   # docling
curl -s http://localhost:9091/healthz  # milvus
```

## 4. Build and run the app

```bash
cargo build --release
cargo run --release
```

The server starts on `http://localhost:8080` by default.

Verify:

```bash
curl http://localhost:8080/api/v1/health
# {"status":"ok","version":"0.3.0"}
```

## 5. Access the UI

- **Swagger UI** (API docs): http://localhost:8080/swagger-ui/
- **Chat UI** (built-in): http://localhost:8080/static/chat.html
- **Docling UI** (document extraction): http://localhost:5001/ui

## Quick test

Upload a document and ask a question:

```bash
# Upload a PDF
curl -X POST http://localhost:8080/api/v1/documents/upload \
  -F "file=@your-document.pdf" \
  -F "collection_name=test"

# Ask a question (non-streaming)
curl -X POST http://localhost:8080/api/v1/chat-rag \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is this document about?",
    "collection_name": "test",
    "model": "llama-3.3-70b-versatile",
    "provider": "groq"
  }'

# Stream a response (SSE)
curl -N -X POST http://localhost:8080/api/v1/chat-rag/stream \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Summarize the key points",
    "collection_name": "test",
    "model": "llama-3.3-70b-versatile",
    "provider": "groq"
  }'
```

## API endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/health` | Health check |
| GET | `/api/v1/llms` | List available LLM models |
| POST | `/api/v1/documents/upload` | Upload and index a document |
| POST | `/api/v1/documents/search` | Semantic search across documents |
| POST | `/api/v1/chat-rag` | RAG chat (full response) |
| POST | `/api/v1/chat-rag/stream` | RAG chat (SSE stream) |

## Running as a background service

```bash
# Option 1: systemd service
sudo tee /etc/systemd/system/rustyrag.service > /dev/null <<EOF
[Unit]
Description=RustyRAG API Server
After=network.target docker.service
Requires=docker.service

[Service]
Type=simple
User=$USER
WorkingDirectory=$(pwd)
EnvironmentFile=$(pwd)/.env
ExecStart=$(pwd)/target/release/rustyrag
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now rustyrag

# Check logs
journalctl -u rustyrag -f

# Option 2: simple background with nohup
nohup cargo run --release > rustyrag.log 2>&1 &
```

## Exposing to the internet (VastAI)

VastAI maps container ports automatically. If running on a VastAI instance, the app is already accessible via the instance's public IP on the mapped port. For other setups:

```bash
# Change bind address from localhost to all interfaces
# In .env:
HOST=0.0.0.0
PORT=8080
```

## Troubleshooting

**GPU not available in Docker containers:**
```bash
# Verify NVIDIA runtime is configured
docker info | grep -i runtime
# Should show: Runtimes: nvidia runc

# If not, re-run:
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

**TEI containers crash or OOM:**
```bash
# Check logs
docker compose logs embeddings
docker compose logs reranker

# If VRAM is limited, fall back to CPU for the reranker:
# In docker-compose.yml, change reranker image to cpu-1.9
# and remove the GPU device reservation
```

**Milvus won't start:**
```bash
# Check if port 19530 is already in use
sudo lsof -i :19530

# Reset Milvus data (warning: deletes all indexed documents)
docker compose down -v
docker compose up -d
```

**Model download slow on first startup:**
```bash
# TEI downloads models to the hf_cache volume on first run
# Check download progress:
docker compose logs -f embeddings
docker compose logs -f reranker
```
