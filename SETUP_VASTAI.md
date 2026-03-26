# RustyRAG - Vast.ai RTX 4090 Setup Guide

Setup for a Vast.ai RTX 4090 instance (24 GB VRAM, 16 CPU, 48 GB RAM, 150 GB disk).

## 1. SSH into the instance

```bash
ssh -p <PORT> root@<HOST>
```

## 2. Install system dependencies

```bash
apt update && apt install -y build-essential pkg-config libssl-dev curl git

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

# Docker + NVIDIA Container Toolkit should already be available on Vast.ai
# Verify:
docker --version
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

## 3. Clone and configure

```bash
git clone --branch pre-release/v0.4 --single-branch https://github.com/AlphaCorp-AI/RustyRAG.git
cd RustyRAG

cat > .env << 'EOF'
HOST=0.0.0.0
PORT=8080
RUST_LOG=info
GROQ_API_KEY=gsk_your_key_here
CEREBRAS_API_KEY=csk_your_key_here
EOF
```

## 4. Start infrastructure services

```bash
docker compose up -d --build
```

Services and VRAM allocation:

| Service | Port | Model | GPU VRAM |
|---------|------|-------|----------|
| embeddings (FastAPI + Jina v5 nano) | 7997 | 239M, 768-dim | ~0.5 GB |
| reranker (FastAPI + Jina Reranker v3) | 7998 | 0.6B, listwise | ~1.2 GB |
| docling | 5001 | Document extraction | ~4 GB |
| milvus | 19530 | Vector DB (CPU only) | — |
| **Total** | | | **~6 GB / 24 GB** |

Wait for healthy:

```bash
docker compose ps

# Check individual services:
curl -s http://localhost:7997/health   # embeddings
curl -s http://localhost:7998/health   # reranker
curl -s http://localhost:5001/health   # docling
curl -s http://localhost:9091/healthz  # milvus
```

## 5. Build and run the app

```bash
cargo build --release
cargo run --release
```

Verify:

```bash
curl http://localhost:8080/api/v1/health
```

## 6. Quick test

```bash
# Upload a PDF
curl -X POST http://localhost:8080/api/v1/documents/upload \
  -F "file=@your-document.pdf" \
  -F "collection_name=test"

# Stream a response
curl -N -X POST http://localhost:8080/api/v1/chat-rag/stream \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is this document about?",
    "collection_name": "test",
    "model": "llama-3.3-70b-versatile",
    "provider": "groq"
  }'
```

## Troubleshooting

**Reranker won't start:**
```bash
docker compose logs reranker
docker compose build reranker
docker compose up -d reranker
```

**Check GPU usage:**
```bash
nvidia-smi
```

**Reset everything:**
```bash
docker compose down -v
docker compose up -d --build
```
