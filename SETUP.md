# RustyRAG GPU Setup on Vast.ai

## Prerequisites

- A Vast.ai instance with an NVIDIA GPU (RTX 3000+ series recommended)
- Ubuntu 22.04+ base image

## 1. Install NVIDIA Container Toolkit

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
```

Configure Docker to use the NVIDIA runtime and restart:

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Verify the GPU is visible to Docker:

```bash
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```

You should see your GPU listed in the output.

## 2. Install Rust

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
```

Install build dependencies:

```bash
sudo apt-get install -y pkg-config libssl-dev build-essential
```

## 3. Clone and Build

```bash
git clone https://github.com/AlphaCorp-AI/RustyRAG-GPU.git
cd RustyRAG-GPU
cargo build --release
```

## 4. Configure Environment

```bash
cp .env.gpu.example .env
```

Edit `.env` with your API keys and settings:

```bash
nano .env
```

Key GPU-specific values (already set in `.env.gpu.example`):

```
EMBEDDING_MODEL=jinaai/jina-embeddings-v5-text-small-retrieval
EMBEDDING_DIMENSION=1024
```

## 5. Start GPU Services

```bash
docker compose -f docker-compose.gpu.yml up -d
```

Wait for the services to become healthy (model download + warmup can take a few minutes on first run):

```bash
# Check container status
docker compose -f docker-compose.gpu.yml ps

# Check GPU usage
nvidia-smi

# Check service health
curl http://localhost:7997/health    # TEI embeddings
curl http://localhost:9091/healthz   # Milvus
```

## 6. Run the Application

```bash
./target/release/rustyrag
```

## 7. Test

```bash
curl -N -s -o /dev/null -w "TTFT: %{time_starttransfer}s\nTotal: %{time_total}s\n" \
  -X POST http://localhost:8080/api/v1/chat-rag/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "Are static and dynamic obstacles considered in the obstacle space?", "collection_name": "documents", "provider": "cerebras", "model": "qwen-3-235b-a22b-instruct-2507"}'
```

## Troubleshooting

### `could not select device driver "nvidia" with capabilities: [[gpu]]`

The NVIDIA Container Toolkit is not installed or Docker was not restarted. Re-run step 1.

### `Could not find directory of OpenSSL installation`

Missing build dependencies. Run:

```bash
sudo apt-get install -y pkg-config libssl-dev build-essential
```

### High CPU usage on embeddings container

This is normal during model loading/warmup. The 12 tokenization workers always run on CPU. Verify GPU is being used with `nvidia-smi` — you should see `text-embeddings-router` in the processes list.
