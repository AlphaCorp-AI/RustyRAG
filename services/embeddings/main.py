"""Jina v5 nano embedding sidecar — OpenAI-compatible /v1/embeddings endpoint."""

import os
import asyncio
from contextlib import asynccontextmanager
from functools import partial

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
torch.set_float32_matmul_precision("high")
from fastapi import FastAPI, Response
from pydantic import BaseModel

MODEL_ID = os.getenv("MODEL_ID", "jinaai/jina-embeddings-v5-text-nano-retrieval")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = None
tokenizer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    )
    model.to(DEVICE)
    model.eval()
    if DEVICE == "cuda":
        model = torch.compile(model)

    # Warmup — triggers torch.compile tracing + CUDA kernel gen
    if DEVICE == "cuda":
        with torch.inference_mode():
            dummy = tokenizer(["warmup"], padding=True, truncation=True, max_length=64, return_tensors="pt").to(DEVICE)
            model(**dummy)
        torch.cuda.empty_cache()

    yield


app = FastAPI(lifespan=lifespan)


class EmbeddingRequest(BaseModel):
    model: str = ""
    input: list[str] | str = []
    encoding_format: str = "float"
    embedding_type: str | None = None
    task: str | None = None


def _embed_sync(texts: list[str]) -> list[list[float]]:
    with torch.inference_mode():
        encoded = tokenizer(texts, padding=True, truncation=True, max_length=8192, return_tensors="pt").to(DEVICE)
        outputs = model(**encoded)
        # Last token pooling (matching jina v5 retrieval config)
        attention_mask = encoded["attention_mask"]
        last_idx = attention_mask.sum(dim=1) - 1
        embeddings = outputs.last_hidden_state[torch.arange(len(texts), device=DEVICE), last_idx]
        # Normalize
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu().float().tolist()


@app.post("/v1/embeddings")
async def embeddings(req: EmbeddingRequest):
    texts = [req.input] if isinstance(req.input, str) else req.input
    if not texts:
        return {"object": "list", "data": [], "model": MODEL_ID, "usage": {"prompt_tokens": 0, "total_tokens": 0}}

    loop = asyncio.get_running_loop()
    vecs = await loop.run_in_executor(None, partial(_embed_sync, texts))

    data = [
        {"object": "embedding", "embedding": vec, "index": i}
        for i, vec in enumerate(vecs)
    ]
    return {
        "object": "list",
        "data": data,
        "model": MODEL_ID,
        "usage": {"prompt_tokens": 0, "total_tokens": 0},
    }


@app.get("/health")
async def health(response: Response):
    if model is None:
        response.status_code = 503
        return {"status": "loading"}
    return {"status": "ok"}
