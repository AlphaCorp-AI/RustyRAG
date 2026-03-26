"""Jina Reranker v3 sidecar — TEI-compatible /rerank endpoint."""

import os
import asyncio
from contextlib import asynccontextmanager
from functools import partial

import torch
from fastapi import FastAPI, Response
from pydantic import BaseModel

MODEL_ID = os.getenv("MODEL_ID", "jinaai/jina-reranker-v3")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    from transformers import AutoModel

    model = AutoModel.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        dtype="auto",
    )
    model.to(DEVICE)
    model.eval()

    # Warmup: run a dummy rerank to trigger CUDA kernels / JIT compilation
    if DEVICE == "cuda":
        with torch.inference_mode():
            model.rerank("warmup", ["warmup"], top_n=1)
        torch.cuda.empty_cache()

    yield


app = FastAPI(lifespan=lifespan)


class RerankRequest(BaseModel):
    query: str
    texts: list[str]
    raw_scores: bool = False


class RerankResult(BaseModel):
    index: int
    score: float


def _rerank_sync(query: str, texts: list[str]) -> list[dict]:
    with torch.inference_mode():
        return model.rerank(query, texts)


@app.post("/rerank")
async def rerank(req: RerankRequest) -> list[RerankResult]:
    loop = asyncio.get_running_loop()
    results = await loop.run_in_executor(
        None, partial(_rerank_sync, req.query, req.texts)
    )
    return [
        RerankResult(index=r["index"], score=r["relevance_score"])
        for r in results
    ]


@app.get("/health")
async def health(response: Response):
    if model is None:
        response.status_code = 503
        return {"status": "loading"}
    return {"status": "ok"}
