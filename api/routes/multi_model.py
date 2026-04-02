"""Multi-model comparison endpoint — run a query on multiple models in parallel."""

import asyncio
import logging
import os
import time
from typing import List, Optional

import httpx
from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from api.auth import require_api_key

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/compare", tags=["compare"], dependencies=[Depends(require_api_key)])

OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL") or os.getenv("OLLAMA_HOST", "http://localhost:11434")

def _get_default_compare_models():
    """Get default models from Config to avoid hardcoding."""
    try:
        from aura.config import Config
        return [Config.MODEL_CODE, Config.MODEL_THINK, Config.MODEL_REASON]
    except Exception:
        return ["minimax-m2.7:cloud", "qwen3.5:397b-cloud", "kimi-k2.5:cloud"]


class CompareRequest(BaseModel):
    message: str = Field(..., max_length=8000)
    models: Optional[List[str]] = None  # Ollama model names; defaults to top 3 cloud


class ModelResult(BaseModel):
    model: str
    response: str
    elapsed_ms: int
    error: Optional[str] = None


class CompareResponse(BaseModel):
    results: List[ModelResult]
    fastest: str
    query: str


async def _query_model(client: httpx.AsyncClient, model: str, prompt: str) -> ModelResult:
    """POST to Ollama /api/generate for one model, non-streaming."""
    start = time.monotonic()
    try:
        r = await client.post(
            f"{OLLAMA_BASE}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=60.0,
        )
        r.raise_for_status()
        data = r.json()
        response_text = data.get("response", "")
        elapsed = int((time.monotonic() - start) * 1000)
        return ModelResult(model=model, response=response_text, elapsed_ms=elapsed)
    except Exception as exc:
        elapsed = int((time.monotonic() - start) * 1000)
        logger.error("[Compare] Model %s failed: %s", model, exc)
        return ModelResult(model=model, response="", elapsed_ms=elapsed, error=str(exc))


@router.post("", response_model=CompareResponse)
async def compare_models(request: CompareRequest):
    """Run a prompt on multiple Ollama models in parallel and return side-by-side results."""
    models = request.models if request.models else _get_default_compare_models()
    # Cap at 6 to avoid hammering the bridge
    models = models[:6]
    # Validate model name format
    import re
    _model_re = re.compile(r'^[a-zA-Z0-9._:\-/]{1,128}$')
    for m in models:
        if not _model_re.match(m):
            raise HTTPException(400, f"Invalid model name: {m}")

    async with httpx.AsyncClient() as client:
        tasks = [_query_model(client, m, request.message) for m in models]
        results: List[ModelResult] = await asyncio.gather(*tasks)

    # Sort by elapsed_ms (fastest first); errors sort last
    results.sort(key=lambda r: (r.error is not None, r.elapsed_ms))

    # Pick fastest successful result; fall back to first entry
    fastest = next((r.model for r in results if not r.error), results[0].model)

    return CompareResponse(results=results, fastest=fastest, query=request.message)
