"""Multi-model comparison endpoint — run a query on multiple models in parallel."""

import asyncio
import logging
import time
from typing import List, Optional

from fastapi import APIRouter
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/compare", tags=["compare"])


class CompareRequest(BaseModel):
    message: str
    models: Optional[List[str]] = None  # Model IDs to compare; defaults to 3 representative models


class ModelResult(BaseModel):
    model: str
    response: str
    time_ms: int
    error: Optional[str] = None


class CompareResponse(BaseModel):
    results: List[ModelResult]
    query: str


DEFAULT_COMPARE_MODELS = [
    "gemini-3-flash-preview:cloud",
    "qwen3.5:397b-cloud",
    "cogito-2.1:671b-cloud",
]


@router.post("", response_model=CompareResponse)
async def compare_models(request: CompareRequest):
    """Run a query on 2-3 models in parallel and return side-by-side responses."""
    from api.services.agent_service import agent_service

    models = request.models or DEFAULT_COMPARE_MODELS
    # Limit to 4 models max to avoid excessive API calls
    models = models[:4]

    loop = asyncio.get_running_loop()

    async def run_model(model_id: str) -> ModelResult:
        start = time.time()
        try:
            def _think():
                agent = agent_service.agent
                brain = agent.brain
                # Temporarily use a per-call approach with the brain
                with agent_service._agent_lock:
                    brain.set_model_override(model_id)
                response = brain.think(request.message, use_history=False)
                return response

            response = await loop.run_in_executor(None, _think)
            elapsed = int((time.time() - start) * 1000)
            return ModelResult(model=model_id, response=response, time_ms=elapsed)
        except Exception as e:
            elapsed = int((time.time() - start) * 1000)
            logger.error(f"[Compare] Model {model_id} failed: {e}")
            return ModelResult(model=model_id, response="", time_ms=elapsed, error=str(e))

    # Run all models in parallel
    tasks = [run_model(m) for m in models]
    results = await asyncio.gather(*tasks, return_exceptions=False)

    # Restore to auto mode
    try:
        agent_service.agent.brain.set_model_override(None)
    except Exception:
        pass

    return CompareResponse(results=list(results), query=request.message)
