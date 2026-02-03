"""Status and health check endpoints."""

import asyncio
import logging
from typing import List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from api.models.schemas import StatusResponse, HealthResponse, MoodState
from api.services.agent_service import agent_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["status"])


class ModelsResponse(BaseModel):
    """Response with available models."""
    local_models: List[str]
    cloud_models: List[str]
    current_model: str


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(status="ok", version="1.0.0")


@router.get("/status", response_model=StatusResponse)
async def get_status() -> StatusResponse:
    """Get agent status including mood and stats.

    Returns:
        Status response with agent state
    """
    try:
        loop = asyncio.get_event_loop()
        status = await loop.run_in_executor(None, agent_service.get_status)

        mood = status.get("mood")
        if mood and isinstance(mood, dict):
            mood = MoodState(**mood)

        return StatusResponse(
            online=status.get("online", True),
            model=status.get("model", "unknown"),
            aura_enabled=status.get("aura_enabled", False),
            mood=mood,
            memory_count=status.get("memory_count", 0),
            query_count=status.get("query_count", 0),
            last_model_used=status.get("last_model_used")
        )

    except Exception as e:
        logger.error(f"[Status] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models", response_model=ModelsResponse)
async def get_models() -> ModelsResponse:
    """Get available models (local and cloud).

    Returns:
        List of available local and cloud models
    """
    try:
        loop = asyncio.get_event_loop()
        models = await loop.run_in_executor(None, agent_service.get_available_models)

        return ModelsResponse(
            local_models=models.get("local", []),
            cloud_models=models.get("cloud", []),
            current_model=models.get("current", "auto")
        )

    except Exception as e:
        logger.error(f"[Models] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
