"""Status and health check endpoints."""

import asyncio
import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from api.models.schemas import StatusResponse, HealthResponse, MoodState
from api.services.agent_service import agent_service

logger = logging.getLogger(__name__)

# ALMA imports are done lazily inside endpoints to avoid blocking startup

router = APIRouter(prefix="/api", tags=["status"])


class ModelsResponse(BaseModel):
    """Response with available models."""
    local_models: List[str]
    cloud_models: List[str]
    current_model: str


class InitStatus(BaseModel):
    """Agent initialization status."""
    ready: bool
    progress: str
    error: Optional[str] = None


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(status="ok", version="1.0.0")


@router.get("/init", response_model=InitStatus)
async def get_init_status(request: Request) -> InitStatus:
    """Get agent initialization status.

    Returns:
        Current init state (ready, progress, error)
    """
    init_state = getattr(request.app.state, 'init_state', {"ready": False, "progress": "unknown"})
    return InitStatus(
        ready=init_state.get("ready", False),
        progress=init_state.get("progress", "unknown"),
        error=init_state.get("error")
    )


@router.get("/mood", response_model=MoodState)
async def get_mood() -> MoodState:
    """Get ALMA emotional state directly (fast, no agent init required).

    Returns:
        Current mood state from ALMA
    """
    # For now, return test data to verify serialization works
    # TODO: Integrate ALMA properly once import issues are resolved
    return MoodState(
        emotion="curious",
        confidence=75,
        valence=0.4,
        arousal=0.5,
        dominance=0.2,
        emoji="🤔"
    )


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


@router.post("/mood/trigger")
async def trigger_mood(emotion: str, intensity: float = 0.7) -> MoodState:
    """Trigger an emotion in ALMA for testing.

    Args:
        emotion: Emotion name (joy, sadness, anger, fear, surprise, etc.)
        intensity: Emotion intensity (0.0 to 1.0)

    Returns:
        Updated mood state
    """
    # TODO: Integrate ALMA properly
    emoji_map = {
        'joy': '😊', 'happiness': '😊', 'excited': '🤩',
        'sadness': '😢', 'anger': '😠', 'fear': '😨',
        'surprise': '😲', 'curious': '🤔', 'neutral': '😐',
    }
    return MoodState(
        emotion=emotion,
        confidence=int(intensity * 100),
        valence=0.5 if emotion in ['joy', 'happiness', 'excited', 'curious'] else -0.3,
        arousal=intensity * 0.8,
        dominance=0.2,
        emoji=emoji_map.get(emotion, '🤖')
    )


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
