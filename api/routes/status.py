"""Status and health check endpoints."""

import asyncio
import logging
import random
import time
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from api.models.schemas import StatusResponse, HealthResponse, MoodState
from api.services.agent_service import agent_service

logger = logging.getLogger(__name__)

# ALMA imports are done lazily inside endpoints to avoid blocking startup

router = APIRouter(prefix="/api", tags=["status"])


# ============================================================================
# AURA "Consideration" State - Makes AURA feel alive by showing deliberation
# ============================================================================

class ConsiderationState:
    """Tracks AURA's internal deliberation about whether to speak."""

    def __init__(self):
        self.is_considering = False
        self.decided_against = False
        self.consideration_topic: Optional[str] = None
        self.last_consideration_time = 0.0
        self.last_decision_time = 0.0
        self.consideration_count = 0
        self.declined_count = 0

        # Possible things AURA might consider mentioning
        self.consideration_topics = [
            "a pattern in your recent questions",
            "something from our earlier conversation",
            "a connection to your interests",
            "an observation about your workflow",
            "a thought about the current topic",
            "a memory that seemed relevant",
            "an insight from recent context",
            "something that might be helpful",
        ]

    def maybe_trigger_consideration(self) -> bool:
        """Randomly decide whether to start a new consideration.

        Returns True if a new consideration was triggered.
        """
        now = time.time()

        # Don't consider if already considering or recently decided
        if self.is_considering:
            return False
        if now - self.last_decision_time < 10:  # Min 10s between decisions
            return False
        if now - self.last_consideration_time < 20:  # Min 20s between considerations
            return False

        # 15% chance to start considering when polled
        if random.random() < 0.15:
            self.is_considering = True
            self.decided_against = False
            self.consideration_topic = random.choice(self.consideration_topics)
            self.last_consideration_time = now
            self.consideration_count += 1

            # Schedule the decision (will be resolved on next poll)
            return True

        return False

    def resolve_consideration(self) -> bool:
        """Resolve an ongoing consideration - decide whether to speak.

        Returns True if decided NOT to speak (the interesting case).
        """
        if not self.is_considering:
            return False

        now = time.time()
        consideration_duration = now - self.last_consideration_time

        # Need at least 2 seconds of "thinking"
        if consideration_duration < 2.0:
            return False

        # After 2-5 seconds, make a decision
        # 75% chance to decide NOT to speak (makes it feel more selective)
        self.is_considering = False
        self.last_decision_time = now

        if random.random() < 0.75:
            self.decided_against = True
            self.declined_count += 1
            return True
        else:
            # Would have spoken - but we don't actually generate content here
            # This just means the "decided against" won't show
            self.decided_against = False
            return False

    def clear_decided_against(self):
        """Clear the decided_against flag after frontend has shown it."""
        self.decided_against = False

    def get_state(self) -> dict:
        """Get current consideration state for API."""
        return {
            "is_considering": self.is_considering,
            "decided_against": self.decided_against,
            "topic": self.consideration_topic if (self.is_considering or self.decided_against) else None,
            "consideration_count": self.consideration_count,
            "declined_count": self.declined_count,
        }


# Global consideration state
_consideration_state = ConsiderationState()


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


@router.get("/alma/state")
async def get_alma_state():
    """Get full ALMA emotional state including neuromodulators and active emotions.

    Returns:
        Complete ALMA state with all emotional data
    """
    try:
        # Import ALMA lazily
        from apprentice_agent.emotion.alma_engine import alma_engine

        if alma_engine:
            state = alma_engine.get_emotional_state()
            if state:
                return {
                    "available": True,
                    "dominant_emotion": state.get("dominant_emotion", "neutral"),
                    "intensity": state.get("intensity", 0.5),
                    "pad": state.get("pad", {"pleasure": 0, "arousal": 0, "dominance": 0}),
                    "mood": state.get("mood", {}),
                    "active_emotions": state.get("active_emotions", []),
                    "neuromodulators": state.get("neuromodulators", {
                        "dopamine": 0.5,
                        "serotonin": 0.5,
                        "norepinephrine": 0.5,
                        "oxytocin": 0.5
                    }),
                    "personality": state.get("personality", {
                        "openness": 0.8,
                        "conscientiousness": 0.7,
                        "extraversion": 0.5,
                        "agreeableness": 0.75,
                        "neuroticism": 0.25
                    }),
                    "timestamp": state.get("timestamp", 0)
                }
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"[ALMA State] Error: {e}")

    # Return default state if ALMA not available
    return {
        "available": False,
        "dominant_emotion": "neutral",
        "intensity": 0.5,
        "pad": {"pleasure": 0, "arousal": 0, "dominance": 0},
        "mood": {"label": "neutral", "intensity": 0.5},
        "active_emotions": [],
        "neuromodulators": {
            "dopamine": 0.5,
            "serotonin": 0.5,
            "norepinephrine": 0.5,
            "oxytocin": 0.5
        },
        "personality": {
            "openness": 0.8,
            "conscientiousness": 0.7,
            "extraversion": 0.5,
            "agreeableness": 0.75,
            "neuroticism": 0.25
        },
        "timestamp": 0
    }


@router.get("/aura/consideration")
async def get_consideration_state():
    """Get AURA's current consideration state.

    This endpoint is polled by the frontend to show when AURA is
    "thinking about saying something" and when it "decides not to speak".

    Returns:
        Consideration state with is_considering, decided_against, and topic
    """
    global _consideration_state

    # First, try to resolve any ongoing consideration
    _consideration_state.resolve_consideration()

    # Then, maybe trigger a new consideration
    _consideration_state.maybe_trigger_consideration()

    state = _consideration_state.get_state()

    # If we just showed "decided against", clear it for next poll
    # (frontend gets one chance to see it)
    if state["decided_against"]:
        # Keep it for this response, clear after
        asyncio.get_event_loop().call_later(0.1, _consideration_state.clear_decided_against)

    return state


@router.post("/aura/consideration/trigger")
async def trigger_consideration(topic: Optional[str] = None):
    """Manually trigger a consideration (for testing).

    Args:
        topic: Optional custom topic to consider
    """
    global _consideration_state

    _consideration_state.is_considering = True
    _consideration_state.decided_against = False
    _consideration_state.consideration_topic = topic or random.choice(_consideration_state.consideration_topics)
    _consideration_state.last_consideration_time = time.time()
    _consideration_state.consideration_count += 1

    return {"status": "considering", "topic": _consideration_state.consideration_topic}


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
