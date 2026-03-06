"""Status and health check endpoints."""

import asyncio
import logging
import random
import time
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from api.models.schemas import StatusResponse, HealthResponse, MoodState

logger = logging.getLogger(__name__)

# Lazy import to avoid blocking event loop at module load
def _get_agent_service():
    """Get agent_service with lazy loading."""
    from api.services.agent_service import agent_service
    return agent_service

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
async def get_init_status(request: Request):
    """Get agent initialization status."""
    try:
        from api.services.agent_service import agent_service
        if agent_service.is_ready:
            return {"ready": True, "progress": "complete"}
        elif getattr(agent_service, '_initializing', False):
            return {"ready": False, "progress": "initializing"}
        else:
            return {"ready": False, "progress": "not_started"}
    except Exception as e:
        return {"ready": False, "progress": f"error: {str(e)[:100]}"}


@router.get("/status", response_model=StatusResponse)
async def get_status() -> StatusResponse:
    """Get agent status including mood and stats.

    Returns:
        Status response with agent state
    """
    try:
        svc = _get_agent_service()

        # If agent isn't ready yet, return a lightweight status
        if not svc.is_ready:
            return StatusResponse(
                online=True,
                model="initializing...",
                aura_enabled=False,
                mood=MoodState(
                    emotion='neutral', confidence=50,
                    valence=0.3, arousal=0.1, dominance=0.3, emoji='🔄'
                ),
                memory_count=0,
                query_count=0,
                last_model_used=None
            )

        loop = asyncio.get_running_loop()
        status = await asyncio.wait_for(
            loop.run_in_executor(None, svc.get_status),
            timeout=10.0
        )

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

    except asyncio.TimeoutError:
        logger.warning("[Status] Timed out getting agent status")
        return StatusResponse(
            online=True, model="loading...", aura_enabled=False,
            mood=MoodState(emotion='neutral', confidence=50, valence=0.0, arousal=0.0, dominance=0.0, emoji='🔄'),
            memory_count=0, query_count=0, last_model_used=None
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
    try:
        from aura.emotion.alma_engine import alma_engine, trigger_emotion
        trigger_emotion(emotion, intensity=intensity)
        # Return updated state
        state = alma_engine.get_emotional_state()
        pad = state.get("pad", {})
        return MoodState(
            emotion=state.get("dominant_emotion", "neutral"),
            confidence=int(state.get("intensity", 0.5) * 100),
            valence=pad.get("pleasure", 0.0),
            arousal=pad.get("arousal", 0.0),
            dominance=pad.get("dominance", 0.0),
        )
    except ImportError:
        raise HTTPException(status_code=503, detail="ALMA engine not available")
    except Exception as e:
        logger.warning(f"[API] Mood trigger failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to trigger emotion")


@router.get("/alma/state")
async def get_alma_state():
    """Get full ALMA emotional state including neuromodulators and active emotions.

    Returns:
        Complete ALMA state with all emotional data
    """
    try:
        # Import ALMA lazily
        from aura.emotion.alma_engine import alma_engine

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
        asyncio.get_running_loop().call_later(0.1, _consideration_state.clear_decided_against)

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


# ============================================================================
# AURA Personality Editor - Adjust OCEAN traits
# ============================================================================

class PersonalityUpdate(BaseModel):
    """Request model for updating personality traits."""
    openness: Optional[float] = None
    conscientiousness: Optional[float] = None
    extraversion: Optional[float] = None
    agreeableness: Optional[float] = None
    neuroticism: Optional[float] = None


@router.get("/alma/personality")
async def get_personality():
    """Get AURA's current personality traits (OCEAN model).

    Returns:
        Personality traits with descriptions
    """
    try:
        from aura.emotion.alma_engine import alma_engine

        if alma_engine:
            personality = alma_engine.personality.to_dict()
            return {
                "available": True,
                "traits": personality,
                "descriptions": {
                    "openness": {
                        "name": "Openness",
                        "low": "Conventional, practical",
                        "high": "Creative, curious",
                        "description": "Openness to new experiences and ideas"
                    },
                    "conscientiousness": {
                        "name": "Conscientiousness",
                        "low": "Flexible, spontaneous",
                        "high": "Organized, reliable",
                        "description": "Tendency to be organized and dependable"
                    },
                    "extraversion": {
                        "name": "Extraversion",
                        "low": "Reserved, reflective",
                        "high": "Outgoing, energetic",
                        "description": "Energy derived from social interaction"
                    },
                    "agreeableness": {
                        "name": "Agreeableness",
                        "low": "Analytical, detached",
                        "high": "Warm, empathetic",
                        "description": "Tendency to be compassionate and cooperative"
                    },
                    "neuroticism": {
                        "name": "Emotional Sensitivity",
                        "low": "Calm, stable",
                        "high": "Sensitive, reactive",
                        "description": "Tendency to experience negative emotions"
                    }
                }
            }
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"[Personality] Error: {e}")

    # Default response if ALMA not available
    return {
        "available": False,
        "traits": {
            "openness": 0.8,
            "conscientiousness": 0.7,
            "extraversion": 0.5,
            "agreeableness": 0.75,
            "neuroticism": 0.25
        },
        "descriptions": {}
    }


@router.post("/alma/personality")
async def update_personality(update: PersonalityUpdate):
    """Update AURA's personality traits.

    Args:
        update: Personality traits to update (only specified traits are changed)

    Returns:
        Updated personality and the effect on baseline mood
    """
    try:
        from aura.emotion.alma_engine import alma_engine, PersonalityProfile

        if alma_engine:
            # Get current values
            current = alma_engine.personality

            # Update only provided values (clamp to 0-1)
            new_openness = max(0, min(1, update.openness)) if update.openness is not None else current.openness
            new_conscientiousness = max(0, min(1, update.conscientiousness)) if update.conscientiousness is not None else current.conscientiousness
            new_extraversion = max(0, min(1, update.extraversion)) if update.extraversion is not None else current.extraversion
            new_agreeableness = max(0, min(1, update.agreeableness)) if update.agreeableness is not None else current.agreeableness
            new_neuroticism = max(0, min(1, update.neuroticism)) if update.neuroticism is not None else current.neuroticism

            # Create new personality profile
            alma_engine.personality = PersonalityProfile(
                openness=new_openness,
                conscientiousness=new_conscientiousness,
                extraversion=new_extraversion,
                agreeableness=new_agreeableness,
                neuroticism=new_neuroticism
            )

            # Update mood baseline
            new_baseline = alma_engine.personality.get_baseline_mood()

            # Save state
            alma_engine._save_state()

            return {
                "success": True,
                "traits": alma_engine.personality.to_dict(),
                "baseline_mood": {
                    "pleasure": new_baseline.pleasure,
                    "arousal": new_baseline.arousal,
                    "dominance": new_baseline.dominance
                }
            }

    except ImportError:
        pass
    except Exception as e:
        logger.error(f"[Personality Update] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    raise HTTPException(status_code=503, detail="ALMA not available")


@router.post("/alma/personality/reset")
async def reset_personality():
    """Reset AURA's personality to default values."""
    try:
        from aura.emotion.alma_engine import alma_engine, PersonalityProfile

        if alma_engine:
            alma_engine.personality = PersonalityProfile.aura_default()
            alma_engine._save_state()

            return {
                "success": True,
                "traits": alma_engine.personality.to_dict(),
                "message": "Personality reset to AURA defaults"
            }

    except ImportError:
        pass
    except Exception as e:
        logger.error(f"[Personality Reset] Error: {e}")

    raise HTTPException(status_code=503, detail="ALMA not available")


@router.get("/models", response_model=ModelsResponse)
async def get_models() -> ModelsResponse:
    """Get available models (local and cloud).

    Returns:
        List of available local and cloud models
    """
    try:
        svc = _get_agent_service()

        # If agent isn't ready, return empty lists
        if not svc.is_ready:
            return ModelsResponse(
                local_models=[], cloud_models=[], current_model="initializing..."
            )

        loop = asyncio.get_running_loop()
        models = await asyncio.wait_for(
            loop.run_in_executor(None, svc.get_available_models),
            timeout=10.0
        )

        return ModelsResponse(
            local_models=models.get("local", []),
            cloud_models=models.get("cloud", []),
            current_model=models.get("current", "auto")
        )

    except asyncio.TimeoutError:
        logger.warning("[Models] Timed out getting models")
        return ModelsResponse(local_models=[], cloud_models=[], current_model="loading...")
    except Exception as e:
        logger.error(f"[Models] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
