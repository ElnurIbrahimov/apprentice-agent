"""Routing API — feedback signals, stats, and conversation profiles."""

import logging
from typing import Optional
from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from api.auth import require_api_key

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/routing", tags=["routing"], dependencies=[Depends(require_api_key)])


class FeedbackRequest(BaseModel):
    signal: str = Field(..., max_length=50)  # regeneration, model_switch, thumbs_up, thumbs_down, abort
    model: str = Field(..., max_length=200)
    conversation_id: Optional[str] = Field(None, max_length=200)
    task_dimensions: Optional[dict] = None
    switched_to: Optional[str] = Field(None, max_length=200)


@router.post("/feedback")
async def submit_feedback(body: FeedbackRequest):
    """Record a feedback signal for the learning loop."""
    try:
        from aura.routing.router import get_router
        from aura.routing.learning import process_feedback

        r = get_router()

        # Get task dimensions from conversation if not provided
        task_dims = body.task_dimensions or {
            "code": 0.5, "reason": 0.5, "speed": 0.5,
            "context": 0.0, "quality": 0.5, "vision": 0.0,
        }

        # Update learning
        process_feedback(
            body.signal, body.model, task_dims, r.profiles,
            switched_to=body.switched_to,
        )

        # Update conversation tracker
        if body.conversation_id:
            r.conversations.record_feedback(
                body.conversation_id, body.signal, model=body.switched_to,
            )

        return {"ok": True, "signal": body.signal}
    except Exception as e:
        logger.error(f"[Routing] Feedback failed: {e}")
        return {"ok": False, "error": str(e)}


@router.get("/stats")
async def get_routing_stats():
    """Return current model profiles + learning stats."""
    try:
        from aura.routing.router import get_router
        r = get_router()
        return {
            "profiles": r.profiles.all_profiles(),
            "total_models": len(r.profiles.all_profiles()),
        }
    except Exception as e:
        return {"error": str(e)}


@router.get("/conversation/{conversation_id}")
async def get_conversation_profile(conversation_id: str):
    """Return conversation routing profile."""
    try:
        from aura.routing.router import get_router
        r = get_router()
        p = r.conversations.get_profile(conversation_id)
        return {
            "conversation_id": conversation_id,
            "turn_count": p.turn_count,
            "total_tokens": p.total_tokens,
            "in_code_mode": p.in_code_mode,
            "complexity_trend": p.complexity_trend,
            "last_model": p.last_model,
            "regen_count": p.regen_count,
            "model_switches": p.model_switches,
            "models_used": p.models_used,
        }
    except Exception as e:
        return {"error": str(e)}
