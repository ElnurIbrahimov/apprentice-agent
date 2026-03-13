"""API routes for Agent State Machine observability."""

import logging
from fastapi import APIRouter, Query, Depends

from api.auth import require_api_key

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/state-machine", tags=["state-machine"], dependencies=[Depends(require_api_key)])


@router.get("/state")
async def get_state_machine_state():
    """Get current phase, goal, iteration, and elapsed time."""
    try:
        from aura.state_machine import get_agent_state_machine
        sm = get_agent_state_machine()
        return {"success": True, **sm.get_state()}
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/timings")
async def get_state_machine_timings():
    """Get per-phase timing statistics."""
    try:
        from aura.state_machine import get_agent_state_machine
        sm = get_agent_state_machine()
        return {"success": True, "timings": sm.get_timings()}
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/transitions")
async def get_state_machine_transitions(limit: int = Query(default=20, ge=1, le=200)):
    """Get recent phase transition history."""
    try:
        from aura.state_machine import get_agent_state_machine
        sm = get_agent_state_machine()
        return {"success": True, "transitions": sm.get_recent_transitions(limit)}
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.post("/reset-stats")
async def reset_state_machine_stats():
    """Reset timing statistics and transition history."""
    try:
        from aura.state_machine import get_agent_state_machine
        sm = get_agent_state_machine()
        sm.reset_stats()
        return {"success": True, "message": "State machine stats reset"}
    except Exception as e:
        return {"success": False, "error": str(e)}
