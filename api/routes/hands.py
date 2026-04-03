"""Autonomous Hands API — CRUD, execution, history, and approval endpoints."""

import asyncio
import logging
import threading
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from pydantic import BaseModel

from api.auth import require_api_key

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/hands", tags=["hands"], dependencies=[Depends(require_api_key)])


# ============================================================================
# Helpers
# ============================================================================

def _get_manager():
    from aura.hands.manager import get_hand_manager
    from aura.hands.researcher import ResearcherHand
    from aura.hands.guardian import GuardianHand
    from aura.hands.memory_hand import MemoryHand

    manager = get_hand_manager()
    # Lazy-register built-in Hands on first API access
    if not manager.list_hands():
        manager.register(ResearcherHand())
        manager.register(GuardianHand())
        manager.register(MemoryHand())
    return manager


def _get_agent():
    """Get the agent instance for brain/tools access."""
    try:
        from api.services.agent_service import get_agent_service
        svc = get_agent_service()
        return getattr(svc, '_agent', None) or getattr(svc, 'agent', None)
    except Exception:
        pass
    # Fallback: try the singleton agent
    try:
        from aura.agent import _agent_instance
        return _agent_instance
    except Exception:
        return None


# ============================================================================
# Response Models
# ============================================================================

class HandSummary(BaseModel):
    name: str
    description: str
    state: str
    total_runs: int
    total_cost: float
    consecutive_failures: int
    last_run: Optional[str] = None
    model_preference: str
    idle_only: bool
    trigger_on_drive: Optional[str] = None


class HandRunResult(BaseModel):
    hand: str
    success: bool
    summary: str
    iterations: int = 0
    tokens_used: int = 0
    cost_usd: float = 0.0
    duration_seconds: float = 0.0
    error: Optional[str] = None


class ApprovalAction(BaseModel):
    approved: bool


# ============================================================================
# Endpoints
# ============================================================================

@router.get("")
async def list_hands() -> Dict[str, Any]:
    """List all registered Hands with their stats."""
    manager = _get_manager()
    hands = manager.list_hands()
    return {"hands": hands, "count": len(hands)}


@router.get("/history")
async def get_hand_history(
    limit: int = Query(20, ge=1, le=100),
) -> Dict[str, Any]:
    """Get recent Hand execution results from the audit chain."""
    try:
        from aura.security.audit_chain import get_audit_chain
        chain = get_audit_chain()
        entries = chain.search(action_type="hand_complete", limit=limit)
        return {"history": entries, "count": len(entries)}
    except Exception as e:
        logger.debug(f"[Hands API] History lookup failed: {e}")
        return {"history": [], "count": 0}


@router.get("/approvals")
async def get_pending_approvals() -> Dict[str, Any]:
    """Get pending approval requests from Hands."""
    manager = _get_manager()
    pending = manager.get_pending_approvals()
    return {"approvals": pending, "count": len(pending)}


@router.get("/{name}")
async def get_hand_status(name: str) -> Dict[str, Any]:
    """Get detailed status for a specific Hand."""
    manager = _get_manager()
    hand = manager.get_hand(name)
    if not hand:
        raise HTTPException(status_code=404, detail=f"Unknown hand: {name}")
    return hand.get_stats()


@router.post("/{name}/run")
async def run_hand(name: str, background_tasks: BackgroundTasks) -> Dict[str, Any]:
    """Trigger a Hand to run immediately."""
    manager = _get_manager()
    hand = manager.get_hand(name)
    if not hand:
        raise HTTPException(status_code=404, detail=f"Unknown hand: {name}")

    from aura.hands.base import HandState
    if hand.state == HandState.RUNNING:
        raise HTTPException(status_code=409, detail=f"Hand '{name}' is already running")

    agent = _get_agent()
    brain = getattr(agent, 'brain', None) if agent else None
    tools = getattr(agent, 'tools', {}) if agent else {}

    if not brain:
        raise HTTPException(status_code=503, detail="Agent brain not available")

    # Run in background thread
    def _run():
        try:
            result = asyncio.run(manager.run_hand(name, brain, tools))
            logger.info(f"[Hands API] Hand '{name}' completed: {result.success}")
        except Exception as e:
            logger.error(f"[Hands API] Hand '{name}' execution failed: {e}")

    thread = threading.Thread(target=_run, daemon=True, name=f"hand-api-{name}")
    thread.start()

    return {"status": "started", "hand": name, "message": f"Hand '{name}' is now running"}


@router.post("/{name}/activate")
async def activate_hand(name: str) -> Dict[str, Any]:
    """Activate a Hand for scheduled execution."""
    manager = _get_manager()
    if manager.activate(name):
        return {"status": "activated", "hand": name}
    raise HTTPException(status_code=404, detail=f"Unknown or already running hand: {name}")


@router.post("/{name}/deactivate")
async def deactivate_hand(name: str) -> Dict[str, Any]:
    """Deactivate a Hand (stop scheduling it)."""
    manager = _get_manager()
    if manager.deactivate(name):
        return {"status": "deactivated", "hand": name}
    raise HTTPException(status_code=404, detail=f"Unknown hand: {name}")


@router.post("/{name}/approve")
async def approve_hand_action(name: str, body: ApprovalAction) -> Dict[str, Any]:
    """Approve or deny a pending Hand action."""
    manager = _get_manager()
    # Find the pending approval for this hand
    pending = manager.get_pending_approvals()
    request_id = None
    for req in pending:
        if req.get("hand_name") == name:
            request_id = req.get("request_id")
            break

    if not request_id:
        raise HTTPException(status_code=404, detail=f"No pending approval for hand: {name}")

    manager.resolve_approval(request_id, body.approved)
    action = "approved" if body.approved else "denied"
    return {"status": action, "hand": name, "request_id": request_id}
