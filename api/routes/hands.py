"""Autonomous Hands API — CRUD, execution, history, and approval endpoints."""

import asyncio
import copy
import logging
import threading
from typing import Any, Dict, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from api.auth import require_api_key

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/hands", tags=["hands"], dependencies=[Depends(require_api_key)])


# ============================================================================
# Helpers
# ============================================================================

def _get_manager():
    from aura.hands.collector import CollectorHand
    from aura.hands.custom_store import get_custom_hand_store
    from aura.hands.dynamic_hand import DynamicHand
    from aura.hands.guardian import GuardianHand
    from aura.hands.manager import get_hand_manager
    from aura.hands.memory_hand import MemoryHand
    from aura.hands.morning_briefing import MorningBriefingHand
    from aura.hands.researcher import ResearcherHand

    manager = get_hand_manager()
    # Lazy-register built-in Hands on first API access
    if not manager.list_hands():
        manager.register(ResearcherHand())
        manager.register(GuardianHand())
        manager.register(MemoryHand())
        manager.register(CollectorHand())
        manager.register(MorningBriefingHand())
        # Restore custom hands persisted from previous sessions
        store = get_custom_hand_store()
        for config in store.load_all():
            try:
                manager.register(DynamicHand(config))
                logger.debug(f"[Hands API] Restored custom hand: {config.get('name')}")
            except Exception as exc:
                logger.warning(f"[Hands API] Failed to restore custom hand '{config.get('name')}': {exc}")
    return manager


def _get_agent():
    """Get the agent instance for brain/tools access."""
    try:
        from api.utils import get_agent
        return get_agent()
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


class CreateHandRequest(BaseModel):
    description: str = Field(..., min_length=5, max_length=2000)


class FromTemplateRequest(BaseModel):
    template_name: str
    variables: Optional[Dict[str, str]] = None


class HandConfigResponse(BaseModel):
    status: str
    hand: str
    config: dict


# ============================================================================
# Hand templates
# ============================================================================

HAND_TEMPLATES: list[dict] = [
    {
        "name": "daily_news_monitor",
        "description": "Monitor a topic in the news daily",
        "goal": "Find and summarize today's top news on {topic}",
        "search_queries": ["{topic} news today", "latest {topic} developments"],
        "interval_minutes": 1440,
        "idle_only": True,
        "model_preference": "fast",
        "max_tokens": 15000,
        "max_cost_usd": 0.15,
    },
    {
        "name": "research_tracker",
        "description": "Track new research papers on a subject",
        "goal": "Find recent academic papers and breakthroughs in {subject}",
        "search_queries": ["{subject} research papers 2025", "arxiv {subject} latest"],
        "interval_minutes": 720,
        "idle_only": True,
        "model_preference": "reasoning",
        "max_tokens": 25000,
        "max_cost_usd": 0.30,
    },
    {
        "name": "price_watcher",
        "description": "Watch prices or market movements for an asset",
        "goal": "Monitor price and market sentiment for {asset}",
        "search_queries": ["{asset} price today", "{asset} market analysis"],
        "interval_minutes": 480,
        "idle_only": False,
        "model_preference": "fast",
        "max_tokens": 10000,
        "max_cost_usd": 0.10,
    },
    {
        "name": "github_tracker",
        "description": "Track activity and releases in a GitHub repository",
        "goal": "Check for new releases, issues, and PRs in {repo}",
        "search_queries": ["{repo} github releases", "{repo} changelog"],
        "interval_minutes": 360,
        "idle_only": True,
        "model_preference": "fast",
        "max_tokens": 12000,
        "max_cost_usd": 0.12,
    },
]


# ============================================================================
# Endpoints
# ============================================================================

@router.post("/create")
async def create_hand(body: CreateHandRequest) -> HandConfigResponse:
    """Create a custom Hand from a natural language description."""
    from api.services.hand_config_extractor import extract_hand_config
    from api.utils import get_agent
    from aura.hands.custom_store import get_custom_hand_store
    from aura.hands.dynamic_hand import DynamicHand

    agent = get_agent()
    brain = getattr(agent, 'brain', None) if agent else None
    if not brain:
        raise HTTPException(status_code=503, detail="Agent brain not available")

    config = await extract_hand_config(body.description, brain)
    if not config:
        raise HTTPException(status_code=422, detail="Could not extract a valid hand config from the description")

    hand = DynamicHand(config)
    manager = _get_manager()
    manager.register(hand)

    store = get_custom_hand_store()
    store.save(config)

    logger.info(f"[Hands API] Created custom hand: {config['name']}")
    return HandConfigResponse(status="created", hand=config["name"], config=config)


@router.get("/templates")
async def list_templates() -> Dict[str, Any]:
    """Return the list of built-in hand templates."""
    return {"templates": HAND_TEMPLATES, "count": len(HAND_TEMPLATES)}


@router.post("/from-template")
async def create_from_template(body: FromTemplateRequest) -> HandConfigResponse:
    """Create a custom Hand from a named template, applying variable substitutions."""
    from aura.hands.custom_store import get_custom_hand_store
    from aura.hands.dynamic_hand import DynamicHand

    template = next((t for t in HAND_TEMPLATES if t["name"] == body.template_name), None)
    if not template:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown template '{body.template_name}'. Available: {[t['name'] for t in HAND_TEMPLATES]}",
        )

    # Deep-copy so mutations don't affect the template constant
    config = copy.deepcopy(template)
    variables = body.variables or {}

    if variables:
        # Apply substitutions recursively to all string values
        def _apply(obj: Any) -> Any:
            if isinstance(obj, str):
                try:
                    return obj.format_map(variables)
                except KeyError:
                    return obj
            if isinstance(obj, list):
                return [_apply(item) for item in obj]
            if isinstance(obj, dict):
                return {k: _apply(v) for k, v in obj.items()}
            return obj

        config = _apply(config)

        # Derive a unique name from the first variable value if not overridden
        if "name" not in variables:
            first_val = next(iter(variables.values()), "").lower()
            safe = "".join(c if c.isalnum() or c == "_" else "_" for c in first_val)[:20].strip("_")
            if safe:
                config["name"] = f"{template['name']}_{safe}"

    config["is_custom"] = True

    hand = DynamicHand(config)
    manager = _get_manager()
    manager.register(hand)

    store = get_custom_hand_store()
    store.save(config)

    logger.info(f"[Hands API] Created hand from template '{body.template_name}': {config['name']}")
    return HandConfigResponse(status="created", hand=config["name"], config=config)


@router.delete("/{name}")
async def delete_hand(name: str) -> Dict[str, Any]:
    """Delete a custom hand. Only custom (is_custom=True) hands can be deleted."""
    from aura.hands.custom_store import get_custom_hand_store

    manager = _get_manager()
    hand = manager.get_hand(name)
    if not hand:
        raise HTTPException(status_code=404, detail=f"Unknown hand: {name}")

    stats = hand.get_stats()
    if not stats.get("is_custom"):
        raise HTTPException(status_code=403, detail=f"Hand '{name}' is a built-in hand and cannot be deleted")

    # Deactivate before removal
    manager.deactivate(name)

    # Remove from manager's registry
    with manager._lock:
        manager._hands.pop(name, None)

    # Remove from persistent store
    store = get_custom_hand_store()
    store.delete(name)

    logger.info(f"[Hands API] Deleted custom hand: {name}")
    return {"status": "deleted", "hand": name}


@router.get("")
async def list_hands() -> Dict[str, Any]:
    """List all registered Hands with their stats."""
    manager = _get_manager()
    hands = manager.list_hands()
    return {"hands": hands, "count": len(hands)}


@router.get("/history")
async def get_hand_history(
    limit: int = Query(20, ge=1, le=100),
    hand: Optional[str] = Query(None),
) -> Dict[str, Any]:
    """Get recent Hand execution results from the audit chain."""
    try:
        from aura.security.audit_chain import get_audit_chain
        chain = get_audit_chain()
        agent_id = f"hand:{hand}" if hand else None
        entries = chain.search(action_type="hand_complete", agent_id=agent_id, limit=limit)
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


# NOTE: All literal-path GET/POST routes must appear before /{name} to avoid
# shadowing.  The routes above (/create, /templates, /from-template, /history,
# /approvals) are all defined first; /{name} is intentionally last.


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


@router.post("/{name}/pause")
async def pause_hand(name: str) -> Dict[str, Any]:
    """Pause a Hand (suspend scheduling without deactivating)."""
    manager = _get_manager()
    hand = manager._hands.get(name)
    if not hand:
        raise HTTPException(status_code=404, detail=f"Unknown hand: {name}")
    from aura.hands.base import HandState
    if hand.state in (HandState.ACTIVE, HandState.RUNNING):
        hand._state = HandState.PAUSED
        return {"status": "paused", "hand": name}
    raise HTTPException(status_code=400, detail=f"Hand {name} is {hand.state.value}, cannot pause")


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
