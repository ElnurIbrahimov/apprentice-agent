"""API endpoints for the Multi-Agent System."""

import asyncio
import logging
import re
import threading
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from api.auth import require_api_key
from api.utils import get_agent_service as _get_agent_service
from api.utils import safe_error_detail

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/multi-agent", tags=["multi-agent"], dependencies=[Depends(require_api_key)])


# ============================================================================
# Request/Response Models
# ============================================================================

class MultiAgentChatRequest(BaseModel):
    message: str = Field(..., max_length=32_000)
    context: Optional[Dict[str, Any]] = None


class MultiAgentChatResponse(BaseModel):
    response: str
    agents_used: List[str]
    routing_mode: str
    confidence: float


class RoutePreviewRequest(BaseModel):
    query: str = Field(..., max_length=32_000)


class RoutePreviewResponse(BaseModel):
    query: str
    selected_agents: List[str]
    mode: str
    reasoning: str
    confidence: float
    all_scores: Dict[str, float]


class AgentInfo(BaseModel):
    name: str
    description: str
    tools: List[str]
    triggers: List[str]


class MultiAgentStatusResponse(BaseModel):
    enabled: bool
    specialists: List[str]
    specialist_details: Dict[str, AgentInfo]
    conversation_turns: int


# ============================================================================
# Per-Session Orchestrator
# ============================================================================

_orchestrators: dict[str, object] = {}
_orch_lock = threading.Lock()


_SESSION_ID_PATTERN = re.compile(r'^[a-zA-Z0-9_-]{1,64}$')


def get_orchestrator(session_id: str):
    """Get or create the multi-agent orchestrator for a session."""
    # Validate session_id format
    if not _SESSION_ID_PATTERN.match(session_id):
        raise HTTPException(
            status_code=400,
            detail="Invalid session_id format. Must match ^[a-zA-Z0-9_-]{1,64}$"
        )

    with _orch_lock:
        if session_id not in _orchestrators:
            # LRU eviction: drop oldest session when at cap
            _MAX_SESSIONS = 100
            if len(_orchestrators) >= _MAX_SESSIONS:
                oldest_key = next(iter(_orchestrators))
                del _orchestrators[oldest_key]
                logger.info(f"[MultiAgent] Evicted oldest session: {oldest_key}")

            try:
                from aura.multi_agent import MultiAgentOrchestrator

                agent = _get_agent_service().agent

                # Create LLM function wrapper
                def llm_func(system_prompt: str, user_message: str) -> str:
                    return agent.brain.think(user_message, system_prompt=system_prompt)

                # Initialize orchestrator with agent's tools
                _orchestrators[session_id] = MultiAgentOrchestrator(
                    tool_registry=agent.tools,
                    llm_func=llm_func
                )
                logger.info(f"[MultiAgent] Orchestrator initialized for session={session_id}")

            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"[MultiAgent] Failed to initialize orchestrator: {e}")
                raise

        return _orchestrators[session_id]


# ============================================================================
# Endpoints
# ============================================================================

@router.get("/status", response_model=MultiAgentStatusResponse)
async def get_multi_agent_status(session_id: str = Query(default="default")):
    """Get multi-agent system status."""
    try:
        orchestrator = get_orchestrator(session_id)
        status = orchestrator.get_status()

        specialist_details = {}
        for name, details in status.get("specialist_details", {}).items():
            specialist_details[name] = AgentInfo(
                name=name,
                description=details.get("description", ""),
                tools=details.get("tools", []),
                triggers=details.get("triggers", [])
            )

        return MultiAgentStatusResponse(
            enabled=True,
            specialists=status.get("specialists", []),
            specialist_details=specialist_details,
            conversation_turns=status.get("conversation_turns", 0)
        )
    except Exception as e:
        logger.error(f"[MultiAgent] Status error: {e}")
        return MultiAgentStatusResponse(
            enabled=False,
            specialists=[],
            specialist_details={},
            conversation_turns=0
        )


@router.get("/agents")
async def list_agents(session_id: str = Query(default="default")):
    """List available specialist agents."""
    try:
        orchestrator = get_orchestrator(session_id)
        status = orchestrator.get_status()

        agents = []
        for name, details in status.get("specialist_details", {}).items():
            agents.append({
                "name": name,
                "description": details.get("description", ""),
                "tools": details.get("tools", []),
                "triggers": details.get("triggers", [])[:5]
            })

        return {"agents": agents, "count": len(agents)}
    except Exception as e:
        return {"agents": [], "count": 0, "error": safe_error_detail(e)}


@router.post("/chat", response_model=MultiAgentChatResponse)
async def multi_agent_chat(request: MultiAgentChatRequest, session_id: str = Query(default="default")):
    """Chat with the multi-agent system."""
    try:
        orchestrator = get_orchestrator(session_id)

        # Execute in thread pool to avoid blocking
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            lambda: _execute_chat(orchestrator, request.message, request.context)
        )

        return result

    except Exception as e:
        logger.error(f"[MultiAgent] Chat error: {e}")
        raise HTTPException(status_code=500, detail=safe_error_detail(e))


def _execute_chat(orchestrator, message: str, context: Optional[Dict] = None) -> dict:
    """Execute chat in sync context."""
    response = orchestrator.chat(message, context)

    # Get last turn for metadata
    if orchestrator.history:
        last_turn = orchestrator.history[-1]
        agents_used = [r.agent for r in last_turn.results]
        routing_mode = last_turn.routing.mode.value
        confidence = last_turn.routing.confidence
    else:
        agents_used = []
        routing_mode = "unknown"
        confidence = 0.0

    return {
        "response": response,
        "agents_used": agents_used,
        "routing_mode": routing_mode,
        "confidence": confidence
    }


@router.post("/route", response_model=RoutePreviewResponse)
async def preview_routing(request: RoutePreviewRequest, session_id: str = Query(default="default")):
    """Preview routing decision without executing."""
    try:
        orchestrator = get_orchestrator(session_id)

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            lambda: orchestrator.route_preview(request.query)
        )

        return RoutePreviewResponse(**result)

    except Exception as e:
        logger.error(f"[MultiAgent] Route preview error: {e}")
        raise HTTPException(status_code=500, detail=safe_error_detail(e))


@router.post("/clear")
async def clear_history(session_id: str = Query(default="default")):
    """Clear multi-agent conversation history."""
    try:
        loop = asyncio.get_running_loop()
        def _clear():
            with _orch_lock:
                _orchestrators.pop(session_id, None)
        await loop.run_in_executor(None, _clear)
        return {"status": "cleared", "session_id": session_id}
    except Exception as e:
        return {"success": False, "error": safe_error_detail(e)}


@router.get("/history")
async def get_history(session_id: str = Query(default="default")):
    """Get recent conversation history."""
    try:
        orchestrator = get_orchestrator(session_id)

        history = []
        for turn in orchestrator.history[-10:]:  # Last 10 turns
            history.append({
                "query": turn.user_message.content,
                "agents": turn.routing.agents,
                "mode": turn.routing.mode.value,
                "response": turn.final_response[:500] + "..." if len(turn.final_response) > 500 else turn.final_response,
                "timestamp": turn.timestamp.isoformat()
            })

        return {"history": history, "total_turns": len(orchestrator.history)}

    except Exception as e:
        return {"history": [], "total_turns": 0, "error": safe_error_detail(e)}
