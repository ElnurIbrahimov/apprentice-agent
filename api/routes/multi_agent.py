"""API endpoints for the Multi-Agent System."""

import asyncio
import logging
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from api.services.agent_service import agent_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/multi-agent", tags=["multi-agent"])


# ============================================================================
# Request/Response Models
# ============================================================================

class MultiAgentChatRequest(BaseModel):
    message: str
    context: Optional[Dict[str, Any]] = None


class MultiAgentChatResponse(BaseModel):
    response: str
    agents_used: List[str]
    routing_mode: str
    confidence: float


class RoutePreviewRequest(BaseModel):
    query: str


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
# Orchestrator Singleton
# ============================================================================

_orchestrator = None


def get_orchestrator():
    """Get or create the multi-agent orchestrator."""
    global _orchestrator

    if _orchestrator is None:
        try:
            from apprentice_agent.multi_agent import MultiAgentOrchestrator

            agent = agent_service.agent

            # Create LLM function wrapper
            def llm_func(system_prompt: str, user_message: str) -> str:
                return agent.brain.think(user_message, system_prompt=system_prompt)

            # Initialize orchestrator with agent's tools
            _orchestrator = MultiAgentOrchestrator(
                tool_registry=agent.tools,
                llm_func=llm_func
            )
            logger.info("[MultiAgent] Orchestrator initialized")

        except Exception as e:
            logger.error(f"[MultiAgent] Failed to initialize orchestrator: {e}")
            raise

    return _orchestrator


# ============================================================================
# Endpoints
# ============================================================================

@router.get("/status", response_model=MultiAgentStatusResponse)
async def get_multi_agent_status():
    """Get multi-agent system status."""
    try:
        orchestrator = get_orchestrator()
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
async def list_agents():
    """List available specialist agents."""
    try:
        orchestrator = get_orchestrator()
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
        return {"agents": [], "count": 0, "error": str(e)}


@router.post("/chat", response_model=MultiAgentChatResponse)
async def multi_agent_chat(request: MultiAgentChatRequest):
    """Chat with the multi-agent system."""
    try:
        orchestrator = get_orchestrator()

        # Execute in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: _execute_chat(orchestrator, request.message, request.context)
        )

        return result

    except Exception as e:
        logger.error(f"[MultiAgent] Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
async def preview_routing(request: RoutePreviewRequest):
    """Preview routing decision without executing."""
    try:
        orchestrator = get_orchestrator()

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: orchestrator.route_preview(request.query)
        )

        return RoutePreviewResponse(**result)

    except Exception as e:
        logger.error(f"[MultiAgent] Route preview error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/clear")
async def clear_history():
    """Clear multi-agent conversation history."""
    try:
        orchestrator = get_orchestrator()
        orchestrator.clear_history()
        return {"success": True, "message": "History cleared"}
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/history")
async def get_history():
    """Get recent conversation history."""
    try:
        orchestrator = get_orchestrator()

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
        return {"history": [], "total_turns": 0, "error": str(e)}
