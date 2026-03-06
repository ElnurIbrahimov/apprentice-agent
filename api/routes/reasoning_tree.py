"""
Reasoning Tree API Routes
=========================

API endpoints for MCTS-based deep reasoning.
"""

import logging
import asyncio
import uuid
from typing import Optional, List
from datetime import datetime

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/reasoning-tree", tags=["reasoning-tree"])

logger = logging.getLogger(__name__)

# Store for active reasoning sessions
_active_sessions = {}
_session_results = {}


class ReasoningRequest(BaseModel):
    """Request model for deep reasoning"""
    problem: str = Field(..., description="The problem to reason about")
    context: str = Field("", description="Additional context")
    max_iterations: int = Field(30, ge=5, le=100, description="Maximum MCTS iterations")
    max_depth: int = Field(10, ge=3, le=20, description="Maximum tree depth")
    branching_factor: int = Field(5, ge=2, le=10, description="Number of candidates per expansion")


class ExploreRequest(BaseModel):
    """Request model for exploring options"""
    question: str = Field(..., description="The decision question")
    num_options: int = Field(5, ge=2, le=10, description="Number of options to explore")
    context: str = Field("", description="Additional context")


class ReasoningResponse(BaseModel):
    """Response model for reasoning results"""
    success: bool
    session_id: str
    answer: Optional[str] = None
    confidence: Optional[float] = None
    reasoning_steps: Optional[List[dict]] = None
    summary: Optional[str] = None
    metadata: Optional[dict] = None
    error: Optional[str] = None


class TreeVisualization(BaseModel):
    """Response model for tree visualization"""
    success: bool
    tree: Optional[dict] = None
    stats: Optional[dict] = None
    error: Optional[str] = None


class SessionStatus(BaseModel):
    """Status of a reasoning session"""
    session_id: str
    status: str  # "pending", "running", "completed", "error"
    progress: Optional[dict] = None
    result: Optional[dict] = None


@router.post("/think", response_model=ReasoningResponse)
async def think_deeply(request: ReasoningRequest, background_tasks: BackgroundTasks):
    """
    Start deep reasoning on a problem using MCTS.

    This endpoint initiates an MCTS search to explore multiple reasoning paths
    and find the optimal solution.
    """
    from api.services.agent_service import agent_service

    if not agent_service.agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    session_id = f"mcts_{uuid.uuid4()}"

    try:
        # Get or create the reasoning tree tool
        tool = _get_or_create_tool(agent_service)

        # Run reasoning (this can take time)
        logger.info(f"Starting deep reasoning session {session_id}")

        result = tool.think_deeply(
            problem=request.problem,
            context=request.context,
            max_iterations=request.max_iterations,
            max_depth=request.max_depth,
        )

        # Store result (evict oldest if over capacity)
        _session_results[session_id] = result
        if len(_session_results) > 100:
            oldest_key = next(iter(_session_results))
            del _session_results[oldest_key]

        return ReasoningResponse(
            success=result.get("success", False),
            session_id=session_id,
            answer=result.get("answer"),
            confidence=result.get("confidence"),
            reasoning_steps=result.get("reasoning_steps"),
            summary=result.get("summary"),
            metadata=result.get("metadata"),
        )

    except Exception as e:
        logger.error(f"Error in deep reasoning: {e}", exc_info=True)
        return ReasoningResponse(
            success=False,
            session_id=session_id,
            error=str(e),
        )


@router.post("/explore", response_model=ReasoningResponse)
async def explore_options(request: ExploreRequest):
    """
    Explore multiple options/approaches for a decision.

    Uses MCTS to generate and evaluate different approaches.
    """
    from api.services.agent_service import agent_service

    if not agent_service.agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    session_id = f"explore_{uuid.uuid4()}"

    try:
        tool = _get_or_create_tool(agent_service)

        result = tool.explore_options(
            question=request.question,
            num_options=request.num_options,
            context=request.context,
        )

        # Store result (evict oldest if over capacity)
        _session_results[session_id] = result
        if len(_session_results) > 100:
            oldest_key = next(iter(_session_results))
            del _session_results[oldest_key]

        return ReasoningResponse(
            success=result.get("success", False),
            session_id=session_id,
            answer=result.get("recommendation"),
            confidence=result.get("options", [{}])[0].get("score") if result.get("options") else None,
            reasoning_steps=[{"type": "option", **opt} for opt in result.get("options", [])],
            metadata={
                "question": request.question,
                "iterations": result.get("iterations"),
                "time_taken": result.get("time_taken"),
            },
        )

    except Exception as e:
        logger.error(f"Error exploring options: {e}", exc_info=True)
        return ReasoningResponse(
            success=False,
            session_id=session_id,
            error=str(e),
        )


@router.get("/tree/{session_id}", response_model=TreeVisualization)
async def get_tree_visualization(session_id: str):
    """
    Get the reasoning tree visualization for a session.

    Returns the full tree structure for rendering in the UI.
    """
    from api.services.agent_service import agent_service

    if not agent_service.agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    try:
        tool = _get_or_create_tool(agent_service)
        result = tool.get_tree_visualization()

        return TreeVisualization(
            success=result.get("success", False),
            tree=result.get("tree"),
            stats=result.get("stats"),
        )

    except Exception as e:
        logger.error(f"Error getting tree visualization: {e}", exc_info=True)
        return TreeVisualization(
            success=False,
            error=str(e),
        )


@router.get("/path/{session_id}")
async def get_reasoning_path(session_id: str):
    """
    Get the best reasoning path from a session.
    """
    from api.services.agent_service import agent_service

    if not agent_service.agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    try:
        tool = _get_or_create_tool(agent_service)
        return tool.get_reasoning_path()

    except Exception as e:
        logger.error(f"Error getting reasoning path: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


@router.get("/reflections/{session_id}")
async def get_reflections(session_id: str):
    """
    Get reflections/lessons learned from a reasoning session.
    """
    from api.services.agent_service import agent_service

    if not agent_service.agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    try:
        tool = _get_or_create_tool(agent_service)
        return tool.get_reflections()

    except Exception as e:
        logger.error(f"Error getting reflections: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


@router.get("/status")
async def get_reasoning_status():
    """
    Get the status of the reasoning tree tool.
    """
    from api.services.agent_service import agent_service

    if not agent_service.agent:
        return {
            "success": False,
            "enabled": False,
            "error": "Agent not initialized",
        }

    try:
        tool = _get_or_create_tool(agent_service)
        status = tool.status()

        return {
            "success": True,
            "enabled": True,
            **status,
        }

    except Exception as e:
        logger.error(f"Error getting status: {e}", exc_info=True)
        return {
            "success": False,
            "enabled": False,
            "error": str(e),
        }


@router.get("/sessions")
async def list_sessions():
    """
    List recent reasoning sessions.
    """
    sessions = []
    for session_id, result in list(_session_results.items())[-10:]:  # Last 10
        sessions.append({
            "session_id": session_id,
            "success": result.get("success"),
            "confidence": result.get("confidence"),
        })

    return {
        "success": True,
        "sessions": sessions,
        "total": len(_session_results),
    }


def _get_or_create_tool(agent_service):
    """Get or create the reasoning tree tool."""
    from aura.tools.reasoning_tree_tool import ReasoningTreeTool

    # Check if tool exists in agent
    if hasattr(agent_service, '_reasoning_tree_tool'):
        return agent_service._reasoning_tree_tool

    # Create new tool
    def llm_func(prompt, system_prompt=None):
        return agent_service.agent.brain.think(prompt, system_prompt)

    tool = ReasoningTreeTool(llm_func=llm_func)
    agent_service._reasoning_tree_tool = tool

    return tool
