"""
Introspection API Routes
========================

API endpoints for the Introspection Circuit - AURA's uncertainty detection system.
"""

import logging
from typing import Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/introspection", tags=["introspection"])

logger = logging.getLogger(__name__)


class AnalyzeRequest(BaseModel):
    """Request model for analyzing uncertainty"""
    query: str = Field(..., description="The query to analyze")
    response: Optional[str] = Field(None, description="Optional pre-generated response")
    context: str = Field("", description="Additional context")


class PreCheckRequest(BaseModel):
    """Request model for pre-response check"""
    query: str = Field(..., description="The query to check")
    context: str = Field("", description="Additional context")


class WrapRequest(BaseModel):
    """Request model for wrapping response with epistemic markers"""
    query: str = Field(..., description="The original query")
    response: str = Field(..., description="The response to wrap")
    context: str = Field("", description="Additional context")


class ConfigUpdateRequest(BaseModel):
    """Request model for updating configuration"""
    high_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    medium_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    low_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    verify_factual_below: Optional[float] = Field(None, ge=0.0, le=1.0)
    enable_consistency_check: Optional[bool] = None
    enable_auto_verification: Optional[bool] = None
    enable_epistemic_markers: Optional[bool] = None


@router.post("/analyze")
async def analyze_query(request: AnalyzeRequest):
    """
    Analyze a query/response for uncertainty.

    Returns confidence score, query type classification, and recommended action.
    """
    from api.services.agent_service import agent_service

    if not agent_service.agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    try:
        tool = _get_or_create_tool(agent_service)

        result = tool.analyze_query(
            query=request.query,
            response=request.response,
            context=request.context,
        )

        return result

    except Exception as e:
        logger.error(f"Error analyzing query: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


@router.post("/pre-check")
async def pre_check(request: PreCheckRequest):
    """
    Quick pre-check before generating a response.

    Determines if verification is needed before responding.
    """
    from api.services.agent_service import agent_service

    if not agent_service.agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    try:
        tool = _get_or_create_tool(agent_service)

        result = tool.pre_check(
            query=request.query,
            context=request.context,
        )

        return result

    except Exception as e:
        logger.error(f"Error in pre-check: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


@router.post("/wrap")
async def wrap_response(request: WrapRequest):
    """
    Wrap a response with appropriate epistemic markers.

    Adds uncertainty language based on confidence level.
    """
    from api.services.agent_service import agent_service

    if not agent_service.agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    try:
        tool = _get_or_create_tool(agent_service)

        result = tool.wrap_response(
            response=request.response,
            query=request.query,
            context=request.context,
        )

        return result

    except Exception as e:
        logger.error(f"Error wrapping response: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


@router.get("/status")
async def get_status():
    """
    Get the status of the Introspection Circuit.
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


@router.get("/stats")
async def get_stats():
    """
    Get circuit statistics including query type distribution and confidence metrics.
    """
    from api.services.agent_service import agent_service

    if not agent_service.agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    try:
        tool = _get_or_create_tool(agent_service)
        return tool.get_stats()

    except Exception as e:
        logger.error(f"Error getting stats: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


@router.get("/recent")
async def get_recent(limit: int = 10):
    """
    Get recent inner thoughts from the Real Inner Thoughts Engine.

    Returns {results: [{type, content, timestamp}]} matching the
    InnerThoughtsPanel frontend format.
    """
    try:
        from api.services.inner_thoughts_engine import get_inner_thoughts_engine
        engine = get_inner_thoughts_engine()
        thoughts = engine.get_recent(limit=min(limit, 50))

        if thoughts:
            return {"results": thoughts}

        # Fallback: if engine hasn't generated anything yet, return empty
        return {"results": []}

    except Exception as e:
        logger.error(f"Error getting recent thoughts: {e}", exc_info=True)
        return {"results": []}


@router.post("/config")
async def update_config(request: ConfigUpdateRequest):
    """
    Update Introspection Circuit configuration.
    """
    from api.services.agent_service import agent_service

    if not agent_service.agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    try:
        tool = _get_or_create_tool(agent_service)
        config = tool.circuit.config

        # Update config values
        if request.high_threshold is not None:
            config.high_confidence_threshold = request.high_threshold
        if request.medium_threshold is not None:
            config.medium_confidence_threshold = request.medium_threshold
        if request.low_threshold is not None:
            config.low_confidence_threshold = request.low_threshold
        if request.verify_factual_below is not None:
            config.verify_factual_below = request.verify_factual_below
        if request.enable_consistency_check is not None:
            config.enable_consistency_check = request.enable_consistency_check
        if request.enable_auto_verification is not None:
            config.enable_auto_verification = request.enable_auto_verification
        if request.enable_epistemic_markers is not None:
            config.enable_epistemic_markers = request.enable_epistemic_markers

        return {
            "success": True,
            "message": "Configuration updated",
            "config": {
                "high_threshold": config.high_confidence_threshold,
                "medium_threshold": config.medium_confidence_threshold,
                "low_threshold": config.low_confidence_threshold,
                "verify_factual_below": config.verify_factual_below,
                "enable_consistency_check": config.enable_consistency_check,
                "enable_auto_verification": config.enable_auto_verification,
                "enable_epistemic_markers": config.enable_epistemic_markers,
            },
        }

    except Exception as e:
        logger.error(f"Error updating config: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


@router.post("/reset-stats")
async def reset_stats():
    """
    Reset circuit statistics.
    """
    from api.services.agent_service import agent_service

    if not agent_service.agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    try:
        tool = _get_or_create_tool(agent_service)
        tool.circuit.reset_stats()

        return {"success": True, "message": "Statistics reset"}

    except Exception as e:
        logger.error(f"Error resetting stats: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


def _get_or_create_tool(agent_service):
    """Get or create the introspection tool."""
    from aura.tools.introspection_tool import IntrospectionTool
    from aura.tools.introspection_circuit import IntrospectionConfig

    # Check if tool exists
    if hasattr(agent_service, '_introspection_tool'):
        return agent_service._introspection_tool

    # Create LLM function
    def llm_func(prompt, system_prompt=None):
        return agent_service.agent.brain.think(prompt, system_prompt)

    # Try to get search function
    search_func = None
    if hasattr(agent_service.agent, 'search'):
        search_func = agent_service.agent.search
    elif hasattr(agent_service.agent, 'tools'):
        for tool in agent_service.agent.tools.values():
            if hasattr(tool, 'search'):
                search_func = tool.search
                break

    # Try to get FluxMind
    fluxmind = None
    if hasattr(agent_service.agent, 'fluxmind'):
        fluxmind = agent_service.agent.fluxmind
    elif hasattr(agent_service.agent, 'tools'):
        for tool in agent_service.agent.tools.values():
            if 'fluxmind' in tool.__class__.__name__.lower():
                fluxmind = tool
                break

    # Try to get Guardian
    guardian = None
    if hasattr(agent_service.agent, 'guardian'):
        guardian = agent_service.agent.guardian
    elif hasattr(agent_service.agent, 'tools'):
        for tool in agent_service.agent.tools.values():
            if 'guardian' in tool.__class__.__name__.lower():
                guardian = tool
                break

    # Create tool
    config = IntrospectionConfig(
        enable_consistency_check=True,
        enable_auto_verification=True,
        enable_epistemic_markers=True,
        use_fluxmind=fluxmind is not None,
        use_guardian=guardian is not None,
    )

    tool = IntrospectionTool(
        llm_func=llm_func,
        search_func=search_func,
        config=config,
        fluxmind=fluxmind,
        guardian=guardian,
    )

    agent_service._introspection_tool = tool
    logger.info("Introspection Tool created")

    return tool
