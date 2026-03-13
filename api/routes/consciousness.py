"""
Consciousness API Routes
========================

API endpoints for the Global Workspace Theory engine — AURA's conscious
attention mechanism. Exposes current conscious state, broadcast history,
attention schema, and manual focus injection for testing.
"""

import logging
import time
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

from api.auth import require_api_key

router = APIRouter(prefix="/api/consciousness", tags=["consciousness"], dependencies=[Depends(require_api_key)])

logger = logging.getLogger(__name__)


class FocusRequest(BaseModel):
    """Request model for manually directing attention."""
    source_module: str = Field(..., description="Source module name")
    content_type: str = Field("manual_focus", description="Content type")
    summary: str = Field(..., description="Human-readable description")
    activation: float = Field(0.9, ge=0.0, le=1.0, description="Activation level")
    salience: float = Field(0.9, ge=0.0, le=1.0, description="Salience level")


@router.get("/state")
async def get_conscious_state():
    """Get current conscious content — what AURA is attending to right now."""
    try:
        from aura.consciousness.global_workspace import get_global_workspace
        gw = get_global_workspace()
        state = gw.get_conscious_state()
        return {
            "status": "ok",
            "conscious_state": state.to_dict(),
        }
    except Exception as e:
        logger.error(f"[Consciousness API] state error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history")
async def get_broadcast_history(limit: int = 20):
    """Get recent broadcast history — the stream of consciousness."""
    try:
        from aura.consciousness.global_workspace import get_global_workspace
        gw = get_global_workspace()
        history = gw.get_broadcast_history(limit=limit)
        return {
            "status": "ok",
            "count": len(history),
            "broadcasts": [b.to_dict() for b in history],
        }
    except Exception as e:
        logger.error(f"[Consciousness API] history error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/attention")
async def get_attention_schema():
    """Get attention schema (AST self-model) and codelet statistics."""
    try:
        from aura.consciousness.global_workspace import get_global_workspace
        gw = get_global_workspace()
        return {
            "status": "ok",
            "attention_schema": gw.get_attention_schema(),
            "codelet_stats": gw.get_codelet_stats(),
        }
    except Exception as e:
        logger.error(f"[Consciousness API] attention error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/focus")
async def inject_focus(req: FocusRequest):
    """Manually direct attention — injects content that wins next cycle (testing)."""
    try:
        from aura.consciousness.global_workspace import (
            WorkspaceContent,
            get_global_workspace,
        )

        content = WorkspaceContent(
            source_module=req.source_module,
            content_type=req.content_type,
            summary=req.summary,
            activation=req.activation,
            salience=req.salience,
        )

        gw = get_global_workspace()
        gw.inject_content(content)

        return {
            "status": "ok",
            "message": f"Focus injected: {req.summary}",
            "conscious_state": gw.get_conscious_state().to_dict(),
        }
    except Exception as e:
        logger.error(f"[Consciousness API] focus injection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_engine_stats():
    """Get Global Workspace engine statistics."""
    try:
        from aura.consciousness.global_workspace import get_global_workspace
        gw = get_global_workspace()
        return {
            "status": "ok",
            "engine_stats": gw.get_stats(),
        }
    except Exception as e:
        logger.error(f"[Consciousness API] stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
