"""
Self-Improvement API Routes
============================

API endpoints for the Self-Improvement Engine — exposes improvement status,
quality reports, tunable parameters, and manual cycle triggers.
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

from api.auth import require_api_key
from api.utils import safe_error_detail

router = APIRouter(prefix="/api/self-improvement", tags=["self-improvement"], dependencies=[Depends(require_api_key)])

logger = logging.getLogger(__name__)


class TuneParamRequest(BaseModel):
    """Request model for manually tuning a parameter."""
    name: str = Field(..., description="Parameter name (e.g. 'brain.base_temperature')")
    value: float = Field(..., description="New value for the parameter")


@router.get("/status")
async def get_status():
    """Get self-improvement engine state, recent outcomes, and cycle info."""
    try:
        from aura.consciousness.self_improvement import (
            get_self_improvement_engine,
        )
        engine = get_self_improvement_engine()
        return {
            "status": "ok",
            **engine.get_status(),
        }
    except Exception as e:
        logger.error(f"[SelfImprovement API] status error: {e}")
        raise HTTPException(status_code=500, detail=safe_error_detail(e))


@router.get("/report")
async def get_report():
    """Get quality evaluation report with domain trends and strategy effectiveness."""
    try:
        from aura.consciousness.self_improvement import (
            get_self_improvement_engine,
        )
        engine = get_self_improvement_engine()
        return {
            "status": "ok",
            "report": engine.get_improvement_report(),
        }
    except Exception as e:
        logger.error(f"[SelfImprovement API] report error: {e}")
        raise HTTPException(status_code=500, detail=safe_error_detail(e))


@router.get("/params")
async def get_params():
    """Get current tunable parameters registry with values."""
    try:
        from aura.consciousness.self_improvement import (
            get_self_improvement_engine,
        )
        engine = get_self_improvement_engine()
        return {
            "status": "ok",
            "params": engine.get_tunable_params(),
        }
    except Exception as e:
        logger.error(f"[SelfImprovement API] params error: {e}")
        raise HTTPException(status_code=500, detail=safe_error_detail(e))


@router.post("/cycle")
async def trigger_cycle():
    """Manually trigger an improvement cycle."""
    try:
        from aura.consciousness.self_improvement import (
            get_self_improvement_engine,
        )
        engine = get_self_improvement_engine()
        result = engine.trigger_cycle()

        if result:
            return {
                "status": "ok",
                "message": "Improvement cycle completed",
                "result": result.to_dict(),
            }
        else:
            return {
                "status": "ok",
                "message": "Improvement cycle returned no result (may have failed)",
                "result": None,
            }
    except Exception as e:
        logger.error(f"[SelfImprovement API] cycle error: {e}")
        raise HTTPException(status_code=500, detail=safe_error_detail(e))


@router.post("/tune")
async def tune_param(req: TuneParamRequest):
    """Manually adjust a tunable parameter."""
    try:
        from aura.consciousness.self_improvement import (
            get_self_improvement_engine,
        )
        engine = get_self_improvement_engine()
        result = engine.tune_param(req.name, req.value)

        if result.get("success"):
            return {"status": "ok", **result}
        else:
            raise HTTPException(status_code=400, detail=result.get("error", "unknown error"))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[SelfImprovement API] tune error: {e}")
        raise HTTPException(status_code=500, detail=safe_error_detail(e))
