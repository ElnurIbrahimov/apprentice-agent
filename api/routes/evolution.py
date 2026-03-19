"""
Skill Evolution API Routes
============================

API endpoints for GEPA-based skill evolution — run optimization,
check status, and preview evolution plans.
"""

import asyncio
import logging
import threading
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field

from api.auth import require_api_key

router = APIRouter(
    prefix="/api/evolution",
    tags=["evolution"],
    dependencies=[Depends(require_api_key)],
)

logger = logging.getLogger(__name__)

# Track running evolution (protected by _run_lock)
_current_run = {"status": "idle", "result": None}
_run_lock = threading.Lock()


class EvolutionRequest(BaseModel):
    """Request to start a skill evolution run."""
    skill_ids: Optional[List[str]] = Field(None, description="Specific skill IDs (None = all)")
    max_iterations: int = Field(10, ge=1, le=50)
    reflection_model: str = Field("qwen3.5:397b-cloud")
    eval_model: str = Field("nemotron-3-super:cloud")
    timeout_seconds: int = Field(600, ge=60, le=3600)


def _run_evolution_sync(request: EvolutionRequest):
    """Run evolution in background thread."""
    try:
        from aura.evolution.runner import run_evolution
        with _run_lock:
            _current_run["status"] = "running"
        result = run_evolution(
            skill_ids=request.skill_ids,
            config_overrides={
                "max_iterations": request.max_iterations,
                "reflection_model": request.reflection_model,
                "eval_model": request.eval_model,
                "timeout_seconds": request.timeout_seconds,
            },
        )
        with _run_lock:
            _current_run["status"] = "complete"
            _current_run["result"] = result
    except Exception as e:
        logger.error(f"Evolution run failed: {e}")
        with _run_lock:
            _current_run["status"] = "error"
            _current_run["result"] = {"error": str(e)}


@router.post("/run")
async def start_evolution(request: EvolutionRequest, background_tasks: BackgroundTasks):
    """Start a GEPA skill evolution run (runs in background)."""
    with _run_lock:
        if _current_run["status"] in ("running", "starting"):
            raise HTTPException(status_code=409, detail="Evolution already running")
        _current_run["status"] = "starting"
        _current_run["result"] = None

    background_tasks.add_task(_run_evolution_sync, request)

    return {"status": "started", "message": "Evolution running in background. Check /api/evolution/status"}


@router.get("/status")
async def get_status():
    """Get current evolution run status."""
    return _current_run


@router.post("/preview")
async def preview_evolution(request: EvolutionRequest):
    """Preview what would be evolved without running."""
    try:
        from aura.evolution.runner import run_evolution

        def _preview():
            return run_evolution(
                skill_ids=request.skill_ids,
                config_overrides={
                    "max_iterations": request.max_iterations,
                    "reflection_model": request.reflection_model,
                    "eval_model": request.eval_model,
                },
                dry_run=True,
            )

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, _preview)
        return {"status": "ok", "preview": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
