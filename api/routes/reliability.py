"""
Reliability & Telemetry API endpoints.

Provides:
  GET  /api/telemetry/recent        — last N events from ring buffer
  GET  /api/telemetry/stats         — aggregate metrics
  GET  /api/introspection/reliability — reliability summary
  POST /api/evals/run               — run an eval suite (stub, extensible)

Author: Aura reliability upgrade (2026-03)
"""

import logging
from typing import Optional

from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(tags=["reliability"])


# ---------------------------------------------------------------------------
# Telemetry endpoints
# ---------------------------------------------------------------------------

@router.get("/api/telemetry/recent")
async def telemetry_recent(
    n: int = Query(50, ge=1, le=500),
    kind: Optional[str] = Query(None),
):
    """Return the N most recent telemetry events."""
    try:
        from aura.reliability.telemetry import get_telemetry
        return {"events": get_telemetry().recent(n=n, kind=kind)}
    except Exception as e:
        raise HTTPException(500, f"Telemetry unavailable: {e}")


@router.get("/api/telemetry/stats")
async def telemetry_stats():
    """Return aggregate telemetry statistics."""
    try:
        from aura.reliability.telemetry import get_telemetry
        return get_telemetry().stats()
    except Exception as e:
        raise HTTPException(500, f"Telemetry unavailable: {e}")


# ---------------------------------------------------------------------------
# Reliability summary
# ---------------------------------------------------------------------------

@router.get("/api/introspection/reliability")
async def reliability_summary():
    """
    High-level reliability summary combining:
      - telemetry stats
      - memory write gate decisions
      - loop guard status
      - routing stats
    """
    result: dict = {}

    try:
        from aura.reliability.telemetry import get_telemetry
        result["telemetry"] = get_telemetry().stats()
    except Exception as e:
        result["telemetry_error"] = str(e)

    try:
        from aura.reliability.routing_stats import get_routing_stats
        result["routing"] = get_routing_stats().summary()
    except Exception as e:
        result["routing"] = {"error": str(e)}

    try:
        from aura.memory.unified_memory import get_unified_memory
        result["memory"] = get_unified_memory().get_stats()
    except Exception as e:
        result["memory"] = {"error": str(e)}

    return result


# ---------------------------------------------------------------------------
# Eval runner (stub — extend per suite)
# ---------------------------------------------------------------------------

class EvalRunRequest(BaseModel):
    suite: str = "memory_continuity"   # suite name
    user_id: str = "eval_user"
    max_cases: int = 10


@router.post("/api/evals/run")
async def run_eval(req: EvalRunRequest):
    """
    Run a named eval suite. Currently supported:
      memory_continuity, loop_detection, browser_reliability
    """
    try:
        from aura.reliability.eval_harness import run_suite
        results = run_suite(req.suite, user_id=req.user_id, max_cases=req.max_cases)
        return {"suite": req.suite, "results": results}
    except ImportError:
        raise HTTPException(501, "Eval harness not yet implemented for this suite")
    except Exception as e:
        raise HTTPException(500, f"Eval failed: {e}")
