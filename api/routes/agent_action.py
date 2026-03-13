"""
Browser Agent — LLM action planner + executor.

Two modes:
  1. plan   (POST /api/agent/action)     — plan-only, returns next action JSON
                                            (backward-compatible with sidebar loop)
  2. verify (POST /api/agent/action/verify) — plan + parse into PlannedAction
                                              with safety class and success signals

The extension sidebar calls /api/agent/action and handles execution itself.
The backend executor (BrowserSession) is available for server-side automation.

Author: Aura reliability upgrade (2026-03)
"""

import json as _json
import logging
import os
import re
import time

import httpx
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Any, Dict, List, Optional

from api.auth import require_api_key

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/agent", tags=["agent"], dependencies=[Depends(require_api_key)])

OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


# ---------------------------------------------------------------------------
# Original plan endpoint — backward compatible
# ---------------------------------------------------------------------------

@router.post("/action")
async def agent_action(body: dict):
    """
    Given a prompt (task + DOM state + history), return the next action as JSON.
    Response: {"action":"click"|"type"|"scroll"|"navigate"|"done",
               "selector":"", "text":"", "url":"", "amount":300, "description":"",
               "safety_class":"safe"|"sensitive"|"destructive",
               "success_signals":[...]}
    """
    prompt = body.get("prompt", "")
    if not prompt:
        raise HTTPException(400, "prompt is required")
    if len(prompt) > 32_000:
        raise HTTPException(400, "prompt exceeds maximum length of 32000 characters")

    session_id = body.get("session_id", "")
    model = body.get("model") or os.getenv("AURA_AGENT_MODEL", "gemini-3-flash-preview:cloud")

    t0 = time.monotonic()

    # Loop guard check
    try:
        from aura.reliability.loop_guard import get_guard
        if session_id:
            guard = get_guard(session_id)
            guard_result = guard.record("llm_plan_call", prompt[:80])
            if guard_result.triggered:
                logger.warning("[AgentAction] Loop guard fired: %s", guard_result.reason)
                return {
                    "action": "abort",
                    "description": f"Loop guard: {guard_result.reason}. {guard_result.fallback_message}",
                    "loop_guard": True,
                }
    except Exception:
        pass

    try:
        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.post(
                f"{OLLAMA_BASE}/api/generate",
                json={"model": model, "prompt": prompt, "stream": False},
            )
        r.raise_for_status()
        response_text = r.json().get("response", "")
    except Exception as e:
        logger.error("[AgentAction] LLM call failed: %s", e)
        raise HTTPException(500, f"LLM call failed: {e}")

    latency_ms = (time.monotonic() - t0) * 1000

    # Extract the first JSON object containing an "action" key
    m = re.search(r'\{[^{}]*"action"[^{}]*\}', response_text, re.DOTALL)
    raw_action: Dict[str, Any] = {}
    if m:
        try:
            raw_action = _json.loads(m.group())
        except Exception:
            pass

    if not raw_action:
        raw_action = {
            "action": "done",
            "description": (response_text[:200] if response_text else "No response from agent"),
        }

    # Enrich with planner metadata (safety class, success signals)
    try:
        from aura.browser.executor import BrowserPlanner
        planner = BrowserPlanner()
        planned = planner.parse(raw_action)
        raw_action["safety_class"]    = planned.safety_class.value
        raw_action["success_signals"] = planned.success_signals
    except Exception:
        pass

    # Telemetry
    try:
        from aura.reliability.telemetry import emit, TelemetryKind
        emit(
            TelemetryKind.BROWSER_ACTION,
            session_id=session_id,
            latency_ms=latency_ms,
            model_used=model,
            extra={"action": raw_action.get("action"), "phase": "plan"},
        )
    except Exception:
        pass

    return raw_action


# ---------------------------------------------------------------------------
# Verify endpoint — returns planner metadata without executing
# ---------------------------------------------------------------------------

class VerifyRequest(BaseModel):
    action: Dict[str, Any]
    session_id: str = ""


@router.post("/action/verify")
async def verify_action(req: VerifyRequest):
    """
    Parse a raw action dict and return planner metadata (safety class, signals).
    Useful for the extension to check before executing a sensitive action.
    """
    try:
        from aura.browser.executor import BrowserPlanner
        planner = BrowserPlanner()
        planned = planner.parse(req.action)
        return {
            "kind": planned.kind.value,
            "safety_class": planned.safety_class.value,
            "success_signals": planned.success_signals,
            "fallback_selectors": planned.fallback_selectors,
            "description": planned.description,
        }
    except Exception as e:
        raise HTTPException(500, f"Planner error: {e}")


# ---------------------------------------------------------------------------
# Trace endpoint — get loop guard status for a session
# ---------------------------------------------------------------------------

@router.get("/session/{session_id}/status")
async def session_status(session_id: str):
    """Return loop guard status and action count for a session."""
    try:
        from aura.reliability.loop_guard import get_guard
        guard = get_guard(session_id)
        return {
            "session_id": session_id,
            "actions_taken": guard.actions_taken,
            "triggered": guard._triggered,
            "trigger_reason": guard._trigger_reason,
        }
    except Exception as e:
        raise HTTPException(500, str(e))


@router.post("/session/{session_id}/reset")
async def reset_session(session_id: str):
    """Reset loop guard state for a session (call at start of new task)."""
    try:
        from aura.reliability.loop_guard import reset_guard
        reset_guard(session_id)
        return {"ok": True, "session_id": session_id}
    except Exception as e:
        raise HTTPException(500, str(e))
