"""
Browser Agent — LLM action planner.
Called by the sidebar agent loop to decide the next DOM action.
Uses Ollama directly for sync single-shot LLM calls.
"""

import json as _json
import logging
import os
import re

import httpx
from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/agent", tags=["agent"])

OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


@router.post("/action")
async def agent_action(body: dict):
    """
    Given a prompt (task + DOM state + history), return the next action as JSON.
    Response: {"action":"click"|"type"|"scroll"|"navigate"|"done",
               "selector":"", "text":"", "url":"", "amount":300, "description":""}
    """
    prompt = body.get("prompt", "")
    if not prompt:
        raise HTTPException(400, "prompt is required")
    if len(prompt) > 32_000:
        raise HTTPException(400, "prompt exceeds maximum length of 32000 characters")

    # Use model from request body, then env var, then default
    model = body.get("model") or os.getenv("AURA_AGENT_MODEL", "gemini-3-flash-preview:cloud")

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

    # Extract the first JSON object containing an "action" key
    m = re.search(r'\{[^{}]*"action"[^{}]*\}', response_text, re.DOTALL)
    if m:
        try:
            return _json.loads(m.group())
        except Exception:
            pass

    # Fallback — treat the whole response as a done description
    return {
        "action": "done",
        "description": (response_text[:200] if response_text else "No response from agent"),
    }
