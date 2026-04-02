"""Shared utilities for the AURA API layer."""

import asyncio
import functools
import logging
import os
import re
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# Reusable validators for API input parameters
_SAFE_ID_RE = re.compile(r"^[a-zA-Z0-9_\-]{1,128}$")
_SAFE_MODEL_RE = re.compile(r"^[a-zA-Z0-9._:\-/]{1,128}$")
_VALID_EMOTIONS = frozenset({
    "happy", "sad", "angry", "fearful", "surprised", "disgusted",
    "neutral", "curious", "excited", "anxious", "confident", "confused",
    "calm", "frustrated", "hopeful", "bored", "amused", "nostalgic",
    "joy", "trust", "anticipation", "interest", "serenity", "acceptance",
})


def safe_error_detail(e: Exception, default: str = "Internal server error") -> str:
    """Return detailed error in dev, generic in production."""
    if os.environ.get("AURA_ENV") == "production":
        return default
    return str(e)


def validate_id(value: str, name: str = "id") -> str:
    """Validate an ID parameter (conversation_id, session_id, etc.)."""
    if not value or not _SAFE_ID_RE.match(value):
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail=f"Invalid {name}: must be 1-128 alphanumeric/dash/underscore characters")
    return value


def validate_model_name(value: str) -> str:
    """Validate a model name parameter."""
    if not value or not _SAFE_MODEL_RE.match(value):
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="Invalid model name")
    return value


def validate_emotion(value: str) -> str:
    """Validate an emotion string against known values."""
    normalized = value.strip().lower()
    if normalized not in _VALID_EMOTIONS:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail=f"Invalid emotion: '{value}'. Valid: {', '.join(sorted(_VALID_EMOTIONS))}")
    return normalized


class EndpointRateLimiter:
    """Simple per-endpoint rate limiter for expensive operations.

    Usage in a route:
        _code_limiter = EndpointRateLimiter(max_per_minute=10)

        @router.post("/execute")
        async def execute(request):
            _code_limiter.check()  # raises HTTPException(429) if exceeded
    """

    def __init__(self, max_per_minute: int = 10):
        self._max = max_per_minute
        self._timestamps: list = []
        import threading
        self._lock = threading.Lock()

    def check(self):
        import time
        now = time.time()
        with self._lock:
            cutoff = now - 60
            self._timestamps = [t for t in self._timestamps if t > cutoff]
            if len(self._timestamps) >= self._max:
                from fastapi import HTTPException
                raise HTTPException(
                    status_code=429,
                    detail=f"Rate limit exceeded ({self._max}/min for this endpoint)"
                )
            self._timestamps.append(now)


def error_response(message: str, status_code: int = 500, **extra) -> dict:
    """Standardized error response format for non-HTTPException errors.

    Convention:
      - HTTPException → {"detail": "..."} (FastAPI built-in, 4xx)
      - This helper → {"success": false, "error": "..."} (custom, typically 5xx)

    Use HTTPException for client errors, this helper for structured error returns.
    """
    result = {"success": False, "error": message}
    result.update(extra)
    return result


# ---------------------------------------------------------------------------
# Lazy agent service accessor (single source of truth)
# ---------------------------------------------------------------------------

def get_agent_service():
    """Get agent_service with lazy loading. Use this instead of duplicating per-route."""
    from api.services.agent_service import agent_service
    return agent_service


def get_agent():
    """Shorthand for get_agent_service().agent."""
    return get_agent_service().agent


# ---------------------------------------------------------------------------
# Tool access helpers
# ---------------------------------------------------------------------------

def call_tool(tool_name: str, method: str, *args, **kwargs) -> Any:
    """Call a method on a named tool, returning error dict if tool not loaded."""
    agent = get_agent()
    tool = agent.tools.get(tool_name)
    if tool is None:
        return {"success": False, "error": f"{tool_name} tool not loaded"}
    fn = getattr(tool, method, None)
    if fn is None:
        return {"success": False, "error": f"{tool_name} has no method '{method}'"}
    return fn(*args, **kwargs)


def get_amem():
    """Get the A-MEM instance from whichever tool provides it."""
    agent = get_agent()
    for name in ("amem", "hybrid_amem"):
        tool = agent.tools.get(name)
        if tool and hasattr(tool, "amem"):
            return tool.amem
    return None


# ---------------------------------------------------------------------------
# Async executor wrapper
# ---------------------------------------------------------------------------

async def run_sync(fn: Callable, *args):
    """Run a blocking function in the default executor.

    Usage in a route::

        @router.get("/foo")
        async def foo():
            return await run_sync(_foo_sync, arg1, arg2)
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, functools.partial(fn, *args))
