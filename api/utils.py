"""Shared utilities for the AURA API layer."""

import os
import re

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
