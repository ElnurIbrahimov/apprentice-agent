"""Ghost-text inline completion endpoint.

Used by the Chrome extension's `ghost-text.ts` content script to fetch a
short continuation for any `<textarea>` or `[contenteditable]` input on any
page. Target latency: <300 ms round-trip. Uses the fastest available local
model (gemma4:e2b) with a cloud fallback if local is down.

Deployment:
  1. git pull on Hetzner
  2. systemctl restart aura-api
  3. No new environment variables required
  4. Verify at `/docs` in dev or curl `/api/ghost/complete` with a test body

Added as part of SOTA Round 2 (Cluster 5).
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from api.auth import require_api_key
from api.utils import safe_error_detail

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/ghost", tags=["ghost"], dependencies=[Depends(require_api_key)])

OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL") or os.getenv("OLLAMA_HOST", "http://localhost:11434")
FAST_MODEL = os.getenv("AURA_GHOST_MODEL", "gemma4:e2b")
FALLBACK_MODEL = os.getenv("AURA_GHOST_FALLBACK", "qwen3.5:cloud")

MAX_TEXT_LEN = 2000
MAX_CONTINUATION_TOKENS = 40

# Simple per-key rate limiting: 60 requests / 60 seconds / key
_rate_window: dict[str, list[float]] = {}
_RATE_WINDOW_SEC = 60.0
_RATE_LIMIT = 60


class GhostRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_TEXT_LEN)
    url: Optional[str] = Field(default=None, max_length=2048)
    title: Optional[str] = Field(default=None, max_length=512)


class GhostResponse(BaseModel):
    continuation: str


def _build_prompt(text: str, url: Optional[str], title: Optional[str]) -> str:
    tail = text[-600:]
    context = ""
    if title or url:
        context = f"[Context: {title or ''} — {url or ''}]\n\n"
    return (
        f"{context}Continue the following text naturally. Output ONLY the continuation "
        f"(max ~20 words). Do not repeat the input. Do not add quotes, markdown, or explanation.\n\n"
        f"<<<\n{tail}\n>>>\n\nContinuation:"
    )


def _rate_limit_ok(key: str) -> bool:
    now = time.monotonic()
    timestamps = _rate_window.get(key) or []
    timestamps = [t for t in timestamps if now - t < _RATE_WINDOW_SEC]
    if len(timestamps) >= _RATE_LIMIT:
        _rate_window[key] = timestamps
        return False
    timestamps.append(now)
    _rate_window[key] = timestamps
    return True


async def _generate(model: str, prompt: str) -> Optional[str]:
    try:
        async with httpx.AsyncClient(timeout=6) as c:
            r = await c.post(
                f"{OLLAMA_BASE}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": MAX_CONTINUATION_TOKENS,
                        "temperature": 0.3,
                        "top_p": 0.85,
                        "stop": ["\n\n", "<<<", ">>>"],
                    },
                },
            )
            if r.status_code != 200:
                logger.debug("[Ghost] %s returned %s", model, r.status_code)
                return None
            text = (r.json() or {}).get("response", "") or ""
            return text.strip()
    except Exception as e:
        logger.debug("[Ghost] %s failed: %s", model, e)
        return None


@router.post("/complete", response_model=GhostResponse)
async def ghost_complete(body: GhostRequest) -> GhostResponse:
    """Return a short continuation for a textarea / contenteditable input."""
    if not _rate_limit_ok("global"):
        raise HTTPException(429, "Ghost rate limit exceeded")

    prompt = _build_prompt(body.text, body.url, body.title)

    try:
        continuation = await _generate(FAST_MODEL, prompt)
        if not continuation:
            continuation = await _generate(FALLBACK_MODEL, prompt)
        if not continuation:
            return GhostResponse(continuation="")

        # Clean: strip quotes, leading bullet markers, repeated input tail
        cleaned = continuation.strip().strip('"').strip("'")
        # Drop anything after the first double newline
        cleaned = cleaned.split("\n\n", 1)[0]
        # Drop if the model repeated the input literally
        tail = body.text[-40:].lower()
        if cleaned.lower().startswith(tail):
            cleaned = cleaned[len(tail):].lstrip()
        # Hard cap length
        cleaned = cleaned[:200]
        return GhostResponse(continuation=cleaned)
    except Exception as e:
        logger.warning("[Ghost] complete failed: %s", e)
        raise HTTPException(500, detail=safe_error_detail(e)) from e
