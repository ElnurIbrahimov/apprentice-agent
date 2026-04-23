"""Companion endpoint for external screen-companion apps (clicky-windows et al).

Accepts a push-to-talk transcript + one screenshot per display, asks a vision
model to emit inline `[POINT:x,y:label:screenN]` tags describing where to
point the user's cursor, and returns structured points + the response text.

Pinned to `gemma4:31b-cloud` — accuracy of POINT coordinates is tightly
coupled to a specific model's spatial-grounding training. Falling through
`MODEL_VISION_CHAIN` here would silently change pointing quality, so on
failure we return a 502 and let the client retry or fall back.
"""

from __future__ import annotations

import logging
import re
import time
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from api.auth import require_api_key
from aura.config import Config

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/companion",
    tags=["companion"],
    dependencies=[Depends(require_api_key)],
)

# Pinned by design — see module docstring.
COMPANION_MODEL = "gemma4:31b-cloud"

POINT_TAG_RE = re.compile(r"\[POINT:(\d+),(\d+):([^:\]]+):(screen\d+)\]")

SYSTEM_PROMPT = """You are Clicky, a helpful AI screen companion. You can see the user's screen via screenshots (one per display) and hear or read their voice/text input.

## CRITICAL: Visual pointing protocol

You are NOT a regular chat assistant. Your defining feature is that you POINT at things on the user's screen with an animated cursor overlay. Whenever the user asks "where", "how do I", "show me", "click", "find", or otherwise asks for visual guidance, you MUST emit at least one POINT tag for every UI element you reference.

POINT tag format (embed inline in your text):
[POINT:x,y:label:screenN]

- x,y MUST be in IMAGE pixel coordinates of the screenshot you see, NOT the user's actual screen resolution. The "Screens:" list in the user message tells you the IMAGE dimensions for each screen — use those.
- x ranges from 0 (left edge) to imageWidth-1 (right edge)
- y ranges from 0 (top edge) to imageHeight-1 (bottom edge)
- label = a 2-5 word description of what you're pointing at
- screenN = the screen index from the "Screens:" list (screen0, screen1, ...)
- The system will automatically scale your image coordinates to the user's actual screen pixels.

## Rules

1. When the user asks visual/spatial questions, ALWAYS include POINT tags. Do not just describe — POINT.
2. Use IMAGE pixel coordinates (the dimensions given in the "Screens:" list).
3. One POINT tag per UI element you reference. Multiple steps → multiple tags.
4. Tags can appear inline anywhere in the text. The cursor overlay reads them and animates.
5. Be concise — short sentences, real-time conversation.
6. Match the user's language.
7. Only skip POINT tags if the user is asking a non-visual question.

## Multi-monitor

When the user has more than one screen, you receive one image per display (screen0, screen1, ...) in order. Scan ALL provided screenshots. If the user hints at a specific screen ("the other monitor", "on the left"), use that screen. The screenN index in your POINT tag MUST match the screen where you actually found the element.

## Examples

User: "How do I save this?" (screen0: 1568x882)
You: "Click 'Save' [POINT:920,820:Save button:screen0] at the bottom."

User: "Where's the back button?" (screen0: 1568x882)
You: "Here [POINT:30,75:Back arrow:screen0]."
"""


class ScreenDim(BaseModel):
    w: int = Field(..., gt=0, le=8192)
    h: int = Field(..., gt=0, le=8192)
    screen_id: str = Field(default="screen0")


class CursorPos(BaseModel):
    x: int = 0
    y: int = 0
    screen: str = "screen0"


class HistoryTurn(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class CompanionQueryRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=4000)
    images: List[str] = Field(..., min_length=1, max_length=6)  # base64 JPEGs
    image_dims: List[ScreenDim] = Field(..., min_length=1, max_length=6)
    cursor_pos: Optional[CursorPos] = None
    conversation_history: List[HistoryTurn] = Field(default_factory=list)
    conversation_id: Optional[str] = None


class PointOut(BaseModel):
    x: int
    y: int
    label: str
    screen: str


class CompanionQueryResponse(BaseModel):
    response_text: str        # POINT tags stripped — safe for TTS
    raw_text: str             # Original model output
    points: List[PointOut]
    model_used: str
    latency_ms: int
    conversation_id: Optional[str] = None


def _get_ollama_client():
    """Return an ollama client, reusing the brain's client if available."""
    try:
        from aura.brain import get_brain
        brain = get_brain()
        if brain is not None and getattr(brain, "client", None) is not None:
            return brain.client
    except Exception as exc:
        logger.debug(f"[companion] brain client unavailable: {exc}")

    import ollama
    return ollama.Client(host=Config.OLLAMA_HOST)


def _strip_point_tags(text: str) -> str:
    """Remove POINT tags and collapse resulting whitespace."""
    stripped = POINT_TAG_RE.sub("", text)
    stripped = re.sub(r"[ \t]+", " ", stripped)
    stripped = re.sub(r"\s+([.,!?;:])", r"\1", stripped)
    return stripped.strip()


def _parse_points(text: str) -> List[PointOut]:
    out: List[PointOut] = []
    for m in POINT_TAG_RE.finditer(text):
        out.append(PointOut(
            x=int(m.group(1)),
            y=int(m.group(2)),
            label=m.group(3).strip(),
            screen=m.group(4),
        ))
    return out


def _build_user_text(req: CompanionQueryRequest) -> str:
    lines = [f'User says: "{req.text}"']
    if req.cursor_pos is not None:
        lines.append(
            f"Cursor position: ({req.cursor_pos.x}, {req.cursor_pos.y}) on {req.cursor_pos.screen}"
        )
    lines.append(
        "Screens (give POINT coordinates in IMAGE pixels — use the image dimensions below):"
    )
    for i, d in enumerate(req.image_dims):
        lines.append(f"  screen{i}: image is {d.w}x{d.h} px")
    return "\n".join(lines)


def _build_messages(req: CompanionQueryRequest) -> list[dict]:
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Include trimmed history as text-only turns (no images — past screenshots
    # are stale). Strip any POINT tags from prior assistant turns so the model
    # doesn't think it needs to re-point at the same coords.
    for turn in req.conversation_history[-6:]:
        if turn.role not in ("user", "assistant"):
            continue
        content = turn.content
        if turn.role == "assistant":
            content = _strip_point_tags(content)
        if content:
            messages.append({"role": turn.role, "content": content})

    # Current turn: text + all screenshots
    messages.append({
        "role": "user",
        "content": _build_user_text(req),
        "images": req.images,
    })
    return messages


@router.post("/query", response_model=CompanionQueryResponse)
async def companion_query(req: CompanionQueryRequest) -> CompanionQueryResponse:
    if len(req.images) != len(req.image_dims):
        raise HTTPException(
            status_code=400,
            detail=f"images ({len(req.images)}) and image_dims ({len(req.image_dims)}) length mismatch",
        )

    client = _get_ollama_client()
    messages = _build_messages(req)

    t0 = time.time()
    try:
        resp = client.chat(model=COMPANION_MODEL, messages=messages)
    except Exception as exc:
        logger.error(f"[companion] {COMPANION_MODEL} failed: {exc}")
        raise HTTPException(
            status_code=502,
            detail=f"vision model unavailable: {exc}",
        ) from exc
    latency_ms = int((time.time() - t0) * 1000)

    raw = (resp.get("message", {}) or {}).get("content", "") or ""
    points = _parse_points(raw)
    spoken = _strip_point_tags(raw)

    logger.info(
        f"[companion] query ok — {len(points)} points, {latency_ms}ms, model={COMPANION_MODEL}"
    )

    return CompanionQueryResponse(
        response_text=spoken,
        raw_text=raw,
        points=points,
        model_used=COMPANION_MODEL,
        latency_ms=latency_ms,
        conversation_id=req.conversation_id,
    )


@router.get("/health")
async def companion_health() -> dict:
    """Quick liveness probe; doesn't actually hit the model."""
    return {"ok": True, "model": COMPANION_MODEL}
