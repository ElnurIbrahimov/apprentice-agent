"""
Page summarization via local Ollama.
"""

import logging
import math
import os
from typing import Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from api.auth import require_api_key
from api.utils import safe_error_detail

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/summarize", tags=["summarize"], dependencies=[Depends(require_api_key)])

OLLAMA_URL = (os.getenv("OLLAMA_BASE_URL") or os.getenv("OLLAMA_HOST", "http://localhost:11434")) + "/api/generate"
DEFAULT_MODEL = os.getenv("AURA_AGENT_MODEL", "nemotron-3-super:cloud")
MAX_TEXT_CHARS = 50000


class SummarizeRequest(BaseModel):
    # 500 KB of raw text is more than enough; prevents memory exhaustion
    text: str = Field(..., max_length=500_000)
    url: str = Field("", max_length=2048)
    title: str = Field("", max_length=500)
    format: str = "bullets"   # "bullets" | "paragraph" | "tldr"
    model: Optional[str] = None


PROMPTS = {
    "bullets": (
        "Summarize the following web page content into 5-7 concise bullet points. "
        "Each bullet should capture a key idea or fact. "
        "Format as a markdown list using '- ' prefix. "
        "Be direct and information-dense. No preamble.\n\n"
    ),
    "paragraph": (
        "Summarize the following web page content in 3-4 sentences. "
        "Capture the main topic, key arguments, and most important takeaway. "
        "Write in clear, plain prose. No preamble.\n\n"
    ),
    "tldr": (
        "Give a TL;DR of the following web page content in 1-2 sentences maximum. "
        "Be extremely concise. No preamble.\n\n"
    ),
}


def estimate_reading_time(char_count: int) -> float:
    """Estimate reading time in minutes at ~238 wpm average."""
    words = char_count / 5.0  # ~5 chars/word
    return words / 238.0


def format_reading_time(minutes: float) -> str:
    if minutes < 1:
        return "less than a minute"
    elif minutes < 60:
        rounded = math.ceil(minutes)
        return f"~{rounded} min"
    else:
        h = int(minutes // 60)
        m = int(minutes % 60)
        return f"~{h}h {m}m" if m else f"~{h}h"


@router.post("/page")
async def summarize_page(req: SummarizeRequest):
    """Summarize a web page's text content."""
    text = req.text.strip()
    if not text:
        raise HTTPException(400, "No text provided")

    truncated = False
    if len(text) > MAX_TEXT_CHARS:
        text = text[:MAX_TEXT_CHARS]
        truncated = True

    fmt = req.format if req.format in PROMPTS else "bullets"
    system_prompt = PROMPTS[fmt]

    context_header = ""
    if req.title:
        context_header += f"Page Title: {req.title}\n"
    if req.url:
        context_header += f"URL: {req.url}\n"
    if context_header:
        context_header += "\n"

    full_prompt = system_prompt + context_header + "Content:\n" + text

    if truncated:
        full_prompt += "\n\n[Note: Content was truncated to 50,000 characters due to length.]"

    model = req.model or DEFAULT_MODEL

    try:
        async with httpx.AsyncClient(timeout=45.0) as client:
            resp = await client.post(
                OLLAMA_URL,
                json={
                    "model": model,
                    "prompt": full_prompt,
                    "stream": False,
                },
            )
            resp.raise_for_status()
            data = resp.json()
    except httpx.TimeoutException:
        raise HTTPException(504, "Summarization timed out (45s). Try a shorter page or faster model.")
    except httpx.HTTPStatusError as e:
        logger.error("[Summarize] Ollama HTTP error: %s", e)
        raise HTTPException(502, f"Ollama error: {e.response.status_code}")
    except Exception as e:
        logger.error("[Summarize] Unexpected error: %s", e)
        raise HTTPException(500, safe_error_detail(e, "Summarization failed"))

    summary = data.get("response", "").strip()
    if not summary:
        raise HTTPException(500, "Model returned empty summary")

    # Word count of original text
    original_words = len(text.split())

    # Reading time saved: original vs summary
    original_minutes = estimate_reading_time(len(text))
    summary_minutes = estimate_reading_time(len(summary))
    saved_minutes = max(0.0, original_minutes - summary_minutes)

    if saved_minutes < 0.5:
        reading_time_saved = "a few seconds"
    else:
        reading_time_saved = format_reading_time(saved_minutes) + " saved"

    return {
        "summary": summary,
        "word_count": original_words,
        "reading_time_saved": reading_time_saved,
        "truncated": truncated,
    }
