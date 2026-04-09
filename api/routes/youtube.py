"""
YouTube video summarizer.
Fetches transcript via youtube-transcript-api, summarizes with Ollama.
"""

import logging
import re

import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from api.auth import require_api_key
from api.utils import safe_error_detail

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/youtube", tags=["youtube"], dependencies=[Depends(require_api_key)])

import os as _os

OLLAMA_URL = (_os.getenv("OLLAMA_BASE_URL") or _os.getenv("OLLAMA_HOST", "http://localhost:11434")) + "/api/generate"
def _get_summary_model() -> str:
    try:
        from aura.config import Config
        return Config.MODEL_FAST
    except Exception:
        return "nemotron-3-super:cloud"
TRANSCRIPT_CHAR_LIMIT = 12000


def _extract_video_id(url: str) -> str | None:
    """Extract YouTube video ID from various URL formats."""
    patterns = [
        r"(?:youtube\.com/watch\?.*v=)([a-zA-Z0-9_-]{11})",
        r"(?:youtu\.be/)([a-zA-Z0-9_-]{11})",
        r"(?:youtube\.com/embed/)([a-zA-Z0-9_-]{11})",
        r"(?:youtube\.com/shorts/)([a-zA-Z0-9_-]{11})",
    ]
    for pattern in patterns:
        m = re.search(pattern, url)
        if m:
            return m.group(1)
    return None


async def _fetch_video_meta(video_id: str) -> dict:
    """Fetch title and channel from YouTube page og: meta tags."""
    title = ""
    channel = ""
    try:
        url = f"https://www.youtube.com/watch?v={video_id}"
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }
        async with httpx.AsyncClient(timeout=10, follow_redirects=True) as c:
            r = await c.get(url, headers=headers)
        html = r.text

        # og:title
        m = re.search(r'<meta property="og:title" content="([^"]*)"', html)
        if m:
            title = m.group(1)

        # channel name from itemprop or author meta
        mc = re.search(r'"ownerChannelName":"([^"]*)"', html)
        if mc:
            channel = mc.group(1)
        else:
            mc2 = re.search(r'"author":"([^"]*)"', html)
            if mc2:
                channel = mc2.group(1)

        # duration from og:video:duration or approximation from html
        # YouTube embeds duration in structured data
        md = re.search(r'"lengthSeconds":"(\d+)"', html)
        duration = ""
        if md:
            secs = int(md.group(1))
            h = secs // 3600
            m_val = (secs % 3600) // 60
            s = secs % 60
            if h:
                duration = f"{h}:{m_val:02d}:{s:02d}"
            else:
                duration = f"{m_val}:{s:02d}"

    except Exception as e:
        logger.warning("[YouTube] Failed to fetch meta: %s", e)
        title = f"YouTube Video ({video_id})"
        channel = ""
        duration = ""

    return {"title": title, "channel": channel, "duration": duration}


def _get_transcript(video_id: str) -> tuple[str, str]:
    """
    Get transcript text and a short snippet.
    Returns (full_text_truncated, snippet).
    Raises HTTPException on failure.
    """
    try:
        from youtube_transcript_api import (
            NoTranscriptFound,  # noqa: F401
            TranscriptsDisabled,  # noqa: F401
            YouTubeTranscriptApi,
        )
    except ImportError:
        raise HTTPException(
            503,
            "youtube-transcript-api not installed. Run: pip install youtube-transcript-api"
        )

    try:
        entries = YouTubeTranscriptApi.get_transcript(video_id)
    except Exception as e:
        err_str = str(e).lower()
        if "disabled" in err_str or "transcriptsdisabled" in err_str:
            raise HTTPException(422, "Transcripts are disabled for this video.")
        if "notranscriptfound" in err_str or "no transcript" in err_str:
            raise HTTPException(422, "No transcript available for this video (may be private or have no captions).")
        raise HTTPException(500, safe_error_detail(e, "Failed to fetch transcript"))

    full_text = " ".join(e.get("text", "") for e in entries)
    snippet = full_text[:400].strip() + ("…" if len(full_text) > 400 else "")
    truncated = full_text[:TRANSCRIPT_CHAR_LIMIT]
    return truncated, snippet


async def _summarize_with_ollama(transcript: str, title: str) -> tuple[str, list[str]]:
    """
    Call Ollama to produce a summary + 5 key bullet points.
    Returns (summary_text, [key_point, ...]).
    """
    prompt = f"""You are summarizing a YouTube video transcript. Be concise and clear.

Video title: {title}

Transcript:
{transcript}

Respond in exactly this format (no extra text before or after):

SUMMARY:
<2-4 sentence summary of the video>

KEY POINTS:
- <key point 1>
- <key point 2>
- <key point 3>
- <key point 4>
- <key point 5>"""

    payload = {
        "model": _get_summary_model(),
        "prompt": prompt,
        "stream": False,
    }

    try:
        async with httpx.AsyncClient(timeout=120) as c:
            r = await c.post(OLLAMA_URL, json=payload)
        r.raise_for_status()
        data = r.json()
        raw = data.get("response", "").strip()
    except httpx.ConnectError:
        raise HTTPException(503, "Ollama is not running. Start it with: ollama serve")
    except Exception as e:
        raise HTTPException(500, safe_error_detail(e, "LLM summarization failed"))

    # Parse summary
    summary = ""
    key_points = []

    summary_match = re.search(r"SUMMARY:\s*\n(.*?)(?:\n\s*KEY POINTS:|$)", raw, re.DOTALL | re.IGNORECASE)
    if summary_match:
        summary = summary_match.group(1).strip()

    kp_match = re.search(r"KEY POINTS:\s*\n(.*?)$", raw, re.DOTALL | re.IGNORECASE)
    if kp_match:
        kp_block = kp_match.group(1).strip()
        for line in kp_block.split("\n"):
            line = line.strip()
            if line.startswith("- "):
                key_points.append(line[2:].strip())
            elif line.startswith("* "):
                key_points.append(line[2:].strip())
            elif re.match(r"^\d+\.\s+", line):
                key_points.append(re.sub(r"^\d+\.\s+", "", line).strip())

    # Fallback: if parsing failed, use the raw response as summary
    if not summary:
        summary = raw[:800]
    if not key_points:
        key_points = ["See full summary above."]

    return summary, key_points[:5]


class TranscriptRequest(BaseModel):
    video_id: str = Field(..., min_length=11, max_length=11)
    language: str = "en"


# pip install youtube-transcript-api (already used by /summarize above)
@router.post("/transcript")
async def get_transcript(request: TranscriptRequest):
    """Fetch raw YouTube video transcript with timestamps."""
    import asyncio

    def _fetch():
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
        except ImportError:
            raise HTTPException(
                501,
                "youtube-transcript-api not installed. Run: pip install youtube-transcript-api",
            )
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(request.video_id)
            try:
                transcript = transcript_list.find_transcript([request.language])
            except Exception:
                transcript = transcript_list.find_generated_transcript([request.language])
            entries = transcript.fetch()
            return {
                "transcript": [
                    {"text": e["text"], "start": e["start"], "duration": e["duration"]}
                    for e in entries
                ],
                "language": request.language,
                "is_generated": transcript.is_generated,
            }
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(404, f"Transcript not available: {e!s}")

    loop = asyncio.get_running_loop()
    try:
        return await loop.run_in_executor(None, _fetch)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, safe_error_detail(e, "Transcript fetch error"))


@router.post("/summarize")
async def summarize_youtube(body: dict):
    """
    Summarize a YouTube video by URL.

    Body: { "url": "https://youtube.com/watch?v=..." }
    Returns: { "title", "channel", "duration", "summary", "key_points", "transcript_snippet" }
    """
    url = (body.get("url") or "").strip()
    if not url:
        raise HTTPException(400, "url is required")

    video_id = _extract_video_id(url)
    if not video_id:
        raise HTTPException(400, "Could not extract video ID from URL. Supported: youtube.com/watch?v=, youtu.be/, youtube.com/shorts/")

    logger.info("[YouTube] Summarizing video_id=%s", video_id)

    # Fetch transcript (sync call — use run_in_executor to avoid blocking the event loop)
    import asyncio
    loop = asyncio.get_running_loop()

    try:
        transcript_text, snippet = await loop.run_in_executor(
            None, _get_transcript, video_id
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, safe_error_detail(e, "Transcript error"))

    # Fetch meta and summarize in parallel
    meta_task = asyncio.create_task(_fetch_video_meta(video_id))
    title_for_llm = f"Video {video_id}"  # placeholder until meta loads

    # We need title for the prompt — fetch meta first (fast, ~1s)
    meta = await meta_task
    title = meta["title"] or title_for_llm

    summary, key_points = await _summarize_with_ollama(transcript_text, title)

    return {
        "title": meta["title"],
        "channel": meta["channel"],
        "duration": meta["duration"],
        "summary": summary,
        "key_points": key_points,
        "transcript_snippet": snippet,
    }
