"""
Audio transcription via OpenAI Whisper (local).
Requires: pip install openai-whisper  +  ffmpeg on PATH
"""

import asyncio
import os
import tempfile
import logging
from fastapi import APIRouter, File, UploadFile, HTTPException

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["transcribe"])


@router.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    """Transcribe uploaded audio file using Whisper base model."""
    try:
        import whisper  # noqa: F401
    except ImportError:
        raise HTTPException(
            503,
            "whisper not installed. Run: pip install openai-whisper  "
            "(also requires ffmpeg on PATH)",
        )

    data = await file.read()
    # 100 MB limit — Whisper base handles up to ~2h audio; reject anything bigger
    if len(data) > 100 * 1024 * 1024:
        raise HTTPException(413, "Audio file too large. Maximum size is 100 MB.")
    suffix = os.path.splitext(file.filename or ".webm")[1] or ".webm"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(data)
        tmp = f.name

    try:
        import whisper as _whisper
        loop = asyncio.get_running_loop()

        def _run_whisper():
            model = _whisper.load_model("base")
            return model.transcribe(tmp)

        result = await loop.run_in_executor(None, _run_whisper)
        return {"text": result["text"]}
    except Exception as e:
        raise HTTPException(500, f"Whisper error (ensure ffmpeg is on PATH): {e}")
    finally:
        try:
            os.unlink(tmp)
        except Exception:
            pass
