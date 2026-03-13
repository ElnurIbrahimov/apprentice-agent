"""
Audio transcription via OpenAI Whisper (local).
Requires: pip install openai-whisper  +  ffmpeg on PATH
"""

import asyncio
import os
import tempfile
import logging
from fastapi import APIRouter, File, UploadFile, HTTPException, Depends

from api.auth import require_api_key

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["transcribe"], dependencies=[Depends(require_api_key)])

# Cache the Whisper model to avoid reloading ~300MB per request
_whisper_model = None
_whisper_model_lock = __import__("threading").Lock()


def _get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        with _whisper_model_lock:
            if _whisper_model is None:
                import whisper as _w
                _whisper_model = _w.load_model("base")
                logger.info("[Transcribe] Whisper base model loaded and cached")
    return _whisper_model


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
        loop = asyncio.get_running_loop()

        def _run_whisper():
            model = _get_whisper_model()
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
