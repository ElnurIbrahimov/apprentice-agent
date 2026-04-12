"""
Audio transcription via OpenAI Whisper (local).
Requires: pip install openai-whisper  +  ffmpeg on PATH
"""

import asyncio
import logging
import os
import tempfile

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from api.auth import require_api_key
from api.utils import safe_error_detail

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

    # Validate file extension BEFORE reading the body to avoid buffering disallowed files
    suffix = os.path.splitext(file.filename or ".webm")[1] or ".webm"
    _ALLOWED_AUDIO_SUFFIXES = {".webm", ".mp3", ".wav", ".ogg", ".flac", ".m4a", ".mp4", ".mpeg", ".mpga"}
    if suffix.lower() not in _ALLOWED_AUDIO_SUFFIXES:
        raise HTTPException(400, f"Unsupported audio format: {suffix}")

    # 100 MB limit — stream to avoid buffering entire file in memory
    _MAX_AUDIO_SIZE = 100 * 1024 * 1024
    chunks = []
    total = 0
    async for chunk in file.stream():
        total += len(chunk)
        if total > _MAX_AUDIO_SIZE:
            raise HTTPException(413, "Audio file too large. Maximum size is 100 MB.")
        chunks.append(chunk)
    data = b"".join(chunks)

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
        raise HTTPException(500, safe_error_detail(e, "Whisper error (ensure ffmpeg is on PATH)"))
    finally:
        try:
            os.unlink(tmp)
        except Exception:
            pass
