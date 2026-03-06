"""
Audio transcription via OpenAI Whisper (local).
Requires: pip install openai-whisper  +  ffmpeg on PATH
"""

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
    suffix = os.path.splitext(file.filename or ".webm")[1] or ".webm"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(data)
        tmp = f.name

    try:
        import whisper as _whisper
        model = _whisper.load_model("base")
        result = model.transcribe(tmp)
        return {"text": result["text"]}
    except Exception as e:
        raise HTTPException(500, f"Whisper error (ensure ffmpeg is on PATH): {e}")
    finally:
        try:
            os.unlink(tmp)
        except Exception:
            pass
