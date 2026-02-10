"""Audio Transcriber tool — speech-to-text from audio/video files using Whisper."""

import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

# Supported audio/video extensions
AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".wma", ".opus", ".webm"}
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm"}
ALL_EXTENSIONS = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS

TRANSCRIPTS_DIR = Path(__file__).parent.parent.parent / "data" / "transcripts"

# Whisper model singleton (lazy loaded)
_whisper_model = None
_whisper_available: Optional[bool] = None


def _check_whisper_available() -> bool:
    """Check if Whisper is installed."""
    global _whisper_available
    if _whisper_available is not None:
        return _whisper_available
    try:
        import whisper
        _whisper_available = True
    except ImportError:
        _whisper_available = False
        logger.info("[AudioTranscriber] Whisper not installed (pip install openai-whisper)")
    return _whisper_available


def _load_whisper(model_size: str = "base"):
    """Lazy-load Whisper model."""
    global _whisper_model

    if _whisper_model is not None:
        return _whisper_model

    if not _check_whisper_available():
        raise RuntimeError("Whisper not installed. Run: pip install openai-whisper")

    import whisper
    logger.info(f"[AudioTranscriber] Loading Whisper model '{model_size}'...")

    try:
        _whisper_model = whisper.load_model(model_size)
        logger.info(f"[AudioTranscriber] Whisper '{model_size}' loaded")
    except Exception as e:
        raise RuntimeError(f"Whisper model loading failed: {e}") from e

    return _whisper_model


class AudioTranscriberTool:
    """Transcribe audio/video files to text using OpenAI Whisper."""

    name = "audio_transcriber"
    description = "Transcribe audio and video files to text using Whisper"

    def __init__(self, model_size: str = "base"):
        self._model_size = model_size
        TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

    def transcribe(self, file_path: str, language: str = None,
                   model_size: str = None, task: str = "transcribe") -> dict:
        """Transcribe an audio or video file.

        Args:
            file_path: Path to audio/video file
            language: Language code (e.g., 'en', 'es', 'fr') or None for auto-detect
            model_size: Whisper model size (tiny/base/small/medium/large)
            task: 'transcribe' or 'translate' (translate to English)
        """
        p = Path(file_path)
        if not p.exists():
            return {"success": False, "error": f"File not found: {file_path}"}

        if p.suffix.lower() not in ALL_EXTENSIONS:
            return {"success": False, "error": f"Unsupported format: {p.suffix}. Supported: {', '.join(sorted(ALL_EXTENSIONS))}"}

        # Check if ffmpeg is needed for video files
        if p.suffix.lower() in VIDEO_EXTENSIONS:
            try:
                import subprocess
                result = subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=5)
                if result.returncode != 0:
                    return {"success": False, "error": "ffmpeg required for video files but not working"}
            except FileNotFoundError:
                return {"success": False, "error": "ffmpeg required for video files. Install ffmpeg first."}
            except Exception:
                pass

        size = model_size or self._model_size
        try:
            model = _load_whisper(size)
        except RuntimeError as e:
            return {"success": False, "error": str(e)}

        # Transcribe
        start = time.time()
        try:
            options = {"task": task}
            if language:
                options["language"] = language

            result = model.transcribe(str(p), **options)
            elapsed = time.time() - start

        except Exception as e:
            return {"success": False, "error": f"Transcription failed: {e}"}

        text = result.get("text", "").strip()
        detected_lang = result.get("language", "unknown")
        segments = result.get("segments", [])

        # Save transcript
        transcript_name = f"{p.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        transcript_path = TRANSCRIPTS_DIR / transcript_name
        try:
            with open(transcript_path, "w", encoding="utf-8") as f:
                f.write(f"Source: {p.name}\n")
                f.write(f"Language: {detected_lang}\n")
                f.write(f"Duration: {elapsed:.1f}s processing\n")
                f.write(f"{'='*60}\n\n")
                f.write(text)
                if segments:
                    f.write(f"\n\n{'='*60}\nTIMESTAMPED SEGMENTS:\n{'='*60}\n\n")
                    for seg in segments:
                        start_t = seg.get("start", 0)
                        end_t = seg.get("end", 0)
                        seg_text = seg.get("text", "").strip()
                        f.write(f"[{self._format_time(start_t)} -> {self._format_time(end_t)}] {seg_text}\n")
        except IOError:
            transcript_path = None

        # Build segment summaries
        segment_data = []
        for seg in segments[:50]:
            segment_data.append({
                "start": round(seg.get("start", 0), 2),
                "end": round(seg.get("end", 0), 2),
                "text": seg.get("text", "").strip(),
            })

        return {
            "success": True,
            "text": text,
            "language": detected_lang,
            "segments": segment_data,
            "segment_count": len(segments),
            "elapsed_seconds": round(elapsed, 2),
            "transcript_path": str(transcript_path) if transcript_path else None,
            "source_file": str(p),
            "model": size,
            "response": f"Transcribed '{p.name}' ({detected_lang}, {elapsed:.1f}s):\n{text[:1000]}"
                        + (f"\n...[{len(text)} chars total]" if len(text) > 1000 else "")
        }

    def _format_time(self, seconds: float) -> str:
        """Format seconds to MM:SS.ms."""
        m = int(seconds // 60)
        s = seconds % 60
        return f"{m:02d}:{s:05.2f}"

    def list_transcripts(self) -> dict:
        """List saved transcripts."""
        transcripts = sorted(TRANSCRIPTS_DIR.glob("*.txt"), key=lambda p: p.stat().st_mtime, reverse=True)
        items = []
        for t in transcripts[:20]:
            items.append({
                "name": t.name,
                "path": str(t),
                "size": t.stat().st_size,
                "modified": datetime.fromtimestamp(t.stat().st_mtime).isoformat(),
            })

        formatted = [f"  {t['name']} ({t['size']} bytes)" for t in items]
        return {
            "success": True,
            "count": len(items),
            "transcripts": items,
            "response": f"{len(items)} transcript(s):\n" + "\n".join(formatted) if items else "No transcripts saved"
        }

    def read_transcript(self, name: str) -> dict:
        """Read a saved transcript."""
        p = TRANSCRIPTS_DIR / name
        if not p.exists():
            # Try adding .txt
            p = TRANSCRIPTS_DIR / (name + ".txt")
        if not p.exists():
            return {"success": False, "error": f"Transcript not found: {name}"}

        try:
            text = p.read_text(encoding="utf-8")
            return {"success": True, "text": text, "path": str(p),
                    "response": text[:2000]}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def detect_language(self, file_path: str) -> dict:
        """Detect the language of an audio file without full transcription."""
        p = Path(file_path)
        if not p.exists():
            return {"success": False, "error": f"File not found: {file_path}"}

        try:
            model = _load_whisper(self._model_size)
        except RuntimeError as e:
            return {"success": False, "error": str(e)}

        try:
            import whisper
            audio = whisper.load_audio(str(p))
            audio = whisper.pad_or_trim(audio)
            mel = whisper.log_mel_spectrogram(audio).to(model.device)
            _, probs = model.detect_language(mel)

            top_langs = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5]
            detected = top_langs[0][0]
            confidence = top_langs[0][1]

            return {
                "success": True,
                "language": detected,
                "confidence": round(confidence, 4),
                "top_languages": {k: round(v, 4) for k, v in top_langs},
                "response": f"Detected language: {detected} ({confidence:.1%} confidence)"
            }
        except Exception as e:
            return {"success": False, "error": f"Language detection failed: {e}"}

    def status(self) -> dict:
        """Check if Whisper is available and loaded."""
        available = _check_whisper_available()
        loaded = _whisper_model is not None
        return {
            "success": True,
            "whisper_installed": available,
            "model_loaded": loaded,
            "model_size": self._model_size,
            "response": f"Whisper: {'installed' if available else 'NOT installed'}, "
                        f"model: {self._model_size} ({'loaded' if loaded else 'not loaded'})"
        }

    # -- Dispatch -----------------------------------------------------------

    def execute(self, action: str, **kwargs) -> dict:
        action_lower = action.lower().strip()

        # Status
        if action_lower in ("status", "info", "check"):
            return self.status()

        # List transcripts
        if action_lower in ("list", "transcripts", "list_transcripts"):
            return self.list_transcripts()

        # Read transcript
        if action_lower.startswith("read_transcript") or action_lower.startswith("read transcript"):
            name = kwargs.get("name") or (action.split(None, 2)[-1] if len(action.split()) > 2 else "")
            return self.read_transcript(name.strip())

        # Detect language
        if action_lower.startswith("detect") or action_lower.startswith("language"):
            file_path = kwargs.get("file_path") or kwargs.get("path")
            if not file_path and len(action.split()) > 1:
                file_path = action.split(None, 1)[-1].strip()
            if file_path:
                return self.detect_language(file_path)
            return {"success": False, "error": "No file path specified"}

        # Translate (transcribe + translate to English)
        if action_lower.startswith("translate"):
            file_path = kwargs.get("file_path") or kwargs.get("path")
            if not file_path and len(action.split()) > 1:
                file_path = action.split(None, 1)[-1].strip()
            if file_path:
                return self.transcribe(file_path, task="translate",
                                        model_size=kwargs.get("model_size"),
                                        language=kwargs.get("language"))
            return {"success": False, "error": "No file path specified"}

        # Default: transcribe
        file_path = kwargs.get("file_path") or kwargs.get("path")
        language = kwargs.get("language")
        model_size = kwargs.get("model_size")

        if not file_path:
            # Try to extract path from action
            # "transcribe /path/to/file.mp3" or just "/path/to/file.mp3"
            cleaned = re.sub(r'^transcribe\s+', '', action, flags=re.IGNORECASE).strip()
            if cleaned:
                file_path = cleaned

        if file_path:
            return self.transcribe(file_path, language=language, model_size=model_size)

        return {
            "success": False,
            "error": f"Could not parse: {action}. "
                     "Try: 'transcribe <file_path>', 'translate <file_path>', 'detect <file_path>', 'list', 'status'"
        }


# Singleton
audio_transcriber_tool = AudioTranscriberTool()
