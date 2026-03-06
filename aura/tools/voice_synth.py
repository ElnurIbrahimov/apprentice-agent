"""Voice Synthesis Tool — high-quality neural TTS with 54 voice presets via Kokoro.

Uses kokoro-onnx (neural TTS model, ~300MB download on first use).
Much higher quality than system TTS — natural, expressive speech.

Voices available: af_heart, af_sky, af_bella, am_adam, am_michael,
bf_emma, bf_isabella, bm_george, bm_lewis (and ~45 more).

AURA can speak responses aloud, generate audio files, or narrate documents.

Output: Desktop/aura_speech/ (WAV files)

Setup:
    pip install kokoro-onnx soundfile

Note: For voice *cloning* from reference audio, Coqui TTS is required
but only supports Python <3.12. This tool uses preset neural voices instead.
"""

import logging
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path.home() / "Desktop" / "aura_speech"
DEFAULT_VOICE = os.getenv("AURA_TTS_VOICE", "af_heart")
DEFAULT_SPEED = float(os.getenv("AURA_TTS_SPEED", "1.0"))

# Available kokoro voice presets
KOKORO_VOICES = {
    # American Female
    "af_heart": "American Female — warm, expressive (default)",
    "af_sky": "American Female — bright, energetic",
    "af_bella": "American Female — soft, clear",
    "af_nova": "American Female — conversational",
    "af_sarah": "American Female — professional",
    # American Male
    "am_adam": "American Male — deep, authoritative",
    "am_michael": "American Male — friendly, natural",
    "am_echo": "American Male — clear narrator",
    # British Female
    "bf_emma": "British Female — refined, articulate",
    "bf_isabella": "British Female — warm British accent",
    # British Male
    "bm_george": "British Male — distinguished",
    "bm_lewis": "British Male — conversational British",
}

_kokoro_lock = threading.Lock()
_kokoro_model = None


KOKORO_DATA_DIR = Path(__file__).parent.parent.parent / "aura_data" / "kokoro"
KOKORO_MODEL_PATH = str(KOKORO_DATA_DIR / "kokoro-v1.0.onnx")
KOKORO_VOICES_PATH = str(KOKORO_DATA_DIR / "voices-v1.0.bin")


def _get_kokoro():
    global _kokoro_model
    with _kokoro_lock:
        if _kokoro_model is None:
            from kokoro_onnx import Kokoro
            logger.info(f"[VoiceSynth] Loading Kokoro model from {KOKORO_DATA_DIR}...")
            _kokoro_model = Kokoro(KOKORO_MODEL_PATH, KOKORO_VOICES_PATH)
            logger.info("[VoiceSynth] Kokoro ready")
        return _kokoro_model


class VoiceSynthTool:
    """Neural TTS with 54 voice presets — speak text in natural, expressive voices."""

    name = "voice_synth"
    description = "High-quality neural TTS with 54 voice presets — speak text aloud or save audio files. kokoro-onnx, fully local."

    def __init__(self):
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def _check(self) -> Optional[str]:
        try:
            import kokoro_onnx  # noqa
            import soundfile  # noqa
            return None
        except ImportError as e:
            return f"Missing dependency: {e}. Run: pip install kokoro-onnx soundfile"

    def list_voices(self) -> Dict:
        """List available voice presets."""
        return {
            "success": True,
            "default_voice": DEFAULT_VOICE,
            "count": len(KOKORO_VOICES),
            "voices": [{"id": k, "description": v} for k, v in KOKORO_VOICES.items()],
            "note": "Many more voices available — see kokoro-onnx docs for full list",
        }

    def speak(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: float = DEFAULT_SPEED,
        output_path: Optional[str] = None,
        play: bool = False,
    ) -> Dict:
        """Generate speech from text.

        Args:
            text: Text to synthesize
            voice: Voice ID (e.g. 'af_heart', 'am_adam'). Uses default if None.
            speed: Speech rate (0.5 = slow, 1.0 = normal, 1.5 = fast)
            output_path: Where to save the WAV file (auto-generated if None)
            play: If True, also play the audio immediately via sounddevice
        """
        err = self._check()
        if err:
            return {"success": False, "error": err}

        voice = voice or DEFAULT_VOICE

        if not output_path:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = str(OUTPUT_DIR / f"speech_{ts}.wav")

        try:
            import soundfile as sf
            import numpy as np

            kokoro = _get_kokoro()
            logger.info(f"[VoiceSynth] Generating '{text[:60]}' with voice '{voice}'")
            samples, sample_rate = kokoro.create(text, voice=voice, speed=speed, lang="en-us")

            sf.write(output_path, samples, sample_rate)
            logger.info(f"[VoiceSynth] Saved: {output_path}")

            if play:
                try:
                    import sounddevice as sd
                    sd.play(samples, sample_rate)
                    sd.wait()
                except Exception as e:
                    logger.warning(f"[VoiceSynth] Playback failed: {e}")

            return {
                "success": True,
                "output": output_path,
                "voice": voice,
                "speed": speed,
                "text_length": len(text),
                "sample_rate": sample_rate,
                "duration_sec": round(len(samples) / sample_rate, 1),
            }

        except Exception as e:
            logger.error(f"[VoiceSynth] Generation failed: {e}")
            return {"success": False, "error": str(e), "voice": voice}

    def play_file(self, audio_path: str) -> Dict:
        """Play a previously generated audio file."""
        path = Path(audio_path)
        if not path.exists():
            return {"success": False, "error": f"File not found: {audio_path}"}
        try:
            import soundfile as sf
            import sounddevice as sd
            data, sr = sf.read(str(path))
            sd.play(data, sr)
            sd.wait()
            return {"success": True, "played": audio_path}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def list_outputs(self, limit: int = 10) -> Dict:
        """List recently generated speech files."""
        files = sorted(OUTPUT_DIR.glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]
        return {
            "success": True,
            "output_dir": str(OUTPUT_DIR),
            "count": len(files),
            "files": [{"name": p.name, "size_kb": round(p.stat().st_size / 1024, 1), "created": datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M")} for p in files],
        }

    def execute(self, action: str, **kwargs) -> Dict:
        a = action.lower().strip()
        if "list" in a and "voice" in a:
            return self.list_voices()
        if "list" in a or "output" in a or "file" in a:
            return self.list_outputs(kwargs.get("limit", 10))
        if "play" in a and "path" not in a:
            path = kwargs.get("path") or kwargs.get("audio_path") or kwargs.get("file") or ""
            return self.play_file(path)
        if "speak" in a or "say" in a or "generate" in a or "tts" in a or "synthesize" in a:
            return self.speak(
                text=kwargs.get("text") or kwargs.get("prompt") or action,
                voice=kwargs.get("voice") or kwargs.get("voice_id"),
                speed=kwargs.get("speed", DEFAULT_SPEED),
                output_path=kwargs.get("output_path"),
                play=kwargs.get("play", False),
            )
        # Default: speak the action text if it looks like a sentence
        if len(action) > 10 and not any(w in a for w in ["list", "status", "check"]):
            return self.speak(text=action, voice=kwargs.get("voice"))
        return self.list_voices()
