"""Voice reply synthesizer for Telegram — generates voice notes from text responses."""

import asyncio
import logging
import os
import tempfile
from typing import Optional

logger = logging.getLogger(__name__)

class VoiceReply:
    """Synthesizes voice replies using Kokoro-ONNX with pyttsx3 fallback."""

    async def synthesize(self, text: str, voice: str = None, max_chars: int = 500) -> Optional[str]:
        """Generate a WAV file from text. Returns file path or None on failure.
        Truncates to max_chars to keep audio reasonable length."""
        truncated = text[:max_chars]
        # Try Kokoro first (high quality), fall back to pyttsx3
        path = await asyncio.to_thread(self._try_kokoro, truncated, voice)
        if not path:
            path = await asyncio.to_thread(self._try_pyttsx3, truncated)
        return path

    def _try_kokoro(self, text: str, voice: str = None) -> Optional[str]:
        try:
            from aura.tools.voice_synth import VoiceSynthTool
            synth = VoiceSynthTool()
            result = synth.speak(text, voice=voice)
            if result.get("success"):
                return result["output"]
            logger.warning(f"[VoiceReply] Kokoro failed: {result.get('error')}")
        except Exception as e:
            logger.debug(f"[VoiceReply] Kokoro unavailable: {e}")
        return None

    def _try_pyttsx3(self, text: str) -> Optional[str]:
        try:
            import pyttsx3
            fd, tmp_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            engine = pyttsx3.init()
            engine.setProperty('rate', 175)
            engine.save_to_file(text, tmp_path)
            engine.runAndWait()
            if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 0:
                return tmp_path
            logger.warning("[VoiceReply] pyttsx3 produced empty file")
        except Exception as e:
            logger.debug(f"[VoiceReply] pyttsx3 unavailable: {e}")
        return None
