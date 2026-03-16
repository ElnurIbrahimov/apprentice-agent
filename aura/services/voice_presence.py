"""Voice presence stub — Sesame/PersonaPlex removed, using external voice provider.

This stub prevents ImportError in the 7+ files that lazy-import get_voice_presence().
All methods return safe defaults indicating voice is not available.
"""

import logging

logger = logging.getLogger(__name__)


class VoicePresenceStub:
    """No-op voice presence — real implementation removed."""

    _enabled = False

    def is_available(self) -> bool:
        return False

    def speak(self, text: str, **kwargs) -> dict:
        return {"success": False, "error": "Voice presence not configured"}

    def stop(self) -> None:
        pass

    def get_status(self) -> dict:
        return {"available": False, "reason": "Voice modules removed — using external provider"}


_instance = None


def get_voice_presence() -> VoicePresenceStub:
    global _instance
    if _instance is None:
        _instance = VoicePresenceStub()
    return _instance
