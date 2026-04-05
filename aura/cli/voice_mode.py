"""Voice conversation mode — wraps VoiceConversation with error handling."""
from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def run_voice_mode(
    agent: Any,
    enable_barge_in: bool = True,
    whisper_model: str = "base",
    bridge: Optional[Any] = None,
) -> None:
    """Start voice conversation mode.

    Falls back to chat mode if voice dependencies are unavailable.
    """
    try:
        from aura.tools.voice import VoiceConversation
    except ImportError:
        print("\n[AURA] Voice mode requires extra dependencies.")
        print("Install with: pip install aura[voice]")
        print("Falling back to chat mode...\n")
        from .chat_loop import run_chat_mode
        run_chat_mode(agent, speak=True, bridge=bridge)
        return

    try:
        conversation = VoiceConversation(
            agent,
            whisper_model=whisper_model,
            enable_barge_in=enable_barge_in,
        )
    except (OSError, RuntimeError) as e:
        logger.error("Voice initialization failed: %s", e)
        print(f"\n[AURA] Voice initialization failed: {e}")
        print("Check your audio devices and microphone permissions.")
        print("Falling back to chat mode...\n")
        from .chat_loop import run_chat_mode
        run_chat_mode(agent, speak=True, bridge=bridge)
        return

    try:
        conversation.start()
    except KeyboardInterrupt:
        print("\n[dim]Voice mode ended.[/dim]")
    except Exception as e:
        logger.error("Voice mode crashed: %s", e, exc_info=True)
        print(f"\n[AURA] Voice mode error: {e}")
    finally:
        if bridge:
            bridge.stop()
