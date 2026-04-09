"""Voice Manager - Hybrid voice system for Aura agent.

Switches between:
- Pipeline mode (Sesame CSM): Best voice quality, ~4.5GB VRAM
- Duplex mode (PersonaPlex): Natural conversation with interrupts, ~8GB VRAM

Handles VRAM management for RTX 4060 (8GB) - only ONE voice model at a time.
"""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# Disable triton on Windows
os.environ.setdefault('TORCHAO_NO_TRITON', '1')

import requests

# Torch is lazy-loaded — only needed for GPU voice features
_torch = None
_torch_checked = False

def _get_torch():
    """Lazy-load torch on first use. Returns (torch_module, is_available)."""
    global _torch, _torch_checked
    if not _torch_checked:
        try:
            import torch
            _torch = torch
        except ImportError:
            _torch = None
        _torch_checked = True
    return _torch, _torch is not None

# Keep backward-compat module-level flag (evaluated lazily via property-like access)
# Code that checks TORCH_AVAILABLE should use _get_torch() instead.

# Config import for dynamic model list
try:
    from ..config import Config
except ImportError:
    from config import Config

# SesameTTS and PersonaPlexTool lazy-loaded on first use to avoid startup penalty
SesameTTS = None
PersonaPlexTool = None
VOICE_MODELS_AVAILABLE = None  # None = not yet checked

def _ensure_voice_models():
    global SesameTTS, PersonaPlexTool, VOICE_MODELS_AVAILABLE
    if VOICE_MODELS_AVAILABLE is not None:
        return VOICE_MODELS_AVAILABLE
    try:
        from .personaplex.personaplex_tool import PersonaPlexTool as _PP
        from .sesame_tts import SesameTTS as _ST
        SesameTTS = _ST
        PersonaPlexTool = _PP
        VOICE_MODELS_AVAILABLE = True
    except ImportError:
        try:
            from personaplex.personaplex_tool import PersonaPlexTool as _PP
            from sesame_tts import SesameTTS as _ST
            SesameTTS = _ST
            PersonaPlexTool = _PP
            VOICE_MODELS_AVAILABLE = True
        except ImportError:
            VOICE_MODELS_AVAILABLE = False
    return VOICE_MODELS_AVAILABLE


class VoiceManager:
    """Manages hybrid voice system switching between Sesame and PersonaPlex."""

    # Keywords for automatic mode detection
    PIPELINE_KEYWORDS = [
        "read this", "say this", "speak", "read aloud", "read it",
        "tell me", "announce", "narrate", "pronounce"
    ]

    DUPLEX_KEYWORDS = [
        "talk to me", "let's chat", "voice chat", "conversation mode",
        "speak with me", "have a conversation", "real-time chat",
        "duplex mode", "full duplex"
    ]

    def __init__(self):
        self.name = "voice_manager"
        self.description = "Hybrid voice system manager (Sesame + PersonaPlex)"
        self.current_mode: Optional[str] = None
        _ensure_voice_models()
        self.sesame = SesameTTS() if SesameTTS else None
        self.personaplex = PersonaPlexTool() if PersonaPlexTool else None
        self._ollama_host = os.getenv("OLLAMA_HOST") or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.available = VOICE_MODELS_AVAILABLE

    def switch_to_pipeline(self) -> dict:
        """Switch to pipeline mode: Whisper STT -> LLM -> Sesame TTS.

        Best voice quality, supports other tools during conversation.

        Returns:
            dict with switch status
        """
        try:
            # Unload PersonaPlex first if running
            if self.current_mode == "duplex" and self.personaplex is not None:
                logger.debug("Stopping PersonaPlex server...")
                self.personaplex.stop_server()
                _t, _avail = _get_torch()
                if _avail and _t.cuda.is_available():
                    _t.cuda.empty_cache()

            # Load Sesame
            logger.debug("Loading Sesame CSM 1B...")
            result = self.sesame.load()

            if result["success"]:
                self.current_mode = "pipeline"
                return {
                    "success": True,
                    "mode": "pipeline",
                    "tts": "Sesame CSM 1B",
                    "message": "Pipeline mode active. Using Sesame for high-quality TTS.",
                    "vram_used": result.get("vram_used", "N/A")
                }
            else:
                return result

        except Exception as e:
            return {"success": False, "error": str(e)}

    def switch_to_duplex(self, voice: str | None = None, persona: str | None = None) -> dict:
        """Switch to duplex mode: PersonaPlex handles everything.

        Natural conversation with interrupts. Uses full GPU (~8GB).

        Args:
            voice: Optional voice ID (NATM1, NATF2, etc.)
            persona: Optional persona/system prompt

        Returns:
            dict with switch status
        """
        try:
            # Unload Sesame and free VRAM
            if self.current_mode == "pipeline":
                logger.debug("Unloading Sesame CSM...")
                self.sesame.unload()

            # Unload Ollama models to free VRAM (PersonaPlex needs full 8GB)
            self._unload_ollama_models()
            _t, _avail = _get_torch()
            if _avail and _t.cuda.is_available():
                _t.cuda.empty_cache()

            # Start PersonaPlex server
            logger.debug("Starting PersonaPlex server...")
            result = self.personaplex.start_server(
                voice=voice,
                persona=persona,
                cpu_offload=True  # Use CPU offload for 8GB GPU
            )

            if result["success"]:
                self.current_mode = "duplex"
                return {
                    "success": True,
                    "mode": "duplex",
                    "model": "PersonaPlex 7B",
                    "url": result.get("url", "https://localhost:8998"),
                    "voice": result.get("voice", "NATM1"),
                    "message": "Duplex mode active. Open browser for voice chat.",
                    "instructions": result.get("instructions", "")
                }
            else:
                return result

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _unload_ollama_models(self):
        """Tell Ollama to unload models to free VRAM."""
        try:
            # Send request to unload models with keep_alive=0
            models_to_unload = list(Config.get_all_models().values())
            for model in models_to_unload:
                try:
                    requests.post(
                        f"{self._ollama_host}/api/generate",
                        json={"model": model, "keep_alive": 0},
                        timeout=5
                    )
                except (requests.RequestException, ConnectionError, TimeoutError):
                    pass  # Model might not be loaded or Ollama not running
            logger.debug("Ollama models unloaded")
        except Exception as e:
            logger.warning(f"Warning: Could not unload Ollama models: {e}")

    def speak(self, text: str, **kwargs) -> dict:
        """Speak text using current mode.

        Args:
            text: Text to speak
            **kwargs: Additional parameters (speaker, output_path, etc.)

        Returns:
            dict with speak result
        """
        try:
            if self.current_mode == "pipeline":
                return self.sesame.speak(text, **kwargs)
            elif self.current_mode == "duplex":
                # PersonaPlex handles its own audio via web interface
                return {
                    "success": False,
                    "error": "In duplex mode, use the browser interface at https://localhost:8998"
                }
            else:
                # Default to pipeline mode
                switch_result = self.switch_to_pipeline()
                if switch_result["success"]:
                    return self.sesame.speak(text, **kwargs)
                return switch_result

        except Exception as e:
            return {"success": False, "error": str(e)}

    def route_voice(self, user_input: str) -> Optional[str]:
        """Detect voice mode from user input.

        Args:
            user_input: User's text input

        Returns:
            "duplex", "pipeline", or None if no voice action needed
        """
        lower = user_input.lower()

        # Check for duplex keywords first (more specific)
        if any(kw in lower for kw in self.DUPLEX_KEYWORDS):
            return "duplex"

        # Check for pipeline keywords
        if any(kw in lower for kw in self.PIPELINE_KEYWORDS):
            return "pipeline"

        return None

    def auto_switch(self, user_input: str) -> Optional[dict]:
        """Automatically switch mode based on user input.

        Args:
            user_input: User's text input

        Returns:
            Switch result dict or None if no switch needed
        """
        mode = self.route_voice(user_input)

        if mode == "duplex" and self.current_mode != "duplex":
            return self.switch_to_duplex()
        elif mode == "pipeline" and self.current_mode != "pipeline":
            return self.switch_to_pipeline()

        return None

    def stop(self) -> dict:
        """Stop all voice systems and free VRAM.

        Returns:
            dict with stop status
        """
        try:
            results = []

            # Stop PersonaPlex
            if self.current_mode == "duplex" and self.personaplex is not None:
                result = self.personaplex.stop_server()
                results.append(f"PersonaPlex: {result.get('message', 'stopped')}")

            # Unload Sesame
            if self.sesame is not None and self.sesame.is_loaded():
                result = self.sesame.unload()
                results.append(f"Sesame: {result.get('message', 'unloaded')}")

            # Clear CUDA cache
            _t, _avail = _get_torch()
            if _avail and _t.cuda.is_available():
                _t.cuda.empty_cache()

            self.current_mode = None

            return {
                "success": True,
                "message": "All voice systems stopped",
                "details": results
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def status(self) -> dict:
        """Get current voice system status.

        Returns:
            dict with status information
        """
        try:
            vram_free = "N/A"
            vram_total = "N/A"

            _t, _avail = _get_torch()
            if _avail and _t.cuda.is_available():
                free, total = _t.cuda.mem_get_info()
                vram_free = f"{free / 1024**3:.1f}GB"
                vram_total = f"{total / 1024**3:.1f}GB"

            return {
                "success": True,
                "mode": self.current_mode,
                "sesame_loaded": self.sesame.is_loaded() if self.sesame is not None else False,
                "personaplex_status": self.personaplex.status() if self.personaplex is not None else "unavailable",
                "vram_free": vram_free,
                "vram_total": vram_total
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def execute(self, action: str, **kwargs) -> dict:
        """Parse and dispatch actions.

        Args:
            action: Natural language action string
            **kwargs: Additional parameters

        Returns:
            dict with action result
        """
        action_lower = action.lower()

        # Status check
        if any(word in action_lower for word in ["status", "info", "check"]):
            return self.status()

        # Switch to pipeline
        if "pipeline" in action_lower or any(kw in action_lower for kw in ["sesame", "tts mode"]):
            return self.switch_to_pipeline()

        # Switch to duplex
        if "duplex" in action_lower or "personaplex" in action_lower:
            voice = kwargs.get("voice")
            persona = kwargs.get("persona")
            return self.switch_to_duplex(voice=voice, persona=persona)

        # Stop all
        if any(word in action_lower for word in ["stop", "shutdown", "kill all"]):
            return self.stop()

        # Speak
        if any(word in action_lower for word in ["speak", "say"]):
            text = kwargs.get("text", "")
            return self.speak(text, **kwargs)

        # Unknown action
        return {
            "success": False,
            "error": f"Unknown voice manager action: {action}",
            "available_actions": [
                "status - Check voice system status",
                "switch to pipeline - Use Sesame CSM (best quality)",
                "switch to duplex - Use PersonaPlex (conversation mode)",
                "stop - Stop all voice systems",
                "speak <text> - Speak using current mode"
            ]
        }


_voice_manager = None

def get_voice_manager():
    global _voice_manager
    if _voice_manager is None:
        _voice_manager = VoiceManager()
    return _voice_manager
