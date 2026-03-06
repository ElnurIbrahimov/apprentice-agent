"""Sesame CSM 1B - High-quality conversational TTS.

Tool #18: Sesame CSM integration for Aura agent.
Pipeline mode TTS with superior voice quality.

VRAM: ~4.5GB on CUDA
Repository: https://github.com/SesameAILabs/csm

Requires HuggingFace access to:
- sesame/csm-1b
- meta-llama/Llama-3.2-1B
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional, List, Tuple

logger = logging.getLogger(__name__)

# Disable triton on Windows (causes import errors)
os.environ.setdefault('TORCHAO_NO_TRITON', '1')

# torch is lazy-loaded on first use to avoid ~2s startup penalty
_torch = None

def _get_torch():
    global _torch
    if _torch is None:
        import torch
        torch._dynamo.config.suppress_errors = True
        _torch = torch
    return _torch

# Add CSM repo to path
CSM_PATH = Path.home() / "sesame-csm"
if CSM_PATH.exists() and str(CSM_PATH) not in sys.path:
    sys.path.insert(0, str(CSM_PATH))


class SesameTTS:
    """High-quality conversational TTS using Sesame CSM 1B."""

    def __init__(self):
        self.name = "sesame_tts"
        self.description = "High-quality text-to-speech using Sesame CSM 1B"
        self.generator = None
        self.device = "cuda" if _get_torch().cuda.is_available() else "cpu"
        self.sample_rate = 24000
        self._loaded = False

    def _check_hf_token(self) -> dict:
        """Verify HuggingFace token is set."""
        if not os.getenv("HF_TOKEN"):
            return {
                "success": False,
                "error": (
                    "HF_TOKEN environment variable not set. "
                    "Please set your HuggingFace token: export HF_TOKEN=<your_token>\n"
                    "You must also accept the Sesame CSM and Llama-3.2-1B model licenses."
                )
            }
        return {"success": True}

    def load(self) -> dict:
        """Load Sesame CSM model to GPU.

        Returns:
            dict with success status
        """
        try:
            # Check token first
            token_check = self._check_hf_token()
            if not token_check["success"]:
                return token_check

            import torch
            if torch.cuda.is_available():
                free_vram = torch.cuda.mem_get_info()[0] / 1e9
                if free_vram < 5.0:
                    logger.warning(f"[SesameTTS] Only {free_vram:.1f}GB VRAM free, CSM-1B needs ~4.5GB. Skipping GPU load.")
                    return {"success": False, "error": "Insufficient VRAM"}

            if self._loaded and self.generator is not None:
                return {
                    "success": True,
                    "message": "Sesame CSM already loaded",
                    "device": self.device
                }

            logger.debug("Loading Sesame CSM 1B...")

            # Import generator from CSM package
            try:
                from generator import load_csm_1b
            except ImportError:
                # Try adding CSM path if not already added
                csm_path = Path.home() / "sesame-csm"
                if csm_path.exists() and str(csm_path) not in sys.path:
                    sys.path.insert(0, str(csm_path))
                try:
                    from generator import load_csm_1b
                except ImportError:
                    return {
                        "success": False,
                        "error": (
                            "CSM package not found. Clone the repo:\n"
                            "git clone https://github.com/SesameAILabs/csm.git ~/sesame-csm"
                        )
                    }

            self.generator = load_csm_1b(device=self.device)
            self._loaded = True

            # Get VRAM usage
            vram_used = "N/A"
            if _get_torch().cuda.is_available():
                vram_used = f"{_get_torch().cuda.memory_allocated() / 1024**3:.2f}GB"

            logger.debug(f"Sesame CSM loaded on {self.device}")
            return {
                "success": True,
                "message": "Sesame CSM 1B loaded successfully",
                "device": self.device,
                "vram_used": vram_used
            }

        except Exception as e:
            self._loaded = False
            self.generator = None
            return {"success": False, "error": str(e)}

    def unload(self) -> dict:
        """Free VRAM by unloading the model.

        Returns:
            dict with success status
        """
        try:
            if self.generator is not None:
                del self.generator
                self.generator = None

            self._loaded = False

            # Clear CUDA cache
            if _get_torch().cuda.is_available():
                _get_torch().cuda.empty_cache()
                _get_torch().cuda.synchronize()

            return {
                "success": True,
                "message": "Sesame CSM unloaded, VRAM freed"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded and self.generator is not None

    def generate(
        self,
        text: str,
        speaker: int = 0,
        context: Optional[List[Tuple[str, Optional["torch.Tensor"]]]] = None,
        max_audio_length_ms: int = 30000
    ) -> Optional["torch.Tensor"]:
        """Generate speech from text.

        Args:
            text: Text to synthesize
            speaker: Speaker ID (0-based)
            context: Optional conversation context [(text, audio), ...]
            max_audio_length_ms: Maximum audio length in milliseconds

        Returns:
            Audio tensor or None on error
        """
        try:
            if not self._loaded:
                load_result = self.load()
                if not load_result["success"]:
                    logger.debug(f"Failed to load model: {load_result.get('error')}")
                    return None

            audio = self.generator.generate(
                text=text,
                speaker=speaker,
                context=context or [],
                max_audio_length_ms=max_audio_length_ms
            )
            return audio

        except Exception as e:
            logger.error(f"Generation error: {e}")
            return None

    def speak(self, text: str, output_path: Optional[str] = None, speaker: int = 0) -> dict:
        """Generate and play/save audio.

        Args:
            text: Text to speak
            output_path: Optional path to save WAV file
            speaker: Speaker ID

        Returns:
            dict with success status
        """
        try:
            if not text or not text.strip():
                return {"success": False, "error": "No text provided"}

            audio = self.generate(text, speaker=speaker)

            if audio is None:
                return {"success": False, "error": "Audio generation failed"}

            # Save to file if path provided
            if output_path:
                import torchaudio
                torchaudio.save(
                    output_path,
                    audio.unsqueeze(0).cpu(),
                    self.sample_rate
                )

            # Play audio
            self._play(audio)

            return {
                "success": True,
                "text": text,
                "duration_ms": int(len(audio) / self.sample_rate * 1000),
                "saved_to": output_path
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _play(self, audio: "torch.Tensor"):
        """Play audio through speakers.

        Args:
            audio: Audio tensor to play
        """
        try:
            import numpy as np

            # Convert to numpy and ensure float32
            audio_np = audio.cpu().numpy().astype(np.float32)

            # Ensure 1D
            if audio_np.ndim > 1:
                audio_np = audio_np.flatten()

            # Normalize if needed
            max_val = np.abs(audio_np).max()
            if max_val > 1.0:
                audio_np = audio_np / max_val

            logger.debug(f"Playing audio: shape={audio_np.shape}, dtype={audio_np.dtype}, max={np.abs(audio_np).max():.3f}")

            # Try sounddevice first
            try:
                import sounddevice as sd
                sd.play(audio_np, self.sample_rate)
                sd.wait()
                logger.debug("Playback complete (sounddevice)")
                return
            except Exception as e:
                logger.debug(f"sounddevice failed: {e}, trying alternative...")

            # Fallback: save to temp file and play with Windows
            try:
                import tempfile
                import scipy.io.wavfile as wav

                # Save to temp WAV file
                temp_path = tempfile.mktemp(suffix='.wav')
                wav.write(temp_path, self.sample_rate, (audio_np * 32767).astype(np.int16))

                # Play with Windows
                import winsound
                winsound.PlaySound(temp_path, winsound.SND_FILENAME)
                logger.debug("Playback complete (winsound)")

                # Cleanup
                import os
                os.remove(temp_path)
            except Exception as e2:
                logger.debug(f"winsound fallback failed: {e2}")

        except Exception as e:
            logger.error(f"Playback error: {e}")

    def status(self) -> dict:
        """Get current status of Sesame TTS.

        Returns:
            dict with status information
        """
        vram_free = "N/A"
        vram_used = "N/A"

        if _get_torch().cuda.is_available():
            free, total = _get_torch().cuda.mem_get_info()
            vram_free = f"{free / 1024**3:.2f}GB"
            vram_used = f"{(total - free) / 1024**3:.2f}GB"

        return {
            "success": True,
            "loaded": self._loaded,
            "device": self.device,
            "sample_rate": self.sample_rate,
            "vram_used": vram_used,
            "vram_free": vram_free
        }

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

        # Load model
        if any(word in action_lower for word in ["load", "start", "init"]):
            return self.load()

        # Unload model
        if any(word in action_lower for word in ["unload", "stop", "free", "release"]):
            return self.unload()

        # Speak/generate
        if any(word in action_lower for word in ["speak", "say", "generate", "synthesize"]):
            text = kwargs.get("text", "")
            output_path = kwargs.get("output_path")
            speaker = kwargs.get("speaker", 0)
            return self.speak(text, output_path=output_path, speaker=speaker)

        # Unknown action
        return {
            "success": False,
            "error": f"Unknown action: {action}",
            "available_actions": [
                "status - Check if model is loaded",
                "load - Load Sesame CSM to GPU",
                "unload - Free VRAM",
                "speak <text> - Generate and play speech"
            ]
        }
