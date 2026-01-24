"""PersonaPlex Tool - Real-time full-duplex speech-to-speech with NVIDIA PersonaPlex.

Tool #17: PersonaPlex integration for Aura agent.
Replaces Whisper+pyttsx3 voice system for real-time conversations.

Repository: https://github.com/NVIDIA/personaplex
"""

import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional


class PersonaPlexTool:
    """Real-time full-duplex speech-to-speech using NVIDIA PersonaPlex."""

    # Available voices
    VOICES = {
        # Natural voices
        "NATF0": "Natural Female 0",
        "NATF1": "Natural Female 1",
        "NATF2": "Natural Female 2",
        "NATF3": "Natural Female 3",
        "NATM0": "Natural Male 0",
        "NATM1": "Natural Male 1 (Default)",
        "NATM2": "Natural Male 2",
        "NATM3": "Natural Male 3",
        # Variety voices
        "VARF0": "Variety Female 0",
        "VARF1": "Variety Female 1",
        "VARF2": "Variety Female 2",
        "VARF3": "Variety Female 3",
        "VARF4": "Variety Female 4",
        "VARM0": "Variety Male 0",
        "VARM1": "Variety Male 1",
        "VARM2": "Variety Male 2",
        "VARM3": "Variety Male 3",
        "VARM4": "Variety Male 4",
    }

    DEFAULT_VOICE = "NATM1"
    DEFAULT_PERSONA = (
        "You are Aura, an intelligent personal AI assistant. "
        "You are wise, helpful, and occasionally witty with subtle sarcasm."
    )

    def __init__(self):
        self.name = "personaplex"
        self.description = "Real-time full-duplex speech-to-speech with NVIDIA PersonaPlex"
        self._server_process: Optional[subprocess.Popen] = None
        self._current_voice: str = self.DEFAULT_VOICE
        self._current_persona: str = self.DEFAULT_PERSONA
        self._ssl_dir: Optional[str] = None

    def _check_hf_token(self) -> dict:
        """Verify HuggingFace token is set.

        Returns:
            dict with success status and error if missing
        """
        if not os.getenv("HF_TOKEN"):
            return {
                "success": False,
                "error": (
                    "HF_TOKEN environment variable not set. "
                    "Please set your HuggingFace token: export HF_TOKEN=<your_token>\n"
                    "You must also accept the PersonaPlex model license on HuggingFace."
                )
            }
        return {"success": True}

    def status(self) -> dict:
        """Check if PersonaPlex server is running.

        Returns:
            dict with server status information
        """
        try:
            if self._server_process is None:
                return {
                    "success": True,
                    "running": False,
                    "message": "PersonaPlex server is not running",
                    "voice": self._current_voice,
                    "persona": self._current_persona[:50] + "..." if len(self._current_persona) > 50 else self._current_persona
                }

            # Check if process is still alive
            poll_result = self._server_process.poll()
            if poll_result is not None:
                # Process has terminated
                self._server_process = None
                return {
                    "success": True,
                    "running": False,
                    "message": f"PersonaPlex server stopped (exit code: {poll_result})",
                    "voice": self._current_voice,
                    "persona": self._current_persona[:50] + "..."
                }

            return {
                "success": True,
                "running": True,
                "message": "PersonaPlex server is running",
                "url": "https://localhost:8998",
                "voice": self._current_voice,
                "voice_name": self.VOICES.get(self._current_voice, "Unknown"),
                "persona": self._current_persona[:50] + "..." if len(self._current_persona) > 50 else self._current_persona
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def start_server(self, voice: str = None, persona: str = None, cpu_offload: bool = False) -> dict:
        """Launch the PersonaPlex server.

        Args:
            voice: Voice ID (NATF0-3, NATM0-3, VARF0-4, VARM0-4). Default: NATM1
            persona: System prompt for the AI persona. Default: Aura persona
            cpu_offload: Enable CPU offload for low GPU memory (requires accelerate)

        Returns:
            dict with startup status and connection info
        """
        try:
            # Check HF token first
            token_check = self._check_hf_token()
            if not token_check["success"]:
                return token_check

            # Check if already running
            if self._server_process is not None:
                poll = self._server_process.poll()
                if poll is None:
                    return {
                        "success": False,
                        "error": "PersonaPlex server is already running. Stop it first with stop_server()."
                    }
                self._server_process = None

            # Set voice and persona
            if voice:
                voice_upper = voice.upper()
                if voice_upper not in self.VOICES:
                    return {
                        "success": False,
                        "error": f"Invalid voice '{voice}'. Valid options: {', '.join(self.VOICES.keys())}"
                    }
                self._current_voice = voice_upper
            else:
                self._current_voice = self.DEFAULT_VOICE

            if persona:
                self._current_persona = persona
            else:
                self._current_persona = self.DEFAULT_PERSONA

            # Create temporary SSL directory
            self._ssl_dir = tempfile.mkdtemp(prefix="personaplex_ssl_")

            # Build command
            cmd = [
                "python", "-m", "moshi.server",
                "--ssl", self._ssl_dir,
                "--voice", self._current_voice,
                "--system-prompt", self._current_persona
            ]

            if cpu_offload:
                cmd.append("--cpu-offload")

            # Start server process
            self._server_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env={**os.environ, "HF_TOKEN": os.getenv("HF_TOKEN", "")}
            )

            # Give it a moment to start
            time.sleep(2)

            # Check if it started successfully
            if self._server_process.poll() is not None:
                stderr = self._server_process.stderr.read() if self._server_process.stderr else ""
                self._server_process = None
                return {
                    "success": False,
                    "error": f"Server failed to start: {stderr}"
                }

            return {
                "success": True,
                "message": "PersonaPlex server started successfully",
                "url": "https://localhost:8998",
                "voice": self._current_voice,
                "voice_name": self.VOICES[self._current_voice],
                "persona": self._current_persona[:100] + "..." if len(self._current_persona) > 100 else self._current_persona,
                "instructions": (
                    "Open https://localhost:8998 in your browser to access the voice interface. "
                    "Accept the self-signed certificate warning to connect."
                )
            }
        except FileNotFoundError:
            return {
                "success": False,
                "error": (
                    "moshi package not found. Install it with: pip install moshi/. "
                    "from the PersonaPlex repository."
                )
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def stop_server(self) -> dict:
        """Shutdown the PersonaPlex server.

        Returns:
            dict with shutdown status
        """
        try:
            if self._server_process is None:
                return {
                    "success": True,
                    "message": "PersonaPlex server was not running"
                }

            # Check if still running
            if self._server_process.poll() is None:
                # Terminate gracefully
                self._server_process.terminate()
                try:
                    self._server_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    # Force kill if needed
                    self._server_process.kill()
                    self._server_process.wait()

            self._server_process = None

            # Cleanup SSL directory
            if self._ssl_dir and os.path.exists(self._ssl_dir):
                import shutil
                try:
                    shutil.rmtree(self._ssl_dir)
                except Exception:
                    pass
                self._ssl_dir = None

            return {
                "success": True,
                "message": "PersonaPlex server stopped successfully"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def set_voice(self, voice_id: str) -> dict:
        """Set the voice for PersonaPlex.

        Note: If server is running, it must be restarted for changes to take effect.

        Args:
            voice_id: Voice ID (NATF0-3, NATM0-3, VARF0-4, VARM0-4)

        Returns:
            dict with status and restart instructions if needed
        """
        try:
            voice_upper = voice_id.upper()
            if voice_upper not in self.VOICES:
                return {
                    "success": False,
                    "error": f"Invalid voice '{voice_id}'. Valid options: {', '.join(self.VOICES.keys())}",
                    "available_voices": self.VOICES
                }

            old_voice = self._current_voice
            self._current_voice = voice_upper

            result = {
                "success": True,
                "message": f"Voice changed from {old_voice} to {voice_upper}",
                "voice": voice_upper,
                "voice_name": self.VOICES[voice_upper]
            }

            # Check if server is running
            if self._server_process is not None and self._server_process.poll() is None:
                result["note"] = "Server is running. Restart it for the voice change to take effect."
                result["restart_command"] = "personaplex stop_server then start_server"

            return result
        except Exception as e:
            return {"success": False, "error": str(e)}

    def set_persona(self, prompt: str) -> dict:
        """Set the persona/role text for the AI.

        Note: If server is running, it must be restarted for changes to take effect.

        Args:
            prompt: System prompt defining the AI's personality and role

        Returns:
            dict with status and restart instructions if needed
        """
        try:
            if not prompt or not prompt.strip():
                return {
                    "success": False,
                    "error": "Persona prompt cannot be empty"
                }

            old_persona = self._current_persona
            self._current_persona = prompt.strip()

            result = {
                "success": True,
                "message": "Persona updated successfully",
                "persona": self._current_persona[:100] + "..." if len(self._current_persona) > 100 else self._current_persona
            }

            # Check if server is running
            if self._server_process is not None and self._server_process.poll() is None:
                result["note"] = "Server is running. Restart it for the persona change to take effect."
                result["restart_command"] = "personaplex stop_server then start_server"

            return result
        except Exception as e:
            return {"success": False, "error": str(e)}

    def list_voices(self) -> dict:
        """Show all available voices.

        Returns:
            dict with categorized voice listings
        """
        try:
            natural_female = {k: v for k, v in self.VOICES.items() if k.startswith("NATF")}
            natural_male = {k: v for k, v in self.VOICES.items() if k.startswith("NATM")}
            variety_female = {k: v for k, v in self.VOICES.items() if k.startswith("VARF")}
            variety_male = {k: v for k, v in self.VOICES.items() if k.startswith("VARM")}

            return {
                "success": True,
                "current_voice": self._current_voice,
                "current_voice_name": self.VOICES[self._current_voice],
                "categories": {
                    "Natural Female (NATF0-3)": natural_female,
                    "Natural Male (NATM0-3)": natural_male,
                    "Variety Female (VARF0-4)": variety_female,
                    "Variety Male (VARM0-4)": variety_male
                },
                "all_voices": self.VOICES,
                "default_voice": self.DEFAULT_VOICE
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def reset_to_defaults(self) -> dict:
        """Reset voice and persona to Aura defaults.

        Returns:
            dict with reset confirmation
        """
        self._current_voice = self.DEFAULT_VOICE
        self._current_persona = self.DEFAULT_PERSONA

        result = {
            "success": True,
            "message": "Reset to Aura defaults",
            "voice": self._current_voice,
            "voice_name": self.VOICES[self._current_voice],
            "persona": self._current_persona
        }

        if self._server_process is not None and self._server_process.poll() is None:
            result["note"] = "Server is running. Restart it for changes to take effect."

        return result

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
        if any(word in action_lower for word in ["status", "running", "check", "is running"]):
            return self.status()

        # Start server
        if any(word in action_lower for word in ["start", "launch", "begin", "run server"]):
            voice = kwargs.get("voice")
            persona = kwargs.get("persona")
            cpu_offload = kwargs.get("cpu_offload", False)

            # Try to extract voice from action
            if not voice:
                for v in self.VOICES.keys():
                    if v.lower() in action_lower:
                        voice = v
                        break

            return self.start_server(voice=voice, persona=persona, cpu_offload=cpu_offload)

        # Stop server
        if any(word in action_lower for word in ["stop", "shutdown", "kill", "end", "terminate"]):
            return self.stop_server()

        # Set voice
        if "voice" in action_lower and any(word in action_lower for word in ["set", "change", "use", "switch"]):
            voice_id = kwargs.get("voice_id") or kwargs.get("voice")
            if not voice_id:
                # Try to extract from action
                for v in self.VOICES.keys():
                    if v.lower() in action_lower:
                        voice_id = v
                        break
            if voice_id:
                return self.set_voice(voice_id)
            return {"success": False, "error": "No voice ID provided. Use list_voices to see options."}

        # Set persona
        if "persona" in action_lower and any(word in action_lower for word in ["set", "change", "use"]):
            prompt = kwargs.get("prompt") or kwargs.get("persona")
            if prompt:
                return self.set_persona(prompt)
            return {"success": False, "error": "No persona prompt provided."}

        # List voices
        if any(phrase in action_lower for phrase in ["list voice", "show voice", "available voice", "voices"]):
            return self.list_voices()

        # Reset to defaults
        if "reset" in action_lower or "default" in action_lower:
            return self.reset_to_defaults()

        # Unknown action - show help
        return {
            "success": False,
            "error": f"Unknown PersonaPlex action: {action}",
            "available_actions": [
                "status - Check if server is running",
                "start_server - Launch the PersonaPlex server",
                "stop_server - Shutdown the server",
                "set_voice <voice_id> - Change voice (NATF0-3, NATM0-3, VARF0-4, VARM0-4)",
                "set_persona <prompt> - Set role/personality text",
                "list_voices - Show available voices",
                "reset_to_defaults - Reset to Aura defaults"
            ]
        }
