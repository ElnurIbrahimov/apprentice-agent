"""
Aura - Apprentice Agent GUI
Simple, readable chat interface with Sesame CSM voice.
"""

import os
from pathlib import Path

# Load .env file for HF_TOKEN
env_file = Path(__file__).parent / ".env"
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ.setdefault(key.strip(), value.strip())

os.environ.setdefault('TORCHAO_NO_TRITON', '1')

import gradio as gr
import threading
from typing import Generator
from pathlib import Path

from apprentice_agent import ApprenticeAgent
from apprentice_agent.metacognition import MetacognitionLogger


# ============================================================================
# TTS ENGINE - SESAME CSM 1B (Human-quality voice)
# ============================================================================

class TTSEngine:
    """Text-to-speech engine using Sesame CSM 1B for human-quality voice."""
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.sesame = None
        self.pyttsx3_engine = None
        self.available = False
        self.using_sesame = False
        self._init_tts()

    def _init_tts(self):
        """Initialize TTS - try Sesame first, always init pyttsx3 as backup."""
        # Try Sesame CSM first (human-quality)
        try:
            from apprentice_agent.tools.sesame_tts import SesameTTS
            self.sesame = SesameTTS()
            self.using_sesame = True
            print("TTS: Sesame CSM 1B available (human-quality voice)")
        except Exception as e:
            print(f"Sesame not available: {e}")
            self.using_sesame = False

        # Always init pyttsx3 as backup (works immediately, no loading needed)
        try:
            import pyttsx3
            self.pyttsx3_engine = pyttsx3.init()
            self.pyttsx3_engine.setProperty('rate', 175)
            self.pyttsx3_engine.setProperty('volume', 0.9)
            self.available = True
            print("TTS: pyttsx3 ready (backup)")
        except Exception as e2:
            print(f"pyttsx3 not available: {e2}")
            # Only fail if neither Sesame nor pyttsx3 work
            if not self.using_sesame:
                self.available = False
            else:
                self.available = True

    def load_sesame(self) -> str:
        """Load Sesame model to GPU (takes ~30s first time)."""
        if self.sesame is None:
            return "Sesame not initialized"
        try:
            result = self.sesame.load()
            if result.get("success"):
                self.using_sesame = True
                return f"Sesame loaded! VRAM: {result.get('vram_used', 'N/A')}"
            return f"Error: {result.get('error', 'Failed to load')}"
        except Exception as e:
            return f"Error: {e}"

    def speak(self, text: str, use_sesame: bool = False):
        """Speak text. Uses pyttsx3 by default (fast), Sesame on request (quality).

        Args:
            text: Text to speak
            use_sesame: If True, use Sesame for high-quality voice (slower)
        """
        if not self.available or not text:
            return

        def _do_speak():
            try:
                # Use Sesame only if explicitly requested AND loaded
                if use_sesame and self.using_sesame and self.sesame and self.sesame.is_loaded():
                    print(f"Speaking with Sesame (high quality): {text[:50]}...")
                    self._speak_sesame(text)
                else:
                    # Default: fast pyttsx3
                    self._speak_pyttsx3(text)
            except Exception as e:
                print(f"TTS error: {e}")

        thread = threading.Thread(target=_do_speak, daemon=True)
        thread.start()

    def _speak_sesame(self, text: str):
        """Speak using Sesame (human-quality voice)."""
        try:
            import tempfile
            import numpy as np
            import scipy.io.wavfile as wav
            import winsound

            audio = self.sesame.generate(text)
            if audio is None:
                print("Sesame generate returned None, falling back to pyttsx3")
                self._speak_pyttsx3(text)
                return

            audio_np = audio.cpu().numpy().astype(np.float32)
            if audio_np.ndim > 1:
                audio_np = audio_np.flatten()

            # Normalize
            max_val = np.abs(audio_np).max()
            if max_val > 0:
                audio_np = audio_np / max_val

            # Add 0.5s silence at end to prevent cutoff
            silence = np.zeros(int(self.sesame.sample_rate * 0.5), dtype=np.float32)
            audio_np = np.concatenate([audio_np, silence])

            # Convert to int16
            audio_int16 = (audio_np * 32767).astype(np.int16)

            temp_path = tempfile.mktemp(suffix='.wav')
            wav.write(temp_path, self.sesame.sample_rate, audio_int16)
            winsound.PlaySound(temp_path, winsound.SND_FILENAME)

            import os
            os.remove(temp_path)
            print("Sesame playback complete")
        except Exception as e:
            print(f"Sesame speak error: {e}, falling back to pyttsx3")
            self._speak_pyttsx3(text)

    def _speak_pyttsx3(self, text: str):
        """Speak using pyttsx3 (robotic fallback)."""
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty('rate', 175)
            engine.say(text)
            engine.runAndWait()
            engine.stop()
        except Exception as e:
            print(f"pyttsx3 error: {e}")

    def get_status(self) -> str:
        """Get TTS status string."""
        if self.using_sesame and self.sesame:
            if self.sesame.is_loaded():
                return "Sesame CSM (loaded)"
            return "Sesame CSM (not loaded)"
        elif self.pyttsx3_engine:
            return "pyttsx3 (fallback)"
        return "Unavailable"


# Global TTS instance
tts = TTSEngine()


# ============================================================================
# CSS WITH EXACT COLORS
# ============================================================================

CUSTOM_CSS = """
/* Page background */
.gradio-container {
    background: #0f172a !important;
}

/* Sidebar and chat area */
.block {
    background: #1e293b !important;
    border: none !important;
}

/* All text - almost white */
body, .gradio-container, .gradio-container *, p, span, div, label {
    color: #f1f5f9 !important;
}

/* Headers - pure white */
h1, h2, h3, h4, h5, h6 {
    color: #ffffff !important;
}

/* Labels */
label span {
    color: #94a3b8 !important;
}

/* Input box */
textarea, input[type="text"], input[type="number"] {
    background: #1e293b !important;
    border: 1px solid #475569 !important;
    color: #f1f5f9 !important;
}

textarea::placeholder, input::placeholder {
    color: #64748b !important;
}

/* Send button - blue */
button.primary, button[variant="primary"] {
    background: #3b82f6 !important;
    color: #ffffff !important;
    border: none !important;
}

/* Other buttons */
button.secondary, button:not(.primary) {
    background: #334155 !important;
    color: #f1f5f9 !important;
    border: none !important;
}

/* Chatbot container */
.chatbot, .chatbot-container {
    background: #1e293b !important;
}

/* User message bubble - blue */
.message.user, .user-message, [data-testid="user"] {
    background: #3b82f6 !important;
    color: #ffffff !important;
}

/* Aura message bubble - dark gray */
.message.bot, .bot-message, [data-testid="bot"] {
    background: #334155 !important;
    color: #f1f5f9 !important;
}

/* Message text must be visible */
.message p, .message span, .message div {
    color: inherit !important;
}

/* Markdown inside messages */
.message .prose, .message .markdown {
    color: inherit !important;
}

.message .prose *, .message .markdown * {
    color: inherit !important;
}

/* Slider */
input[type="range"] {
    accent-color: #3b82f6 !important;
}

/* Accordion */
.accordion {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
}

/* Status dot */
.status-online {
    display: inline-block;
    width: 10px;
    height: 10px;
    background: #22c55e;
    border-radius: 50%;
    margin-right: 8px;
}

.status-offline {
    display: inline-block;
    width: 10px;
    height: 10px;
    background: #ef4444;
    border-radius: 50%;
    margin-right: 8px;
}

/* Inner Monologue Panel */
#thought-stream {
    font-family: 'Consolas', 'Fira Code', monospace !important;
    font-size: 12px !important;
    max-height: 200px;
    overflow-y: auto;
    background: #1a1a2e !important;
    padding: 10px;
    border-radius: 8px;
    border-left: 3px solid #00d9ff;
}

#thought-stream .thought-line {
    padding: 4px 0;
    border-bottom: 1px solid #2a2a4a;
}

#thought-stream .thought-perceive { color: #60a5fa; }
#thought-stream .thought-recall { color: #a78bfa; }
#thought-stream .thought-reason { color: #f472b6; }
#thought-stream .thought-decide { color: #fbbf24; }
#thought-stream .thought-execute { color: #34d399; }
#thought-stream .thought-reflect { color: #38bdf8; }
#thought-stream .thought-uncertain { color: #fb7185; }
#thought-stream .thought-eureka { color: #facc15; }

.monologue-header {
    color: #00d9ff !important;
    font-weight: bold;
    margin-bottom: 8px;
}
"""


# ============================================================================
# GUI CLASS
# ============================================================================

class AuraGUI:
    def __init__(self):
        self.max_iterations = 10
        self.voice_enabled = False
        # Fast-initialize agent (skip heavy tools like KG, FluxMind)
        print("Initializing Aura agent (fast mode)...")
        self.agent = ApprenticeAgent(fast_init=True)
        print("Agent ready!")

    def _get_agent(self):
        """Get agent instance (already initialized)."""
        return self.agent

    def _check_fluxmind(self) -> dict:
        """Check FluxMind status - load lazily if needed."""
        try:
            agent = self._get_agent()

            # Check if already loaded in agent
            if "fluxmind" in agent.tools:
                tool = agent.tools["fluxmind"]
                if tool.is_available():
                    return {"available": True, "version": "0.75.1"}

            # Try to load FluxMind lazily
            from apprentice_agent.tools import FluxMindTool, FLUXMIND_AVAILABLE

            if FLUXMIND_AVAILABLE and "fluxmind" not in agent.tools:
                # Try multiple possible paths
                possible_paths = [
                    Path(__file__).parent / "models" / "fluxmind_v0751.pt",
                    Path("models/fluxmind_v0751.pt"),
                    Path(__file__).parent.parent / "models" / "fluxmind_v0751.pt",
                ]
                for models_path in possible_paths:
                    if models_path.exists():
                        agent.tools["fluxmind"] = FluxMindTool(str(models_path))
                        if agent.tools["fluxmind"].is_available():
                            return {"available": True, "version": "0.75.1"}
                        break

            return {"available": False}
        except Exception as e:
            print(f"[FluxMind] Check error: {e}")
            return {"available": False}

    def get_status_html(self) -> str:
        status = self._check_fluxmind()
        if status["available"]:
            return f'''<div style="background: #334155; padding: 12px; border-radius: 8px; margin: 8px 0;">
<span class="status-online"></span>
<strong style="color: #22c55e;">FluxMind Online</strong>
<div style="color: #94a3b8; font-size: 13px; margin-top: 4px;">v{status["version"]}</div>
</div>'''
        else:
            return '''<div style="background: #334155; padding: 12px; border-radius: 8px; margin: 8px 0;">
<span class="status-offline"></span>
<strong style="color: #ef4444;">FluxMind Offline</strong>
</div>'''

    def get_voice_status_html(self) -> str:
        """Get voice status HTML."""
        tts_status = tts.get_status()
        if tts.available:
            # Check if Sesame is loaded
            sesame_loaded = tts.using_sesame and tts.sesame and tts.sesame.is_loaded()

            if sesame_loaded:
                status_text = "Sesame Ready"
                status_color = "#22c55e"
                dot_class = "status-online"
            elif tts.using_sesame:
                status_text = "Sesame (click Load)"
                status_color = "#f59e0b"  # Orange - needs loading
                dot_class = "status-offline"
            else:
                status_text = "pyttsx3 Fallback"
                status_color = "#94a3b8"
                dot_class = "status-online"

            voice_state = "ON" if self.voice_enabled else "OFF"
            icon = "🔊" if self.voice_enabled else "🔇"

            return f'''<div style="background: #334155; padding: 12px; border-radius: 8px; margin: 8px 0;">
<span class="{dot_class}"></span>
<strong style="color: {status_color};">{status_text}</strong>
<div style="color: #94a3b8; font-size: 12px; margin-top: 4px;">{icon} Voice {voice_state}</div>
</div>'''
        else:
            return '''<div style="background: #334155; padding: 12px; border-radius: 8px; margin: 8px 0;">
<span class="status-offline"></span>
<strong style="color: #ef4444;">Voice Unavailable</strong>
<div style="color: #94a3b8; font-size: 12px; margin-top: 4px;">No TTS engine available</div>
</div>'''

    def toggle_voice(self, enabled: bool) -> str:
        """Toggle voice output."""
        self.voice_enabled = enabled
        if enabled and tts.available:
            # Load Sesame if not loaded
            if tts.using_sesame and tts.sesame and not tts.sesame.is_loaded():
                print("Loading Sesame CSM... (first time takes ~30s)")
                tts.load_sesame()
            tts.speak("Voice enabled")
        return self.get_voice_status_html()

    def load_sesame(self) -> str:
        """Load Sesame model with timeout and VRAM cleanup."""
        import threading

        # Step 1: Free VRAM by unloading Ollama models
        print("Freeing VRAM before loading Sesame...")
        try:
            import requests
            requests.post("http://localhost:11434/api/generate",
                         json={"model": "qwen2:1.5b", "keep_alive": 0}, timeout=5)
            requests.post("http://localhost:11434/api/generate",
                         json={"model": "llama3:8b", "keep_alive": 0}, timeout=5)
        except Exception as e:
            print(f"Ollama cleanup: {e}")

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                free, total = torch.cuda.mem_get_info()
                print(f"VRAM free: {free/1024**3:.1f}GB / {total/1024**3:.1f}GB")
        except Exception as e:
            print(f"CUDA cleanup: {e}")

        # Step 2: Load Sesame with timeout
        result = {"done": False, "error": None}

        def load_thread():
            try:
                r = tts.load_sesame()
                print(f"Load Sesame result: {r}")
                if "Error" in str(r) or "error" in str(r).lower():
                    result["error"] = r
                result["done"] = True
            except Exception as e:
                import traceback
                traceback.print_exc()
                result["error"] = str(e)
                result["done"] = True

        thread = threading.Thread(target=load_thread, daemon=True)
        thread.start()
        thread.join(timeout=300)  # 5 minute timeout for slow loading

        if not result["done"]:
            return self.get_voice_status_html()  # Still loading, will update on next check
        elif result["error"]:
            print(f"Sesame load error: {result['error']}")

        return self.get_voice_status_html()

    # Keywords that trigger Sesame high-quality voice
    SESAME_KEYWORDS = [
        "with sesame", "use sesame", "sesame voice", "high quality voice",
        "quality voice", "nice voice", "human voice", "natural voice",
        "read nicely", "speak nicely", "say nicely", "read this nicely"
    ]

    def _wants_sesame(self, message: str) -> bool:
        """Check if user wants high-quality Sesame voice."""
        msg_lower = message.lower()
        return any(kw in msg_lower for kw in self.SESAME_KEYWORDS)

    def run_agent(self, message: str, history: list, voice_enabled: bool) -> Generator:
        """Run agent and optionally speak response."""
        if not message.strip():
            yield history
            return

        self.voice_enabled = voice_enabled
        use_sesame = self._wants_sesame(message)

        # Add user message
        history = history + [{"role": "user", "content": message}]
        yield history

        # Run agent
        try:
            agent = self._get_agent()
            agent.max_iterations = self.max_iterations

            # Use chat() for simple messages (faster), run() for complex tasks
            if agent._is_simple_query(message):
                response = agent.chat(message)
            else:
                result = agent.run(message)
                if result.get("fast_path"):
                    response = result.get("response", "Done.")
                else:
                    response = self._build_response(result)

            # Speak response if voice enabled
            if self.voice_enabled and tts.available:
                if use_sesame:
                    print(f"Speaking with Sesame (requested): {response[:50]}...")
                else:
                    print(f"Speaking with pyttsx3 (fast): {response[:50]}...")
                tts.speak(response, use_sesame=use_sesame)

            history = history + [{"role": "assistant", "content": response}]
            yield history

        except Exception as e:
            error_msg = f"Error: {e}"
            history = history + [{"role": "assistant", "content": error_msg}]
            yield history

    def _build_response(self, result: dict) -> str:
        outputs = []

        for item in result.get("history", []):
            item_result = item.get("result", {})
            if not item_result:
                continue

            tool = item.get("action", {}).get("tool", "")

            if tool == "fluxmind":
                conf = item_result.get("confidence", 0)
                if isinstance(conf, (int, float)):
                    pct = conf * 100 if conf <= 1 else conf
                    if pct >= 80:
                        outputs.append(f"FluxMind: {pct:.1f}% confidence (high)")
                    else:
                        outputs.append(f"FluxMind: {pct:.1f}% confidence (uncertain)")

            elif tool == "web_search" and item_result.get("success"):
                for r in item_result.get("results", [])[:3]:
                    title = r.get("title", "")
                    snippet = r.get("snippet", "")
                    if snippet:
                        outputs.append(f"{title}: {snippet}")

            elif tool == "code_executor" and item_result.get("success"):
                output = item_result.get("output", "")
                if output:
                    outputs.append(output[:500])

        if result.get("completed"):
            final_eval = result.get("final_evaluation", {})
            progress = final_eval.get("progress", "") if final_eval else ""
            if outputs:
                return "\n\n".join(outputs) + (f"\n\n{progress}" if progress else "")
            return progress or "Done."
        else:
            return f"Incomplete after {result.get('iterations', 0)} iterations."

    def get_stats(self) -> str:
        try:
            logger = MetacognitionLogger()
            stats = logger.get_stats()
            if "error" in stats:
                return stats['error']
            return f"Actions: {stats['total_actions']}, Success: {stats['success_rate']}%"
        except Exception as e:
            return str(e)

    def test_voice(self) -> str:
        """Test TTS - uses Sesame if loaded, otherwise pyttsx3."""
        try:
            sesame_loaded = tts.using_sesame and tts.sesame and tts.sesame.is_loaded()
            print(f"Test voice: sesame_loaded={sesame_loaded}")

            if sesame_loaded:
                print("Test voice: Using Sesame...")
                self._play_sesame("Hello! I am Aura with human quality voice.")
            else:
                print("Test voice: Using pyttsx3")
                import pyttsx3
                engine = pyttsx3.init()
                engine.setProperty('rate', 175)
                engine.say("Hello! I am Aura, your AI assistant.")
                engine.runAndWait()
                engine.stop()
            return self.get_voice_status_html()
        except Exception as e:
            import traceback
            print(f"Test voice error: {e}")
            traceback.print_exc()
            return self.get_voice_status_html()

    def _play_sesame(self, text: str, background: bool = False):
        """Play text using Sesame TTS."""
        def _generate_and_play():
            try:
                import tempfile
                import numpy as np
                import scipy.io.wavfile as wav
                import winsound

                audio = tts.sesame.generate(text)
                if audio is None:
                    print("Sesame generate returned None")
                    return

                audio_np = audio.cpu().numpy().astype(np.float32)
                if audio_np.ndim > 1:
                    audio_np = audio_np.flatten()

                # Normalize
                max_val = np.abs(audio_np).max()
                if max_val > 0:
                    audio_np = audio_np / max_val

                # Add 0.5s silence at end to prevent cutoff
                silence = np.zeros(int(tts.sesame.sample_rate * 0.5), dtype=np.float32)
                audio_np = np.concatenate([audio_np, silence])

                # Convert to int16
                audio_int16 = (audio_np * 32767).astype(np.int16)

                temp_path = tempfile.mktemp(suffix='.wav')
                wav.write(temp_path, tts.sesame.sample_rate, audio_int16)
                winsound.PlaySound(temp_path, winsound.SND_FILENAME)

                import os
                os.remove(temp_path)
            except Exception as e:
                print(f"Sesame playback error: {e}")

        if background:
            import threading
            thread = threading.Thread(target=_generate_and_play, daemon=True)
            thread.start()
        else:
            _generate_and_play()

    # =========================================================================
    # CLAWDBOT INTEGRATION
    # =========================================================================

    def get_clawdbot_status_html(self) -> str:
        """Get Clawdbot status HTML."""
        try:
            from apprentice_agent.tools.clawdbot import clawdbot
            status = clawdbot.get_status()

            if status.get("running"):
                return '''<div style="background: #334155; padding: 12px; border-radius: 8px; margin: 8px 0;">
<span class="status-online"></span>
<strong style="color: #22c55e;">Clawdbot Online</strong>
<div style="color: #94a3b8; font-size: 12px; margin-top: 4px;">Gateway running</div>
</div>'''
            else:
                error = status.get("error", "Not running")
                return f'''<div style="background: #334155; padding: 12px; border-radius: 8px; margin: 8px 0;">
<span class="status-offline"></span>
<strong style="color: #ef4444;">Clawdbot Offline</strong>
<div style="color: #94a3b8; font-size: 12px; margin-top: 4px;">{error}</div>
</div>'''
        except Exception as e:
            return f'''<div style="background: #334155; padding: 12px; border-radius: 8px; margin: 8px 0;">
<span class="status-offline"></span>
<strong style="color: #ef4444;">Clawdbot Error</strong>
<div style="color: #94a3b8; font-size: 12px; margin-top: 4px;">{str(e)[:50]}</div>
</div>'''

    def start_clawdbot_gateway(self) -> str:
        """Start Clawdbot gateway."""
        try:
            from apprentice_agent.tools.clawdbot import clawdbot
            result = clawdbot.start_gateway()
            if result.get("success"):
                return self.get_clawdbot_status_html()
            return self.get_clawdbot_status_html()
        except Exception as e:
            print(f"Clawdbot start error: {e}")
            return self.get_clawdbot_status_html()

    def send_clawdbot_message(self, to: str, message: str, channel: str) -> str:
        """Send a message via Clawdbot."""
        if not to or not message:
            return "Please enter recipient and message"

        try:
            from apprentice_agent.tools.clawdbot import clawdbot
            result = clawdbot.send_message(to, message, channel)

            if result.get("success"):
                return f"Sent to {to} via {channel}"
            else:
                return f"Failed: {result.get('error', 'Unknown error')}"
        except Exception as e:
            return f"Error: {e}"

    def get_clawdbot_channels(self) -> str:
        """Get list of connected channels."""
        try:
            from apprentice_agent.tools.clawdbot import clawdbot
            result = clawdbot.list_channels()

            if result.get("success"):
                channels = result.get("channels", [])
                if channels:
                    return ", ".join(channels) if isinstance(channels, list) else str(channels)
                return "No channels connected"
            else:
                return result.get("error", "Failed to get channels")
        except Exception as e:
            return f"Error: {e}"

    # =========================================================================
    # EVOEMO - Emotional State Tracking (Tool #20)
    # =========================================================================

    def get_mood_html(self) -> str:
        """Get current mood indicator HTML - simple colored dot."""
        try:
            agent = self._get_agent()
            if "evoemo" in agent.tools:
                evoemo = agent.tools["evoemo"]
                mood = evoemo.get_current_mood()

                if mood:
                    color = evoemo.get_mood_color()
                    # Simple colored dot: 🟢🟡🔴 style
                    dot_colors = {
                        "calm": "🟢",
                        "focused": "🔵",
                        "stressed": "🟠",
                        "frustrated": "🔴",
                        "excited": "🟡",
                        "tired": "🟣",
                        "curious": "🔵"
                    }
                    dot = dot_colors.get(mood.emotion, "⚪")

                    return f'''<div style="display: flex; align-items: center; gap: 6px; padding: 8px; background: #334155; border-radius: 6px;">
    <span style="font-size: 16px;">{dot}</span>
    <span style="color: {color}; font-size: 13px;">{mood.emotion.title()}</span>
    <span style="color: #64748b; font-size: 11px;">({mood.confidence}%)</span>
</div>'''
                else:
                    return '''<div style="display: flex; align-items: center; gap: 6px; padding: 8px; background: #334155; border-radius: 6px;">
    <span style="font-size: 16px;">⚪</span>
    <span style="color: #94a3b8; font-size: 13px;">No data</span>
</div>'''
        except Exception as e:
            return f'''<div style="padding: 8px; background: #334155; border-radius: 6px;">
<span style="color: #ef4444; font-size: 12px;">Error: {str(e)[:20]}</span>
</div>'''

    def get_mood_history_md(self) -> str:
        """Get mood history as markdown."""
        try:
            agent = self._get_agent()
            if "evoemo" in agent.tools:
                evoemo = agent.tools["evoemo"]

                # Get session summary
                session = evoemo.get_session_summary()
                daily = evoemo.get_daily_summary()

                md = f"**Session:** {session.get('dominant', 'calm')} ({session.get('readings', 0)} readings)\n\n"

                if daily:
                    md += f"**Today:** {daily.dominant_emotion} (avg {daily.average_confidence}% confidence)\n"
                    md += f"Distribution: {daily.emotion_distribution}\n\n"

                # Get patterns
                patterns = evoemo.get_patterns()
                if patterns.get("status") == "ok":
                    md += f"**7-day dominant:** {patterns.get('dominant_emotion', 'calm')}\n"
                    stress_hours = patterns.get("stress_hours", [])
                    if stress_hours:
                        md += f"Stress hours: {stress_hours}\n"

                return md if md.strip() else "No mood data yet."
        except Exception as e:
            return f"Error: {e}"

    def clear_mood_history(self) -> str:
        """Clear mood history."""
        try:
            agent = self._get_agent()
            if "evoemo" in agent.tools:
                result = agent.tools["evoemo"].clear_history()
                if result.get("success"):
                    return self.get_mood_html()
        except:
            pass
        return self.get_mood_html()

    def toggle_mood_tracking(self, enabled: bool) -> str:
        """Toggle mood tracking on/off."""
        try:
            agent = self._get_agent()
            if "evoemo" in agent.tools:
                agent.tools["evoemo"].set_enabled(enabled)
        except:
            pass
        return self.get_mood_html()

    # =========================================================================
    # INNER MONOLOGUE - Thought Visualization (Tool #21)
    # =========================================================================

    def get_thoughts_html(self) -> str:
        """Get formatted thoughts for display."""
        try:
            agent = self._get_agent()
            if "inner_monologue" in agent.tools:
                monologue = agent.tools["inner_monologue"]
                thoughts = monologue.get_recent_thoughts(15)

                if not thoughts:
                    return '''<div id="thought-stream" style="color: #64748b; font-style: italic;">
                        Waiting for Aura to think...
                    </div>'''

                # Format thoughts
                thought_icons = {
                    "perceive": "\U0001F50D",
                    "recall": "\U0001F4BE",
                    "reason": "\U0001F9E0",
                    "decide": "\u26A1",
                    "execute": "\U0001F527",
                    "reflect": "\U0001FA9E",
                    "uncertain": "\u2753",
                    "eureka": "\U0001F4A1",
                }

                lines = []
                for t in thoughts:
                    icon = thought_icons.get(t.type, "\U0001F4AD")
                    conf = f" [{t.confidence}%]" if t.confidence else ""
                    css_class = f"thought-{t.type}"
                    content = t.content[:80] + "..." if len(t.content) > 80 else t.content
                    lines.append(f'<div class="thought-line {css_class}">{icon} <strong>{t.type.upper()}</strong>{conf}: {content}</div>')

                return f'<div id="thought-stream">{"".join(lines)}</div>'
        except Exception as e:
            return f'<div id="thought-stream" style="color: #ef4444;">Error: {str(e)[:50]}</div>'

        return '<div id="thought-stream" style="color: #64748b;">Inner monologue not available</div>'

    def get_reasoning_chain(self) -> str:
        """Get the reasoning chain for 'why did you do that?' queries."""
        try:
            agent = self._get_agent()
            if "inner_monologue" in agent.tools:
                return agent.tools["inner_monologue"].get_reasoning_chain()
        except Exception as e:
            return f"Error: {e}"
        return "No reasoning chain available."

    def set_monologue_verbosity(self, level: int) -> str:
        """Set inner monologue verbosity level."""
        try:
            agent = self._get_agent()
            if "inner_monologue" in agent.tools:
                result = agent.tools["inner_monologue"].set_verbosity(int(level))
                return result
        except Exception as e:
            return f"Error: {e}"
        return "Monologue not available"

    def toggle_think_aloud(self, enabled: bool) -> str:
        """Toggle think-aloud mode (Sesame voice for thoughts)."""
        try:
            agent = self._get_agent()
            if "inner_monologue" in agent.tools:
                monologue = agent.tools["inner_monologue"]
                monologue.think_aloud_enabled = enabled

                # Connect TTS if enabling
                if enabled and tts.using_sesame:
                    monologue.connect_tts(tts)

                status = "enabled" if enabled else "disabled"
                return f"Think aloud {status}"
        except Exception as e:
            return f"Error: {e}"
        return "Monologue not available"

    def clear_thoughts(self) -> str:
        """Clear the thought display."""
        try:
            agent = self._get_agent()
            if "inner_monologue" in agent.tools:
                agent.tools["inner_monologue"].stream.clear()
        except:
            pass
        return self.get_thoughts_html()

    def export_thoughts(self) -> str:
        """Export current thoughts to file."""
        try:
            agent = self._get_agent()
            if "inner_monologue" in agent.tools:
                result = agent.tools["inner_monologue"].export_session()
                if result.get("success"):
                    return f"Exported to: {result.get('filepath', 'unknown')}"
                return f"Export failed: {result.get('error', 'unknown')}"
        except Exception as e:
            return f"Error: {e}"
        return "Monologue not available"

    def get_monologue_status(self) -> str:
        """Get inner monologue status."""
        try:
            agent = self._get_agent()
            if "inner_monologue" in agent.tools:
                monologue = agent.tools["inner_monologue"]
                status = monologue.execute("status")
                if status.get("success"):
                    session = status.get("active_session", "None")
                    verbosity = status.get("verbosity", 2)
                    think_aloud = "ON" if status.get("think_aloud") else "OFF"
                    count = status.get("thought_count", 0)
                    return f"Session: {session} | Verbosity: {verbosity} | Think Aloud: {think_aloud} | Thoughts: {count}"
        except Exception as e:
            return f"Error: {e}"
        return "Monologue not available"

    # =========================================================================
    # KNOWLEDGE GRAPH - Graph Memory Visualization (Tool #22)
    # =========================================================================

    def get_knowledge_graph_html(self, center_node: str = "", depth: int = 2) -> str:
        """Generate vis.js graph visualization."""
        try:
            agent = self._get_agent()
            if "knowledge_graph" not in agent.tools:
                return '<div style="color: #64748b;">Knowledge graph not available</div>'

            kg = agent.tools["knowledge_graph"]

            # Get nodes and edges to display
            if center_node and center_node.strip():
                related = kg.get_related(center_node.strip(), depth=depth, min_weight=0.2)
                nodes = related.get("nodes", [])
                edges = related.get("edges", [])
            else:
                nodes = kg.get_recent_nodes(limit=30)
                # Get edges between these nodes
                node_ids = {n.id for n in nodes}
                edges = []
                for edge in kg._edges.values():
                    if edge.source_id in node_ids and edge.target_id in node_ids:
                        edges.append(edge)

            if not nodes:
                return '''<div style="color: #64748b; padding: 20px; text-align: center;">
                    No nodes in knowledge graph yet. Start chatting to build knowledge!
                </div>'''

            # Build vis.js data
            import json
            nodes_js = []
            node_colors = {
                "concept": "#00d9ff",
                "entity": "#ff6b6b",
                "project": "#4ecdc4",
                "tool": "#ffe66d",
                "person": "#95e1d3",
                "emotion": "#f38181",
                "event": "#aa96da",
                "skill": "#fcbad3",
                "location": "#a8d8ea",
                "file": "#dfe6e9",
            }

            for node in nodes:
                color = node_colors.get(node.type, "#888888")
                icon = {
                    "concept": "\U0001F4A1",
                    "entity": "\U0001F4CC",
                    "project": "\U0001F4C1",
                    "tool": "\U0001F527",
                    "person": "\U0001F464",
                    "emotion": "\U0001F49A",
                    "event": "\U0001F4C5",
                    "skill": "\u26A1",
                    "location": "\U0001F4CD",
                    "file": "\U0001F4C4",
                }.get(node.type, "\U0001F4AD")

                nodes_js.append({
                    "id": node.id,
                    "label": f"{icon} {node.label[:20]}",
                    "title": f"{node.type}: {node.label}\\nConfidence: {int(node.confidence * 100)}%",
                    "color": color,
                    "shape": "dot",
                    "size": 15 + (node.access_count * 2),
                })

            edges_js = []
            for edge in edges:
                edges_js.append({
                    "from": edge.source_id,
                    "to": edge.target_id,
                    "label": edge.type,
                    "width": max(1, edge.weight * 3),
                    "arrows": "to",
                    "color": {"opacity": 0.6 + edge.weight * 0.4},
                })

            return f'''
            <div id="kg-graph" style="height: 350px; border: 1px solid #334155; border-radius: 8px; background: #1e293b;"></div>
            <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
            <script>
                (function() {{
                    var nodes = new vis.DataSet({json.dumps(nodes_js)});
                    var edges = new vis.DataSet({json.dumps(edges_js)});
                    var container = document.getElementById('kg-graph');
                    if (!container) return;
                    var data = {{ nodes: nodes, edges: edges }};
                    var options = {{
                        nodes: {{
                            font: {{ color: '#fff', size: 12 }},
                            borderWidth: 2,
                        }},
                        edges: {{
                            font: {{ color: '#94a3b8', size: 10, align: 'middle' }},
                            smooth: {{ type: 'curvedCW', roundness: 0.2 }},
                        }},
                        physics: {{
                            stabilization: {{ iterations: 100 }},
                            barnesHut: {{ gravitationalConstant: -2000, springLength: 100 }}
                        }},
                        interaction: {{
                            hover: true,
                            tooltipDelay: 200,
                        }}
                    }};
                    new vis.Network(container, data, options);
                }})();
            </script>
            '''
        except Exception as e:
            return f'<div style="color: #ef4444;">Error: {str(e)[:100]}</div>'

    def get_kg_stats(self) -> str:
        """Get knowledge graph statistics."""
        try:
            agent = self._get_agent()
            if "knowledge_graph" in agent.tools:
                kg = agent.tools["knowledge_graph"]
                stats = kg.get_stats()
                return f"**Nodes:** {stats['total_nodes']} | **Edges:** {stats['total_edges']} | **Clusters:** {stats['clusters']} | **Avg Confidence:** {stats['avg_confidence']}"
        except Exception as e:
            return f"Error: {e}"
        return "Stats unavailable"

    def kg_search(self, query: str) -> str:
        """Search the knowledge graph."""
        try:
            agent = self._get_agent()
            if "knowledge_graph" in agent.tools:
                kg = agent.tools["knowledge_graph"]
                nodes = kg.find_nodes(query, limit=10)
                if not nodes:
                    return "No matching nodes found."
                lines = []
                for node in nodes:
                    lines.append(f"- {node.format_display()}")
                return "\n".join(lines)
        except Exception as e:
            return f"Error: {e}"
        return "Search unavailable"

    def kg_find_path(self, source: str, target: str) -> str:
        """Find path between two concepts."""
        try:
            agent = self._get_agent()
            if "knowledge_graph" in agent.tools:
                kg = agent.tools["knowledge_graph"]
                path = kg.find_path(source.strip(), target.strip())
                if not path:
                    return f"No path found between '{source}' and '{target}'"
                path_parts = []
                for n1, edge, n2 in path:
                    path_parts.append(f"{n1.label} --{edge.type}--> {n2.label}")
                return " | ".join(path_parts)
        except Exception as e:
            return f"Error: {e}"
        return "Path search unavailable"

    def kg_add_node(self, node_type: str, label: str) -> str:
        """Add a node to the knowledge graph."""
        try:
            agent = self._get_agent()
            if "knowledge_graph" in agent.tools:
                kg = agent.tools["knowledge_graph"]
                node = kg.add_node(node_type, label, source="gui")
                return f"Added: {node.format_display()}"
        except Exception as e:
            return f"Error: {e}"
        return "Add node unavailable"

    def kg_consolidate(self) -> str:
        """Run knowledge graph consolidation."""
        try:
            agent = self._get_agent()
            if "knowledge_graph" in agent.tools:
                kg = agent.tools["knowledge_graph"]
                result = kg.consolidate()
                kg.save()
                return f"Consolidated: {result['merged_nodes']} merged, {result['pruned_edges']} edges pruned"
        except Exception as e:
            return f"Error: {e}"
        return "Consolidation unavailable"

    # ========================================================================
    # METACOGNITIVE GUARDIAN (Tool #23)
    # ========================================================================

    def get_guardian_stats(self) -> dict:
        """Get guardian statistics."""
        try:
            agent = self._get_agent()
            if hasattr(agent, 'guardian') and agent.guardian:
                return agent.guardian.get_stats()
        except Exception:
            pass
        return {
            "session_predictions": 0,
            "interventions_triggered": 0,
            "failure_patterns_learned": 0,
            "monitoring_level": "medium",
            "thresholds": {"warning": 0.3, "intervention": 0.6, "abort": 0.9}
        }

    def get_guardian_status_html(self) -> str:
        """Get guardian status HTML."""
        stats = self.get_guardian_stats()
        return f'''<div style="background: #334155; padding: 12px; border-radius: 8px; margin: 8px 0;">
<span class="status-online"></span>
<strong style="color: #22c55e;">Guardian Active</strong>
<div style="color: #94a3b8; font-size: 12px; margin-top: 4px;">Monitoring: {stats["monitoring_level"]}</div>
</div>'''

    def set_guardian_level(self, level: str) -> dict:
        """Set guardian monitoring level."""
        try:
            agent = self._get_agent()
            if hasattr(agent, 'guardian') and agent.guardian:
                agent.guardian.set_monitoring_level(level)
                return agent.guardian.get_stats()["thresholds"]
        except Exception:
            pass
        return {"warning": 0.3, "intervention": 0.6, "abort": 0.9}

    def get_guardian_predictions(self) -> str:
        """Get recent predictions as markdown."""
        try:
            agent = self._get_agent()
            if hasattr(agent, 'guardian') and agent.guardian:
                predictions = agent.guardian.session_predictions
                if not predictions:
                    return "No predictions this session."
                lines = []
                for p in predictions[-5:]:
                    emoji = "⚠️" if p.probability >= 0.6 else "💡"
                    lines.append(f"{emoji} **{p.failure_type.value}** ({p.probability:.0%})")
                    lines.append(f"   {p.reasoning[:60]}...")
                return "\n".join(lines)
        except Exception:
            pass
        return "Guardian unavailable."

    def reset_guardian_session(self) -> str:
        """Reset guardian session stats."""
        try:
            agent = self._get_agent()
            if hasattr(agent, 'guardian') and agent.guardian:
                agent.guardian.reset_session()
                return "Session reset."
        except Exception:
            pass
        return "Reset unavailable."

    def record_feedback(self, was_helpful: bool) -> str:
        """Record user feedback for guardian learning."""
        try:
            agent = self._get_agent()
            if hasattr(agent, 'record_user_feedback'):
                agent.record_user_feedback(was_helpful)
                return "Feedback recorded." if was_helpful else "Feedback recorded - will improve."
        except Exception:
            pass
        return "Feedback unavailable."

    # ========================================================================
    # NEURODREAM - Sleep/Dream Memory Consolidation (Tool #24)
    # ========================================================================

    def get_neurodream_status_html(self) -> str:
        """Get NeuroDream sleep status HTML."""
        try:
            agent = self._get_agent()
            if hasattr(agent, 'neurodream') and agent.neurodream:
                nd = agent.neurodream
                status = nd.get_status()

                if status.get("is_sleeping"):
                    phase = status.get("current_phase", "unknown")
                    phase_icons = {"light": "🌙", "deep": "💤", "rem": "🌈"}
                    icon = phase_icons.get(phase, "😴")
                    return f'''<div style="background: #334155; padding: 12px; border-radius: 8px; margin: 8px 0;">
<span class="status-online" style="background: #8b5cf6;"></span>
<strong style="color: #a78bfa;">{icon} Dreaming ({phase.upper()})</strong>
<div style="color: #94a3b8; font-size: 12px; margin-top: 4px;">Memory consolidation in progress...</div>
</div>'''
                else:
                    sessions = status.get("total_sessions", 0)
                    insights = status.get("total_insights", 0)
                    return f'''<div style="background: #334155; padding: 12px; border-radius: 8px; margin: 8px 0;">
<span class="status-online"></span>
<strong style="color: #22c55e;">☀️ Awake</strong>
<div style="color: #94a3b8; font-size: 12px; margin-top: 4px;">{sessions} sessions, {insights} insights</div>
</div>'''
        except Exception as e:
            return f'''<div style="background: #334155; padding: 12px; border-radius: 8px; margin: 8px 0;">
<span class="status-offline"></span>
<strong style="color: #ef4444;">NeuroDream Error</strong>
<div style="color: #94a3b8; font-size: 12px; margin-top: 4px;">{str(e)[:30]}</div>
</div>'''

    def start_sleep(self) -> str:
        """Manually trigger sleep cycle (non-blocking)."""
        try:
            agent = self._get_agent()
            if hasattr(agent, 'neurodream') and agent.neurodream:
                # Just trigger sleep and return immediately - don't wait
                result = agent.neurodream.enter_sleep(trigger="manual_gui")
                if result.get("success"):
                    session_id = result.get("session_id", "")
                    return f'''<div style="display: flex; align-items: center; gap: 8px;">
<span style="width: 10px; height: 10px; background: #3b82f6; border-radius: 50%; animation: pulse 1s infinite;"></span>
<span style="color: #60a5fa;">😴 Sleeping</span>
<span style="color: #94a3b8; font-size: 12px;">Session: {session_id[:20]}...</span>
</div>
<style>@keyframes pulse {{ 0%, 100% {{ opacity: 1; }} 50% {{ opacity: 0.5; }} }}</style>'''
                return f"Error: {result.get('error', 'Failed to start')}"
            return "NeuroDream not available"
        except Exception as e:
            return f"Error: {e}"

    def wake_up(self) -> str:
        """Wake up from sleep cycle."""
        try:
            agent = self._get_agent()
            if hasattr(agent, 'neurodream') and agent.neurodream:
                result = agent.neurodream.wake_up(reason="user_gui")
                return self.get_neurodream_status_html()
        except Exception as e:
            return f"Error: {e}"

    def get_dream_journal_md(self) -> str:
        """Get recent dream journal entries as markdown."""
        try:
            agent = self._get_agent()
            if hasattr(agent, 'neurodream') and agent.neurodream:
                entries = agent.neurodream.get_dream_journal(n=5)
                if not entries:
                    return "No dreams recorded yet. Try 'go to sleep' to start a sleep cycle."

                lines = []
                for entry in entries:
                    phase = entry.get("phase", "unknown")
                    phase_icons = {"light": "🌙", "deep": "💤", "rem": "🌈"}
                    icon = phase_icons.get(phase, "💭")
                    timestamp = entry.get("timestamp", "")[:16]
                    content = entry.get("content", "")[:100]
                    lines.append(f"**{icon} {phase.upper()}** ({timestamp})")
                    lines.append(f"> {content}...")
                    lines.append("")
                return "\n".join(lines) if lines else "No dreams yet."
        except Exception as e:
            return f"Error: {e}"

    def get_dream_insights_md(self) -> str:
        """Get dream insights as markdown."""
        try:
            agent = self._get_agent()
            if hasattr(agent, 'neurodream') and agent.neurodream:
                insights = agent.neurodream.get_insights()
                if not insights:
                    return "No insights yet. Sleep cycles generate insights during memory consolidation."

                lines = []
                for insight in insights[-5:]:
                    itype = insight.get("type", "pattern")
                    content = insight.get("content", "")[:80]
                    confidence = insight.get("confidence", 0)
                    icon = {"pattern": "🔮", "connection": "🔗", "creative": "💡", "memory": "🧠"}.get(itype, "✨")
                    lines.append(f"{icon} **{itype.title()}** ({confidence}%): {content}")
                return "\n".join(lines) if lines else "No insights yet."
        except Exception as e:
            return f"Error: {e}"

    def get_sleep_patterns_md(self) -> str:
        """Get consolidated patterns as markdown."""
        try:
            agent = self._get_agent()
            if hasattr(agent, 'neurodream') and agent.neurodream:
                patterns = agent.neurodream.get_patterns()
                if not patterns:
                    return "No patterns consolidated yet."

                lines = []
                for p in patterns[-5:]:
                    name = p.get("pattern_name", "Unknown")
                    strength = p.get("strength", 0)
                    memories = p.get("memories_consolidated", 0)
                    lines.append(f"🔄 **{name}** (strength: {strength:.0%}, {memories} memories)")
                return "\n".join(lines) if lines else "No patterns yet."
        except Exception as e:
            return f"Error: {e}"

    def get_neurodream_stats(self) -> dict:
        """Get NeuroDream statistics."""
        try:
            agent = self._get_agent()
            if hasattr(agent, 'neurodream') and agent.neurodream:
                return agent.neurodream.get_status()
        except Exception:
            pass
        return {"total_sessions": 0, "total_insights": 0, "is_sleeping": False}


# ============================================================================
# CREATE APP
# ============================================================================

def create_app():
    gui = AuraGUI()

    with gr.Blocks(title="Aura") as app:

        gr.HTML("""<div style="text-align: center; padding: 16px; border-bottom: 1px solid #334155;">
            <h1 style="color: #ffffff; margin: 0;">Aura</h1>
            <p style="color: #94a3b8; margin: 4px 0 0 0;">AI Assistant</p>
        </div>""")

        with gr.Row():
            with gr.Column(scale=1, min_width=200):
                gr.HTML("""<div style="text-align: center; padding: 16px;">
                    <div style="font-size: 40px;">🤖</div>
                    <div style="color: #ffffff; font-weight: bold; font-size: 18px;">Aura</div>
                    <div style="color: #94a3b8; font-size: 13px;">Apprentice Agent</div>
                </div>""")

                # EvoEmo Mood Indicator (Tool #20)
                gr.Markdown("**Your Mood**", elem_classes=["section-header"])
                mood_indicator = gr.HTML(value=gui.get_mood_html())

                with gr.Accordion("Mood Details", open=False):
                    mood_history_md = gr.Markdown("No data yet")
                    mood_tracking_toggle = gr.Checkbox(label="Enable Tracking", value=True)
                    with gr.Row():
                        refresh_mood_btn = gr.Button("Refresh", size="sm")
                        clear_mood_btn = gr.Button("Clear History", size="sm")

                # Voice Controls
                voice_status = gr.HTML(value=gui.get_voice_status_html())
                voice_toggle = gr.Checkbox(label="🔊 Enable Voice", value=False)
                with gr.Row():
                    test_voice_btn = gr.Button("Test Voice", size="sm")
                    load_sesame_btn = gr.Button("Load Sesame", size="sm")
                voice_output = gr.Textbox(visible=False)

                # FluxMind Status
                fluxmind_html = gr.HTML(value=gui.get_status_html())
                refresh_btn = gr.Button("Refresh Status", size="sm")

                # Settings
                max_iter = gr.Slider(1, 20, value=10, step=1, label="Max Iterations")

                with gr.Accordion("Stats", open=False):
                    stats_md = gr.Markdown("Click Load")
                    stats_btn = gr.Button("Load", size="sm")

                # Clawdbot Integration
                with gr.Accordion("Clawdbot", open=False):
                    clawdbot_status = gr.HTML(value=gui.get_clawdbot_status_html())
                    with gr.Row():
                        start_gateway_btn = gr.Button("Start Gateway", size="sm")
                        refresh_clawdbot_btn = gr.Button("Refresh", size="sm")

                    clawdbot_channel = gr.Dropdown(
                        choices=["whatsapp", "telegram", "discord", "signal", "imessage"],
                        value="whatsapp",
                        label="Channel",
                        scale=1
                    )
                    clawdbot_to = gr.Textbox(
                        label="Send To",
                        placeholder="+1234567890",
                        scale=1
                    )
                    clawdbot_msg = gr.Textbox(
                        label="Message",
                        placeholder="Hello from Aura!",
                        scale=1
                    )
                    clawdbot_send_btn = gr.Button("Send Message", size="sm")
                    clawdbot_result = gr.Markdown("")

            with gr.Column(scale=4):
                chatbot = gr.Chatbot(height=500, show_label=False)

                # Inner Monologue Panel (Tool #21)
                with gr.Accordion("\U0001F9E0 Aura's Thoughts", open=False) as thoughts_panel:
                    thought_display = gr.HTML(
                        value=gui.get_thoughts_html(),
                        elem_id="thought-stream-container"
                    )
                    monologue_status = gr.Markdown(gui.get_monologue_status())

                    with gr.Row():
                        verbosity_slider = gr.Slider(
                            minimum=0, maximum=3, step=1, value=2,
                            label="Verbosity (0=silent, 3=debug)",
                            scale=2
                        )
                        think_aloud_toggle = gr.Checkbox(
                            label="\U0001F50A Think Aloud",
                            value=False,
                            scale=1
                        )
                    with gr.Row():
                        refresh_thoughts_btn = gr.Button("Refresh", size="sm")
                        clear_thoughts_btn = gr.Button("Clear", size="sm")
                        export_thoughts_btn = gr.Button("Export", size="sm")
                        why_btn = gr.Button("Why?", size="sm")
                    reasoning_output = gr.Markdown(visible=False)

                # Knowledge Graph Panel (Tool #22)
                with gr.Accordion("\U0001F578\uFE0F Knowledge Graph", open=False) as kg_panel:
                    kg_graph_html = gr.HTML(
                        value=gui.get_knowledge_graph_html(),
                        elem_id="knowledge-graph-container"
                    )
                    kg_stats = gr.Markdown(gui.get_kg_stats())

                    with gr.Row():
                        kg_search_input = gr.Textbox(
                            label="Search/Center Node",
                            placeholder="e.g., FluxMind, Python...",
                            scale=3
                        )
                        kg_depth_slider = gr.Slider(
                            minimum=1, maximum=4, step=1, value=2,
                            label="Depth",
                            scale=1
                        )
                    with gr.Row():
                        kg_refresh_btn = gr.Button("\U0001F504 Refresh", size="sm")
                        kg_consolidate_btn = gr.Button("\U0001F9F9 Consolidate", size="sm")

                    with gr.Accordion("Add Knowledge", open=False):
                        with gr.Row():
                            kg_node_type = gr.Dropdown(
                                choices=["concept", "entity", "person", "project", "tool", "event", "skill", "location", "file"],
                                value="concept",
                                label="Type",
                                scale=1
                            )
                            kg_node_label = gr.Textbox(label="Label", scale=2)
                            kg_add_btn = gr.Button("Add", size="sm", scale=1)
                        kg_add_result = gr.Markdown("")

                    with gr.Accordion("Find Path", open=False):
                        with gr.Row():
                            kg_path_source = gr.Textbox(label="From", scale=1)
                            kg_path_target = gr.Textbox(label="To", scale=1)
                            kg_path_btn = gr.Button("Find", size="sm")
                        kg_path_result = gr.Markdown("")

                # Metacognitive Guardian Panel (Tool #23)
                with gr.Accordion("\U0001F52E Metacognitive Guardian", open=False) as guardian_panel:
                    gr.Markdown("### Self-Aware Failure Prediction")

                    with gr.Row():
                        guardian_status = gr.HTML(value=gui.get_guardian_status_html())
                        guardian_level = gr.Dropdown(
                            label="Monitoring Level",
                            choices=["low", "medium", "high", "critical"],
                            value="medium",
                            scale=1
                        )

                    with gr.Row():
                        guardian_interventions = gr.Number(
                            label="Interventions",
                            value=0,
                            interactive=False
                        )
                        guardian_patterns = gr.Number(
                            label="Patterns Learned",
                            value=0,
                            interactive=False
                        )
                        guardian_predictions = gr.Number(
                            label="Session Predictions",
                            value=0,
                            interactive=False
                        )

                    with gr.Row():
                        refresh_guardian_btn = gr.Button("\U0001F504 Refresh", size="sm")
                        reset_guardian_btn = gr.Button("\U0001F5D1 Reset Session", size="sm")

                    with gr.Accordion("\U00002699\uFE0F Thresholds", open=False):
                        gr.Markdown("""
- **Warning** (yellow): Show caution message
- **Intervention** (orange): Take action (clarify, switch tool)
- **Abort** (red): Stop and hand off to user
""")
                        guardian_thresholds = gr.JSON(
                            label="Current Thresholds",
                            value={"warning": 0.3, "intervention": 0.6, "abort": 0.9}
                        )

                    with gr.Accordion("\U0001F4CA Recent Predictions", open=False):
                        guardian_predictions_md = gr.Markdown("No predictions yet.")

                    with gr.Row():
                        feedback_helpful_btn = gr.Button("\U0001F44D Helpful", size="sm", variant="primary")
                        feedback_unhelpful_btn = gr.Button("\U0001F44E Not Helpful", size="sm")
                        feedback_status = gr.Textbox(value="", show_label=False, interactive=False, scale=2)

                # NeuroDream Panel (Tool #24) - Sleep/Dream Memory Consolidation
                with gr.Accordion("😴 NeuroDream", open=False) as neurodream_panel:
                    gr.Markdown("### Sleep/Dream Memory Consolidation")

                    with gr.Row():
                        neurodream_status = gr.HTML(value=gui.get_neurodream_status_html())

                    with gr.Row():
                        neurodream_sessions = gr.Number(
                            label="Sleep Sessions",
                            value=0,
                            interactive=False
                        )
                        neurodream_insights = gr.Number(
                            label="Insights Generated",
                            value=0,
                            interactive=False
                        )

                    with gr.Row():
                        sleep_btn = gr.Button("🌙 Sleep Now", size="sm", variant="primary")
                        wake_btn = gr.Button("☀️ Wake Up", size="sm")
                        refresh_neurodream_btn = gr.Button("🔄 Refresh", size="sm")

                    with gr.Accordion("📖 Dream Journal", open=False):
                        dream_journal_md = gr.Markdown("No dreams recorded yet.")

                    with gr.Accordion("💡 Dream Insights", open=False):
                        dream_insights_md = gr.Markdown("No insights yet.")

                    with gr.Accordion("🔄 Consolidated Patterns", open=False):
                        sleep_patterns_md = gr.Markdown("No patterns yet.")

                with gr.Row():
                    msg = gr.Textbox(placeholder="Type message...", show_label=False, scale=5)
                    send = gr.Button("Send", variant="primary", scale=1)

                clear = gr.Button("Clear", size="sm")

        # Events
        def on_send(message, history, voice_enabled):
            for h in gui.run_agent(message, history, voice_enabled):
                yield h

        send.click(on_send, [msg, chatbot, voice_toggle], chatbot).then(lambda: "", outputs=msg)
        msg.submit(on_send, [msg, chatbot, voice_toggle], chatbot).then(lambda: "", outputs=msg)
        clear.click(lambda: [], outputs=chatbot)
        refresh_btn.click(gui.get_status_html, outputs=fluxmind_html)
        stats_btn.click(gui.get_stats, outputs=stats_md)
        max_iter.change(lambda v: setattr(gui, 'max_iterations', int(v)), inputs=max_iter)
        voice_toggle.change(gui.toggle_voice, inputs=voice_toggle, outputs=voice_status)
        test_voice_btn.click(gui.test_voice, outputs=voice_status)
        load_sesame_btn.click(gui.load_sesame, outputs=voice_status)

        # Clawdbot events
        start_gateway_btn.click(gui.start_clawdbot_gateway, outputs=clawdbot_status)
        refresh_clawdbot_btn.click(gui.get_clawdbot_status_html, outputs=clawdbot_status)
        clawdbot_send_btn.click(
            gui.send_clawdbot_message,
            inputs=[clawdbot_to, clawdbot_msg, clawdbot_channel],
            outputs=clawdbot_result
        )

        # EvoEmo events
        refresh_mood_btn.click(gui.get_mood_html, outputs=mood_indicator)
        refresh_mood_btn.click(gui.get_mood_history_md, outputs=mood_history_md)
        clear_mood_btn.click(gui.clear_mood_history, outputs=mood_indicator)
        mood_tracking_toggle.change(gui.toggle_mood_tracking, inputs=mood_tracking_toggle, outputs=mood_indicator)

        # Update mood indicator after each message
        send.click(gui.get_mood_html, outputs=mood_indicator)
        msg.submit(gui.get_mood_html, outputs=mood_indicator)

        # Inner Monologue events (Tool #21)
        refresh_thoughts_btn.click(gui.get_thoughts_html, outputs=thought_display)
        refresh_thoughts_btn.click(gui.get_monologue_status, outputs=monologue_status)
        clear_thoughts_btn.click(gui.clear_thoughts, outputs=thought_display)
        export_thoughts_btn.click(gui.export_thoughts, outputs=monologue_status)
        verbosity_slider.change(gui.set_monologue_verbosity, inputs=verbosity_slider, outputs=monologue_status)
        think_aloud_toggle.change(gui.toggle_think_aloud, inputs=think_aloud_toggle, outputs=monologue_status)

        # Why button shows reasoning chain
        def show_reasoning():
            return gr.update(visible=True, value=gui.get_reasoning_chain())
        why_btn.click(show_reasoning, outputs=reasoning_output)

        # Update thoughts after each message
        send.click(gui.get_thoughts_html, outputs=thought_display)
        msg.submit(gui.get_thoughts_html, outputs=thought_display)

        # Knowledge Graph events (Tool #22)
        def refresh_kg(search_query, depth):
            return gui.get_knowledge_graph_html(search_query, int(depth))

        kg_refresh_btn.click(refresh_kg, inputs=[kg_search_input, kg_depth_slider], outputs=kg_graph_html)
        kg_refresh_btn.click(gui.get_kg_stats, outputs=kg_stats)
        kg_search_input.submit(refresh_kg, inputs=[kg_search_input, kg_depth_slider], outputs=kg_graph_html)
        kg_consolidate_btn.click(gui.kg_consolidate, outputs=kg_stats)
        kg_add_btn.click(gui.kg_add_node, inputs=[kg_node_type, kg_node_label], outputs=kg_add_result)
        kg_path_btn.click(gui.kg_find_path, inputs=[kg_path_source, kg_path_target], outputs=kg_path_result)

        # Update knowledge graph after messages (learn from conversation)
        send.click(lambda: gui.get_knowledge_graph_html("", 2), outputs=kg_graph_html)

        # Metacognitive Guardian events (Tool #23)
        def refresh_guardian():
            stats = gui.get_guardian_stats()
            return (
                stats["interventions_triggered"],
                stats["failure_patterns_learned"],
                stats["session_predictions"],
                stats["thresholds"],
                gui.get_guardian_predictions(),
                gui.get_guardian_status_html()
            )

        refresh_guardian_btn.click(
            refresh_guardian,
            outputs=[guardian_interventions, guardian_patterns, guardian_predictions,
                    guardian_thresholds, guardian_predictions_md, guardian_status]
        )

        guardian_level.change(
            gui.set_guardian_level,
            inputs=[guardian_level],
            outputs=[guardian_thresholds]
        )

        reset_guardian_btn.click(
            gui.reset_guardian_session,
            outputs=[feedback_status]
        )

        feedback_helpful_btn.click(
            lambda: gui.record_feedback(True),
            outputs=[feedback_status]
        )

        feedback_unhelpful_btn.click(
            lambda: gui.record_feedback(False),
            outputs=[feedback_status]
        )

        # Update guardian stats after each message
        send.click(refresh_guardian, outputs=[guardian_interventions, guardian_patterns,
                  guardian_predictions, guardian_thresholds, guardian_predictions_md, guardian_status])

        # NeuroDream events (Tool #24)
        def refresh_neurodream():
            stats = gui.get_neurodream_stats()
            return (
                stats.get("total_sessions", 0),
                stats.get("total_insights", 0),
                gui.get_neurodream_status_html(),
                gui.get_dream_journal_md(),
                gui.get_dream_insights_md(),
                gui.get_sleep_patterns_md()
            )

        refresh_neurodream_btn.click(
            refresh_neurodream,
            outputs=[neurodream_sessions, neurodream_insights, neurodream_status,
                    dream_journal_md, dream_insights_md, sleep_patterns_md]
        )

        sleep_btn.click(
            gui.start_sleep,
            outputs=[neurodream_status]
        )

        wake_btn.click(
            gui.wake_up,
            outputs=[neurodream_status]
        )

        # Update neurodream status after messages (may trigger idle sleep)
        send.click(gui.get_neurodream_status_html, outputs=neurodream_status)
        msg.submit(gui.get_neurodream_status_html, outputs=neurodream_status)

    return app


if __name__ == "__main__":
    print("Starting Aura GUI...")
    print(f"TTS Engine: {tts.get_status()}")
    print(f"TTS Available: {tts.available}")
    app = create_app()
    app.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True, css=CUSTOM_CSS)
