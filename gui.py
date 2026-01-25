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
"""


# ============================================================================
# GUI CLASS
# ============================================================================

class AuraGUI:
    def __init__(self):
        self.max_iterations = 10
        self.voice_enabled = False
        self.agent = None

    def _get_agent(self):
        """Get or create agent instance."""
        if self.agent is None:
            self.agent = ApprenticeAgent()
        return self.agent

    def _check_fluxmind(self) -> dict:
        try:
            from apprentice_agent.tools import FluxMindTool, FLUXMIND_AVAILABLE
            if FLUXMIND_AVAILABLE:
                models_path = Path(__file__).parent / "models" / "fluxmind_v0751.pt"
                tool = FluxMindTool(str(models_path))
                if tool.is_available():
                    return {"available": True, "version": "0.75.1"}
            return {"available": False}
        except:
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
        """Get current mood indicator HTML."""
        try:
            agent = self._get_agent()
            if "evoemo" in agent.tools:
                evoemo = agent.tools["evoemo"]
                mood = evoemo.get_current_mood()

                if mood:
                    emoji = evoemo.get_mood_emoji()
                    color = evoemo.get_mood_color()
                    confidence = mood.confidence

                    return f'''<div style="background: #334155; padding: 12px; border-radius: 8px; margin: 8px 0;">
<div style="display: flex; align-items: center; gap: 8px;">
    <span style="font-size: 24px;">{emoji}</span>
    <div>
        <strong style="color: {color};">{mood.emotion.title()}</strong>
        <div style="color: #94a3b8; font-size: 11px;">{confidence}% confidence</div>
    </div>
</div>
</div>'''
                else:
                    return '''<div style="background: #334155; padding: 12px; border-radius: 8px; margin: 8px 0;">
<div style="display: flex; align-items: center; gap: 8px;">
    <span style="font-size: 24px;">😐</span>
    <div>
        <strong style="color: #94a3b8;">No data yet</strong>
        <div style="color: #64748b; font-size: 11px;">Start chatting</div>
    </div>
</div>
</div>'''
        except Exception as e:
            return f'''<div style="background: #334155; padding: 12px; border-radius: 8px; margin: 8px 0;">
<span style="color: #ef4444;">EvoEmo error: {str(e)[:30]}</span>
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

    return app


if __name__ == "__main__":
    print("Starting Aura GUI...")
    print(f"TTS Engine: {tts.get_status()}")
    print(f"TTS Available: {tts.available}")
    app = create_app()
    app.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True, css=CUSTOM_CSS)
