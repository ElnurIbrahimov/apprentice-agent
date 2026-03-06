"""VoicePresenceService — singleton TTS service using Kokoro neural TTS.

Replaces pyttsx3 (robotic) with Kokoro ONNX (natural, ~300MB, ~1GB VRAM).
Same public API — speak(), synthesize_wav(), synthesize_audio_array(), get_status().

Kokoro is thread-safe; worker thread kept for queue ordering + non-blocking speak().
"""

import io
import logging
import queue
import threading
import wave
from typing import Optional

logger = logging.getLogger(__name__)

_instance: Optional["VoicePresenceService"] = None
_instance_lock = threading.Lock()

DEFAULT_VOICE = "af_heart"
DEFAULT_SPEED = 1.0


def _get_kokoro():
    """Reuse the Kokoro singleton from voice_synth (one model load per process)."""
    try:
        from aura.tools.voice_synth import _get_kokoro as _synth_get
        return _synth_get()
    except Exception:
        # Fallback: load independently if voice_synth unavailable
        import threading as _t
        global _fallback_model, _fallback_lock
        if "_fallback_lock" not in globals():
            _fallback_lock = _t.Lock()
            _fallback_model = None
        with _fallback_lock:
            if _fallback_model is None:
                from kokoro_onnx import Kokoro
                logger.info("[VoicePresence] Loading Kokoro (fallback)...")
                _fallback_model = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")
            return _fallback_model


def _synthesize(text: str, voice: str, speed: float):
    """Return (samples: np.ndarray, sample_rate: int) from Kokoro."""
    kokoro = _get_kokoro()
    samples, sample_rate = kokoro.create(text, voice=voice, speed=speed, lang="en-us")
    return samples, sample_rate


def _samples_to_wav_bytes(samples, sample_rate: int) -> bytes:
    """Convert float32 numpy samples to 16-bit PCM WAV bytes."""
    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, samples, sample_rate, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf.read()


class VoicePresenceService:
    """Singleton TTS service — Kokoro neural TTS engine."""

    def __init__(self, enabled: bool = True):
        self._enabled = enabled
        self._speech_queue: queue.Queue = queue.Queue()
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._speaking = threading.Event()
        self._ready = False
        self._started = False

    def start(self) -> None:
        """Start the worker thread."""
        if self._started:
            return
        self._stop_event.clear()
        self._worker_thread = threading.Thread(
            target=self._worker, name="VoicePresenceWorker", daemon=True
        )
        self._worker_thread.start()
        self._started = True
        logger.info("[VoicePresence] Worker thread started (Kokoro TTS)")

    def stop(self) -> None:
        """Signal the worker to exit and join the thread."""
        if not self._started:
            return
        self._stop_event.set()
        self._speech_queue.put(None)
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=5.0)
        while not self._speech_queue.empty():
            try:
                self._speech_queue.get_nowait()
            except queue.Empty:
                break
        self._started = False
        logger.info("[VoicePresence] Worker thread stopped")

    def speak(self, text: str, emotion: Optional[str] = None, block: bool = False) -> None:
        """Queue text for speech through server speakers."""
        if not self._enabled or not self._started:
            return
        if block:
            done_event = threading.Event()
            result_holder = {"done": done_event, "wav_bytes": None, "_block_only": True}
            self._speech_queue.put((text, emotion, result_holder))
            done_event.wait(timeout=30.0)
        else:
            self._speech_queue.put((text, emotion, None))

    def synthesize_wav(self, text: str, emotion: Optional[str] = None) -> bytes:
        """Generate WAV bytes for the given text (used by REST endpoint)."""
        done_event = threading.Event()
        result_holder = {"done": done_event, "wav_bytes": b""}
        self._speech_queue.put((text, emotion, result_holder))
        done_event.wait(timeout=30.0)
        return result_holder.get("wav_bytes", b"")

    def set_enabled(self, enabled: bool) -> None:
        """Toggle voice on/off at runtime."""
        self._enabled = enabled
        logger.info(f"[VoicePresence] Enabled: {enabled}")

    def synthesize_audio_array(self, text: str) -> tuple:
        """Synthesize text to a numpy array — returns (np.ndarray, sample_rate)."""
        import numpy as np
        wav_bytes = self.synthesize_wav(text)
        if not wav_bytes:
            return None, 0
        try:
            with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
                sr = wf.getframerate()
                frames = wf.readframes(wf.getnframes())
                audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                return audio, sr
        except Exception:
            return None, 0

    def is_speaking(self) -> bool:
        return self._speaking.is_set()

    def get_status(self) -> dict:
        return {
            "available": self._started and self._ready,
            "engine": "kokoro",
            "enabled": self._enabled,
            "speaking": self.is_speaking(),
            "sesame_loaded": False,
        }

    # ------------------------------------------------------------------
    # Worker thread
    # ------------------------------------------------------------------

    def _worker(self) -> None:
        """Worker thread — pre-loads Kokoro then processes the speech queue."""
        try:
            _get_kokoro()
            self._ready = True
            logger.info("[VoicePresence] Kokoro loaded and ready on worker thread")
        except Exception as e:
            logger.error(f"[VoicePresence] Failed to load Kokoro: {e}")
            return

        while not self._stop_event.is_set():
            try:
                item = self._speech_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if item is None:
                break

            text, emotion, result_holder = item
            voice, speed = self._emotion_to_params(emotion)

            try:
                samples, sample_rate = _synthesize(text, voice, speed)

                if result_holder is not None and not result_holder.get("_block_only"):
                    # WAV synthesis — return bytes to caller
                    result_holder["wav_bytes"] = _samples_to_wav_bytes(samples, sample_rate)
                    result_holder["done"].set()
                elif result_holder is not None and result_holder.get("_block_only"):
                    # Blocking speak
                    self._speaking.set()
                    self._play(samples, sample_rate)
                    self._speaking.clear()
                    result_holder["done"].set()
                else:
                    # Non-blocking speak through server speakers
                    self._speaking.set()
                    self._play(samples, sample_rate)
                    self._speaking.clear()

            except Exception as e:
                logger.error(f"[VoicePresence] Speech error: {e}")
                self._speaking.clear()
                if result_holder is not None:
                    result_holder.get("done") and result_holder["done"].set()

        logger.info("[VoicePresence] Worker thread exiting")

    def _play(self, samples, sample_rate: int) -> None:
        """Play audio through server speakers — sounddevice with winsound fallback."""
        try:
            import sounddevice as sd
            sd.play(samples, sample_rate)
            sd.wait()
        except Exception as e:
            logger.warning(f"[VoicePresence] sounddevice failed: {e}, trying winsound")
            try:
                import tempfile, os, soundfile as sf, winsound
                fd, path = tempfile.mkstemp(suffix=".wav")
                os.close(fd)
                sf.write(path, samples, sample_rate)
                winsound.PlaySound(path, winsound.SND_FILENAME)
                os.unlink(path)
            except Exception as e2:
                logger.error(f"[VoicePresence] All playback methods failed: {e2}")

    def _emotion_to_params(self, emotion: Optional[str]) -> tuple:
        """Map emotion string → (voice, speed) for Kokoro."""
        speed = DEFAULT_SPEED
        try:
            from aura.tools.evoemo_prompts import VOICE_PARAMS
            params = VOICE_PARAMS.get(emotion or "calm", VOICE_PARAMS["calm"])
            speed = float(params.get("rate", 1.0)) * DEFAULT_SPEED
        except Exception:
            pass
        return DEFAULT_VOICE, max(0.5, min(2.0, speed))


def get_voice_presence(enabled: bool = True) -> VoicePresenceService:
    """Get or create the singleton VoicePresenceService instance."""
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = VoicePresenceService(enabled=enabled)
    return _instance
