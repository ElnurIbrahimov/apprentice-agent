"""VoicePresenceService — singleton TTS service with dedicated pyttsx3 worker thread.

Handles all AURA voice output (chat responses + proactive messages) through
server speakers and provides WAV synthesis for REST endpoint playback.

pyttsx3 requires engine init + runAndWait() on the SAME thread (Windows COM
constraint), so all TTS work is funneled through a single worker thread.
"""

import logging
import os
import queue
import tempfile
import threading
from typing import Optional

logger = logging.getLogger(__name__)

# Singleton instance
_instance: Optional["VoicePresenceService"] = None
_instance_lock = threading.Lock()


class VoicePresenceService:
    """Singleton TTS service with a dedicated pyttsx3 worker thread."""

    def __init__(self, enabled: bool = True):
        self._enabled = enabled
        self._speech_queue: queue.Queue = queue.Queue()
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._speaking = threading.Event()
        self._tts_engine = None
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
        logger.info("[VoicePresence] Worker thread started")

    def stop(self) -> None:
        """Signal the worker to exit and join the thread."""
        if not self._started:
            return
        self._stop_event.set()
        # Poison pill to unblock the queue.get()
        self._speech_queue.put(None)
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=5.0)
        # Drain remaining items
        while not self._speech_queue.empty():
            try:
                self._speech_queue.get_nowait()
            except queue.Empty:
                break
        self._started = False
        self._tts_engine = None
        logger.info("[VoicePresence] Worker thread stopped")

    def speak(self, text: str, emotion: Optional[str] = None, block: bool = False) -> None:
        """Queue text for speech through server speakers.

        Args:
            text: Text to speak.
            emotion: Optional emotion name for voice parameter adaptation.
            block: If True, wait for this utterance to finish before returning.
        """
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
        """Generate WAV bytes for the given text. Runs on the worker thread.

        Args:
            text: Text to synthesize.
            emotion: Optional emotion name for voice parameter adaptation.

        Returns:
            WAV file bytes.
        """
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
        """Synthesize text to a numpy array via the worker thread.

        Returns:
            (np.ndarray, sample_rate) or (None, 0) on failure.
        """
        import wave as _wave
        import numpy as _np

        wav_bytes = self.synthesize_wav(text)
        if not wav_bytes:
            return None, 0

        try:
            import io as _io
            with _wave.open(_io.BytesIO(wav_bytes), 'rb') as wf:
                sr = wf.getframerate()
                frames = wf.readframes(wf.getnframes())
                audio = _np.frombuffer(frames, dtype=_np.int16).astype(_np.float32) / 32768.0
                return audio, sr
        except Exception:
            return None, 0

    def is_speaking(self) -> bool:
        """Check if currently speaking."""
        return self._speaking.is_set()

    def get_status(self) -> dict:
        """Status dict for API responses."""
        return {
            "available": self._started and self._tts_engine is not None,
            "engine": "pyttsx3",
            "enabled": self._enabled,
            "speaking": self.is_speaking(),
            "sesame_loaded": False,
        }

    # ------------------------------------------------------------------
    # Worker thread
    # ------------------------------------------------------------------

    def _worker(self) -> None:
        """Dedicated worker thread — initializes pyttsx3 here (COM constraint)."""
        try:
            import pyttsx3

            self._tts_engine = pyttsx3.init()
            self._tts_engine.setProperty("rate", 175)
            self._tts_engine.setProperty("volume", 0.9)
            logger.info("[VoicePresence] pyttsx3 engine initialized on worker thread")
        except Exception as e:
            logger.error(f"[VoicePresence] Failed to init pyttsx3: {e}")
            self._tts_engine = None
            return

        while not self._stop_event.is_set():
            try:
                item = self._speech_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if item is None:
                # Poison pill — exit
                break

            text, emotion, result_holder = item
            self._apply_emotion_params(emotion)

            try:
                if result_holder is not None and not result_holder.get("_block_only"):
                    # WAV synthesis mode
                    fd, path = tempfile.mkstemp(suffix=".wav")
                    os.close(fd)
                    try:
                        self._tts_engine.save_to_file(text, path)
                        self._tts_engine.runAndWait()
                        with open(path, "rb") as f:
                            result_holder["wav_bytes"] = f.read()
                    finally:
                        try:
                            os.unlink(path)
                        except OSError:
                            pass
                    result_holder["done"].set()
                elif result_holder is not None and result_holder.get("_block_only"):
                    # Blocking speak mode
                    self._speaking.set()
                    self._tts_engine.say(text)
                    self._tts_engine.runAndWait()
                    self._speaking.clear()
                    result_holder["done"].set()
                else:
                    # Non-blocking speak through speakers
                    self._speaking.set()
                    self._tts_engine.say(text)
                    self._tts_engine.runAndWait()
                    self._speaking.clear()
            except Exception as e:
                logger.error(f"[VoicePresence] Speech error: {e}")
                self._speaking.clear()
                if result_holder is not None:
                    result_holder["done"].set()

        logger.info("[VoicePresence] Worker thread exiting")

    def _apply_emotion_params(self, emotion: Optional[str]) -> None:
        """Apply emotion-based voice parameters from evoemo_prompts.VOICE_PARAMS."""
        if self._tts_engine is None:
            return
        try:
            from aura.tools.evoemo_prompts import VOICE_PARAMS

            params = VOICE_PARAMS.get(emotion or "calm", VOICE_PARAMS["calm"])
            base_rate = 175
            self._tts_engine.setProperty("rate", int(base_rate * params["rate"]))
            self._tts_engine.setProperty("volume", min(1.0, 0.9 * params["volume"]))
        except Exception as e:
            logger.debug(f"[VoicePresence] Could not apply emotion params: {e}")


def get_voice_presence(enabled: bool = True) -> VoicePresenceService:
    """Get or create the singleton VoicePresenceService instance."""
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = VoicePresenceService(enabled=enabled)
    return _instance
