"""Voice interface tool for speech-to-text and text-to-speech with barge-in detection."""

import io
import logging
import os
import queue
import tempfile
import threading
import time
import wave
from enum import Enum
from pathlib import Path
from typing import Optional, Callable, Tuple

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Barge-in constants
# ---------------------------------------------------------------------------
BARGE_IN_ENERGY_THRESHOLD = 0.03    # 3x silence threshold (0.01)
BARGE_IN_CONSECUTIVE_FRAMES = 3     # Confirm after 3 consecutive speech frames
BARGE_IN_FRAME_DURATION_MS = 100    # Analyze 100ms frames
BARGE_IN_ZCR_MIN = 0.02            # Zero-crossing rate min for speech
BARGE_IN_ZCR_MAX = 0.30            # Zero-crossing rate max for speech

# Set up ffmpeg path for Whisper (use bundled ffmpeg from imageio-ffmpeg)
def _setup_ffmpeg():
    """Add bundled ffmpeg to PATH if available."""
    try:
        import imageio_ffmpeg
        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        ffmpeg_dir = str(Path(ffmpeg_path).parent)
        if ffmpeg_dir not in os.environ.get('PATH', ''):
            os.environ['PATH'] = ffmpeg_dir + os.pathsep + os.environ.get('PATH', '')
    except ImportError:
        pass  # imageio-ffmpeg not installed, hope ffmpeg is in PATH

_setup_ffmpeg()


# ---------------------------------------------------------------------------
# Conversation state
# ---------------------------------------------------------------------------
class ConversationState(Enum):
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    INTERRUPTED = "interrupted"


# ---------------------------------------------------------------------------
# Barge-in Detector
# ---------------------------------------------------------------------------
class BargeInDetector:
    """Monitors microphone during TTS playback to detect user speech interruptions.

    Opens a sounddevice InputStream, analyzes 100ms frames for RMS energy +
    zero-crossing rate (ZCR), and confirms barge-in after N consecutive
    speech-like frames. Buffers captured audio for later Whisper transcription.
    """

    def __init__(self, sample_rate: int = 16000):
        self._sample_rate = sample_rate
        self._frame_size = int(sample_rate * BARGE_IN_FRAME_DURATION_MS / 1000)
        self._detected = threading.Event()
        self._stop = threading.Event()
        self._consecutive_speech = 0
        self._audio_buffer: list = []
        self._stream: Optional[sd.InputStream] = None
        self._lock = threading.Lock()

    def start(self) -> None:
        """Start monitoring the microphone for barge-in."""
        self._detected.clear()
        self._stop.clear()
        self._consecutive_speech = 0
        self._audio_buffer = []

        try:
            self._stream = sd.InputStream(
                samplerate=self._sample_rate,
                channels=1,
                dtype=np.float32,
                callback=self._audio_callback,
                blocksize=self._frame_size,
            )
            self._stream.start()
        except Exception as e:
            logger.warning(f"[BargeIn] Failed to start mic monitor: {e}")

    def stop(self) -> None:
        """Stop monitoring."""
        self._stop.set()
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    @property
    def detected(self) -> bool:
        return self._detected.is_set()

    def wait(self, timeout: float = 0.05) -> bool:
        """Block until barge-in detected or timeout. Returns True if detected."""
        return self._detected.wait(timeout=timeout)

    def get_buffered_audio(self) -> Optional[np.ndarray]:
        """Return audio captured after barge-in detection."""
        with self._lock:
            if self._audio_buffer:
                return np.concatenate(self._audio_buffer, axis=0).flatten()
        return None

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        if self._stop.is_set():
            return

        frame = indata.flatten()

        # Compute RMS energy
        rms = np.sqrt(np.mean(frame ** 2))

        # Compute zero-crossing rate
        signs = np.sign(frame)
        zero_crossings = np.sum(np.abs(np.diff(signs)) > 0)
        zcr = zero_crossings / len(frame) if len(frame) > 0 else 0

        is_speech = (
            rms > BARGE_IN_ENERGY_THRESHOLD
            and BARGE_IN_ZCR_MIN <= zcr <= BARGE_IN_ZCR_MAX
        )

        if is_speech:
            self._consecutive_speech += 1
        else:
            self._consecutive_speech = 0

        # Buffer audio once we start detecting potential speech
        if self._consecutive_speech >= 1:
            with self._lock:
                self._audio_buffer.append(indata.copy())

        # Confirm barge-in after consecutive frames
        if self._consecutive_speech >= BARGE_IN_CONSECUTIVE_FRAMES:
            self._detected.set()


# ---------------------------------------------------------------------------
# Interruptible Audio Player
# ---------------------------------------------------------------------------
class InterruptiblePlayer:
    """Non-blocking audio player that can be stopped immediately.

    Uses sd.play() (non-blocking) + sd.stop() for immediate cancellation.
    Tracks playback position.
    """

    def __init__(self) -> None:
        self._playing = threading.Event()
        self._audio: Optional[np.ndarray] = None
        self._sample_rate: int = 22050
        self._start_time: float = 0
        self._total_duration: float = 0

    def play(self, audio: np.ndarray, sample_rate: int) -> None:
        """Start non-blocking playback."""
        self._audio = audio
        self._sample_rate = sample_rate
        self._total_duration = len(audio) / sample_rate
        self._start_time = time.time()
        self._playing.set()
        sd.play(audio, samplerate=sample_rate)

    def stop(self) -> float:
        """Stop playback immediately. Returns seconds of audio that was heard."""
        sd.stop()
        elapsed = time.time() - self._start_time
        self._playing.clear()
        return min(elapsed, self._total_duration)

    def is_playing(self) -> bool:
        """Check if audio is currently playing."""
        if not self._playing.is_set():
            return False
        elapsed = time.time() - self._start_time
        if elapsed >= self._total_duration:
            self._playing.clear()
            return False
        return True

    def wait(self, timeout: Optional[float] = None) -> None:
        """Wait for playback to complete."""
        if self._audio is None:
            return
        remaining = self._total_duration - (time.time() - self._start_time)
        if remaining > 0:
            sd.sleep(int(min(remaining, timeout or remaining) * 1000))
        self._playing.clear()


# ---------------------------------------------------------------------------
# VoiceTool (existing, unchanged except TTS now also supports WAV synthesis)
# ---------------------------------------------------------------------------
class VoiceTool:
    """Voice interface with speech-to-text (Whisper) and text-to-speech (pyttsx3)."""

    def __init__(self, whisper_model: str = "base"):
        self._whisper_model_name = whisper_model
        self._whisper_model = None
        self._tts_engine = None
        self._is_recording = False
        self._audio_queue = queue.Queue()
        self._sample_rate = 16000  # Whisper expects 16kHz
        self._channels = 1

    def _load_whisper(self):
        """Lazy load Whisper model."""
        if self._whisper_model is None:
            logger.debug(f"Loading Whisper model '{self._whisper_model_name}'...")
            import whisper
            self._whisper_model = whisper.load_model(self._whisper_model_name)
            logger.debug("Whisper model loaded.")
        return self._whisper_model

    def is_whisper_loaded(self) -> bool:
        return self._whisper_model is not None

    def unload_whisper(self) -> dict:
        try:
            if self._whisper_model is None:
                return {"success": True, "message": "Whisper model not loaded, nothing to unload"}
            del self._whisper_model
            self._whisper_model = None
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except ImportError:
                pass
            logger.debug("Whisper model unloaded")
            return {"success": True, "message": "Whisper model unloaded successfully"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _get_tts_engine(self):
        if self._tts_engine is None:
            import pyttsx3
            self._tts_engine = pyttsx3.init()
            self._tts_engine.setProperty('rate', 175)
            self._tts_engine.setProperty('volume', 0.9)
        return self._tts_engine

    def listen(
        self,
        duration: float = 5.0,
        silence_threshold: float = 0.01,
        silence_duration: float = 1.5,
        max_duration: float = 30.0,
        on_listening: Optional[Callable] = None
    ) -> dict:
        try:
            model = self._load_whisper()
            if on_listening:
                on_listening()
            logger.debug("Listening... (speak now)")
            audio_data = self._record_with_silence_detection(
                silence_threshold=silence_threshold,
                silence_duration=silence_duration,
                max_duration=max_duration
            )
            if audio_data is None or len(audio_data) < self._sample_rate * 0.5:
                return {"success": False, "error": "No audio recorded or too short", "text": ""}
            logger.debug("Processing speech...")
            result = model.transcribe(audio_data, language="en", fp16=False)
            text = result["text"].strip()
            return {"success": True, "text": text, "language": result.get("language", "en")}
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e), "text": ""}

    def transcribe(self, audio_data: np.ndarray) -> str:
        """Transcribe a numpy audio array via Whisper. Returns text."""
        try:
            model = self._load_whisper()
            result = model.transcribe(audio_data, language="en", fp16=False)
            return result["text"].strip()
        except Exception as e:
            logger.error(f"[Voice] Transcribe error: {e}")
            return ""

    def _record_with_silence_detection(
        self,
        silence_threshold: float = 0.01,
        silence_duration: float = 1.5,
        max_duration: float = 30.0
    ) -> Optional[np.ndarray]:
        audio_chunks = []
        silence_samples = 0
        silence_samples_threshold = int(silence_duration * self._sample_rate)
        max_samples = int(max_duration * self._sample_rate)
        total_samples = 0
        has_speech = False

        def audio_callback(indata, frames, time, status):
            nonlocal silence_samples, total_samples, has_speech
            if status:
                logger.debug(f"Audio status: {status}")
            audio_chunks.append(indata.copy())
            total_samples += frames
            rms = np.sqrt(np.mean(indata**2))
            if rms > silence_threshold:
                has_speech = True
                silence_samples = 0
            else:
                silence_samples += frames

        try:
            with sd.InputStream(
                samplerate=self._sample_rate,
                channels=self._channels,
                dtype=np.float32,
                callback=audio_callback,
                blocksize=int(self._sample_rate * 0.1)
            ):
                while not has_speech and total_samples < max_samples:
                    sd.sleep(100)
                if not has_speech:
                    logger.debug("No speech detected")
                    return None
                while (silence_samples < silence_samples_threshold and
                       total_samples < max_samples):
                    sd.sleep(100)
            if audio_chunks:
                return np.concatenate(audio_chunks, axis=0).flatten()
            return None
        except Exception as e:
            logger.error(f"Recording error: {e}")
            return None

    def _save_wav(self, file_obj, audio_data: np.ndarray):
        audio_int16 = (audio_data * 32767).astype(np.int16)
        with wave.open(file_obj, 'wb') as wav:
            wav.setnchannels(self._channels)
            wav.setsampwidth(2)
            wav.setframerate(self._sample_rate)
            wav.writeframes(audio_int16.tobytes())

    def speak(self, text: str, block: bool = True) -> dict:
        if not text or not text.strip():
            return {"success": False, "error": "No text provided"}
        try:
            engine = self._get_tts_engine()
            if block:
                engine.say(text)
                engine.runAndWait()
            else:
                def speak_async():
                    engine.say(text)
                    engine.runAndWait()
                thread = threading.Thread(target=speak_async, daemon=True)
                thread.start()
            return {"success": True, "text": text}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def set_voice(self, voice_id: Optional[str] = None, rate: Optional[int] = None, volume: Optional[float] = None):
        engine = self._get_tts_engine()
        if voice_id is not None:
            engine.setProperty('voice', voice_id)
        if rate is not None:
            engine.setProperty('rate', rate)
        if volume is not None:
            engine.setProperty('volume', max(0.0, min(1.0, volume)))

    def get_voices(self) -> list:
        engine = self._get_tts_engine()
        voices = engine.getProperty('voices')
        return [
            {"id": voice.id, "name": voice.name,
             "languages": getattr(voice, 'languages', []),
             "gender": getattr(voice, 'gender', None)}
            for voice in voices
        ]

    def list_audio_devices(self) -> dict:
        try:
            devices = sd.query_devices()
            input_devices = []
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:
                    input_devices.append({
                        "id": i, "name": device['name'],
                        "channels": device['max_input_channels'],
                        "sample_rate": device['default_samplerate']
                    })
            return {"success": True, "devices": input_devices, "default": sd.default.device[0]}
        except Exception as e:
            return {"success": False, "error": str(e), "devices": []}


# ---------------------------------------------------------------------------
# VoiceConversation with barge-in support
# ---------------------------------------------------------------------------
class VoiceConversation:
    """Voice-based conversation interface with barge-in detection."""

    MAX_BARGE_IN_DEPTH = 3  # Max recursion for chained interruptions

    def __init__(self, agent, whisper_model: str = "base", enable_barge_in: bool = True):
        self.agent = agent
        self.voice = VoiceTool(whisper_model=whisper_model)
        self._running = False
        self._state = ConversationState.LISTENING
        self._enable_barge_in = enable_barge_in

    def start(self):
        """Start voice conversation loop."""
        self._running = True

        logger.debug("\n" + "=" * 60)
        logger.debug("Voice Mode - Speak to interact with the agent")
        if self._enable_barge_in:
            logger.debug("Barge-in ENABLED: interrupt the agent while it speaks")
        logger.debug("Say 'exit', 'quit', or 'goodbye' to end")
        logger.debug("=" * 60 + "\n")

        # Greet the user
        self._speak_and_handle_barge_in("Hello! I'm AURA. How can I help you?")

        while self._running:
            try:
                self._state = ConversationState.LISTENING

                # Listen for input
                result = self.voice.listen(
                    silence_threshold=0.01,
                    silence_duration=1.5,
                    max_duration=30.0
                )

                if not result["success"]:
                    if "No audio" not in result.get("error", ""):
                        logger.error(f"Listen error: {result.get('error')}")
                    continue

                user_text = result["text"]
                if not user_text:
                    continue

                logger.debug(f"\nYou: {user_text}")

                # Check for exit commands
                exit_phrases = ['exit', 'quit', 'goodbye', 'bye', 'stop listening']
                if any(phrase in user_text.lower() for phrase in exit_phrases):
                    self.voice.speak("Goodbye! Have a great day.")
                    self._running = False
                    break

                # Get agent response
                self._state = ConversationState.THINKING
                response = self.agent.chat(user_text)
                logger.debug(f"\nAgent: {response}")

                # Speak with barge-in handling
                self._speak_and_handle_barge_in(response)

            except KeyboardInterrupt:
                logger.debug("\nVoice mode interrupted.")
                self._running = False
                break
            except Exception as e:
                logger.error(f"Error: {e}")
                self.voice.speak("I encountered an error. Please try again.")

    def stop(self):
        """Stop voice conversation."""
        self._running = False

    def _speak_and_handle_barge_in(self, text: str, depth: int = 0) -> None:
        """Speak text with barge-in handling, with recursion limit."""
        if depth >= self.MAX_BARGE_IN_DEPTH:
            # Too many chained interruptions, just speak blocking
            self.voice.speak(text)
            return

        interruption_text = self._speak_interruptible(text)

        if interruption_text:
            logger.debug(f"\n[Barge-in detected] You: {interruption_text}")

            # Check for exit
            if any(phrase in interruption_text.lower() for phrase in ['exit', 'quit', 'goodbye', 'bye']):
                self.voice.speak("Goodbye!")
                self._running = False
                return

            # Process the interruption as a new query
            self._state = ConversationState.THINKING
            response = self.agent.chat(interruption_text)
            logger.debug(f"\nAgent: {response}")

            # Recursively speak with barge-in
            self._speak_and_handle_barge_in(response, depth + 1)

    def _speak_interruptible(self, text: str) -> Optional[str]:
        """Speak text while monitoring mic for barge-in.

        Returns:
            Transcribed interruption text if barge-in detected, None otherwise.
        """
        if not self._enable_barge_in:
            self.voice.speak(text)
            return None

        self._state = ConversationState.SPEAKING

        # Synthesize audio to numpy array
        audio, sr = self._synthesize_audio(text)
        if audio is None:
            # Fallback to blocking TTS
            self.voice.speak(text)
            return None

        # Start barge-in detector
        detector = BargeInDetector(sample_rate=self.voice._sample_rate)
        detector.start()

        # Start non-blocking playback
        player = InterruptiblePlayer()
        player.play(audio, sr)

        # Poll for barge-in or playback completion
        try:
            while player.is_playing():
                if detector.wait(timeout=0.05):
                    # Barge-in confirmed!
                    self._state = ConversationState.INTERRUPTED
                    heard_seconds = player.stop()
                    logger.info(f"[Voice] Barge-in after {heard_seconds:.1f}s of TTS")

                    # Continue listening for the rest of the user's speech
                    detector.stop()

                    # Get what was captured + continue recording
                    barge_audio = detector.get_buffered_audio()

                    # Record the rest of the user's utterance
                    remaining = self.voice._record_with_silence_detection(
                        silence_threshold=0.01,
                        silence_duration=1.0,
                        max_duration=15.0,
                    )

                    # Combine barge-in buffer + remaining
                    parts = []
                    if barge_audio is not None:
                        parts.append(barge_audio)
                    if remaining is not None:
                        parts.append(remaining)

                    if parts:
                        full_audio = np.concatenate(parts)
                        transcription = self.voice.transcribe(full_audio)
                        if transcription:
                            return transcription

                    return None
        finally:
            detector.stop()

        return None

    def _synthesize_audio(self, text: str) -> Tuple[Optional[np.ndarray], int]:
        """Synthesize text to a numpy audio array.

        Tries VoicePresenceService WAV bridge first, falls back to pyttsx3 file synthesis.

        Returns:
            (audio_array, sample_rate) or (None, 0) on failure.
        """
        # Try VoicePresenceService synthesize_audio_array
        try:
            from aura.services.voice_presence import get_voice_presence
            vp = get_voice_presence()
            audio, sr = vp.synthesize_audio_array(text)
            if audio is not None and len(audio) > 0:
                return audio, sr
        except Exception:
            pass

        # Fallback: use pyttsx3 save_to_file + WAV parse
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty('rate', 175)
            engine.setProperty('volume', 0.9)

            fd, path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            try:
                engine.save_to_file(text, path)
                engine.runAndWait()

                with wave.open(path, 'rb') as wf:
                    sr = wf.getframerate()
                    frames = wf.readframes(wf.getnframes())
                    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                    return audio, sr
            finally:
                try:
                    os.unlink(path)
                except OSError:
                    pass
        except Exception as e:
            logger.warning(f"[Voice] Audio synthesis failed: {e}")
            return None, 0
