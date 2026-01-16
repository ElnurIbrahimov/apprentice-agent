"""Voice interface tool for speech-to-text and text-to-speech."""

import io
import os
import queue
import tempfile
import threading
import wave
from pathlib import Path
from typing import Optional, Callable

import numpy as np
import sounddevice as sd

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


class VoiceTool:
    """Voice interface with speech-to-text (Whisper) and text-to-speech (pyttsx3)."""

    def __init__(self, whisper_model: str = "base"):
        """Initialize voice tool.

        Args:
            whisper_model: Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
        """
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
            print(f"Loading Whisper model '{self._whisper_model_name}'...")
            import whisper
            self._whisper_model = whisper.load_model(self._whisper_model_name)
            print("Whisper model loaded.")
        return self._whisper_model

    def _get_tts_engine(self):
        """Get or create TTS engine."""
        if self._tts_engine is None:
            import pyttsx3
            self._tts_engine = pyttsx3.init()
            # Configure voice settings
            self._tts_engine.setProperty('rate', 175)  # Speed
            self._tts_engine.setProperty('volume', 0.9)  # Volume (0.0 to 1.0)
        return self._tts_engine

    def listen(
        self,
        duration: float = 5.0,
        silence_threshold: float = 0.01,
        silence_duration: float = 1.5,
        max_duration: float = 30.0,
        on_listening: Optional[Callable] = None
    ) -> dict:
        """Record audio from microphone and transcribe using Whisper.

        Args:
            duration: Fixed recording duration in seconds (if not using silence detection)
            silence_threshold: RMS threshold below which audio is considered silence
            silence_duration: Seconds of silence to stop recording
            max_duration: Maximum recording duration
            on_listening: Callback when recording starts

        Returns:
            dict with 'success', 'text', and optional 'error'
        """
        try:
            model = self._load_whisper()

            # Notify that we're listening
            if on_listening:
                on_listening()

            print("Listening... (speak now)")

            # Record with silence detection
            audio_data = self._record_with_silence_detection(
                silence_threshold=silence_threshold,
                silence_duration=silence_duration,
                max_duration=max_duration
            )

            if audio_data is None or len(audio_data) < self._sample_rate * 0.5:
                return {
                    "success": False,
                    "error": "No audio recorded or too short",
                    "text": ""
                }

            print("Processing speech...")

            # Pass audio data directly to Whisper (avoids ffmpeg dependency)
            # Whisper expects float32 audio at 16kHz, which is what we have
            result = model.transcribe(
                audio_data,  # Pass numpy array directly
                language="en",
                fp16=False  # Use FP32 for CPU compatibility
            )
            text = result["text"].strip()

            return {
                "success": True,
                "text": text,
                "language": result.get("language", "en")
            }

        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e),
                "text": ""
            }

    def _record_with_silence_detection(
        self,
        silence_threshold: float = 0.01,
        silence_duration: float = 1.5,
        max_duration: float = 30.0
    ) -> Optional[np.ndarray]:
        """Record audio with automatic stop on silence.

        Args:
            silence_threshold: RMS threshold for silence detection
            silence_duration: Seconds of continuous silence to trigger stop
            max_duration: Maximum recording duration

        Returns:
            numpy array of audio data or None
        """
        audio_chunks = []
        silence_samples = 0
        silence_samples_threshold = int(silence_duration * self._sample_rate)
        max_samples = int(max_duration * self._sample_rate)
        total_samples = 0
        has_speech = False

        def audio_callback(indata, frames, time, status):
            nonlocal silence_samples, total_samples, has_speech

            if status:
                print(f"Audio status: {status}")

            audio_chunks.append(indata.copy())
            total_samples += frames

            # Calculate RMS for silence detection
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
                blocksize=int(self._sample_rate * 0.1)  # 100ms blocks
            ):
                # Wait for speech to start or timeout
                while not has_speech and total_samples < max_samples:
                    sd.sleep(100)

                if not has_speech:
                    print("No speech detected")
                    return None

                # Continue recording until silence or max duration
                while (silence_samples < silence_samples_threshold and
                       total_samples < max_samples):
                    sd.sleep(100)

            if audio_chunks:
                return np.concatenate(audio_chunks, axis=0).flatten()
            return None

        except Exception as e:
            print(f"Recording error: {e}")
            return None

    def _save_wav(self, file_obj, audio_data: np.ndarray):
        """Save audio data to WAV file."""
        # Convert float32 to int16
        audio_int16 = (audio_data * 32767).astype(np.int16)

        with wave.open(file_obj, 'wb') as wav:
            wav.setnchannels(self._channels)
            wav.setsampwidth(2)  # 16-bit
            wav.setframerate(self._sample_rate)
            wav.writeframes(audio_int16.tobytes())

    def speak(self, text: str, block: bool = True) -> dict:
        """Convert text to speech using pyttsx3.

        Args:
            text: Text to speak
            block: If True, wait for speech to complete

        Returns:
            dict with 'success' and optional 'error'
        """
        if not text or not text.strip():
            return {"success": False, "error": "No text provided"}

        try:
            engine = self._get_tts_engine()

            if block:
                engine.say(text)
                engine.runAndWait()
            else:
                # Run in background thread
                def speak_async():
                    engine.say(text)
                    engine.runAndWait()

                thread = threading.Thread(target=speak_async, daemon=True)
                thread.start()

            return {"success": True, "text": text}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def set_voice(self, voice_id: Optional[str] = None, rate: Optional[int] = None, volume: Optional[float] = None):
        """Configure TTS voice settings.

        Args:
            voice_id: Voice identifier (use get_voices() to list available)
            rate: Speech rate in words per minute
            volume: Volume level (0.0 to 1.0)
        """
        engine = self._get_tts_engine()

        if voice_id is not None:
            engine.setProperty('voice', voice_id)
        if rate is not None:
            engine.setProperty('rate', rate)
        if volume is not None:
            engine.setProperty('volume', max(0.0, min(1.0, volume)))

    def get_voices(self) -> list:
        """Get list of available TTS voices.

        Returns:
            List of dicts with voice info
        """
        engine = self._get_tts_engine()
        voices = engine.getProperty('voices')

        return [
            {
                "id": voice.id,
                "name": voice.name,
                "languages": getattr(voice, 'languages', []),
                "gender": getattr(voice, 'gender', None)
            }
            for voice in voices
        ]

    def list_audio_devices(self) -> dict:
        """List available audio input devices.

        Returns:
            dict with 'success' and 'devices' list
        """
        try:
            devices = sd.query_devices()
            input_devices = []

            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:
                    input_devices.append({
                        "id": i,
                        "name": device['name'],
                        "channels": device['max_input_channels'],
                        "sample_rate": device['default_samplerate']
                    })

            return {
                "success": True,
                "devices": input_devices,
                "default": sd.default.device[0]
            }
        except Exception as e:
            return {"success": False, "error": str(e), "devices": []}


class VoiceConversation:
    """Voice-based conversation interface for the agent."""

    def __init__(self, agent, whisper_model: str = "base"):
        """Initialize voice conversation.

        Args:
            agent: ApprenticeAgent instance
            whisper_model: Whisper model size
        """
        self.agent = agent
        self.voice = VoiceTool(whisper_model=whisper_model)
        self._running = False

    def start(self):
        """Start voice conversation loop."""
        self._running = True

        print("\n" + "="*60)
        print("Voice Mode - Speak to interact with the agent")
        print("Say 'exit', 'quit', or 'goodbye' to end")
        print("="*60 + "\n")

        # Greet the user
        self.voice.speak("Hello! I'm your apprentice agent. How can I help you?")

        while self._running:
            try:
                # Listen for input
                result = self.voice.listen(
                    silence_threshold=0.01,
                    silence_duration=1.5,
                    max_duration=30.0
                )

                if not result["success"]:
                    if "No audio" not in result.get("error", ""):
                        print(f"Listen error: {result.get('error')}")
                    continue

                user_text = result["text"]
                if not user_text:
                    continue

                print(f"\nYou: {user_text}")

                # Check for exit commands
                exit_phrases = ['exit', 'quit', 'goodbye', 'bye', 'stop listening']
                if any(phrase in user_text.lower() for phrase in exit_phrases):
                    self.voice.speak("Goodbye! Have a great day.")
                    self._running = False
                    break

                # Get agent response
                response = self.agent.chat(user_text)
                print(f"\nAgent: {response}")

                # Speak the response
                self.voice.speak(response)

            except KeyboardInterrupt:
                print("\nVoice mode interrupted.")
                self._running = False
                break
            except Exception as e:
                print(f"Error: {e}")
                self.voice.speak("I encountered an error. Please try again.")

    def stop(self):
        """Stop voice conversation."""
        self._running = False
