"""Unit tests for VoicePresenceService."""

import queue
import threading
import time
import unittest
from unittest.mock import MagicMock, patch, mock_open

# Reset the singleton before each test module run
import aura.services.voice_presence as vp_module


class TestVoicePresenceService(unittest.TestCase):
    """Tests for VoicePresenceService with mocked pyttsx3."""

    def setUp(self):
        """Reset singleton before each test."""
        vp_module._instance = None

    def tearDown(self):
        """Stop service if running."""
        if vp_module._instance is not None:
            try:
                vp_module._instance.stop()
            except Exception:
                pass
            vp_module._instance = None

    def test_singleton(self):
        """get_voice_presence() returns the same instance."""
        a = vp_module.get_voice_presence()
        b = vp_module.get_voice_presence()
        self.assertIs(a, b)

    @patch("pyttsx3.init")
    def test_start_stop(self, mock_init):
        """Worker thread starts and stops cleanly."""
        mock_engine = MagicMock()
        mock_init.return_value = mock_engine

        svc = vp_module.get_voice_presence()
        svc.start()
        self.assertTrue(svc._started)
        self.assertTrue(svc._worker_thread.is_alive())

        svc.stop()
        self.assertFalse(svc._started)
        self.assertFalse(svc._worker_thread.is_alive())

    @patch("pyttsx3.init")
    def test_speak_queues(self, mock_init):
        """speak() adds items to the queue."""
        mock_engine = MagicMock()
        mock_init.return_value = mock_engine

        svc = vp_module.get_voice_presence()
        svc.start()
        # Give worker time to start
        time.sleep(0.2)

        svc.speak("Hello world")
        # Wait for worker to process
        time.sleep(0.5)

        mock_engine.say.assert_called_with("Hello world")
        mock_engine.runAndWait.assert_called()
        svc.stop()

    def test_enabled_toggle(self):
        """set_enabled(False) prevents speech."""
        svc = vp_module.get_voice_presence()
        svc.set_enabled(False)
        self.assertFalse(svc._enabled)
        svc.set_enabled(True)
        self.assertTrue(svc._enabled)

    @patch("pyttsx3.init")
    def test_emotion_params(self, mock_init):
        """_apply_emotion_params sets rate and volume based on emotion."""
        mock_engine = MagicMock()
        mock_init.return_value = mock_engine

        svc = vp_module.get_voice_presence()
        svc.start()
        time.sleep(0.2)

        svc._tts_engine = mock_engine
        svc._apply_emotion_params("stressed")

        # stressed: rate=0.85, volume=0.9
        # base_rate=175 * 0.85 = 148.75 -> 148
        mock_engine.setProperty.assert_any_call("rate", 148)
        # volume: min(1.0, 0.9 * 0.9) = 0.81
        mock_engine.setProperty.assert_any_call("volume", 0.81)
        svc.stop()

    @patch("pyttsx3.init")
    def test_synthesize_wav(self, mock_init):
        """synthesize_wav returns bytes."""
        mock_engine = MagicMock()
        mock_init.return_value = mock_engine

        # Make save_to_file write a fake RIFF header
        def fake_save(text, path):
            with open(path, "wb") as f:
                f.write(b"RIFF" + b"\x00" * 40)

        mock_engine.save_to_file.side_effect = fake_save

        svc = vp_module.get_voice_presence()
        svc.start()
        time.sleep(0.2)

        wav_bytes = svc.synthesize_wav("Test synthesis")
        self.assertTrue(wav_bytes.startswith(b"RIFF"))
        self.assertTrue(len(wav_bytes) > 0)
        svc.stop()

    @patch("pyttsx3.init")
    def test_get_status(self, mock_init):
        """get_status returns correct dict shape."""
        mock_engine = MagicMock()
        mock_init.return_value = mock_engine

        svc = vp_module.get_voice_presence()
        svc.start()
        time.sleep(0.2)

        status = svc.get_status()
        self.assertIn("available", status)
        self.assertIn("engine", status)
        self.assertIn("enabled", status)
        self.assertIn("speaking", status)
        self.assertEqual(status["engine"], "pyttsx3")
        self.assertTrue(status["available"])
        self.assertTrue(status["enabled"])
        self.assertFalse(status["speaking"])
        svc.stop()

    @patch("pyttsx3.init")
    def test_speak_while_disabled(self, mock_init):
        """speak() is a no-op when disabled."""
        mock_engine = MagicMock()
        mock_init.return_value = mock_engine

        svc = vp_module.get_voice_presence()
        svc.start()
        time.sleep(0.2)

        svc.set_enabled(False)
        svc.speak("Should not speak")
        time.sleep(0.3)

        mock_engine.say.assert_not_called()
        svc.stop()

    @patch("pyttsx3.init")
    def test_block_mode(self, mock_init):
        """speak(block=True) waits for completion."""
        mock_engine = MagicMock()
        mock_init.return_value = mock_engine

        svc = vp_module.get_voice_presence()
        svc.start()
        time.sleep(0.2)

        # block=True should return only after speech completes
        start = time.time()
        svc.speak("Blocking test", block=True)
        elapsed = time.time() - start

        mock_engine.say.assert_called_with("Blocking test")
        mock_engine.runAndWait.assert_called()
        svc.stop()

    @patch("pyttsx3.init")
    def test_stop_clears_queue(self, mock_init):
        """Pending items are discarded on stop."""
        mock_engine = MagicMock()
        mock_init.return_value = mock_engine

        # Make runAndWait slow so items pile up
        mock_engine.runAndWait.side_effect = lambda: time.sleep(0.3)

        svc = vp_module.get_voice_presence()
        svc.start()
        time.sleep(0.2)

        # Queue several items
        for i in range(5):
            svc.speak(f"Message {i}")

        svc.stop()
        self.assertTrue(svc._speech_queue.empty())

    @patch("pyttsx3.init")
    def test_is_speaking(self, mock_init):
        """is_speaking() reflects the _speaking event."""
        mock_engine = MagicMock()
        mock_init.return_value = mock_engine

        svc = vp_module.get_voice_presence()
        self.assertFalse(svc.is_speaking())

        svc._speaking.set()
        self.assertTrue(svc.is_speaking())

        svc._speaking.clear()
        self.assertFalse(svc.is_speaking())

    def test_speak_before_start(self):
        """speak() before start() is a no-op (not started)."""
        svc = vp_module.get_voice_presence()
        # Should not raise
        svc.speak("No crash please")


if __name__ == "__main__":
    unittest.main()
