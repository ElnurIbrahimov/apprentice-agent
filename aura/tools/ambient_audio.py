"""Ambient Audio Monitor — continuous mic transcription like a local Limitless Pendant.

Captures mic audio, detects speech with VAD (webrtcvad), transcribes with local
Whisper (faster-whisper), stores transcripts in ChromaDB for semantic search.

"What did I say at 2pm?" → AURA retrieves it from local memory.
100% local — no audio ever leaves the machine.

Setup:
    pip install faster-whisper sounddevice soundfile webrtcvad-wheels

Config (.env):
    AMBIENT_AUDIO_MODEL   — Whisper model: tiny | base | small (default: base)
    AMBIENT_AUDIO_ENABLED — Auto-start on boot: true | false (default: false)
"""

import logging
import os
import queue
import threading
import io
import wave
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any

logger = logging.getLogger(__name__)

WHISPER_MODEL = os.getenv("AMBIENT_AUDIO_MODEL", "base")
AUTO_START = os.getenv("AMBIENT_AUDIO_ENABLED", "false").lower() == "true"
SAMPLE_RATE = 16000
FRAME_DURATION_MS = 30  # webrtcvad supports 10, 20, 30ms
FRAME_BYTES = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000) * 2  # int16 = 2 bytes/sample
SILENCE_TIMEOUT = 1.5   # seconds of silence to end a speech segment
MIN_SPEECH_SEC = 0.4    # ignore segments shorter than this

_monitor_lock = threading.Lock()
_monitor: Optional["AmbientAudioMonitor"] = None


class AmbientAudioMonitor:
    """Background audio capture + VAD + Whisper transcription."""

    def __init__(self):
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._whisper = None
        self._vad = None
        self._collection = None
        self._stats: Dict[str, Any] = {
            "transcripts": 0,
            "errors": 0,
            "started_at": None,
        }
        self._in_memory: List[Dict] = []  # fallback if ChromaDB unavailable

    # ------------------------------------------------------------------ #
    # Lazy init
    # ------------------------------------------------------------------ #

    def _init_deps(self) -> Optional[str]:
        """Load heavy deps on first use. Returns error string or None."""
        try:
            import webrtcvad
            self._vad = webrtcvad.Vad(2)  # aggressiveness 0-3
        except ImportError:
            return "webrtcvad not installed. Run: pip install webrtcvad-wheels"

        try:
            import sounddevice  # just verify it's importable
        except ImportError:
            return "sounddevice not installed. Run: pip install sounddevice"

        try:
            from faster_whisper import WhisperModel
            logger.info(f"[AmbientAudio] Loading Whisper '{WHISPER_MODEL}' ...")
            self._whisper = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
            logger.info("[AmbientAudio] Whisper ready")
        except ImportError:
            return "faster-whisper not installed. Run: pip install faster-whisper"
        except Exception as e:
            return f"Whisper load failed: {e}"

        try:
            import chromadb
            client = chromadb.PersistentClient(path=os.getenv("CHROMADB_PATH", "./data/chromadb"))
            self._collection = client.get_or_create_collection("ambient_audio")
        except Exception as e:
            logger.warning(f"[AmbientAudio] ChromaDB unavailable ({e}) — using in-memory storage")

        return None

    # ------------------------------------------------------------------ #
    # Start / stop
    # ------------------------------------------------------------------ #

    def start(self) -> Dict:
        with _monitor_lock:
            if self._running:
                return {"success": False, "error": "Already running"}
            err = self._init_deps()
            if err:
                return {"success": False, "error": err}
            self._running = True
            self._stats["started_at"] = datetime.now().isoformat()
            self._thread = threading.Thread(
                target=self._capture_loop, daemon=True, name="ambient-audio"
            )
            self._thread.start()
            return {"success": True, "status": "monitoring", "model": WHISPER_MODEL}

    def stop(self) -> Dict:
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
        self._stats["started_at"] = None
        return {"success": True, "status": "stopped", "transcripts": self._stats["transcripts"]}

    # ------------------------------------------------------------------ #
    # Capture loop
    # ------------------------------------------------------------------ #

    def _capture_loop(self):
        import sounddevice as sd
        import numpy as np

        raw_q: queue.Queue = queue.Queue()

        def _callback(indata, frames, time_info, status):
            if status:
                logger.debug(f"[AmbientAudio] sounddevice status: {status}")
            raw_q.put(bytes(indata))

        # sounddevice gives float32 by default — we need int16 for webrtcvad
        CHUNK_FRAMES = FRAME_BYTES // 2  # samples per frame

        try:
            with sd.RawInputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="int16",
                blocksize=CHUNK_FRAMES,
                callback=_callback,
            ):
                logger.info("[AmbientAudio] Mic stream open")
                speech_frames: List[bytes] = []
                silence_count = 0
                in_speech = False
                frames_for_silence = int(SILENCE_TIMEOUT * 1000 / FRAME_DURATION_MS)

                while self._running:
                    try:
                        frame = raw_q.get(timeout=0.1)
                    except queue.Empty:
                        continue

                    # Pad/trim to exact VAD frame size
                    if len(frame) < FRAME_BYTES:
                        frame = frame + b"\x00" * (FRAME_BYTES - len(frame))
                    frame = frame[:FRAME_BYTES]

                    try:
                        is_speech = self._vad.is_speech(frame, SAMPLE_RATE)
                    except Exception:
                        is_speech = False

                    if is_speech:
                        speech_frames.append(frame)
                        silence_count = 0
                        in_speech = True
                    elif in_speech:
                        speech_frames.append(frame)
                        silence_count += 1
                        if silence_count >= frames_for_silence:
                            duration = len(speech_frames) * FRAME_DURATION_MS / 1000
                            if duration >= MIN_SPEECH_SEC:
                                self._transcribe_segment(speech_frames)
                            speech_frames = []
                            silence_count = 0
                            in_speech = False

        except Exception as e:
            logger.error(f"[AmbientAudio] Capture loop error: {e}")
            self._stats["errors"] += 1
            self._running = False

    def _transcribe_segment(self, frames: List[bytes]):
        """Transcribe and store a speech segment."""
        if self._whisper is None:
            logger.warning("[AmbientAudio] Whisper not initialized, skipping transcription")
            return
        try:
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(b"".join(frames))
            buf.seek(0)

            segments, _ = self._whisper.transcribe(buf, beam_size=1, language="en")
            text = " ".join(seg.text for seg in segments).strip()

            if not text or len(text) < 3:
                return

            ts = datetime.now()
            doc_id = f"audio_{ts.strftime('%Y%m%d_%H%M%S_%f')}"
            meta = {"timestamp": ts.isoformat(), "source": "ambient_audio", "date": ts.strftime("%Y-%m-%d")}
            logger.info(f"[AmbientAudio] '{text[:80]}'")

            if self._collection:
                self._collection.add(documents=[text], ids=[doc_id], metadatas=[meta])
            else:
                self._in_memory.append({"text": text, **meta})

            self._stats["transcripts"] += 1

        except Exception as e:
            logger.error(f"[AmbientAudio] Transcribe error: {e}")
            self._stats["errors"] += 1

    # ------------------------------------------------------------------ #
    # Query
    # ------------------------------------------------------------------ #

    def recall(self, query: str, limit: int = 10) -> Dict:
        """Semantic search over ambient audio memory."""
        if self._collection:
            try:
                results = self._collection.query(query_texts=[query], n_results=min(limit, 20))
                docs = results.get("documents", [[]])[0]
                metas = results.get("metadatas", [[]])[0]
                return {
                    "success": True,
                    "query": query,
                    "count": len(docs),
                    "results": [{"text": d, "timestamp": m.get("timestamp", "")} for d, m in zip(docs, metas)],
                }
            except Exception as e:
                return {"success": False, "error": str(e)}
        # Fallback: simple substring search in memory
        q = query.lower()
        hits = [e for e in self._in_memory if q in e["text"].lower()][-limit:]
        return {"success": True, "query": query, "count": len(hits), "results": hits}

    def get_today_log(self) -> Dict:
        """Get all transcripts from today in chronological order."""
        today = datetime.now().strftime("%Y-%m-%d")
        if self._collection:
            try:
                all_docs = self._collection.get(where={"date": today})
                docs = all_docs.get("documents", [])
                metas = all_docs.get("metadatas", [])
                entries = sorted(
                    [{"text": d, "time": m.get("timestamp", "")[11:16]} for d, m in zip(docs, metas)],
                    key=lambda x: x["time"],
                )
                return {"success": True, "date": today, "count": len(entries), "entries": entries}
            except Exception as e:
                return {"success": False, "error": str(e)}
        entries = [{"text": e["text"], "time": e["timestamp"][11:16]} for e in self._in_memory if e.get("date") == today]
        return {"success": True, "date": today, "count": len(entries), "entries": entries}

    def get_status(self) -> Dict:
        return {
            "running": self._running,
            "model": WHISPER_MODEL,
            "transcripts": self._stats["transcripts"],
            "errors": self._stats["errors"],
            "started_at": self._stats["started_at"],
            "storage": "chromadb" if self._collection else "memory",
        }


def _get_monitor() -> AmbientAudioMonitor:
    global _monitor
    if _monitor is None:
        _monitor = AmbientAudioMonitor()
    return _monitor


class AmbientAudioTool:
    """Continuous mic transcription — 'What did I say at 2pm?' → real answer from local memory."""

    name = "ambient_audio"
    description = "Record and recall ambient audio locally — Whisper transcription, ChromaDB memory. Like Limitless Pendant but offline."

    def __init__(self):
        self._mon = _get_monitor()
        if AUTO_START:
            self._mon.start()

    def execute(self, action: str, **kwargs) -> Dict:
        a = action.lower().strip()
        if "start" in a or "begin" in a or "monitor" in a or "listen" in a:
            return self._mon.start()
        if "stop" in a or "pause" in a or "end" in a:
            return self._mon.stop()
        if "status" in a or "check" in a:
            return {"success": True, **self._mon.get_status()}
        if "today" in a or "log" in a or "transcript" in a:
            return self._mon.get_today_log()
        if "recall" in a or "search" in a or "find" in a or "what" in a:
            return self._mon.recall(kwargs.get("query") or action, kwargs.get("limit", 10))
        return {"success": True, **self._mon.get_status()}
