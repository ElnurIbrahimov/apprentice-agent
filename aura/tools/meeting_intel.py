"""Meeting Intelligence — record, transcribe, and analyze meetings locally.

Records from mic, transcribes with local Whisper (faster-whisper), extracts
action items and decisions, stores searchable meeting records.

"Last 3 meetings you keep mentioning the auth system — draft a decision doc?"

100% offline — audio never leaves the machine.

Storage: data/meetings/ (JSON records)

Setup:
    pip install faster-whisper sounddevice soundfile

Config (.env):
    AMBIENT_AUDIO_MODEL — Whisper model: tiny | base | small (default: base)
"""

import json
import logging
import os
import queue
import threading
import uuid
import wave
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

MEETINGS_DIR = Path(os.getenv("AURA_DATA_DIR", "data")) / "meetings"
WHISPER_MODEL = os.getenv("AMBIENT_AUDIO_MODEL", "base")
SAMPLE_RATE = 16000
CHUNK_FRAMES = 512

_record_lock = threading.Lock()
_active: Optional["MeetingRecorder"] = None


class MeetingRecorder:
    """Records mic audio for a single meeting session."""

    def __init__(self):
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._frames: List[bytes] = []
        self._raw_q: queue.Queue = queue.Queue()
        self._meeting_id: Optional[str] = None
        self._title: str = ""
        self._start_time: Optional[datetime] = None

    def start(self, title: str = "") -> Dict:
        """Start recording."""
        global _active
        with _record_lock:
            if self._running:
                return {"success": False, "error": "Already recording"}
            # Verify deps
            try:
                import sounddevice  # noqa
            except ImportError:
                return {"success": False, "error": "sounddevice not installed. Run: pip install sounddevice soundfile"}
            try:
                from faster_whisper import WhisperModel  # noqa
            except ImportError:
                return {"success": False, "error": "faster-whisper not installed. Run: pip install faster-whisper"}

            MEETINGS_DIR.mkdir(parents=True, exist_ok=True)
            self._meeting_id = str(uuid.uuid4())[:8]
            self._start_time = datetime.now()
            self._frames = []
            self._title = title or f"Meeting {self._start_time.strftime('%Y-%m-%d %H:%M')}"
            self._running = True
            _active = self

            self._thread = threading.Thread(target=self._record_loop, daemon=True, name="meeting-record")
            self._thread.start()

            return {
                "success": True,
                "meeting_id": self._meeting_id,
                "title": self._title,
                "started_at": self._start_time.isoformat(),
                "tip": "Call stop_and_analyze when done.",
            }

    def stop_and_analyze(self) -> Dict:
        """Stop recording, transcribe, and extract structure."""
        global _active
        with _record_lock:
            if not self._running:
                return {"success": False, "error": "Not recording"}
            self._running = False

        if self._thread:
            self._thread.join(timeout=5)

        duration = int((datetime.now() - self._start_time).total_seconds())
        logger.info(f"[MeetingIntel] Stopped — {duration}s captured, transcribing...")

        if not self._frames:
            return {"success": False, "error": "No audio captured — check microphone"}

        # Save audio file
        audio_path = MEETINGS_DIR / f"{self._meeting_id}_audio.wav"
        try:
            with wave.open(str(audio_path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(b"".join(self._frames))
        except Exception as e:
            logger.warning(f"[MeetingIntel] Audio save failed: {e}")
            audio_path = None

        transcript = self._transcribe(audio_path)
        analysis = self._analyze_transcript(transcript)

        record = {
            "id": self._meeting_id,
            "title": self._title,
            "started_at": self._start_time.isoformat(),
            "duration_sec": duration,
            "transcript": transcript,
            "summary": analysis["summary"],
            "action_items": analysis["action_items"],
            "decisions": analysis["decisions"],
            "key_topics": analysis["key_topics"],
            "audio_path": str(audio_path) if audio_path else None,
        }

        record_path = MEETINGS_DIR / f"{self._meeting_id}.json"
        with open(record_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)

        _active = None
        logger.info(f"[MeetingIntel] Saved: {record_path}")
        return {"success": True, **record}

    def _record_loop(self):
        import sounddevice as sd

        def _cb(indata, frames, time_info, status):
            self._raw_q.put(bytes(indata))

        try:
            with sd.RawInputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="int16",
                blocksize=CHUNK_FRAMES,
                callback=_cb,
            ):
                while self._running:
                    try:
                        chunk = self._raw_q.get(timeout=0.1)
                        self._frames.append(chunk)
                    except queue.Empty:
                        pass
        except Exception as e:
            logger.error(f"[MeetingIntel] Record error: {e}")
            self._running = False

    def _transcribe(self, audio_path: Optional[Path]) -> str:
        if not audio_path or not audio_path.exists():
            return ""
        try:
            from faster_whisper import WhisperModel
            logger.info("[MeetingIntel] Transcribing ...")
            model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
            segments, _ = model.transcribe(str(audio_path), beam_size=1)
            return " ".join(seg.text for seg in segments).strip()
        except Exception as e:
            logger.error(f"[MeetingIntel] Transcription error: {e}")
            return ""

    def _analyze_transcript(self, transcript: str) -> Dict:
        """Extract structure from transcript using keyword heuristics."""
        if not transcript:
            return {"summary": "No speech detected.", "action_items": [], "decisions": [], "key_topics": []}

        sentences = [s.strip() for s in transcript.replace(".", ".\n").splitlines() if s.strip()]

        action_keywords = ["action:", "todo:", "i'll", "we'll", "will do", "follow up", "next step", "need to", "should", "must", "going to"]
        decision_keywords = ["decided", "agreed", "going with", "we're using", "final decision", "we chose", "let's go with", "confirmed"]

        action_items = [s for s in sentences if any(k in s.lower() for k in action_keywords)][:10]
        decisions = [s for s in sentences if any(k in s.lower() for k in decision_keywords)][:10]

        # Simple topic extraction: find most repeated nouns (basic word frequency)
        import re
        from collections import Counter
        words = re.findall(r"\b[A-Za-z]{4,}\b", transcript.lower())
        stop = {"that", "this", "with", "from", "they", "have", "been", "will", "were", "would", "could", "should", "when", "what", "then", "just", "also", "some", "more", "very", "than", "about"}
        freq = Counter(w for w in words if w not in stop)
        key_topics = [word for word, _ in freq.most_common(8)]

        # Summary: first 400 chars
        summary = transcript[:400] + ("..." if len(transcript) > 400 else "")

        return {
            "summary": summary,
            "action_items": action_items,
            "decisions": decisions,
            "key_topics": key_topics,
        }


class MeetingIntelTool:
    """Record meetings locally, transcribe with Whisper, extract action items and decisions."""

    name = "meeting_intel"
    description = "Record and analyze meetings locally — Whisper transcription, action items, decisions. Fully offline."

    def __init__(self):
        MEETINGS_DIR.mkdir(parents=True, exist_ok=True)
        self._recorder = MeetingRecorder()

    def list_meetings(self, limit: int = 10) -> Dict:
        """List recent meeting records."""
        records = sorted(MEETINGS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]
        meetings = []
        for path in records:
            try:
                with open(path, encoding="utf-8") as f:
                    m = json.load(f)
                meetings.append({
                    "id": m.get("id"),
                    "title": m.get("title"),
                    "started_at": m.get("started_at", "")[:16],
                    "duration_min": round(m.get("duration_sec", 0) / 60, 1),
                    "action_items": len(m.get("action_items", [])),
                    "decisions": len(m.get("decisions", [])),
                })
            except Exception:
                pass
        return {"success": True, "count": len(meetings), "meetings": meetings}

    def get_meeting(self, meeting_id: str) -> Dict:
        """Retrieve a full meeting record."""
        path = MEETINGS_DIR / f"{meeting_id}.json"
        if not path.exists():
            matches = list(MEETINGS_DIR.glob(f"{meeting_id}*.json"))
            if not matches:
                return {"success": False, "error": f"Meeting '{meeting_id}' not found"}
            path = matches[0]
        try:
            with open(path, encoding="utf-8") as f:
                return {"success": True, **json.load(f)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def search_meetings(self, query: str, limit: int = 10) -> Dict:
        """Search all meeting transcripts and summaries."""
        q = query.lower()
        results = []
        for path in sorted(MEETINGS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                with open(path, encoding="utf-8") as f:
                    m = json.load(f)
                text = (m.get("transcript", "") + " " + m.get("summary", "")).lower()
                if q in text:
                    idx = text.find(q)
                    snippet = text[max(0, idx - 60): idx + 120].strip()
                    results.append({
                        "id": m.get("id"),
                        "title": m.get("title"),
                        "started_at": m.get("started_at", "")[:16],
                        "snippet": snippet,
                    })
                    if len(results) >= limit:
                        break
            except Exception:
                pass
        return {"success": True, "query": query, "count": len(results), "results": results}

    def get_cross_meeting_themes(self, limit: int = 5) -> Dict:
        """Find themes that appear across multiple meetings."""
        import re
        from collections import Counter

        word_freq: Counter = Counter()
        total = 0
        for path in sorted(MEETINGS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:20]:
            try:
                with open(path, encoding="utf-8") as f:
                    m = json.load(f)
                text = m.get("transcript", "") + " " + " ".join(m.get("key_topics", []))
                words = re.findall(r"\b[A-Za-z]{5,}\b", text.lower())
                stop = {"that", "this", "with", "from", "they", "have", "been", "will", "were", "would", "could", "should", "about", "there", "their", "these", "those", "really"}
                word_freq.update(w for w in words if w not in stop)
                total += 1
            except Exception:
                pass

        themes = [{"topic": w, "mentions": c} for w, c in word_freq.most_common(limit * 2) if c >= 2][:limit]
        return {"success": True, "meetings_analyzed": total, "recurring_themes": themes}

    def execute(self, action: str, **kwargs) -> Dict:
        a = action.lower().strip()
        if "start" in a or "record" in a or "begin" in a:
            return self._recorder.start(kwargs.get("title", ""))
        if "stop" in a or "end" in a or "finish" in a or "done" in a or "analyze" in a:
            return self._recorder.stop_and_analyze()
        if "list" in a or "show" in a:
            return self.list_meetings(kwargs.get("limit", 10))
        if "get" in a or "view" in a:
            return self.get_meeting(kwargs.get("meeting_id") or kwargs.get("id") or "")
        if "search" in a:
            return self.search_meetings(kwargs.get("query") or action)
        if "theme" in a or "cross" in a or "recurring" in a:
            return self.get_cross_meeting_themes()
        if "status" in a:
            return {
                "success": True,
                "recording": self._recorder._running,
                "current_meeting": self._recorder._meeting_id,
                "title": self._recorder._title,
            }
        return self.list_meetings()
