"""
EvoEmo - Emotional State Tracking System for Aura
Tool #20: Detects user emotions and adapts responses accordingly.
"""

import json
import re
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, asdict
from collections import Counter
import uuid


# Emotion states
EMOTION_STATES = ["calm", "focused", "stressed", "frustrated", "excited", "tired", "curious"]

# Emotion indicators
EMOTION_MARKERS = {
    "frustrated": {
        "words": ["ugh", "argh", "damn", "annoying", "stupid", "hate", "broken", "doesn't work",
                  "not working", "failed", "again", "still", "why won't", "can't believe",
                  "ridiculous", "useless", "waste", "impossible", "give up"],
        "patterns": [r"!{2,}", r"\?{2,}", r"\.{3,}", r"wtf", r"ffs", r"smh"],
        "caps_threshold": 0.4,  # 40%+ caps indicates frustration
        "weight": 1.5
    },
    "stressed": {
        "words": ["urgent", "asap", "hurry", "deadline", "emergency", "quick", "fast",
                  "immediately", "right now", "need", "must", "have to", "running out",
                  "pressure", "overwhelmed", "too much", "help me", "please help"],
        "patterns": [r"!+", r"ASAP", r"URGENT"],
        "weight": 1.3
    },
    "excited": {
        "words": ["awesome", "amazing", "great", "love", "fantastic", "wonderful", "perfect",
                  "excited", "can't wait", "yes", "finally", "brilliant", "incredible",
                  "wow", "cool", "nice", "sweet", "yay", "woohoo"],
        "patterns": [r"!{1,}", r":D", r":)", r"<3", r"\byes\b"],
        "weight": 1.2
    },
    "tired": {
        "words": ["tired", "exhausted", "sleepy", "long day", "drained", "worn out",
                  "can't think", "brain fog", "whatever", "don't care", "just",
                  "too tired", "need rest", "zzz", "yawn", "meh"],
        "patterns": [r"\.{2,}", r"zzz+", r"meh"],
        "short_response_indicator": True,
        "weight": 1.1
    },
    "curious": {
        "words": ["how", "why", "what", "wonder", "curious", "interesting", "explain",
                  "tell me", "learn", "understand", "could you", "what if", "how does",
                  "wondering", "question", "know more"],
        "patterns": [r"\?$", r"^(how|why|what|when|where|who)"],
        "weight": 1.0
    },
    "focused": {
        "words": ["specifically", "exactly", "precisely", "detail", "step by step",
                  "focus", "only", "just need", "the answer", "directly", "simply"],
        "patterns": [r"^[a-z]", r"\.$"],  # Lowercase start, period end = measured
        "weight": 1.0
    },
    "calm": {
        "words": ["thanks", "thank you", "please", "appreciate", "no rush", "whenever",
                  "take your time", "no problem", "all good", "sounds good", "okay"],
        "patterns": [r"^[A-Z][a-z].*\.$"],  # Proper sentence structure
        "weight": 0.8
    }
}

# Voice tone markers (from Whisper transcriptions)
VOICE_MARKERS = {
    "fast_speech": ["stressed", "excited"],
    "slow_speech": ["tired", "calm"],
    "high_pitch": ["excited", "stressed"],
    "low_pitch": ["tired", "calm"],
    "loud": ["frustrated", "excited"],
    "quiet": ["tired", "calm", "focused"]
}


@dataclass
class EmotionReading:
    """Single emotion detection result."""
    emotion: str
    confidence: int  # 0-100
    timestamp: str
    trigger_text: str
    session_id: str
    markers_found: List[str]
    voice_markers: Optional[List[str]] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MoodSummary:
    """Daily mood summary."""
    date: str
    dominant_emotion: str
    emotion_distribution: Dict[str, int]
    average_confidence: float
    total_readings: int
    peak_stress_time: Optional[str] = None
    calmest_period: Optional[str] = None


class EvoEmoTool:
    """Emotional state tracking and adaptive response system."""

    def __init__(self, data_dir: Optional[str] = None):
        self.name = "evoemo"
        self.description = "Track emotional state and adapt responses"

        # Data directory
        if data_dir:
            self.data_dir = Path(data_dir)
        else:
            self.data_dir = Path(__file__).parent.parent.parent / "data" / "evoemo"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Files
        self.mood_history_file = self.data_dir / "mood_history.jsonl"
        self.settings_file = self.data_dir / "settings.json"

        # Current session
        self.session_id = str(uuid.uuid4())[:8]
        self.current_mood: Optional[EmotionReading] = None
        self.session_history: List[EmotionReading] = []

        # Settings
        self.settings = self._load_settings()

        # Tracking enabled by default
        self.enabled = self.settings.get("enabled", True)

    def _load_settings(self) -> dict:
        """Load settings from file."""
        if self.settings_file.exists():
            try:
                return json.loads(self.settings_file.read_text())
            except:
                pass
        return {"enabled": True, "history_days": 7}

    def _save_settings(self):
        """Save settings to file."""
        self.settings_file.write_text(json.dumps(self.settings, indent=2))

    def analyze_text(self, text: str, voice_markers: Optional[List[str]] = None) -> EmotionReading:
        """
        Analyze text for emotional signals.

        Args:
            text: User input text
            voice_markers: Optional voice characteristics from Whisper

        Returns:
            EmotionReading with detected emotion and confidence
        """
        if not self.enabled:
            return EmotionReading(
                emotion="calm",
                confidence=50,
                timestamp=datetime.now().isoformat(),
                trigger_text=text[:100],
                session_id=self.session_id,
                markers_found=["tracking_disabled"]
            )

        text_lower = text.lower()
        scores: Dict[str, float] = {emotion: 0.0 for emotion in EMOTION_STATES}
        markers_found: List[str] = []

        # Analyze text patterns
        for emotion, markers in EMOTION_MARKERS.items():
            # Word matching
            for word in markers.get("words", []):
                if word in text_lower:
                    scores[emotion] += markers.get("weight", 1.0)
                    markers_found.append(f"word:{word}")

            # Pattern matching
            for pattern in markers.get("patterns", []):
                if re.search(pattern, text, re.IGNORECASE):
                    scores[emotion] += markers.get("weight", 1.0) * 0.5
                    markers_found.append(f"pattern:{pattern[:20]}")

            # Caps analysis for frustration
            if emotion == "frustrated" and len(text) > 10:
                caps_ratio = sum(1 for c in text if c.isupper()) / len(text)
                if caps_ratio > markers.get("caps_threshold", 0.4):
                    scores[emotion] += 2.0
                    markers_found.append(f"caps:{caps_ratio:.0%}")

            # Short response for tired
            if emotion == "tired" and markers.get("short_response_indicator"):
                if len(text.split()) <= 3 and not text.endswith("?"):
                    scores[emotion] += 0.5
                    markers_found.append("short_response")

        # Voice marker analysis
        if voice_markers:
            for marker in voice_markers:
                if marker in VOICE_MARKERS:
                    for emotion in VOICE_MARKERS[marker]:
                        scores[emotion] += 0.8
                        markers_found.append(f"voice:{marker}")

        # Determine dominant emotion
        if max(scores.values()) == 0:
            # Default to calm if no signals detected
            dominant_emotion = "calm"
            confidence = 60
        else:
            dominant_emotion = max(scores, key=scores.get)
            # Confidence based on score strength and marker count
            raw_confidence = min(scores[dominant_emotion] * 20, 100)
            marker_bonus = min(len(markers_found) * 5, 20)
            confidence = int(min(raw_confidence + marker_bonus, 100))

        # Create reading
        reading = EmotionReading(
            emotion=dominant_emotion,
            confidence=confidence,
            timestamp=datetime.now().isoformat(),
            trigger_text=text[:100] if len(text) > 100 else text,
            session_id=self.session_id,
            markers_found=markers_found,
            voice_markers=voice_markers
        )

        # Update current mood
        self.current_mood = reading
        self.session_history.append(reading)

        # Persist to history
        self._append_to_history(reading)

        return reading

    def _append_to_history(self, reading: EmotionReading):
        """Append reading to mood history file."""
        try:
            with open(self.mood_history_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(reading.to_dict()) + "\n")
        except Exception as e:
            print(f"[EvoEmo] Failed to save mood: {e}")

    def get_current_mood(self) -> Optional[EmotionReading]:
        """Get the most recent mood reading."""
        return self.current_mood

    def get_mood_emoji(self) -> str:
        """Get emoji representation of current mood."""
        if not self.current_mood:
            return "😐"

        emoji_map = {
            "calm": "😌",
            "focused": "🎯",
            "stressed": "😰",
            "frustrated": "😤",
            "excited": "🤩",
            "tired": "😴",
            "curious": "🤔"
        }
        return emoji_map.get(self.current_mood.emotion, "😐")

    def get_mood_color(self) -> str:
        """Get color code for current mood (for GUI)."""
        if not self.current_mood:
            return "#94a3b8"  # Gray

        color_map = {
            "calm": "#22c55e",      # Green
            "focused": "#3b82f6",   # Blue
            "stressed": "#f97316",  # Orange
            "frustrated": "#ef4444", # Red
            "excited": "#eab308",   # Yellow
            "tired": "#8b5cf6",     # Purple
            "curious": "#06b6d4"    # Cyan
        }
        return color_map.get(self.current_mood.emotion, "#94a3b8")

    def get_session_summary(self) -> dict:
        """Get summary of current session emotions."""
        if not self.session_history:
            return {"dominant": "calm", "readings": 0, "distribution": {}}

        emotions = [r.emotion for r in self.session_history]
        distribution = dict(Counter(emotions))
        dominant = max(distribution, key=distribution.get)
        avg_confidence = sum(r.confidence for r in self.session_history) / len(self.session_history)

        return {
            "dominant": dominant,
            "readings": len(self.session_history),
            "distribution": distribution,
            "average_confidence": round(avg_confidence, 1),
            "session_id": self.session_id
        }

    def get_daily_summary(self, date: Optional[str] = None) -> Optional[MoodSummary]:
        """Get mood summary for a specific day."""
        if not self.mood_history_file.exists():
            return None

        target_date = date or datetime.now().strftime("%Y-%m-%d")
        readings = []

        try:
            with open(self.mood_history_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        if data["timestamp"].startswith(target_date):
                            readings.append(data)
                    except:
                        continue
        except Exception as e:
            print(f"[EvoEmo] Error reading history: {e}")
            return None

        if not readings:
            return None

        emotions = [r["emotion"] for r in readings]
        distribution = dict(Counter(emotions))
        dominant = max(distribution, key=distribution.get)
        avg_confidence = sum(r["confidence"] for r in readings) / len(readings)

        # Find peak stress time
        stress_readings = [r for r in readings if r["emotion"] in ["stressed", "frustrated"]]
        peak_stress_time = None
        if stress_readings:
            peak_stress_time = max(stress_readings, key=lambda x: x["confidence"])["timestamp"]

        # Find calmest period
        calm_readings = [r for r in readings if r["emotion"] in ["calm", "focused"]]
        calmest_period = None
        if calm_readings:
            calmest_period = max(calm_readings, key=lambda x: x["confidence"])["timestamp"]

        return MoodSummary(
            date=target_date,
            dominant_emotion=dominant,
            emotion_distribution=distribution,
            average_confidence=round(avg_confidence, 1),
            total_readings=len(readings),
            peak_stress_time=peak_stress_time,
            calmest_period=calmest_period
        )

    def get_history(self, days: int = 7) -> List[dict]:
        """Get mood history for the last N days."""
        if not self.mood_history_file.exists():
            return []

        cutoff = datetime.now() - timedelta(days=days)
        history = []

        try:
            with open(self.mood_history_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        timestamp = datetime.fromisoformat(data["timestamp"])
                        if timestamp >= cutoff:
                            history.append(data)
                    except:
                        continue
        except Exception as e:
            print(f"[EvoEmo] Error reading history: {e}")

        return history

    def get_patterns(self) -> dict:
        """Analyze emotional patterns over time."""
        history = self.get_history(days=7)
        if len(history) < 5:
            return {"status": "insufficient_data", "readings": len(history)}

        # Time-based patterns
        hour_emotions: Dict[int, List[str]] = {}
        for reading in history:
            try:
                hour = datetime.fromisoformat(reading["timestamp"]).hour
                if hour not in hour_emotions:
                    hour_emotions[hour] = []
                hour_emotions[hour].append(reading["emotion"])
            except:
                continue

        # Find stress hours
        stress_hours = []
        for hour, emotions in hour_emotions.items():
            stress_ratio = emotions.count("stressed") + emotions.count("frustrated")
            if stress_ratio / len(emotions) > 0.3:
                stress_hours.append(hour)

        # Overall distribution
        all_emotions = [r["emotion"] for r in history]
        distribution = dict(Counter(all_emotions))

        return {
            "status": "ok",
            "readings": len(history),
            "distribution": distribution,
            "stress_hours": sorted(stress_hours),
            "dominant_emotion": max(distribution, key=distribution.get) if distribution else "calm"
        }

    def clear_history(self) -> dict:
        """Clear all mood history (privacy feature)."""
        try:
            if self.mood_history_file.exists():
                self.mood_history_file.unlink()
            self.session_history.clear()
            self.current_mood = None
            return {"success": True, "message": "Mood history cleared"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def set_enabled(self, enabled: bool) -> dict:
        """Enable or disable mood tracking."""
        self.enabled = enabled
        self.settings["enabled"] = enabled
        self._save_settings()
        return {"success": True, "enabled": enabled}

    def is_enabled(self) -> bool:
        """Check if mood tracking is enabled."""
        return self.enabled

    def execute(self, action: str, **kwargs) -> dict:
        """Execute an EvoEmo action."""
        action_lower = action.lower()

        if "analyze" in action_lower or "detect" in action_lower:
            # Extract text to analyze
            text = kwargs.get("text", action)
            reading = self.analyze_text(text)
            return {
                "success": True,
                "emotion": reading.emotion,
                "confidence": reading.confidence,
                "emoji": self.get_mood_emoji(),
                "markers": reading.markers_found
            }

        elif "current" in action_lower or "mood" in action_lower:
            mood = self.get_current_mood()
            if mood:
                return {
                    "success": True,
                    "emotion": mood.emotion,
                    "confidence": mood.confidence,
                    "emoji": self.get_mood_emoji(),
                    "color": self.get_mood_color()
                }
            return {"success": True, "emotion": "unknown", "message": "No mood data yet"}

        elif "summary" in action_lower:
            if "session" in action_lower:
                return {"success": True, **self.get_session_summary()}
            else:
                summary = self.get_daily_summary()
                if summary:
                    return {"success": True, **asdict(summary)}
                return {"success": True, "message": "No data for today"}

        elif "history" in action_lower:
            days = kwargs.get("days", 7)
            return {"success": True, "history": self.get_history(days)}

        elif "patterns" in action_lower:
            return {"success": True, **self.get_patterns()}

        elif "clear" in action_lower:
            return self.clear_history()

        elif "enable" in action_lower:
            return self.set_enabled(True)

        elif "disable" in action_lower:
            return self.set_enabled(False)

        elif "status" in action_lower:
            return {
                "success": True,
                "enabled": self.enabled,
                "current_mood": self.current_mood.emotion if self.current_mood else None,
                "session_readings": len(self.session_history),
                "session_id": self.session_id
            }

        return {"success": False, "error": f"Unknown action: {action}"}


# Singleton instance
evoemo = EvoEmoTool()


# Convenience functions
def analyze_emotion(text: str, voice_markers: Optional[List[str]] = None) -> EmotionReading:
    """Analyze text for emotional signals."""
    return evoemo.analyze_text(text, voice_markers)


def get_current_mood() -> Optional[EmotionReading]:
    """Get the current mood reading."""
    return evoemo.get_current_mood()


def get_mood_emoji() -> str:
    """Get emoji for current mood."""
    return evoemo.get_mood_emoji()


def get_mood_color() -> str:
    """Get color for current mood."""
    return evoemo.get_mood_color()


def is_tracking_enabled() -> bool:
    """Check if mood tracking is enabled."""
    return evoemo.is_enabled()
