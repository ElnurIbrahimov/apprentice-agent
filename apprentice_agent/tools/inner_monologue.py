"""
Inner Monologue System for Aura
Makes Aura's thinking visible and audible in real-time.

Author: Aura Development Team
Created: 2026-01-26
"""

import json
import uuid
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any, Generator, Callable
from dataclasses import dataclass, asdict, field
import time


# Thought types for different reasoning stages
THOUGHT_TYPES = {
    "perceive": "Perceiving...",       # Understanding user input
    "recall": "Recalling...",           # Memory retrieval
    "reason": "Reasoning...",           # Logic/planning
    "decide": "Deciding...",            # Tool/action selection
    "execute": "Executing...",          # Running tool
    "reflect": "Reflecting...",         # Evaluating result
    "uncertain": "Uncertain...",        # Low confidence moment
    "eureka": "Eureka!",                # Insight/breakthrough
}

THOUGHT_ICONS = {
    "perceive": "\U0001F50D",   # magnifying glass
    "recall": "\U0001F4BE",     # floppy disk
    "reason": "\U0001F9E0",     # brain
    "decide": "\u26A1",         # lightning
    "execute": "\U0001F527",    # wrench
    "reflect": "\U0001FA9E",    # mirror (or fallback)
    "uncertain": "\u2753",      # question mark
    "eureka": "\U0001F4A1",     # light bulb
}


@dataclass
class Thought:
    """Represents a single thought in the inner monologue."""
    id: str
    timestamp: str
    type: str
    content: str
    confidence: Optional[int] = None
    duration_ms: Optional[int] = None
    parent_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert thought to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Thought':
        """Create Thought from dictionary."""
        return cls(**data)

    def format_display(self, show_confidence: bool = True, show_time: bool = False) -> str:
        """Format thought for display."""
        icon = THOUGHT_ICONS.get(self.type, "\U0001F4AD")  # default: thought bubble
        conf_str = f" [{self.confidence}%]" if self.confidence is not None and show_confidence else ""
        time_str = f" ({self.timestamp[11:19]})" if show_time else ""
        return f"{icon} **{self.type.upper()}**{conf_str}{time_str}: {self.content}"


@dataclass
class MonologueSession:
    """Represents a full inner monologue session."""
    session_id: str
    started_at: str
    ended_at: Optional[str] = None
    thoughts: List[Thought] = field(default_factory=list)
    verbosity: int = 2
    total_duration_ms: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary."""
        return {
            "session_id": self.session_id,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "thoughts": [t.to_dict() for t in self.thoughts],
            "verbosity": self.verbosity,
            "total_duration_ms": self.total_duration_ms,
        }


class MonologueStream:
    """Real-time streaming infrastructure for thoughts."""

    def __init__(self, max_buffer: int = 100):
        self.subscribers: List[Callable[[Thought], None]] = []
        self.buffer: List[Thought] = []
        self.max_buffer = max_buffer
        self._lock = threading.Lock()

    def subscribe(self, callback: Callable[[Thought], None]) -> None:
        """Subscribe to thought stream."""
        with self._lock:
            self.subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[Thought], None]) -> None:
        """Unsubscribe from thought stream."""
        with self._lock:
            if callback in self.subscribers:
                self.subscribers.remove(callback)

    def emit(self, thought: Thought) -> None:
        """Emit a thought to all subscribers."""
        with self._lock:
            self.buffer.append(thought)
            # Trim buffer if needed
            if len(self.buffer) > self.max_buffer:
                self.buffer = self.buffer[-self.max_buffer:]

            # Notify all subscribers
            for cb in self.subscribers:
                try:
                    cb(thought)
                except Exception as e:
                    print(f"[Monologue] Subscriber error: {e}")

    def get_recent(self, n: int = 10) -> List[Thought]:
        """Get n most recent thoughts."""
        with self._lock:
            return self.buffer[-n:]

    def clear(self) -> None:
        """Clear the buffer."""
        with self._lock:
            self.buffer.clear()


class InnerMonologueTool:
    """
    Core Inner Monologue engine for Aura.

    Makes Aura's thinking visible in real-time, supports
    streaming to GUI, voice output via Sesame, and logging.
    """

    name = "inner_monologue"
    description = "Aura's inner monologue system - makes thinking visible"

    def __init__(self, logs_dir: Optional[Path] = None):
        """
        Initialize the inner monologue system.

        Args:
            logs_dir: Directory for storing monologue logs.
                      Defaults to data/inner_monologue/
        """
        # Session state
        self.current_session: Optional[MonologueSession] = None
        self._thought_counter = 0
        self._session_start_time: Optional[float] = None
        self._last_thought_time: Optional[float] = None

        # Settings
        self.verbosity = 2  # 0=silent, 1=key, 2=verbose, 3=debug
        self.think_aloud_enabled = False

        # Streaming
        self.stream = MonologueStream()

        # Logging paths
        if logs_dir is None:
            self.logs_dir = Path(__file__).parent.parent / "data" / "inner_monologue"
        else:
            self.logs_dir = Path(logs_dir)

        self.sessions_dir = self.logs_dir / "sessions"
        self.summaries_dir = self.logs_dir / "summaries"

        # Ensure directories exist
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.summaries_dir.mkdir(parents=True, exist_ok=True)

        # TTS reference (set externally)
        self.tts_engine = None

        # EvoEmo reference (set externally)
        self.evoemo = None

        self._lock = threading.Lock()

    def start_session(self) -> str:
        """
        Initialize a new monologue session.

        Returns:
            Session UUID
        """
        with self._lock:
            session_id = str(uuid.uuid4())[:8]
            now = datetime.now()

            self.current_session = MonologueSession(
                session_id=session_id,
                started_at=now.isoformat(),
                verbosity=self.verbosity,
            )

            self._thought_counter = 0
            self._session_start_time = time.time()
            self._last_thought_time = self._session_start_time

            # Clear stream buffer for new session
            self.stream.clear()

            return session_id

    def think(
        self,
        thought_type: str,
        content: str,
        confidence: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        parent_id: Optional[str] = None,
    ) -> Optional[Thought]:
        """
        Log a thought to the inner monologue.

        Args:
            thought_type: Type of thought (perceive, recall, reason, etc.)
            content: The thought content
            confidence: Optional confidence score 0-100
            metadata: Optional additional data
            parent_id: Optional parent thought ID for nested chains

        Returns:
            The created Thought object, or None if filtered by verbosity
        """
        # Validate thought type
        if thought_type not in THOUGHT_TYPES:
            thought_type = "reason"  # default fallback

        # Check verbosity filter
        if not self._should_emit(thought_type):
            return None

        with self._lock:
            # Auto-start session if needed
            if self.current_session is None:
                self.start_session()

            # Calculate timing
            now = time.time()
            duration_ms = None
            if self._last_thought_time is not None:
                duration_ms = int((now - self._last_thought_time) * 1000)
            self._last_thought_time = now

            # Create thought
            self._thought_counter += 1
            thought = Thought(
                id=f"thought_{self._thought_counter:03d}",
                timestamp=datetime.now().isoformat(timespec='milliseconds'),
                type=thought_type,
                content=content,
                confidence=confidence,
                duration_ms=duration_ms,
                parent_id=parent_id,
                metadata=metadata or {},
            )

            # Add to session
            self.current_session.thoughts.append(thought)

        # Emit to stream (outside lock to prevent deadlock)
        self.stream.emit(thought)

        # Think aloud if enabled
        if self.think_aloud_enabled:
            self._think_aloud(thought)

        return thought

    def _should_emit(self, thought_type: str) -> bool:
        """Check if thought should be emitted based on verbosity."""
        if self.verbosity == 0:
            return False
        elif self.verbosity == 1:
            # Key thoughts only
            return thought_type in ["decide", "eureka", "uncertain"]
        elif self.verbosity == 2:
            # All except debug-level
            return thought_type != "execute" or True  # Show all for now
        else:  # verbosity == 3
            return True

    def _think_aloud(self, thought: Thought) -> None:
        """
        Speak thought via TTS if enabled.

        Only speaks key thoughts, not debug noise.
        """
        if self.tts_engine is None:
            return

        # Only speak certain thought types
        if thought.type not in ["reason", "decide", "eureka", "uncertain"]:
            return

        try:
            # Simplify for speech
            text = self._simplify_for_speech(thought.content)

            # Speak at lower volume / faster pace
            # Note: Sesame TTS may not support all these params
            self.tts_engine.speak(
                text,
                use_sesame=True,
                # speed=1.2,  # Faster for inner thoughts
                # volume=0.5,  # Lower volume - it's internal
            )
        except Exception as e:
            print(f"[Monologue] Think aloud error: {e}")

    def _simplify_for_speech(self, content: str) -> str:
        """Simplify content for natural speech."""
        # Remove technical jargon
        content = content.replace("tool:", "")
        content = content.replace("confidence:", "I'm about")
        content = content.replace("_", " ")

        # Truncate long content
        if len(content) > 100:
            content = content[:100] + "..."

        return content

    def get_stream(self) -> Generator[Thought, None, None]:
        """
        Generator yielding thoughts in real-time.

        Usage:
            for thought in monologue.get_stream():
                print(thought.format_display())
        """
        # Create a queue for this consumer
        import queue
        thought_queue: queue.Queue = queue.Queue()

        def on_thought(thought: Thought):
            thought_queue.put(thought)

        self.stream.subscribe(on_thought)

        try:
            while True:
                try:
                    thought = thought_queue.get(timeout=0.1)
                    yield thought
                except queue.Empty:
                    # Check if session ended
                    if self.current_session is None:
                        break
                    continue
        finally:
            self.stream.unsubscribe(on_thought)

    def get_session_log(self) -> List[Dict[str, Any]]:
        """Return full session as list of thought dicts."""
        if self.current_session is None:
            return []

        with self._lock:
            return [t.to_dict() for t in self.current_session.thoughts]

    def get_recent_thoughts(self, n: int = 20) -> List[Thought]:
        """Get n most recent thoughts from current session."""
        return self.stream.get_recent(n)

    def set_verbosity(self, level: int) -> str:
        """
        Set verbosity level.

        Args:
            level: 0=silent, 1=key thoughts, 2=verbose, 3=debug

        Returns:
            Confirmation message
        """
        self.verbosity = max(0, min(3, level))

        if self.current_session:
            self.current_session.verbosity = self.verbosity

        levels = {0: "silent", 1: "key thoughts only", 2: "verbose", 3: "debug"}
        return f"Monologue verbosity set to {self.verbosity} ({levels.get(self.verbosity, 'unknown')})"

    def end_session(self) -> Dict[str, Any]:
        """
        Finalize and save the current session.

        Returns:
            Session summary dict
        """
        with self._lock:
            if self.current_session is None:
                return {"success": False, "error": "No active session"}

            now = datetime.now()
            self.current_session.ended_at = now.isoformat()

            if self._session_start_time:
                self.current_session.total_duration_ms = int(
                    (time.time() - self._session_start_time) * 1000
                )

            # Save session to file
            session_file = self.sessions_dir / f"{now.strftime('%Y-%m-%d')}_session_{self.current_session.session_id}.jsonl"

            try:
                with open(session_file, 'w', encoding='utf-8') as f:
                    for thought in self.current_session.thoughts:
                        f.write(json.dumps(thought.to_dict()) + '\n')
            except Exception as e:
                print(f"[Monologue] Failed to save session: {e}")

            # Create summary
            summary = self._create_session_summary()

            # Update daily summary
            self._update_daily_summary(now.strftime('%Y-%m-%d'))

            # Clear session
            session_id = self.current_session.session_id
            self.current_session = None
            self._thought_counter = 0
            self._session_start_time = None
            self._last_thought_time = None

            return {
                "success": True,
                "session_id": session_id,
                "summary": summary,
            }

    def _create_session_summary(self) -> Dict[str, Any]:
        """Create summary statistics for current session."""
        if not self.current_session or not self.current_session.thoughts:
            return {}

        thoughts = self.current_session.thoughts

        # Count by type
        type_counts = {}
        confidence_by_type = {}

        for t in thoughts:
            type_counts[t.type] = type_counts.get(t.type, 0) + 1

            if t.confidence is not None:
                if t.type not in confidence_by_type:
                    confidence_by_type[t.type] = []
                confidence_by_type[t.type].append(t.confidence)

        # Average confidence by type
        avg_confidence = {}
        for ttype, confs in confidence_by_type.items():
            avg_confidence[ttype] = sum(confs) / len(confs)

        # Find eureka moments
        eureka_count = type_counts.get("eureka", 0)

        # Find uncertainty triggers
        uncertain_thoughts = [t for t in thoughts if t.type == "uncertain"]

        return {
            "total_thoughts": len(thoughts),
            "thoughts_by_type": type_counts,
            "avg_confidence_by_type": avg_confidence,
            "eureka_moments": eureka_count,
            "uncertain_moments": len(uncertain_thoughts),
            "duration_ms": self.current_session.total_duration_ms,
        }

    def _update_daily_summary(self, date_str: str) -> None:
        """Update daily aggregate summary."""
        summary_file = self.summaries_dir / f"{date_str}_summary.json"

        try:
            # Load existing summary or create new
            if summary_file.exists():
                with open(summary_file, 'r', encoding='utf-8') as f:
                    daily = json.load(f)
            else:
                daily = {
                    "date": date_str,
                    "total_sessions": 0,
                    "total_thoughts": 0,
                    "thoughts_by_type": {},
                    "total_eureka": 0,
                    "total_uncertain": 0,
                    "sessions": [],
                }

            # Get session summary
            session_summary = self._create_session_summary()

            # Update aggregates
            daily["total_sessions"] += 1
            daily["total_thoughts"] += session_summary.get("total_thoughts", 0)
            daily["total_eureka"] += session_summary.get("eureka_moments", 0)
            daily["total_uncertain"] += session_summary.get("uncertain_moments", 0)

            # Merge type counts
            for ttype, count in session_summary.get("thoughts_by_type", {}).items():
                daily["thoughts_by_type"][ttype] = daily["thoughts_by_type"].get(ttype, 0) + count

            # Add session reference
            daily["sessions"].append({
                "session_id": self.current_session.session_id,
                "thoughts": session_summary.get("total_thoughts", 0),
                "duration_ms": session_summary.get("duration_ms"),
            })

            # Save
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(daily, f, indent=2)

        except Exception as e:
            print(f"[Monologue] Failed to update daily summary: {e}")

    def get_reasoning_chain(self, last_n: int = 10) -> str:
        """
        Get formatted reasoning chain for "why did you do that?" queries.

        Returns markdown-formatted explanation of recent reasoning.
        """
        thoughts = self.get_recent_thoughts(last_n)

        if not thoughts:
            return "*No recent thoughts recorded.*"

        lines = ["## Aura's Recent Reasoning\n"]

        for thought in thoughts:
            lines.append(thought.format_display(show_confidence=True, show_time=True))

        return "\n".join(lines)

    def export_session(self, filepath: Optional[str] = None) -> Dict[str, Any]:
        """Export current session to JSON file."""
        if self.current_session is None:
            return {"success": False, "error": "No active session"}

        if filepath is None:
            filepath = str(
                self.sessions_dir /
                f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.current_session.to_dict(), f, indent=2)

            return {"success": True, "filepath": filepath}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def connect_evoemo(self, evoemo) -> None:
        """Connect EvoEmo for emotionally-aware reasoning."""
        self.evoemo = evoemo

    def connect_tts(self, tts_engine) -> None:
        """Connect TTS engine for think-aloud mode."""
        self.tts_engine = tts_engine

    def think_with_emotion(
        self,
        thought_type: str,
        content: str,
        confidence: Optional[int] = None,
        user_mood: Optional[str] = None,
    ) -> Optional[Thought]:
        """
        Think with emotional awareness from EvoEmo.

        If user_mood is stressed/frustrated and confidence is low,
        adjusts the thought accordingly.
        """
        # Get user mood from EvoEmo if not provided
        if user_mood is None and self.evoemo is not None:
            try:
                mood_result = self.evoemo.get_current_mood()
                if mood_result.get("success"):
                    user_mood = mood_result.get("emotion")
            except Exception:
                pass

        # Adjust for emotional context
        if user_mood in ["frustrated", "stressed"]:
            if thought_type == "uncertain" or (confidence and confidence < 50):
                content = f"User seems {user_mood}. Being careful: {content}"

        return self.think(
            thought_type=thought_type,
            content=content,
            confidence=confidence,
            metadata={"user_mood": user_mood} if user_mood else None,
        )

    def execute(self, action: str, **kwargs) -> Dict[str, Any]:
        """
        Execute inner monologue actions (tool interface).

        Actions:
            - "show thoughts" / "get thoughts": Display recent thoughts
            - "think aloud on/off": Toggle voice output
            - "verbosity N": Set detail level
            - "why" / "explain": Get reasoning chain
            - "export": Save session to file
            - "start": Start new session
            - "end": End current session
            - "status": Get current status
        """
        action_lower = action.lower().strip()

        # Show thoughts
        if any(kw in action_lower for kw in ["show thought", "get thought", "recent thought"]):
            n = kwargs.get("n", 20)
            thoughts = self.get_recent_thoughts(n)
            formatted = "\n".join([t.format_display() for t in thoughts])
            return {
                "success": True,
                "count": len(thoughts),
                "thoughts": formatted or "*No thoughts yet.*",
            }

        # Think aloud toggle
        if "think aloud" in action_lower:
            if "on" in action_lower or "enable" in action_lower:
                self.think_aloud_enabled = True
                return {"success": True, "message": "Think aloud enabled. Aura will speak her thoughts."}
            elif "off" in action_lower or "disable" in action_lower:
                self.think_aloud_enabled = False
                return {"success": True, "message": "Think aloud disabled."}

        # Verbosity
        if "verbosity" in action_lower:
            # Extract number
            import re
            match = re.search(r'(\d+)', action_lower)
            if match:
                level = int(match.group(1))
                msg = self.set_verbosity(level)
                return {"success": True, "message": msg}

        # Why / Explain reasoning
        if any(kw in action_lower for kw in ["why", "explain", "reasoning", "how did you"]):
            chain = self.get_reasoning_chain()
            return {"success": True, "reasoning_chain": chain}

        # Export
        if "export" in action_lower:
            filepath = kwargs.get("filepath")
            return self.export_session(filepath)

        # Start session
        if "start" in action_lower:
            session_id = self.start_session()
            return {"success": True, "session_id": session_id, "message": f"Started monologue session {session_id}"}

        # End session
        if "end" in action_lower:
            return self.end_session()

        # Status
        if "status" in action_lower:
            return {
                "success": True,
                "active_session": self.current_session.session_id if self.current_session else None,
                "verbosity": self.verbosity,
                "think_aloud": self.think_aloud_enabled,
                "thought_count": len(self.current_session.thoughts) if self.current_session else 0,
                "has_tts": self.tts_engine is not None,
                "has_evoemo": self.evoemo is not None,
            }

        return {"success": False, "error": f"Unknown action: {action}"}


# Singleton instance for global access
_monologue_instance: Optional[InnerMonologueTool] = None


def get_monologue() -> InnerMonologueTool:
    """Get or create the global InnerMonologueTool instance."""
    global _monologue_instance
    if _monologue_instance is None:
        _monologue_instance = InnerMonologueTool()
    return _monologue_instance
