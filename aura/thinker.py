"""Talker/Thinker Split — MIRROR-architecture dual-process system.

The Thinker runs asynchronously between user turns using a fast/8B model.
It produces structured private thoughts across 3 threads:
  1. Goal tracking — What am I trying to accomplish?
  2. Reasoning audit — Did my last response make sense?
  3. Memory integration — What should I surface next time?

The Talker (main LLM in agent.py) reads the last Thinker output before
generating each response, injecting it as private system context.

Thinker output is NEVER shown to the user.

Implements roadmap item 3.6 based on the MIRROR Architecture (2025).
"""

import json
import logging
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# How long to keep thinker state before it goes stale (seconds)
THINKER_STATE_TTL = 300  # 5 minutes


@dataclass
class ThinkerState:
    """Structured output from one Thinker cycle."""

    # Thread 1: Goal tracking
    goal_state: str = ""          # Current goal / what we're trying to accomplish
    goal_progress: str = ""       # How far along we are

    # Thread 2: Reasoning audit
    reasoning_audit: str = ""     # Was the last response coherent / accurate?
    reasoning_gaps: str = ""      # What did we miss or get wrong?

    # Thread 3: Memory integration
    memory_suggestions: str = ""  # What should we surface next time?
    context_notes: str = ""       # Key facts to keep in mind

    # Meta
    timestamp: float = 0.0
    turn_number: int = 0
    processing_ms: int = 0
    last_touched: float = 0.0     # Last activity time (reset on each user message)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def touch(self) -> None:
        """Reset staleness clock — call on each new user message."""
        self.last_touched = time.time()

    def is_stale(self) -> bool:
        if self.timestamp == 0.0:
            return True
        # Use the most recent of creation time or last touch time
        anchor = max(self.timestamp, self.last_touched)
        return (time.time() - anchor) > THINKER_STATE_TTL

    def to_system_context(self) -> str:
        """Format thinker state as private context for the Talker.

        Returns a compact string to inject into the system prompt.
        Empty string if nothing useful.
        """
        if self.is_stale():
            return ""

        parts = []
        if self.goal_state:
            parts.append(f"GOAL: {self.goal_state}")
        if self.goal_progress:
            parts.append(f"PROGRESS: {self.goal_progress}")
        if self.reasoning_audit:
            parts.append(f"SELF-CHECK: {self.reasoning_audit}")
        if self.reasoning_gaps:
            parts.append(f"GAPS: {self.reasoning_gaps}")
        if self.memory_suggestions:
            parts.append(f"SURFACE NEXT: {self.memory_suggestions}")
        if self.context_notes:
            parts.append(f"CONTEXT: {self.context_notes}")

        if not parts:
            return ""

        return "THINKER (private reflection):\n" + "\n".join(parts)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ThinkerState":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in known})


# ─── Thinker prompts ───────────────────────────────────────────────

GOAL_TRACKING_PROMPT = """\
You are the Goal Tracker inside an AI assistant named Aura.
Given the conversation so far, identify:
1. What is the user's current goal or intent? (1 sentence)
2. How much progress has been made toward it? (1 sentence)

Conversation:
{conversation}

Reply ONLY in this exact JSON format, nothing else:
{{"goal_state": "...", "goal_progress": "..."}}"""

REASONING_AUDIT_PROMPT = """\
You are the Reasoning Auditor inside an AI assistant named Aura.
Review the last assistant response and determine:
1. Was the response accurate and coherent? Any errors or weak reasoning? (1 sentence)
2. What gaps or missed points should be addressed next? (1 sentence, or "none")

Last exchange:
User: {user_message}
Assistant: {assistant_response}

Reply ONLY in this exact JSON format, nothing else:
{{"reasoning_audit": "...", "reasoning_gaps": "..."}}"""

MEMORY_INTEGRATION_PROMPT = """\
You are the Memory Integrator inside an AI assistant named Aura.
Given this exchange, determine:
1. What information should be surfaced or remembered for the next response? (1 sentence)
2. Any key facts, preferences, or context to keep in mind? (1 sentence, or "none")

User: {user_message}
Assistant: {assistant_response}

Reply ONLY in this exact JSON format, nothing else:
{{"memory_suggestions": "...", "context_notes": "..."}}"""


# ─── Thinker Engine ────────────────────────────────────────────────

class ThinkerEngine:
    """Async background Thinker that runs between user turns.

    Usage:
        thinker = ThinkerEngine(brain)
        # After each response:
        thinker.run_async(user_msg, assistant_response, conversation_history)
        # Before next response:
        context = thinker.get_talker_context()
    """

    def __init__(self, brain=None, state_dir: Optional[Path] = None):
        self._brain = brain
        self._state: ThinkerState = ThinkerState()
        self._lock = threading.Lock()
        self._turn_counter = 0
        self._running = threading.Event()  # Set while a thinker cycle is in progress

        # Persistent state (survives restarts)
        if state_dir is None:
            self._state_dir = Path(__file__).parent.parent / "data" / "thinker"
        else:
            self._state_dir = Path(state_dir)
        self._state_dir.mkdir(parents=True, exist_ok=True)
        self._state_file = self._state_dir / "last_state.json"

        # Load persisted state on startup
        self._load_state()

    def set_brain(self, brain) -> None:
        """Set or update the brain reference (for late binding)."""
        self._brain = brain

    # ─── Public API ─────────────────────────────────────────────

    def run_async(
        self,
        user_message: str,
        assistant_response: str,
        conversation_history: list | None = None,
    ) -> None:
        """Kick off the Thinker asynchronously. Non-blocking.

        Called right after the Talker generates a response.
        """
        if self._brain is None:
            logger.debug("[Thinker] No brain set, skipping")
            return

        if not hasattr(self._brain, "_quick_generate"):
            logger.debug("[Thinker] Brain has no _quick_generate, skipping")
            return

        # Use lock for atomic check-and-start to prevent duplicate threads
        with self._lock:
            # Don't stack thinker runs — skip if one is already running
            if self._running.is_set():
                logger.debug("[Thinker] Previous cycle still running, skipping")
                return

            self._turn_counter += 1
            turn = self._turn_counter

            # Mark as running before releasing lock to prevent races
            self._running.set()

        # Build a short conversation summary for the goal tracker
        conv_summary = self._build_conversation_summary(
            user_message, assistant_response, conversation_history
        )

        thread = threading.Thread(
            target=self._run_cycle,
            args=(user_message, assistant_response, conv_summary, turn),
            daemon=True,
            name=f"thinker-cycle-{turn}",
        )
        thread.start()

    def get_talker_context(self) -> str:
        """Get the Thinker's latest output formatted for the Talker's system prompt.

        Returns empty string if no useful state or state is stale.
        """
        with self._lock:
            return self._state.to_system_context()

    def get_state(self) -> ThinkerState:
        """Get a copy of the current thinker state."""
        with self._lock:
            return ThinkerState(**asdict(self._state))

    def touch(self) -> None:
        """Reset staleness clock — call when a new user message arrives.

        This makes the TTL measure "time since last activity" instead of
        "time since thinker cycle completed", so a user pausing and
        resuming the same conversation keeps the thinker state alive.
        """
        with self._lock:
            self._state.touch()

    def is_running(self) -> bool:
        """Check if a thinker cycle is currently in progress."""
        return self._running.is_set()

    # ─── Internal ───────────────────────────────────────────────

    def _run_cycle(
        self,
        user_message: str,
        assistant_response: str,
        conv_summary: str,
        turn: int,
    ) -> None:
        """Execute one full Thinker cycle (3 threads in parallel)."""
        # _running is already set by run_async before thread launch
        start = time.time()

        try:
            # Run all 3 threads in parallel using threading
            results = {}
            errors = {}
            threads = []

            def _run_thread(name, prompt):
                try:
                    raw = self._brain._quick_generate(prompt, timeout=8)
                    results[name] = self._parse_json_response(raw)
                except Exception as e:
                    errors[name] = str(e)
                    logger.debug(f"[Thinker] {name} error: {e}")

            # Thread 1: Goal Tracking
            goal_prompt = GOAL_TRACKING_PROMPT.format(conversation=conv_summary[:1500])
            t1 = threading.Thread(target=_run_thread, args=("goal", goal_prompt), daemon=True)
            threads.append(t1)

            # Thread 2: Reasoning Audit
            audit_prompt = REASONING_AUDIT_PROMPT.format(
                user_message=user_message[:500],
                assistant_response=assistant_response[:800],
            )
            t2 = threading.Thread(target=_run_thread, args=("audit", audit_prompt), daemon=True)
            threads.append(t2)

            # Thread 3: Memory Integration
            memory_prompt = MEMORY_INTEGRATION_PROMPT.format(
                user_message=user_message[:500],
                assistant_response=assistant_response[:800],
            )
            t3 = threading.Thread(target=_run_thread, args=("memory", memory_prompt), daemon=True)
            threads.append(t3)

            # Start all
            for t in threads:
                t.start()

            # Wait for all (with timeout)
            for t in threads:
                t.join(timeout=12)

            # Build new state from results
            elapsed_ms = int((time.time() - start) * 1000)

            goal_data = results.get("goal", {})
            audit_data = results.get("audit", {})
            memory_data = results.get("memory", {})

            now = time.time()
            new_state = ThinkerState(
                goal_state=goal_data.get("goal_state", "")[:300],
                goal_progress=goal_data.get("goal_progress", "")[:300],
                reasoning_audit=audit_data.get("reasoning_audit", "")[:300],
                reasoning_gaps=audit_data.get("reasoning_gaps", "")[:300],
                memory_suggestions=memory_data.get("memory_suggestions", "")[:300],
                context_notes=memory_data.get("context_notes", "")[:300],
                timestamp=now,
                last_touched=now,
                turn_number=turn,
                processing_ms=elapsed_ms,
            )

            with self._lock:
                self._state = new_state

            # Persist state
            self._save_state()

            logger.debug(
                f"[Thinker] Cycle {turn} complete in {elapsed_ms}ms "
                f"(goal={bool(new_state.goal_state)}, "
                f"audit={bool(new_state.reasoning_audit)}, "
                f"memory={bool(new_state.memory_suggestions)})"
            )

        except Exception as e:
            logger.debug(f"[Thinker] Cycle {turn} failed: {e}")
        finally:
            self._running.clear()

    def _build_conversation_summary(
        self,
        user_message: str,
        assistant_response: str,
        conversation_history: list | None = None,
    ) -> str:
        """Build a short conversation summary for the goal tracker."""
        parts = []

        # Include last few exchanges from history
        if conversation_history:
            recent = conversation_history[-6:]  # Last 3 exchanges
            for msg in recent:
                role = msg.get("role", "?").upper()
                content = msg.get("content", "")[:200]
                if role == "SYSTEM":
                    continue
                parts.append(f"{role}: {content}")

        # Always include the current exchange
        parts.append(f"USER: {user_message[:300]}")
        parts.append(f"ASSISTANT: {assistant_response[:500]}")

        return "\n".join(parts)

    def _parse_json_response(self, raw: str) -> dict:
        """Parse JSON from LLM response, handling markdown fences and junk."""
        from aura.core.json_utils import parse_llm_json

        result = parse_llm_json(raw, default={})
        return result if isinstance(result, dict) else {}

    def _save_state(self) -> None:
        """Persist current state to disk atomically (tempfile + os.replace)."""
        import os
        import tempfile
        try:
            with self._lock:
                data = self._state.to_dict()
            content = json.dumps(data, indent=2)
            dir_ = self._state_file.parent
            dir_.mkdir(parents=True, exist_ok=True)
            fd, tmp_path = tempfile.mkstemp(dir=str(dir_), suffix=".tmp")
            try:
                os.write(fd, content.encode("utf-8"))
                os.close(fd)
                os.replace(tmp_path, str(self._state_file))
            except Exception:
                try:
                    os.close(fd)
                except OSError:
                    pass
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                raise
        except Exception as e:
            logger.debug(f"[Thinker] State save failed: {e}")

    def _load_state(self) -> None:
        """Load persisted state from disk."""
        try:
            if self._state_file.exists():
                data = json.loads(self._state_file.read_text(encoding="utf-8"))
                loaded = ThinkerState.from_dict(data)
                if not loaded.is_stale():
                    self._state = loaded
                    self._turn_counter = loaded.turn_number
                    logger.debug(f"[Thinker] Loaded state from turn {loaded.turn_number}")
                else:
                    logger.debug("[Thinker] Persisted state is stale, starting fresh")
        except Exception as e:
            logger.debug(f"[Thinker] State load failed: {e}")


# ─── Singleton ──────────────────────────────────────────────────

_thinker_instance: Optional[ThinkerEngine] = None
_thinker_lock = threading.Lock()


def get_thinker(brain=None) -> ThinkerEngine:
    """Get or create the global ThinkerEngine instance."""
    global _thinker_instance
    if _thinker_instance is None:
        with _thinker_lock:
            if _thinker_instance is None:
                _thinker_instance = ThinkerEngine(brain=brain)
    elif brain is not None and _thinker_instance._brain is None:
        _thinker_instance.set_brain(brain)
    return _thinker_instance
