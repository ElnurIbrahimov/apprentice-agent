"""
Loop Guard — Phase 1.

Prevents AURA from getting stuck in reasoning/tool/memory cycles.

Algorithm:
1. Maintain a sliding window of recent (action, argument_fingerprint) tuples
   per session.
2. Detect cycles: same (action, fingerprint) appearing ≥ LOOP_GUARD_MAX_REPETITIONS
   times within the window.
3. Novelty scoring: hash the last N intermediate artifacts; if entropy is below
   LOOP_GUARD_NOVELTY_THRESHOLD the session is flagged as low-novelty.
4. On trigger: emit telemetry, return a stop signal + compact fallback message.

Thread-safe — one LoopGuard instance per session is typical, but the global
registry is thread-safe for shared access.

Author: Aura reliability upgrade (2026-03)
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config defaults (can be overridden by Config class)
# ---------------------------------------------------------------------------

_DEFAULT_MAX_REPETITIONS  = 3
_DEFAULT_NOVELTY_THRESHOLD = 0.25   # below this → low novelty
_DEFAULT_WINDOW_SIZE       = 20     # number of recent actions to examine
_DEFAULT_BUDGET            = 40     # hard max actions per session


def _load_config() -> Tuple[bool, int, float, int, int]:
    try:
        from aura.config import Config
        enabled = getattr(Config, "ENABLE_LOOP_GUARD", True)
        max_rep = getattr(Config, "LOOP_GUARD_MAX_REPETITIONS", _DEFAULT_MAX_REPETITIONS)
        nov_thr = getattr(Config, "LOOP_GUARD_NOVELTY_THRESHOLD", _DEFAULT_NOVELTY_THRESHOLD)
        window  = getattr(Config, "LOOP_GUARD_WINDOW_SIZE", _DEFAULT_WINDOW_SIZE)
        budget  = getattr(Config, "LOOP_GUARD_BUDGET", _DEFAULT_BUDGET)
        return enabled, max_rep, nov_thr, window, budget
    except Exception:
        return True, _DEFAULT_MAX_REPETITIONS, _DEFAULT_NOVELTY_THRESHOLD, _DEFAULT_WINDOW_SIZE, _DEFAULT_BUDGET


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class ActionRecord:
    action: str           # e.g. "search_memory", "browser_click", "llm_call"
    fingerprint: str      # short hash of argument/context
    ts: float = field(default_factory=time.time)


@dataclass
class LoopGuardResult:
    triggered: bool
    reason: str = ""
    repetitions: int = 0
    novelty_score: float = 1.0
    fallback_message: str = ""
    actions_taken: int = 0


# ---------------------------------------------------------------------------
# Per-session guard
# ---------------------------------------------------------------------------

class SessionLoopGuard:
    """
    Tracks actions within one session and raises loop alarms.
    """

    FALLBACK_MESSAGE = (
        "I've noticed I'm going in circles. "
        "Let me give you my best current answer without further searching: "
    )

    def __init__(self, session_id: str = "") -> None:
        self.session_id = session_id
        enabled, max_rep, nov_thr, window, budget = _load_config()
        self._enabled      = enabled
        self._max_rep      = max_rep
        self._nov_thr      = nov_thr
        self._window       = window
        self._budget       = budget
        self._history: Deque[ActionRecord] = deque(maxlen=window)
        self._total_actions = 0
        self._triggered     = False
        self._trigger_reason = ""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(self, action: str, context: Any = "") -> LoopGuardResult:
        """
        Record an action and check for loops.

        Args:
            action:  Name of the action (e.g. "search_memory").
            context: Relevant argument / query string (used for fingerprinting).

        Returns:
            LoopGuardResult — check .triggered before proceeding.
        """
        if not self._enabled:
            return LoopGuardResult(triggered=False)

        fingerprint = _fingerprint(context)
        record = ActionRecord(action=action, fingerprint=fingerprint)
        self._history.append(record)
        self._total_actions += 1

        # Hard budget
        if self._total_actions >= self._budget:
            return self._trigger("budget_exhausted",
                                  f"Exceeded {self._budget}-action budget")

        # Repetition check
        rep_count = sum(
            1 for r in self._history
            if r.action == action and r.fingerprint == fingerprint
        )
        if rep_count >= self._max_rep:
            return self._trigger("repeated_action",
                                  f"Action '{action}' repeated {rep_count}× with same fingerprint")

        # Low novelty check
        novelty = self._compute_novelty()
        if novelty < self._nov_thr:
            return self._trigger("low_novelty",
                                  f"Novelty score {novelty:.2f} below threshold {self._nov_thr}")

        return LoopGuardResult(
            triggered=False,
            novelty_score=novelty,
            actions_taken=self._total_actions,
        )

    def reset(self) -> None:
        """Clear state (e.g., when a new user message arrives)."""
        self._history.clear()
        self._total_actions = 0
        self._triggered = False
        self._trigger_reason = ""

    @property
    def actions_taken(self) -> int:
        return self._total_actions

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _trigger(self, reason: str, detail: str) -> LoopGuardResult:
        if not self._triggered:
            logger.warning(
                "[LoopGuard] TRIGGERED session=%s reason=%s actions=%d detail=%s",
                self.session_id, reason, self._total_actions, detail,
            )
            self._triggered = True
            self._trigger_reason = reason
            # Emit telemetry
            try:
                from aura.reliability.telemetry import TelemetryKind, emit
                emit(
                    TelemetryKind.LOOP_GUARD,
                    session_id=self.session_id,
                    success=False,
                    loop_warnings=1,
                    extra={"reason": reason, "detail": detail,
                           "actions": self._total_actions},
                )
            except Exception:
                pass

        return LoopGuardResult(
            triggered=True,
            reason=reason,
            repetitions=self._total_actions,
            novelty_score=self._compute_novelty(),
            fallback_message=self.FALLBACK_MESSAGE,
            actions_taken=self._total_actions,
        )

    def _compute_novelty(self) -> float:
        """
        Measure information entropy of recent fingerprints.
        Returns 0.0 (all identical) to 1.0 (all unique).
        """
        if not self._history:
            return 1.0
        fps = [r.action + ":" + r.fingerprint for r in self._history]
        unique = len(set(fps))
        return unique / len(fps)


# ---------------------------------------------------------------------------
# Global session registry
# ---------------------------------------------------------------------------

import threading as _threading

_guards: Dict[str, SessionLoopGuard] = {}
_guards_lock = _threading.Lock()


def get_guard(session_id: str) -> SessionLoopGuard:
    """Get or create a per-session loop guard."""
    with _guards_lock:
        if session_id not in _guards:
            _guards[session_id] = SessionLoopGuard(session_id=session_id)
        return _guards[session_id]


def reset_guard(session_id: str) -> None:
    """Reset the guard for a session (call at start of each user message)."""
    with _guards_lock:
        if session_id in _guards:
            _guards[session_id].reset()


def purge_old_guards(max_age_s: float = 3600.0) -> int:
    """Remove guards for sessions idle > max_age_s. Returns count removed."""
    now = time.time()
    with _guards_lock:
        to_remove = [
            sid for sid, g in _guards.items()
            if (not g._history or now - g._history[-1].ts > max_age_s)
        ]
        for sid in to_remove:
            del _guards[sid]
    return len(to_remove)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _fingerprint(obj: Any, length: int = 8) -> str:
    """Create a short stable fingerprint of an arbitrary object."""
    try:
        text = str(obj)[:200]
    except Exception:
        text = ""
    return hashlib.md5(text.encode()).hexdigest()[:length]


__all__ = [
    "ActionRecord",
    "LoopGuardResult",
    "SessionLoopGuard",
    "get_guard",
    "purge_old_guards",
    "reset_guard",
]
