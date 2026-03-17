"""System 1/2 Explicit Switching — user-controllable thinking mode with cognitive load tracking.

Provides:
- ThinkingMode enum (AUTO / SYSTEM1 / SYSTEM2)
- CognitiveLoadState: rolling-window tracker for query complexity and escalation history
- ThinkingModeManager: thread-safe singleton that wraps the automatic S1/S2 decision
  with explicit user overrides and cognitive-load based escalation.
"""

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ThinkingMode(Enum):
    """User-selectable thinking mode."""
    AUTO = "auto"       # Default: let the brain decide
    SYSTEM1 = "system1" # Force fast/intuitive
    SYSTEM2 = "system2" # Force deliberative/reasoning


# ---------------------------------------------------------------------------
# Cognitive Load Tracker
# ---------------------------------------------------------------------------

COGNITIVE_LOAD_WINDOW = 10  # Rolling window size

# Thresholds for load-based escalation
LOAD_HIGH_THRESHOLD = 0.70   # Above this -> suggest S2
LOAD_LOW_THRESHOLD = 0.30    # Below this -> suggest S1


@dataclass
class CognitiveLoadState:
    """Tracks recent query complexity to inform S1/S2 decisions."""

    _confidences: deque = field(default_factory=lambda: deque(maxlen=COGNITIVE_LOAD_WINDOW))
    _escalations: deque = field(default_factory=lambda: deque(maxlen=COGNITIVE_LOAD_WINDOW))
    _complexities: deque = field(default_factory=lambda: deque(maxlen=COGNITIVE_LOAD_WINDOW))
    _timestamps: deque = field(default_factory=lambda: deque(maxlen=COGNITIVE_LOAD_WINDOW))

    def record_query(self, confidence: float, was_complex: bool, escalated: bool) -> None:
        """Record one query outcome into the rolling window."""
        self._confidences.append(confidence)
        self._escalations.append(escalated)
        self._complexities.append(was_complex)
        self._timestamps.append(time.time())

    @property
    def load_score(self) -> float:
        """Weighted combination: inverse-confidence (0.4) + escalation ratio (0.35) + complexity ratio (0.25).

        Returns a value in [0, 1] where higher = heavier cognitive load.
        """
        n = len(self._confidences)
        if n == 0:
            return 0.0

        avg_inv_conf = 1.0 - (sum(self._confidences) / n)
        escalation_ratio = sum(self._escalations) / n
        complexity_ratio = sum(self._complexities) / n

        return 0.40 * avg_inv_conf + 0.35 * escalation_ratio + 0.25 * complexity_ratio

    def should_escalate_from_load(self) -> Optional[bool]:
        """Return True (suggest S2), False (suggest S1), or None (no opinion)."""
        if len(self._confidences) < 3:
            return None  # Not enough data
        score = self.load_score
        if score >= LOAD_HIGH_THRESHOLD:
            return True
        if score <= LOAD_LOW_THRESHOLD:
            return False
        return None

    def reset(self) -> None:
        """Clear all tracked state."""
        self._confidences.clear()
        self._escalations.clear()
        self._complexities.clear()
        self._timestamps.clear()

    def to_dict(self) -> Dict:
        """Serialize for API responses."""
        return {
            "load_score": round(self.load_score, 3),
            "window_size": len(self._confidences),
            "suggestion": {
                True: "system2",
                False: "system1",
                None: "none",
            }[self.should_escalate_from_load()],
        }


# ---------------------------------------------------------------------------
# ThinkingModeManager (singleton)
# ---------------------------------------------------------------------------

_instance: Optional["ThinkingModeManager"] = None
_instance_lock = threading.Lock()


class ThinkingModeManager:
    """Thread-safe manager for explicit thinking-mode control."""

    def __init__(self) -> None:
        self._mode = ThinkingMode.AUTO
        self._lock = threading.Lock()
        self._callbacks: List[Callable[[ThinkingMode, ThinkingMode], None]] = []
        self.cognitive_load = CognitiveLoadState()

    # -- Mode control -------------------------------------------------------

    @property
    def mode(self) -> ThinkingMode:
        with self._lock:
            return self._mode

    @mode.setter
    def mode(self, new_mode: ThinkingMode) -> None:
        with self._lock:
            old = self._mode
            self._mode = new_mode
            cbs = list(self._callbacks)
        if old != new_mode:
            logger.info(f"[ThinkingMode] Changed: {old.value} -> {new_mode.value}")
            for cb in cbs:
                try:
                    cb(old, new_mode)
                except Exception:
                    pass

    def on_change(self, callback: Callable[[ThinkingMode, ThinkingMode], None]) -> None:
        """Register a callback fired on mode change (old, new)."""
        self._callbacks.append(callback)

    # -- Decision wrapper ---------------------------------------------------

    def get_effective_decision(self, auto_decision: bool) -> Tuple[bool, str]:
        """Wrap the automatic S1/S2 decision with user override + cognitive load.

        Args:
            auto_decision: True if automatic logic says S2, False for S1.

        Returns:
            (use_system2: bool, reason: str)
        """
        current = self.mode

        if current == ThinkingMode.SYSTEM1:
            return (False, "explicit_system1")
        if current == ThinkingMode.SYSTEM2:
            return (True, "explicit_system2")

        # AUTO mode: check cognitive load as tie-breaker
        load_suggestion = self.cognitive_load.should_escalate_from_load()
        if load_suggestion is True and not auto_decision:
            return (True, "cognitive_load_escalation")
        if load_suggestion is False and auto_decision:
            # Don't override a genuine S2 decision downward
            pass

        reason = "auto_system2" if auto_decision else "auto_system1"
        return (auto_decision, reason)

    # -- Serialization ------------------------------------------------------

    def get_state(self) -> Dict:
        """Full state dict for API."""
        return {
            "mode": self.mode.value,
            "cognitive_load": self.cognitive_load.to_dict(),
        }


def get_thinking_mode_manager() -> ThinkingModeManager:
    """Get or create the singleton ThinkingModeManager."""
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = ThinkingModeManager()
    return _instance
