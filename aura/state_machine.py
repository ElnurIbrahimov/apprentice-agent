"""Agent State Machine — lightweight observability wrapper around the agent loop.

Provides:
- Phase enum mapping the implicit OBSERVE/PLAN/ACT/EVALUATE/REMEMBER phases
- Validated transitions with hooks (on_enter / on_exit per phase)
- Per-phase timing statistics
- Bounded transition history (last 200)
- Singleton via get_agent_state_machine()
"""

import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Phase enum
# ---------------------------------------------------------------------------

class Phase(Enum):
    IDLE = "idle"
    OBSERVE = "observe"
    PLAN = "plan"
    ACT = "act"
    EVALUATE = "evaluate"
    REMEMBER = "remember"
    COMPLETED = "completed"
    ERROR = "error"


# Legal transitions
VALID_TRANSITIONS: Dict[Phase, List[Phase]] = {
    Phase.IDLE:      [Phase.OBSERVE],
    Phase.OBSERVE:   [Phase.PLAN, Phase.ERROR],
    Phase.PLAN:      [Phase.ACT, Phase.ERROR],
    Phase.ACT:       [Phase.EVALUATE, Phase.ERROR],
    Phase.EVALUATE:  [Phase.REMEMBER, Phase.ERROR],
    Phase.REMEMBER:  [Phase.OBSERVE, Phase.COMPLETED, Phase.ERROR],  # loop or done
    Phase.COMPLETED: [Phase.IDLE],
    Phase.ERROR:     [Phase.OBSERVE, Phase.IDLE, Phase.COMPLETED],   # recover or abort
}

MAX_HISTORY = 200


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PhaseTransition:
    from_phase: str
    to_phase: str
    timestamp: float
    iteration: int
    duration_ms: float  # time spent in from_phase
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "from": self.from_phase,
            "to": self.to_phase,
            "timestamp": self.timestamp,
            "iteration": self.iteration,
            "duration_ms": round(self.duration_ms, 2),
            "metadata": self.metadata,
        }


@dataclass
class PhaseTimings:
    count: int = 0
    total_ms: float = 0.0
    min_ms: float = float("inf")
    max_ms: float = 0.0

    @property
    def avg_ms(self) -> float:
        return self.total_ms / self.count if self.count else 0.0

    def record(self, duration_ms: float) -> None:
        self.count += 1
        self.total_ms += duration_ms
        if duration_ms < self.min_ms:
            self.min_ms = duration_ms
        if duration_ms > self.max_ms:
            self.max_ms = duration_ms

    def to_dict(self) -> Dict:
        return {
            "count": self.count,
            "total_ms": round(self.total_ms, 2),
            "avg_ms": round(self.avg_ms, 2),
            "min_ms": round(self.min_ms, 2) if self.count else 0.0,
            "max_ms": round(self.max_ms, 2),
        }


# ---------------------------------------------------------------------------
# AgentStateMachine
# ---------------------------------------------------------------------------

class AgentStateMachine:
    """Observability wrapper that tracks phase transitions and timing."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._phase = Phase.IDLE
        self._phase_start: float = time.time()
        self._goal: Optional[str] = None
        self._iteration: int = 0
        self._run_start: Optional[float] = None

        # History (bounded)
        self._transitions: List[PhaseTransition] = []

        # Timing stats per phase
        self._timings: Dict[Phase, PhaseTimings] = defaultdict(PhaseTimings)

        # Hooks: phase -> list of callbacks
        self._on_enter: Dict[Phase, List[Callable]] = defaultdict(list)
        self._on_exit: Dict[Phase, List[Callable]] = defaultdict(list)

    # -- Lifecycle ----------------------------------------------------------

    def start_run(self, goal: str, iteration: int = 0) -> None:
        """Begin an agent run: IDLE -> OBSERVE."""
        with self._lock:
            self._goal = goal
            self._iteration = iteration
            self._run_start = time.time()
        self.transition(Phase.OBSERVE)

    def end_run(self, completed: bool = True) -> None:
        """Finish an agent run -> COMPLETED -> IDLE."""
        target = Phase.COMPLETED if completed else Phase.ERROR
        self.transition(target, metadata={"completed": completed})
        self.transition(Phase.IDLE)

    # -- Transitions --------------------------------------------------------

    def transition(self, to_phase: Phase, metadata: Optional[Dict] = None) -> None:
        """Validate and record a phase transition."""
        with self._lock:
            from_phase = self._phase
            now = time.time()
            duration_ms = (now - self._phase_start) * 1000

            # Validate
            allowed = VALID_TRANSITIONS.get(from_phase, [])
            if to_phase not in allowed:
                logger.warning(
                    f"[StateMachine] Invalid transition {from_phase.value} -> {to_phase.value} "
                    f"(allowed: {[p.value for p in allowed]})"
                )
                return

            # Record timing for outgoing phase
            if from_phase != Phase.IDLE:
                self._timings[from_phase].record(duration_ms)

            # Record transition
            t = PhaseTransition(
                from_phase=from_phase.value,
                to_phase=to_phase.value,
                timestamp=now,
                iteration=self._iteration,
                duration_ms=duration_ms,
                metadata=metadata or {},
            )
            self._transitions.append(t)
            if len(self._transitions) > MAX_HISTORY:
                self._transitions = self._transitions[-MAX_HISTORY:]

            # Update state
            self._phase = to_phase
            self._phase_start = now

            # Capture hooks to fire outside lock
            exit_hooks = list(self._on_exit.get(from_phase, []))
            enter_hooks = list(self._on_enter.get(to_phase, []))

        # Fire hooks outside lock
        for hook in exit_hooks:
            try:
                hook(from_phase, to_phase, duration_ms, metadata or {})
            except Exception as e:
                logger.debug(f"[StateMachine] Exit hook error: {e}")

        for hook in enter_hooks:
            try:
                hook(from_phase, to_phase, duration_ms, metadata or {})
            except Exception as e:
                logger.debug(f"[StateMachine] Enter hook error: {e}")

    def set_iteration(self, iteration: int) -> None:
        """Update the current iteration counter."""
        with self._lock:
            self._iteration = iteration

    # -- Hooks --------------------------------------------------------------

    def on_enter(self, phase: Phase, callback: Callable) -> None:
        """Register a callback for when a phase is entered.

        Callback signature: (from_phase, to_phase, duration_ms, metadata)
        """
        if not callable(callback):
            raise TypeError(f"on_enter callback must be callable, got {type(callback)!r}")
        self._on_enter[phase].append(callback)

    def on_exit(self, phase: Phase, callback: Callable) -> None:
        """Register a callback for when a phase is exited.

        Callback signature: (from_phase, to_phase, duration_ms, metadata)
        """
        if not callable(callback):
            raise TypeError(f"on_exit callback must be callable, got {type(callback)!r}")
        self._on_exit[phase].append(callback)

    # -- Queries ------------------------------------------------------------

    def get_state(self) -> Dict:
        """Current state snapshot."""
        with self._lock:
            elapsed = (time.time() - self._run_start) if self._run_start else 0
            return {
                "phase": self._phase.value,
                "goal": self._goal,
                "iteration": self._iteration,
                "elapsed_seconds": round(elapsed, 2),
            }

    def get_timings(self) -> Dict[str, Dict]:
        """Per-phase timing statistics."""
        with self._lock:
            return {
                phase.value: self._timings[phase].to_dict()
                for phase in Phase
                if self._timings[phase].count > 0
            }

    def get_recent_transitions(self, limit: int = 20) -> List[Dict]:
        """Last N transitions."""
        with self._lock:
            return [t.to_dict() for t in self._transitions[-limit:]]

    def reset_stats(self) -> None:
        """Reset timing stats and history."""
        with self._lock:
            self._timings.clear()
            self._transitions.clear()


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_instance: Optional[AgentStateMachine] = None
_instance_lock = threading.Lock()


def get_agent_state_machine() -> AgentStateMachine:
    """Get or create the singleton AgentStateMachine."""
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = AgentStateMachine()
    return _instance
