"""Hand base class — the autonomous task unit.

Each Hand is a self-contained package:
- Manifest: name, schedule, triggers, resources, guardrails
- System prompt: domain expertise baked in
- Lifecycle: inactive → active → running → paused → inactive
- Budget: token/cost limits, max duration
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class HandState(Enum):
    INACTIVE = "inactive"      # Not activated
    ACTIVE = "active"          # Activated, waiting for schedule/trigger
    RUNNING = "running"        # Currently executing
    PAUSED = "paused"          # Temporarily suspended
    COOLDOWN = "cooldown"      # Finished, waiting before next run
    ERROR = "error"            # Failed, needs attention


@dataclass
class HandManifest:
    """Configuration manifest for a Hand (equivalent to OpenFang's HAND.toml)."""
    name: str
    version: str = "0.1.0"
    description: str = ""

    # Schedule
    cron: Optional[str] = None          # Cron expression (e.g., "0 */4 * * *")
    interval_minutes: Optional[int] = None  # Simple interval alternative
    idle_only: bool = True              # Only run when user is idle
    min_idle_seconds: int = 300         # Minimum idle time before triggering

    # Resources
    max_tokens: int = 50000             # Token budget per run
    max_cost_usd: float = 0.50         # Cost cap per run
    max_duration_seconds: int = 1800    # 30 min max
    model_preference: str = "reasoning" # "fast", "reasoning", "code", "vision"

    # Guardrails
    require_approval_for: list[str] = field(default_factory=lambda: ["publish", "send_message", "write_file"])
    blocked_tools: list[str] = field(default_factory=lambda: ["shell", "deploy_tool"])
    extra_blocked_tools: list[str] = field(default_factory=list)  # Extends blocked_tools per-Hand
    max_iterations: int = 10           # ReAct loop cap

    @property
    def all_blocked_tools(self) -> set[str]:
        """Combined blocked_tools + extra_blocked_tools."""
        return set(self.blocked_tools) | set(self.extra_blocked_tools)

    # Triggers (beyond schedule)
    trigger_on_drive: Optional[str] = None   # Intrinsic motivation drive (curiosity, competence, social, coherence)
    trigger_drive_threshold: float = 0.7     # Minimum drive urgency to trigger


@dataclass
class HandResult:
    """Result of a Hand execution."""
    hand_name: str
    success: bool
    summary: str                        # Human-readable summary of what was done
    iterations: int = 0
    tokens_used: int = 0
    cost_usd: float = 0.0
    duration_seconds: float = 0.0
    artifacts: list[dict] = field(default_factory=list)  # Files created, memories stored, etc.
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "hand": self.hand_name,
            "success": self.success,
            "summary": self.summary,
            "iterations": self.iterations,
            "tokens_used": self.tokens_used,
            "cost_usd": round(self.cost_usd, 4),
            "duration_seconds": round(self.duration_seconds, 1),
            "artifacts": self.artifacts,
            **({"error": self.error} if self.error else {}),
        }


class Hand(ABC):
    """Base class for autonomous Hands.

    Subclasses implement `execute()` with their domain logic.
    The HandManager handles scheduling, lifecycle, and budget enforcement.
    """

    def __init__(self):
        self._manifest: Optional[HandManifest] = None
        self._state = HandState.INACTIVE
        self._last_run: float = 0.0
        self._total_runs: int = 0
        self._total_tokens: int = 0
        self._total_cost: float = 0.0
        self._consecutive_failures: int = 0
        self._last_error: Optional[str] = None

    @abstractmethod
    def get_manifest(self) -> HandManifest:
        """Return this Hand's manifest."""
        ...

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Return the domain-specific system prompt for this Hand."""
        ...

    @abstractmethod
    async def execute(self, brain: Any, tools: dict, context: dict) -> HandResult:
        """Execute this Hand's autonomous task.

        Args:
            brain: OllamaBrain instance for LLM calls
            tools: Available tool instances (filtered by guardrails)
            context: Runtime context (memory, world model state, drives, etc.)

        Returns:
            HandResult with execution summary
        """
        ...

    @property
    def manifest(self) -> HandManifest:
        if self._manifest is None:
            self._manifest = self.get_manifest()
        return self._manifest

    @property
    def name(self) -> str:
        return self.manifest.name

    @property
    def state(self) -> HandState:
        return self._state

    @state.setter
    def state(self, new_state: HandState):
        old = self._state
        self._state = new_state
        if old != new_state:
            logger.info(f"[Hand:{self.name}] {old.value} → {new_state.value}")

    def can_run(self, idle_seconds: float = 0, drive_urgencies: Optional[dict] = None) -> bool:
        """Check if this Hand should run right now."""
        if self._state not in (HandState.ACTIVE, HandState.COOLDOWN):
            return False

        # Check idle requirement
        if self.manifest.idle_only and idle_seconds < self.manifest.min_idle_seconds:
            return False

        # Check drive trigger
        if self.manifest.trigger_on_drive and drive_urgencies:
            drive = drive_urgencies.get(self.manifest.trigger_on_drive, 0.0)
            if drive >= self.manifest.trigger_drive_threshold:
                return True

        # Check cooldown (don't run too frequently)
        if self._last_run > 0:
            elapsed = time.time() - self._last_run
            min_interval = (self.manifest.interval_minutes or 60) * 60
            if elapsed < min_interval:
                return False

        # Circuit breaker: 3 consecutive failures = wait longer
        if self._consecutive_failures >= 3:
            cooldown = min(3600, 300 * (2 ** (self._consecutive_failures - 3)))
            if time.time() - self._last_run < cooldown:
                return False

        return True

    def record_run(self, result: HandResult):
        """Update stats after a run."""
        self._last_run = time.time()
        self._total_runs += 1
        self._total_tokens += result.tokens_used
        self._total_cost += result.cost_usd

        if result.success:
            self._consecutive_failures = 0
            self._last_error = None
        else:
            self._consecutive_failures += 1
            self._last_error = result.error

    def get_stats(self) -> dict:
        from datetime import datetime
        return {
            "name": self.name,
            "description": self.manifest.description,
            "state": self._state.value,
            "total_runs": self._total_runs,
            "total_tokens": self._total_tokens,
            "total_cost": round(self._total_cost, 4),
            "consecutive_failures": self._consecutive_failures,
            "last_run": datetime.fromtimestamp(self._last_run).isoformat() if self._last_run > 0 else None,
            "last_run_ts": self._last_run,
            "last_error": self._last_error,
            "model_preference": self.manifest.model_preference,
            "idle_only": self.manifest.idle_only,
            "trigger_on_drive": self.manifest.trigger_on_drive,
        }
