"""
Workflow Boundary Detection — Phase 5B.

Detects natural interruption points in user's workflow:
- File saves, git commits
- App/tab switches
- Idle periods (typing silence)
- Task completion signals

Based on CHI 2025 findings:
  52% engagement at workflow boundaries vs 38% dismissed mid-task.

Emits events when safe interruption opportunities are detected,
allowing the proactive suggestion engine to time interventions well.

Author: Aura Development Team
Created: 2026-02-07
"""

import logging
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any

from ..event_bus import Event, EventBus, EventPriority
from .base_monitor import BaseMonitor

logger = logging.getLogger(__name__)

# Perceptual hashing for visual change detection
try:
    import imagehash
    from PIL import Image
    IMAGEHASH_AVAILABLE = True
except ImportError:
    IMAGEHASH_AVAILABLE = False


class BoundaryType(Enum):
    """Types of workflow boundaries."""
    APP_SWITCH = "app_switch"           # User switched applications
    FILE_SAVE = "file_save"             # File was saved (detected via screen)
    IDLE_PAUSE = "idle_pause"           # Typing silence > threshold
    TASK_COMPLETE = "task_complete"     # Signals like closing a document/tab
    CONTEXT_SHIFT = "context_shift"    # Major content change on screen
    GIT_COMMIT = "git_commit"          # Git commit detected on screen


class FocusState(Enum):
    """Estimated user focus state."""
    DEEP_WORK = "deep_work"     # Actively coding/writing, don't interrupt
    SHALLOW = "shallow"         # Browsing, reading, light work
    TRANSITIONING = "transitioning"  # Between tasks, good time to suggest
    IDLE = "idle"               # Not active, safe to suggest


class WorkflowDetector(BaseMonitor):
    """
    Monitors user workflow to detect natural interruption boundaries.

    Uses screen context (via Screenpipe) and timing heuristics to
    identify when the user is between tasks and receptive to suggestions.
    """

    # Timing thresholds
    IDLE_THRESHOLD_SECONDS = 5.0       # 5s silence = micro-boundary
    DEEP_WORK_MIN_SECONDS = 120.0      # 2min in same app = deep work
    APP_SWITCH_COOLDOWN = 3.0          # Don't re-fire for rapid switches
    BOUNDARY_COOLDOWN = 15.0           # Min seconds between boundary events

    @property
    def source(self) -> str:
        return "workflow"

    def __init__(
        self,
        event_bus: Optional[EventBus] = None,
        poll_interval: float = 3.0,
    ):
        super().__init__(event_bus=event_bus, poll_interval=poll_interval)

        # Detect headless: no display on Linux = nothing to monitor
        import sys as _sys, os as _os
        self._headless = bool(_os.environ.get("AURA_HEADLESS")) or (
            _sys.platform != "win32" and not _os.environ.get("DISPLAY")
        )
        if self._headless:
            logger.info("[WorkflowDetector] Headless mode detected -- polling disabled")

        # State tracking
        self._current_app: str = ""
        self._current_window: str = ""
        self._last_app_change: float = time.time()
        self._last_activity: float = time.time()
        self._last_boundary_event: float = 0
        self._last_content_hash: str = ""
        self._focus_state: FocusState = FocusState.SHALLOW

        # App dwell times (how long in each app)
        self._app_dwell_start: float = time.time()
        self._app_history: List[Dict[str, Any]] = []  # Recent app switches

        # Perceptual hash for visual change detection
        self._last_visual_hash = None  # imagehash.ImageHash or None

        # Boundary scoring
        self._boundary_score: float = 0.0  # 0-1, higher = better time to interrupt

        # Deep work detection
        self._deep_work_apps = {
            "code", "vs code", "visual studio", "pycharm", "intellij",
            "vim", "neovim", "emacs", "sublime", "atom",
            "word", "docs", "notion", "obsidian",
            "terminal", "cmd", "powershell", "wezterm", "iterm",
        }

    async def _poll(self) -> List[Event]:
        """Poll screen state to detect workflow boundaries."""
        if self._headless:
            return []  # Nothing to monitor without a display

        events = []

        try:
            screen_data = self._get_screen_state()
        except Exception as e:
            logger.debug(f"[WorkflowDetector] Screen state unavailable: {e}")
            return events

        if not screen_data or not screen_data.get("available"):
            return events

        now = time.time()
        new_app = screen_data.get("current_app", "")
        new_window = screen_data.get("current_window", "")

        # === Detect app switch ===
        if new_app and new_app != self._current_app and self._current_app:
            dwell_time = now - self._app_dwell_start

            # Record in history
            self._app_history.append({
                "from_app": self._current_app,
                "to_app": new_app,
                "dwell_seconds": dwell_time,
                "timestamp": now,
            })
            # Keep last 20 switches
            self._app_history = self._app_history[-20:]

            # Emit app switch boundary if cooldown passed
            if now - self._last_app_change > self.APP_SWITCH_COOLDOWN:
                if self._can_emit_boundary(now):
                    events.append(self.create_event(
                        event_type="boundary_detected",
                        payload={
                            "boundary_type": BoundaryType.APP_SWITCH.value,
                            "from_app": self._current_app,
                            "to_app": new_app,
                            "dwell_seconds": dwell_time,
                            "boundary_score": self._compute_boundary_score(
                                BoundaryType.APP_SWITCH, dwell_time
                            ),
                        },
                        priority=EventPriority.LOW,
                    ))
                    self._last_boundary_event = now

            self._current_app = new_app
            self._last_app_change = now
            self._app_dwell_start = now

        elif new_app:
            self._current_app = new_app

        self._current_window = new_window

        # === Detect content changes (context shift) ===
        content_hash = screen_data.get("content_hash", "")
        if content_hash and content_hash != self._last_content_hash:
            self._last_activity = now
            self._last_content_hash = content_hash

        # === Detect visual changes via perceptual hashing ===
        if IMAGEHASH_AVAILABLE:
            screenshot_path = screen_data.get("screenshot_path", "")
            if screenshot_path:
                try:
                    img = Image.open(screenshot_path)
                    new_visual_hash = imagehash.dhash(img, hash_size=16)
                    if self._last_visual_hash is not None:
                        visual_distance = new_visual_hash - self._last_visual_hash
                        from aura.config import Config
                        if visual_distance > Config.PHASH_MAJOR_THRESHOLD:
                            # Rapid large visual change → likely transitioning
                            self._focus_state = FocusState.TRANSITIONING
                            self._last_activity = now
                    self._last_visual_hash = new_visual_hash
                except Exception:
                    pass

        # === Detect idle pause (typing silence) ===
        silence_seconds = now - self._last_activity
        if (silence_seconds >= self.IDLE_THRESHOLD_SECONDS
                and self._focus_state != FocusState.IDLE):
            if self._can_emit_boundary(now):
                events.append(self.create_event(
                    event_type="boundary_detected",
                    payload={
                        "boundary_type": BoundaryType.IDLE_PAUSE.value,
                        "silence_seconds": silence_seconds,
                        "current_app": self._current_app,
                        "boundary_score": self._compute_boundary_score(
                            BoundaryType.IDLE_PAUSE, silence_seconds
                        ),
                    },
                    priority=EventPriority.BACKGROUND,
                ))
                self._last_boundary_event = now

        # === Detect screen signals (file save, git commit) ===
        screen_text = screen_data.get("recent_text", "").lower()
        if screen_text:
            # Git commit detection
            if any(sig in screen_text for sig in [
                "committed", "commit successful", "changes committed",
                "git push", "pushed to", "[main", "[master",
            ]):
                if self._can_emit_boundary(now):
                    events.append(self.create_event(
                        event_type="boundary_detected",
                        payload={
                            "boundary_type": BoundaryType.GIT_COMMIT.value,
                            "current_app": self._current_app,
                            "boundary_score": 0.85,
                        },
                        priority=EventPriority.LOW,
                    ))
                    self._last_boundary_event = now

        # === Update focus state ===
        self._update_focus_state(now)

        return events

    def _get_screen_state(self) -> Optional[Dict[str, Any]]:
        """Get current screen state from Screenpipe."""
        try:
            from aura.tools.screenpipe import get_screenpipe_client
            client = get_screenpipe_client()
            if not client.is_available():
                return None
            return client.get_screen_context_filtered(minutes=1, max_chars=500)
        except Exception:
            return None

    def _can_emit_boundary(self, now: float) -> bool:
        """Check if we can emit a boundary event (cooldown)."""
        return (now - self._last_boundary_event) >= self.BOUNDARY_COOLDOWN

    def _compute_boundary_score(
        self, boundary_type: BoundaryType, magnitude: float
    ) -> float:
        """
        Compute an interruption-opportunity score (0-1).

        Higher = better time to interrupt.
        Based on boundary type and context.
        """
        base_scores = {
            BoundaryType.APP_SWITCH: 0.6,
            BoundaryType.FILE_SAVE: 0.5,
            BoundaryType.IDLE_PAUSE: 0.7,
            BoundaryType.TASK_COMPLETE: 0.9,
            BoundaryType.CONTEXT_SHIFT: 0.4,
            BoundaryType.GIT_COMMIT: 0.85,
        }

        score = base_scores.get(boundary_type, 0.5)

        # Adjust based on magnitude
        if boundary_type == BoundaryType.APP_SWITCH:
            # Longer dwell before switch = more likely a real task boundary
            if magnitude > 300:  # > 5 min in previous app
                score += 0.15
            elif magnitude < 10:  # Quick switch, probably not a boundary
                score -= 0.2

        elif boundary_type == BoundaryType.IDLE_PAUSE:
            # Longer silence = more definitive boundary
            if magnitude > 30:
                score += 0.15
            elif magnitude < 8:
                score -= 0.1

        # Penalize if in deep work
        if self._focus_state == FocusState.DEEP_WORK:
            score -= 0.3

        # Boost if already transitioning
        if self._focus_state == FocusState.TRANSITIONING:
            score += 0.15

        return max(0.0, min(1.0, score))

    def _update_focus_state(self, now: float):
        """Update estimated user focus state."""
        silence = now - self._last_activity
        dwell = now - self._app_dwell_start
        app_lower = self._current_app.lower()

        # Idle if no activity for a while
        if silence > 60:
            self._focus_state = FocusState.IDLE
            return

        # Deep work if in a coding/writing app for > threshold
        is_deep_app = any(da in app_lower for da in self._deep_work_apps)
        if is_deep_app and dwell > self.DEEP_WORK_MIN_SECONDS:
            self._focus_state = FocusState.DEEP_WORK
            return

        # Transitioning if recently switched apps
        if now - self._last_app_change < 10:
            self._focus_state = FocusState.TRANSITIONING
            return

        # Default: shallow work
        self._focus_state = FocusState.SHALLOW

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def get_focus_state(self) -> Dict[str, Any]:
        """Get current estimated focus state."""
        now = time.time()
        return {
            "focus_state": self._focus_state.value,
            "current_app": self._current_app,
            "current_window": self._current_window,
            "dwell_seconds": now - self._app_dwell_start,
            "silence_seconds": now - self._last_activity,
            "boundary_score": self._boundary_score,
            "recent_switches": len(self._app_history),
            "is_interruptible": self._focus_state in (
                FocusState.SHALLOW, FocusState.TRANSITIONING, FocusState.IDLE
            ),
        }

    def should_interrupt(self, importance: float = 0.5) -> bool:
        """
        Check if it's a good time to interrupt the user.

        Args:
            importance: How important the interruption is (0-1).
                       Higher importance = interrupt even during focus.

        Returns:
            True if it's okay to interrupt.
        """
        if self._focus_state == FocusState.IDLE:
            return True
        if self._focus_state == FocusState.TRANSITIONING:
            return importance >= 0.3
        if self._focus_state == FocusState.SHALLOW:
            return importance >= 0.5
        if self._focus_state == FocusState.DEEP_WORK:
            return importance >= 0.85  # Only interrupt for critical things
        return False


# Singleton
_detector_instance: Optional[WorkflowDetector] = None


def get_workflow_detector(event_bus: Optional[EventBus] = None) -> WorkflowDetector:
    """Get or create the workflow detector singleton."""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = WorkflowDetector(event_bus=event_bus)
    return _detector_instance
