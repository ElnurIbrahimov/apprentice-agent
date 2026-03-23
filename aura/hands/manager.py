"""HandManager — scheduling, lifecycle, and budget enforcement for Hands.

Integrates with Aura's consciousness stack:
- intrinsic_motivation drives influence Hand priority
- idle_presence triggers Hand activation during idle periods
- metacognition evaluates Hand performance
- audit_chain logs every Hand action
"""

import asyncio
import logging
import threading
import time
from datetime import datetime
from typing import Any, Optional

from aura.hands.base import Hand, HandState, HandResult

logger = logging.getLogger(__name__)


class HandManager:
    """Manages the lifecycle and scheduling of all Hands."""

    def __init__(self):
        self._hands: dict[str, Hand] = {}
        self._lock = threading.Lock()
        self._running = False
        self._scheduler_thread: Optional[threading.Thread] = None
        self._check_interval = 30  # seconds between schedule checks

    def register(self, hand: Hand):
        """Register a Hand with the manager."""
        with self._lock:
            name = hand.name
            if name in self._hands:
                logger.warning(f"[HandManager] Overwriting existing hand: {name}")
            self._hands[name] = hand
            logger.info(f"[HandManager] Registered hand: {name} (v{hand.manifest.version})")

    def activate(self, name: str) -> bool:
        """Activate a Hand (make it eligible for scheduling)."""
        with self._lock:
            hand = self._hands.get(name)
            if not hand:
                logger.error(f"[HandManager] Unknown hand: {name}")
                return False
            if hand.state == HandState.RUNNING:
                logger.warning(f"[HandManager] Hand {name} is already running")
                return False
            hand.state = HandState.ACTIVE
            return True

    def deactivate(self, name: str) -> bool:
        """Deactivate a Hand (stop scheduling it)."""
        with self._lock:
            hand = self._hands.get(name)
            if not hand:
                return False
            if hand.state == HandState.RUNNING:
                logger.warning(f"[HandManager] Hand {name} is running — will deactivate after completion")
                hand.state = HandState.PAUSED  # Will transition to INACTIVE after run
            else:
                hand.state = HandState.INACTIVE
            return True

    def pause(self, name: str) -> bool:
        """Pause a Hand."""
        with self._lock:
            hand = self._hands.get(name)
            if not hand:
                return False
            hand.state = HandState.PAUSED
            return True

    def list_hands(self) -> list[dict]:
        """List all registered Hands with their stats."""
        with self._lock:
            return [hand.get_stats() for hand in self._hands.values()]

    def get_hand(self, name: str) -> Optional[Hand]:
        with self._lock:
            return self._hands.get(name)

    async def run_hand(
        self,
        name: str,
        brain: Any,
        tools: dict,
        context: Optional[dict] = None,
    ) -> HandResult:
        """Execute a specific Hand immediately.

        This is the main execution path. Handles:
        - State transitions
        - Tool filtering (guardrails)
        - Budget enforcement
        - Audit logging
        - Metacognition recording
        """
        hand = self._hands.get(name)
        if not hand:
            return HandResult(hand_name=name, success=False, summary="", error=f"Unknown hand: {name}")

        if hand.state == HandState.RUNNING:
            return HandResult(hand_name=name, success=False, summary="", error="Hand is already running")

        # Filter tools by guardrails (base blocked + per-Hand extras)
        manifest = hand.manifest
        blocked = manifest.all_blocked_tools
        filtered_tools = {
            k: v for k, v in tools.items()
            if k not in blocked
        }

        # Build context
        run_context = context or {}
        run_context["manifest"] = manifest
        run_context["system_prompt"] = hand.get_system_prompt()
        run_context["max_iterations"] = manifest.max_iterations
        run_context["max_tokens"] = manifest.max_tokens
        run_context["model_preference"] = manifest.model_preference
        run_context["require_approval_for"] = manifest.require_approval_for

        # State transition
        hand.state = HandState.RUNNING
        start_time = time.time()

        try:
            # Audit entry
            try:
                from aura.security.audit_chain import get_audit_chain
                get_audit_chain().append(
                    action_type="hand_start",
                    action_data={"hand": name, "model": manifest.model_preference},
                    agent_id=f"hand:{name}",
                )
            except Exception:
                pass

            # Execute with duration enforcement
            try:
                result = await asyncio.wait_for(
                    hand.execute(brain, filtered_tools, run_context),
                    timeout=manifest.max_duration_seconds,
                )
            except asyncio.TimeoutError:
                elapsed = time.time() - start_time
                logger.warning(f"[HandManager] Hand {name} timed out after {elapsed:.1f}s (limit: {manifest.max_duration_seconds}s)")
                result = HandResult(
                    hand_name=name,
                    success=False,
                    summary=f"Hand timed out after {elapsed:.0f}s",
                    duration_seconds=elapsed,
                    error=f"Exceeded max_duration_seconds ({manifest.max_duration_seconds})",
                )
            result.duration_seconds = time.time() - start_time

            # Budget check (warn but don't block — the Hand already ran)
            if result.cost_usd > manifest.max_cost_usd:
                logger.warning(
                    f"[HandManager] Hand {name} exceeded cost budget: "
                    f"${result.cost_usd:.4f} > ${manifest.max_cost_usd:.4f}"
                )

            # Record stats
            hand.record_run(result)

            # Audit entry
            try:
                from aura.security.audit_chain import get_audit_chain
                get_audit_chain().append(
                    action_type="hand_complete",
                    action_data=result.to_dict(),
                    agent_id=f"hand:{name}",
                )
            except Exception:
                pass

            # Metacognition recording
            try:
                from aura.consciousness.metacognition import get_metacognition_engine
                mc = get_metacognition_engine()
                if mc:
                    mc.log_iteration({
                        "tool": f"hand:{name}",
                        "action": "autonomous_run",
                        "success": result.success,
                        "confidence": 0.8 if result.success else 0.3,
                        "duration_s": result.duration_seconds,
                    })
            except Exception:
                pass

            logger.info(
                f"[HandManager] Hand {name} completed: "
                f"{'SUCCESS' if result.success else 'FAILED'} "
                f"({result.iterations} iterations, {result.duration_seconds:.1f}s, "
                f"${result.cost_usd:.4f})"
            )
            return result

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[HandManager] Hand {name} crashed: {e}", exc_info=True)
            result = HandResult(
                hand_name=name,
                success=False,
                summary=f"Hand crashed: {e}",
                duration_seconds=elapsed,
                error=str(e),
            )
            hand.record_run(result)
            return result

        finally:
            # State transition
            if hand.state == HandState.PAUSED:
                hand.state = HandState.INACTIVE  # Deactivation was requested during run
            elif hand.state == HandState.RUNNING:
                hand.state = HandState.COOLDOWN

    def check_and_run(
        self,
        brain: Any,
        tools: dict,
        idle_seconds: float = 0,
        drive_urgencies: Optional[dict] = None,
        context: Optional[dict] = None,
    ) -> Optional[str]:
        """Check if any Hand should run and execute the highest-priority one.

        Called by idle_presence or the scheduler. Returns the name of the Hand
        that was triggered, or None.
        """
        # Find eligible Hands
        eligible = []
        with self._lock:
            for hand in self._hands.values():
                if hand.can_run(idle_seconds, drive_urgencies):
                    # Priority: drive-triggered > scheduled > cooldown
                    priority = 0
                    if hand.manifest.trigger_on_drive and drive_urgencies:
                        drive_val = drive_urgencies.get(hand.manifest.trigger_on_drive, 0.0)
                        if drive_val >= hand.manifest.trigger_drive_threshold:
                            priority = 100 + int(drive_val * 100)
                    else:
                        priority = 50  # Scheduled
                    eligible.append((priority, hand))

        if not eligible:
            return None

        # Run highest priority
        eligible.sort(key=lambda x: x[0], reverse=True)
        _, chosen = eligible[0]

        logger.info(f"[HandManager] Triggering hand: {chosen.name} (idle={idle_seconds:.0f}s)")

        # Run in a dedicated event loop on a background thread to avoid blocking
        def _run_async():
            try:
                asyncio.run(self.run_hand(chosen.name, brain, tools, context))
            except Exception as e:
                logger.error(f"[HandManager] Async execution error for {chosen.name}: {e}")

        thread = threading.Thread(target=_run_async, daemon=True, name=f"hand-run-{chosen.name}")
        thread.start()

        return chosen.name

    def start_scheduler(self, brain: Any, tools: dict, get_idle_seconds: callable,
                        get_drive_urgencies: callable):
        """Start the background scheduler thread."""
        if self._running:
            return

        self._running = True

        def _scheduler_loop():
            while self._running:
                try:
                    idle = get_idle_seconds()
                    drives = get_drive_urgencies()
                    self.check_and_run(brain, tools, idle, drives)
                except Exception as e:
                    logger.error(f"[HandManager] Scheduler error: {e}")
                time.sleep(self._check_interval)

        self._scheduler_thread = threading.Thread(
            target=_scheduler_loop,
            daemon=True,
            name="hand-scheduler",
        )
        self._scheduler_thread.start()
        logger.info("[HandManager] Scheduler started")

    def stop_scheduler(self):
        """Stop the background scheduler."""
        self._running = False
        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=5)
            self._scheduler_thread = None
        logger.info("[HandManager] Scheduler stopped")


# Global singleton
_manager: Optional[HandManager] = None
_manager_lock = threading.Lock()


def get_hand_manager() -> HandManager:
    global _manager
    if _manager is None:
        with _manager_lock:
            if _manager is None:
                _manager = HandManager()
    return _manager
