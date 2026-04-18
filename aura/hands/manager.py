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
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from aura.hands.base import Hand, HandResult, HandState

logger = logging.getLogger(__name__)


@dataclass
class ApprovalRequest:
    """A pending approval request from a Hand."""
    request_id: str
    hand_name: str
    tool_name: str
    args: dict
    timestamp: float = field(default_factory=time.time)
    resolved: bool = False
    approved: bool = False


class HandManager:
    """Manages the lifecycle and scheduling of all Hands."""

    MAX_CONCURRENT_HANDS = 3  # Cap concurrent hand execution threads

    def __init__(self):
        self._hands: dict[str, Hand] = {}
        self._lock = threading.Lock()
        self._running = False
        self._scheduler_thread: Optional[threading.Thread] = None
        self._check_interval = 30  # seconds between schedule checks
        self._active_threads: dict[str, threading.Thread] = {}
        self._threads_lock = threading.Lock()

        # Notification callback (set by API/Telegram to receive results)
        self._notify_callback: Optional[Callable[[HandResult], None]] = None

        # Approval workflow. Each entry is (asyncio.Event, owning_loop), produced
        # by request_approval_async and consumed by resolve_approval.
        self._pending_approvals: dict[str, ApprovalRequest] = {}
        self._approval_events: dict[str, tuple[asyncio.Event, asyncio.AbstractEventLoop]] = {}
        self._approval_lock = threading.Lock()

        # Event loop reference for thread-safe broadcast (set by API at startup)
        self._approval_loop: Optional[asyncio.AbstractEventLoop] = None

        # Hand-to-hand trigger support
        self._event_bus = None
        self._pending_triggers: dict[str, dict] = {}
        self._triggers_lock = threading.Lock()

        # Mid-execution command support
        self._pending_commands: dict[str, dict] = {}
        self._commands_lock = threading.Lock()

        # Snooze state: hand_name -> unix timestamp until which scheduler
        # and trigger_hand_async should skip this hand. Set via user action
        # on a proactive card (e.g. "⏰ Snooze 1h").
        self._snoozed_until: dict[str, float] = {}
        self._snooze_lock = threading.Lock()

    def register(self, hand: Hand):
        """Register a Hand with the manager."""
        with self._lock:
            name = hand.name
            if name in self._hands:
                logger.warning(f"[HandManager] Overwriting existing hand: {name}")
            # Recovery: if hand is stuck in RUNNING from a prior crash, reset it
            if hand.state == HandState.RUNNING:
                logger.warning(f"[HandManager] Hand {name} stuck in RUNNING (stale), resetting to COOLDOWN")
                hand._state = HandState.COOLDOWN
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

    def snooze(self, name: str, seconds: int) -> bool:
        """Snooze a Hand for N seconds. check_and_run and trigger_hand_async
        will skip it until time.time() exceeds the snooze deadline.

        Returns True if the hand exists, False otherwise."""
        if seconds <= 0:
            return False
        with self._lock:
            if name not in self._hands:
                return False
        with self._snooze_lock:
            self._snoozed_until[name] = time.time() + seconds
        logger.info(f"[HandManager] Snoozed '{name}' for {seconds}s")
        return True

    def unsnooze(self, name: str) -> None:
        """Clear any active snooze on a Hand."""
        with self._snooze_lock:
            self._snoozed_until.pop(name, None)

    def is_snoozed(self, name: str) -> bool:
        """True if the Hand is currently snoozed (deadline not yet reached)."""
        with self._snooze_lock:
            deadline = self._snoozed_until.get(name)
            if deadline is None:
                return False
            if time.time() >= deadline:
                # Expired — clean up lazily
                self._snoozed_until.pop(name, None)
                return False
            return True

    def snooze_remaining(self, name: str) -> float:
        """Seconds remaining on a Hand's snooze, or 0 if not snoozed."""
        with self._snooze_lock:
            deadline = self._snoozed_until.get(name)
            if deadline is None:
                return 0.0
            remaining = deadline - time.time()
            return max(0.0, remaining)

    async def trigger_hand_async(
        self,
        name: str,
        context: Optional[dict] = None,
    ) -> Optional[str]:
        """Queue a Hand for immediate execution, bypassing schedule/idle/drive gates.

        Used by webhook receivers and other push-driven triggers. Honors the
        snooze state — returns None if the Hand is currently snoozed.

        Returns the Hand name if queued, ``None`` if the Hand is unknown or snoozed.
        """
        with self._lock:
            if name not in self._hands:
                logger.warning(f"[HandManager] trigger_hand_async: unknown hand '{name}'")
                return None
        if self.is_snoozed(name):
            logger.info(f"[HandManager] trigger_hand_async: '{name}' is snoozed — skipping")
            return None
        with self._triggers_lock:
            self._pending_triggers[name] = {
                "triggered_by": (context or {}).get("source", "external"),
                "context": context or {},
                "timestamp": time.time(),
            }
        logger.info(f"[HandManager] Queued push-trigger for '{name}'")
        return name

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

        if self.is_snoozed(name):
            remaining = int(self.snooze_remaining(name))
            return HandResult(
                hand_name=name,
                success=False,
                summary="",
                error=f"Hand is snoozed for {remaining}s",
            )

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
        run_context["request_approval"] = lambda tool_name, args=None: self.request_approval_async(
            name, tool_name, args or {}
        )

        async def _step_cb(step: int, description: str):
            try:
                from api.services.websocket_hub import websocket_hub

                asyncio.get_running_loop()  # raises RuntimeError if no loop
                from aura.pools import fire_and_forget

                fire_and_forget(
                    websocket_hub.broadcast_action_trace(name, step, description),
                )
            except Exception:
                pass

        run_context["step_callback"] = _step_cb
        run_context["check_command"] = lambda: self.check_command(name)

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
                logger.warning("[HandManager] Audit chain write failed for hand_start(%s)", name)

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
                logger.debug("[HandManager] Audit chain write failed", exc_info=True)

            # Metacognition recording
            try:
                from aura.consciousness.metacognition import get_metacognitive_engine
                mc = get_metacognitive_engine()
                if mc:
                    mc.log_iteration({
                        "tool": f"hand:{name}",
                        "action": "autonomous_run",
                        "success": result.success,
                        "confidence": 0.8 if result.success else 0.3,
                        "duration_s": result.duration_seconds,
                    })
            except Exception:
                logger.debug("[HandManager] Metacognition recording failed", exc_info=True)

            logger.info(
                f"[HandManager] Hand {name} completed: "
                f"{'SUCCESS' if result.success else 'FAILED'} "
                f"({result.iterations} iterations, {result.duration_seconds:.1f}s, "
                f"${result.cost_usd:.4f})"
            )

            # Notify listeners (API WebSocket, Telegram, etc.)
            self._notify(result)

            # Publish hand_completed event for hand-to-hand triggers
            if self._event_bus is not None:
                try:
                    self._event_bus.publish("hand_completed", {
                        "hand_name": name,
                        "success": result.success,
                        "result": result,
                    })
                except Exception as e:
                    logger.debug(f"[HandManager] Event bus publish failed: {e}")

            # Queue hand-to-hand triggered hands
            self._on_hand_event(name, result.success)

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
            # State transition — wrapped in try/except to prevent leaving hand
            # stuck in RUNNING if the transition itself fails
            try:
                if hand.state == HandState.PAUSED:
                    hand.state = HandState.INACTIVE  # Deactivation was requested during run
                elif hand.state == HandState.RUNNING:
                    hand.state = HandState.COOLDOWN
            except Exception as state_err:
                logger.error(f"[HandManager] State transition failed for {name}: {state_err}")
                # Force to COOLDOWN rather than leaving in RUNNING
                try:
                    hand._state = HandState.COOLDOWN
                except Exception as inner_err:
                    logger.critical("[HandManager] Cannot force %s to COOLDOWN: %s — hand may be stuck in RUNNING", name, inner_err)

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
        with self._triggers_lock:
            pending_trigger_names = set(self._pending_triggers.keys())
        with self._lock:
            for hand in self._hands.values():
                if self.is_snoozed(hand.name):
                    continue  # Skip snoozed hands entirely
                # Hand-to-hand triggers have highest priority (200)
                if hand.name in pending_trigger_names and hand.state in (
                    HandState.ACTIVE, HandState.COOLDOWN
                ):
                    eligible.append((200, hand))
                    continue
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

        # Consume pending trigger if this was a hand-to-hand trigger
        with self._triggers_lock:
            trigger_info = self._pending_triggers.pop(chosen.name, None)

        if trigger_info:
            logger.info(
                f"[HandManager] Triggering hand: {chosen.name} "
                f"(hand-to-hand trigger from {trigger_info['triggered_by']})"
            )
        else:
            logger.info(f"[HandManager] Triggering hand: {chosen.name} (idle={idle_seconds:.0f}s)")

        # Check concurrent hand cap
        with self._threads_lock:
            # Clean up finished threads
            self._active_threads = {
                n: t for n, t in self._active_threads.items() if t.is_alive()
            }
            if len(self._active_threads) >= self.MAX_CONCURRENT_HANDS:
                logger.info(f"[HandManager] Skipping {chosen.name} — {len(self._active_threads)} hands already running")
                return None
            if chosen.name in self._active_threads:
                logger.info(f"[HandManager] Skipping {chosen.name} — already running")
                return None

        # Run in a dedicated event loop on a background thread to avoid blocking
        def _run_async():
            try:
                asyncio.run(self.run_hand(chosen.name, brain, tools, context))
            except Exception as e:
                logger.error(f"[HandManager] Async execution error for {chosen.name}: {e}")
            finally:
                with self._threads_lock:
                    self._active_threads.pop(chosen.name, None)

        thread = threading.Thread(target=_run_async, daemon=True, name=f"hand-run-{chosen.name}")
        with self._threads_lock:
            self._active_threads[chosen.name] = thread
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

    # ====================================================================
    # Event loop management (for thread-safe async operations)
    # ====================================================================

    def set_event_loop(self, loop: asyncio.AbstractEventLoop):
        """Set the event loop reference for thread-safe broadcast operations.

        Called by API at startup to enable approval request broadcasting from threads.
        """
        self._approval_loop = loop

    def set_event_bus(self, bus):
        """Set the event bus for publishing hand_completed events (hand-to-hand triggers)."""
        self._event_bus = bus

    # ====================================================================
    # Notification callback
    # ====================================================================

    def set_notify_callback(self, callback: Callable[[HandResult], None]):
        """Set a callback that fires when any Hand completes."""
        self._notify_callback = callback

    def _notify(self, result: HandResult):
        """Call the notification callback if set."""
        if self._notify_callback:
            try:
                self._notify_callback(result)
            except Exception as e:
                logger.debug(f"[HandManager] Notify callback error: {e}")

        # Broadcast proactive card for notable findings
        if result.success and result.summary and len(result.summary) > 50:
            try:
                self._broadcast_finding_card(result)
            except Exception as e:
                logger.debug(f"[HandManager] Finding card broadcast failed: {e}")

    def _broadcast_finding_card(self, result: HandResult):
        """Broadcast a proactive card for notable hand findings."""
        if not self._approval_loop or self._approval_loop.is_closed():
            return
        try:
            import time as _time

            from api.routes.chat import _broadcast_json
            payload = {
                "type": "proactive",
                "content": f"**{result.hand_name}** found something:\n\n{result.summary[:300]}",
                "action": "notify",
                "priority": "MEDIUM",
                "timestamp": _time.time(),
                "metadata": {"source": "hand", "hand_name": result.hand_name},
            }
            self._approval_loop.call_soon_threadsafe(
                self._approval_loop.create_task,
                _broadcast_json(payload),
            )
        except Exception as e:
            logger.debug(f"[HandManager] Proactive card broadcast failed: {e}")

    # ====================================================================
    # Approval workflow
    # ====================================================================

    async def request_approval_async(self, hand_name: str, tool_name: str, args: dict) -> bool:
        """Request approval for a Hand to use a sensitive tool (async version).

        Awaits until approved/denied or timeout (60s). Does NOT block a thread.
        Returns True if approved, False if denied or timed out.
        """
        request_id = f"apr_{uuid.uuid4().hex}"
        request = ApprovalRequest(
            request_id=request_id,
            hand_name=hand_name,
            tool_name=tool_name,
            args=args,
        )
        async_event = asyncio.Event()
        owner_loop = asyncio.get_running_loop()

        with self._approval_lock:
            self._pending_approvals[request_id] = request
            # Store (event, loop) tuple so resolve_approval can set from any thread
            self._approval_events[request_id] = (async_event, owner_loop)

        logger.info(f"[HandManager] Approval requested: {hand_name} wants to use {tool_name}")

        # Broadcast to WebSocket clients
        try:
            self._broadcast_approval_request(request)
        except Exception as e:
            logger.debug(f"[HandManager] Approval broadcast failed: {e}")

        # Notify Telegram with InlineKeyboard
        try:
            from aura.messaging.telegram_bot import notify_hand_approval_request
            notify_hand_approval_request({
                "request_id": request_id,
                "hand_name": hand_name,
                "tool_name": tool_name,
                "args": args,
            })
        except Exception as e:
            logger.debug(f"[HandManager] Telegram approval notify failed: {e}")

        # Await resolution (non-blocking — frees the event loop for other work)
        try:
            await asyncio.wait_for(async_event.wait(), timeout=60)
        except asyncio.TimeoutError:
            logger.info(f"[HandManager] Approval timed out for {hand_name}/{tool_name}")
            with self._approval_lock:
                self._pending_approvals.pop(request_id, None)
                self._approval_events.pop(request_id, None)
            return False

        with self._approval_lock:
            req = self._pending_approvals.pop(request_id, request)
            self._approval_events.pop(request_id, None)

        logger.info(f"[HandManager] Approval {'granted' if req.approved else 'denied'} for {hand_name}/{tool_name}")
        return req.approved

    def _notify_approval(self, request_id: str, hand_name: str, tool_name: str, args: dict):
        """Send approval notifications via WebSocket and Telegram."""
        request = self._pending_approvals.get(request_id)
        if request:
            try:
                self._broadcast_approval_request(request)
            except Exception as e:
                logger.debug(f"[HandManager] Approval broadcast failed: {e}")
            try:
                from aura.messaging.telegram_bot import notify_hand_approval_request
                notify_hand_approval_request({
                    "request_id": request_id,
                    "hand_name": hand_name,
                    "tool_name": tool_name,
                    "args": args,
                })
            except Exception as e:
                logger.debug(f"[HandManager] Telegram approval notify failed: {e}")

    def resolve_approval(self, request_id: str, approved: bool):
        """Resolve a pending approval request (called by API or Telegram).

        Thread-safe: the only producer is request_approval_async, which stores
        (asyncio.Event, owning_loop) tuples. Signalling goes through the loop
        via call_soon_threadsafe, so this can be called from any thread.
        """
        with self._approval_lock:
            request = self._pending_approvals.get(request_id)
            event_entry = self._approval_events.get(request_id)

        if not request or not event_entry:
            logger.warning(f"[HandManager] Unknown approval request: {request_id}")
            return

        request.approved = approved
        request.resolved = True

        async_event, owner_loop = event_entry
        owner_loop.call_soon_threadsafe(async_event.set)

    def get_pending_approvals(self) -> list[dict]:
        """Get all pending approval requests."""
        with self._approval_lock:
            return [
                {
                    "request_id": r.request_id,
                    "hand_name": r.hand_name,
                    "tool_name": r.tool_name,
                    "args": r.args,
                    "timestamp": r.timestamp,
                    "age_seconds": round(time.time() - r.timestamp, 1),
                }
                for r in self._pending_approvals.values()
                if not r.resolved
            ]

    def _broadcast_approval_request(self, request: ApprovalRequest):
        """Broadcast an approval request to WebSocket clients.

        Uses thread-safe event loop scheduling. Requires set_event_loop() to be called
        at API startup (in main.py lifespan handler).
        """
        if not self._approval_loop or self._approval_loop.is_closed():
            logger.debug("[HandManager] Event loop not available for approval broadcast (API not started)")
            return

        try:
            from api.services.websocket_hub import websocket_hub

            payload = {
                "request_id": request.request_id,
                "hand_name": request.hand_name,
                "tool_name": request.tool_name,
                "args": request.args,
                "timestamp": request.timestamp,
            }

            # Schedule the broadcast coroutine on the main event loop from this thread
            self._approval_loop.call_soon_threadsafe(
                self._approval_loop.create_task,
                websocket_hub.broadcast_hand_approval_request(payload),
            )
        except Exception as e:
            logger.debug(f"[HandManager] WebSocket approval broadcast failed: {e}")

    # ====================================================================
    # Hand-to-hand trigger support
    # ====================================================================

    def _on_hand_event(self, completed_hand_name: str, success: bool):
        """Queue hands that are triggered by the completion of another hand."""
        with self._lock:
            for hand in self._hands.values():
                if hand.manifest.trigger_on_hand != completed_hand_name:
                    continue
                filter_val = hand.manifest.trigger_on_hand_filter
                if filter_val == "success" and not success:
                    continue
                if filter_val == "failure" and success:
                    continue
                with self._triggers_lock:
                    self._pending_triggers[hand.name] = {
                        "triggered_by": completed_hand_name,
                        "success": success,
                        "queued_at": time.time(),
                    }
                logger.info(
                    f"[HandManager] Queued trigger: {hand.name} "
                    f"(triggered by {completed_hand_name}, success={success})"
                )

    # ====================================================================
    # Adaptive scheduling
    # ====================================================================

    def record_finding_referenced(self, hand_name: str):
        """Signal that a hand's finding was referenced/useful — adjust adaptive interval."""
        with self._lock:
            hand = self._hands.get(hand_name)
            if hand:
                hand._adaptive_referenced_count += 1
                # More references → run more often (reduce multiplier, floor 0.5)
                hand._adaptive_interval_multiplier = max(
                    0.5,
                    hand._adaptive_interval_multiplier * 0.9,
                )
                logger.debug(
                    f"[HandManager] {hand_name} referenced — "
                    f"adaptive_multiplier={hand._adaptive_interval_multiplier:.2f}"
                )

    # ====================================================================
    # Mid-execution command support
    # ====================================================================

    def send_command(self, hand_name: str, command: str, new_goal: Optional[str] = None):
        """Send a mid-execution command to a running hand (e.g., 'stop', 'pivot').

        The hand reads this via check_command() in its run_context.
        """
        with self._commands_lock:
            self._pending_commands[hand_name] = {
                "command": command,
                "new_goal": new_goal,
                "sent_at": time.time(),
            }
        logger.info(f"[HandManager] Command sent to {hand_name}: {command}")

    def check_command(self, hand_name: str) -> Optional[dict]:
        """Check for a pending command for this hand and consume it (one-shot).

        Called from inside run_context['check_command'] during hand execution.
        Returns the command dict or None.
        """
        with self._commands_lock:
            return self._pending_commands.pop(hand_name, None)


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
