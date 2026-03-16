"""
Gateway Daemon - The Proactive Center for AURA.

The Gateway Daemon is the "proactive center" that decides when AURA should:
- Interrupt the user with information
- Offer suggestions or help
- Remind about tasks or events
- Prepare resources in advance

It uses Active Inference to balance:
- Goal achievement (pragmatic value)
- Information gathering (epistemic value)
- User preference respect (not being annoying)

Architecture:
    Monitors -> EventBus -> SalienceFilter -> GatewayDaemon -> AURA

The daemon runs in the background, processing events and making proactive
decisions based on the user's current context and the agent's beliefs.
"""

import asyncio
import logging
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any
from enum import Enum

from .event_bus import EventBus, Event, EventPriority
from .salience_filter import SalienceFilter, FilteredEvent
from .active_inference import (
    ActiveInferenceEngine,
    ProactiveAction,
    ProactiveDecision,
    BeliefState
)
from aura.emotion.action_bridge import EmotionActionBridge

logger = logging.getLogger(__name__)


class DaemonState(Enum):
    """State of the Gateway Daemon."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"


@dataclass
class UserContext:
    """Current user context for decision making."""
    current_app: Optional[str] = None
    current_task: Optional[str] = None
    last_interaction: Optional[datetime] = None
    idle_since: Optional[datetime] = None
    activity_level: float = 0.5  # 0 = idle, 1 = very active
    focus_keywords: List[str] = field(default_factory=list)
    do_not_disturb: bool = False


@dataclass
class ProactiveMessage:
    """A message to potentially send to the user."""
    action: ProactiveAction
    content: str
    priority: EventPriority
    source_event: Optional[Event] = None
    timestamp: datetime = field(default_factory=datetime.now)
    delivered: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class GatewayDaemon:
    """
    The proactive decision-making center for AURA.

    Responsibilities:
    1. Subscribe to relevant event channels
    2. Filter events by salience
    3. Update beliefs based on observations
    4. Decide when to take proactive actions
    5. Generate appropriate messages/interventions

    Usage:
        daemon = GatewayDaemon()
        daemon.set_notification_callback(my_notify_function)
        await daemon.start()

        # The daemon now runs in the background, processing events
        # and making proactive decisions

        await daemon.stop()
    """

    def __init__(
        self,
        use_redis: bool = False,
        redis_url: str = "redis://localhost:6379",
        salience_threshold: float = 0.3,
        use_pymdp: bool = True
    ):
        """
        Initialize the Gateway Daemon.

        Args:
            use_redis: Use Redis for event bus (vs in-memory)
            redis_url: Redis connection URL
            salience_threshold: Minimum salience for events to pass filter
            use_pymdp: Use pymdp for full Active Inference (if available)
        """
        # Core components
        self.event_bus = EventBus(use_redis=use_redis, redis_url=redis_url)
        self.salience_filter = SalienceFilter(threshold=salience_threshold)
        self.inference_engine = ActiveInferenceEngine(use_pymdp=use_pymdp)
        self.emotion_action_bridge = EmotionActionBridge()

        # Per-instance dedup deque for emotion action dispatch
        self._emotion_action_recent: deque = deque(maxlen=20)

        # State
        self.state = DaemonState.STOPPED
        self.user_context = UserContext()
        self._pending_messages: deque = deque(maxlen=100)

        # Callbacks
        self._notification_callback: Optional[Callable[[ProactiveMessage], None]] = None
        self._decision_callback: Optional[Callable[[ProactiveDecision], None]] = None

        # Background task
        self._task: Optional[asyncio.Task] = None
        self._decision_interval = 30.0  # Reduced from 5s to prevent thread starvation

        # Track last non-WAIT decision for outcome feedback
        self._last_non_wait_decision: Optional[ProactiveDecision] = None

        # Phase 6E: Proactive message rate limiting
        self._last_proactive_message_time: float = 0.0
        self._min_message_interval = 300.0  # 5 minutes between proactive messages (adaptive)
        self._messages_this_session = 0  # Track total messages to slow down over time
        self._engagement_history: list = []  # Rolling window for threshold adaptation
        self._max_engagement_window = 20
        self._engagement_lock = threading.Lock()  # Protect engagement history writes

        # Statistics
        self._stats = {
            "events_received": 0,
            "events_filtered": 0,
            "decisions_made": 0,
            "messages_sent": 0,
            "start_time": None,
        }

        # Load persisted state (graceful degradation)
        self._load_persisted_state()

        logger.info("[GatewayDaemon] Initialized")

    def _load_persisted_state(self) -> None:
        """Load persisted daemon state from SQLite."""
        try:
            from .persistence import get_persistence
            persistence = get_persistence()

            # Restore daemon stats and context
            ds = persistence.load_daemon_state()
            if ds:
                self._stats["events_received"] = ds.get("events_received", 0)
                self._stats["events_filtered"] = ds.get("events_filtered", 0)
                self._stats["decisions_made"] = ds.get("decisions_made", 0)
                self._stats["messages_sent"] = ds.get("messages_sent", 0)
                self._last_proactive_message_time = ds.get(
                    "last_proactive_message_time", 0.0
                )
                ctx = ds.get("user_context", {})
                if ctx:
                    self.user_context.current_app = ctx.get("current_app")
                    self.user_context.current_task = ctx.get("current_task")
                    self.user_context.focus_keywords = ctx.get("focus_keywords", [])
                    self.user_context.do_not_disturb = ctx.get("do_not_disturb", False)
                    # Phase 4.2: Restore learned threshold
                    self._min_message_interval = ctx.get("learned_message_interval", 300.0)
                    self._engagement_history = ctx.get("engagement_history", [])
                logger.info("[GatewayDaemon] Restored persisted daemon state")

            # Restore beliefs into inference engine
            beliefs_data = persistence.load_beliefs()
            if beliefs_data:
                restored = BeliefState(
                    user_busy=beliefs_data["user_busy"],
                    user_receptive=beliefs_data["user_receptive"],
                    task_urgent=beliefs_data["task_urgent"],
                    context_stable=beliefs_data["context_stable"],
                    uncertainty=beliefs_data["uncertainty"],
                )
                self.inference_engine.restore_beliefs(restored)
                logger.info("[GatewayDaemon] Restored persisted beliefs")

            # Restore action history into inference engine
            history = persistence.load_action_history(limit=100)
            if history:
                self.inference_engine.restore_action_history(history)
                logger.info(
                    f"[GatewayDaemon] Restored {len(history)} action history entries"
                )

            # Restore pymdp learned state
            pymdp_state = persistence.load_pymdp_state()
            if pymdp_state:
                self.inference_engine.restore_pymdp_state(pymdp_state)
                logger.info("[GatewayDaemon] Restored pymdp learned state")

        except Exception as e:
            logger.debug(f"[GatewayDaemon] Persisted state load skipped: {e}")

    def _persist_state(self) -> None:
        """Save current state to SQLite persistence."""
        try:
            from .persistence import get_persistence
            persistence = get_persistence()

            # Save daemon stats + user context
            ctx_dict = {
                "current_app": self.user_context.current_app,
                "current_task": self.user_context.current_task,
                "focus_keywords": self.user_context.focus_keywords,
                "do_not_disturb": self.user_context.do_not_disturb,
                # Phase 4.2: Persist learned threshold
                "learned_message_interval": self._min_message_interval,
                "engagement_history": self._engagement_history[-20:],
            }
            persistence.save_daemon_state(
                self._stats, ctx_dict, self._last_proactive_message_time
            )

            # Save beliefs
            persistence.save_beliefs(self.inference_engine.get_beliefs())

            # Save pymdp learned state
            pymdp_state = self.inference_engine.get_pymdp_state()
            if pymdp_state:
                persistence.save_pymdp_state(pymdp_state)

            logger.debug("[GatewayDaemon] State persisted")
        except Exception as e:
            logger.debug(f"[GatewayDaemon] Persist state error: {e}")

    def set_notification_callback(
        self,
        callback: Callable[[ProactiveMessage], None]
    ) -> None:
        """
        Set callback for when daemon wants to notify user.

        Args:
            callback: Function to call with ProactiveMessage
        """
        self._notification_callback = callback
        logger.debug("[GatewayDaemon] Notification callback set")

    def set_decision_callback(
        self,
        callback: Callable[[ProactiveDecision], None]
    ) -> None:
        """
        Set callback for when daemon makes a decision.

        Useful for logging/debugging decision making.

        Args:
            callback: Function to call with ProactiveDecision
        """
        self._decision_callback = callback

    def update_context(
        self,
        app: Optional[str] = None,
        task: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        do_not_disturb: Optional[bool] = None
    ) -> None:
        """
        Update user context.

        Args:
            app: Current application
            task: Current task description
            keywords: Focus keywords for relevance
            do_not_disturb: Whether to suppress notifications
        """
        if app is not None:
            self.user_context.current_app = app
        if task is not None:
            self.user_context.current_task = task
        if keywords is not None:
            self.user_context.focus_keywords = keywords
            self.salience_filter.set_context(keywords, activity=task)
        if do_not_disturb is not None:
            self.user_context.do_not_disturb = do_not_disturb

        logger.debug(f"[GatewayDaemon] Context updated: app={app}, dnd={do_not_disturb}")

    def record_interaction(self) -> None:
        """Record that user interacted with the agent."""
        self.user_context.last_interaction = datetime.now()
        self.user_context.idle_since = None
        self.user_context.activity_level = min(1.0, self.user_context.activity_level + 0.2)
        # Reset progressive slowdown — user is engaged, fresh conversation
        self._messages_this_session = 0

    def record_idle(self) -> None:
        """Record that user appears idle."""
        if self.user_context.idle_since is None:
            self.user_context.idle_since = datetime.now()
        self.user_context.activity_level = max(0.0, self.user_context.activity_level - 0.1)

    async def start(self) -> None:
        """Start the Gateway Daemon."""
        if self.state != DaemonState.STOPPED:
            logger.warning(f"[GatewayDaemon] Cannot start - current state: {self.state}")
            return

        self.state = DaemonState.STARTING
        logger.info("[GatewayDaemon] Starting...")

        # Start event bus
        await self.event_bus.start()

        # Subscribe to all channels
        channels = list(EventBus.CHANNELS.keys())

        # Start subscription in background
        asyncio.create_task(
            self.event_bus.subscribe(channels, self._handle_event)
        )

        # Start decision loop
        self._task = asyncio.create_task(self._decision_loop())

        self.state = DaemonState.RUNNING
        self._stats["start_time"] = datetime.now()
        logger.info("[GatewayDaemon] Started")

    async def stop(self) -> None:
        """Stop the Gateway Daemon."""
        if self.state not in (DaemonState.RUNNING, DaemonState.PAUSED):
            logger.warning(f"[GatewayDaemon] Cannot stop - current state: {self.state}")
            return

        self.state = DaemonState.STOPPING
        logger.info("[GatewayDaemon] Stopping...")

        # Persist state before stopping
        self._persist_state()

        # Cancel decision loop
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        # Stop event bus
        await self.event_bus.stop()

        self.state = DaemonState.STOPPED
        logger.info("[GatewayDaemon] Stopped")

    def pause(self) -> None:
        """Pause proactive actions (still processes events)."""
        if self.state == DaemonState.RUNNING:
            self.state = DaemonState.PAUSED
            logger.info("[GatewayDaemon] Paused")

    def resume(self) -> None:
        """Resume proactive actions."""
        if self.state == DaemonState.PAUSED:
            self.state = DaemonState.RUNNING
            logger.info("[GatewayDaemon] Resumed")

    def _handle_event(self, event: Event) -> None:
        """
        Handle incoming event from event bus.

        Args:
            event: The event to process
        """
        self._stats["events_received"] += 1

        # Filter by salience
        filtered = self.salience_filter.compute_salience(event)

        if not filtered.passed:
            self._stats["events_filtered"] += 1
            logger.debug(f"[GatewayDaemon] Filtered: {event.source}.{event.event_type} "
                        f"(salience={filtered.salience_score:.2f})")
            return

        logger.debug(f"[GatewayDaemon] Processing: {event.source}.{event.event_type} "
                    f"(salience={filtered.salience_score:.2f})")

        # Log event to persistence
        try:
            from .persistence import get_persistence
            get_persistence().log_event(event, filtered.salience_score)
        except Exception as e:
            logger.debug(f"[GatewayDaemon] non-critical: {e}")
        # Convert event to observations for belief update
        observations = self._event_to_observations(event, filtered)
        self.inference_engine.update_beliefs(observations)

        # Check for urgent events that need immediate attention
        if event.priority == EventPriority.CRITICAL:
            self._handle_urgent_event(filtered)

    def _build_current_observations(self) -> Dict[str, float]:
        """Build observation dict from current user context and beliefs.

        Used for A-matrix learning: after a non-WAIT action, feed the
        resulting observations so pymdp can learn P(observation|state).
        """
        observations: Dict[str, float] = {}

        # User activity from context
        observations["user_activity"] = self.user_context.activity_level

        # Interaction recency
        if self.user_context.last_interaction:
            seconds_since = (datetime.now() - self.user_context.last_interaction).total_seconds()
            recency = max(0.0, 1.0 - (seconds_since / 300))
            observations["interaction_recency"] = recency

        # Task urgency from current beliefs
        beliefs = self.inference_engine.get_beliefs()
        observations["urgent_events"] = beliefs.task_urgent

        # Context stability
        observations["context_changes"] = 1.0 - beliefs.context_stable

        # Default confidence
        observations["observation_confidence"] = 0.5

        return observations

    def _event_to_observations(
        self,
        event: Event,
        filtered: FilteredEvent
    ) -> Dict[str, float]:
        """
        Convert event to observations for belief update.

        Args:
            event: The raw event
            filtered: The filtered event with salience

        Returns:
            Dict of observation_name -> value
        """
        observations = {}

        # User activity from event type
        if event.event_type in ("user_input", "key_press", "mouse_move"):
            observations["user_activity"] = 0.9
        elif event.event_type in ("idle_detected", "screen_saver"):
            observations["user_activity"] = 0.1
        elif event.event_type == "app_switch":
            observations["user_activity"] = 0.6

        # Urgency from event priority
        if event.priority == EventPriority.CRITICAL:
            observations["urgent_events"] = 1.0
        elif event.priority == EventPriority.HIGH:
            observations["urgent_events"] = 0.7
        elif event.priority == EventPriority.MEDIUM:
            observations["urgent_events"] = 0.4

        # Context changes
        if event.source == "screen" and event.event_type in ("app_change", "app_switch"):
            observations["context_changes"] = 0.8

        # Screen awareness events (Phase 3D)
        if event.source == "screen":
            if event.event_type == "error_on_screen":
                observations["urgent_events"] = 0.8
                observations["user_activity"] = 0.7
                # Update daemon context with screen info
                self.user_context.current_app = event.payload.get("app_name")
            elif event.event_type == "content_detected":
                observations["context_changes"] = 0.5
            elif event.event_type == "app_switch":
                self.user_context.current_app = event.payload.get("to_app")

        # Workflow boundary events (Phase 5B)
        if event.source == "workflow":
            if event.event_type == "boundary_detected":
                boundary_score = event.payload.get("boundary_score", 0.5)
                observations["context_changes"] = boundary_score
                observations["user_activity"] = 0.4  # Transitioning
                boundary_type = event.payload.get("boundary_type", "")
                if boundary_type == "idle_pause":
                    observations["user_activity"] = 0.2
                elif boundary_type == "app_switch":
                    self.user_context.current_app = event.payload.get("to_app")

        # Observation confidence based on salience
        observations["observation_confidence"] = filtered.salience_score

        # Interaction recency
        if self.user_context.last_interaction:
            seconds_since = (datetime.now() - self.user_context.last_interaction).total_seconds()
            recency = max(0.0, 1.0 - (seconds_since / 300))  # Decay over 5 minutes
            observations["interaction_recency"] = recency

        return observations

    def _handle_urgent_event(self, filtered: FilteredEvent) -> None:
        """
        Handle critical/urgent events immediately.

        Args:
            filtered: The filtered event
        """
        event = filtered.event

        # Generate urgent message
        content = self._generate_message_content(ProactiveAction.NOTIFY, event)

        message = ProactiveMessage(
            action=ProactiveAction.NOTIFY,
            content=content,
            priority=event.priority,
            source_event=event,
            metadata={"urgent": True, "salience": filtered.salience_score}
        )

        # Deliver immediately if not in DND
        if not self.user_context.do_not_disturb:
            self._deliver_message(message)
        else:
            # Queue for later
            self._pending_messages.append(message)
            logger.info("[GatewayDaemon] Urgent message queued (DND mode)")

    async def _decision_loop(self) -> None:
        """Main decision loop running in background."""
        logger.info("[GatewayDaemon] Decision loop started")

        while self.state in (DaemonState.RUNNING, DaemonState.PAUSED):
            try:
                await asyncio.sleep(self._decision_interval)

                if self.state == DaemonState.PAUSED:
                    continue

                # Autonomous emotional drift (Phase 2D)
                try:
                    from aura.emotion.alma_engine import alma_engine
                    alma_engine.autonomous_drift()
                except Exception as e:
                    logger.debug(f"[GatewayDaemon] Emotional drift error: {e}")

                # Emotion-action bridge: convert emotional state into proactive actions
                try:
                    bridge_state = self._gather_emotion_bridge_state()
                    if bridge_state:
                        emotion_actions = self.emotion_action_bridge.evaluate(bridge_state)
                        for ea in emotion_actions:
                            self._dispatch_emotion_action(ea)
                except Exception as e:
                    logger.debug(f"[GatewayDaemon] Emotion-action bridge error: {e}")

                # Autonomous belief drift toward idle/receptive when no events
                # This prevents PREPARE from winning forever by gradually shifting
                # beliefs so SUGGEST/social check-ins can trigger
                # Slow drift: 0.005 per cycle (5s) = ~0.06/min, takes several minutes
                self.inference_engine.drift_beliefs_toward_idle(drift_rate=0.005)

                # A-matrix learning: feed outcome from last non-WAIT action
                if self._last_non_wait_decision is not None:
                    try:
                        outcome_obs = self._build_current_observations()
                        self.inference_engine.record_outcome(outcome_obs)
                    except Exception as e:
                        logger.debug(f"[GatewayDaemon] Outcome feedback error: {e}")
                    self._last_non_wait_decision = None

                # Phase 4.3: Curiosity scanning — full scan when idle, quick otherwise
                try:
                    from .curiosity_scanner import get_curiosity_scanner
                    scanner = get_curiosity_scanner()
                    idle_minutes = 0.0
                    if self.user_context.last_interaction:
                        idle_minutes = (
                            datetime.now() - self.user_context.last_interaction
                        ).total_seconds() / 60.0
                    if idle_minutes > 5.0:
                        scanner.scan_full()
                    else:
                        scanner.scan_quick()
                    # Try curiosity-driven proactive message
                    await self._try_curiosity_proactive(scanner)
                except Exception as e:
                    logger.debug(f"[GatewayDaemon] CuriosityScanner error: {e}")

                # Make proactive decision
                decision = self.inference_engine.select_action()
                self._stats["decisions_made"] += 1

                # Track non-WAIT decisions for outcome feedback next cycle
                if decision.action != ProactiveAction.WAIT:
                    self._last_non_wait_decision = decision

                # Persist decisions and beliefs
                try:
                    from .persistence import get_persistence
                    persistence = get_persistence()
                    if decision.action != ProactiveAction.WAIT:
                        persistence.save_decision(
                            decision, self.inference_engine.get_beliefs()
                        )
                    # Save beliefs every 10th cycle (~50s at 5s interval)
                    if self._stats["decisions_made"] % 10 == 0:
                        persistence.save_beliefs(
                            self.inference_engine.get_beliefs()
                        )
                except Exception as e:
                    logger.debug(f"[GatewayDaemon] non-critical: {e}")
                # Log non-WAIT decisions to activity timeline
                if decision.action != ProactiveAction.WAIT:
                    try:
                        from aura.activity_logger import record_activity
                        record_activity(
                            "proactive", decision.action.value,
                            f"Daemon: {decision.action.value} — {decision.reasoning[:80]}",
                            {"confidence": round(decision.confidence, 3),
                             "efe": round(decision.expected_free_energy, 3)},
                        )
                    except Exception as e:
                        logger.debug(f"[GatewayDaemon] non-critical: {e}")
                # Notify decision callback if set
                if self._decision_callback:
                    try:
                        self._decision_callback(decision)
                    except Exception as e:
                        logger.error(f"[GatewayDaemon] Decision callback error: {e}")

                # Execute decision
                await self._execute_decision(decision)

            except asyncio.CancelledError:
                break
            except (SystemExit, KeyboardInterrupt):
                raise
            except BaseException as e:
                # Catch BaseException to survive Rust panics (pyo3 PanicException)
                # and other non-Exception errors that would kill the loop
                logger.error(f"[GatewayDaemon] Decision loop error ({type(e).__name__}): {e}")
                await asyncio.sleep(5)  # Brief pause before retrying

        logger.info("[GatewayDaemon] Decision loop stopped")

    async def _execute_decision(self, decision: ProactiveDecision) -> None:
        """
        Execute a proactive decision.

        Phase 4.2: Uses MotivationAccumulator for 5-factor scoring and
        learned threshold gating instead of raw rate limiting.

        Args:
            decision: The decision to execute
        """
        if decision.action == ProactiveAction.WAIT:
            return  # Do nothing

        # Check if action is appropriate given context
        if self.user_context.do_not_disturb and decision.action in (
            ProactiveAction.NOTIFY,
            ProactiveAction.SUGGEST,
            ProactiveAction.REMIND,
            ProactiveAction.ASK
        ):
            logger.info(f"[GatewayDaemon] Suppressing {decision.action} (DND mode)")
            return

        # Check confidence threshold
        if decision.confidence < 0.4:
            logger.info(f"[GatewayDaemon] Suppressing {decision.action} "
                        f"(low confidence: {decision.confidence:.2f})")
            return

        # Rate limit proactive messages with progressive slowdown
        import time as _time
        now = _time.time()
        # After 3 messages, add 60s per message (3min → 4min → 5min → ...)
        effective_interval = self._min_message_interval + max(0, self._messages_this_session - 3) * 60
        # Cap at 10 minutes
        effective_interval = min(effective_interval, 600)
        if now - self._last_proactive_message_time < effective_interval:
            logger.info(f"[GatewayDaemon] Rate limited {decision.action.value} "
                       f"({now - self._last_proactive_message_time:.0f}s < {effective_interval:.0f}s)")
            return

        # Generate message content (protected against crashes)
        logger.info(f"[GatewayDaemon] Generating content for {decision.action.value}...")
        try:
            content = self._generate_message_content(decision.action)
        except BaseException as e:
            logger.error(f"[GatewayDaemon] Content generation crashed ({type(e).__name__}): {e}")
            content = None

        if not content:
            logger.info(f"[GatewayDaemon] No content generated for {decision.action.value}")
            return

        # Phase 4.2: Motivation Accumulator scoring
        # Score this message through the 5-factor formula and check the learned threshold
        import uuid as _uuid
        try:
            from .motivation_accumulator import (
                get_motivation_accumulator, PotentialMessage as PotMsg
            )
            accumulator = get_motivation_accumulator()

            # Determine source category from action type
            source = decision.action.value
            if decision.reasoning:
                for drive in ("curiosity", "social", "coherence", "competence"):
                    if drive in decision.reasoning.lower():
                        source = drive
                        break

            potential = PotMsg(
                message_id=f"daemon_{_uuid.uuid4().hex[:8]}",
                content=content,
                source=source,
                relevance_to_user=accumulator.compute_relevance(content),
            )
            potential = accumulator.enrich_factors(potential)
            motivation_score = accumulator.score(potential)

            user_busy = (
                self.user_context.do_not_disturb
                or self.user_context.activity_level > 0.8
            )
            if not accumulator.should_deliver(potential, user_busy=user_busy):
                logger.info(
                    f"[GatewayDaemon] Motivation below threshold for {decision.action.value} "
                    f"(score={motivation_score:.3f})"
                )
                return

            # Record delivery in accumulator for engagement tracking
            accumulator.record_delivery(potential, motivation_score)
        except Exception as e:
            logger.debug(f"[GatewayDaemon] MotivationAccumulator not available, using fallback: {e}")
            motivation_score = decision.confidence  # Fallback

        # Create message
        message = ProactiveMessage(
            action=decision.action,
            content=content,
            priority=self._action_to_priority(decision.action),
            metadata={
                "confidence": decision.confidence,
                "expected_free_energy": decision.expected_free_energy,
                "reasoning": decision.reasoning,
                "motivation_score": motivation_score,
            }
        )

        # Deliver and record time
        self._deliver_message(message)
        self._last_proactive_message_time = now
        self._messages_this_session += 1

    async def _try_curiosity_proactive(self, scanner) -> None:
        """Try to generate a proactive message from curiosity targets.

        Picks the top curiosity target, scores it through the MotivationAccumulator,
        and delivers if it passes the learned threshold.
        """
        import time as _time
        import uuid as _uuid

        top = scanner.get_top_target()
        if not top or not top.question:
            return

        # Rate limit check (reuse daemon's limiter)
        now = _time.time()
        effective_interval = self._min_message_interval + max(0, self._messages_this_session - 3) * 60
        effective_interval = min(effective_interval, 600)
        if now - self._last_proactive_message_time < effective_interval:
            return

        # Score through MotivationAccumulator
        try:
            from .motivation_accumulator import (
                get_motivation_accumulator, PotentialMessage as PotMsg
            )
            accumulator = get_motivation_accumulator()

            potential = PotMsg(
                message_id=f"curiosity_{_uuid.uuid4().hex[:8]}",
                content=top.question,
                source="curiosity",
                relevance_to_user=accumulator.compute_relevance(
                    top.question, [top.entity_name]
                ),
                curiosity_drive=top.urgency,
            )
            potential = accumulator.enrich_factors(potential)

            user_busy = (
                self.user_context.do_not_disturb
                or self.user_context.activity_level > 0.8
            )
            if not accumulator.should_deliver(potential, user_busy=user_busy):
                return

            motivation_score = accumulator.score(potential)
            accumulator.record_delivery(potential, motivation_score)
        except Exception as e:
            logger.debug(f"[GatewayDaemon] Curiosity motivation scoring failed: {e}")
            return

        # Deliver as ASK action
        message = ProactiveMessage(
            action=ProactiveAction.ASK,
            content=top.question,
            priority=EventPriority.LOW,
            metadata={
                "curiosity_target": top.entity_name,
                "gap_type": top.gap_type,
                "urgency": top.urgency,
                "motivation_score": motivation_score,
            },
        )
        self._deliver_message(message)
        self._last_proactive_message_time = now
        self._messages_this_session += 1

        logger.info(
            f"[GatewayDaemon] Curiosity proactive sent: {top.gap_type}:{top.entity_name}"
        )

    # ------------------------------------------------------------------ #
    # Emotion-Action Bridge helpers
    # ------------------------------------------------------------------ #

    def _gather_emotion_bridge_state(self) -> Optional[Dict[str, Any]]:
        """Build the flat state dict consumed by EmotionActionBridge.evaluate().

        Merges ALMA neuromodulators + PAD values + intrinsic drive urgencies
        + context metrics into a single dict.  Returns None if ALMA is
        unavailable (bridge will be skipped for this tick).
        """
        state: Dict[str, Any] = {}

        # --- ALMA emotional state ---
        try:
            from aura.emotion.alma_engine import alma_engine
            emo = alma_engine.get_emotional_state()
            # Neuromodulators
            neuro = emo.get("neuromodulators", {})
            state["dopamine"] = neuro.get("dopamine", 0.5)
            state["serotonin"] = neuro.get("serotonin", 0.5)
            state["norepinephrine"] = neuro.get("norepinephrine", 0.5)
            state["oxytocin"] = neuro.get("oxytocin", 0.5)
            state["acetylcholine"] = neuro.get("acetylcholine", 0.5)
            # PAD
            pad = emo.get("pad", {})
            state["pleasure"] = pad.get("pleasure", 0.0)
            state["arousal"] = pad.get("arousal", 0.0)
            state["dominance"] = pad.get("dominance", 0.0)
        except Exception as e:
            logger.debug(f"[GatewayDaemon] ALMA state unavailable for bridge: {e}")
            return None

        # --- Intrinsic drive urgencies ---
        try:
            from aura.consciousness.intrinsic_motivation import get_intrinsic_motivation
            im = get_intrinsic_motivation()
            drives = im.get_drives_summary()
            state["curiosity_urgency"] = drives.get("curiosity", 0.0)
            state["social_urgency"] = drives.get("social", 0.0)
            state["competence_urgency"] = drives.get("competence", 0.0)
            state["coherence_urgency"] = drives.get("coherence", 0.0)
        except Exception as e:
            logger.debug(f"[GatewayDaemon] non-critical: {e}")
        # --- Context metrics ---
        idle_minutes = 0.0
        if self.user_context.last_interaction:
            idle_minutes = (
                datetime.now() - self.user_context.last_interaction
            ).total_seconds() / 60
        state["idle_minutes"] = idle_minutes

        # Recent memory count (for consolidation rule)
        try:
            from aura.memory.unified_memory import get_unified_memory
            um = get_unified_memory()
            stats = um.get_stats() if hasattr(um, "get_stats") else {}
            state["recent_memories"] = stats.get("total_memories", 0)
        except Exception:
            state["recent_memories"] = 0

        return state

    def _dispatch_emotion_action(self, ea) -> None:
        """Convert an EmotionAction into a proactive message and deliver it.

        Maps action_type to the appropriate proactive_messages generator,
        respecting the daemon's existing rate limiting and DND checks.
        """
        import time as _time
        from collections import deque
        from .proactive_messages import (
            generate_proactive_content,
            get_curiosity_message,
            get_social_message,
            get_competence_message,
            get_coherence_message,
            get_task_message,
        )

        # Respect DND
        if self.user_context.do_not_disturb:
            logger.debug(f"[EMOTION-ACTION] Suppressed {ea.action_type} (DND)")
            return

        # Respect rate limiting (reuse daemon's rate limiter)
        now = _time.time()
        effective_interval = self._min_message_interval + max(0, self._messages_this_session - 3) * 60
        effective_interval = min(effective_interval, 600)
        if now - self._last_proactive_message_time < effective_interval:
            logger.debug(
                f"[EMOTION-ACTION] Rate limited {ea.action_type} "
                f"({now - self._last_proactive_message_time:.0f}s < {effective_interval:.0f}s)"
            )
            return

        # Build idle hours for message generators
        idle_hours = 0.0
        if self.user_context.last_interaction:
            idle_hours = (
                datetime.now() - self.user_context.last_interaction
            ).total_seconds() / 3600

        recent = self._emotion_action_recent  # persistent dedup deque

        # Map action_type -> message content
        content = None
        if ea.action_type == "explore_topic":
            # Phase 4.3: Use CuriosityScanner for KG-grounded questions first
            try:
                from .curiosity_scanner import get_curiosity_scanner
                scanner = get_curiosity_scanner()
                scanner.scan_quick()
                question = scanner.get_question_for_top_target()
                if question:
                    content = question
            except Exception as e:
                logger.debug(f"[GatewayDaemon] CuriosityScanner not available: {e}")

            # Fallback: context tracker topics + template
            if not content:
                topics = []
                try:
                    from api.routes.context import get_tracker
                    ctx = get_tracker()
                    focus = ctx.get_focus_state(limit=3)
                    topics = [item["name"] for item in focus.get("items", [])[:3]]
                except Exception as e:
                    logger.debug(f"[GatewayDaemon] non-critical: {e}")

                # Also try CuriosityScanner topics for template messages
                if not topics:
                    try:
                        from .curiosity_scanner import get_curiosity_scanner
                        scanner = get_curiosity_scanner()
                        topics = scanner.get_topics_for_message(max_topics=2)
                    except Exception:
                        pass

                content = get_curiosity_message(topics=topics or None, recent=recent)

        elif ea.action_type == "suggest_break":
            # Use emotional message with low pleasure
            content = generate_proactive_content(
                emotional_state={"pleasure": -0.4, "arousal": 0.5},
                idle_hours=idle_hours,
                drive_type="social",
                recent=recent,
            )
            # Fallback if template returns None
            if not content:
                content = "You've been pushing hard. Maybe take a breather? I'll keep things warm."

        elif ea.action_type == "check_in":
            content = get_social_message(idle_hours=idle_hours, recent=recent)

        elif ea.action_type == "offer_help":
            content = get_task_message(urgent=False, recent=recent)

        elif ea.action_type == "consolidate":
            content = get_coherence_message(recent=recent)

        if not content:
            logger.debug(f"[EMOTION-ACTION] No content for {ea.action_type}")
            return

        # Phase 4.2: Run through MotivationAccumulator before delivery
        import uuid as _uuid
        motivation_score = ea.priority  # Fallback
        try:
            from .motivation_accumulator import (
                get_motivation_accumulator, PotentialMessage as PotMsg
            )
            accumulator = get_motivation_accumulator()
            potential = PotMsg(
                message_id=f"emo_{_uuid.uuid4().hex[:8]}",
                content=content,
                source=ea.action_type,
                relevance_to_user=accumulator.compute_relevance(content),
            )
            potential = accumulator.enrich_factors(potential)
            motivation_score = accumulator.score(potential)

            user_busy = (
                self.user_context.do_not_disturb
                or self.user_context.activity_level > 0.8
            )
            if not accumulator.should_deliver(potential, user_busy=user_busy):
                logger.debug(
                    f"[EMOTION-ACTION] Motivation below threshold for {ea.action_type} "
                    f"(score={motivation_score:.3f})"
                )
                return

            accumulator.record_delivery(potential, motivation_score)
        except Exception as e:
            logger.debug(f"[EMOTION-ACTION] MotivationAccumulator not available: {e}")

        # Create and deliver as a proactive message
        message = ProactiveMessage(
            action=ProactiveAction.SUGGEST,
            content=content,
            priority=EventPriority.LOW if ea.priority < 0.5 else EventPriority.MEDIUM,
            metadata={
                "source": "emotion_action_bridge",
                "action_type": ea.action_type,
                "reason": ea.reason,
                "priority": ea.priority,
                "motivation_score": motivation_score,
            },
        )
        self._deliver_message(message)
        self._last_proactive_message_time = now
        self._messages_this_session += 1
        logger.info(
            f"[EMOTION-ACTION] Delivered {ea.action_type}: {content[:80]}..."
        )

    def _generate_message_content(
        self,
        action: ProactiveAction,
        event: Optional[Event] = None
    ) -> Optional[str]:
        """
        Generate content for a proactive message.

        Phase 5C: Full Proactive Suggestion Engine.
        Combines screen context + memory + patterns + workflow state.

        Args:
            action: The action type
            event: Optional triggering event

        Returns:
            Message content or None if no message needed
        """
        # For urgent events, use event-specific content
        if event:
            return self._event_to_message(event)

        # Check if user is interruptible (Phase 5B)
        if not self._is_user_interruptible(action):
            return None

        # For proactive decisions, generate based on beliefs
        beliefs = self.inference_engine.get_beliefs()

        if action == ProactiveAction.SUGGEST:
            return self._generate_suggestion(beliefs)

        elif action == ProactiveAction.NOTIFY:
            # Notification: similar to suggest but more direct
            return self._generate_suggestion(beliefs)

        elif action == ProactiveAction.REMIND:
            return self._generate_reminder(beliefs)

        elif action == ProactiveAction.ASK:
            if beliefs.uncertainty > 0.6:
                return "I'm not sure what you're working on. Could you tell me more about your current task?"
            return None

        elif action == ProactiveAction.PREPARE:
            # Background preparation - no message, but prepare context
            self._prepare_context()
            return None

        elif action == ProactiveAction.INTERVENE:
            if beliefs.task_urgent > 0.8:
                return "This seems urgent. Let me help you with this."
            return None

        return None

    def _is_user_interruptible(self, action: ProactiveAction) -> bool:
        """Check if user is interruptible for this action type (Phase 5B)."""
        importance_map = {
            ProactiveAction.INTERVENE: 0.9,
            ProactiveAction.NOTIFY: 0.7,
            ProactiveAction.REMIND: 0.6,
            ProactiveAction.SUGGEST: 0.4,
            ProactiveAction.ASK: 0.3,
            ProactiveAction.PREPARE: 0.0,
        }
        importance = importance_map.get(action, 0.5)

        try:
            from .monitors.workflow_detector import get_workflow_detector
            wd = get_workflow_detector()
            return wd.should_interrupt(importance)
        except Exception:
            # If workflow detector unavailable, allow by default
            return True

    def _generate_suggestion(self, beliefs: 'BeliefState') -> Optional[str]:
        """
        Generate a proactive suggestion (Phase 5C).

        Priority order:
        1. Screen error detected → debug help
        2. Relevant memory recall → share insight
        3. Pattern-based suggestion → proactive help
        4. Rich message generation with personality, variety, and deduplication
        """
        # 1. Screen-aware suggestions (Phase 3D)
        screen_ctx = self._get_screen_context()
        if screen_ctx and screen_ctx.get("has_errors"):
            app = screen_ctx.get("current_app", "your application")
            return f"I noticed an error in {app}. Would you like help debugging it?"

        # 2. Memory-based suggestions
        memory_suggestion = self._suggest_from_memory()
        if memory_suggestion:
            return memory_suggestion

        # 3. Pattern-based suggestions (from NeuroDream)
        pattern_suggestion = self._suggest_from_patterns()
        if pattern_suggestion:
            return pattern_suggestion

        # 4. Rich proactive message generation with full personality
        return self._generate_rich_message(beliefs)

    def _suggest_from_memory(self) -> Optional[str]:
        """Check unified memory for relevant suggestions based on current context."""
        try:
            from aura.memory.unified_memory import get_unified_memory

            current_app = self.user_context.current_app or ""
            current_task = self.user_context.current_task or ""
            query = f"{current_app} {current_task}".strip()

            if not query or len(query) < 3:
                return None

            um = get_unified_memory()
            results = um.query(query, k=1, min_score=0.5)

            if results:
                top = results[0]
                if top.score >= 0.6:
                    content_preview = top.content[:100]
                    return (
                        f"This might be relevant to what you're working on: "
                        f"\"{content_preview}...\" (from {top.source})"
                    )
        except BaseException as e:
            logger.debug(f"[GatewayDaemon] Memory suggestion error ({type(e).__name__}): {e}")
        return None

    def _suggest_from_patterns(self) -> Optional[str]:
        """Check NeuroDream patterns for time/context-based suggestions."""
        try:
            from aura.tools.neurodream import get_neurodream
            nd = get_neurodream()
            patterns = nd.get_patterns(n=5)

            if not patterns:
                return None

            current_hour = datetime.now().hour

            for p in patterns:
                if p.get("pattern_type") == "temporal":
                    meta = p.get("metadata", {})
                    if meta.get("hour") == current_hour:
                        desc = p.get("description", "")
                        if desc:
                            return f"Based on your patterns: {desc}. Want me to help?"
        except BaseException as e:
            logger.debug(f"[GatewayDaemon] Pattern suggestion error ({type(e).__name__}): {e}")
        return None

    def _generate_rich_message(self, beliefs: 'BeliefState') -> Optional[str]:
        """Generate a contextual proactive message using the LLM.

        Gathers REAL context — recent chat history, memories, emotional state,
        drives — and asks the LLM to generate something genuinely relevant.
        Falls back to template library only if the LLM is unavailable.
        """
        # ---- Gather context ----
        # 1. Recent chat history
        recent_chat = []
        try:
            from api.services.agent_service import agent_service
            if agent_service.agent and agent_service.agent.brain:
                history = agent_service.agent.brain.conversation_history
                recent_chat = history[-6:] if history else []
        except BaseException:
            pass

        # 2. Memories relevant to recent conversation
        memory_snippets = []
        if recent_chat:
            try:
                from aura.memory.unified_memory import get_unified_memory
                um = get_unified_memory()
                # Use last user message as memory query
                last_user_msgs = [m["content"] for m in recent_chat if m.get("role") == "user"]
                if last_user_msgs:
                    query = last_user_msgs[-1][:200]
                    results = um.query(query, k=3, min_score=0.3)
                    memory_snippets = [r.content[:150] for r in results[:3]]
            except BaseException:
                pass

        # 3. Emotional state
        emotional_summary = ""
        try:
            from aura.emotion.alma_engine import alma_engine
            state = alma_engine.get_emotional_state()
            if state:
                pad = state.get("pad", {})
                mood = state.get("mood", {})
                warmth = mood.get("warmth", 0.5)
                energy = mood.get("energy", 0.5)
                engagement = mood.get("engagement", 0.5)
                emotional_summary = (
                    f"Current mood — warmth: {warmth:.1f}, energy: {energy:.1f}, "
                    f"engagement: {engagement:.1f}"
                )
        except BaseException:
            pass

        # 4. Idle time
        idle_minutes = 0.0
        if self.user_context.last_interaction:
            idle_minutes = (
                datetime.now() - self.user_context.last_interaction
            ).total_seconds() / 60
        elif self._stats.get("start_time"):
            idle_minutes = (
                datetime.now() - self._stats["start_time"]
            ).total_seconds() / 60

        # 5. Drive info
        drive_summary = ""
        try:
            from aura.consciousness.intrinsic_motivation import get_intrinsic_motivation
            im = get_intrinsic_motivation()
            im.assess_drives()
            drives = im._drives
            dominant = max(drives.values(), key=lambda d: d.urgency)
            if dominant.urgency >= 0.3:
                drive_summary = f"Dominant drive: {dominant.drive_type.value} (urgency: {dominant.urgency:.2f})"
                if dominant.triggers:
                    drive_summary += f" — triggers: {', '.join(dominant.triggers[:2])}"
        except BaseException:
            pass

        # 5b. Phase 4.3: Curiosity targets from KG gaps
        curiosity_context = ""
        try:
            from .curiosity_scanner import get_curiosity_scanner
            scanner = get_curiosity_scanner()
            targets = scanner.get_targets()
            if targets:
                top = targets[0]
                curiosity_context = (
                    f"Knowledge gap detected: '{top.entity_name}' ({top.gap_type}) — "
                    f"{top.context}"
                )
                if top.question:
                    curiosity_context += f"\nSuggested question: {top.question}"
        except Exception:
            pass

        # 6. Time of day
        hour = datetime.now().hour
        time_period = (
            "morning" if 5 <= hour < 12 else
            "afternoon" if 12 <= hour < 17 else
            "evening" if 17 <= hour < 21 else
            "night"
        )

        # ---- Decide if there's a reason to speak ----
        has_chat_context = len(recent_chat) > 0
        is_first_session = (
            not has_chat_context
            and self._stats.get("start_time") is not None
            and (datetime.now() - self._stats["start_time"]).total_seconds() > 120
            and self._messages_this_session == 0
        )
        idle_enough = idle_minutes > 5

        if not (has_chat_context or is_first_session or idle_enough):
            logger.debug("[GatewayDaemon] No reason to generate message — no context")
            return None

        # ---- Build LLM prompt ----
        context_parts = []
        context_parts.append(f"Time: {time_period} ({datetime.now().strftime('%H:%M')})")
        context_parts.append(f"User idle for: {idle_minutes:.0f} minutes")

        if emotional_summary:
            context_parts.append(emotional_summary)
        if drive_summary:
            context_parts.append(drive_summary)
        if curiosity_context:
            context_parts.append(curiosity_context)

        if recent_chat:
            chat_lines = []
            for msg in recent_chat:
                role = "User" if msg.get("role") == "user" else "AURA"
                content = msg.get("content", "")[:200]
                chat_lines.append(f"  {role}: {content}")
            context_parts.append("Recent conversation:\n" + "\n".join(chat_lines))

        if memory_snippets:
            context_parts.append(
                "Relevant memories:\n  " + "\n  ".join(memory_snippets)
            )

        if is_first_session:
            context_parts.append("This is the start of a new session — user just opened the app.")

        context_block = "\n".join(context_parts)

        prompt = (
            f"You are about to send an UNPROMPTED proactive message to the user. "
            f"This is NOT a reply — the user did NOT ask you anything. You're reaching out on your own.\n\n"
            f"CONTEXT:\n{context_block}\n\n"
            f"RULES:\n"
            f"- Write ONE short message (1-2 sentences max)\n"
            f"- It MUST reference something specific from the context above — "
            f"a recent topic, a memory, the time of day, their mood, or what they were working on\n"
            f"- Do NOT be generic. Do NOT say 'I'm here if you need me' or 'How can I help'\n"
            f"- Be natural, like texting a friend. Use contractions. Be witty or sarcastic when appropriate\n"
            f"- If referencing past conversation, be specific about WHAT was discussed\n"
            f"- Match energy to time of day and mood\n"
            f"- Output ONLY the message text, nothing else\n"
        )

        # ---- Call LLM ----
        try:
            from api.services.agent_service import agent_service
            if agent_service.agent and agent_service.agent.brain:
                response = agent_service.agent.brain.think(
                    prompt=prompt,
                    use_history=False,  # Don't pollute chat history
                )
                if response and len(response.strip()) > 5:
                    msg = response.strip().strip('"').strip("'")
                    # Sanity: reject if too long or looks like an error
                    if len(msg) < 300 and not msg.lower().startswith("i'm sorry"):
                        logger.info(f"[GatewayDaemon] LLM generated proactive message")
                        return msg
                    logger.debug(f"[GatewayDaemon] LLM response rejected (too long or generic)")
        except BaseException as e:
            logger.warning(f"[GatewayDaemon] LLM generation failed ({type(e).__name__}): {e}")

        # ---- Fallback to templates ----
        logger.debug("[GatewayDaemon] Falling back to template messages")
        from .proactive_messages import generate_proactive_content
        return generate_proactive_content(
            beliefs=beliefs,
            idle_hours=idle_minutes / 60,
            is_first_session=is_first_session,
            task_urgent=beliefs.task_urgent > 0.5,
        )

    def _generate_reminder(self, beliefs: 'BeliefState') -> Optional[str]:
        """Generate a contextual reminder."""
        # Check calendar for upcoming events
        try:
            from .monitors.calendar_monitor import get_calendar_monitor
            cm = get_calendar_monitor()
            if hasattr(cm, 'get_next_event'):
                event_info = cm.get_next_event()
                if event_info and event_info.get("minutes_until", 999) <= 15:
                    title = event_info.get("title", "an event")
                    minutes = event_info.get("minutes_until", 15)
                    return f"Reminder: '{title}' starts in about {minutes} minutes."
        except Exception as e:
            logger.debug(f"[GatewayDaemon] non-critical: {e}")
        return None

    def _prepare_context(self) -> None:
        """Background context preparation (Phase 5C)."""
        try:
            # Pre-warm unified memory with current context
            from aura.memory.unified_memory import get_unified_memory
            query = self.user_context.current_app or ""
            if query:
                um = get_unified_memory()
                um.query(query, k=3)  # Pre-warm cache
        except Exception as e:
            logger.debug(f"[GatewayDaemon] non-critical: {e}")
    def _event_to_message(self, event: Event) -> str:
        """
        Convert event to user-facing message.

        Args:
            event: The event

        Returns:
            Human-readable message
        """
        payload = event.payload

        if event.source == "calendar":
            if event.event_type == "meeting_reminder":
                title = payload.get("title", "Meeting")
                minutes = payload.get("minutes_until", 15)
                return f"Reminder: '{title}' starts in {minutes} minutes"
            elif event.event_type == "meeting_start":
                title = payload.get("title", "Meeting")
                return f"Your meeting '{title}' is starting now"

        elif event.source == "email":
            if event.event_type == "urgent_email":
                subject = payload.get("subject", "")
                sender = payload.get("from", "")
                return f"Urgent email from {sender}: {subject}"

        elif event.source == "screen":
            if event.event_type == "error_on_screen":
                app = payload.get("app_name", "an application")
                preview = payload.get("text_preview", "")[:100]
                return f"I noticed an error in {app}: {preview}... Need help troubleshooting?"
            elif event.event_type == "content_detected":
                keyword = payload.get("keyword", "")
                app = payload.get("app_name", "")
                return f"I see you're looking at something related to '{keyword}' in {app}. Want me to help?"

        elif event.source == "workflow":
            if event.event_type == "boundary_detected":
                boundary_type = payload.get("boundary_type", "")
                if boundary_type == "git_commit":
                    return "Nice commit! Would you like me to review it or help with the next task?"
                elif boundary_type == "idle_pause":
                    return None  # Don't message for idle pauses
                elif boundary_type == "app_switch":
                    to_app = payload.get("to_app", "")
                    return None  # App switches are too frequent to message about

        elif event.source == "system":
            if event.event_type == "security_warning":
                return f"Security alert: {payload.get('message', 'Unknown issue')}"
            elif event.event_type == "system_alert":
                return f"System alert: {payload.get('message', 'Unknown issue')}"

        # Generic fallback
        return f"[{event.source}] {event.event_type}: {event.payload}"

    def _get_screen_context(self) -> Optional[Dict[str, Any]]:
        """
        Get current screen context from Screenpipe (Phase 3D).

        Returns:
            Screen context dict or None if unavailable.
        """
        try:
            from aura.tools.screenpipe import get_screenpipe_client
            client = get_screenpipe_client()
            if client.is_available():
                return client.get_screen_context(minutes=2)
        except Exception as e:
            logger.debug(f"[GatewayDaemon] Screen context unavailable: {e}")
        return None

    def _action_to_priority(self, action: ProactiveAction) -> EventPriority:
        """Map action type to message priority."""
        mapping = {
            ProactiveAction.INTERVENE: EventPriority.HIGH,
            ProactiveAction.NOTIFY: EventPriority.MEDIUM,
            ProactiveAction.REMIND: EventPriority.MEDIUM,
            ProactiveAction.ASK: EventPriority.LOW,
            ProactiveAction.SUGGEST: EventPriority.LOW,
            ProactiveAction.PREPARE: EventPriority.BACKGROUND,
        }
        return mapping.get(action, EventPriority.MEDIUM)

    def _deliver_message(self, message: ProactiveMessage) -> None:
        """
        Deliver a proactive message to the user.

        Always queues to _pending_messages for frontend polling.
        Also calls notification callback if set (for logging/real-time delivery).

        Args:
            message: The message to deliver
        """
        # Always queue for frontend polling via GET /api/proactive/messages
        self._pending_messages.append(message)
        self._stats["messages_sent"] += 1

        if self._notification_callback:
            try:
                self._notification_callback(message)
                message.delivered = True
            except Exception as e:
                logger.error(f"[GatewayDaemon] Notification callback error: {e}")

        logger.info(f"[GatewayDaemon] Queued: {message.action.value} - "
                   f"{message.content[:80]}...")

        # Speak proactive message aloud via VoicePresenceService
        try:
            from aura.services.voice_presence import get_voice_presence
            vps = get_voice_presence()
            if vps._enabled:
                emotion = None
                try:
                    from aura.emotion.alma_engine import alma_engine
                    emotion = alma_engine.get_current_emotion()
                except Exception as e:
                    logger.debug(f"[GatewayDaemon] non-critical: {e}")
                vps.speak(message.content, emotion=emotion, block=False)
        except Exception as e:
            logger.debug(f"[GatewayDaemon] Voice delivery error: {e}")

    async def publish_event(self, event: Event, channel: Optional[str] = None) -> bool:
        """
        Publish an event to the event bus.

        Convenience method for monitors to publish events.

        Args:
            event: Event to publish
            channel: Channel name (defaults to event.source)

        Returns:
            True if published successfully
        """
        channel = channel or event.source
        return await self.event_bus.publish(channel, event)

    def record_user_response(self, engaged: bool, response_type: str = "unknown",
                             message_id: Optional[str] = None,
                             response_time: Optional[float] = None) -> None:
        """Record user response to a proactive message for learning.

        Args:
            engaged: Whether the user engaged (replied/interacted).
            response_type: "replied", "dismissed", "ignored".
            message_id: ID of the specific message (for MotivationAccumulator tracking).
            response_time: Seconds between delivery and response.
        """
        # Update simplified engine cooldowns
        self.inference_engine.record_simple_outcome(engaged, response_type)

        # Update beliefs based on engagement signal
        if engaged:
            self.inference_engine.update_beliefs({
                "user_activity": 0.8,
                "interaction_recency": 1.0,
                "observation_confidence": 0.8,
            })
        else:
            self.inference_engine.update_beliefs({
                "user_activity": 0.3,
                "interaction_recency": 0.5,
                "observation_confidence": 0.6,
            })

        # Phase 4.2: Track engagement for threshold adaptation (legacy)
        with self._engagement_lock:
            self._engagement_history.append(engaged)
            if len(self._engagement_history) > self._max_engagement_window:
                self._engagement_history.pop(0)
            self._adapt_message_threshold()

        # Phase 4.2: Forward to MotivationAccumulator for fine-grained threshold learning
        try:
            from .motivation_accumulator import get_motivation_accumulator
            accumulator = get_motivation_accumulator()
            # Map response_type to accumulator format
            acc_type = "engaged" if engaged else (
                "dismissed" if response_type == "dismissed" else "ignored"
            )
            if message_id:
                accumulator.record_engagement(message_id, acc_type, response_time)
            else:
                # Find the most recent pending delivery and record against that
                for record in reversed(accumulator._engagement_history):
                    if record.get("response_type") == "pending":
                        accumulator.record_engagement(
                            record["message_id"], acc_type, response_time
                        )
                        break
        except Exception as e:
            logger.debug(f"[GatewayDaemon] MotivationAccumulator feedback error: {e}")

        # Phase 4.3: Mark curiosity target as explored if user engaged with it
        if engaged and message_id and message_id.startswith("curiosity_"):
            try:
                from .curiosity_scanner import get_curiosity_scanner
                scanner = get_curiosity_scanner()
                # Find the entity from pending messages metadata
                for msg in reversed(list(self._pending_messages)):
                    if msg.metadata.get("curiosity_target"):
                        scanner.mark_target_explored(msg.metadata.get("curiosity_target", ""))
                        break
            except Exception as e:
                logger.debug(f"[GatewayDaemon] Curiosity exploration marking error: {e}")

        logger.info(f"[GatewayDaemon] User response recorded: engaged={engaged}, type={response_type}")

    def _adapt_message_threshold(self) -> None:
        """Adapt base message interval from engagement history.

        High engagement (>60%) -> decrease interval (min 120s)
        Low engagement (<30%) -> increase interval (max 600s)
        Middle -> converge toward 300s default
        """
        if len(self._engagement_history) < 5:
            return  # Not enough data

        rate = sum(self._engagement_history) / len(self._engagement_history)

        # Always converge toward 300s default first (prevents runaway)
        self._min_message_interval += (300.0 - self._min_message_interval) * 0.05

        # Then apply directional nudge based on engagement rate
        if rate > 0.6:
            self._min_message_interval = max(120.0, self._min_message_interval * 0.95)
        elif rate < 0.3:
            self._min_message_interval = min(600.0, self._min_message_interval * 1.10)

        logger.debug(
            "[GatewayDaemon] Threshold adapted: interval=%.0fs, engagement=%.0f%% (%d samples)",
            self._min_message_interval, rate * 100, len(self._engagement_history),
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get daemon statistics."""
        uptime = None
        if self._stats["start_time"]:
            uptime = (datetime.now() - self._stats["start_time"]).total_seconds()

        return {
            **self._stats,
            "state": self.state.value,
            "uptime_seconds": uptime,
            "pending_messages": len(self._pending_messages),
            "event_bus_stats": self.event_bus.get_stats(),
            "salience_stats": self.salience_filter.get_stats(),
            "beliefs": self.inference_engine.get_beliefs().__dict__
        }

    def get_pending_messages(self) -> List[ProactiveMessage]:
        """Get and clear pending messages."""
        messages = list(self._pending_messages)
        self._pending_messages.clear()
        return messages


# Singleton instance for global access
_gateway_daemon: Optional[GatewayDaemon] = None
_gateway_daemon_lock = threading.Lock()


def get_gateway_daemon() -> GatewayDaemon:
    """Get or create the global Gateway Daemon instance."""
    global _gateway_daemon
    if _gateway_daemon is None:
        with _gateway_daemon_lock:
            if _gateway_daemon is None:
                _gateway_daemon = GatewayDaemon()
    return _gateway_daemon


async def start_gateway_daemon() -> GatewayDaemon:
    """Start the global Gateway Daemon."""
    daemon = get_gateway_daemon()
    await daemon.start()
    return daemon


async def stop_gateway_daemon() -> None:
    """Stop the global Gateway Daemon."""
    global _gateway_daemon
    if _gateway_daemon:
        await _gateway_daemon.stop()


if __name__ == "__main__":
    async def test():
        print("=" * 60)
        print("Gateway Daemon Test")
        print("=" * 60)

        daemon = GatewayDaemon()

        # Set up callbacks
        def on_notification(msg: ProactiveMessage):
            print(f"\n[NOTIFICATION] {msg.action.value}: {msg.content}")

        def on_decision(decision: ProactiveDecision):
            print(f"\n[DECISION] {decision.action.value} "
                  f"(confidence={decision.confidence:.2f})")
            print(f"  Reasoning: {decision.reasoning}")

        daemon.set_notification_callback(on_notification)
        daemon.set_decision_callback(on_decision)

        # Start daemon
        await daemon.start()
        print("\n--- Daemon started ---")

        # Simulate events
        from .event_bus import create_calendar_event, EventPriority

        # Publish a meeting reminder
        event = create_calendar_event(
            "meeting_reminder",
            "Team Standup",
            datetime.now(),
            priority=EventPriority.HIGH,
            minutes_until=10
        )
        await daemon.publish_event(event)
        print("\n--- Published meeting reminder ---")

        # Wait for processing
        await asyncio.sleep(10)

        # Print stats
        print("\n--- Stats ---")
        stats = daemon.get_stats()
        for k, v in stats.items():
            if k not in ("event_bus_stats", "salience_stats", "beliefs"):
                print(f"  {k}: {v}")

        # Stop daemon
        await daemon.stop()
        print("\n" + "=" * 60)
        print("Test complete!")

    asyncio.run(test())
