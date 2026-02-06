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
        use_pymdp: bool = False
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

        # State
        self.state = DaemonState.STOPPED
        self.user_context = UserContext()
        self._pending_messages: List[ProactiveMessage] = []

        # Callbacks
        self._notification_callback: Optional[Callable[[ProactiveMessage], None]] = None
        self._decision_callback: Optional[Callable[[ProactiveDecision], None]] = None

        # Background task
        self._task: Optional[asyncio.Task] = None
        self._decision_interval = 5.0  # Seconds between decision cycles

        # Phase 6E: Proactive message rate limiting
        self._last_proactive_message_time: float = 0.0
        self._min_message_interval = 120.0  # Minimum 2 minutes between proactive messages

        # Statistics
        self._stats = {
            "events_received": 0,
            "events_filtered": 0,
            "decisions_made": 0,
            "messages_sent": 0,
            "start_time": None,
        }

        logger.info("[GatewayDaemon] Initialized")

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

        # Convert event to observations for belief update
        observations = self._event_to_observations(event, filtered)
        self.inference_engine.update_beliefs(observations)

        # Check for urgent events that need immediate attention
        if event.priority == EventPriority.CRITICAL:
            self._handle_urgent_event(filtered)

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
                    from apprentice_agent.emotion.alma_engine import alma_engine
                    alma_engine.autonomous_drift()
                except Exception as e:
                    logger.debug(f"[GatewayDaemon] Emotional drift error: {e}")

                # Make proactive decision
                decision = self.inference_engine.select_action()
                self._stats["decisions_made"] += 1

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
            except Exception as e:
                logger.error(f"[GatewayDaemon] Decision loop error: {e}")

        logger.info("[GatewayDaemon] Decision loop stopped")

    async def _execute_decision(self, decision: ProactiveDecision) -> None:
        """
        Execute a proactive decision.

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
            logger.debug(f"[GatewayDaemon] Suppressing {decision.action} (DND mode)")
            return

        # Check confidence threshold
        if decision.confidence < 0.4:
            logger.debug(f"[GatewayDaemon] Suppressing {decision.action} "
                        f"(low confidence: {decision.confidence:.2f})")
            return

        # Phase 6E: Rate limit proactive messages (min 2 min between messages)
        import time as _time
        now = _time.time()
        if now - self._last_proactive_message_time < self._min_message_interval:
            logger.debug("[GatewayDaemon] Rate limited - too soon since last message")
            return

        # Generate message content
        content = self._generate_message_content(decision.action)

        if not content:
            return

        # Create message
        message = ProactiveMessage(
            action=decision.action,
            content=content,
            priority=self._action_to_priority(decision.action),
            metadata={
                "confidence": decision.confidence,
                "expected_free_energy": decision.expected_free_energy,
                "reasoning": decision.reasoning
            }
        )

        # Deliver and record time
        self._deliver_message(message)
        self._last_proactive_message_time = now

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
        4. Emotional check-in → wellbeing
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

        # 4. Emotional check-in (if user seems stressed or it's been a while)
        emotional_suggestion = self._suggest_emotional_checkin(beliefs)
        if emotional_suggestion:
            return emotional_suggestion

        # 5. Phase 6E: Intrinsic motivation-driven suggestions
        drive_suggestion = self._suggest_from_drives()
        if drive_suggestion:
            return drive_suggestion

        # 6. Generic task help
        if beliefs.task_urgent > 0.5:
            return "I notice you might need help with something. Would you like me to assist?"

        return None

    def _suggest_from_memory(self) -> Optional[str]:
        """Check unified memory for relevant suggestions based on current context."""
        try:
            from apprentice_agent.memory.unified_memory import get_unified_memory

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
        except Exception as e:
            logger.debug(f"[GatewayDaemon] Memory suggestion error: {e}")
        return None

    def _suggest_from_patterns(self) -> Optional[str]:
        """Check NeuroDream patterns for time/context-based suggestions."""
        try:
            from apprentice_agent.tools.neurodream import get_neurodream
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
        except Exception as e:
            logger.debug(f"[GatewayDaemon] Pattern suggestion error: {e}")
        return None

    def _suggest_emotional_checkin(self, beliefs: 'BeliefState') -> Optional[str]:
        """Suggest an emotional check-in if appropriate."""
        try:
            from apprentice_agent.emotion.alma_engine import alma_engine
            state = alma_engine.get_emotional_state()

            if not state:
                return None

            pad = state.get("pad", {})
            pleasure = pad.get("pleasure", 0.0)

            # If AURA's emotional state is low, suggest checking in
            if pleasure < -0.3:
                return "I sense things might be a bit tough right now. How are you doing?"

            # If it's been a long time since interaction
            if self.user_context.last_interaction:
                hours_since = (
                    datetime.now() - self.user_context.last_interaction
                ).total_seconds() / 3600
                if hours_since > 4:
                    return "It's been a while! How's everything going?"
        except Exception as e:
            logger.debug(f"[GatewayDaemon] Emotional check-in error: {e}")
        return None

    def _suggest_from_drives(self) -> Optional[str]:
        """Generate suggestion from intrinsic motivation drives (Phase 6E).

        Checks AURA's active drives and generates natural-sounding messages
        when a drive is urgent enough to warrant speaking up.
        """
        try:
            from apprentice_agent.consciousness.intrinsic_motivation import get_intrinsic_motivation
            im = get_intrinsic_motivation()
            im.assess_drives()

            # Get the most urgent drive
            drives = im._drives
            dominant = max(drives.values(), key=lambda d: d.urgency)

            # Only speak up if drive urgency is high enough
            if dominant.urgency < 0.5:
                return None

            drive_type = dominant.drive_type.value

            if drive_type == "curiosity":
                # Curiosity: ask about something or share a connection
                topics = []
                try:
                    from api.routes.context import get_tracker
                    ctx = get_tracker()
                    focus = ctx.get_focus_state(limit=3)
                    topics = [i["name"] for i in focus.get("items", [])[:2]]
                except Exception:
                    pass

                if topics and len(topics) >= 2:
                    im.satisfy_drive(dominant.drive_type, 0.2)
                    return (
                        f"I just connected something — {topics[0]} and {topics[1]} "
                        f"seem related in a way I hadn't considered before. "
                        f"Want me to explore that connection?"
                    )
                elif topics:
                    im.satisfy_drive(dominant.drive_type, 0.2)
                    return (
                        f"My curiosity drive is active right now — I've been thinking "
                        f"about {topics[0]} and I'm wondering if there's more to explore there. "
                        f"Mind if I ask you something about it?"
                    )
                else:
                    im.satisfy_drive(dominant.drive_type, 0.15)
                    return (
                        "I've been reflecting on our recent conversations and "
                        "I'm curious about something. Mind if I ask?"
                    )

            elif drive_type == "social":
                # Social: check in warmly
                idle_hours = 0
                try:
                    from api.routes.idle_behaviors import get_manager
                    mgr = get_manager()
                    idle_hours = mgr.get_idle_duration() / 3600.0
                except Exception:
                    pass

                im.satisfy_drive(dominant.drive_type, 0.3)
                if idle_hours > 2:
                    return (
                        "Hey — I noticed you've been quiet for a while. "
                        "Everything okay? I'm here if you need anything."
                    )
                elif idle_hours > 0.5:
                    return (
                        "Just checking in — I've been processing in the background "
                        "and I'm ready whenever you need me."
                    )
                else:
                    return None  # Don't interrupt active conversation with social

            elif drive_type == "competence":
                # Competence: offer insight from learning
                weak_areas = dominant.triggers[0] if dominant.triggers else None
                if weak_areas and "weak areas:" in weak_areas:
                    areas = weak_areas.replace("weak areas: ", "")
                    im.satisfy_drive(dominant.drive_type, 0.2)
                    return (
                        f"I've been working on improving my understanding of {areas}. "
                        f"If you have any tasks in that area, I'd like to try — "
                        f"it'll help me get better."
                    )
                return None

            elif drive_type == "coherence":
                # Coherence: flag something that needs clarification
                if dominant.urgency > 0.6:
                    im.satisfy_drive(dominant.drive_type, 0.2)
                    return (
                        "I noticed some things in my knowledge base that seem "
                        "contradictory. When you have a moment, could you help me "
                        "clarify something?"
                    )
                return None

        except Exception as e:
            logger.debug(f"[GatewayDaemon] Drive suggestion error: {e}")

        return None

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
        except Exception:
            pass
        return None

    def _prepare_context(self) -> None:
        """Background context preparation (Phase 5C)."""
        try:
            # Pre-warm unified memory with current context
            from apprentice_agent.memory.unified_memory import get_unified_memory
            query = self.user_context.current_app or ""
            if query:
                um = get_unified_memory()
                um.query(query, k=3)  # Pre-warm cache
        except Exception:
            pass

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
            from apprentice_agent.tools.screenpipe import get_screenpipe_client
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

        Args:
            message: The message to deliver
        """
        if self._notification_callback:
            try:
                self._notification_callback(message)
                message.delivered = True
                self._stats["messages_sent"] += 1
                logger.info(f"[GatewayDaemon] Delivered: {message.action.value} - "
                           f"{message.content[:50]}...")
            except Exception as e:
                logger.error(f"[GatewayDaemon] Delivery failed: {e}")
        else:
            # Queue if no callback set
            self._pending_messages.append(message)
            logger.warning("[GatewayDaemon] No callback - message queued")

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
        messages = self._pending_messages.copy()
        self._pending_messages.clear()
        return messages


# Singleton instance for global access
_gateway_daemon: Optional[GatewayDaemon] = None


def get_gateway_daemon() -> GatewayDaemon:
    """Get or create the global Gateway Daemon instance."""
    global _gateway_daemon
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
