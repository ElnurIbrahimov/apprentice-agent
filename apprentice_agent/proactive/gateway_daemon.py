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
        if event.source == "screen" and event.event_type == "app_change":
            observations["context_changes"] = 0.8

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

        # Deliver
        self._deliver_message(message)

    def _generate_message_content(
        self,
        action: ProactiveAction,
        event: Optional[Event] = None
    ) -> Optional[str]:
        """
        Generate content for a proactive message.

        Args:
            action: The action type
            event: Optional triggering event

        Returns:
            Message content or None if no message needed
        """
        # For urgent events, use event-specific content
        if event:
            return self._event_to_message(event)

        # For proactive decisions, generate based on beliefs
        beliefs = self.inference_engine.get_beliefs()

        if action == ProactiveAction.SUGGEST:
            if beliefs.task_urgent > 0.5:
                return "I notice you might need help with something. Would you like me to assist?"
            return None

        elif action == ProactiveAction.REMIND:
            # Would integrate with calendar/task system
            return None

        elif action == ProactiveAction.ASK:
            if beliefs.uncertainty > 0.6:
                return "I'm not sure what you're working on. Could you tell me more about your current task?"
            return None

        elif action == ProactiveAction.PREPARE:
            # Background preparation - no message needed
            return None

        elif action == ProactiveAction.INTERVENE:
            if beliefs.task_urgent > 0.8:
                return "This seems urgent. Let me help you with this."
            return None

        return None

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

        elif event.source == "system":
            if event.event_type == "security_warning":
                return f"Security alert: {payload.get('message', 'Unknown issue')}"
            elif event.event_type == "system_alert":
                return f"System alert: {payload.get('message', 'Unknown issue')}"

        # Generic fallback
        return f"[{event.source}] {event.event_type}: {event.payload}"

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
