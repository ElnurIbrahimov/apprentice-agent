"""
Base Monitor - Abstract base class for all monitors.

Monitors are responsible for:
1. Watching a specific source (screen, calendar, system, etc.)
2. Detecting interesting events
3. Publishing events to the EventBus

Each monitor runs in its own asyncio task or thread.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Optional, Callable, List, Any

from ..event_bus import Event, EventBus, EventPriority

logger = logging.getLogger(__name__)


class MonitorState(Enum):
    """State of a monitor."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"


class BaseMonitor(ABC):
    """
    Abstract base class for event monitors.

    Subclasses must implement:
    - _poll(): Check for events (called periodically)
    - source property: Return source name (e.g., "screen", "calendar")

    Optional overrides:
    - _on_start(): Called when monitor starts
    - _on_stop(): Called when monitor stops
    """

    def __init__(
        self,
        event_bus: Optional[EventBus] = None,
        poll_interval: float = 5.0
    ):
        """
        Initialize the monitor.

        Args:
            event_bus: EventBus to publish to (or creates internal one)
            poll_interval: Seconds between polls
        """
        self._event_bus = event_bus
        self._poll_interval = poll_interval
        self._state = MonitorState.STOPPED
        self._task: Optional[asyncio.Task] = None
        self._error: Optional[str] = None

        # Statistics
        self._stats = {
            "events_published": 0,
            "polls": 0,
            "errors": 0,
            "start_time": None,
        }

        logger.debug(f"[{self.source}Monitor] Initialized")

    @property
    @abstractmethod
    def source(self) -> str:
        """Return the source name for events from this monitor."""
        pass

    @property
    def state(self) -> MonitorState:
        """Current monitor state."""
        return self._state

    @property
    def is_running(self) -> bool:
        """Check if monitor is running."""
        return self._state == MonitorState.RUNNING

    @abstractmethod
    async def _poll(self) -> List[Event]:
        """
        Poll for new events.

        Subclasses should implement this to check their source
        and return any new events detected.

        Returns:
            List of events to publish
        """
        pass

    async def _on_start(self) -> None:
        """Called when monitor starts. Override for setup."""
        pass

    async def _on_stop(self) -> None:
        """Called when monitor stops. Override for cleanup."""
        pass

    async def start(self) -> None:
        """Start the monitor."""
        if self._state != MonitorState.STOPPED:
            logger.warning(f"[{self.source}Monitor] Cannot start - state: {self._state}")
            return

        self._state = MonitorState.STARTING
        logger.info(f"[{self.source}Monitor] Starting...")

        try:
            await self._on_start()
            self._task = asyncio.create_task(self._run_loop())
            self._state = MonitorState.RUNNING
            self._stats["start_time"] = datetime.now()
            logger.info(f"[{self.source}Monitor] Started")

        except Exception as e:
            self._state = MonitorState.ERROR
            self._error = str(e)
            logger.error(f"[{self.source}Monitor] Start failed: {e}")
            raise

    async def stop(self) -> None:
        """Stop the monitor."""
        if self._state not in (MonitorState.RUNNING, MonitorState.PAUSED):
            logger.warning(f"[{self.source}Monitor] Cannot stop - state: {self._state}")
            return

        logger.info(f"[{self.source}Monitor] Stopping...")

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        try:
            await self._on_stop()
        except Exception as e:
            logger.error(f"[{self.source}Monitor] Stop cleanup failed: {e}")

        self._state = MonitorState.STOPPED
        logger.info(f"[{self.source}Monitor] Stopped")

    def pause(self) -> None:
        """Pause the monitor (stop polling but stay ready)."""
        if self._state == MonitorState.RUNNING:
            self._state = MonitorState.PAUSED
            logger.info(f"[{self.source}Monitor] Paused")

    def resume(self) -> None:
        """Resume a paused monitor."""
        if self._state == MonitorState.PAUSED:
            self._state = MonitorState.RUNNING
            logger.info(f"[{self.source}Monitor] Resumed")

    async def _run_loop(self) -> None:
        """Main polling loop."""
        while self._state in (MonitorState.RUNNING, MonitorState.PAUSED):
            try:
                if self._state == MonitorState.PAUSED:
                    await asyncio.sleep(1.0)
                    continue

                # Poll for events
                events = await self._poll()
                self._stats["polls"] += 1

                # Publish events
                for event in events:
                    await self._publish(event)

                # Wait for next poll
                await asyncio.sleep(self._poll_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._stats["errors"] += 1
                logger.error(f"[{self.source}Monitor] Poll error: {e}")
                await asyncio.sleep(self._poll_interval)

    async def _publish(self, event: Event) -> bool:
        """
        Publish an event to the event bus.

        Args:
            event: Event to publish

        Returns:
            True if published successfully
        """
        if not self._event_bus:
            logger.warning(f"[{self.source}Monitor] No event bus - event dropped")
            return False

        success = await self._event_bus.publish(event.source, event)
        if success:
            self._stats["events_published"] += 1
            logger.debug(f"[{self.source}Monitor] Published: {event.event_type}")
        return success

    def create_event(
        self,
        event_type: str,
        payload: dict,
        priority: EventPriority = EventPriority.MEDIUM,
        **metadata
    ) -> Event:
        """
        Create an event with this monitor's source.

        Args:
            event_type: Type of event
            payload: Event data
            priority: Event priority
            **metadata: Additional metadata

        Returns:
            Event instance
        """
        return Event(
            source=self.source,
            event_type=event_type,
            payload=payload,
            priority=priority,
            metadata=metadata
        )

    def get_stats(self) -> dict:
        """Get monitor statistics."""
        uptime = None
        if self._stats["start_time"]:
            uptime = (datetime.now() - self._stats["start_time"]).total_seconds()

        return {
            **self._stats,
            "source": self.source,
            "state": self._state.value,
            "uptime_seconds": uptime,
            "poll_interval": self._poll_interval,
            "error": self._error,
        }
