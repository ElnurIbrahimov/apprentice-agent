"""
Event Bus - Pub/Sub system for proactive events.

Supports both Redis (production) and in-memory (development) backends.
Events flow from monitors -> event bus -> salience filter -> daemon.
"""

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import IntEnum
from typing import Any, Callable, Dict, List, Set

logger = logging.getLogger(__name__)


class EventPriority(IntEnum):
    """Priority levels for events."""
    CRITICAL = 1    # Immediate attention needed
    HIGH = 2        # Important, handle soon
    MEDIUM = 3      # Normal priority
    LOW = 4         # Background, handle when convenient
    BACKGROUND = 5  # Lowest priority, process during idle


@dataclass
class Event:
    """
    An event from a monitor source.

    Events are the fundamental unit of proactive processing.
    They flow through: Source -> EventBus -> SalienceFilter -> GatewayDaemon
    """
    source: str                          # Monitor that generated this (e.g., "calendar", "screen")
    event_type: str                      # Type within source (e.g., "meeting_reminder", "app_change")
    payload: Dict[str, Any]              # Event-specific data
    priority: EventPriority = EventPriority.MEDIUM
    timestamp: float = field(default_factory=time.time)
    event_id: str = field(default_factory=lambda: f"{time.time():.6f}")
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __lt__(self, other: 'Event') -> bool:
        """Compare by priority (lower = higher priority) then timestamp."""
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.timestamp < other.timestamp

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['priority'] = self.priority.value
        return data

    @classmethod
    def from_dict(cls, data: dict) -> 'Event':
        """Create from dictionary."""
        data['priority'] = EventPriority(data.get('priority', 3))
        return cls(**data)

    def age_seconds(self) -> float:
        """Get event age in seconds."""
        return time.time() - self.timestamp

    def is_stale(self, max_age_seconds: float = 300) -> bool:
        """Check if event is too old to be relevant."""
        return self.age_seconds() > max_age_seconds


class EventBusBackend(ABC):
    """Abstract backend for event bus."""

    @abstractmethod
    async def publish(self, channel: str, event: Event) -> bool:
        """Publish event to channel."""
        pass

    @abstractmethod
    async def subscribe(self, channels: List[str], callback: Callable[[Event], None]) -> None:
        """Subscribe to channels with callback."""
        pass

    @abstractmethod
    async def unsubscribe(self, channels: List[str]) -> None:
        """Unsubscribe from channels."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close the backend connection."""
        pass


class InMemoryBackend(EventBusBackend):
    """
    In-memory event bus backend for development/testing.

    Uses asyncio queues for pub/sub within the same process.
    """

    def __init__(self):
        self._channels: Dict[str, List[asyncio.Queue]] = {}
        self._subscriptions: Dict[str, Set[asyncio.Queue]] = {}
        self._lock = asyncio.Lock()
        self._running = True
        logger.info("[EventBus] In-memory backend initialized")

    def _get_lock(self) -> asyncio.Lock:
        """Return the asyncio lock (eagerly initialized in __init__)."""
        return self._lock

    async def publish(self, channel: str, event: Event) -> bool:
        """Publish event to all subscribers of a channel."""
        async with self._get_lock():
            if channel not in self._channels:
                self._channels[channel] = []

            # Send to all queues subscribed to this channel
            for queue in self._channels[channel]:
                try:
                    await queue.put(event)
                except Exception as e:
                    logger.error(f"[EventBus] Failed to publish to queue: {e}")

            return True

    async def subscribe(self, channels: List[str], callback: Callable[[Event], None]) -> None:
        """Subscribe to channels and process events with callback."""
        queue: asyncio.Queue = asyncio.Queue()

        async with self._get_lock():
            for channel in channels:
                if channel not in self._channels:
                    self._channels[channel] = []
                self._channels[channel].append(queue)

        # Process events from queue
        try:
            while self._running:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=1.0)
                    try:
                        callback(event)
                    except Exception as e:
                        logger.error(f"[EventBus] Callback error: {e}")
                except asyncio.TimeoutError:
                    continue
        finally:
            # Cleanup
            async with self._get_lock():
                for channel in channels:
                    if channel in self._channels:
                        try:
                            self._channels[channel].remove(queue)
                        except ValueError:
                            pass

    async def unsubscribe(self, channels: List[str]) -> None:
        """Unsubscribe from channels (handled in subscribe cleanup)."""
        pass

    async def close(self) -> None:
        """Close the backend."""
        self._running = False
        logger.info("[EventBus] In-memory backend closed")


class RedisBackend(EventBusBackend):
    """
    Redis-based event bus backend for production.

    Enables distributed event processing across multiple processes.
    """

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self._redis = None
        self._pubsub = None
        self._running = True
        logger.info(f"[EventBus] Redis backend initialized: {redis_url}")

    async def _ensure_connected(self):
        """Ensure Redis connection is established."""
        if self._redis is None:
            try:
                import redis.asyncio as redis
                self._redis = await redis.from_url(self.redis_url)
                logger.info("[EventBus] Redis connected")
            except ImportError:
                logger.error("[EventBus] redis package not installed")
                raise RuntimeError("redis package required for RedisBackend") from None
            except Exception as e:
                logger.error(f"[EventBus] Redis connection failed: {e}")
                raise

    async def publish(self, channel: str, event: Event) -> bool:
        """Publish event to Redis channel."""
        await self._ensure_connected()
        try:
            message = json.dumps(event.to_dict())
            await self._redis.publish(channel, message)
            return True
        except Exception as e:
            logger.error(f"[EventBus] Redis publish error: {e}")
            return False

    async def subscribe(self, channels: List[str], callback: Callable[[Event], None]) -> None:
        """Subscribe to Redis channels."""
        await self._ensure_connected()

        self._pubsub = self._redis.pubsub()
        await self._pubsub.subscribe(*channels)

        try:
            async for message in self._pubsub.listen():
                if not self._running:
                    break

                if message["type"] == "message":
                    try:
                        event_data = json.loads(message["data"])
                        event = Event.from_dict(event_data)
                        callback(event)
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.error(f"[EventBus] Invalid event message: {e}")
        finally:
            await self._pubsub.unsubscribe(*channels)

    async def unsubscribe(self, channels: List[str]) -> None:
        """Unsubscribe from Redis channels."""
        if self._pubsub:
            await self._pubsub.unsubscribe(*channels)

    async def close(self) -> None:
        """Close Redis connection."""
        self._running = False
        if self._pubsub:
            await self._pubsub.close()
        if self._redis:
            await self._redis.close()
        logger.info("[EventBus] Redis backend closed")


class EventBus:
    """
    Main event bus for proactive system.

    Provides a unified interface for publishing and subscribing to events
    regardless of backend (in-memory or Redis).

    Usage:
        bus = EventBus()
        await bus.start()

        # Publish an event
        event = Event(source="calendar", event_type="meeting_soon", payload={...})
        await bus.publish("calendar", event)

        # Subscribe to events
        await bus.subscribe(["calendar", "screen"], my_handler)
    """

    # Standard channels
    CHANNELS = {
        "calendar": "Calendar and scheduling events",
        "email": "Email notifications",
        "screen": "Screen/UI change events",
        "file_system": "File system changes",
        "system": "System events (low battery, etc.)",
        "user": "User activity events",
        "aura": "AURA internal events",
        "workspace": "Global Workspace conscious broadcasts",
    }

    def __init__(self, use_redis: bool = False, redis_url: str = "redis://localhost:6379"):
        """
        Initialize event bus.

        Args:
            use_redis: Use Redis backend (False = in-memory)
            redis_url: Redis URL if using Redis backend
        """
        if use_redis:
            self._backend = RedisBackend(redis_url)
        else:
            self._backend = InMemoryBackend()

        self._subscribers: Dict[str, List[Callable]] = {}
        self._running = False
        self._stats = {
            "events_published": 0,
            "events_processed": 0,
            "errors": 0,
        }

    async def start(self) -> None:
        """Start the event bus."""
        self._running = True
        logger.info("[EventBus] Started")

    async def stop(self) -> None:
        """Stop the event bus."""
        self._running = False
        await self._backend.close()
        logger.info("[EventBus] Stopped")

    async def publish(self, channel: str, event: Event) -> bool:
        """
        Publish an event to a channel.

        Args:
            channel: Channel name (e.g., "calendar", "screen")
            event: Event to publish

        Returns:
            True if published successfully
        """
        if not self._running:
            logger.warning("[EventBus] Attempted to publish while stopped")
            return False

        success = await self._backend.publish(channel, event)
        if success:
            self._stats["events_published"] += 1
        else:
            self._stats["errors"] += 1

        return success

    async def subscribe(
        self,
        channels: List[str],
        callback: Callable[[Event], None]
    ) -> None:
        """
        Subscribe to channels with a callback.

        Args:
            channels: List of channel names
            callback: Function to call for each event
        """
        def wrapped_callback(event: Event):
            try:
                callback(event)
                self._stats["events_processed"] += 1
            except Exception as e:
                logger.error(f"[EventBus] Subscriber callback error: {e}")
                self._stats["errors"] += 1

        await self._backend.subscribe(channels, wrapped_callback)

    def get_stats(self) -> Dict[str, Any]:
        """Get event bus statistics."""
        return {
            **self._stats,
            "running": self._running,
        }


# Convenience functions for creating common events

def create_calendar_event(
    event_type: str,
    title: str,
    start_time: datetime,
    priority: EventPriority = EventPriority.MEDIUM,
    **extra
) -> Event:
    """Create a calendar event."""
    return Event(
        source="calendar",
        event_type=event_type,
        priority=priority,
        payload={
            "title": title,
            "start_time": start_time.isoformat(),
            **extra
        }
    )


def create_screen_event(
    event_type: str,
    app_name: str,
    window_title: str,
    **extra
) -> Event:
    """Create a screen/UI event."""
    return Event(
        source="screen",
        event_type=event_type,
        priority=EventPriority.LOW,
        payload={
            "app_name": app_name,
            "window_title": window_title,
            **extra
        }
    )


def create_system_event(
    event_type: str,
    priority: EventPriority = EventPriority.MEDIUM,
    **payload
) -> Event:
    """Create a system event."""
    return Event(
        source="system",
        event_type=event_type,
        priority=priority,
        payload=payload
    )


if __name__ == "__main__":
    import asyncio

    async def test():
        print("=" * 60)
        print("EventBus Test")
        print("=" * 60)

        bus = EventBus(use_redis=False)
        await bus.start()

        received_events = []

        def handler(event: Event):
            print(f"  Received: [{event.source}] {event.event_type}")
            received_events.append(event)

        # Start subscriber in background
        subscriber_task = asyncio.create_task(
            bus.subscribe(["calendar", "screen"], handler)
        )

        # Give subscriber time to start
        await asyncio.sleep(0.1)

        # Publish some events
        print("\n--- Publishing events ---")

        await bus.publish("calendar", create_calendar_event(
            "meeting_reminder",
            "Team Standup",
            datetime.now(),
            minutes_until=15
        ))

        await bus.publish("screen", create_screen_event(
            "app_change",
            "VSCode",
            "gateway_daemon.py"
        ))

        # Wait for processing
        await asyncio.sleep(0.5)

        print(f"\n--- Received {len(received_events)} events ---")
        print(f"Stats: {bus.get_stats()}")

        # Cleanup
        await bus.stop()
        subscriber_task.cancel()

        print("\n" + "=" * 60)
        print("Test complete!")

    asyncio.run(test())
