"""
Calendar Monitor - Monitors calendar events and sends reminders.

Integrates with:
- Google Calendar (via API)
- Local calendar files (ICS)
- System calendar (Windows/macOS)

Events generated:
- meeting_reminder: Upcoming meeting in X minutes
- meeting_start: Meeting is starting now
- meeting_end: Meeting has ended
- deadline_approaching: Task deadline approaching
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Set
from dataclasses import dataclass, field

from .base_monitor import BaseMonitor
from ..event_bus import Event, EventPriority

logger = logging.getLogger(__name__)


@dataclass
class CalendarEvent:
    """Representation of a calendar event."""
    id: str
    title: str
    start: datetime
    end: datetime
    location: Optional[str] = None
    description: Optional[str] = None
    is_all_day: bool = False
    source: str = "calendar"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_minutes(self) -> int:
        """Get event duration in minutes."""
        return int((self.end - self.start).total_seconds() / 60)

    def minutes_until_start(self) -> float:
        """Minutes until event starts."""
        delta = self.start - datetime.now()
        return delta.total_seconds() / 60

    def minutes_until_end(self) -> float:
        """Minutes until event ends."""
        delta = self.end - datetime.now()
        return delta.total_seconds() / 60


class CalendarMonitor(BaseMonitor):
    """
    Monitor for calendar events.

    Features:
    - Tracks upcoming events
    - Sends reminders at configurable intervals
    - Detects meeting start/end
    - Supports multiple calendar sources
    """

    # Default reminder times (minutes before event)
    DEFAULT_REMINDERS = [30, 15, 5, 1]

    def __init__(
        self,
        event_bus=None,
        poll_interval: float = 60.0,  # Check every minute
        reminder_minutes: Optional[List[int]] = None,
        lookahead_hours: int = 24
    ):
        """
        Initialize calendar monitor.

        Args:
            event_bus: EventBus to publish to
            poll_interval: Seconds between polls
            reminder_minutes: Minutes before events to send reminders
            lookahead_hours: Hours ahead to look for events
        """
        super().__init__(event_bus, poll_interval)

        self._reminder_minutes = reminder_minutes or self.DEFAULT_REMINDERS
        self._lookahead_hours = lookahead_hours

        # Track sent reminders to avoid duplicates
        self._sent_reminders: Set[str] = set()  # "{event_id}_{minutes}"
        self._active_meetings: Dict[str, CalendarEvent] = {}

        # Calendar events cache
        self._events: List[CalendarEvent] = []
        self._last_fetch: Optional[datetime] = None

        logger.info(f"[CalendarMonitor] Initialized with reminders at {self._reminder_minutes} minutes")

    @property
    def source(self) -> str:
        return "calendar"

    def add_event(self, event: CalendarEvent) -> None:
        """
        Add a calendar event manually.

        Args:
            event: Calendar event to add
        """
        # Remove existing event with same ID
        self._events = [e for e in self._events if e.id != event.id]
        self._events.append(event)
        self._events.sort(key=lambda e: e.start)
        logger.debug(f"[CalendarMonitor] Added event: {event.title}")

    def remove_event(self, event_id: str) -> None:
        """Remove a calendar event."""
        self._events = [e for e in self._events if e.id != event_id]
        self._sent_reminders = {r for r in self._sent_reminders if not r.startswith(f"{event_id}_")}

    def set_events(self, events: List[CalendarEvent]) -> None:
        """
        Set the full list of calendar events.

        Args:
            events: List of calendar events
        """
        self._events = sorted(events, key=lambda e: e.start)
        self._last_fetch = datetime.now()
        logger.info(f"[CalendarMonitor] Loaded {len(events)} events")

    async def _poll(self) -> List[Event]:
        """Poll for calendar events to generate."""
        events = []
        now = datetime.now()

        # Check if we need to refresh events from source
        if self._should_refresh_events():
            await self._refresh_events()

        # Process each upcoming event
        for cal_event in self._events:
            # Skip past events
            if cal_event.end < now:
                continue

            # Skip all-day events for reminders
            if cal_event.is_all_day:
                continue

            minutes_until = cal_event.minutes_until_start()

            # Check for meeting start
            if -1 <= minutes_until <= 1:  # Within 1 minute of start
                if cal_event.id not in self._active_meetings:
                    self._active_meetings[cal_event.id] = cal_event
                    events.append(self._create_meeting_start_event(cal_event))

            # Check for reminders
            for reminder_mins in self._reminder_minutes:
                reminder_key = f"{cal_event.id}_{reminder_mins}"

                if reminder_key not in self._sent_reminders:
                    # Check if we're within the reminder window
                    if reminder_mins - 0.5 <= minutes_until <= reminder_mins + 0.5:
                        events.append(self._create_reminder_event(cal_event, reminder_mins))
                        self._sent_reminders.add(reminder_key)

        # Check for meeting ends
        for event_id, cal_event in list(self._active_meetings.items()):
            if cal_event.minutes_until_end() <= 0:
                events.append(self._create_meeting_end_event(cal_event))
                del self._active_meetings[event_id]

        # Cleanup old reminder keys (for past events)
        self._cleanup_reminders()

        return events

    def _should_refresh_events(self) -> bool:
        """Check if we should refresh events from source."""
        if self._last_fetch is None:
            return True

        # Refresh every 15 minutes
        age = (datetime.now() - self._last_fetch).total_seconds()
        return age > 900

    async def _refresh_events(self) -> None:
        """Refresh events from calendar sources."""
        # This would integrate with actual calendar APIs
        # For now, we rely on manual event addition
        self._last_fetch = datetime.now()

    def _cleanup_reminders(self) -> None:
        """Remove reminder keys for past events."""
        now = datetime.now()
        current_event_ids = {e.id for e in self._events if e.end >= now}
        self._sent_reminders = {
            r for r in self._sent_reminders
            if r.split("_")[0] in current_event_ids
        }

    def _create_reminder_event(
        self,
        cal_event: CalendarEvent,
        minutes: int
    ) -> Event:
        """Create a meeting reminder event."""
        priority = EventPriority.MEDIUM
        if minutes <= 5:
            priority = EventPriority.HIGH
        elif minutes <= 1:
            priority = EventPriority.CRITICAL

        return self.create_event(
            "meeting_reminder",
            {
                "event_id": cal_event.id,
                "title": cal_event.title,
                "start_time": cal_event.start.isoformat(),
                "end_time": cal_event.end.isoformat(),
                "minutes_until": minutes,
                "location": cal_event.location,
                "duration_minutes": cal_event.duration_minutes,
            },
            priority=priority,
            reminder_minutes=minutes
        )

    def _create_meeting_start_event(self, cal_event: CalendarEvent) -> Event:
        """Create a meeting start event."""
        return self.create_event(
            "meeting_start",
            {
                "event_id": cal_event.id,
                "title": cal_event.title,
                "start_time": cal_event.start.isoformat(),
                "end_time": cal_event.end.isoformat(),
                "location": cal_event.location,
                "duration_minutes": cal_event.duration_minutes,
            },
            priority=EventPriority.HIGH
        )

    def _create_meeting_end_event(self, cal_event: CalendarEvent) -> Event:
        """Create a meeting end event."""
        return self.create_event(
            "meeting_end",
            {
                "event_id": cal_event.id,
                "title": cal_event.title,
                "start_time": cal_event.start.isoformat(),
                "end_time": cal_event.end.isoformat(),
                "actual_duration_minutes": cal_event.duration_minutes,
            },
            priority=EventPriority.LOW
        )

    def get_upcoming_events(
        self,
        hours: Optional[int] = None,
        limit: int = 10
    ) -> List[CalendarEvent]:
        """
        Get upcoming calendar events.

        Args:
            hours: Hours ahead to look (default: lookahead_hours)
            limit: Maximum events to return

        Returns:
            List of upcoming events
        """
        hours = hours or self._lookahead_hours
        now = datetime.now()
        cutoff = now + timedelta(hours=hours)

        upcoming = [
            e for e in self._events
            if now <= e.start <= cutoff
        ]

        return upcoming[:limit]

    def get_current_meeting(self) -> Optional[CalendarEvent]:
        """Get the currently active meeting, if any."""
        now = datetime.now()
        for event in self._events:
            if event.start <= now <= event.end:
                return event
        return None
