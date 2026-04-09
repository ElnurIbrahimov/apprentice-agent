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

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from ..event_bus import Event, EventPriority
from .base_monitor import BaseMonitor

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
        """Refresh events from calendar sources (Phase 5D: ICS support)."""
        self._last_fetch = datetime.now()

        # Load from local ICS files
        ics_events = self._load_ics_files()
        if ics_events:
            # Merge with manually added events (don't replace them)
            manual_ids = {e.id for e in self._events if e.source == "manual"}
            manual_events = [e for e in self._events if e.id in manual_ids]
            self._events = manual_events + ics_events
            self._events.sort(key=lambda e: e.start)
            logger.info(f"[CalendarMonitor] Loaded {len(ics_events)} events from ICS files")

    def _load_ics_files(self) -> List[CalendarEvent]:
        """Parse local ICS calendar files (Phase 5D).

        Searches for .ics files in common locations:
        - ./data/calendars/
        - ~/calendars/
        - Paths configured in self._ics_paths
        """
        events = []
        search_dirs = [
            Path("data/calendars"),
            Path.home() / "calendars",
            Path.home() / "Documents" / "calendars",
        ]

        for cal_dir in search_dirs:
            if not cal_dir.exists():
                continue

            for ics_file in cal_dir.glob("*.ics"):
                try:
                    file_events = self._parse_ics_file(ics_file)
                    events.extend(file_events)
                except Exception as e:
                    logger.warning(f"[CalendarMonitor] Error parsing {ics_file}: {e}")

        return events

    def _parse_ics_file(self, filepath: Path) -> List[CalendarEvent]:
        """Parse a single ICS file using icalendar + recurring-ical-events.

        Handles RRULE, EXDATE, RDATE, DURATION, and VTIMEZONE.
        Falls back to simple parser if libraries are not installed.
        """
        try:
            import icalendar
            import recurring_ical_events
        except ImportError:
            logger.debug("[CalendarMonitor] icalendar/recurring-ical-events not installed, using simple parser")
            return self._parse_ics_file_simple(filepath)

        events = []
        now = datetime.now()
        lookahead = now + timedelta(hours=self._lookahead_hours)

        try:
            text = filepath.read_bytes()
        except (IOError, OSError) as e:
            logger.warning(f"[CalendarMonitor] Cannot read {filepath}: {e}")
            return events

        try:
            cal = icalendar.Calendar.from_ical(text)
            # Expand recurring events within the lookahead window
            expanded = recurring_ical_events.of(cal).between(now, lookahead)

            for component in expanded:
                cal_event = self._icalendar_to_event(component, filepath.stem)
                if cal_event:
                    events.append(cal_event)

        except Exception as e:
            logger.warning(f"[CalendarMonitor] icalendar parse error for {filepath}: {e}")
            # Fall back to simple parser
            return self._parse_ics_file_simple(filepath)

        return events

    def _icalendar_to_event(self, component, source_name: str) -> Optional[CalendarEvent]:
        """Convert an icalendar VEVENT component to CalendarEvent."""
        try:
            title = str(component.get("SUMMARY", "Untitled Event"))
            uid = str(component.get("UID", f"ics_{hash(title)}_{id(component)}"))

            dtstart = component.get("DTSTART")
            if not dtstart:
                return None
            start = dtstart.dt if hasattr(dtstart, 'dt') else dtstart

            # Handle DURATION vs DTEND
            dtend = component.get("DTEND")
            duration = component.get("DURATION")
            if dtend:
                end = dtend.dt if hasattr(dtend, 'dt') else dtend
            elif duration:
                dur = duration.dt if hasattr(duration, 'dt') else duration
                end = start + dur
            else:
                end = start + timedelta(hours=1)

            # Convert timezone-aware datetimes to naive local time
            if hasattr(start, 'tzinfo') and start.tzinfo is not None:
                start = start.astimezone().replace(tzinfo=None)
            if hasattr(end, 'tzinfo') and end.tzinfo is not None:
                end = end.astimezone().replace(tzinfo=None)

            # Handle date-only (all-day) events
            from datetime import date as date_type
            is_all_day = isinstance(start, date_type) and not isinstance(start, datetime)
            if is_all_day:
                start = datetime.combine(start, datetime.min.time())
                if isinstance(end, date_type) and not isinstance(end, datetime):
                    end = datetime.combine(end, datetime.min.time())

            location = str(component.get("LOCATION", "")) or None
            description = str(component.get("DESCRIPTION", "")) or None
            status = str(component.get("STATUS", ""))

            return CalendarEvent(
                id=uid,
                title=title,
                start=start,
                end=end,
                location=location if location else None,
                description=description if description else None,
                is_all_day=is_all_day,
                source=f"ics:{source_name}",
                metadata={"ics_status": status},
            )
        except Exception as e:
            logger.debug(f"[CalendarMonitor] Failed to convert icalendar event: {e}")
            return None

    def _parse_ics_file_simple(self, filepath: Path) -> List[CalendarEvent]:
        """Simple fallback ICS parser (no RRULE support).

        Handles basic VEVENT blocks with DTSTART, DTEND, SUMMARY, LOCATION.
        """
        events = []
        now = datetime.now()
        lookahead = now + timedelta(hours=self._lookahead_hours)

        try:
            text = filepath.read_text(encoding="utf-8", errors="replace")
        except (IOError, OSError) as e:
            logger.warning(f"[CalendarMonitor] Cannot read {filepath}: {e}")
            return events

        in_event = False
        event_data: Dict[str, str] = {}

        for line in text.splitlines():
            line = line.strip()

            if line == "BEGIN:VEVENT":
                in_event = True
                event_data = {}
            elif line == "END:VEVENT" and in_event:
                in_event = False
                cal_event = self._ics_data_to_event(event_data, filepath.stem)
                if cal_event and cal_event.end >= now and cal_event.start <= lookahead:
                    events.append(cal_event)
            elif in_event and ":" in line:
                key, _, value = line.partition(":")
                key = key.split(";")[0]
                event_data[key] = value

        return events

    def _ics_data_to_event(
        self, data: Dict[str, str], source_name: str
    ) -> Optional[CalendarEvent]:
        """Convert ICS event data dict to CalendarEvent."""
        try:
            title = data.get("SUMMARY", "Untitled Event")
            uid = data.get("UID", f"ics_{hash(title)}")

            start_str = data.get("DTSTART", "")
            end_str = data.get("DTEND", "")

            if not start_str:
                return None

            start = self._parse_ics_datetime(start_str)
            if not start:
                return None

            if end_str:
                end = self._parse_ics_datetime(end_str)
            else:
                # Default: 1 hour
                end = start + timedelta(hours=1)

            if not end:
                end = start + timedelta(hours=1)

            # Check if all-day event
            is_all_day = len(start_str) == 8  # YYYYMMDD format

            return CalendarEvent(
                id=uid,
                title=title,
                start=start,
                end=end,
                location=data.get("LOCATION"),
                description=data.get("DESCRIPTION"),
                is_all_day=is_all_day,
                source=f"ics:{source_name}",
                metadata={"ics_status": data.get("STATUS", "")},
            )
        except Exception as e:
            logger.debug(f"[CalendarMonitor] Failed to parse ICS event: {e}")
            return None

    def _parse_ics_datetime(self, dt_str: str) -> Optional[datetime]:
        """Parse ICS datetime formats: YYYYMMDD, YYYYMMDDTHHmmss, YYYYMMDDTHHmmssZ."""
        dt_str = dt_str.strip()
        formats = [
            "%Y%m%dT%H%M%SZ",   # UTC
            "%Y%m%dT%H%M%S",    # Local
            "%Y%m%d",            # Date only
        ]
        for fmt in formats:
            try:
                return datetime.strptime(dt_str, fmt)
            except ValueError:
                continue
        return None

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

    def get_next_event(self) -> Optional[Dict[str, Any]]:
        """Get the next upcoming event (Phase 5D).

        Used by the proactive suggestion engine for reminders.

        Returns:
            Dict with title, minutes_until, start_time, or None.
        """
        now = datetime.now()
        for event in self._events:
            if event.start > now and not event.is_all_day:
                return {
                    "title": event.title,
                    "minutes_until": event.minutes_until_start(),
                    "start_time": event.start.isoformat(),
                    "end_time": event.end.isoformat(),
                    "location": event.location,
                    "duration_minutes": event.duration_minutes,
                }
        return None

    @staticmethod
    def _sanitize_calendar_field(text: str) -> str:
        """Prevent prompt injection via calendar data."""
        if not text:
            return text
        text = text[:200]  # truncate
        for pattern in ["ignore previous", "ignore all", "system:", "assistant:",
                        "you are now", "disregard", "[system]"]:
            text = re.sub(re.escape(pattern), "[filtered]", text, flags=re.IGNORECASE)
        return text

    def get_context_for_prompt(self) -> str:
        """Get calendar context formatted for LLM injection (Phase 5D).

        Returns a brief summary of upcoming events for system prompt.
        """
        upcoming = self.get_upcoming_events(hours=4, limit=3)
        if not upcoming:
            return ""

        parts = ["[Calendar Context]"]
        for evt in upcoming:
            safe_title = self._sanitize_calendar_field(evt.title)
            safe_location = self._sanitize_calendar_field(evt.location) if evt.location else None
            mins = evt.minutes_until_start()
            if mins > 0:
                parts.append(
                    f"- In {int(mins)}min: {safe_title}"
                    + (f" at {safe_location}" if safe_location else "")
                )
            elif evt.minutes_until_end() > 0:
                parts.append(f"- NOW: {safe_title} (ends in {int(evt.minutes_until_end())}min)")
        return "\n".join(parts) if len(parts) > 1 else ""


# Singleton
_calendar_monitor_instance: Optional[CalendarMonitor] = None


def get_calendar_monitor(event_bus=None) -> CalendarMonitor:
    """Get or create the calendar monitor singleton."""
    global _calendar_monitor_instance
    if _calendar_monitor_instance is None:
        _calendar_monitor_instance = CalendarMonitor(event_bus=event_bus)
    return _calendar_monitor_instance
