"""Calendar tool for managing events, appointments, and schedules."""

import json
import os
import re
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any


@dataclass
class CalendarEvent:
    """A calendar event."""
    id: str
    title: str
    start: str                          # ISO 8601
    end: Optional[str] = None           # ISO 8601 or None (all-day)
    description: str = ""
    location: str = ""
    recurrence: Optional[str] = None    # daily/weekly/monthly/yearly
    reminders: List[int] = field(default_factory=list)  # minutes before
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CalendarEvent':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class CalendarTool:
    """Tool for managing events, appointments, and schedules."""

    name = "calendar"
    description = "Manage events, appointments, and schedules"

    def __init__(self, user_id: str = "default"):
        self.user_id = user_id
        data_dir = Path(__file__).parent.parent.parent / "data" / "users" / user_id
        data_dir.mkdir(parents=True, exist_ok=True)
        self.calendar_file = data_dir / "calendar_events.json"
        self._ensure_file()

    def _ensure_file(self):
        """Ensure the events file and directory exist."""
        self.calendar_file.parent.mkdir(parents=True, exist_ok=True)
        if not self.calendar_file.exists():
            self._save_events([])

    def _load_events(self) -> List[Dict[str, Any]]:
        """Load events from JSON file."""
        try:
            with open(self.calendar_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return []

    def _save_events(self, events: List[Dict[str, Any]]) -> bool:
        """Save events to JSON file."""
        try:
            with open(self.calendar_file, "w", encoding="utf-8") as f:
                json.dump(events, f, indent=4)
            return True
        except IOError:
            return False

    def _generate_id(self) -> str:
        return uuid.uuid4().hex[:8]

    def _parse_datetime(self, dt_str: str) -> Optional[datetime]:
        """Parse a datetime string (absolute or relative)."""
        if not dt_str:
            return None

        dt_str = dt_str.strip()

        # Try relative time first (reuse notification pattern)
        relative = self._parse_relative_time(dt_str)
        if relative:
            return relative

        # Try dateutil for absolute parsing
        try:
            from dateutil.parser import parse as dateutil_parse
            return dateutil_parse(dt_str)
        except Exception:
            pass

        # Manual ISO / common formats
        formats = [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
            "%Y-%m-%d",
            "%m/%d/%Y %H:%M",
            "%m/%d/%Y",
            "%d/%m/%Y %H:%M",
            "%d/%m/%Y",
            "%B %d, %Y %H:%M",
            "%B %d, %Y",
            "%b %d, %Y",
        ]
        for fmt in formats:
            try:
                return datetime.strptime(dt_str, fmt)
            except ValueError:
                continue

        # Try "today at HH:MM" or "tomorrow at HH:MM"
        today_match = re.match(r'today\s+(?:at\s+)?(\d{1,2}):(\d{2})\s*(am|pm)?', dt_str, re.IGNORECASE)
        if today_match:
            h, m = int(today_match.group(1)), int(today_match.group(2))
            ampm = today_match.group(3)
            if ampm and ampm.lower() == 'pm' and h != 12:
                h += 12
            elif ampm and ampm.lower() == 'am' and h == 12:
                h = 0
            return datetime.now().replace(hour=h, minute=m, second=0, microsecond=0)

        tomorrow_match = re.match(r'tomorrow\s+(?:at\s+)?(\d{1,2}):(\d{2})\s*(am|pm)?', dt_str, re.IGNORECASE)
        if tomorrow_match:
            h, m = int(tomorrow_match.group(1)), int(tomorrow_match.group(2))
            ampm = tomorrow_match.group(3)
            if ampm and ampm.lower() == 'pm' and h != 12:
                h += 12
            elif ampm and ampm.lower() == 'am' and h == 12:
                h = 0
            return (datetime.now() + timedelta(days=1)).replace(hour=h, minute=m, second=0, microsecond=0)

        return None

    def _parse_relative_time(self, time_str: str) -> Optional[datetime]:
        """Parse relative time like 'in 30 minutes', 'in 2 hours'."""
        time_str = time_str.lower().strip()
        if time_str.startswith("in "):
            time_str = time_str[3:]

        patterns = [
            (r"(\d+)\s*(?:minutes?|mins?)", "minutes"),
            (r"(\d+)\s*(?:hours?|hrs?)", "hours"),
            (r"(\d+)\s*(?:days?)", "days"),
        ]
        for pattern, unit in patterns:
            match = re.search(pattern, time_str)
            if match:
                value = int(match.group(1))
                now = datetime.now()
                if unit == "minutes":
                    return now + timedelta(minutes=value)
                elif unit == "hours":
                    return now + timedelta(hours=value)
                elif unit == "days":
                    return now + timedelta(days=value)
        return None

    def add_event(self, title: str, start: str, end: str = None,
                  description: str = "", location: str = "",
                  recurrence: str = None, reminders: List[int] = None) -> dict:
        """Create a new calendar event."""
        if not title:
            return {"success": False, "error": "No title provided"}

        start_dt = self._parse_datetime(start)
        if not start_dt:
            return {"success": False, "error": f"Could not parse start time: {start}"}

        end_dt = None
        if end:
            end_dt = self._parse_datetime(end)
            if not end_dt:
                return {"success": False, "error": f"Could not parse end time: {end}"}

        if recurrence and recurrence not in ("daily", "weekly", "monthly", "yearly"):
            return {"success": False, "error": f"Invalid recurrence: {recurrence}. Use daily/weekly/monthly/yearly"}

        event = CalendarEvent(
            id=self._generate_id(),
            title=title,
            start=start_dt.isoformat(),
            end=end_dt.isoformat() if end_dt else None,
            description=description,
            location=location,
            recurrence=recurrence,
            reminders=reminders or [],
        )

        events = self._load_events()
        events.append(event.to_dict())
        self._save_events(events)

        return {
            "success": True,
            "event_id": event.id,
            "title": event.title,
            "start": event.start,
            "end": event.end,
            "response": f"Event '{title}' created on {start_dt.strftime('%Y-%m-%d %H:%M')}"
        }

    def list_events(self, date: str = None, range_days: int = 1) -> dict:
        """List events for a specific date or range."""
        events = self._load_events()

        if date:
            target_dt = self._parse_datetime(date)
            if not target_dt:
                return {"success": False, "error": f"Could not parse date: {date}"}
        else:
            target_dt = datetime.now()

        start_of_range = target_dt.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_range = start_of_range + timedelta(days=range_days)

        matching = []
        for ev in events:
            try:
                ev_start = datetime.fromisoformat(ev["start"])
                if start_of_range <= ev_start < end_of_range:
                    matching.append(ev)
            except (ValueError, KeyError):
                continue

        # Also include recurring events that fall in range
        for ev in events:
            if ev.get("recurrence") and ev not in matching:
                try:
                    ev_start = datetime.fromisoformat(ev["start"])
                    occurrences = self._get_recurring_occurrences(ev, start_of_range, end_of_range)
                    for occ in occurrences:
                        occ_ev = dict(ev)
                        occ_ev["start"] = occ.isoformat()
                        occ_ev["_recurring_instance"] = True
                        matching.append(occ_ev)
                except (ValueError, KeyError):
                    continue

        matching.sort(key=lambda e: e.get("start", ""))

        formatted = []
        for ev in matching:
            start_dt = datetime.fromisoformat(ev["start"])
            time_str = start_dt.strftime("%H:%M")
            loc = f" @ {ev['location']}" if ev.get("location") else ""
            formatted.append(f"[{ev['id']}] {time_str} - {ev['title']}{loc}")

        return {
            "success": True,
            "count": len(matching),
            "events": matching,
            "formatted": "\n".join(formatted) if formatted else "No events found",
            "date_range": f"{start_of_range.strftime('%Y-%m-%d')} to {end_of_range.strftime('%Y-%m-%d')}",
            "response": f"Found {len(matching)} event(s)" + ("\n" + "\n".join(formatted) if formatted else "")
        }

    def _get_recurring_occurrences(self, event: dict, range_start: datetime, range_end: datetime) -> List[datetime]:
        """Get recurring event occurrences within a date range."""
        occurrences = []
        try:
            ev_start = datetime.fromisoformat(event["start"])
        except (ValueError, KeyError):
            return occurrences

        recurrence = event.get("recurrence")
        if not recurrence:
            return occurrences

        delta_map = {
            "daily": timedelta(days=1),
            "weekly": timedelta(weeks=1),
            "monthly": None,  # handled separately
            "yearly": None,
        }

        if recurrence in ("daily", "weekly"):
            delta = delta_map[recurrence]
            current = ev_start
            while current < range_end:
                if range_start <= current < range_end and current != ev_start:
                    occurrences.append(current)
                current += delta
                if len(occurrences) >= 50:
                    break
        elif recurrence == "monthly":
            current = ev_start
            for _ in range(120):
                month = current.month + 1
                year = current.year
                if month > 12:
                    month = 1
                    year += 1
                try:
                    current = current.replace(year=year, month=month)
                except ValueError:
                    current = current.replace(year=year, month=month, day=28)
                if current >= range_end:
                    break
                if range_start <= current < range_end:
                    occurrences.append(current)
        elif recurrence == "yearly":
            current = ev_start
            for _ in range(20):
                try:
                    current = current.replace(year=current.year + 1)
                except ValueError:
                    current = current.replace(year=current.year + 1, day=28)
                if current >= range_end:
                    break
                if range_start <= current < range_end:
                    occurrences.append(current)

        return occurrences

    def remove_event(self, event_id: str) -> dict:
        """Remove an event by ID."""
        if not event_id:
            return {"success": False, "error": "No event ID provided"}

        events = self._load_events()
        original_count = len(events)
        events = [e for e in events if e.get("id") != event_id]

        if len(events) == original_count:
            return {"success": False, "error": f"Event not found: {event_id}"}

        self._save_events(events)
        return {
            "success": True,
            "removed_id": event_id,
            "response": f"Removed event {event_id}"
        }

    def today(self) -> dict:
        """Get today's agenda."""
        return self.list_events(date=datetime.now().strftime("%Y-%m-%d"), range_days=1)

    def upcoming(self, days: int = 7) -> dict:
        """Get upcoming events for the next N days."""
        return self.list_events(date=datetime.now().strftime("%Y-%m-%d"), range_days=days)

    def search_events(self, query: str) -> dict:
        """Full-text search across event titles and descriptions."""
        if not query:
            return {"success": False, "error": "No search query provided"}

        events = self._load_events()
        query_lower = query.lower()
        matching = [
            e for e in events
            if query_lower in e.get("title", "").lower()
            or query_lower in e.get("description", "").lower()
            or query_lower in e.get("location", "").lower()
        ]

        formatted = []
        for ev in matching:
            start_dt = datetime.fromisoformat(ev["start"])
            formatted.append(f"[{ev['id']}] {start_dt.strftime('%Y-%m-%d %H:%M')} - {ev['title']}")

        return {
            "success": True,
            "count": len(matching),
            "events": matching,
            "formatted": "\n".join(formatted) if formatted else "No events found",
            "response": f"Found {len(matching)} event(s) matching '{query}'" + ("\n" + "\n".join(formatted) if formatted else "")
        }

    def import_ics(self, path: str) -> dict:
        """Import events from an .ics file."""
        try:
            from icalendar import Calendar as iCalCalendar
        except ImportError:
            return {"success": False, "error": "icalendar library not installed. Run: pip install icalendar"}

        ALLOWED_DIRS = [Path.home() / "Downloads", Path.home() / "Documents", Path(__file__).parent.parent.parent / "data"]
        resolved = Path(path).resolve()
        if not any(str(resolved).startswith(str(d.resolve()) + os.sep) or str(resolved) == str(d.resolve()) for d in ALLOWED_DIRS):
            return {"success": False, "error": "ICS path not in an allowed directory"}

        ics_path = Path(path)
        if not ics_path.exists():
            return {"success": False, "error": f"File not found: {path}"}

        try:
            with open(ics_path, "r", encoding="utf-8") as f:
                cal = iCalCalendar.from_ical(f.read())
        except Exception as e:
            return {"success": False, "error": f"Failed to parse .ics file: {e}"}

        events = self._load_events()
        imported = 0

        for component in cal.walk():
            if component.name == "VEVENT":
                dtstart = component.get("dtstart")
                dtend = component.get("dtend")
                summary = str(component.get("summary", "Untitled"))
                desc = str(component.get("description", ""))
                loc = str(component.get("location", ""))

                start_dt = dtstart.dt if dtstart else None
                end_dt = dtend.dt if dtend else None

                if start_dt:
                    if hasattr(start_dt, 'isoformat'):
                        start_iso = start_dt.isoformat()
                    else:
                        start_iso = datetime.combine(start_dt, datetime.min.time()).isoformat()
                else:
                    continue

                end_iso = None
                if end_dt:
                    if hasattr(end_dt, 'isoformat'):
                        end_iso = end_dt.isoformat()
                    else:
                        end_iso = datetime.combine(end_dt, datetime.min.time()).isoformat()

                event = CalendarEvent(
                    id=self._generate_id(),
                    title=summary,
                    start=start_iso,
                    end=end_iso,
                    description=desc,
                    location=loc,
                )
                events.append(event.to_dict())
                imported += 1

        self._save_events(events)
        return {
            "success": True,
            "imported": imported,
            "response": f"Imported {imported} event(s) from {ics_path.name}"
        }

    def export_ics(self, event_id: str) -> dict:
        """Export a single event as .ics file."""
        try:
            from icalendar import Calendar as iCalCalendar, Event as iCalEvent
        except ImportError:
            return {"success": False, "error": "icalendar library not installed. Run: pip install icalendar"}

        events = self._load_events()
        event = next((e for e in events if e.get("id") == event_id), None)
        if not event:
            return {"success": False, "error": f"Event not found: {event_id}"}

        cal = iCalCalendar()
        cal.add("prodid", "-//AURA Calendar//EN")
        cal.add("version", "2.0")

        ical_event = iCalEvent()
        ical_event.add("summary", event["title"])
        ical_event.add("dtstart", datetime.fromisoformat(event["start"]))
        if event.get("end"):
            ical_event.add("dtend", datetime.fromisoformat(event["end"]))
        if event.get("description"):
            ical_event.add("description", event["description"])
        if event.get("location"):
            ical_event.add("location", event["location"])
        ical_event.add("uid", event["id"] + "@aura")
        cal.add_component(ical_event)

        export_dir = self.calendar_file.parent / "exports"
        export_dir.mkdir(parents=True, exist_ok=True)
        export_path = export_dir / f"event_{event_id}.ics"
        with open(export_path, "wb") as f:
            f.write(cal.to_ical())

        return {
            "success": True,
            "path": str(export_path),
            "event_id": event_id,
            "response": f"Exported event '{event['title']}' to {export_path.name}"
        }

    def _extract_event_info(self, action: str) -> dict:
        """Extract event info from natural language action string."""
        result = {}

        # Title in quotes
        quote_match = re.search(r'["\']([^"\']+)["\']', action)
        if quote_match:
            result["title"] = quote_match.group(1)

        # Date/time patterns
        on_match = re.search(r'on\s+(.+?)(?:\s+at\s+|\s*$)', action, re.IGNORECASE)
        at_match = re.search(r'at\s+(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)', action, re.IGNORECASE)

        if on_match:
            date_part = on_match.group(1).strip()
            if at_match:
                result["start"] = f"{date_part} {at_match.group(1)}"
            else:
                result["start"] = date_part
        elif at_match:
            result["start"] = f"today at {at_match.group(1)}"

        # Location after "at" (but not time)
        loc_match = re.search(r'(?:at|in|@)\s+([A-Z][^,.\d]+)', action)
        if loc_match and not at_match:
            result["location"] = loc_match.group(1).strip()

        # Title from "add <title>" if not in quotes
        if "title" not in result:
            add_match = re.search(r'add\s+(.+?)(?:\s+on\s+|\s+at\s+|\s+from\s+|\s*$)', action, re.IGNORECASE)
            if add_match:
                result["title"] = add_match.group(1).strip()

        return result

    def execute(self, action: str, **kwargs) -> dict:
        """Execute a calendar action."""
        action_lower = action.lower().strip()

        # Today
        if action_lower in ("today", "today's agenda", "agenda"):
            return self.today()

        # Upcoming
        if action_lower.startswith("upcoming") or action_lower.startswith("next"):
            days_match = re.search(r'(\d+)', action)
            days = int(days_match.group(1)) if days_match else 7
            return self.upcoming(days=days)

        # List
        if action_lower.startswith("list"):
            date_part = action[4:].strip() if len(action) > 4 else None
            if date_part:
                return self.list_events(date=date_part)
            return self.list_events()

        # Remove
        if action_lower.startswith("remove") or action_lower.startswith("delete") or action_lower.startswith("cancel"):
            event_id = kwargs.get("event_id")
            if not event_id:
                id_match = re.search(r'\b([a-f0-9]{8})\b', action)
                event_id = id_match.group(1) if id_match else None
            if event_id:
                return self.remove_event(event_id)
            return {"success": False, "error": "No event ID specified"}

        # Search
        if action_lower.startswith("search") or action_lower.startswith("find"):
            query = kwargs.get("query") or action.split(None, 1)[-1] if len(action.split()) > 1 else ""
            return self.search_events(query)

        # Import .ics
        if action_lower.startswith("import"):
            path = kwargs.get("path") or action.split(None, 1)[-1] if len(action.split()) > 1 else ""
            return self.import_ics(path.strip())

        # Export .ics
        if action_lower.startswith("export"):
            event_id = kwargs.get("event_id")
            if not event_id:
                id_match = re.search(r'\b([a-f0-9]{8})\b', action)
                event_id = id_match.group(1) if id_match else None
            if event_id:
                return self.export_ics(event_id)
            return {"success": False, "error": "No event ID specified for export"}

        # Default: try to add event
        title = kwargs.get("title")
        start = kwargs.get("start")
        end = kwargs.get("end")
        description = kwargs.get("description", "")
        location = kwargs.get("location", "")
        recurrence = kwargs.get("recurrence")
        reminders = kwargs.get("reminders")

        if not title or not start:
            extracted = self._extract_event_info(action)
            title = title or extracted.get("title")
            start = start or extracted.get("start")
            location = location or extracted.get("location", "")

        if title and start:
            return self.add_event(
                title=title, start=start, end=end,
                description=description, location=location,
                recurrence=recurrence, reminders=reminders
            )

        return {
            "success": False,
            "error": f"Could not understand calendar action: {action}. "
                     "Try: 'add <title> on <date> at <time>', 'today', 'upcoming', 'list', 'search <query>', 'remove <id>'"
        }


# Singleton (default user)
calendar_tool = CalendarTool()
