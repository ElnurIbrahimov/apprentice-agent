"""Calendar tool for managing events, appointments, and schedules.

Supports:
- Local JSON storage (default)
- ICS import/export with recurring event expansion
- Google Calendar API (optional, graceful fallback)
- CalDAV sources (optional)
- Calendar intelligence: free slots, conflicts, deadline scoring
"""

import json
import logging
import os
import re
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, date, time
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  Optional imports — never break if missing
# ---------------------------------------------------------------------------
try:
    from dateutil.parser import parse as dateutil_parse
    from dateutil.rrule import rrulestr
    from dateutil import tz as dateutil_tz
    HAS_DATEUTIL = True
except ImportError:
    dateutil_parse = None
    rrulestr = None
    dateutil_tz = None
    HAS_DATEUTIL = False

try:
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request as GRequest
    from googleapiclient.discovery import build as google_build
    HAS_GOOGLE_CAL = True
except ImportError:
    HAS_GOOGLE_CAL = False

try:
    from icalendar import Calendar as iCalCalendar, Event as iCalEvent
    HAS_ICAL = True
except ImportError:
    HAS_ICAL = False

try:
    import pytz
    HAS_PYTZ = True
except ImportError:
    HAS_PYTZ = False

# Google Calendar API scopes
GCAL_SCOPES = ["https://www.googleapis.com/auth/calendar"]
GCAL_CREDS_PATH = Path.home() / ".aura" / "google_calendar_creds.json"
GCAL_TOKEN_PATH = Path.home() / ".aura" / "google_calendar_token.json"


# ---------------------------------------------------------------------------
#  Data classes
# ---------------------------------------------------------------------------
@dataclass
class CalendarEvent:
    """A calendar event."""
    id: str
    title: str
    start: str                          # ISO 8601
    end: Optional[str] = None           # ISO 8601 or None (all-day)
    description: str = ""
    location: str = ""
    recurrence: Optional[str] = None    # daily/weekly/monthly/yearly or RRULE string
    reminders: List[int] = field(default_factory=list)  # minutes before
    created_at: str = ""
    source: str = "local"               # local / google / ics

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CalendarEvent":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
#  Google Calendar backend
# ---------------------------------------------------------------------------
class GoogleCalendarBackend:
    """Optional Google Calendar API integration."""

    def __init__(self):
        self._service = None

    @property
    def available(self) -> bool:
        return HAS_GOOGLE_CAL and GCAL_CREDS_PATH.exists()

    def _get_service(self):
        """Authenticate and return the Google Calendar service."""
        if self._service is not None:
            return self._service

        creds = None
        if GCAL_TOKEN_PATH.exists():
            creds = Credentials.from_authorized_user_file(str(GCAL_TOKEN_PATH), GCAL_SCOPES)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(GRequest())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(str(GCAL_CREDS_PATH), GCAL_SCOPES)
                creds = flow.run_local_server(port=0)
            GCAL_TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(GCAL_TOKEN_PATH, "w") as f:
                f.write(creds.to_json())

        self._service = google_build("calendar", "v3", credentials=creds)
        return self._service

    def list_events(self, days: int = 7) -> List[Dict[str, Any]]:
        """Fetch events from Google Calendar for the next N days."""
        service = self._get_service()
        now = datetime.utcnow().isoformat() + "Z"
        end = (datetime.utcnow() + timedelta(days=days)).isoformat() + "Z"

        results = service.events().list(
            calendarId="primary",
            timeMin=now,
            timeMax=end,
            maxResults=100,
            singleEvents=True,
            orderBy="startTime",
        ).execute()

        events = []
        for item in results.get("items", []):
            start = item["start"].get("dateTime", item["start"].get("date", ""))
            end_val = item["end"].get("dateTime", item["end"].get("date", ""))
            events.append({
                "id": item["id"],
                "title": item.get("summary", "(No title)"),
                "start": start,
                "end": end_val,
                "description": item.get("description", ""),
                "location": item.get("location", ""),
                "source": "google",
            })
        return events

    def create_event(self, title: str, start: str, end: str,
                     description: str = "", location: str = "",
                     timezone: str = None) -> Dict[str, Any]:
        """Create a Google Calendar event."""
        service = self._get_service()
        tz = timezone or self._get_user_timezone()

        body = {
            "summary": title,
            "description": description,
            "location": location,
            "start": {"dateTime": start, "timeZone": tz},
            "end": {"dateTime": end, "timeZone": tz},
        }
        event = service.events().insert(calendarId="primary", body=body).execute()
        return {
            "id": event["id"],
            "title": event.get("summary", ""),
            "start": event["start"].get("dateTime", event["start"].get("date", "")),
            "end": event["end"].get("dateTime", event["end"].get("date", "")),
            "link": event.get("htmlLink", ""),
            "source": "google",
        }

    def delete_event(self, event_id: str) -> bool:
        """Delete a Google Calendar event."""
        service = self._get_service()
        service.events().delete(calendarId="primary", eventId=event_id).execute()
        return True

    def update_event(self, event_id: str, changes: Dict[str, Any]) -> Dict[str, Any]:
        """Update a Google Calendar event."""
        service = self._get_service()
        event = service.events().get(calendarId="primary", eventId=event_id).execute()

        field_map = {
            "title": "summary",
            "description": "description",
            "location": "location",
        }
        for key, val in changes.items():
            gcal_key = field_map.get(key, key)
            if gcal_key in ("summary", "description", "location"):
                event[gcal_key] = val
            elif key == "start":
                tz = changes.get("timezone") or self._get_user_timezone()
                event["start"] = {"dateTime": val, "timeZone": tz}
            elif key == "end":
                tz = changes.get("timezone") or self._get_user_timezone()
                event["end"] = {"dateTime": val, "timeZone": tz}

        updated = service.events().update(
            calendarId="primary", eventId=event_id, body=event
        ).execute()
        return {
            "id": updated["id"],
            "title": updated.get("summary", ""),
            "start": updated["start"].get("dateTime", updated["start"].get("date", "")),
            "end": updated["end"].get("dateTime", updated["end"].get("date", "")),
            "source": "google",
        }

    def _get_user_timezone(self) -> str:
        """Get user's timezone from Google Calendar settings."""
        try:
            service = self._get_service()
            settings = service.settings().get(setting="timezone").execute()
            return settings.get("value", "UTC")
        except Exception as e:
            logger.debug(f"[Calendar] Timezone fetch failed, defaulting to UTC: {e}")
            return "UTC"


# ---------------------------------------------------------------------------
#  ICS / CalDAV backend
# ---------------------------------------------------------------------------
class ICSBackend:
    """Read events from local .ics files or CalDAV URLs."""

    def __init__(self):
        self._sources: List[str] = []  # paths or URLs

    def add_source(self, source: str):
        """Add an ICS file path or CalDAV URL."""
        if source not in self._sources:
            self._sources.append(source)

    def remove_source(self, source: str):
        if source in self._sources:
            self._sources.remove(source)

    @property
    def sources(self) -> List[str]:
        return list(self._sources)

    def fetch_events(self, range_start: datetime, range_end: datetime) -> List[Dict[str, Any]]:
        """Fetch and expand events from all ICS sources within a date range."""
        if not HAS_ICAL:
            return []

        all_events = []
        for source in self._sources:
            try:
                ics_text = self._read_source(source)
                if ics_text:
                    events = self._parse_ics(ics_text, range_start, range_end)
                    all_events.extend(events)
            except Exception as e:
                logger.warning(f"[Calendar] ICS source error ({source}): {e}")
        return all_events

    def _read_source(self, source: str) -> Optional[str]:
        """Read ICS content from file or URL."""
        if source.startswith(("http://", "https://")):
            try:
                import urllib.request
                with urllib.request.urlopen(source, timeout=15) as resp:
                    return resp.read().decode("utf-8", errors="replace")
            except Exception as e:
                logger.warning(f"[Calendar] Failed to fetch ICS URL {source}: {e}")
                return None
        else:
            p = Path(source)
            if p.exists():
                return p.read_text(encoding="utf-8", errors="replace")
            return None

    def _parse_ics(self, ics_text: str, range_start: datetime, range_end: datetime) -> List[Dict[str, Any]]:
        """Parse ICS text and expand recurring events into the given range."""
        cal = iCalCalendar.from_ical(ics_text)
        events = []

        for component in cal.walk():
            if component.name != "VEVENT":
                continue

            dtstart = component.get("dtstart")
            dtend = component.get("dtend")
            summary = str(component.get("summary", "Untitled"))
            desc = str(component.get("description", ""))
            loc = str(component.get("location", ""))
            uid = str(component.get("uid", uuid.uuid4().hex[:8]))
            rrule = component.get("rrule")

            if not dtstart:
                continue

            start_dt = self._to_datetime(dtstart.dt)
            end_dt = self._to_datetime(dtend.dt) if dtend else None
            duration = (end_dt - start_dt) if end_dt else timedelta(hours=1)

            if rrule and HAS_DATEUTIL:
                # Expand recurring events
                try:
                    rule_str = component.get("rrule").to_ical().decode("utf-8")
                    rule = rrulestr(f"RRULE:{rule_str}", dtstart=start_dt)
                    occurrences = list(rule.between(range_start, range_end, inc=True))
                    for occ in occurrences[:100]:
                        events.append({
                            "id": f"{uid[:8]}_{occ.strftime('%Y%m%d')}",
                            "title": summary,
                            "start": occ.isoformat(),
                            "end": (occ + duration).isoformat(),
                            "description": desc,
                            "location": loc,
                            "source": "ics",
                            "_recurring": True,
                        })
                except Exception as e:
                    logger.warning(f"[Calendar] RRULE expansion failed for '{summary}': {e}")
                    # Fall back to single instance
                    if range_start <= start_dt <= range_end:
                        events.append(self._make_event_dict(uid, summary, start_dt, end_dt, desc, loc))
            else:
                if range_start <= start_dt <= range_end:
                    events.append(self._make_event_dict(uid, summary, start_dt, end_dt, desc, loc))

        return events

    def _to_datetime(self, dt_val) -> datetime:
        """Convert date or datetime to timezone-aware datetime."""
        if isinstance(dt_val, datetime):
            if dt_val.tzinfo is None and HAS_DATEUTIL:
                return dt_val.replace(tzinfo=dateutil_tz.tzlocal())
            return dt_val
        elif isinstance(dt_val, date):
            dt = datetime.combine(dt_val, time.min)
            if HAS_DATEUTIL:
                dt = dt.replace(tzinfo=dateutil_tz.tzlocal())
            return dt
        return datetime.now()

    @staticmethod
    def _make_event_dict(uid, summary, start_dt, end_dt, desc, loc) -> Dict[str, Any]:
        return {
            "id": uid[:8] if len(uid) > 8 else uid,
            "title": summary,
            "start": start_dt.isoformat(),
            "end": end_dt.isoformat() if end_dt else None,
            "description": desc,
            "location": loc,
            "source": "ics",
        }


# ---------------------------------------------------------------------------
#  Smart date parsing
# ---------------------------------------------------------------------------
_DAY_NAMES = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
    "mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6,
}


def smart_parse_datetime(text: str) -> Optional[datetime]:
    """Parse natural language date/time strings.

    Supports:
    - ISO 8601 / common formats
    - "next Tuesday at 3pm"
    - "tomorrow at 14:00"
    - "in 2 hours"
    - "today at 3pm"
    - dateutil fallback
    """
    if not text:
        return None

    text = text.strip()
    now = datetime.now()

    # --- Relative time: "in 30 minutes", "in 2 hours", "in 3 days" ---
    rel_match = re.match(
        r"in\s+(\d+)\s*(minutes?|mins?|hours?|hrs?|days?|weeks?)",
        text, re.IGNORECASE,
    )
    if rel_match:
        val = int(rel_match.group(1))
        unit = rel_match.group(2).lower()
        if unit.startswith("min"):
            return now + timedelta(minutes=val)
        elif unit.startswith("h"):
            return now + timedelta(hours=val)
        elif unit.startswith("d"):
            return now + timedelta(days=val)
        elif unit.startswith("w"):
            return now + timedelta(weeks=val)

    # --- "next <dayname> at <time>" ---
    next_day_match = re.match(
        r"next\s+(\w+)(?:\s+at\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?)?",
        text, re.IGNORECASE,
    )
    if next_day_match:
        day_name = next_day_match.group(1).lower()
        if day_name in _DAY_NAMES:
            target_day = _DAY_NAMES[day_name]
            days_ahead = (target_day - now.weekday()) % 7
            if days_ahead == 0:
                days_ahead = 7
            result = now + timedelta(days=days_ahead)
            h, m = _extract_time(next_day_match.group(2), next_day_match.group(3), next_day_match.group(4))
            return result.replace(hour=h, minute=m, second=0, microsecond=0)

    # --- "this <dayname> at <time>" ---
    this_day_match = re.match(
        r"this\s+(\w+)(?:\s+at\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?)?",
        text, re.IGNORECASE,
    )
    if this_day_match:
        day_name = this_day_match.group(1).lower()
        if day_name in _DAY_NAMES:
            target_day = _DAY_NAMES[day_name]
            days_ahead = (target_day - now.weekday()) % 7
            result = now + timedelta(days=days_ahead)
            h, m = _extract_time(this_day_match.group(2), this_day_match.group(3), this_day_match.group(4))
            return result.replace(hour=h, minute=m, second=0, microsecond=0)

    # --- "today at HH:MM" / "tomorrow at HH:MM" ---
    today_match = re.match(
        r"today\s+(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)?",
        text, re.IGNORECASE,
    )
    if today_match:
        h, m = _extract_time(today_match.group(1), today_match.group(2), today_match.group(3))
        return now.replace(hour=h, minute=m, second=0, microsecond=0)

    tomorrow_match = re.match(
        r"tomorrow\s+(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)?",
        text, re.IGNORECASE,
    )
    if tomorrow_match:
        h, m = _extract_time(tomorrow_match.group(1), tomorrow_match.group(2), tomorrow_match.group(3))
        return (now + timedelta(days=1)).replace(hour=h, minute=m, second=0, microsecond=0)

    # --- dateutil fallback ---
    if HAS_DATEUTIL:
        try:
            return dateutil_parse(text)
        except (ValueError, OverflowError) as e:
            logger.debug(f"[Calendar] dateutil parse failed for '{text[:50]}': {e}")

    # --- Manual ISO / common formats ---
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
            return datetime.strptime(text, fmt)
        except ValueError:
            continue

    return None


def _extract_time(hour_str: Optional[str], min_str: Optional[str], ampm: Optional[str]) -> Tuple[int, int]:
    """Extract hour and minute from parsed groups."""
    h = int(hour_str) if hour_str else 9
    m = int(min_str) if min_str else 0
    if ampm:
        ap = ampm.lower()
        if ap == "pm" and h != 12:
            h += 12
        elif ap == "am" and h == 12:
            h = 0
    return h, m


# ---------------------------------------------------------------------------
#  Main CalendarTool
# ---------------------------------------------------------------------------
class CalendarTool:
    """Tool for managing events, appointments, and schedules.

    Backends (in priority order):
    1. Google Calendar API (if credentials exist)
    2. ICS/CalDAV sources (if configured)
    3. Local JSON storage (always available)
    """

    name = "calendar"
    description = "Manage events, appointments, and schedules"

    def __init__(self, user_id: str = "default"):
        if not re.match(r'^[a-zA-Z0-9_-]+$', user_id):
            raise ValueError(f"Invalid user_id: {user_id!r}. Only alphanumeric, underscore, and hyphen allowed.")
        self.user_id = user_id
        data_dir = Path(__file__).parent.parent.parent / "data" / "users" / user_id
        data_dir.mkdir(parents=True, exist_ok=True)
        self.calendar_file = data_dir / "calendar_events.json"
        self._ensure_file()

        # Optional backends
        self._google = GoogleCalendarBackend() if HAS_GOOGLE_CAL else None
        self._ics = ICSBackend()
        self._ics_sources_file = data_dir / "ics_sources.json"
        self._load_ics_sources()

    # ------------------------------------------------------------------
    #  Local JSON persistence (unchanged interface)
    # ------------------------------------------------------------------
    def _ensure_file(self):
        self.calendar_file.parent.mkdir(parents=True, exist_ok=True)
        if not self.calendar_file.exists():
            self._save_events([])

    def _load_events(self) -> List[Dict[str, Any]]:
        try:
            with open(self.calendar_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return []

    def _save_events(self, events: List[Dict[str, Any]]) -> bool:
        try:
            with open(self.calendar_file, "w", encoding="utf-8") as f:
                json.dump(events, f, indent=4)
            return True
        except IOError:
            return False

    def _generate_id(self) -> str:
        return uuid.uuid4().hex[:8]

    # ------------------------------------------------------------------
    #  ICS source management
    # ------------------------------------------------------------------
    def _load_ics_sources(self):
        if self._ics_sources_file.exists():
            try:
                with open(self._ics_sources_file, "r") as f:
                    sources = json.load(f)
                for s in sources:
                    self._ics.add_source(s)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"[Calendar] Failed to load ICS sources: {e}")

    def _save_ics_sources(self):
        try:
            with open(self._ics_sources_file, "w") as f:
                json.dump(self._ics.sources, f, indent=2)
        except IOError as e:
            logger.warning(f"[Calendar] Failed to save ICS sources: {e}")

    def add_ics_source(self, source: str) -> dict:
        """Add a CalDAV URL or local .ics file path as a source."""
        self._ics.add_source(source)
        self._save_ics_sources()
        return {"success": True, "source": source, "response": f"Added ICS source: {source}"}

    def remove_ics_source(self, source: str) -> dict:
        """Remove an ICS source."""
        self._ics.remove_source(source)
        self._save_ics_sources()
        return {"success": True, "response": f"Removed ICS source: {source}"}

    def list_ics_sources(self) -> dict:
        return {"success": True, "sources": self._ics.sources, "count": len(self._ics.sources)}

    # ------------------------------------------------------------------
    #  Date parsing (uses smart_parse_datetime)
    # ------------------------------------------------------------------
    def _parse_datetime(self, dt_str: str) -> Optional[datetime]:
        return smart_parse_datetime(dt_str)

    # ------------------------------------------------------------------
    #  Google Calendar convenience wrappers
    # ------------------------------------------------------------------
    @property
    def google_available(self) -> bool:
        return self._google is not None and self._google.available

    def google_list_events(self, days: int = 7) -> dict:
        """List events from Google Calendar."""
        if not self.google_available:
            return {"success": False, "error": "Google Calendar not configured. Place OAuth2 credentials at ~/.aura/google_calendar_creds.json"}
        try:
            events = self._google.list_events(days=days)
            return {"success": True, "count": len(events), "events": events, "source": "google",
                    "response": f"Found {len(events)} Google Calendar event(s) for next {days} days"}
        except Exception as e:
            return {"success": False, "error": f"Google Calendar error: {e}"}

    def google_create_event(self, title: str, start: str, end: str,
                            description: str = "", location: str = "") -> dict:
        """Create event on Google Calendar."""
        if not self.google_available:
            return {"success": False, "error": "Google Calendar not configured."}
        try:
            start_dt = self._parse_datetime(start)
            if not start_dt:
                return {"success": False, "error": f"Cannot parse start: {start}"}
            end_dt = self._parse_datetime(end)
            if not end_dt:
                return {"success": False, "error": f"Cannot parse end: {end}"}

            result = self._google.create_event(
                title=title,
                start=start_dt.isoformat(),
                end=end_dt.isoformat(),
                description=description,
                location=location,
            )
            return {"success": True, **result, "response": f"Created Google Calendar event: {title}"}
        except Exception as e:
            return {"success": False, "error": f"Google Calendar create error: {e}"}

    def google_delete_event(self, event_id: str) -> dict:
        """Delete event from Google Calendar."""
        if not self.google_available:
            return {"success": False, "error": "Google Calendar not configured."}
        try:
            self._google.delete_event(event_id)
            return {"success": True, "response": f"Deleted Google Calendar event: {event_id}"}
        except Exception as e:
            return {"success": False, "error": f"Google Calendar delete error: {e}"}

    def google_update_event(self, event_id: str, changes: Dict[str, Any]) -> dict:
        """Update event on Google Calendar. changes can include title, start, end, description, location."""
        if not self.google_available:
            return {"success": False, "error": "Google Calendar not configured."}
        try:
            # Parse datetime strings in changes
            for key in ("start", "end"):
                if key in changes and isinstance(changes[key], str):
                    dt = self._parse_datetime(changes[key])
                    if dt:
                        changes[key] = dt.isoformat()

            result = self._google.update_event(event_id, changes)
            return {"success": True, **result, "response": f"Updated Google Calendar event: {event_id}"}
        except Exception as e:
            return {"success": False, "error": f"Google Calendar update error: {e}"}

    # ------------------------------------------------------------------
    #  Core CRUD (local events, unchanged interface)
    # ------------------------------------------------------------------
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
            source="local",
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

    def update_event(self, event_id: str, changes: Dict[str, Any]) -> dict:
        """Update a local event by ID. changes can include title, start, end, description, location, recurrence."""
        if not event_id:
            return {"success": False, "error": "No event ID provided"}

        events = self._load_events()
        target = None
        for ev in events:
            if ev.get("id") == event_id:
                target = ev
                break

        if not target:
            return {"success": False, "error": f"Event not found: {event_id}"}

        allowed_fields = {"title", "start", "end", "description", "location", "recurrence", "reminders"}
        for key, val in changes.items():
            if key not in allowed_fields:
                continue
            if key in ("start", "end") and isinstance(val, str):
                dt = self._parse_datetime(val)
                if dt:
                    target[key] = dt.isoformat()
                else:
                    return {"success": False, "error": f"Cannot parse {key}: {val}"}
            else:
                target[key] = val

        self._save_events(events)
        return {"success": True, "event_id": event_id, "event": target,
                "response": f"Updated event {event_id}"}

    def list_events(self, date: str = None, range_days: int = 1) -> dict:
        """List events for a specific date or range. Merges local + ICS + Google sources."""
        if date:
            target_dt = self._parse_datetime(date)
            if not target_dt:
                return {"success": False, "error": f"Could not parse date: {date}"}
        else:
            target_dt = datetime.now()

        start_of_range = target_dt.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_range = start_of_range + timedelta(days=range_days)

        # 1) Local events
        matching = self._get_local_events_in_range(start_of_range, end_of_range)

        # 2) ICS sources
        if self._ics.sources:
            try:
                ics_events = self._ics.fetch_events(start_of_range, end_of_range)
                matching.extend(ics_events)
            except Exception as e:
                logger.warning(f"[Calendar] ICS fetch error: {e}")

        # 3) Google Calendar (merge if available and not explicitly skipped)
        if self.google_available:
            try:
                gcal_events = self._google.list_events(days=range_days)
                for ev in gcal_events:
                    try:
                        ev_start = datetime.fromisoformat(ev["start"].replace("Z", "+00:00"))
                        # Strip tz for comparison if local events are naive
                        ev_start_naive = ev_start.replace(tzinfo=None) if ev_start.tzinfo else ev_start
                        if start_of_range <= ev_start_naive < end_of_range:
                            matching.append(ev)
                    except (ValueError, KeyError):
                        matching.append(ev)  # include if we can't parse
            except Exception as e:
                logger.warning(f"[Calendar] Google Calendar fetch error: {e}")

        matching.sort(key=lambda e: e.get("start", ""))

        formatted = []
        for ev in matching:
            try:
                start_dt = datetime.fromisoformat(ev["start"].replace("Z", "+00:00"))
                time_str = start_dt.strftime("%H:%M")
            except (ValueError, KeyError) as e:
                logger.debug(f"[Calendar] Time format failed: {e}")
                time_str = "??:??"
            loc = f" @ {ev['location']}" if ev.get("location") else ""
            src = f" [{ev.get('source', 'local')}]" if ev.get("source", "local") != "local" else ""
            formatted.append(f"[{ev['id']}] {time_str} - {ev['title']}{loc}{src}")

        return {
            "success": True,
            "count": len(matching),
            "events": matching,
            "formatted": "\n".join(formatted) if formatted else "No events found",
            "date_range": f"{start_of_range.strftime('%Y-%m-%d')} to {end_of_range.strftime('%Y-%m-%d')}",
            "response": f"Found {len(matching)} event(s)" + ("\n" + "\n".join(formatted) if formatted else "")
        }

    def _get_local_events_in_range(self, start_of_range: datetime, end_of_range: datetime) -> List[Dict[str, Any]]:
        """Get local events in range, including recurring expansions."""
        events = self._load_events()
        matching = []

        for ev in events:
            try:
                ev_start = datetime.fromisoformat(ev["start"])
                if start_of_range <= ev_start < end_of_range:
                    matching.append(ev)
            except (ValueError, KeyError):
                continue

        # Recurring events
        for ev in events:
            if ev.get("recurrence") and ev not in matching:
                try:
                    occurrences = self._get_recurring_occurrences(ev, start_of_range, end_of_range)
                    for occ in occurrences:
                        occ_ev = dict(ev)
                        occ_ev["start"] = occ.isoformat()
                        occ_ev["_recurring_instance"] = True
                        matching.append(occ_ev)
                except (ValueError, KeyError):
                    continue

        return matching

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
            "monthly": None,
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
        return {"success": True, "removed_id": event_id, "response": f"Removed event {event_id}"}

    def today(self) -> dict:
        return self.list_events(date=datetime.now().strftime("%Y-%m-%d"), range_days=1)

    def upcoming(self, days: int = 7) -> dict:
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

    # ------------------------------------------------------------------
    #  Calendar Intelligence
    # ------------------------------------------------------------------
    def find_free_slots(self, date: str = None, duration_min: int = 30,
                        work_start: int = 9, work_end: int = 18) -> dict:
        """Find available time slots on a given date.

        Args:
            date: Date to check (default: today)
            duration_min: Minimum slot duration in minutes
            work_start: Workday start hour (0-23)
            work_end: Workday end hour (0-23)

        Returns:
            List of free time slots.
        """
        target_dt = self._parse_datetime(date) if date else datetime.now()
        if not target_dt:
            return {"success": False, "error": f"Cannot parse date: {date}"}

        day_start = target_dt.replace(hour=work_start, minute=0, second=0, microsecond=0)
        day_end = target_dt.replace(hour=work_end, minute=0, second=0, microsecond=0)

        result = self.list_events(date=target_dt.strftime("%Y-%m-%d"), range_days=1)
        busy_intervals = []

        for ev in result.get("events", []):
            try:
                ev_start = datetime.fromisoformat(ev["start"].replace("Z", "+00:00")).replace(tzinfo=None)
                if ev.get("end"):
                    ev_end = datetime.fromisoformat(ev["end"].replace("Z", "+00:00")).replace(tzinfo=None)
                else:
                    ev_end = ev_start + timedelta(hours=1)
                busy_intervals.append((max(ev_start, day_start), min(ev_end, day_end)))
            except (ValueError, KeyError):
                continue

        # Sort and merge overlapping intervals
        busy_intervals.sort(key=lambda x: x[0])
        merged = []
        for start, end in busy_intervals:
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))

        # Find gaps
        free_slots = []
        current = day_start
        min_duration = timedelta(minutes=duration_min)

        for busy_start, busy_end in merged:
            if busy_start > current:
                gap = busy_start - current
                if gap >= min_duration:
                    free_slots.append({
                        "start": current.strftime("%H:%M"),
                        "end": busy_start.strftime("%H:%M"),
                        "duration_min": int(gap.total_seconds() / 60),
                    })
            current = max(current, busy_end)

        # Final gap after last event
        if current < day_end:
            gap = day_end - current
            if gap >= min_duration:
                free_slots.append({
                    "start": current.strftime("%H:%M"),
                    "end": day_end.strftime("%H:%M"),
                    "duration_min": int(gap.total_seconds() / 60),
                })

        formatted = [f"{s['start']}-{s['end']} ({s['duration_min']}min)" for s in free_slots]

        return {
            "success": True,
            "date": target_dt.strftime("%Y-%m-%d"),
            "slots": free_slots,
            "count": len(free_slots),
            "response": f"Free slots on {target_dt.strftime('%Y-%m-%d')}:\n" + ("\n".join(formatted) if formatted else "No free slots found")
        }

    def conflicts(self, start: str, end: str) -> dict:
        """Check for scheduling conflicts with a proposed time range.

        Returns any events that overlap with the given start-end.
        """
        start_dt = self._parse_datetime(start)
        end_dt = self._parse_datetime(end)
        if not start_dt:
            return {"success": False, "error": f"Cannot parse start: {start}"}
        if not end_dt:
            return {"success": False, "error": f"Cannot parse end: {end}"}

        # Check a 1-day window around the proposed time
        result = self.list_events(date=start_dt.strftime("%Y-%m-%d"), range_days=2)
        conflicts = []

        for ev in result.get("events", []):
            try:
                ev_start = datetime.fromisoformat(ev["start"].replace("Z", "+00:00")).replace(tzinfo=None)
                if ev.get("end"):
                    ev_end = datetime.fromisoformat(ev["end"].replace("Z", "+00:00")).replace(tzinfo=None)
                else:
                    ev_end = ev_start + timedelta(hours=1)

                # Overlap check: events overlap if one starts before the other ends
                if ev_start < end_dt and ev_end > start_dt:
                    conflicts.append(ev)
            except (ValueError, KeyError):
                continue

        has_conflicts = len(conflicts) > 0
        return {
            "success": True,
            "has_conflicts": has_conflicts,
            "conflicts": conflicts,
            "count": len(conflicts),
            "proposed_start": start_dt.isoformat(),
            "proposed_end": end_dt.isoformat(),
            "response": (
                f"CONFLICT: {len(conflicts)} overlapping event(s) found"
                if has_conflicts else "No conflicts — time slot is clear"
            ),
        }

    def upcoming_deadlines(self, days: int = 7) -> dict:
        """Return upcoming events with urgency scoring.

        Urgency score:
        - 1.0 = happening now or overdue
        - 0.8+ = within 24 hours
        - 0.5+ = within 3 days
        - 0.2+ = within a week
        - <0.2 = further out
        """
        now = datetime.now()
        result = self.list_events(date=now.strftime("%Y-%m-%d"), range_days=days)
        deadlines = []

        for ev in result.get("events", []):
            try:
                ev_start = datetime.fromisoformat(ev["start"].replace("Z", "+00:00")).replace(tzinfo=None)
                hours_until = (ev_start - now).total_seconds() / 3600

                if hours_until <= 0:
                    urgency = 1.0
                    urgency_label = "NOW/OVERDUE"
                elif hours_until <= 24:
                    urgency = round(0.8 + 0.2 * (1 - hours_until / 24), 2)
                    urgency_label = "TODAY"
                elif hours_until <= 72:
                    urgency = round(0.5 + 0.3 * (1 - hours_until / 72), 2)
                    urgency_label = "SOON"
                elif hours_until <= 168:
                    urgency = round(0.2 + 0.3 * (1 - hours_until / 168), 2)
                    urgency_label = "THIS WEEK"
                else:
                    urgency = round(max(0.05, 0.2 * (1 - hours_until / (days * 24))), 2)
                    urgency_label = "LATER"

                deadlines.append({
                    **ev,
                    "urgency": urgency,
                    "urgency_label": urgency_label,
                    "hours_until": round(hours_until, 1),
                })
            except (ValueError, KeyError):
                continue

        # Sort by urgency descending
        deadlines.sort(key=lambda d: d["urgency"], reverse=True)

        formatted = []
        for d in deadlines:
            formatted.append(f"[{d['urgency_label']}] {d['title']} — in {d['hours_until']}h (urgency: {d['urgency']})")

        return {
            "success": True,
            "deadlines": deadlines,
            "count": len(deadlines),
            "response": f"Upcoming deadlines ({days} days):\n" + ("\n".join(formatted) if formatted else "No upcoming events")
        }

    # ------------------------------------------------------------------
    #  ICS import/export (kept from original)
    # ------------------------------------------------------------------
    def import_ics(self, path: str) -> dict:
        """Import events from an .ics file."""
        if not HAS_ICAL:
            return {"success": False, "error": "icalendar library not installed. Run: pip install icalendar"}

        ALLOWED_DIRS = [Path.home() / "Downloads", Path.home() / "Documents",
                        Path(__file__).parent.parent.parent / "data"]
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
                    source="ics",
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
        if not HAS_ICAL:
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

    # ------------------------------------------------------------------
    #  Natural language extraction (kept from original)
    # ------------------------------------------------------------------
    def _extract_event_info(self, action: str) -> dict:
        result = {}

        quote_match = re.search(r'["\']([^"\']+)["\']', action)
        if quote_match:
            result["title"] = quote_match.group(1)

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

        loc_match = re.search(r'(?:at|in|@)\s+([A-Z][^,.\d]+)', action)
        if loc_match and not at_match:
            result["location"] = loc_match.group(1).strip()

        if "title" not in result:
            add_match = re.search(r'add\s+(.+?)(?:\s+on\s+|\s+at\s+|\s+from\s+|\s*$)', action, re.IGNORECASE)
            if add_match:
                result["title"] = add_match.group(1).strip()

        return result

    # ------------------------------------------------------------------
    #  execute() — extended with new actions
    # ------------------------------------------------------------------
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

        # Free slots
        if action_lower.startswith("free") or "free slot" in action_lower or "available" in action_lower:
            date_arg = kwargs.get("date")
            duration = kwargs.get("duration_min", 30)
            return self.find_free_slots(date=date_arg, duration_min=duration)

        # Conflicts
        if action_lower.startswith("conflict"):
            start = kwargs.get("start")
            end = kwargs.get("end")
            if start and end:
                return self.conflicts(start, end)
            return {"success": False, "error": "Need start and end for conflict check"}

        # Deadlines
        if action_lower.startswith("deadline") or action_lower.startswith("urgent"):
            days_match = re.search(r'(\d+)', action)
            days = int(days_match.group(1)) if days_match else 7
            return self.upcoming_deadlines(days=days)

        # Google Calendar commands
        if action_lower.startswith("google") or action_lower.startswith("gcal"):
            sub = action_lower.replace("google", "").replace("gcal", "").strip()
            if sub.startswith("list") or sub.startswith("events"):
                days_match = re.search(r'(\d+)', action)
                days = int(days_match.group(1)) if days_match else 7
                return self.google_list_events(days=days)
            elif sub.startswith("create") or sub.startswith("add"):
                return self.google_create_event(
                    title=kwargs.get("title", ""),
                    start=kwargs.get("start", ""),
                    end=kwargs.get("end", ""),
                    description=kwargs.get("description", ""),
                    location=kwargs.get("location", ""),
                )
            elif sub.startswith("delete") or sub.startswith("remove"):
                return self.google_delete_event(kwargs.get("event_id", ""))
            elif sub.startswith("update"):
                return self.google_update_event(kwargs.get("event_id", ""), kwargs.get("changes", {}))
            return self.google_list_events()

        # ICS source management
        if action_lower.startswith("add source") or action_lower.startswith("add ics"):
            source = kwargs.get("source") or action.split(None, 2)[-1] if len(action.split()) > 2 else ""
            return self.add_ics_source(source.strip())
        if action_lower.startswith("remove source") or action_lower.startswith("remove ics"):
            source = kwargs.get("source") or action.split(None, 2)[-1] if len(action.split()) > 2 else ""
            return self.remove_ics_source(source.strip())
        if action_lower in ("sources", "list sources", "ics sources"):
            return self.list_ics_sources()

        # Remove
        if action_lower.startswith("remove") or action_lower.startswith("delete") or action_lower.startswith("cancel"):
            event_id = kwargs.get("event_id")
            if not event_id:
                id_match = re.search(r'\b([a-f0-9]{8})\b', action)
                event_id = id_match.group(1) if id_match else None
            if event_id:
                return self.remove_event(event_id)
            return {"success": False, "error": "No event ID specified"}

        # Update
        if action_lower.startswith("update") or action_lower.startswith("edit") or action_lower.startswith("change"):
            event_id = kwargs.get("event_id")
            changes = kwargs.get("changes", {})
            if not event_id:
                id_match = re.search(r'\b([a-f0-9]{8})\b', action)
                event_id = id_match.group(1) if id_match else None
            if event_id:
                return self.update_event(event_id, changes)
            return {"success": False, "error": "No event ID specified for update"}

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
                     "Try: 'add <title> on <date> at <time>', 'today', 'upcoming', 'list', "
                     "'free slots', 'conflicts', 'deadlines', 'search <query>', 'remove <id>', "
                     "'google list', 'add source <url>'"
        }


# Singleton (default user)
calendar_tool = CalendarTool()
