"""
Temporal Query Parser for AURA Episodic Memory.

Parses natural language time references into datetime ranges
for temporal memory queries.
"""

import re
from datetime import datetime, timedelta
from typing import Optional, Tuple, List
from dataclasses import dataclass

# Check dateparser availability
try:
    import dateparser
    DATEPARSER_AVAILABLE = True
except ImportError:
    DATEPARSER_AVAILABLE = False


@dataclass
class TemporalRange:
    """A time range for querying."""
    start: datetime
    end: datetime
    description: str = ""

    def __str__(self):
        return f"{self.description}: {self.start.isoformat()} to {self.end.isoformat()}"


class TemporalParser:
    """
    Parse natural language temporal expressions.

    Supports:
    - Relative times: "yesterday", "last week", "2 hours ago"
    - Specific dates: "January 15th", "2024-01-15"
    - Ranges: "between Monday and Friday", "from last week to now"
    - Recurring: "every morning", "on weekends"
    """

    # Time-of-day mappings
    TIME_OF_DAY = {
        "morning": (5, 12),
        "afternoon": (12, 17),
        "evening": (17, 21),
        "night": (21, 5),
        "dawn": (4, 7),
        "noon": (11, 13),
        "midnight": (23, 1),
    }

    # Relative time patterns
    RELATIVE_PATTERNS = [
        (r"(\d+)\s*(minute|min|m)s?\s*ago", "minutes"),
        (r"(\d+)\s*(hour|hr|h)s?\s*ago", "hours"),
        (r"(\d+)\s*(day|d)s?\s*ago", "days"),
        (r"(\d+)\s*(week|wk|w)s?\s*ago", "weeks"),
        (r"(\d+)\s*(month|mo)s?\s*ago", "months"),
        (r"(\d+)\s*(year|yr|y)s?\s*ago", "years"),
    ]

    # Named relative times
    NAMED_RELATIVES = {
        "now": timedelta(seconds=0),
        "today": timedelta(days=0),
        "yesterday": timedelta(days=1),
        "last night": timedelta(days=1),
        "this morning": timedelta(hours=12),
        "this week": timedelta(days=7),
        "last week": timedelta(days=7),
        "this month": timedelta(days=30),
        "last month": timedelta(days=30),
        "this year": timedelta(days=365),
        "last year": timedelta(days=365),
    }

    def __init__(self, base_time: Optional[datetime] = None):
        """
        Initialize parser.

        Args:
            base_time: Reference time for relative expressions (defaults to now)
        """
        self._fixed_base_time = base_time  # None means use current time

    @property
    def base_time(self) -> datetime:
        """Return fixed base time or current time for relative expressions."""
        return self._fixed_base_time or datetime.now()

    def parse(self, text: str) -> Optional[TemporalRange]:
        """
        Parse temporal expression from text.

        Args:
            text: Natural language text containing time reference

        Returns:
            TemporalRange or None if no time found
        """
        text_lower = text.lower().strip()

        # Try range patterns first
        range_result = self._parse_range(text_lower)
        if range_result:
            return range_result

        # Try relative patterns
        relative_result = self._parse_relative(text_lower)
        if relative_result:
            return relative_result

        # Try named relatives
        named_result = self._parse_named_relative(text_lower)
        if named_result:
            return named_result

        # Try time of day
        tod_result = self._parse_time_of_day(text_lower)
        if tod_result:
            return tod_result

        # Try day of week
        dow_result = self._parse_day_of_week(text_lower)
        if dow_result:
            return dow_result

        # Fall back to dateparser if available
        if DATEPARSER_AVAILABLE:
            return self._parse_with_dateparser(text)

        return None

    def _parse_range(self, text: str) -> Optional[TemporalRange]:
        """Parse range expressions like 'between X and Y' or 'from X to Y'."""
        # Pattern: between X and Y
        between_match = re.search(
            r"between\s+(.+?)\s+and\s+(.+?)(?:\s|$)",
            text
        )
        if between_match:
            start_text = between_match.group(1)
            end_text = between_match.group(2)
            start = self._parse_single(start_text)
            end = self._parse_single(end_text)
            if start and end:
                return TemporalRange(
                    start=min(start, end),
                    end=max(start, end),
                    description=f"between {start_text} and {end_text}"
                )

        # Pattern: from X to Y
        from_to_match = re.search(
            r"from\s+(.+?)\s+to\s+(.+?)(?:\s|$)",
            text
        )
        if from_to_match:
            start_text = from_to_match.group(1)
            end_text = from_to_match.group(2)
            start = self._parse_single(start_text)
            end = self._parse_single(end_text)
            if start and end:
                return TemporalRange(
                    start=min(start, end),
                    end=max(start, end),
                    description=f"from {start_text} to {end_text}"
                )

        return None

    def _parse_relative(self, text: str) -> Optional[TemporalRange]:
        """Parse relative time expressions like '2 hours ago'."""
        for pattern, unit in self.RELATIVE_PATTERNS:
            match = re.search(pattern, text)
            if match:
                amount = int(match.group(1))

                if unit == "minutes":
                    delta = timedelta(minutes=amount)
                elif unit == "hours":
                    delta = timedelta(hours=amount)
                elif unit == "days":
                    delta = timedelta(days=amount)
                elif unit == "weeks":
                    delta = timedelta(weeks=amount)
                elif unit == "months":
                    delta = timedelta(days=amount * 30)
                elif unit == "years":
                    delta = timedelta(days=amount * 365)
                else:
                    continue

                end_time = self.base_time
                start_time = end_time - delta

                return TemporalRange(
                    start=start_time,
                    end=end_time,
                    description=f"{amount} {unit} ago"
                )

        return None

    def _parse_named_relative(self, text: str) -> Optional[TemporalRange]:
        """Parse named relative times like 'yesterday', 'last week'."""
        for name, delta in self.NAMED_RELATIVES.items():
            if name in text:
                if name in ("today", "now"):
                    # Today: start of day to now
                    start = self.base_time.replace(hour=0, minute=0, second=0)
                    end = self.base_time
                elif name == "yesterday":
                    # Yesterday: full day
                    day = self.base_time - timedelta(days=1)
                    start = day.replace(hour=0, minute=0, second=0)
                    end = day.replace(hour=23, minute=59, second=59)
                elif name == "last night":
                    # Last night: yesterday 9pm to midnight
                    day = self.base_time - timedelta(days=1)
                    start = day.replace(hour=21, minute=0, second=0)
                    end = day.replace(hour=23, minute=59, second=59)
                elif name == "this morning":
                    # This morning: today 5am to noon
                    start = self.base_time.replace(hour=5, minute=0, second=0)
                    end = self.base_time.replace(hour=12, minute=0, second=0)
                elif "week" in name:
                    # Last/this week
                    if "last" in name:
                        end = self.base_time - timedelta(days=self.base_time.weekday())
                        start = end - timedelta(days=7)
                    else:
                        start = self.base_time - timedelta(days=self.base_time.weekday())
                        end = self.base_time
                    start = start.replace(hour=0, minute=0, second=0)
                    end = end.replace(hour=23, minute=59, second=59)
                elif "month" in name:
                    # Last/this month
                    if "last" in name:
                        first_of_month = self.base_time.replace(day=1, hour=0, minute=0, second=0)
                        end = first_of_month - timedelta(seconds=1)
                        start = (first_of_month - timedelta(days=1)).replace(day=1, hour=0, minute=0, second=0)
                    else:
                        start = self.base_time.replace(day=1, hour=0, minute=0, second=0)
                        end = self.base_time
                elif "year" in name:
                    # Last/this year
                    if "last" in name:
                        start = datetime(self.base_time.year - 1, 1, 1)
                        end = datetime(self.base_time.year - 1, 12, 31, 23, 59, 59)
                    else:
                        start = datetime(self.base_time.year, 1, 1)
                        end = self.base_time
                else:
                    continue

                return TemporalRange(start=start, end=end, description=name)

        return None

    def _parse_time_of_day(self, text: str) -> Optional[TemporalRange]:
        """Parse time-of-day references like 'this morning', 'in the evening'."""
        for tod_name, (start_hour, end_hour) in self.TIME_OF_DAY.items():
            if tod_name in text:
                # Determine which day
                if "yesterday" in text:
                    day = self.base_time - timedelta(days=1)
                elif "tomorrow" in text:
                    day = self.base_time + timedelta(days=1)
                else:
                    day = self.base_time

                # Handle overnight periods
                if start_hour > end_hour:  # e.g., night: 21 to 5
                    start = day.replace(hour=start_hour, minute=0, second=0)
                    end = (day + timedelta(days=1)).replace(hour=end_hour, minute=0, second=0)
                else:
                    start = day.replace(hour=start_hour, minute=0, second=0)
                    end = day.replace(hour=end_hour, minute=0, second=0)

                return TemporalRange(
                    start=start,
                    end=end,
                    description=f"{tod_name}"
                )

        return None

    def _parse_day_of_week(self, text: str) -> Optional[TemporalRange]:
        """Parse day-of-week references like 'on Monday', 'last Friday'."""
        days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]

        for i, day_name in enumerate(days):
            if day_name in text:
                # Find the most recent occurrence
                current_dow = self.base_time.weekday()
                days_ago = (current_dow - i) % 7

                if days_ago == 0 and "last" in text:
                    days_ago = 7

                target_day = self.base_time - timedelta(days=days_ago)
                start = target_day.replace(hour=0, minute=0, second=0)
                end = target_day.replace(hour=23, minute=59, second=59)

                return TemporalRange(
                    start=start,
                    end=end,
                    description=f"on {day_name}"
                )

        return None

    def _parse_single(self, text: str) -> Optional[datetime]:
        """Parse a single datetime from text."""
        result = self._parse_named_relative(text.lower())
        if result:
            return result.start

        result = self._parse_relative(text.lower())
        if result:
            return result.start

        if DATEPARSER_AVAILABLE:
            parsed = dateparser.parse(text, settings={
                'PREFER_DATES_FROM': 'past',
                'RELATIVE_BASE': self.base_time
            })
            return parsed

        return None

    def _parse_with_dateparser(self, text: str) -> Optional[TemporalRange]:
        """Use dateparser library for complex expressions."""
        parsed = dateparser.parse(text, settings={
            'PREFER_DATES_FROM': 'past',
            'RELATIVE_BASE': self.base_time
        })

        if parsed:
            # Create a range around the parsed time
            # Default to a 1-day window
            start = parsed.replace(hour=0, minute=0, second=0)
            end = parsed.replace(hour=23, minute=59, second=59)

            return TemporalRange(
                start=start,
                end=end,
                description=text
            )

        return None

    def extract_temporal_mentions(self, text: str) -> List[TemporalRange]:
        """
        Extract all temporal mentions from text.

        Args:
            text: Text to search for time references

        Returns:
            List of found temporal ranges
        """
        results = []

        # Try all parsing methods
        sentences = text.replace(",", ".").split(".")

        for sentence in sentences:
            result = self.parse(sentence.strip())
            if result:
                results.append(result)

        return results

    def get_time_of_day(self, dt: datetime) -> str:
        """Get time-of-day label for a datetime."""
        hour = dt.hour
        if 5 <= hour < 12:
            return "morning"
        elif 12 <= hour < 17:
            return "afternoon"
        elif 17 <= hour < 21:
            return "evening"
        else:
            return "night"

    def get_recency_description(self, dt: datetime) -> str:
        """Get human-readable recency description."""
        delta = self.base_time - dt

        if delta.total_seconds() < 60:
            return "just now"
        elif delta.total_seconds() < 3600:
            mins = int(delta.total_seconds() / 60)
            return f"{mins} minute{'s' if mins != 1 else ''} ago"
        elif delta.total_seconds() < 86400:
            hours = int(delta.total_seconds() / 3600)
            return f"{hours} hour{'s' if hours != 1 else ''} ago"
        elif delta.days == 1:
            return "yesterday"
        elif delta.days < 7:
            return f"{delta.days} days ago"
        elif delta.days < 14:
            return "last week"
        elif delta.days < 30:
            weeks = delta.days // 7
            return f"{weeks} week{'s' if weeks != 1 else ''} ago"
        elif delta.days < 60:
            return "last month"
        elif delta.days < 365:
            months = delta.days // 30
            return f"{months} month{'s' if months != 1 else ''} ago"
        else:
            years = delta.days // 365
            return f"{years} year{'s' if years != 1 else ''} ago"
