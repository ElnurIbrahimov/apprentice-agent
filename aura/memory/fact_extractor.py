"""
Fact Extractor

Automatically extracts facts from conversations to store in memory.
"""

import logging
import re
from typing import List, Tuple

logger = logging.getLogger(__name__)


class FactExtractor:
    """Extracts storable facts from messages."""

    def __init__(self):
        # Patterns that indicate facts to extract
        self.patterns = [
            # Name
            (r"(?:my name is|i'm|i am|call me)\s+([A-Z][a-z]+)", "Name"),

            # Work
            (r"(?:i work at|i work for|i'm (?:a|an)|my job is)\s+(.+?)(?:\.|,|$)", "Work"),

            # Location
            (r"(?:i live in|i'm from|i'm in|based in)\s+(.+?)(?:\.|,|$)", "Location"),

            # Events (interviews, meetings)
            (r"(?:i have (?:a|an)|got (?:a|an))\s+(interview|meeting|appointment).+?(?:on|tomorrow|today|next)", "Event"),

            # Preferences
            (r"(?:i (?:love|like|enjoy|prefer|hate|dislike))\s+(.+?)(?:\.|,|$)", "Preference"),

            # Goals
            (r"(?:i want to|i'm trying to|my goal is|i need to)\s+(.+?)(?:\.|,|$)", "Goal"),

            # Family/Relationships
            (r"(?:my (?:wife|husband|partner|girlfriend|boyfriend|son|daughter|dog|cat)(?:'s name is| is named| is))\s+(.+?)(?:\.|,|$)", "Relationship"),
        ]

    def extract_facts(self, message: str) -> List[Tuple[str, str]]:
        """
        Extract facts from a message.

        Returns list of (category, fact) tuples.
        """

        facts = []
        message_lower = message.lower()

        for pattern, category in self.patterns:
            matches = re.finditer(pattern, message_lower, re.IGNORECASE)
            for match in matches:
                fact = match.group(0).strip()
                # Clean up the fact
                fact = fact.rstrip('.,!?')
                if len(fact) > 5:  # Avoid tiny matches
                    facts.append((category, fact))

        return facts

    def extract_name(self, message: str) -> Optional[str]:
        """Extract a name if mentioned."""
        patterns = [
            r"(?:my name is|i'm|i am|call me)\s+([A-Z][a-z]+)",
            r"(?:this is|it's|i am)\s+([A-Z][a-z]+)\b",
        ]

        for pattern in patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                name = match.group(1)
                # Capitalize properly
                return name.capitalize()

        return None

    def extract_follow_up_triggers(self, message: str) -> List[Tuple[str, str]]:
        """
        Extract events that need follow-up.

        Returns list of (topic, when) tuples.
        """

        triggers = []
        message_lower = message.lower()

        # Interview patterns
        if 'interview' in message_lower:
            if 'tomorrow' in message_lower:
                triggers.append(("the interview", "tomorrow"))
            elif 'today' in message_lower:
                triggers.append(("the interview", "today"))
            elif re.search(r'on (monday|tuesday|wednesday|thursday|friday|saturday|sunday)', message_lower):
                day = re.search(r'on (monday|tuesday|wednesday|thursday|friday|saturday|sunday)', message_lower).group(1)
                triggers.append(("the interview", f"next_{day}"))
            elif 'next week' in message_lower:
                triggers.append(("the interview", "next_week"))
            else:
                # Generic interview mention - follow up in 2 days
                triggers.append(("the interview", "in_2_days"))

        # Meeting patterns
        if 'meeting' in message_lower:
            if 'tomorrow' in message_lower:
                triggers.append(("that meeting", "tomorrow"))
            elif 'today' in message_lower:
                triggers.append(("that meeting", "today"))

        # Deadline patterns
        deadline_match = re.search(r'deadline.+?(tomorrow|today|next week|monday|tuesday|wednesday|thursday|friday)', message_lower)
        if deadline_match:
            triggers.append(("the deadline", deadline_match.group(1)))

        # Exam patterns
        if 'exam' in message_lower or 'test' in message_lower:
            if 'tomorrow' in message_lower:
                triggers.append(("the exam", "tomorrow"))

        return triggers


# For backwards compatibility
def extract_name(message: str) -> Optional[str]:
    extractor = FactExtractor()
    return extractor.extract_name(message)
