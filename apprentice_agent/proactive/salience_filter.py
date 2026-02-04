"""
Salience Filter - Determines which events are worth attention.

Filters events based on:
- Recency: How recent is the event?
- Relevance: How related to current context?
- Importance: How critical is this event type?
- Novelty: Have we seen this before?

Only events passing the salience threshold reach the Gateway Daemon.
"""

import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Set, Any, Optional
import hashlib
import json

from .event_bus import Event, EventPriority

logger = logging.getLogger(__name__)


@dataclass
class SalienceWeights:
    """Configurable weights for salience computation."""
    recency: float = 0.25      # How much recent events matter
    relevance: float = 0.35    # How much context match matters
    importance: float = 0.25   # How much event type priority matters
    novelty: float = 0.15      # How much uniqueness matters

    def __post_init__(self):
        """Validate weights sum to 1.0."""
        total = self.recency + self.relevance + self.importance + self.novelty
        if abs(total - 1.0) > 0.01:
            logger.warning(f"Salience weights sum to {total}, normalizing...")
            self.recency /= total
            self.relevance /= total
            self.importance /= total
            self.novelty /= total


@dataclass
class FilteredEvent:
    """An event with computed salience score."""
    event: Event
    salience_score: float
    salience_breakdown: Dict[str, float]
    filtered_at: float = field(default_factory=time.time)

    @property
    def passed(self) -> bool:
        """Check if event passed the filter."""
        return self.salience_score >= 0.3  # Default threshold


class SalienceFilter:
    """
    Filters events by computed salience score.

    Salience = weighted combination of:
    - Recency: Exponential decay based on event age
    - Relevance: Keyword/context matching
    - Importance: Event type priority mapping
    - Novelty: Whether we've seen similar events recently

    Usage:
        filter = SalienceFilter()
        filter.set_context(["python", "coding", "project"])

        for event in events:
            result = filter.compute_salience(event)
            if result.passed:
                process(result.event)
    """

    # Default importance scores by event type
    DEFAULT_IMPORTANCE = {
        # Critical events
        "urgent_email": 0.95,
        "system_alert": 0.90,
        "security_warning": 0.95,

        # High importance
        "meeting_start": 0.85,
        "meeting_reminder": 0.80,
        "deadline_approaching": 0.85,
        "error_detected": 0.80,

        # Medium importance
        "calendar_upcoming": 0.60,
        "new_email": 0.55,
        "task_reminder": 0.65,
        "file_changed": 0.50,

        # Low importance
        "screen_change": 0.30,
        "app_switch": 0.25,
        "idle_detected": 0.20,
        "background_update": 0.15,
    }

    def __init__(
        self,
        weights: Optional[SalienceWeights] = None,
        threshold: float = 0.3,
        seen_event_ttl: float = 3600.0  # 1 hour
    ):
        """
        Initialize the salience filter.

        Args:
            weights: Custom salience weights
            threshold: Minimum salience to pass filter
            seen_event_ttl: How long to remember seen events
        """
        self.weights = weights or SalienceWeights()
        self.threshold = threshold
        self.seen_event_ttl = seen_event_ttl

        # Context for relevance matching
        self.context_keywords: Set[str] = set()
        self.current_activity: Optional[str] = None

        # Tracking seen events for novelty
        self._seen_events: Dict[str, float] = {}  # hash -> timestamp
        self._seen_event_limit = 1000

        # Custom importance rules
        self.importance_rules: Dict[str, float] = self.DEFAULT_IMPORTANCE.copy()

        # Statistics
        self._stats = {
            "events_processed": 0,
            "events_passed": 0,
            "events_filtered": 0,
        }

        logger.info(f"[SalienceFilter] Initialized with threshold={threshold}")

    def set_context(self, keywords: List[str], activity: Optional[str] = None) -> None:
        """
        Set current context for relevance matching.

        Args:
            keywords: Keywords relevant to current user focus
            activity: Current activity description
        """
        self.context_keywords = set(kw.lower() for kw in keywords)
        self.current_activity = activity
        logger.debug(f"[SalienceFilter] Context updated: {len(self.context_keywords)} keywords")

    def add_context_keywords(self, keywords: List[str]) -> None:
        """Add keywords to current context."""
        self.context_keywords.update(kw.lower() for kw in keywords)

    def clear_context(self) -> None:
        """Clear current context."""
        self.context_keywords.clear()
        self.current_activity = None

    def set_importance(self, event_type: str, importance: float) -> None:
        """
        Set importance score for an event type.

        Args:
            event_type: Event type name
            importance: Importance score (0.0 to 1.0)
        """
        self.importance_rules[event_type] = max(0.0, min(1.0, importance))

    def _compute_recency(self, event: Event) -> float:
        """
        Compute recency score using exponential decay.

        Score decreases as event ages:
        - 0 seconds old: 1.0
        - 1 minute old: ~0.9
        - 5 minutes old: ~0.6
        - 30 minutes old: ~0.1
        """
        age_seconds = event.age_seconds()
        half_life = 300.0  # 5 minutes
        decay = math.exp(-0.693 * age_seconds / half_life)
        return max(0.0, min(1.0, decay))

    def _compute_relevance(self, event: Event) -> float:
        """
        Compute relevance based on context keyword matching.

        Looks for keyword matches in event payload.
        """
        if not self.context_keywords:
            return 0.5  # Neutral if no context set

        # Extract text from event payload
        event_text = json.dumps(event.payload).lower()

        # Also include source and type
        event_text += f" {event.source} {event.event_type}".lower()

        # Count keyword matches
        matches = sum(1 for kw in self.context_keywords if kw in event_text)

        if matches == 0:
            return 0.1  # No matches, low relevance

        # Normalize by number of keywords
        relevance = min(1.0, matches / max(1, len(self.context_keywords) * 0.5))
        return relevance

    def _compute_importance(self, event: Event) -> float:
        """
        Compute importance based on event type rules.

        Falls back to priority-based scoring if no rule exists.
        """
        # Check custom rules
        if event.event_type in self.importance_rules:
            return self.importance_rules[event.event_type]

        # Check source-prefixed rules (e.g., "calendar.meeting_reminder")
        prefixed = f"{event.source}.{event.event_type}"
        if prefixed in self.importance_rules:
            return self.importance_rules[prefixed]

        # Fall back to priority-based importance
        priority_importance = {
            EventPriority.CRITICAL: 0.95,
            EventPriority.HIGH: 0.75,
            EventPriority.MEDIUM: 0.50,
            EventPriority.LOW: 0.30,
            EventPriority.BACKGROUND: 0.15,
        }
        return priority_importance.get(event.priority, 0.5)

    def _compute_novelty(self, event: Event) -> float:
        """
        Compute novelty based on whether we've seen similar events.

        Events with same source + type + key payload fields are considered similar.
        """
        # Create hash of event "signature"
        signature = {
            "source": event.source,
            "type": event.event_type,
            # Include key payload fields that define uniqueness
            "payload_keys": sorted(event.payload.keys())[:5],
        }

        # Add specific payload values for certain event types
        if "title" in event.payload:
            signature["title"] = event.payload["title"]
        if "app_name" in event.payload:
            signature["app_name"] = event.payload["app_name"]

        event_hash = hashlib.md5(
            json.dumps(signature, sort_keys=True).encode()
        ).hexdigest()[:16]

        # Check if seen recently
        now = time.time()
        if event_hash in self._seen_events:
            last_seen = self._seen_events[event_hash]
            age = now - last_seen

            if age < 60:  # Seen in last minute
                novelty = 0.1
            elif age < 300:  # Seen in last 5 minutes
                novelty = 0.3
            elif age < self.seen_event_ttl:  # Seen within TTL
                novelty = 0.5
            else:
                novelty = 1.0
        else:
            novelty = 1.0  # Never seen

        # Update seen events
        self._seen_events[event_hash] = now

        # Cleanup old entries
        if len(self._seen_events) > self._seen_event_limit:
            self._cleanup_seen_events()

        return novelty

    def _cleanup_seen_events(self) -> None:
        """Remove expired entries from seen events."""
        now = time.time()
        expired = [
            h for h, t in self._seen_events.items()
            if now - t > self.seen_event_ttl
        ]
        for h in expired:
            del self._seen_events[h]

    def compute_salience(self, event: Event) -> FilteredEvent:
        """
        Compute salience score for an event.

        Args:
            event: Event to evaluate

        Returns:
            FilteredEvent with score and breakdown
        """
        # Compute individual components
        recency = self._compute_recency(event)
        relevance = self._compute_relevance(event)
        importance = self._compute_importance(event)
        novelty = self._compute_novelty(event)

        # Weighted combination
        salience = (
            self.weights.recency * recency +
            self.weights.relevance * relevance +
            self.weights.importance * importance +
            self.weights.novelty * novelty
        )

        # Build breakdown
        breakdown = {
            "recency": round(recency, 3),
            "relevance": round(relevance, 3),
            "importance": round(importance, 3),
            "novelty": round(novelty, 3),
        }

        # Update stats
        self._stats["events_processed"] += 1
        if salience >= self.threshold:
            self._stats["events_passed"] += 1
        else:
            self._stats["events_filtered"] += 1

        return FilteredEvent(
            event=event,
            salience_score=round(salience, 4),
            salience_breakdown=breakdown
        )

    def filter_events(self, events: List[Event]) -> List[FilteredEvent]:
        """
        Filter a list of events, returning only those passing threshold.

        Args:
            events: Events to filter

        Returns:
            Filtered events sorted by salience (highest first)
        """
        results = [self.compute_salience(e) for e in events]
        passed = [r for r in results if r.salience_score >= self.threshold]
        return sorted(passed, key=lambda x: x.salience_score, reverse=True)

    def get_stats(self) -> Dict[str, Any]:
        """Get filter statistics."""
        return {
            **self._stats,
            "threshold": self.threshold,
            "context_keywords": len(self.context_keywords),
            "seen_events_cached": len(self._seen_events),
            "pass_rate": (
                self._stats["events_passed"] / max(1, self._stats["events_processed"])
            )
        }


if __name__ == "__main__":
    from .event_bus import Event, EventPriority, create_calendar_event, create_screen_event

    print("=" * 60)
    print("SalienceFilter Test")
    print("=" * 60)

    filter = SalienceFilter(threshold=0.3)

    # Set context
    filter.set_context(["python", "coding", "ai", "project"], activity="programming")

    # Create test events
    events = [
        create_calendar_event(
            "meeting_reminder",
            "AI Project Standup",
            datetime.now(),
            priority=EventPriority.HIGH,
            minutes_until=15
        ),
        create_screen_event(
            "app_switch",
            "Slack",
            "general"
        ),
        Event(
            source="email",
            event_type="new_email",
            priority=EventPriority.MEDIUM,
            payload={
                "subject": "Python code review needed",
                "from": "colleague@company.com"
            }
        ),
        Event(
            source="system",
            event_type="idle_detected",
            priority=EventPriority.LOW,
            payload={"idle_seconds": 300}
        ),
    ]

    print("\n--- Computing salience ---")
    for event in events:
        result = filter.compute_salience(event)
        status = "PASS" if result.salience_score >= filter.threshold else "FILTER"
        print(f"\n[{status}] {event.source}.{event.event_type}")
        print(f"  Score: {result.salience_score:.3f}")
        print(f"  Breakdown: {result.salience_breakdown}")

    print("\n--- Filter batch ---")
    passed = filter.filter_events(events)
    print(f"Passed: {len(passed)}/{len(events)} events")

    print("\n--- Stats ---")
    stats = filter.get_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 60)
    print("Test complete!")
