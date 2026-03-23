"""
Salience Filter - Determines which events are worth attention.

Filters events based on:
- Recency: How recent is the event?
- Relevance: How related to current context?
- Importance: How critical is this event type?
- Novelty: Have we seen this before?

Only events passing the salience threshold reach the Gateway Daemon.
"""

import atexit
import logging
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Set, Any, Optional
import hashlib
import json

from aura.config import Config
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
        return self.salience_score >= Config.SALIENCE_FILTER_THRESHOLD


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

        # Screen awareness (Phase 3D)
        "error_on_screen": 0.85,
        "content_detected": 0.55,
        "window_change": 0.20,

        # Low importance
        "screen_change": 0.30,
        "app_switch": 0.25,
        "idle_detected": 0.20,
        "background_update": 0.15,
    }

    def __init__(
        self,
        weights: Optional[SalienceWeights] = None,
        threshold: Optional[float] = None,
        seen_event_ttl: Optional[float] = None,
    ):
        """
        Initialize the salience filter.

        Args:
            weights: Custom salience weights
            threshold: Minimum salience to pass filter (default from Config)
            seen_event_ttl: How long to remember seen events (default from Config)
        """
        self.weights = weights or SalienceWeights()
        self.threshold = threshold if threshold is not None else Config.SALIENCE_FILTER_THRESHOLD
        self.seen_event_ttl = seen_event_ttl if seen_event_ttl is not None else Config.SALIENCE_SEEN_EVENT_TTL

        # Context for relevance matching
        self.context_keywords: Set[str] = set()
        self.current_activity: Optional[str] = None

        # Tracking seen events for novelty
        self._seen_events: Dict[str, float] = {}  # hash -> timestamp
        self._seen_event_count_since_cleanup = 0
        self._last_cleanup_time: float = time.monotonic()

        # Custom importance rules
        self.importance_rules: Dict[str, float] = self.DEFAULT_IMPORTANCE.copy()

        # LLM-powered scoring config
        self._llm_enabled = True
        self._llm_pre_filter_threshold = 0.25
        self._llm_skip_threshold = 0.75
        self._llm_weight = 0.6
        self._llm_skip_types: Set[str] = {
            "screen_change", "app_switch", "idle_detected",
            "background_update", "window_change",
        }
        self._context_cache_ttl = 30.0
        self._context_cache: Optional[Dict[str, Any]] = None
        self._context_cache_time: float = 0.0

        # Statistics
        self._stats = {
            "events_processed": 0,
            "events_passed": 0,
            "events_filtered": 0,
            "llm_scored": 0,
            "llm_failures": 0,
            "llm_skipped_low": 0,
            "llm_skipped_high": 0,
            "llm_skipped_type": 0,
        }

        # Shared LLM pool for scoring
        from aura.pools import llm_pool
        self._llm_pool = llm_pool()

        # Load persisted seen events (graceful degradation)
        try:
            from .persistence import get_persistence
            self._seen_events = get_persistence().load_seen_events(self.seen_event_ttl)
            if self._seen_events:
                logger.info(
                    f"[SalienceFilter] Restored {len(self._seen_events)} seen events"
                )
        except Exception:
            pass  # Graceful degradation — start with empty dict

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
        event_text = json.dumps(event.payload, default=str).lower()

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

        # Persist seen event
        try:
            from .persistence import get_persistence
            get_persistence().save_seen_event(event_hash, now)
        except Exception as e:
            logger.debug(f"[SalienceFilter] non-critical: {e}")
        # Periodic cleanup: every N events OR every M seconds, whichever comes first
        self._seen_event_count_since_cleanup += 1
        mono_now = time.monotonic()
        if (
            self._seen_event_count_since_cleanup >= Config.SALIENCE_CLEANUP_INTERVAL
            or mono_now - self._last_cleanup_time >= Config.SALIENCE_CLEANUP_PERIOD
        ):
            self._cleanup_seen_events()
            self._seen_event_count_since_cleanup = 0
            self._last_cleanup_time = mono_now

        return novelty

    def _cleanup_seen_events(self) -> None:
        """Remove expired entries from seen events (TTL-based, runs periodically)."""
        now = time.time()
        expired = [
            h for h, t in self._seen_events.items()
            if now - t > self.seen_event_ttl
        ]
        for h in expired:
            del self._seen_events[h]

    # ================================================================
    # LLM-Powered Scoring
    # ================================================================

    def enable_llm_scoring(self) -> None:
        """Enable LLM-powered salience scoring for mid-range events."""
        self._llm_enabled = True
        logger.info("[SalienceFilter] LLM scoring enabled")

    def disable_llm_scoring(self) -> None:
        """Disable LLM-powered salience scoring (heuristic only)."""
        self._llm_enabled = False
        logger.info("[SalienceFilter] LLM scoring disabled")

    def _gather_context(self) -> Dict[str, Any]:
        """Gather user context for LLM scoring. Cached for 30s."""
        now = time.time()
        if (
            self._context_cache is not None
            and now - self._context_cache_time < self._context_cache_ttl
        ):
            return self._context_cache

        ctx: Dict[str, Any] = {}

        # Recent chat (last 3 user messages)
        try:
            from api.services.agent_service import agent_service
            if agent_service.agent and agent_service.agent.brain:
                history = agent_service.agent.brain.conversation_history
                if history:
                    user_msgs = [
                        m["content"][:150]
                        for m in history[-6:]
                        if m.get("role") == "user"
                    ]
                    ctx["recent_chat"] = user_msgs[-3:]
        except Exception as e:
            logger.debug(f"[SalienceFilter] non-critical: {e}")
        # Focus keywords and current activity
        if self.context_keywords:
            ctx["focus_keywords"] = list(self.context_keywords)[:10]
        if self.current_activity:
            ctx["current_activity"] = self.current_activity

        # User mood from Theory of Mind
        try:
            from .theory_of_mind import get_theory_of_mind
            tom = get_theory_of_mind()
            emo = tom.get_emotional_state()
            ctx["user_mood"] = emo.describe()
        except Exception as e:
            logger.debug(f"[SalienceFilter] non-critical: {e}")
        self._context_cache = ctx
        self._context_cache_time = now
        return ctx

    def _llm_score_event(self, event: Event, heuristic_score: float) -> Optional[float]:
        """Ask the LLM to score an event on urgency/relevance/preference/impact.

        Returns a normalized 0.0-1.0 score, or None on failure.
        """
        ctx = self._gather_context()

        # Build compact event summary
        event_summary = (
            f"type={event.event_type}, source={event.source}, "
            f"priority={event.priority.value}"
        )
        payload_preview = json.dumps(event.payload, default=str)[:200]

        # Build context block
        ctx_lines = []
        if ctx.get("recent_chat"):
            ctx_lines.append(
                "Recent user messages: " + " | ".join(ctx["recent_chat"])
            )
        if ctx.get("focus_keywords"):
            ctx_lines.append(
                "User focus: " + ", ".join(ctx["focus_keywords"])
            )
        if ctx.get("current_activity"):
            ctx_lines.append(f"Activity: {ctx['current_activity']}")
        if ctx.get("user_mood"):
            ctx_lines.append(f"Mood: {ctx['user_mood']}")
        context_block = "\n".join(ctx_lines) if ctx_lines else "No user context available."

        prompt = (
            f"Score this event's salience (importance to the user right now) from 0 to 10.\n\n"
            f"EVENT: {event_summary}\n"
            f"PAYLOAD: {payload_preview}\n"
            f"HEURISTIC SCORE: {heuristic_score:.2f}\n\n"
            f"USER CONTEXT:\n{context_block}\n\n"
            f"Score on four dimensions, then give a FINAL score:\n"
            f"- Urgency (time-sensitive?)\n"
            f"- Relevance (related to user's current focus?)\n"
            f"- User preference (would user want to know?)\n"
            f"- Impact of inaction (what happens if ignored?)\n\n"
            f"Reply with ONLY a single integer 0-10 as the final salience score."
        )

        try:
            from api.services.agent_service import agent_service
            if not (agent_service.agent and agent_service.agent.brain):
                return None

            response = agent_service.agent.brain.think(
                prompt=prompt,
                use_history=False,
            )
            return self._parse_llm_score(response)
        except Exception as e:
            logger.debug(f"[SalienceFilter] LLM scoring failed: {e}")
            return None

    def _parse_llm_score(self, response: Optional[str]) -> Optional[float]:
        """Extract first integer 0-10 from LLM response. Returns 0.0-1.0 or None."""
        if not response:
            return None
        match = re.search(r'\b(10|[0-9])\b', response)
        if not match:
            return None
        value = int(match.group(1))
        return value / 10.0

    # ================================================================
    # Salience Computation
    # ================================================================

    def compute_salience(self, event: Event) -> FilteredEvent:
        """
        Compute salience score for an event.

        Uses a two-tier hybrid approach:
        1. Heuristic score (always computed)
        2. LLM score for mid-range events (0.25-0.75) when enabled
        Final = blended score or heuristic-only fallback.

        Args:
            event: Event to evaluate

        Returns:
            FilteredEvent with score and breakdown
        """
        # Compute individual heuristic components
        recency = self._compute_recency(event)
        relevance = self._compute_relevance(event)
        importance = self._compute_importance(event)
        novelty = self._compute_novelty(event)

        # Weighted combination (heuristic)
        heuristic_score = (
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
            "scoring_method": "heuristic",
        }

        final_score = heuristic_score

        # LLM scoring for mid-range events
        if self._llm_enabled:
            if heuristic_score < self._llm_pre_filter_threshold:
                self._stats["llm_skipped_low"] += 1
            elif heuristic_score > self._llm_skip_threshold:
                self._stats["llm_skipped_high"] += 1
            elif event.event_type in self._llm_skip_types:
                self._stats["llm_skipped_type"] += 1
            else:
                # Mid-range event — ask the LLM (timeout to avoid blocking event thread)
                try:
                    llm_score = self._llm_pool.submit(self._llm_score_event, event, heuristic_score).result(timeout=Config.SALIENCE_LLM_TIMEOUT)
                except FuturesTimeoutError:
                    llm_score = None
                    logger.debug("[SalienceFilter] LLM scoring timed out")
                if llm_score is not None:
                    heuristic_weight = 1.0 - self._llm_weight
                    final_score = (
                        heuristic_weight * heuristic_score
                        + self._llm_weight * llm_score
                    )
                    breakdown["llm_score"] = round(llm_score, 3)
                    breakdown["scoring_method"] = "hybrid"
                    self._stats["llm_scored"] += 1
                    logger.info(
                        f"[SalienceFilter] LLM scoring {event.source}.{event.event_type}: "
                        f"heuristic={heuristic_score:.3f} llm={llm_score:.3f} "
                        f"final={final_score:.3f}"
                    )
                else:
                    # LLM failed — fall back to heuristic
                    self._stats["llm_failures"] += 1
                    logger.debug(
                        f"[SalienceFilter] LLM fallback for "
                        f"{event.source}.{event.event_type}"
                    )

        breakdown["heuristic_score"] = round(heuristic_score, 3)

        # Update stats
        self._stats["events_processed"] += 1
        if final_score >= self.threshold:
            self._stats["events_passed"] += 1
        else:
            self._stats["events_filtered"] += 1

        return FilteredEvent(
            event=event,
            salience_score=round(final_score, 4),
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
            ),
            "llm_enabled": self._llm_enabled,
            "llm_score_rate": (
                self._stats["llm_scored"]
                / max(1, self._stats["events_processed"])
            ),
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
