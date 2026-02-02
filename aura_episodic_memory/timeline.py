"""
Timeline Query Engine for AURA Episodic Memory.

Provides temporal navigation and story-like memory retrieval.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from .episode import Episode, EpisodeType, EpisodeQuery
from .temporal_parser import TemporalParser, TemporalRange


@dataclass
class TimelineSegment:
    """A segment of the timeline with grouped episodes."""
    start_time: datetime
    end_time: datetime
    episodes: List[Episode]
    label: str = ""
    summary: Optional[str] = None

    @property
    def duration(self) -> timedelta:
        return self.end_time - self.start_time

    @property
    def episode_count(self) -> int:
        return len(self.episodes)


@dataclass
class TimelineView:
    """A view of episodes organized by time."""
    segments: List[TimelineSegment]
    total_episodes: int
    time_range: TemporalRange
    granularity: str  # hour, day, week, month

    def to_narrative(self) -> str:
        """Generate narrative description of timeline."""
        lines = [f"Timeline from {self.time_range.start.date()} to {self.time_range.end.date()}"]
        lines.append(f"Total: {self.total_episodes} episodes across {len(self.segments)} {self.granularity}s\n")

        for segment in self.segments:
            if segment.episodes:
                lines.append(f"## {segment.label}")
                for ep in segment.episodes[:5]:  # Limit per segment
                    lines.append(f"  - [{ep.episode_type.value}] {ep.title or ep.content[:50]}")
                if len(segment.episodes) > 5:
                    lines.append(f"  ... and {len(segment.episodes) - 5} more")
                lines.append("")

        return "\n".join(lines)


class TimelineEngine:
    """
    Engine for temporal queries and timeline navigation.

    Supports:
    - Timeline views at different granularities
    - Temporal pattern detection
    - Story-like episode chains
    - "Time travel" queries
    """

    def __init__(self, memory_store):
        """
        Initialize timeline engine.

        Args:
            memory_store: EpisodicMemoryStore instance
        """
        self.memory_store = memory_store
        self.parser = TemporalParser()

    def get_timeline(
        self,
        time_range: TemporalRange,
        granularity: str = "day",
        episode_types: Optional[List[EpisodeType]] = None
    ) -> TimelineView:
        """
        Get timeline view for a time range.

        Args:
            time_range: Time range to query
            granularity: 'hour', 'day', 'week', or 'month'
            episode_types: Optional filter by episode types

        Returns:
            TimelineView with segmented episodes
        """
        # Get episodes in range
        episodes = self.memory_store.get_timeline(
            start_time=time_range.start,
            end_time=time_range.end,
            episode_types=episode_types
        )

        # Group by granularity
        segments = self._segment_episodes(episodes, granularity, time_range)

        return TimelineView(
            segments=segments,
            total_episodes=len(episodes),
            time_range=time_range,
            granularity=granularity
        )

    def _segment_episodes(
        self,
        episodes: List[Episode],
        granularity: str,
        time_range: TemporalRange
    ) -> List[TimelineSegment]:
        """Segment episodes by time granularity."""
        segments = []

        # Determine segment boundaries
        if granularity == "hour":
            delta = timedelta(hours=1)
            fmt = "%Y-%m-%d %H:00"
        elif granularity == "day":
            delta = timedelta(days=1)
            fmt = "%A, %B %d"
        elif granularity == "week":
            delta = timedelta(weeks=1)
            fmt = "Week of %B %d"
        elif granularity == "month":
            delta = timedelta(days=30)
            fmt = "%B %Y"
        else:
            delta = timedelta(days=1)
            fmt = "%Y-%m-%d"

        # Create time buckets
        current = time_range.start
        while current < time_range.end:
            segment_end = min(current + delta, time_range.end)

            # Find episodes in this segment
            segment_episodes = [
                ep for ep in episodes
                if current <= ep.temporal_context.timestamp < segment_end
            ]

            segments.append(TimelineSegment(
                start_time=current,
                end_time=segment_end,
                episodes=segment_episodes,
                label=current.strftime(fmt)
            ))

            current = segment_end

        return segments

    def query_by_time(self, natural_query: str) -> List[Episode]:
        """
        Query episodes using natural language time reference.

        Args:
            natural_query: e.g., "what happened yesterday", "last week's tasks"

        Returns:
            List of matching episodes
        """
        # Parse temporal reference
        time_range = self.parser.parse(natural_query)

        if not time_range:
            # Default to last 24 hours
            time_range = TemporalRange(
                start=datetime.now() - timedelta(days=1),
                end=datetime.now(),
                description="last 24 hours"
            )

        return self.memory_store.get_timeline(
            start_time=time_range.start,
            end_time=time_range.end
        )

    def get_day_summary(self, date: datetime) -> Dict[str, Any]:
        """
        Get summary of a specific day.

        Args:
            date: Date to summarize

        Returns:
            Dictionary with day statistics and highlights
        """
        start = date.replace(hour=0, minute=0, second=0)
        end = date.replace(hour=23, minute=59, second=59)

        episodes = self.memory_store.get_timeline(start, end)

        # Group by type
        by_type = defaultdict(list)
        for ep in episodes:
            by_type[ep.episode_type.value].append(ep)

        # Find highlights (high importance)
        highlights = sorted(episodes, key=lambda e: e.importance, reverse=True)[:5]

        # Time distribution
        by_time = defaultdict(list)
        for ep in episodes:
            by_time[ep.temporal_context.time_of_day].append(ep)

        return {
            "date": date.date().isoformat(),
            "total_episodes": len(episodes),
            "by_type": {k: len(v) for k, v in by_type.items()},
            "by_time_of_day": {k: len(v) for k, v in by_time.items()},
            "highlights": [
                {"title": ep.title or ep.content[:50], "importance": ep.importance}
                for ep in highlights
            ],
            "entities_mentioned": list(set(
                entity for ep in episodes for entity in ep.entities_involved
            ))[:10],
            "tools_used": list(set(
                tool for ep in episodes for tool in ep.tools_used
            ))
        }

    def find_episode_chains(
        self,
        episode: Episode,
        direction: str = "both",
        max_chain_length: int = 10
    ) -> List[Episode]:
        """
        Find chain of related episodes (story mode).

        Args:
            episode: Starting episode
            direction: 'before', 'after', or 'both'
            max_chain_length: Maximum episodes to return

        Returns:
            Ordered list of related episodes
        """
        chain = [episode]

        # Get episodes with shared entities or in similar timeframe
        time_window = timedelta(hours=4)

        if direction in ("before", "both"):
            # Look backwards
            before_start = episode.temporal_context.timestamp - timedelta(days=1)
            before_end = episode.temporal_context.timestamp - timedelta(minutes=1)

            before_episodes = self.memory_store.get_timeline(before_start, before_end)

            # Score by entity overlap and temporal proximity
            scored = []
            for ep in before_episodes:
                if ep.id == episode.id:
                    continue

                score = 0.0
                # Entity overlap
                overlap = set(ep.entities_involved) & set(episode.entities_involved)
                score += len(overlap) * 0.3

                # Related episode links
                if episode.id in ep.related_episode_ids:
                    score += 0.5

                # Temporal proximity
                time_diff = (episode.temporal_context.timestamp - ep.temporal_context.timestamp).total_seconds()
                score += max(0, 1.0 - time_diff / (4 * 3600))  # Decay over 4 hours

                if score > 0.1:
                    scored.append((ep, score))

            # Add top related before episodes
            scored.sort(key=lambda x: x[1], reverse=True)
            for ep, _ in scored[:max_chain_length // 2]:
                chain.insert(0, ep)

        if direction in ("after", "both"):
            # Look forwards
            after_start = episode.temporal_context.timestamp + timedelta(minutes=1)
            after_end = episode.temporal_context.timestamp + timedelta(days=1)

            after_episodes = self.memory_store.get_timeline(after_start, after_end)

            scored = []
            for ep in after_episodes:
                if ep.id == episode.id:
                    continue

                score = 0.0
                overlap = set(ep.entities_involved) & set(episode.entities_involved)
                score += len(overlap) * 0.3

                if episode.id in ep.related_episode_ids:
                    score += 0.5

                time_diff = (ep.temporal_context.timestamp - episode.temporal_context.timestamp).total_seconds()
                score += max(0, 1.0 - time_diff / (4 * 3600))

                if score > 0.1:
                    scored.append((ep, score))

            scored.sort(key=lambda x: x[1], reverse=True)
            for ep, _ in scored[:max_chain_length // 2]:
                chain.append(ep)

        return chain

    def detect_patterns(
        self,
        episode_type: Optional[EpisodeType] = None,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Detect temporal patterns in memory.

        Args:
            episode_type: Optional filter by type
            days: Number of days to analyze

        Returns:
            Dictionary describing detected patterns
        """
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)

        episodes = self.memory_store.get_timeline(
            start_time, end_time,
            episode_types=[episode_type] if episode_type else None
        )

        patterns = {
            "time_of_day_distribution": defaultdict(int),
            "day_of_week_distribution": defaultdict(int),
            "hourly_activity": defaultdict(int),
            "recurring_entities": defaultdict(int),
            "type_distribution": defaultdict(int),
            "peak_hours": [],
            "active_days": [],
        }

        for ep in episodes:
            tc = ep.temporal_context

            # Time distributions
            patterns["time_of_day_distribution"][tc.time_of_day] += 1
            patterns["day_of_week_distribution"][tc.day_of_week] += 1
            patterns["hourly_activity"][tc.timestamp.hour] += 1
            patterns["type_distribution"][ep.episode_type.value] += 1

            # Entity frequency
            for entity in ep.entities_involved:
                patterns["recurring_entities"][entity] += 1

        # Find peaks
        if patterns["hourly_activity"]:
            peak_hour = max(patterns["hourly_activity"].items(), key=lambda x: x[1])
            patterns["peak_hours"] = [peak_hour[0]]

        if patterns["day_of_week_distribution"]:
            sorted_days = sorted(
                patterns["day_of_week_distribution"].items(),
                key=lambda x: x[1],
                reverse=True
            )
            patterns["active_days"] = [d[0] for d in sorted_days[:3]]

        # Convert defaultdicts to regular dicts
        for key in patterns:
            if isinstance(patterns[key], defaultdict):
                patterns[key] = dict(patterns[key])

        return patterns

    def time_travel(
        self,
        reference: str,
        context_window: int = 5
    ) -> Tuple[List[Episode], str]:
        """
        "Time travel" to a point in memory.

        Args:
            reference: Natural language time reference
            context_window: Number of episodes around the target

        Returns:
            Tuple of (episodes, narrative_summary)
        """
        # Parse time reference
        time_range = self.parser.parse(reference)

        if not time_range:
            return [], "Could not parse time reference."

        # Get episodes around that time
        extended_range = TemporalRange(
            start=time_range.start - timedelta(hours=2),
            end=time_range.end + timedelta(hours=2),
            description=time_range.description
        )

        episodes = self.memory_store.get_timeline(extended_range.start, extended_range.end)

        if not episodes:
            return [], f"No memories found for {time_range.description}."

        # Build narrative
        narrative_parts = [
            f"Traveling back to {time_range.description}...",
            f"Found {len(episodes)} memories from that time.\n"
        ]

        # Group by proximity
        current_group = []
        groups = []

        for ep in episodes:
            if not current_group:
                current_group.append(ep)
            else:
                last_time = current_group[-1].temporal_context.timestamp
                if (ep.temporal_context.timestamp - last_time).total_seconds() < 3600:
                    current_group.append(ep)
                else:
                    groups.append(current_group)
                    current_group = [ep]

        if current_group:
            groups.append(current_group)

        # Narrate each group
        for group in groups:
            time_label = self.parser.get_recency_description(group[0].temporal_context.timestamp)
            narrative_parts.append(f"**{time_label}:**")

            for ep in group[:3]:
                ep_desc = ep.title or ep.content[:80]
                narrative_parts.append(f"  - {ep_desc}")

            if len(group) > 3:
                narrative_parts.append(f"  ... and {len(group) - 3} more memories")
            narrative_parts.append("")

        return episodes, "\n".join(narrative_parts)
