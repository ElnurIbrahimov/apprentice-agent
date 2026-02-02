"""
Titans Memory Integration for AURA Episodic Memory.

Bridges the episodic memory system with Titans Neural Memory,
enabling surprise-driven episode formation and memory consolidation.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

from .episode import (
    Episode, EpisodeType, EpisodeQuery, TemporalContext,
    EmotionalValence
)
from .memory_store import EpisodicMemoryStore
from .memory_scorer import MemoryScorer, ScoringConfig
from .timeline import TimelineEngine

logger = logging.getLogger(__name__)


@dataclass
class TitansEpisodicConfig:
    """Configuration for Titans-Episodic integration."""
    # Surprise threshold for episode formation
    surprise_threshold: float = 0.5

    # Minimum content length for episode
    min_content_length: int = 50

    # Maximum episodes to store per session
    max_episodes_per_session: int = 100

    # Auto-link episodes within time window (seconds)
    auto_link_window: int = 300  # 5 minutes

    # Episode formation from conversation turns
    turns_per_episode: int = 3

    # Memory retrieval settings
    default_retrieval_limit: int = 5
    retrieval_recency_weight: float = 0.3
    retrieval_importance_weight: float = 0.3
    retrieval_relevance_weight: float = 0.4


class TitansEpisodicBridge:
    """
    Bridge between Titans Memory and Episodic Memory.

    Responsibilities:
    - Form episodes from surprising/important moments
    - Provide episodic context for Titans queries
    - Enable "time travel" through conversation history
    - Support memory consolidation
    """

    def __init__(
        self,
        memory_store: EpisodicMemoryStore,
        config: Optional[TitansEpisodicConfig] = None,
        session_id: Optional[str] = None
    ):
        """
        Initialize bridge.

        Args:
            memory_store: EpisodicMemoryStore instance
            config: Integration configuration
            session_id: Current session identifier
        """
        self.memory_store = memory_store
        self.config = config or TitansEpisodicConfig()
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")

        # Initialize components
        self.scorer = MemoryScorer(ScoringConfig(
            recency_weight=self.config.retrieval_recency_weight,
            importance_weight=self.config.retrieval_importance_weight,
            relevance_weight=self.config.retrieval_relevance_weight
        ))
        self.timeline = TimelineEngine(memory_store)

        # Session state
        self._pending_turns: List[Dict[str, Any]] = []
        self._recent_episode_ids: List[str] = []
        self._session_episode_count = 0

        # Statistics
        self._stats = {
            "episodes_formed": 0,
            "surprise_triggers": 0,
            "context_retrievals": 0,
            "time_travels": 0
        }

        logger.info(f"TitansEpisodicBridge initialized for session {self.session_id}")

    def on_titans_trace(
        self,
        trace_content: str,
        surprise_score: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Process a Titans memory trace.

        Called when Titans processes new information. Forms episodes
        when surprise exceeds threshold.

        Args:
            trace_content: Content of the memory trace
            surprise_score: Surprise score from Titans (0-1)
            metadata: Additional metadata

        Returns:
            Episode ID if formed, None otherwise
        """
        metadata = metadata or {}

        # Check if surprise warrants episode formation
        if surprise_score >= self.config.surprise_threshold:
            self._stats["surprise_triggers"] += 1

            # Form episode from surprising content
            episode = self._form_episode(
                content=trace_content,
                episode_type=EpisodeType.LEARNING if surprise_score > 0.7 else EpisodeType.INSIGHT,
                importance=min(1.0, surprise_score * 1.2),
                metadata={
                    "source": "titans_surprise",
                    "surprise_score": surprise_score,
                    **metadata
                }
            )

            return self.memory_store.store_episode(episode)

        return None

    def on_conversation_turn(
        self,
        user_message: str,
        assistant_response: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Process a conversation turn.

        Accumulates turns and forms episode when threshold reached
        or content is significant.

        Args:
            user_message: User's message
            assistant_response: Agent's response
            metadata: Additional metadata

        Returns:
            Episode ID if formed, None otherwise
        """
        metadata = metadata or {}

        # Add to pending turns
        self._pending_turns.append({
            "user": user_message,
            "assistant": assistant_response,
            "timestamp": datetime.now(),
            "metadata": metadata
        })

        # Check if we should form an episode
        should_form = (
            len(self._pending_turns) >= self.config.turns_per_episode or
            self._is_significant_turn(user_message, assistant_response, metadata)
        )

        if should_form and self._session_episode_count < self.config.max_episodes_per_session:
            return self._form_conversation_episode()

        return None

    def _is_significant_turn(
        self,
        user_message: str,
        assistant_response: str,
        metadata: Dict[str, Any]
    ) -> bool:
        """Check if turn warrants immediate episode formation."""
        # Check for significance markers
        significance_markers = [
            "error" in metadata,
            metadata.get("tools_used"),
            len(assistant_response) > 500,
            any(kw in user_message.lower() for kw in [
                "important", "remember", "don't forget",
                "milestone", "achievement", "completed"
            ])
        ]

        return any(significance_markers)

    def _form_conversation_episode(self) -> str:
        """Form episode from accumulated conversation turns."""
        if not self._pending_turns:
            return None

        # Combine turns into episode content
        content_parts = []
        entities = set()
        tools = set()
        timestamps = []

        for turn in self._pending_turns:
            content_parts.append(f"User: {turn['user'][:200]}")
            content_parts.append(f"Assistant: {turn['assistant'][:300]}")
            timestamps.append(turn["timestamp"])

            if turn["metadata"].get("entities"):
                entities.update(turn["metadata"]["entities"])
            if turn["metadata"].get("tools_used"):
                tools.update(turn["metadata"]["tools_used"])

        content = "\n".join(content_parts)

        # Determine episode type
        if any(t["metadata"].get("error") for t in self._pending_turns):
            ep_type = EpisodeType.ERROR
        elif tools:
            ep_type = EpisodeType.TASK_EXECUTION
        else:
            ep_type = EpisodeType.CONVERSATION

        # Create episode
        episode = self._form_episode(
            content=content,
            episode_type=ep_type,
            importance=0.5,
            entities=list(entities),
            tools=list(tools),
            metadata={
                "source": "conversation",
                "turn_count": len(self._pending_turns)
            }
        )

        # Auto-link to recent episodes
        if self._recent_episode_ids:
            episode.related_episode_ids = self._recent_episode_ids[-3:]

        # Store and update state
        episode_id = self.memory_store.store_episode(episode)
        self._recent_episode_ids.append(episode_id)
        self._pending_turns.clear()
        self._session_episode_count += 1

        return episode_id

    def _form_episode(
        self,
        content: str,
        episode_type: EpisodeType,
        importance: float = 0.5,
        entities: Optional[List[str]] = None,
        tools: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Episode:
        """Create an episode with current context."""
        self._stats["episodes_formed"] += 1

        return Episode(
            content=content,
            episode_type=episode_type,
            temporal_context=TemporalContext(
                timestamp=datetime.now(),
                session_id=self.session_id
            ),
            importance=importance,
            entities_involved=entities or [],
            tools_used=tools or [],
            metadata=metadata or {}
        )

    def get_context_for_query(
        self,
        query: str,
        limit: int = None,
        include_timeline: bool = False
    ) -> str:
        """
        Get episodic context for a query.

        Retrieves relevant episodes and formats as context string
        for use with Titans Memory.

        Args:
            query: Query text
            limit: Maximum episodes to include
            include_timeline: Include timeline summary

        Returns:
            Formatted context string
        """
        self._stats["context_retrievals"] += 1
        limit = limit or self.config.default_retrieval_limit

        # Search for relevant episodes
        search_query = EpisodeQuery(
            query_text=query,
            limit=limit,
            recency_weight=self.config.retrieval_recency_weight,
            importance_weight=self.config.retrieval_importance_weight,
            relevance_weight=self.config.retrieval_relevance_weight
        )

        results = self.memory_store.search(search_query)

        if not results:
            return ""

        # Format context
        lines = ["[EPISODIC MEMORY CONTEXT]"]

        for i, result in enumerate(results, 1):
            ep = result.episode
            recency = self.timeline.parser.get_recency_description(
                ep.temporal_context.timestamp
            )

            lines.append(f"\n--- Memory {i} ({recency}, score: {result.score:.2f}) ---")
            lines.append(f"Type: {ep.episode_type.value}")

            if ep.title:
                lines.append(f"Title: {ep.title}")

            lines.append(f"Content: {ep.content[:500]}")

            if ep.entities_involved:
                lines.append(f"Entities: {', '.join(ep.entities_involved[:5])}")

            if ep.tools_used:
                lines.append(f"Tools: {', '.join(ep.tools_used)}")

        if include_timeline:
            # Add brief timeline context
            from .temporal_parser import TemporalRange
            from datetime import timedelta

            timeline_range = TemporalRange(
                start=datetime.now() - timedelta(days=1),
                end=datetime.now(),
                description="last 24 hours"
            )
            timeline_view = self.timeline.get_timeline(timeline_range, granularity="hour")
            lines.append(f"\n[Recent Activity: {timeline_view.total_episodes} episodes in last 24h]")

        lines.append("\n[/EPISODIC MEMORY CONTEXT]")

        return "\n".join(lines)

    def time_travel(self, time_reference: str) -> Tuple[List[Episode], str]:
        """
        Travel to a point in memory.

        Args:
            time_reference: Natural language time reference

        Returns:
            Tuple of (episodes, narrative)
        """
        self._stats["time_travels"] += 1
        return self.timeline.time_travel(time_reference)

    def on_task_complete(
        self,
        task_description: str,
        result: str,
        success: bool,
        tools_used: Optional[List[str]] = None,
        entities: Optional[List[str]] = None
    ) -> str:
        """
        Record task completion as an episode.

        Args:
            task_description: What was attempted
            result: Outcome description
            success: Whether task succeeded
            tools_used: Tools that were used
            entities: Entities involved

        Returns:
            Episode ID
        """
        content = f"Task: {task_description}\nResult: {result}"

        episode = self._form_episode(
            content=content,
            episode_type=EpisodeType.TASK_EXECUTION if success else EpisodeType.ERROR,
            importance=0.7 if success else 0.8,  # Failures slightly more important
            entities=entities,
            tools=tools_used,
            metadata={
                "source": "task_completion",
                "success": success
            }
        )

        if not success:
            episode.emotional_valence = EmotionalValence.NEGATIVE

        return self.memory_store.store_episode(episode)

    def on_milestone(
        self,
        milestone_description: str,
        details: Optional[str] = None,
        entities: Optional[List[str]] = None
    ) -> str:
        """
        Record a milestone achievement.

        Args:
            milestone_description: What was achieved
            details: Additional details
            entities: Related entities

        Returns:
            Episode ID
        """
        content = milestone_description
        if details:
            content += f"\n\nDetails: {details}"

        episode = self._form_episode(
            content=content,
            episode_type=EpisodeType.MILESTONE,
            importance=0.9,  # Milestones are important
            entities=entities,
            metadata={"source": "milestone"}
        )
        episode.emotional_valence = EmotionalValence.POSITIVE

        return self.memory_store.store_episode(episode)

    def on_user_preference(
        self,
        preference: str,
        context: Optional[str] = None
    ) -> str:
        """
        Record a learned user preference.

        Args:
            preference: The preference learned
            context: How it was learned

        Returns:
            Episode ID
        """
        content = f"User Preference: {preference}"
        if context:
            content += f"\nLearned from: {context}"

        episode = self._form_episode(
            content=content,
            episode_type=EpisodeType.USER_PREFERENCE,
            importance=0.75,
            metadata={"source": "user_preference"}
        )

        return self.memory_store.store_episode(episode)

    def flush_pending(self) -> List[str]:
        """
        Force-form episodes from any pending turns.

        Returns:
            List of formed episode IDs
        """
        episode_ids = []

        while self._pending_turns:
            ep_id = self._form_conversation_episode()
            if ep_id:
                episode_ids.append(ep_id)

        return episode_ids

    def get_statistics(self) -> Dict[str, Any]:
        """Get bridge statistics."""
        store_stats = self.memory_store.get_statistics()

        return {
            **self._stats,
            "session_id": self.session_id,
            "session_episode_count": self._session_episode_count,
            "pending_turns": len(self._pending_turns),
            "recent_episode_ids": self._recent_episode_ids[-5:],
            "store": store_stats,
            "config": {
                "surprise_threshold": self.config.surprise_threshold,
                "turns_per_episode": self.config.turns_per_episode,
                "max_episodes_per_session": self.config.max_episodes_per_session
            }
        }
