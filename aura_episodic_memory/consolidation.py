"""
Memory Consolidation for AURA Episodic Memory.

Implements memory maintenance operations:
- Importance decay over time
- Episode merging for similar memories
- Garbage collection of low-value memories
- Summary generation for long-term storage
"""

import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from .episode import Episode, EpisodeType, EpisodeQuery, TemporalContext
from .memory_store import EpisodicMemoryStore
from .temporal_parser import TemporalRange

logger = logging.getLogger(__name__)


@dataclass
class ConsolidationConfig:
    """Configuration for memory consolidation."""
    # Importance decay
    decay_rate: float = 0.05  # Daily decay rate
    min_importance: float = 0.1  # Minimum importance threshold
    protected_types: List[str] = None  # Types exempt from decay

    # Merging
    similarity_threshold: float = 0.85  # Threshold for merging similar episodes
    merge_time_window_hours: int = 24  # Only merge episodes within this window

    # Garbage collection
    gc_importance_threshold: float = 0.15  # Episodes below this may be deleted
    gc_age_days: int = 90  # Episodes older than this with low importance
    gc_preserve_count: int = 1000  # Always preserve at least this many episodes

    # Summary generation
    summary_chunk_size: int = 10  # Episodes to summarize together
    summary_age_days: int = 30  # Summarize episodes older than this

    def __post_init__(self):
        if self.protected_types is None:
            self.protected_types = ["milestone", "user_preference", "error"]


@dataclass
class ConsolidationResult:
    """Result of a consolidation operation."""
    operation: str
    episodes_affected: int
    details: Dict[str, Any]
    duration_seconds: float


class MemoryConsolidator:
    """
    Handles memory maintenance and consolidation.

    Operations:
    - decay_importance(): Apply time-based importance decay
    - merge_similar(): Merge semantically similar episodes
    - garbage_collect(): Remove low-value memories
    - generate_summaries(): Create summary episodes
    """

    def __init__(
        self,
        memory_store: EpisodicMemoryStore,
        config: Optional[ConsolidationConfig] = None,
        llm_func: Optional[Callable[[str], str]] = None
    ):
        """
        Initialize consolidator.

        Args:
            memory_store: EpisodicMemoryStore instance
            config: Consolidation configuration
            llm_func: Optional LLM function for summary generation
        """
        self.memory_store = memory_store
        self.config = config or ConsolidationConfig()
        self.llm_func = llm_func

        self._consolidation_history: List[ConsolidationResult] = []
        self._MAX_HISTORY = 100  # Prevent unbounded growth

    def run_full_consolidation(self) -> List[ConsolidationResult]:
        """
        Run all consolidation operations.

        Returns:
            List of consolidation results
        """
        results = []

        # 1. Decay importance
        results.append(self.decay_importance())

        # 2. Merge similar episodes
        results.append(self.merge_similar())

        # 3. Generate summaries (if LLM available)
        if self.llm_func:
            results.append(self.generate_summaries())

        # 4. Garbage collection
        results.append(self.garbage_collect())

        self._consolidation_history.extend(results)
        # Keep only the most recent entries to prevent unbounded growth
        if len(self._consolidation_history) > self._MAX_HISTORY:
            self._consolidation_history = self._consolidation_history[-self._MAX_HISTORY:]

        return results

    def decay_importance(self, days_elapsed: float = 1.0) -> ConsolidationResult:
        """
        Apply importance decay to all episodes.

        Importance decays exponentially: new = old * (1 - rate)^days

        Args:
            days_elapsed: Number of days since last decay

        Returns:
            ConsolidationResult
        """
        start_time = datetime.now()

        # Get all episodes
        query = EpisodeQuery(limit=10000)  # Large limit
        results = self.memory_store.search(query)

        affected = 0
        decay_factor = math.pow(1 - self.config.decay_rate, days_elapsed)

        episodes_to_update: List[Episode] = []
        for result in results:
            episode = result.episode

            # Skip protected types
            if episode.episode_type.value in self.config.protected_types:
                continue

            # Calculate new importance
            old_importance = episode.importance
            new_importance = max(
                self.config.min_importance,
                old_importance * decay_factor
            )

            # Collect episodes that changed significantly
            if abs(new_importance - old_importance) > 0.01:
                episode.importance = new_importance
                episodes_to_update.append(episode)

        # Batch-write all changed episodes instead of one-by-one
        if episodes_to_update:
            self.memory_store.store_episodes_batch(episodes_to_update)
            affected = len(episodes_to_update)

        duration = (datetime.now() - start_time).total_seconds()

        return ConsolidationResult(
            operation="decay_importance",
            episodes_affected=affected,
            details={
                "decay_rate": self.config.decay_rate,
                "days_elapsed": days_elapsed,
                "decay_factor": decay_factor
            },
            duration_seconds=duration
        )

    def merge_similar(self) -> ConsolidationResult:
        """
        Merge semantically similar episodes within time windows.

        Returns:
            ConsolidationResult
        """
        start_time = datetime.now()

        # Get recent episodes
        cutoff = datetime.now() - timedelta(days=7)  # Only recent episodes
        query = EpisodeQuery(
            start_time=cutoff,
            limit=500
        )
        results = self.memory_store.search(query)

        episodes = [r.episode for r in results]
        merged_count = 0
        deleted_ids: Set[str] = set()

        # Group by type and time window
        for i, ep1 in enumerate(episodes):
            if ep1.id in deleted_ids:
                continue

            candidates = []
            for ep2 in episodes[i + 1:]:
                if ep2.id in deleted_ids:
                    continue

                # Check same type
                if ep1.episode_type != ep2.episode_type:
                    continue

                # Check time proximity
                time_diff = abs(
                    (ep1.temporal_context.timestamp - ep2.temporal_context.timestamp).total_seconds()
                )
                if time_diff > self.config.merge_time_window_hours * 3600:
                    continue

                # Check semantic similarity using embeddings
                if ep1.embedding and ep2.embedding:
                    similarity = self._cosine_similarity(ep1.embedding, ep2.embedding)
                    if similarity >= self.config.similarity_threshold:
                        candidates.append((ep2, similarity))

            # Merge candidates into ep1
            if candidates:
                merged = self._merge_episodes(ep1, [c[0] for c in candidates])
                self.memory_store.store_episode(merged)

                # Mark merged episodes for deletion
                for candidate, _ in candidates:
                    deleted_ids.add(candidate.id)
                    self.memory_store.delete_episode(candidate.id)

                merged_count += len(candidates)

        duration = (datetime.now() - start_time).total_seconds()

        return ConsolidationResult(
            operation="merge_similar",
            episodes_affected=merged_count,
            details={
                "similarity_threshold": self.config.similarity_threshold,
                "time_window_hours": self.config.merge_time_window_hours,
                "episodes_deleted": len(deleted_ids)
            },
            duration_seconds=duration
        )

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between vectors."""
        dot = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot / (norm1 * norm2)

    def _merge_episodes(self, primary: Episode, others: List[Episode]) -> Episode:
        """Merge multiple episodes into one."""
        # Combine content
        all_content = [primary.content]
        all_entities = set(primary.entities_involved)
        all_tools = set(primary.tools_used)
        max_importance = primary.importance

        for other in others:
            all_content.append(other.content)
            all_entities.update(other.entities_involved)
            all_tools.update(other.tools_used)
            max_importance = max(max_importance, other.importance)
            primary.related_episode_ids.append(other.id)

        # Update primary episode
        primary.content = "\n---\n".join(all_content)
        primary.entities_involved = list(all_entities)
        primary.tools_used = list(all_tools)
        primary.importance = min(1.0, max_importance * 1.1)  # Slight boost for merged
        primary.metadata["merged_count"] = len(others) + 1

        return primary

    def garbage_collect(self) -> ConsolidationResult:
        """
        Remove low-value memories.

        Deletes episodes that:
        - Have importance below threshold
        - Are older than age threshold
        - Are not protected types

        Returns:
            ConsolidationResult
        """
        start_time = datetime.now()

        # Get all episodes
        query = EpisodeQuery(limit=10000)
        results = self.memory_store.search(query)

        # Sort by importance to preserve best ones
        results.sort(key=lambda r: r.episode.importance, reverse=True)

        deleted_count = 0
        age_threshold = datetime.now() - timedelta(days=self.config.gc_age_days)

        # Keep at least preserve_count episodes
        candidates_for_deletion = results[self.config.gc_preserve_count:]

        for result in candidates_for_deletion:
            episode = result.episode

            # Skip protected types
            if episode.episode_type.value in self.config.protected_types:
                continue

            # Check deletion criteria
            is_old = episode.temporal_context.timestamp < age_threshold
            is_low_importance = episode.importance < self.config.gc_importance_threshold

            if is_old and is_low_importance:
                self.memory_store.delete_episode(episode.id)
                deleted_count += 1

        duration = (datetime.now() - start_time).total_seconds()

        return ConsolidationResult(
            operation="garbage_collect",
            episodes_affected=deleted_count,
            details={
                "importance_threshold": self.config.gc_importance_threshold,
                "age_threshold_days": self.config.gc_age_days,
                "preserve_count": self.config.gc_preserve_count
            },
            duration_seconds=duration
        )

    def generate_summaries(self) -> ConsolidationResult:
        """
        Generate summary episodes for old memories.

        Uses LLM to create condensed summaries of related episodes.

        Returns:
            ConsolidationResult
        """
        if not self.llm_func:
            return ConsolidationResult(
                operation="generate_summaries",
                episodes_affected=0,
                details={"error": "No LLM function provided"},
                duration_seconds=0
            )

        start_time = datetime.now()

        # Get old episodes that haven't been summarized
        age_threshold = datetime.now() - timedelta(days=self.config.summary_age_days)
        query = EpisodeQuery(
            end_time=age_threshold,
            limit=100
        )
        results = self.memory_store.search(query)

        # Filter out already-summarized episodes
        episodes = [
            r.episode for r in results
            if not r.episode.metadata.get("is_summary")
            and not r.episode.metadata.get("summarized")
        ]

        summaries_created = 0

        # Group by type and create summaries
        by_type = defaultdict(list)
        for ep in episodes:
            by_type[ep.episode_type].append(ep)

        for ep_type, type_episodes in by_type.items():
            # Chunk episodes
            for i in range(0, len(type_episodes), self.config.summary_chunk_size):
                chunk = type_episodes[i:i + self.config.summary_chunk_size]

                if len(chunk) < 3:  # Don't summarize tiny groups
                    continue

                # Generate summary
                summary = self._generate_chunk_summary(chunk)

                if summary:
                    # Store summary episode
                    self.memory_store.store_episode(summary)

                    # Mark original episodes as summarized
                    for ep in chunk:
                        ep.metadata["summarized"] = True
                        ep.metadata["summary_id"] = summary.id
                        self.memory_store.store_episode(ep)

                    summaries_created += 1

        duration = (datetime.now() - start_time).total_seconds()

        return ConsolidationResult(
            operation="generate_summaries",
            episodes_affected=summaries_created,
            details={
                "chunk_size": self.config.summary_chunk_size,
                "age_threshold_days": self.config.summary_age_days,
                "episodes_processed": len(episodes)
            },
            duration_seconds=duration
        )

    def _generate_chunk_summary(self, episodes: List[Episode]) -> Optional[Episode]:
        """Generate a summary episode from a chunk of episodes."""
        # Build prompt
        episode_texts = []
        for ep in episodes:
            episode_texts.append(
                f"- [{ep.episode_type.value}] {ep.content[:200]}..."
            )

        prompt = f"""Summarize these {len(episodes)} related memories into a single coherent summary:

{chr(10).join(episode_texts)}

Create a concise summary (2-3 sentences) that captures the key information and themes."""

        try:
            summary_text = self.llm_func(prompt)
        except Exception as e:
            logger.error(f"Failed to generate summary: {e}")
            return None

        # Calculate aggregate properties
        avg_importance = sum(ep.importance for ep in episodes) / len(episodes)
        all_entities = set()
        all_tools = set()

        for ep in episodes:
            all_entities.update(ep.entities_involved)
            all_tools.update(ep.tools_used)

        # Determine time range
        timestamps = [ep.temporal_context.timestamp for ep in episodes]
        earliest = min(timestamps)
        latest = max(timestamps)

        return Episode(
            content=summary_text,
            title=f"Summary: {episodes[0].episode_type.value} ({earliest.date()} - {latest.date()})",
            episode_type=episodes[0].episode_type,
            temporal_context=TemporalContext(
                timestamp=earliest,
                duration_seconds=(latest - earliest).total_seconds()
            ),
            importance=min(1.0, avg_importance * 1.2),  # Boost for summaries
            entities_involved=list(all_entities)[:10],
            tools_used=list(all_tools),
            related_episode_ids=[ep.id for ep in episodes],
            metadata={
                "is_summary": True,
                "source_count": len(episodes),
                "time_range": {
                    "start": earliest.isoformat(),
                    "end": latest.isoformat()
                }
            }
        )

    def get_consolidation_history(self) -> List[Dict[str, Any]]:
        """Get history of consolidation operations."""
        return [
            {
                "operation": r.operation,
                "episodes_affected": r.episodes_affected,
                "details": r.details,
                "duration_seconds": r.duration_seconds
            }
            for r in self._consolidation_history
        ]

    def get_health_report(self) -> Dict[str, Any]:
        """
        Generate a health report for the memory system.

        Returns:
            Dictionary with health metrics and recommendations
        """
        stats = self.memory_store.get_statistics()

        # Analyze importance distribution
        query = EpisodeQuery(limit=1000)
        results = self.memory_store.search(query)

        if not results:
            return {
                "status": "empty",
                "total_episodes": 0,
                "recommendations": ["Start storing memories!"]
            }

        importances = [r.episode.importance for r in results]
        avg_importance = sum(importances) / len(importances)
        low_importance_count = sum(1 for i in importances if i < 0.2)

        # Age distribution
        now = datetime.now()
        ages_days = [
            (now - r.episode.temporal_context.timestamp).days
            for r in results
        ]
        old_count = sum(1 for a in ages_days if a > 30)

        # Type distribution
        type_counts = defaultdict(int)
        for r in results:
            type_counts[r.episode.episode_type.value] += 1

        # Generate recommendations
        recommendations = []

        if low_importance_count > len(results) * 0.3:
            recommendations.append(
                f"Consider running garbage_collect: {low_importance_count} low-importance episodes"
            )

        if old_count > len(results) * 0.5 and self.llm_func:
            recommendations.append(
                f"Consider running generate_summaries: {old_count} old episodes"
            )

        if stats["total_episodes"] > 5000:
            recommendations.append(
                "Memory is growing large. Consider more aggressive consolidation."
            )

        return {
            "status": "healthy" if len(recommendations) == 0 else "needs_attention",
            "total_episodes": stats["total_episodes"],
            "average_importance": round(avg_importance, 3),
            "low_importance_count": low_importance_count,
            "old_episodes_count": old_count,
            "type_distribution": dict(type_counts),
            "recommendations": recommendations,
            "last_consolidation": (
                self._consolidation_history[-1].operation
                if self._consolidation_history else None
            )
        }
