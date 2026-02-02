"""
Memory Scorer for AURA Episodic Memory.

Implements multi-factor scoring combining recency, importance,
relevance, and other factors for memory retrieval.
"""

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

from .episode import Episode, EpisodeQuery, EpisodeType, EmotionalValence


@dataclass
class ScoringConfig:
    """Configuration for memory scoring."""
    # Weight factors (should sum to ~1.0)
    recency_weight: float = 0.25
    importance_weight: float = 0.25
    relevance_weight: float = 0.30
    access_weight: float = 0.10
    emotional_weight: float = 0.10

    # Recency decay parameters
    recency_half_life_hours: float = 168.0  # 1 week

    # Access frequency parameters
    access_decay_days: float = 30.0  # Time for access bonus to decay

    # Emotional salience boost
    emotional_boost: float = 0.1  # Bonus for emotional memories

    # Type-specific boosts
    type_boosts: Dict[str, float] = field(default_factory=lambda: {
        "milestone": 0.15,
        "error": 0.10,
        "insight": 0.10,
        "user_preference": 0.05,
    })

    # Minimum score threshold
    min_score: float = 0.0


class MemoryScorer:
    """
    Multi-factor memory scorer.

    Combines multiple signals to rank memories:
    - Recency: How recently the episode occurred
    - Importance: Assigned importance value
    - Relevance: Semantic similarity to query
    - Access frequency: How often retrieved
    - Emotional salience: Emotional intensity
    """

    def __init__(self, config: Optional[ScoringConfig] = None):
        """
        Initialize scorer.

        Args:
            config: Scoring configuration
        """
        self.config = config or ScoringConfig()
        self._custom_scorers: List[Callable] = []

    def score(
        self,
        episode: Episode,
        query: EpisodeQuery,
        vector_similarity: float = 0.5
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate composite score for an episode.

        Args:
            episode: Episode to score
            query: Current query context
            vector_similarity: Similarity from vector search (0-1)

        Returns:
            Tuple of (total_score, breakdown_dict)
        """
        breakdown = {}

        # 1. Recency score
        recency = self._calculate_recency(episode)
        breakdown["recency"] = recency

        # 2. Importance score
        importance = episode.importance
        breakdown["importance"] = importance

        # 3. Relevance score (from vector similarity)
        relevance = vector_similarity
        breakdown["relevance"] = relevance

        # 4. Access frequency score
        access = self._calculate_access_score(episode)
        breakdown["access"] = access

        # 5. Emotional salience
        emotional = self._calculate_emotional_score(episode)
        breakdown["emotional"] = emotional

        # 6. Type boost
        type_boost = self._calculate_type_boost(episode)
        breakdown["type_boost"] = type_boost

        # Calculate weighted sum
        base_score = (
            self.config.recency_weight * recency +
            self.config.importance_weight * importance +
            self.config.relevance_weight * relevance +
            self.config.access_weight * access +
            self.config.emotional_weight * emotional
        )

        # Apply type boost
        final_score = min(1.0, base_score + type_boost)

        # Apply custom scorers
        for custom_scorer in self._custom_scorers:
            bonus = custom_scorer(episode, query)
            final_score = min(1.0, final_score + bonus)

        breakdown["final"] = final_score

        return final_score, breakdown

    def _calculate_recency(self, episode: Episode) -> float:
        """Calculate recency score using exponential decay."""
        return episode.get_recency_score(self.config.recency_half_life_hours)

    def _calculate_access_score(self, episode: Episode) -> float:
        """Calculate access frequency score."""
        if episode.access_count == 0:
            return 0.0

        # More accesses = higher score, but with diminishing returns
        access_score = 1.0 - math.exp(-episode.access_count / 5.0)

        # Decay based on last access time
        if episode.last_accessed:
            days_since_access = (datetime.now() - episode.last_accessed).days
            decay = math.exp(-days_since_access / self.config.access_decay_days)
            access_score *= decay

        return access_score

    def _calculate_emotional_score(self, episode: Episode) -> float:
        """Calculate emotional salience score."""
        if episode.emotional_valence == EmotionalValence.NEUTRAL:
            return 0.0

        # Non-neutral emotions get a boost
        base = 0.5 if episode.emotional_valence != EmotionalValence.MIXED else 0.3

        # Negative emotions slightly stronger (negativity bias)
        if episode.emotional_valence == EmotionalValence.NEGATIVE:
            base *= 1.2

        return min(1.0, base + self.config.emotional_boost)

    def _calculate_type_boost(self, episode: Episode) -> float:
        """Calculate boost based on episode type."""
        type_name = episode.episode_type.value
        return self.config.type_boosts.get(type_name, 0.0)

    def add_custom_scorer(self, scorer: Callable[[Episode, EpisodeQuery], float]):
        """
        Add a custom scoring function.

        Args:
            scorer: Function taking (episode, query) returning bonus score
        """
        self._custom_scorers.append(scorer)

    def batch_score(
        self,
        episodes: List[Episode],
        query: EpisodeQuery,
        similarities: Optional[List[float]] = None
    ) -> List[Tuple[Episode, float, Dict[str, float]]]:
        """
        Score multiple episodes.

        Args:
            episodes: List of episodes to score
            query: Query context
            similarities: Optional list of vector similarities

        Returns:
            List of (episode, score, breakdown) tuples sorted by score
        """
        if similarities is None:
            similarities = [0.5] * len(episodes)

        results = []
        for episode, sim in zip(episodes, similarities):
            score, breakdown = self.score(episode, query, sim)
            if score >= self.config.min_score:
                results.append((episode, score, breakdown))

        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)

        return results


class AdaptiveScorer(MemoryScorer):
    """
    Adaptive scorer that learns from user feedback.

    Tracks which scoring factors lead to useful retrievals
    and adjusts weights accordingly.
    """

    def __init__(self, config: Optional[ScoringConfig] = None):
        super().__init__(config)

        # Learning state
        self._feedback_history: List[Dict[str, Any]] = []
        self._weight_adjustments: Dict[str, float] = {
            "recency": 0.0,
            "importance": 0.0,
            "relevance": 0.0,
            "access": 0.0,
            "emotional": 0.0
        }

    def record_feedback(
        self,
        episode: Episode,
        query: EpisodeQuery,
        was_useful: bool,
        breakdown: Dict[str, float]
    ):
        """
        Record feedback for learning.

        Args:
            episode: Episode that was retrieved
            query: Query that triggered retrieval
            was_useful: Whether the retrieval was helpful
            breakdown: Score breakdown from retrieval
        """
        self._feedback_history.append({
            "episode_id": episode.id,
            "was_useful": was_useful,
            "breakdown": breakdown,
            "timestamp": datetime.now()
        })

        # Simple online learning: adjust weights based on feedback
        learning_rate = 0.1
        reward = 1.0 if was_useful else -0.5

        for factor in self._weight_adjustments.keys():
            if factor in breakdown:
                # Increase weight if factor was high and result was useful
                # Decrease if factor was high but result was not useful
                contribution = breakdown[factor] * reward
                self._weight_adjustments[factor] += learning_rate * contribution

    def score(
        self,
        episode: Episode,
        query: EpisodeQuery,
        vector_similarity: float = 0.5
    ) -> Tuple[float, Dict[str, float]]:
        """Score with adaptive weights."""
        # Get base score
        base_score, breakdown = super().score(episode, query, vector_similarity)

        # Apply learned adjustments
        adjustment = 0.0
        for factor, adj in self._weight_adjustments.items():
            if factor in breakdown:
                adjustment += adj * breakdown[factor]

        # Normalize adjustment to prevent runaway scores
        adjustment = max(-0.2, min(0.2, adjustment))

        final_score = max(0.0, min(1.0, base_score + adjustment))
        breakdown["adaptive_adjustment"] = adjustment
        breakdown["final"] = final_score

        return final_score, breakdown

    def get_learned_weights(self) -> Dict[str, float]:
        """Get current learned weight adjustments."""
        return dict(self._weight_adjustments)

    def reset_learning(self):
        """Reset learned adjustments."""
        self._feedback_history.clear()
        for key in self._weight_adjustments:
            self._weight_adjustments[key] = 0.0


class ContextualScorer(MemoryScorer):
    """
    Context-aware scorer that considers current situation.

    Boosts memories relevant to:
    - Current time of day
    - Current task/activity
    - Recent conversation topics
    """

    def __init__(
        self,
        config: Optional[ScoringConfig] = None,
        context_provider: Optional[Callable[[], Dict[str, Any]]] = None
    ):
        super().__init__(config)
        self._context_provider = context_provider or (lambda: {})

    def score(
        self,
        episode: Episode,
        query: EpisodeQuery,
        vector_similarity: float = 0.5
    ) -> Tuple[float, Dict[str, float]]:
        """Score with contextual awareness."""
        base_score, breakdown = super().score(episode, query, vector_similarity)

        # Get current context
        context = self._context_provider()

        # Time-of-day matching
        tod_bonus = 0.0
        if context.get("time_of_day"):
            if episode.temporal_context.time_of_day == context["time_of_day"]:
                tod_bonus = 0.05
        breakdown["tod_bonus"] = tod_bonus

        # Day-of-week matching (for recurring patterns)
        dow_bonus = 0.0
        if context.get("day_of_week"):
            if episode.temporal_context.day_of_week == context["day_of_week"]:
                dow_bonus = 0.03
        breakdown["dow_bonus"] = dow_bonus

        # Current task/entity matching
        entity_bonus = 0.0
        if context.get("current_entities"):
            overlap = set(episode.entities_involved) & set(context["current_entities"])
            if overlap:
                entity_bonus = 0.1 * min(1.0, len(overlap) / 2.0)
        breakdown["entity_bonus"] = entity_bonus

        # Tool usage matching
        tool_bonus = 0.0
        if context.get("current_tools"):
            overlap = set(episode.tools_used) & set(context["current_tools"])
            if overlap:
                tool_bonus = 0.05 * min(1.0, len(overlap) / 2.0)
        breakdown["tool_bonus"] = tool_bonus

        # Calculate final score
        final_score = min(1.0, base_score + tod_bonus + dow_bonus + entity_bonus + tool_bonus)
        breakdown["final"] = final_score

        return final_score, breakdown
