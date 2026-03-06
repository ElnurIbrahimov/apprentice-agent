"""
Episode Data Model for AURA Episodic Time-Travel Memory.

Defines the core Episode structure and related types for storing
autobiographical memories with temporal context.
"""

import hashlib
import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class EpisodeType(Enum):
    """Types of episodes that can be stored."""
    CONVERSATION = "conversation"      # User-agent dialogue
    TASK_EXECUTION = "task_execution"  # Tool usage and results
    LEARNING = "learning"              # New knowledge acquired
    ERROR = "error"                    # Failures and recovery
    MILESTONE = "milestone"            # Important achievements
    INSIGHT = "insight"                # Agent realizations
    USER_PREFERENCE = "user_preference"  # Learned preferences
    SYSTEM_EVENT = "system_event"      # System-level events


class EmotionalValence(Enum):
    """Emotional tone of an episode."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"


@dataclass
class TemporalContext:
    """Temporal metadata for an episode."""
    timestamp: datetime
    duration_seconds: Optional[float] = None
    time_of_day: Optional[str] = None  # morning, afternoon, evening, night
    day_of_week: Optional[str] = None
    is_weekend: bool = False
    session_id: Optional[str] = None

    def __post_init__(self):
        """Derive temporal fields from timestamp."""
        if self.time_of_day is None:
            hour = self.timestamp.hour
            if 5 <= hour < 12:
                self.time_of_day = "morning"
            elif 12 <= hour < 17:
                self.time_of_day = "afternoon"
            elif 17 <= hour < 21:
                self.time_of_day = "evening"
            else:
                self.time_of_day = "night"

        if self.day_of_week is None:
            self.day_of_week = self.timestamp.strftime("%A").lower()

        self.is_weekend = self.timestamp.weekday() >= 5

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "duration_seconds": self.duration_seconds,
            "time_of_day": self.time_of_day,
            "day_of_week": self.day_of_week,
            "is_weekend": self.is_weekend,
            "session_id": self.session_id
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TemporalContext":
        """Create from dictionary."""
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            duration_seconds=data.get("duration_seconds"),
            time_of_day=data.get("time_of_day"),
            day_of_week=data.get("day_of_week"),
            is_weekend=data.get("is_weekend", False),
            session_id=data.get("session_id")
        )


@dataclass
class Episode:
    """
    A single episodic memory unit.

    Episodes are autobiographical memories with temporal context,
    content, and metadata for retrieval and scoring.
    """
    content: str                           # Main content/description
    episode_type: EpisodeType              # Type of episode
    temporal_context: TemporalContext      # When it happened

    # Optional fields
    id: Optional[str] = None               # Unique ID (generated if not provided)
    title: Optional[str] = None            # Brief title/summary
    importance: float = 0.5                # 0.0 to 1.0
    emotional_valence: EmotionalValence = EmotionalValence.NEUTRAL

    # Context and relationships
    entities_involved: List[str] = field(default_factory=list)  # Related entities
    tools_used: List[str] = field(default_factory=list)         # Tools involved
    related_episode_ids: List[str] = field(default_factory=list)  # Linked episodes

    # Retrieval metadata
    access_count: int = 0                  # How often retrieved
    last_accessed: Optional[datetime] = None
    embedding: Optional[List[float]] = None  # Vector embedding

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Generate ID if not provided."""
        if self.id is None:
            self.id = f"ep_{uuid.uuid4().hex}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "content": self.content,
            "title": self.title,
            "episode_type": self.episode_type.value,
            "temporal_context": self.temporal_context.to_dict(),
            "importance": self.importance,
            "emotional_valence": self.emotional_valence.value,
            "entities_involved": self.entities_involved,
            "tools_used": self.tools_used,
            "related_episode_ids": self.related_episode_ids,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], embedding: Optional[List[float]] = None) -> "Episode":
        """Create Episode from dictionary."""
        return cls(
            id=data.get("id"),
            content=data["content"],
            title=data.get("title"),
            episode_type=EpisodeType(data["episode_type"]),
            temporal_context=TemporalContext.from_dict(data["temporal_context"]),
            importance=data.get("importance", 0.5),
            emotional_valence=EmotionalValence(data.get("emotional_valence", "neutral")),
            entities_involved=data.get("entities_involved", []),
            tools_used=data.get("tools_used", []),
            related_episode_ids=data.get("related_episode_ids", []),
            access_count=data.get("access_count", 0),
            last_accessed=datetime.fromisoformat(data["last_accessed"]) if data.get("last_accessed") else None,
            embedding=embedding,
            metadata=data.get("metadata", {})
        )

    def mark_accessed(self):
        """Update access metadata."""
        self.access_count += 1
        self.last_accessed = datetime.now()

    def get_age_hours(self) -> float:
        """Get age of episode in hours."""
        delta = datetime.now() - self.temporal_context.timestamp
        return delta.total_seconds() / 3600

    def get_recency_score(self, half_life_hours: float = 168.0) -> float:
        """
        Calculate recency score using exponential decay.

        Args:
            half_life_hours: Hours for score to decay to 0.5 (default: 1 week)

        Returns:
            Score from 0.0 to 1.0
        """
        age_hours = self.get_age_hours()
        decay_rate = math.log(2) / half_life_hours
        return math.exp(-decay_rate * age_hours)


@dataclass
class EpisodeQuery:
    """Query parameters for episode retrieval."""
    query_text: Optional[str] = None

    # Temporal filters
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    time_of_day: Optional[str] = None
    day_of_week: Optional[str] = None

    # Type filters
    episode_types: Optional[List[EpisodeType]] = None
    emotional_valence: Optional[EmotionalValence] = None

    # Entity/tool filters
    entities: Optional[List[str]] = None
    tools: Optional[List[str]] = None

    # Scoring weights
    recency_weight: float = 0.3
    importance_weight: float = 0.3
    relevance_weight: float = 0.4

    # Emotional congruence
    emotional_pad: Optional[Dict[str, float]] = None
    emotional_weight: float = 0.0

    # Limits
    limit: int = 10
    min_score: float = 0.0


@dataclass
class EpisodeSearchResult:
    """Result from episode search."""
    episode: Episode
    score: float
    score_breakdown: Dict[str, float] = field(default_factory=dict)

    def __lt__(self, other):
        """Enable sorting by score."""
        return self.score < other.score
