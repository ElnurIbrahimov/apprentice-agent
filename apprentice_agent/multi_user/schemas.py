"""
Shared enums and dataclasses for multi-user consciousness (ADV-04).

These types are used across user_mind_model, identity_core, manager,
and knowledge_abstractor modules.
"""

import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional


# ============================================================================
# Enums
# ============================================================================

class TrustLevel(str, Enum):
    """Calibrated trust levels for a user."""
    NEW = "new"                    # First interactions, high uncertainty
    ACQUAINTANCE = "acquaintance"  # Some history, building model
    FAMILIAR = "familiar"          # Reliable model, can anticipate needs
    TRUSTED = "trusted"            # Deep relationship, high mutual trust
    CAUTIOUS = "cautious"          # Trust reduced due to adversarial signals


class IdentityLayer(str, Enum):
    """AURA's four-layer identity model."""
    CONSTITUTIONAL = "constitutional"  # L0: Immutable values
    DEEP = "deep"                      # L1: Slow-changing personality
    ADAPTIVE = "adaptive"              # L2: Context-sensitive
    EXPRESSIVE = "expressive"          # L3: Per-user style


# ============================================================================
# Dataclasses
# ============================================================================

@dataclass
class MetaKnowledge:
    """What the user knows they know, and known blind spots."""
    # Topics user has explicitly acknowledged knowing
    self_reported_expertise: Dict[str, float] = field(default_factory=dict)
    # Topics where user shows Dunning-Kruger signals
    overconfidence_topics: List[str] = field(default_factory=list)
    # Topics user has asked about but forgotten (re-asked)
    forgotten_topics: List[str] = field(default_factory=list)
    # What user believes AURA knows/remembers
    user_expectations_of_aura: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RelationshipState:
    """Tracks the evolving relationship between AURA and a user."""
    first_interaction: str = ""          # ISO timestamp
    last_interaction: str = ""           # ISO timestamp
    total_interactions: int = 0
    total_messages: int = 0
    total_sessions: int = 0
    avg_session_duration_min: float = 0.0
    trust_level: TrustLevel = TrustLevel.NEW
    trust_score: float = 0.5             # 0-1 continuous
    rapport_score: float = 0.5           # 0-1 (warmth of relationship)
    cooperation_score: float = 0.5       # 0-1 (how well we work together)
    # Memorable shared experiences
    shared_milestones: List[str] = field(default_factory=list)
    # Inside jokes or recurring references
    shared_references: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["trust_level"] = self.trust_level.value
        return d


@dataclass
class EmotionalResonance:
    """How this user typically affects AURA's emotional state."""
    # Average emotional impact of interactions with this user
    avg_pleasure_delta: float = 0.0
    avg_arousal_delta: float = 0.0
    avg_dominance_delta: float = 0.0
    # User's typical emotional triggers
    positive_triggers: List[str] = field(default_factory=list)
    negative_triggers: List[str] = field(default_factory=list)
    # Humor compatibility (0 = mismatch, 1 = great match)
    humor_compatibility: float = 0.5
    samples: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class UserSession:
    """Active session for a user."""
    user_id: str
    session_id: str
    started_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    message_count: int = 0
    is_active: bool = True
    platform: str = "web"
    channel_id: str = ""
    conversation_id: Optional[str] = None

    @property
    def duration_minutes(self) -> float:
        return (time.time() - self.started_at) / 60

    @property
    def idle_minutes(self) -> float:
        return (time.time() - self.last_activity) / 60

    def touch(self) -> None:
        self.last_activity = time.time()
        self.message_count += 1
