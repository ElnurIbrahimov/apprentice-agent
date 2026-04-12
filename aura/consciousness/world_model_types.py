"""Shared schema types and helpers for the persistent world model."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class ProjectStatus(str, Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    ABANDONED = "abandoned"


class ProjectHealth(str, Enum):
    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"


class GoalHorizon(str, Enum):
    SHORT_TERM = "short_term"
    MEDIUM_TERM = "medium_term"
    LONG_TERM = "long_term"


class BeliefCategory(str, Enum):
    USER_INTENT = "user_intent"
    TECHNICAL_CONSTRAINT = "technical_constraint"
    PREFERENCE = "preference"
    PROJECT_STATE = "project_state"
    RELATIONSHIP = "relationship"
    SCHEDULE = "schedule"
    HABIT = "habit"
    ENVIRONMENT = "environment"


class ChangeType(str, Enum):
    PROJECT_UPDATE = "project_update"
    GOAL_UPDATE = "goal_update"
    BELIEF_FORMED = "belief_formed"
    BELIEF_REVISED = "belief_revised"
    CONTRADICTION_DETECTED = "contradiction_detected"
    RELATIONSHIP_UPDATE = "relationship_update"
    ENVIRONMENT_UPDATE = "environment_update"
    BLOCKER_ADDED = "blocker_added"
    BLOCKER_RESOLVED = "blocker_resolved"


@dataclass
class Project:
    """A user project tracked by the world model."""

    id: str
    name: str
    status: ProjectStatus = ProjectStatus.ACTIVE
    description: str = ""
    created_at: str = ""
    last_mentioned: str = ""
    last_activity: str = ""
    mention_count: int = 1
    priority: float = 0.5
    health: ProjectHealth = ProjectHealth.GREEN
    technologies: list[str] = field(default_factory=list)


@dataclass
class Goal:
    """A user goal at a specific time horizon."""

    id: str
    description: str
    horizon: GoalHorizon = GoalHorizon.SHORT_TERM
    created_at: str = ""
    target_date: str | None = None
    progress: float = 0.0
    status: str = "active"
    related_project_ids: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)


@dataclass
class Belief:
    """A structured belief about the user's world."""

    id: str
    statement: str
    confidence: float = 0.7
    category: BeliefCategory = BeliefCategory.USER_INTENT
    evidence: list[str] = field(default_factory=list)
    first_formed: str = ""
    last_reinforced: str = ""
    valid_from: str = ""
    valid_to: str | None = None
    superseded_by: str | None = None
    source_conversation_ids: list[str] = field(default_factory=list)


@dataclass
class Relationship:
    """A person mentioned in conversations."""

    id: str
    name: str
    role: str = ""
    relationship_type: str = "mentioned_person"
    first_mentioned: str = ""
    last_mentioned: str = ""
    mention_count: int = 1
    context_notes: list[str] = field(default_factory=list)
    sentiment: str = "neutral"


@dataclass
class Contradiction:
    """A detected contradiction between beliefs."""

    id: str
    belief_a_id: str
    belief_b_id: str
    description: str
    detected_at: str
    resolution: str | None = None
    resolution_details: str | None = None
    resolved_at: str | None = None


@dataclass
class StateChange:
    """A logged change to the world model."""

    timestamp: str
    conversation_id: str | None
    change_type: ChangeType
    entity_type: str
    entity_id: str
    old_value: dict[str, Any] | None = None
    new_value: dict[str, Any] | None = None
    reasoning: str = ""


def now_iso() -> str:
    """Return current UTC time as ISO 8601 string."""

    return datetime.now(timezone.utc).isoformat()


def gen_id(prefix: str = "") -> str:
    """Generate a unique ID with an optional prefix."""

    short = uuid.uuid4().hex[:12]
    return f"{prefix}_{short}" if prefix else short


def json_dumps(obj: Any) -> str:
    """Safe JSON serialization."""

    return json.dumps(obj, default=str)


def json_loads(value: str | None) -> Any:
    """Safe JSON deserialization."""

    if not value:
        return None
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return None


__all__ = [
    "Belief",
    "BeliefCategory",
    "ChangeType",
    "Contradiction",
    "Goal",
    "GoalHorizon",
    "Project",
    "ProjectHealth",
    "ProjectStatus",
    "Relationship",
    "StateChange",
    "gen_id",
    "json_dumps",
    "json_loads",
    "now_iso",
]
