"""
Persistent World Model for AURA — ADV-02 Phase 1.

Maintains a continuously updated structured representation of the user's
projects, goals, environment, relationships, and active beliefs.

Implements Endsley's Situation Awareness framework:
  Level 1 (Perception): Extract state changes from conversations
  Level 2 (Comprehension): Update structured world state
  Level 3 (Projection): Infer implications, generate proactive insights (Phase 3)

Phase 1 delivers: SQLite schema, WorldModel class with in-memory cache,
CRUD operations, context summary for brain.py injection, maintenance
routines, and JSON snapshot persistence. The world model starts empty
and is populated manually or via API.

Storage: SQLite (data/world_model.db) + JSON snapshot (data/world_state.json)
Canonical DB: world_model.db (with underscore). Any "worldmodel.db" is stale.
"""

import json
import logging
import math
import os
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ============================================================================
# Enums
# ============================================================================

class ProjectStatus(str, Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    ABANDONED = "abandoned"


class ProjectHealth(str, Enum):
    GREEN = "green"      # On track, recent activity
    YELLOW = "yellow"    # Slowing down or minor blockers
    RED = "red"          # Stale, major blockers, or at risk


class GoalHorizon(str, Enum):
    SHORT_TERM = "short_term"    # Days to weeks
    MEDIUM_TERM = "medium_term"  # Weeks to months
    LONG_TERM = "long_term"      # Months to years


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


# ============================================================================
# Dataclasses
# ============================================================================

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
    technologies: List[str] = field(default_factory=list)


@dataclass
class Goal:
    """A user goal at a specific time horizon."""
    id: str
    description: str
    horizon: GoalHorizon = GoalHorizon.SHORT_TERM
    created_at: str = ""
    target_date: Optional[str] = None
    progress: float = 0.0
    status: str = "active"
    related_project_ids: List[str] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)


@dataclass
class Belief:
    """A structured belief about the user's world."""
    id: str
    statement: str
    confidence: float = 0.7
    category: BeliefCategory = BeliefCategory.USER_INTENT
    evidence: List[str] = field(default_factory=list)
    first_formed: str = ""
    last_reinforced: str = ""
    valid_from: str = ""
    valid_to: Optional[str] = None
    superseded_by: Optional[str] = None
    source_conversation_ids: List[str] = field(default_factory=list)


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
    context_notes: List[str] = field(default_factory=list)
    sentiment: str = "neutral"


@dataclass
class Contradiction:
    """A detected contradiction between beliefs."""
    id: str
    belief_a_id: str
    belief_b_id: str
    description: str
    detected_at: str
    resolution: Optional[str] = None
    resolution_details: Optional[str] = None
    resolved_at: Optional[str] = None


@dataclass
class StateChange:
    """A logged change to the world model."""
    timestamp: str
    conversation_id: Optional[str]
    change_type: ChangeType
    entity_type: str
    entity_id: str
    old_value: Optional[Dict] = None
    new_value: Optional[Dict] = None
    reasoning: str = ""


# ============================================================================
# Helper
# ============================================================================

def _now_iso() -> str:
    """Return current UTC time as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _gen_id(prefix: str = "") -> str:
    """Generate a unique ID with an optional prefix."""
    short = uuid.uuid4().hex[:12]
    return f"{prefix}_{short}" if prefix else short


def _json_dumps(obj: Any) -> str:
    """Safe JSON serialization."""
    return json.dumps(obj, default=str)


def _json_loads(s: Optional[str]) -> Any:
    """Safe JSON deserialization."""
    if not s:
        return None
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return None


# ============================================================================
# WorldModel Core
# ============================================================================

class WorldModel:
    """
    Persistent World Model — AURA's structured understanding of the user's world.

    Architecture:
    - SQLite for persistent storage with full audit trail
    - In-memory cache for fast access during conversations
    - JSON snapshot for export and quick startup
    - Context summary injected into brain.py system prompt

    Phase 1: Foundation — CRUD, context summary, maintenance.
    Phase 2 adds LLM-powered extraction pipeline.
    Phase 3 adds ProactiveAwarenessEngine.
    """

    # Default thresholds
    PROJECT_STALE_DAYS = 7        # Project goes yellow after 7 days silence
    PROJECT_CRITICAL_DAYS = 14    # Project goes red after 14 days silence
    BELIEF_DECAY_HALF_LIFE = 336  # Hours (2 weeks) for belief confidence decay
    STATE_CHANGE_RETENTION = 1000  # Keep last N state changes

    def __init__(
        self,
        db_path: Optional[str] = None,
        snapshot_path: Optional[str] = None,
        enabled: bool = True,
    ):
        self.enabled = enabled
        self._lock = threading.RLock()

        # Resolve paths
        if db_path:
            self._db_path = str(Path(db_path))
        else:
            data_dir = Path(os.getenv("AURA_DATA_DIR", "data"))
            data_dir.mkdir(parents=True, exist_ok=True)
            self._db_path = str(data_dir / "world_model.db")

        if snapshot_path:
            self._snapshot_path = str(Path(snapshot_path))
        else:
            data_dir = Path(os.getenv("AURA_DATA_DIR", "data"))
            data_dir.mkdir(parents=True, exist_ok=True)
            self._snapshot_path = str(data_dir / "world_state.json")

        # In-memory cache
        self._projects: Dict[str, Project] = {}
        self._goals: Dict[str, Goal] = {}
        self._beliefs: Dict[str, Belief] = {}  # Only current (valid_to=None)
        self._relationships: Dict[str, Relationship] = {}
        self._environment: Dict[str, Dict[str, Any]] = {}
        self._contradictions: Dict[str, Contradiction] = {}  # Only unresolved

        if self.enabled:
            self._init_db()
            self._load_from_db()
            logger.info(f"[WorldModel] Initialized at {self._db_path}")

    # ----------------------------------------------------------------
    # Database helpers
    # ----------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        """Create a connection with WAL mode and row factory."""
        conn = sqlite3.connect(self._db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Create all 11 tables and 9 indices."""
        conn = sqlite3.connect(self._db_path)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")
            conn.executescript("""
            CREATE TABLE IF NOT EXISTS projects (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                status TEXT DEFAULT 'active',
                description TEXT,
                created_at TEXT NOT NULL,
                last_mentioned TEXT NOT NULL,
                last_activity TEXT NOT NULL,
                mention_count INTEGER DEFAULT 1,
                priority REAL DEFAULT 0.5,
                health TEXT DEFAULT 'green',
                technologies TEXT,
                metadata TEXT
            );

            CREATE TABLE IF NOT EXISTS project_blockers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id TEXT NOT NULL REFERENCES projects(id),
                description TEXT NOT NULL,
                severity TEXT DEFAULT 'medium',
                identified_at TEXT NOT NULL,
                resolved_at TEXT,
                status TEXT DEFAULT 'ongoing',
                resolution TEXT
            );

            CREATE TABLE IF NOT EXISTS project_milestones (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id TEXT NOT NULL REFERENCES projects(id),
                name TEXT NOT NULL,
                status TEXT DEFAULT 'pending',
                target_date TEXT,
                completed_at TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS goals (
                id TEXT PRIMARY KEY,
                description TEXT NOT NULL,
                horizon TEXT NOT NULL,
                created_at TEXT NOT NULL,
                target_date TEXT,
                progress REAL DEFAULT 0.0,
                status TEXT DEFAULT 'active',
                related_project_ids TEXT,
                evidence TEXT
            );

            CREATE TABLE IF NOT EXISTS goal_blockers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                goal_id TEXT NOT NULL REFERENCES goals(id),
                blocker_description TEXT NOT NULL,
                blocker_source TEXT,
                inferred_at TEXT NOT NULL,
                confidence REAL DEFAULT 0.5
            );

            CREATE TABLE IF NOT EXISTS relationships (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                role TEXT,
                relationship_type TEXT,
                first_mentioned TEXT NOT NULL,
                last_mentioned TEXT NOT NULL,
                mention_count INTEGER DEFAULT 1,
                context_notes TEXT,
                sentiment TEXT DEFAULT 'neutral'
            );

            CREATE TABLE IF NOT EXISTS beliefs (
                id TEXT PRIMARY KEY,
                statement TEXT NOT NULL,
                confidence REAL DEFAULT 0.7,
                category TEXT NOT NULL,
                evidence TEXT NOT NULL,
                first_formed TEXT NOT NULL,
                last_reinforced TEXT NOT NULL,
                valid_from TEXT NOT NULL,
                valid_to TEXT,
                superseded_by TEXT,
                source_conversation_ids TEXT
            );

            CREATE TABLE IF NOT EXISTS contradictions (
                id TEXT PRIMARY KEY,
                belief_a_id TEXT NOT NULL REFERENCES beliefs(id),
                belief_b_id TEXT NOT NULL REFERENCES beliefs(id),
                description TEXT NOT NULL,
                detected_at TEXT NOT NULL,
                resolution TEXT,
                resolution_details TEXT,
                resolved_at TEXT
            );

            CREATE TABLE IF NOT EXISTS environment (
                key TEXT PRIMARY KEY,
                category TEXT NOT NULL,
                value TEXT NOT NULL,
                confidence REAL DEFAULT 0.8,
                first_observed TEXT NOT NULL,
                last_observed TEXT NOT NULL,
                observation_count INTEGER DEFAULT 1
            );

            CREATE TABLE IF NOT EXISTS state_changes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                conversation_id TEXT,
                change_type TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                old_value TEXT,
                new_value TEXT,
                reasoning TEXT
            );

            CREATE TABLE IF NOT EXISTS proactive_insights (
                id TEXT PRIMARY KEY,
                insight_type TEXT NOT NULL,
                title TEXT NOT NULL,
                description TEXT NOT NULL,
                urgency REAL DEFAULT 0.5,
                confidence REAL DEFAULT 0.5,
                generated_at TEXT NOT NULL,
                delivered_at TEXT,
                dismissed_at TEXT,
                acted_on_at TEXT,
                related_entity_type TEXT,
                related_entity_id TEXT,
                reasoning TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_projects_status
                ON projects(status);
            CREATE INDEX IF NOT EXISTS idx_projects_last_activity
                ON projects(last_activity);
            CREATE INDEX IF NOT EXISTS idx_beliefs_category
                ON beliefs(category);
            CREATE INDEX IF NOT EXISTS idx_beliefs_valid
                ON beliefs(valid_to);
            CREATE INDEX IF NOT EXISTS idx_state_changes_timestamp
                ON state_changes(timestamp);
            CREATE INDEX IF NOT EXISTS idx_state_changes_entity
                ON state_changes(entity_type, entity_id);
            CREATE INDEX IF NOT EXISTS idx_insights_type
                ON proactive_insights(insight_type);
            CREATE INDEX IF NOT EXISTS idx_insights_urgency
                ON proactive_insights(urgency);
            CREATE INDEX IF NOT EXISTS idx_insights_delivered
                ON proactive_insights(delivered_at);
        """)
            conn.commit()
        finally:
            conn.close()

    def _load_from_db(self) -> None:
        """Hydrate in-memory cache from SQLite."""
        conn = self._connect()
        try:
            # Projects
            for row in conn.execute("SELECT * FROM projects"):
                self._projects[row["id"]] = Project(
                    id=row["id"],
                    name=row["name"],
                    status=ProjectStatus(row["status"]),
                    description=row["description"] or "",
                    created_at=row["created_at"],
                    last_mentioned=row["last_mentioned"],
                    last_activity=row["last_activity"],
                    mention_count=row["mention_count"],
                    priority=row["priority"],
                    health=ProjectHealth(row["health"]),
                    technologies=_json_loads(row["technologies"]) or [],
                )

            # Goals
            for row in conn.execute("SELECT * FROM goals"):
                self._goals[row["id"]] = Goal(
                    id=row["id"],
                    description=row["description"],
                    horizon=GoalHorizon(row["horizon"]),
                    created_at=row["created_at"],
                    target_date=row["target_date"],
                    progress=row["progress"],
                    status=row["status"],
                    related_project_ids=_json_loads(row["related_project_ids"]) or [],
                    evidence=_json_loads(row["evidence"]) or [],
                )

            # Beliefs (only current — valid_to IS NULL)
            for row in conn.execute("SELECT * FROM beliefs WHERE valid_to IS NULL"):
                self._beliefs[row["id"]] = Belief(
                    id=row["id"],
                    statement=row["statement"],
                    confidence=row["confidence"],
                    category=BeliefCategory(row["category"]),
                    evidence=_json_loads(row["evidence"]) or [],
                    first_formed=row["first_formed"],
                    last_reinforced=row["last_reinforced"],
                    valid_from=row["valid_from"],
                    valid_to=None,
                    superseded_by=None,
                    source_conversation_ids=_json_loads(row["source_conversation_ids"]) or [],
                )

            # Relationships
            for row in conn.execute("SELECT * FROM relationships"):
                self._relationships[row["id"]] = Relationship(
                    id=row["id"],
                    name=row["name"],
                    role=row["role"] or "",
                    relationship_type=row["relationship_type"] or "mentioned_person",
                    first_mentioned=row["first_mentioned"],
                    last_mentioned=row["last_mentioned"],
                    mention_count=row["mention_count"],
                    context_notes=_json_loads(row["context_notes"]) or [],
                    sentiment=row["sentiment"] or "neutral",
                )

            # Environment
            for row in conn.execute("SELECT * FROM environment"):
                self._environment[row["key"]] = {
                    "key": row["key"],
                    "category": row["category"],
                    "value": _json_loads(row["value"]) if row["value"] and row["value"].startswith(("{", "[", '"')) else row["value"],
                    "confidence": row["confidence"],
                    "first_observed": row["first_observed"],
                    "last_observed": row["last_observed"],
                    "observation_count": row["observation_count"],
                }

            # Contradictions (only unresolved)
            for row in conn.execute("SELECT * FROM contradictions WHERE resolved_at IS NULL"):
                self._contradictions[row["id"]] = Contradiction(
                    id=row["id"],
                    belief_a_id=row["belief_a_id"],
                    belief_b_id=row["belief_b_id"],
                    description=row["description"],
                    detected_at=row["detected_at"],
                    resolution=row["resolution"],
                    resolution_details=row["resolution_details"],
                    resolved_at=None,
                )
        finally:
            conn.close()

    # ----------------------------------------------------------------
    # Projects
    # ----------------------------------------------------------------

    def add_project(
        self,
        name: str,
        description: str = "",
        status: ProjectStatus = ProjectStatus.ACTIVE,
        technologies: Optional[List[str]] = None,
        conversation_id: Optional[str] = None,
    ) -> Project:
        """Create a new project and dual-write to DB + cache."""
        with self._lock:
            now = _now_iso()
            project = Project(
                id=_gen_id("proj"),
                name=name,
                status=status,
                description=description,
                created_at=now,
                last_mentioned=now,
                last_activity=now,
                mention_count=1,
                priority=0.5,
                health=ProjectHealth.GREEN,
                technologies=technologies or [],
            )

            conn = self._connect()
            try:
                conn.execute(
                    """INSERT INTO projects
                       (id, name, status, description, created_at,
                        last_mentioned, last_activity, mention_count,
                        priority, health, technologies)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        project.id, project.name, project.status.value,
                        project.description, project.created_at,
                        project.last_mentioned, project.last_activity,
                        project.mention_count, project.priority,
                        project.health.value,
                        _json_dumps(project.technologies),
                    ),
                )
                conn.commit()
            finally:
                conn.close()

            self._projects[project.id] = project

            self._log_state_change(
                ChangeType.PROJECT_UPDATE, "project", project.id,
                None, asdict(project), conversation_id,
                reasoning=f"New project created: {name}",
            )

            self._update_snapshot()
            logger.info(f"[WorldModel] Added project: {name} ({project.id})")
            return project

    def update_project(
        self,
        project_id: str,
        conversation_id: Optional[str] = None,
        **fields,
    ) -> Optional[Project]:
        """Update non-None fields on a project, bump mention_count, dual-write."""
        with self._lock:
            project = self._projects.get(project_id)
            if not project:
                return None

            old_state = asdict(project)
            now = _now_iso()

            # Apply field updates
            for key, value in fields.items():
                if value is not None and hasattr(project, key):
                    if key == "status":
                        value = ProjectStatus(value) if isinstance(value, str) else value
                    elif key == "health":
                        value = ProjectHealth(value) if isinstance(value, str) else value
                    setattr(project, key, value)

            project.mention_count += 1
            project.last_mentioned = now
            project.last_activity = now

            conn = self._connect()
            try:
                conn.execute(
                    """UPDATE projects SET
                       name=?, status=?, description=?, last_mentioned=?,
                       last_activity=?, mention_count=?, priority=?,
                       health=?, technologies=?
                       WHERE id=?""",
                    (
                        project.name, project.status.value, project.description,
                        project.last_mentioned, project.last_activity,
                        project.mention_count, project.priority,
                        project.health.value,
                        _json_dumps(project.technologies),
                        project.id,
                    ),
                )
                conn.commit()
            finally:
                conn.close()

            self._log_state_change(
                ChangeType.PROJECT_UPDATE, "project", project.id,
                old_state, asdict(project), conversation_id,
            )

            self._update_snapshot()
            return project

    def get_project(self, project_id: str) -> Optional[Project]:
        """Get a project by ID from cache."""
        return self._projects.get(project_id)

    def get_projects_by_status(self, status: ProjectStatus) -> List[Project]:
        """Get all projects with a given status from cache."""
        return [p for p in self._projects.values() if p.status == status]

    def get_all_projects(self) -> List[Project]:
        """Get all cached projects."""
        return list(self._projects.values())

    # ----------------------------------------------------------------
    # Blockers
    # ----------------------------------------------------------------

    def add_blocker(
        self,
        project_id: str,
        description: str,
        severity: str = "medium",
        conversation_id: Optional[str] = None,
    ) -> int:
        """Insert a project blocker. Returns the blocker row ID."""
        with self._lock:
            now = _now_iso()
            conn = self._connect()
            try:
                cursor = conn.execute(
                    """INSERT INTO project_blockers
                       (project_id, description, severity, identified_at, status)
                       VALUES (?, ?, ?, ?, 'ongoing')""",
                    (project_id, description, severity, now),
                )
                blocker_id = cursor.lastrowid
                conn.commit()
            finally:
                conn.close()

            self._log_state_change(
                ChangeType.BLOCKER_ADDED, "project_blocker", str(blocker_id),
                None, {"project_id": project_id, "description": description, "severity": severity},
                conversation_id,
            )
            return blocker_id

    def resolve_blocker(
        self,
        blocker_id: int,
        resolution: str = "",
        conversation_id: Optional[str] = None,
    ) -> bool:
        """Resolve a blocker by setting resolved_at and status."""
        with self._lock:
            now = _now_iso()
            conn = self._connect()
            try:
                cursor = conn.execute(
                    """UPDATE project_blockers
                       SET resolved_at=?, status='resolved', resolution=?
                       WHERE id=? AND resolved_at IS NULL""",
                    (now, resolution, blocker_id),
                )
                conn.commit()
                updated = cursor.rowcount > 0
            finally:
                conn.close()

            if updated:
                self._log_state_change(
                    ChangeType.BLOCKER_RESOLVED, "project_blocker", str(blocker_id),
                    None, {"resolution": resolution},
                    conversation_id,
                )
            return updated

    def get_project_blockers(self, project_id: str) -> List[Dict]:
        """Get all blockers for a project from DB."""
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM project_blockers WHERE project_id=?",
                (project_id,),
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    # ----------------------------------------------------------------
    # Goals
    # ----------------------------------------------------------------

    def add_goal(
        self,
        description: str,
        horizon: GoalHorizon = GoalHorizon.SHORT_TERM,
        target_date: Optional[str] = None,
        related_project_ids: Optional[List[str]] = None,
        conversation_id: Optional[str] = None,
    ) -> Goal:
        """Create a new goal and dual-write."""
        with self._lock:
            now = _now_iso()
            goal = Goal(
                id=_gen_id("goal"),
                description=description,
                horizon=horizon,
                created_at=now,
                target_date=target_date,
                progress=0.0,
                status="active",
                related_project_ids=related_project_ids or [],
                evidence=[],
            )

            conn = self._connect()
            try:
                conn.execute(
                    """INSERT INTO goals
                       (id, description, horizon, created_at, target_date,
                        progress, status, related_project_ids, evidence)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        goal.id, goal.description, goal.horizon.value,
                        goal.created_at, goal.target_date, goal.progress,
                        goal.status, _json_dumps(goal.related_project_ids),
                        _json_dumps(goal.evidence),
                    ),
                )
                conn.commit()
            finally:
                conn.close()

            self._goals[goal.id] = goal

            self._log_state_change(
                ChangeType.GOAL_UPDATE, "goal", goal.id,
                None, asdict(goal), conversation_id,
                reasoning=f"New goal created: {description}",
            )

            self._update_snapshot()
            return goal

    def update_goal(
        self,
        goal_id: str,
        conversation_id: Optional[str] = None,
        **fields,
    ) -> Optional[Goal]:
        """Update goal fields and dual-write."""
        with self._lock:
            goal = self._goals.get(goal_id)
            if not goal:
                return None

            old_state = asdict(goal)

            for key, value in fields.items():
                if value is not None and hasattr(goal, key):
                    if key == "horizon":
                        value = GoalHorizon(value) if isinstance(value, str) else value
                    setattr(goal, key, value)

            conn = self._connect()
            try:
                conn.execute(
                    """UPDATE goals SET
                       description=?, horizon=?, target_date=?,
                       progress=?, status=?, related_project_ids=?, evidence=?
                       WHERE id=?""",
                    (
                        goal.description, goal.horizon.value, goal.target_date,
                        goal.progress, goal.status,
                        _json_dumps(goal.related_project_ids),
                        _json_dumps(goal.evidence),
                        goal.id,
                    ),
                )
                conn.commit()
            finally:
                conn.close()

            self._log_state_change(
                ChangeType.GOAL_UPDATE, "goal", goal.id,
                old_state, asdict(goal), conversation_id,
            )

            self._update_snapshot()
            return goal

    def get_active_goals(self, horizon: Optional[GoalHorizon] = None) -> List[Goal]:
        """Get active goals, optionally filtered by horizon."""
        results = [g for g in self._goals.values() if g.status == "active"]
        if horizon:
            results = [g for g in results if g.horizon == horizon]
        return results

    # ----------------------------------------------------------------
    # Beliefs
    # ----------------------------------------------------------------

    def add_belief(
        self,
        statement: str,
        category: BeliefCategory = BeliefCategory.USER_INTENT,
        confidence: float = 0.7,
        evidence: Optional[List[str]] = None,
        conversation_id: Optional[str] = None,
    ) -> Belief:
        """Create a new belief with valid_from=now, valid_to=None."""
        with self._lock:
            now = _now_iso()
            belief = Belief(
                id=_gen_id("belief"),
                statement=statement,
                confidence=min(max(confidence, 0.0), 1.0),
                category=category,
                evidence=evidence or [],
                first_formed=now,
                last_reinforced=now,
                valid_from=now,
                valid_to=None,
                superseded_by=None,
                source_conversation_ids=[conversation_id] if conversation_id else [],
            )

            conn = self._connect()
            try:
                conn.execute(
                    """INSERT INTO beliefs
                       (id, statement, confidence, category, evidence,
                        first_formed, last_reinforced, valid_from, valid_to,
                        superseded_by, source_conversation_ids)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        belief.id, belief.statement, belief.confidence,
                        belief.category.value, _json_dumps(belief.evidence),
                        belief.first_formed, belief.last_reinforced,
                        belief.valid_from, belief.valid_to,
                        belief.superseded_by,
                        _json_dumps(belief.source_conversation_ids),
                    ),
                )
                conn.commit()
            finally:
                conn.close()

            self._beliefs[belief.id] = belief

            self._log_state_change(
                ChangeType.BELIEF_FORMED, "belief", belief.id,
                None, asdict(belief), conversation_id,
                reasoning=f"New belief formed: {statement}",
            )

            self._update_snapshot()
            return belief

    def reinforce_belief(
        self,
        belief_id: str,
        evidence: Optional[str] = None,
        confidence_boost: float = 0.05,
    ) -> Optional[Belief]:
        """Reinforce a belief: update last_reinforced, append evidence, boost confidence."""
        with self._lock:
            belief = self._beliefs.get(belief_id)
            if not belief:
                return None

            now = _now_iso()
            belief.last_reinforced = now
            belief.confidence = min(belief.confidence + confidence_boost, 1.0)
            if evidence:
                belief.evidence.append(evidence)

            conn = self._connect()
            try:
                conn.execute(
                    """UPDATE beliefs SET
                       last_reinforced=?, confidence=?, evidence=?
                       WHERE id=?""",
                    (
                        belief.last_reinforced, belief.confidence,
                        _json_dumps(belief.evidence), belief.id,
                    ),
                )
                conn.commit()
            finally:
                conn.close()

            self._update_snapshot()
            return belief

    def supersede_belief(
        self,
        old_id: str,
        new_statement: str,
        new_category: Optional[BeliefCategory] = None,
        new_confidence: float = 0.7,
        new_evidence: Optional[List[str]] = None,
        conversation_id: Optional[str] = None,
    ) -> Optional[Belief]:
        """Supersede an old belief: mark valid_to=now on old, create new, log BELIEF_REVISED.

        All DB writes are wrapped in a single transaction to prevent orphaned
        beliefs if a crash occurs between the UPDATE and INSERT.
        """
        with self._lock:
            old_belief = self._beliefs.get(old_id)
            if not old_belief:
                return None

            now = _now_iso()

            # Build the new belief object in memory first
            new_belief = Belief(
                id=_gen_id("belief"),
                statement=new_statement,
                confidence=min(max(new_confidence, 0.0), 1.0),
                category=new_category or old_belief.category,
                evidence=new_evidence or [],
                first_formed=now,
                last_reinforced=now,
                valid_from=now,
                valid_to=None,
                superseded_by=None,
                source_conversation_ids=[conversation_id] if conversation_id else [],
            )

            # Single transaction: mark old as superseded + insert new belief
            conn = self._connect()
            try:
                conn.execute("BEGIN")
                conn.execute(
                    "UPDATE beliefs SET valid_to=?, superseded_by=? WHERE id=?",
                    (now, new_belief.id, old_id),
                )
                conn.execute(
                    """INSERT INTO beliefs
                       (id, statement, confidence, category, evidence,
                        first_formed, last_reinforced, valid_from, valid_to,
                        superseded_by, source_conversation_ids)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        new_belief.id, new_belief.statement, new_belief.confidence,
                        new_belief.category.value, _json_dumps(new_belief.evidence),
                        new_belief.first_formed, new_belief.last_reinforced,
                        new_belief.valid_from, new_belief.valid_to,
                        new_belief.superseded_by,
                        _json_dumps(new_belief.source_conversation_ids),
                    ),
                )
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
            finally:
                conn.close()

            # Update in-memory state only after successful DB commit
            old_belief.valid_to = now
            old_belief.superseded_by = new_belief.id
            del self._beliefs[old_id]
            self._beliefs[new_belief.id] = new_belief

            self._log_state_change(
                ChangeType.BELIEF_REVISED, "belief", old_id,
                {"statement": old_belief.statement},
                {"statement": new_statement, "new_id": new_belief.id},
                conversation_id,
                reasoning=f"Belief revised: '{old_belief.statement}' -> '{new_statement}'",
            )

            self._log_state_change(
                ChangeType.BELIEF_FORMED, "belief", new_belief.id,
                None, asdict(new_belief), conversation_id,
                reasoning=f"New belief formed: {new_statement}",
            )

            self._update_snapshot()
            return new_belief

    def get_current_beliefs(
        self, category: Optional[BeliefCategory] = None
    ) -> List[Belief]:
        """Get all current beliefs (valid_to=None), optionally filtered by category."""
        results = list(self._beliefs.values())
        if category:
            results = [b for b in results if b.category == category]
        return results

    # ----------------------------------------------------------------
    # Relationships
    # ----------------------------------------------------------------

    def add_relationship(
        self,
        name: str,
        role: str = "",
        relationship_type: str = "mentioned_person",
        context: Optional[str] = None,
        sentiment: str = "neutral",
        conversation_id: Optional[str] = None,
    ) -> Relationship:
        """Create a new relationship and dual-write."""
        with self._lock:
            now = _now_iso()
            rel = Relationship(
                id=_gen_id("rel"),
                name=name,
                role=role,
                relationship_type=relationship_type,
                first_mentioned=now,
                last_mentioned=now,
                mention_count=1,
                context_notes=[context] if context else [],
                sentiment=sentiment,
            )

            conn = self._connect()
            try:
                conn.execute(
                    """INSERT INTO relationships
                       (id, name, role, relationship_type, first_mentioned,
                        last_mentioned, mention_count, context_notes, sentiment)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        rel.id, rel.name, rel.role, rel.relationship_type,
                        rel.first_mentioned, rel.last_mentioned,
                        rel.mention_count, _json_dumps(rel.context_notes),
                        rel.sentiment,
                    ),
                )
                conn.commit()
            finally:
                conn.close()

            self._relationships[rel.id] = rel

            self._log_state_change(
                ChangeType.RELATIONSHIP_UPDATE, "relationship", rel.id,
                None, asdict(rel), conversation_id,
            )

            self._update_snapshot()
            return rel

    def update_relationship(
        self,
        rel_id: str,
        role: Optional[str] = None,
        context_note: Optional[str] = None,
        sentiment: Optional[str] = None,
        conversation_id: Optional[str] = None,
    ) -> Optional[Relationship]:
        """Update a relationship: bump mention_count, optionally append context."""
        with self._lock:
            rel = self._relationships.get(rel_id)
            if not rel:
                return None

            now = _now_iso()
            rel.mention_count += 1
            rel.last_mentioned = now

            if role is not None:
                rel.role = role
            if context_note:
                rel.context_notes.append(context_note)
            if sentiment is not None:
                rel.sentiment = sentiment

            conn = self._connect()
            try:
                conn.execute(
                    """UPDATE relationships SET
                       role=?, last_mentioned=?, mention_count=?,
                       context_notes=?, sentiment=?
                       WHERE id=?""",
                    (
                        rel.role, rel.last_mentioned, rel.mention_count,
                        _json_dumps(rel.context_notes), rel.sentiment,
                        rel.id,
                    ),
                )
                conn.commit()
            finally:
                conn.close()

            self._log_state_change(
                ChangeType.RELATIONSHIP_UPDATE, "relationship", rel.id,
                None, asdict(rel), conversation_id,
            )

            self._update_snapshot()
            return rel

    def get_relationship(self, name: str) -> Optional[Relationship]:
        """Find a relationship by person name (case-insensitive)."""
        name_lower = name.lower()
        for rel in self._relationships.values():
            if rel.name.lower() == name_lower:
                return rel
        return None

    # ----------------------------------------------------------------
    # Environment
    # ----------------------------------------------------------------

    def set_environment(
        self,
        key: str,
        category: str,
        value: Any,
        confidence: float = 0.8,
        conversation_id: Optional[str] = None,
    ) -> None:
        """Set an environment observation. INSERT OR REPLACE, increment observation_count."""
        with self._lock:
            now = _now_iso()
            existing = self._environment.get(key)
            obs_count = (existing["observation_count"] + 1) if existing else 1
            first_obs = existing["first_observed"] if existing else now

            value_str = _json_dumps(value) if not isinstance(value, str) else value

            conn = self._connect()
            try:
                conn.execute(
                    """INSERT OR REPLACE INTO environment
                       (key, category, value, confidence, first_observed,
                        last_observed, observation_count)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (key, category, value_str, confidence, first_obs, now, obs_count),
                )
                conn.commit()
            finally:
                conn.close()

            self._environment[key] = {
                "key": key,
                "category": category,
                "value": value,
                "confidence": confidence,
                "first_observed": first_obs,
                "last_observed": now,
                "observation_count": obs_count,
            }

            self._log_state_change(
                ChangeType.ENVIRONMENT_UPDATE, "environment", key,
                existing, self._environment[key], conversation_id,
            )

            self._update_snapshot()

    def get_environment(self, key: str) -> Optional[Dict]:
        """Get an environment observation by key from cache."""
        return self._environment.get(key)

    def get_environment_by_category(self, category: str) -> Dict[str, Any]:
        """Get all environment entries for a category."""
        return {
            k: v for k, v in self._environment.items()
            if v.get("category") == category
        }

    # ----------------------------------------------------------------
    # Contradictions
    # ----------------------------------------------------------------

    def add_contradiction(
        self,
        belief_a_id: str,
        belief_b_id: str,
        description: str,
        conversation_id: Optional[str] = None,
    ) -> Contradiction:
        """Create a new contradiction and log it."""
        with self._lock:
            now = _now_iso()
            contradiction = Contradiction(
                id=_gen_id("contra"),
                belief_a_id=belief_a_id,
                belief_b_id=belief_b_id,
                description=description,
                detected_at=now,
            )

            conn = self._connect()
            try:
                conn.execute(
                    """INSERT INTO contradictions
                       (id, belief_a_id, belief_b_id, description, detected_at)
                       VALUES (?, ?, ?, ?, ?)""",
                    (
                        contradiction.id, belief_a_id, belief_b_id,
                        description, now,
                    ),
                )
                conn.commit()
            finally:
                conn.close()

            self._contradictions[contradiction.id] = contradiction

            self._log_state_change(
                ChangeType.CONTRADICTION_DETECTED, "contradiction", contradiction.id,
                None, asdict(contradiction), conversation_id,
                reasoning=description,
            )

            self._update_snapshot()
            return contradiction

    def resolve_contradiction(
        self,
        contradiction_id: str,
        resolution: str,
        details: str = "",
        conversation_id: Optional[str] = None,
    ) -> bool:
        """Resolve a contradiction. Set resolved_at, remove from cache."""
        with self._lock:
            if contradiction_id not in self._contradictions:
                return False

            now = _now_iso()
            conn = self._connect()
            try:
                cursor = conn.execute(
                    """UPDATE contradictions SET
                       resolution=?, resolution_details=?, resolved_at=?
                       WHERE id=? AND resolved_at IS NULL""",
                    (resolution, details, now, contradiction_id),
                )
                conn.commit()
                updated = cursor.rowcount > 0
            finally:
                conn.close()

            if updated:
                del self._contradictions[contradiction_id]
                self._update_snapshot()
            return updated

    def get_unresolved_contradictions(self) -> List[Contradiction]:
        """Get all unresolved contradictions from cache."""
        return list(self._contradictions.values())

    # ----------------------------------------------------------------
    # State Changes
    # ----------------------------------------------------------------

    def _log_state_change(
        self,
        change_type: ChangeType,
        entity_type: str,
        entity_id: str,
        old_value: Any,
        new_value: Any,
        conversation_id: Optional[str] = None,
        reasoning: str = "",
    ) -> None:
        """Insert an audit row into state_changes."""
        now = _now_iso()
        conn = self._connect()
        try:
            conn.execute(
                """INSERT INTO state_changes
                   (timestamp, conversation_id, change_type, entity_type,
                    entity_id, old_value, new_value, reasoning)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    now, conversation_id, change_type.value, entity_type,
                    entity_id, _json_dumps(old_value) if old_value else None,
                    _json_dumps(new_value) if new_value else None,
                    reasoning,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def get_state_change_counts(
        self, entity_type: str, since: str, until: Optional[str] = None
    ) -> Dict[str, int]:
        """Count state changes per entity_id within a time window.

        Args:
            entity_type: Filter by entity type (e.g. "project").
            since: ISO timestamp for window start.
            until: ISO timestamp for window end (default: now).

        Returns:
            Dict mapping entity_id to change count.
        """
        conn = self._connect()
        try:
            if until:
                rows = conn.execute(
                    """SELECT entity_id, COUNT(*) as cnt
                       FROM state_changes
                       WHERE entity_type = ? AND timestamp >= ? AND timestamp < ?
                       GROUP BY entity_id""",
                    (entity_type, since, until),
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT entity_id, COUNT(*) as cnt
                       FROM state_changes
                       WHERE entity_type = ? AND timestamp >= ?
                       GROUP BY entity_id""",
                    (entity_type, since),
                ).fetchall()
            return {row["entity_id"]: row["cnt"] for row in rows}
        finally:
            conn.close()

    def get_recent_changes(self, limit: int = 20) -> List[StateChange]:
        """Get the most recent state changes from DB."""
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM state_changes ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [
                StateChange(
                    timestamp=row["timestamp"],
                    conversation_id=row["conversation_id"],
                    change_type=ChangeType(row["change_type"]),
                    entity_type=row["entity_type"],
                    entity_id=row["entity_id"],
                    old_value=_json_loads(row["old_value"]),
                    new_value=_json_loads(row["new_value"]),
                    reasoning=row["reasoning"] or "",
                )
                for row in rows
            ]
        finally:
            conn.close()

    # ----------------------------------------------------------------
    # Proactive Insights — ADV-02 Phase 3
    # ----------------------------------------------------------------

    def store_insight(self, insight) -> None:
        """Store a ProactiveInsight into the proactive_insights table."""
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """INSERT OR REPLACE INTO proactive_insights
                       (id, insight_type, title, description, urgency,
                        confidence, generated_at, delivered_at, dismissed_at,
                        acted_on_at, related_entity_type, related_entity_id,
                        reasoning)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        insight.id, insight.insight_type.value, insight.title,
                        insight.description, insight.urgency, insight.confidence,
                        insight.generated_at, insight.delivered_at,
                        insight.dismissed_at, insight.acted_on_at,
                        insight.related_entity_type, insight.related_entity_id,
                        insight.reasoning,
                    ),
                )
                conn.commit()
            finally:
                conn.close()

    def get_pending_insights(self, max_count: int = 3) -> List[Dict]:
        """Get undelivered, non-dismissed insights ordered by urgency."""
        conn = self._connect()
        try:
            rows = conn.execute(
                """SELECT * FROM proactive_insights
                   WHERE delivered_at IS NULL AND dismissed_at IS NULL
                   ORDER BY urgency DESC
                   LIMIT ?""",
                (max_count,),
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def update_insight_feedback(self, insight_id: str, feedback: str) -> bool:
        """Update insight based on feedback: 'engaged', 'dismissed', or 'delivered'."""
        now = _now_iso()
        with self._lock:
            conn = self._connect()
            try:
                if feedback == "engaged":
                    cursor = conn.execute(
                        "UPDATE proactive_insights SET acted_on_at=? WHERE id=?",
                        (now, insight_id),
                    )
                elif feedback == "dismissed":
                    cursor = conn.execute(
                        "UPDATE proactive_insights SET dismissed_at=? WHERE id=?",
                        (now, insight_id),
                    )
                elif feedback == "delivered":
                    cursor = conn.execute(
                        "UPDATE proactive_insights SET delivered_at=? WHERE id=?",
                        (now, insight_id),
                    )
                else:
                    return False
                conn.commit()
                return cursor.rowcount > 0
            finally:
                conn.close()

    def get_insight_engagement_rates(self, days: int = 30) -> Dict[str, Dict]:
        """Compute engagement rates per insight type over the last N days.

        Returns {insight_type: {total, engaged, dismissed, engagement_rate}}.
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        conn = self._connect()
        try:
            rows = conn.execute(
                """SELECT insight_type,
                          COUNT(*) as total,
                          SUM(CASE WHEN acted_on_at IS NOT NULL THEN 1 ELSE 0 END) as engaged,
                          SUM(CASE WHEN dismissed_at IS NOT NULL THEN 1 ELSE 0 END) as dismissed
                   FROM proactive_insights
                   WHERE generated_at > ?
                   GROUP BY insight_type""",
                (cutoff,),
            ).fetchall()
            result = {}
            for row in rows:
                engaged = row["engaged"]
                dismissed = row["dismissed"]
                engagement_rate = engaged / max(1, engaged + dismissed)
                result[row["insight_type"]] = {
                    "total": row["total"],
                    "engaged": engaged,
                    "dismissed": dismissed,
                    "engagement_rate": engagement_rate,
                }
            return result
        finally:
            conn.close()

    def get_recent_insights(
        self, insight_type: Optional[str] = None, hours: int = 24
    ) -> List[Dict]:
        """Get insights generated within the last N hours, optionally filtered by type."""
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        conn = self._connect()
        try:
            if insight_type:
                rows = conn.execute(
                    """SELECT * FROM proactive_insights
                       WHERE generated_at > ? AND insight_type = ?
                       ORDER BY generated_at DESC""",
                    (cutoff, insight_type),
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT * FROM proactive_insights
                       WHERE generated_at > ?
                       ORDER BY generated_at DESC""",
                    (cutoff,),
                ).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    # ----------------------------------------------------------------
    # Context Summary — for brain.py system prompt injection
    # ----------------------------------------------------------------

    def get_context_summary(self, max_tokens: int = 500) -> str:
        """
        Generate a concise summary of current world state for LLM context.

        Injected into every LLM call via brain.py system prompt.
        Empty sections are omitted. Returns "" if disabled or no data.
        """
        if not self.enabled:
            return ""

        sections = []

        # Active projects (top 5 by priority)
        active = self.get_projects_by_status(ProjectStatus.ACTIVE)
        if active:
            active.sort(key=lambda p: p.priority, reverse=True)
            parts = []
            for p in active[:5]:
                staleness = self._days_since(p.last_activity)
                if staleness is not None and staleness > 0:
                    parts.append(f"{p.name} ({staleness}d stale)")
                else:
                    parts.append(f"{p.name} (today)")
            if len(active) > 5:
                parts.append(f"+{len(active) - 5} more")
            sections.append(f"Active projects: {', '.join(parts)}")

        # Current goals (top 3 by progress)
        goals = self.get_active_goals()
        if goals:
            goals.sort(key=lambda g: g.progress, reverse=True)
            parts = []
            for g in goals[:3]:
                pct = int(g.progress * 100)
                parts.append(f"{g.description[:60]} ({pct}%)")
            if len(goals) > 3:
                parts.append(f"+{len(goals) - 3} more")
            sections.append(f"Current goals: {', '.join(parts)}")

        # Active blockers (top 3 from active projects)
        if active:
            all_blockers = []
            for p in active[:5]:
                blockers = self.get_project_blockers(p.id)
                for b in blockers:
                    if b.get("status") == "ongoing":
                        all_blockers.append(f"{b['description'][:60]} ({b['severity']})")
                    if len(all_blockers) >= 3:
                        break
                if len(all_blockers) >= 3:
                    break
            if all_blockers:
                sections.append(f"Blockers: {', '.join(all_blockers)}")

        # Key beliefs (top 3 by confidence)
        beliefs = self.get_current_beliefs()
        if beliefs:
            beliefs.sort(key=lambda b: b.confidence, reverse=True)
            parts = [b.statement for b in beliefs[:3]]
            sections.append(f"Key beliefs: {', '.join(parts)}")

        # People
        if self._relationships:
            parts = []
            rels = sorted(
                self._relationships.values(),
                key=lambda r: r.last_mentioned,
                reverse=True,
            )
            for r in rels[:3]:
                days = self._days_since(r.last_mentioned)
                if days is not None and days > 0:
                    parts.append(f"{r.name} ({r.role or r.relationship_type}, {days}d ago)")
                else:
                    parts.append(f"{r.name} ({r.role or r.relationship_type}, today)")
            sections.append(f"People: {', '.join(parts)}")

        # Contradictions (just count — 7000+ contradictions don't need detail)
        try:
            unresolved_count = len(self.get_unresolved_contradictions())
            if unresolved_count > 0:
                sections.append(f"Contradictions: {unresolved_count} unresolved")
        except Exception:
            pass

        # Pending proactive insight
        try:
            pending = self.get_pending_insights(max_count=1)
            if pending:
                sections.append(f"Pending insight: {pending[0]['title']}")
        except Exception as e:
            logger.debug(f"[WorldModel] pending insight fetch failed: {e}")

        if not sections:
            return ""

        return "[World State]\n" + "\n".join(sections)

    def _days_since(self, iso_str: str) -> Optional[int]:
        """Compute days since an ISO timestamp."""
        if not iso_str:
            return None
        try:
            dt = datetime.fromisoformat(iso_str)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            delta = datetime.now(timezone.utc) - dt
            return max(0, delta.days)
        except (ValueError, TypeError):
            return None

    # ----------------------------------------------------------------
    # Public query API (used by ProactiveAwareness, etc.)
    # ----------------------------------------------------------------

    def days_since(self, iso_str: str) -> Optional[int]:
        """Public wrapper: compute days since an ISO timestamp."""
        return self._days_since(iso_str)

    def get_relationships_snapshot(self) -> list:
        """Return a thread-safe copy of all relationships."""
        with self._lock:
            return list(self._relationships.values())

    # ----------------------------------------------------------------
    # Full state export
    # ----------------------------------------------------------------

    def get_full_state_json(self) -> Dict:
        """Get complete world state as JSON dict (for snapshot/export)."""
        return {
            "version": "1.0.0",
            "last_updated": _now_iso(),
            "projects": [asdict(p) for p in self._projects.values()],
            "goals": [asdict(g) for g in self._goals.values()],
            "beliefs": [asdict(b) for b in self._beliefs.values()],
            "relationships": [asdict(r) for r in self._relationships.values()],
            "environment": dict(self._environment),
            "contradictions": [asdict(c) for c in self._contradictions.values()],
        }

    # ----------------------------------------------------------------
    # Snapshot persistence
    # ----------------------------------------------------------------

    def _update_snapshot(self) -> None:
        """Atomic write of current state to JSON snapshot file.

        Must be called while holding self._lock, or from a path where no
        concurrent mutation is possible.
        """
        try:
            state = self.get_full_state_json()
            # Use tempfile to avoid concurrent writes clobbering a shared .tmp path
            import tempfile
            _dir = os.path.dirname(self._snapshot_path) or "."
            fd, tmp_path = tempfile.mkstemp(dir=_dir, suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(state, f, indent=2, default=str)
                # Atomic rename
                os.replace(tmp_path, self._snapshot_path)
            except Exception:
                # Clean up temp file on failure
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        except Exception as e:
            logger.debug(f"[WorldModel] Snapshot write failed: {e}")

    def _load_snapshot(self) -> Optional[Dict]:
        """Read JSON snapshot file. Returns dict or None."""
        try:
            with open(self._snapshot_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    # ----------------------------------------------------------------
    # Maintenance
    # ----------------------------------------------------------------

    def compute_adaptive_half_life(self) -> float:
        """Compute adaptive belief decay half-life from actual reinforcement intervals.

        Analyzes beliefs that have been reinforced at least once (last_reinforced != first_formed)
        to calibrate the half-life. Clamped between 168h (1 week) and 672h (4 weeks).
        Falls back to BELIEF_DECAY_HALF_LIFE (336h) if insufficient data.
        Results are cached with a 1-hour TTL.
        """
        now = time.monotonic()
        if (
            hasattr(self, "_adaptive_half_life")
            and hasattr(self, "_half_life_computed_at")
            and now - self._half_life_computed_at < 3600
        ):
            return self._adaptive_half_life

        conn = self._connect()
        try:
            rows = conn.execute(
                """SELECT first_formed, last_reinforced FROM beliefs
                   WHERE valid_to IS NULL
                   AND last_reinforced != first_formed"""
            ).fetchall()
        finally:
            conn.close()

        if len(rows) < 5:
            self._adaptive_half_life = self.BELIEF_DECAY_HALF_LIFE
            self._half_life_computed_at = now
            return self._adaptive_half_life

        intervals = []
        for row in rows:
            try:
                formed = datetime.fromisoformat(row["first_formed"])
                reinforced = datetime.fromisoformat(row["last_reinforced"])
                if formed.tzinfo is None:
                    formed = formed.replace(tzinfo=timezone.utc)
                if reinforced.tzinfo is None:
                    reinforced = reinforced.replace(tzinfo=timezone.utc)
                hours = (reinforced - formed).total_seconds() / 3600
                if hours > 0:
                    intervals.append(hours)
            except (ValueError, TypeError):
                continue

        if len(intervals) < 5:
            self._adaptive_half_life = self.BELIEF_DECAY_HALF_LIFE
            self._half_life_computed_at = now
            return self._adaptive_half_life

        intervals.sort()
        median_interval = intervals[len(intervals) // 2]
        self._adaptive_half_life = max(168, min(672, median_interval * 2))
        self._half_life_computed_at = now
        return self._adaptive_half_life

    def decay_beliefs(self) -> int:
        """
        Apply Ebbinghaus-style decay to stale belief confidence.

        confidence *= e^(-decay_rate * hours_since_reinforced)
        where decay_rate = ln(2) / adaptive_half_life

        Returns count of beliefs whose confidence dropped.
        """
        half_life = self.compute_adaptive_half_life()
        decay_rate = math.log(2) / half_life
        decayed_count = 0
        now = datetime.now(timezone.utc)

        with self._lock:
            conn = self._connect()
            try:
                for belief in list(self._beliefs.values()):
                    try:
                        last = datetime.fromisoformat(belief.last_reinforced)
                        if last.tzinfo is None:
                            last = last.replace(tzinfo=timezone.utc)
                        hours_since = (now - last).total_seconds() / 3600
                    except (ValueError, TypeError):
                        continue

                    if hours_since <= 0:
                        continue

                    decay_factor = math.exp(-decay_rate * hours_since)
                    new_conf = belief.confidence * decay_factor

                    if new_conf < belief.confidence:
                        belief.confidence = max(new_conf, 0.01)
                        conn.execute(
                            "UPDATE beliefs SET confidence=? WHERE id=?",
                            (belief.confidence, belief.id),
                        )
                        decayed_count += 1

                conn.commit()
            finally:
                conn.close()

        return decayed_count

    def update_project_health(self) -> int:
        """
        Recompute project health based on activity recency and blockers.

        Green: active in last 3 days, no critical blockers
        Yellow: active in last 7 days, or has medium blockers
        Red: stale > 7 days, or has critical blockers

        Returns count of projects whose health changed.
        """
        changed = 0
        with self._lock:
            conn = self._connect()
            try:
                for project in list(self._projects.values()):
                    if project.status != ProjectStatus.ACTIVE:
                        continue

                    days = self._days_since(project.last_activity)
                    if days is None:
                        continue

                    # Check blockers
                    blockers = self.get_project_blockers(project.id)
                    has_critical = any(
                        b.get("severity") == "critical" and b.get("status") == "ongoing"
                        for b in blockers
                    )
                    has_medium = any(
                        b.get("severity") in ("medium", "high") and b.get("status") == "ongoing"
                        for b in blockers
                    )

                    # Determine health
                    if has_critical or days > self.PROJECT_STALE_DAYS:
                        new_health = ProjectHealth.RED
                    elif has_medium or days > 3:
                        new_health = ProjectHealth.YELLOW
                    else:
                        new_health = ProjectHealth.GREEN

                    if new_health != project.health:
                        project.health = new_health
                        conn.execute(
                            "UPDATE projects SET health=? WHERE id=?",
                            (new_health.value, project.id),
                        )
                        changed += 1

                conn.commit()
            finally:
                conn.close()

        return changed

    def compute_project_priority(self, project: Project) -> float:
        """
        Compute project priority from multiple signals.

        Factors: recency (40%), mention_frequency (30%),
                 blocker_severity (15%), goal_alignment (15%).
        """
        # Recency signal (0-1): decays over 30 days
        days = self._days_since(project.last_activity)
        if days is not None:
            recency = max(0.0, 1.0 - (days / 30.0))
        else:
            recency = 0.5

        # Mention frequency signal (0-1): normalized by max
        max_mentions = max(
            (p.mention_count for p in self._projects.values()), default=1
        )
        frequency = project.mention_count / max(max_mentions, 1)

        # Blocker penalty (0-1): more severe blockers = lower score
        blockers = self.get_project_blockers(project.id)
        active_blockers = [b for b in blockers if b.get("status") == "ongoing"]
        severity_map = {"low": 0.1, "medium": 0.2, "high": 0.4, "critical": 0.6}
        blocker_penalty = sum(
            severity_map.get(b.get("severity", "medium"), 0.2)
            for b in active_blockers
        )
        blocker_score = max(0.0, 1.0 - blocker_penalty)

        # Goal alignment (0-1): how many active goals reference this project
        goal_count = sum(
            1 for g in self._goals.values()
            if g.status == "active" and project.id in g.related_project_ids
        )
        goal_score = min(1.0, goal_count * 0.3)

        priority = (
            recency * 0.4
            + frequency * 0.3
            + blocker_score * 0.15
            + goal_score * 0.15
        )
        return round(min(max(priority, 0.0), 1.0), 3)

    # ----------------------------------------------------------------
    # Extraction Pipeline — ADV-02 Phase 2
    # ----------------------------------------------------------------

    def process_conversation(
        self,
        conversation_id: Optional[str],
        messages: List[Dict[str, str]],
    ) -> Dict[str, int]:
        """
        Orchestrate extraction and application of world state changes.

        Called from brain.py after each think()/think_stream() call.
        Runs the StateExtractor, then applies each entity type.

        Returns:
            Counts dict: {projects_updated, goals_updated, beliefs_updated,
                          relationships_updated, environment_updated,
                          contradictions_detected}
        """
        counts = {
            "projects_updated": 0,
            "goals_updated": 0,
            "beliefs_updated": 0,
            "relationships_updated": 0,
            "environment_updated": 0,
            "contradictions_detected": 0,
        }

        if not self.enabled:
            return counts

        # Get extractor
        try:
            from aura.consciousness.state_extractor import get_state_extractor
            extractor = get_state_extractor()
        except Exception as e:
            logger.debug(f"[WorldModel] Failed to get StateExtractor: {e}")
            return counts

        if extractor is None:
            return counts

        # Check if extraction should run
        if not extractor.should_extract(messages):
            return counts

        # Run extraction
        state_summary = self.get_context_summary()
        try:
            extraction = extractor.extract(messages, state_summary)
        except Exception as e:
            logger.debug(f"[WorldModel] Extraction failed: {e}")
            return counts

        if not extraction or not isinstance(extraction, dict):
            return counts

        # Apply each entity type
        for proj_data in extraction.get("projects", []):
            try:
                counts["projects_updated"] += self._apply_project(proj_data, conversation_id)
            except Exception as e:
                logger.debug(f"[WorldModel] Apply project failed: {e}")

        for goal_data in extraction.get("goals", []):
            try:
                counts["goals_updated"] += self._apply_goal(goal_data, conversation_id)
            except Exception as e:
                logger.debug(f"[WorldModel] Apply goal failed: {e}")

        for belief_data in extraction.get("beliefs", []):
            try:
                updated, contradictions = self._apply_belief(belief_data, conversation_id)
                counts["beliefs_updated"] += updated
                counts["contradictions_detected"] += contradictions
            except Exception as e:
                logger.debug(f"[WorldModel] Apply belief failed: {e}")

        for person_data in extraction.get("people_mentioned", []):
            try:
                counts["relationships_updated"] += self._apply_relationship(person_data, conversation_id)
            except Exception as e:
                logger.debug(f"[WorldModel] Apply relationship failed: {e}")

        for env_data in extraction.get("environment_changes", []):
            try:
                counts["environment_updated"] += self._apply_environment(env_data, conversation_id)
            except Exception as e:
                logger.debug(f"[WorldModel] Apply environment failed: {e}")

        logger.info(f"[WorldModel] Extraction applied: {counts}")
        return counts

    def _apply_project(self, data: Dict, conv_id: Optional[str]) -> int:
        """Apply a project extraction. Create or update. Returns 1 if applied."""
        name = data.get("name", "").strip()
        if not name:
            return 0

        action = data.get("action", "mention")
        existing = self._find_project_by_name(name)

        if existing:
            # Update existing project
            update_fields = {}
            status_change = data.get("status_change")
            if status_change:
                update_fields["status"] = status_change

            techs = data.get("technologies_mentioned", [])
            if techs and isinstance(techs, list):
                merged = list(set(existing.technologies + techs))
                update_fields["technologies"] = merged

            desc = data.get("progress_notes", "")
            if desc and existing.description:
                update_fields["description"] = f"{existing.description}; {desc}"
            elif desc:
                update_fields["description"] = desc

            self.update_project(existing.id, conversation_id=conv_id, **update_fields)

            # Handle blockers
            for blocker_desc in data.get("new_blockers", []):
                if blocker_desc:
                    self.add_blocker(existing.id, blocker_desc, conversation_id=conv_id)

            for resolved_desc in data.get("resolved_blockers", []):
                if resolved_desc:
                    self._resolve_blocker_by_description(existing.id, resolved_desc, conv_id)

            return 1

        elif action == "new":
            # Create new project
            techs = data.get("technologies_mentioned", [])
            desc = data.get("progress_notes", "")
            proj = self.add_project(
                name=name,
                description=desc,
                technologies=techs if isinstance(techs, list) else [],
                conversation_id=conv_id,
            )

            # Add any blockers
            for blocker_desc in data.get("new_blockers", []):
                if blocker_desc:
                    self.add_blocker(proj.id, blocker_desc, conversation_id=conv_id)

            return 1

        return 0

    def _apply_goal(self, data: Dict, conv_id: Optional[str]) -> int:
        """Apply a goal extraction. Create or update. Returns 1 if applied."""
        desc = data.get("description", "").strip()
        if not desc:
            return 0

        action = data.get("action", "new")
        horizon_str = data.get("horizon", "short_term")

        # Try to find existing similar goal
        existing = self._find_similar_goal(desc)

        if existing and action in ("update", "achieved"):
            update_fields = {}

            if action == "achieved":
                update_fields["status"] = "achieved"
                update_fields["progress"] = 1.0
            else:
                delta = data.get("progress_delta", 0.0)
                if isinstance(delta, (int, float)) and delta > 0:
                    update_fields["progress"] = min(1.0, existing.progress + delta)

            evidence_text = data.get("evidence", "")
            if evidence_text:
                new_evidence = list(existing.evidence) + [evidence_text]
                update_fields["evidence"] = new_evidence

            self.update_goal(existing.id, conversation_id=conv_id, **update_fields)
            return 1

        elif action == "new" and not existing:
            try:
                horizon = GoalHorizon(horizon_str)
            except ValueError:
                horizon = GoalHorizon.SHORT_TERM

            evidence_text = data.get("evidence", "")
            self.add_goal(
                description=desc,
                horizon=horizon,
                conversation_id=conv_id,
            )
            return 1

        elif existing:
            # "mention" of existing goal — just reinforce
            evidence_text = data.get("evidence", "")
            if evidence_text:
                update_fields = {"evidence": list(existing.evidence) + [evidence_text]}
                self.update_goal(existing.id, conversation_id=conv_id, **update_fields)
            return 1

        return 0

    def _apply_belief(self, data: Dict, conv_id: Optional[str]) -> Tuple[int, int]:
        """
        Apply a belief extraction. Create, reinforce, or supersede.

        Returns (beliefs_updated, contradictions_detected).
        """
        statement = data.get("statement", "").strip()
        if not statement:
            return 0, 0

        category_str = data.get("category", "user_intent")
        try:
            category = BeliefCategory(category_str)
        except ValueError:
            category = BeliefCategory.USER_INTENT

        confidence = data.get("confidence", 0.7)
        if not isinstance(confidence, (int, float)):
            confidence = 0.7
        confidence = min(max(float(confidence), 0.0), 1.0)

        contradicts = data.get("contradicts_existing")
        contradictions = 0

        # Check for similar existing belief
        existing = self._find_similar_belief(statement, category)

        if existing and contradicts:
            # LLM flagged a contradiction — supersede old belief
            new_belief = self.supersede_belief(
                old_id=existing.id,
                new_statement=statement,
                new_category=category,
                new_confidence=confidence,
                new_evidence=[f"Contradicts: {contradicts}"],
                conversation_id=conv_id,
            )
            if new_belief:
                self.add_contradiction(
                    belief_a_id=existing.id,
                    belief_b_id=new_belief.id,
                    description=str(contradicts),
                    conversation_id=conv_id,
                )
                contradictions = 1
            return 1, contradictions

        elif existing:
            # Check if new statement negates the existing one
            negation_signals = [
                "not ", "no longer", "switched from", "instead of",
                "stopped", "don't", "doesn't", "won't", "never",
            ]
            has_negation = any(sig in statement.lower() for sig in negation_signals)

            if has_negation:
                # Negation detected — create as new belief and run contradiction check
                new_belief = self.add_belief(
                    statement=statement,
                    category=category,
                    confidence=confidence,
                    evidence=[],
                    conversation_id=conv_id,
                )
                contradictions = self._check_local_contradictions(new_belief, conv_id)
                return 1, contradictions
            else:
                # Reinforce existing belief
                self.reinforce_belief(
                    existing.id,
                    evidence=statement,
                    confidence_boost=0.05,
                )
                return 1, 0

        else:
            # New belief
            new_belief = self.add_belief(
                statement=statement,
                category=category,
                confidence=confidence,
                evidence=[],
                conversation_id=conv_id,
            )

            # Local contradiction check
            contradictions = self._check_local_contradictions(new_belief, conv_id)
            return 1, contradictions

    def _apply_relationship(self, data: Dict, conv_id: Optional[str]) -> int:
        """Apply a relationship extraction. Create or update. Returns 1 if applied."""
        name = data.get("name", "").strip()
        if not name:
            return 0

        # Try to find existing relationship
        existing = self.get_relationship(name)

        if existing:
            role = data.get("role")
            context = data.get("context")
            sentiment = data.get("sentiment")
            self.update_relationship(
                existing.id,
                role=role,
                context_note=context,
                sentiment=sentiment,
                conversation_id=conv_id,
            )
            return 1
        else:
            self.add_relationship(
                name=name,
                role=data.get("role", ""),
                context=data.get("context"),
                sentiment=data.get("sentiment", "neutral"),
                conversation_id=conv_id,
            )
            return 1

    def _apply_environment(self, data: Dict, conv_id: Optional[str]) -> int:
        """Apply an environment extraction. Returns 1 if applied."""
        key = data.get("key", "").strip()
        if not key:
            return 0

        category = data.get("category", "preference")
        value = data.get("value", "")
        if not value:
            return 0

        self.set_environment(
            key=key,
            category=category,
            value=value,
            conversation_id=conv_id,
        )
        return 1

    # ----------------------------------------------------------------
    # Extraction helpers
    # ----------------------------------------------------------------

    def _find_project_by_name(self, name: str) -> Optional[Project]:
        """Find a project by name: case-insensitive exact, then substring match."""
        name_lower = name.lower()

        # Exact match (case-insensitive)
        for proj in self._projects.values():
            if proj.name.lower() == name_lower:
                return proj

        # Substring match
        for proj in self._projects.values():
            if name_lower in proj.name.lower() or proj.name.lower() in name_lower:
                return proj

        return None

    def _find_similar_goal(self, description: str) -> Optional[Goal]:
        """Find a similar active goal using word overlap (Jaccard > 0.4)."""
        desc_words = set(description.lower().split())
        if not desc_words:
            return None

        best_match = None
        best_score = 0.0

        for goal in self._goals.values():
            if goal.status != "active":
                continue
            goal_words = set(goal.description.lower().split())
            if not goal_words:
                continue

            intersection = desc_words & goal_words
            union = desc_words | goal_words
            jaccard = len(intersection) / len(union) if union else 0.0

            if jaccard > 0.4 and jaccard > best_score:
                best_score = jaccard
                best_match = goal

        return best_match

    def _find_similar_belief(
        self, statement: str, category: BeliefCategory
    ) -> Optional[Belief]:
        """Find a similar current belief using Jaccard word overlap > 0.4."""
        stmt_words = set(statement.lower().split())
        if not stmt_words:
            return None

        best_match = None
        best_score = 0.0

        for belief in self._beliefs.values():
            # Prefer same-category matches
            belief_words = set(belief.statement.lower().split())
            if not belief_words:
                continue

            intersection = stmt_words & belief_words
            union = stmt_words | belief_words
            jaccard = len(intersection) / len(union) if union else 0.0

            # Lower threshold for same-category
            threshold = 0.3 if belief.category == category else 0.5
            if jaccard > threshold and jaccard > best_score:
                best_score = jaccard
                best_match = belief

        return best_match

    def _check_local_contradictions(
        self, new_belief: Belief, conv_id: Optional[str]
    ) -> int:
        """
        Scan same-category beliefs for local contradiction signals.

        Looks for negation patterns: "not", "no longer", "switched from",
        "instead of", "stopped", "don't".

        Returns count of contradictions detected.
        """
        negation_signals = [
            "not ", "no longer", "switched from", "instead of",
            "stopped", "don't", "doesn't", "won't", "never",
        ]

        new_lower = new_belief.statement.lower()
        has_negation = any(sig in new_lower for sig in negation_signals)
        if not has_negation:
            return 0

        contradictions = 0
        new_words = set(new_lower.split()) - {"not", "no", "don't", "doesn't", "won't", "never", "longer"}

        for belief in list(self._beliefs.values()):
            if belief.id == new_belief.id:
                continue
            if belief.category != new_belief.category:
                continue

            belief_words = set(belief.statement.lower().split())
            overlap = new_words & belief_words

            # Need significant word overlap (excluding negation words) to flag
            if len(overlap) >= 2:
                self.add_contradiction(
                    belief_a_id=belief.id,
                    belief_b_id=new_belief.id,
                    description=f"Potential negation: '{new_belief.statement}' vs '{belief.statement}'",
                    conversation_id=conv_id,
                )
                contradictions += 1

        return contradictions

    def _resolve_blocker_by_description(
        self, project_id: str, description: str, conv_id: Optional[str]
    ) -> bool:
        """Match a blocker by description substring and resolve it."""
        desc_lower = description.lower()
        blockers = self.get_project_blockers(project_id)

        for blocker in blockers:
            if blocker.get("status") != "ongoing":
                continue
            blocker_desc = (blocker.get("description") or "").lower()
            if desc_lower in blocker_desc or blocker_desc in desc_lower:
                return self.resolve_blocker(
                    blocker["id"],
                    resolution=description,
                    conversation_id=conv_id,
                )

        return False

    def run_maintenance(self) -> Dict[str, int]:
        """
        Run all maintenance tasks.

        1. Decay belief confidence
        2. Update project health
        3. Recompute priorities
        4. Clean up old state_changes (keep last STATE_CHANGE_RETENTION)
        """
        results = {}

        results["beliefs_decayed"] = self.decay_beliefs()
        results["health_changes"] = self.update_project_health()

        # Recompute priorities
        priority_updates = 0
        with self._lock:
            conn = self._connect()
            try:
                for project in self._projects.values():
                    new_priority = self.compute_project_priority(project)
                    if abs(new_priority - project.priority) > 0.01:
                        project.priority = new_priority
                        conn.execute(
                            "UPDATE projects SET priority=? WHERE id=?",
                            (new_priority, project.id),
                        )
                        priority_updates += 1
                conn.commit()
            finally:
                conn.close()
        results["priority_updates"] = priority_updates

        # Clean old state changes
        conn = self._connect()
        try:
            count_row = conn.execute(
                "SELECT COUNT(*) as cnt FROM state_changes"
            ).fetchone()
            total = count_row["cnt"] if count_row else 0
            if total > self.STATE_CHANGE_RETENTION:
                excess = total - self.STATE_CHANGE_RETENTION
                conn.execute(
                    """DELETE FROM state_changes WHERE id IN (
                       SELECT id FROM state_changes
                       ORDER BY timestamp ASC LIMIT ?)""",
                    (excess,),
                )
                conn.commit()
                results["state_changes_cleaned"] = excess
            else:
                results["state_changes_cleaned"] = 0
        finally:
            conn.close()

        with self._lock:
            self._update_snapshot()
        logger.info(f"[WorldModel] Maintenance complete: {results}")
        return results


# ============================================================================
# Singleton
# ============================================================================

_world_model: Optional[WorldModel] = None
_singleton_lock = threading.Lock()


def get_world_model() -> WorldModel:
    """Get or create the singleton WorldModel (double-checked locking)."""
    global _world_model
    if _world_model is None:
        with _singleton_lock:
            if _world_model is None:
                from aura.config import Config
                _world_model = WorldModel(
                    db_path=getattr(Config, "WORLD_MODEL_DB_PATH", "") or None,
                    enabled=getattr(Config, "WORLD_MODEL_ENABLED", True),
                )
    return _world_model
