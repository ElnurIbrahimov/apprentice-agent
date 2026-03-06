"""Tests for WorldModel — ADV-02 Phase 1: Persistent World Model Foundation."""

import json
import os
import shutil
import tempfile
import threading
import time
from pathlib import Path

import pytest

from aura.consciousness.world_model import (
    Belief,
    BeliefCategory,
    ChangeType,
    Contradiction,
    Goal,
    GoalHorizon,
    Project,
    ProjectHealth,
    ProjectStatus,
    Relationship,
    StateChange,
    WorldModel,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test databases and snapshots."""
    tmpdir = tempfile.mkdtemp(prefix="aura_test_wm_")
    yield Path(tmpdir)
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def wm(temp_dir):
    """Create a WorldModel with test database."""
    db_path = str(temp_dir / "test_world_model.db")
    snapshot_path = str(temp_dir / "test_world_state.json")
    return WorldModel(db_path=db_path, snapshot_path=snapshot_path, enabled=True)


@pytest.fixture
def disabled_wm(temp_dir):
    """Create a disabled WorldModel."""
    db_path = str(temp_dir / "disabled_world_model.db")
    snapshot_path = str(temp_dir / "disabled_world_state.json")
    return WorldModel(db_path=db_path, snapshot_path=snapshot_path, enabled=False)


# ============================================================================
# TestWorldModelInit
# ============================================================================


class TestWorldModelInit:
    """Tests for WorldModel initialization."""

    def test_db_created(self, temp_dir):
        """DB file is created on init."""
        db_path = str(temp_dir / "init_test.db")
        WorldModel(db_path=db_path, snapshot_path=str(temp_dir / "s.json"))
        assert Path(db_path).exists()

    def test_empty_on_init(self, wm):
        """Freshly created world model has no data."""
        assert wm.get_all_projects() == []
        assert wm.get_active_goals() == []
        assert wm.get_current_beliefs() == []
        assert wm.get_unresolved_contradictions() == []

    def test_disabled_skips_db(self, temp_dir):
        """Disabled world model does not create DB file."""
        db_path = str(temp_dir / "should_not_exist.db")
        wm = WorldModel(db_path=db_path, snapshot_path=str(temp_dir / "s.json"), enabled=False)
        assert not Path(db_path).exists()
        assert wm.get_context_summary() == ""


# ============================================================================
# TestProjectCRUD
# ============================================================================


class TestProjectCRUD:
    """Tests for project create, read, update operations."""

    def test_add_project(self, wm):
        """Adding a project returns Project with generated ID."""
        proj = wm.add_project("TestApp", description="A test app")
        assert proj.id.startswith("proj_")
        assert proj.name == "TestApp"
        assert proj.status == ProjectStatus.ACTIVE
        assert proj.description == "A test app"
        assert proj.mention_count == 1

    def test_get_project(self, wm):
        """Can retrieve a project by ID."""
        proj = wm.add_project("TestApp")
        retrieved = wm.get_project(proj.id)
        assert retrieved is not None
        assert retrieved.name == "TestApp"

    def test_get_project_not_found(self, wm):
        """Getting non-existent project returns None."""
        assert wm.get_project("nonexistent") is None

    def test_update_project(self, wm):
        """Updating a project changes fields and bumps mention_count."""
        proj = wm.add_project("TestApp")
        initial_count = proj.mention_count

        updated = wm.update_project(proj.id, description="Updated desc")
        assert updated is not None
        assert updated.description == "Updated desc"
        assert updated.mention_count == initial_count + 1

    def test_update_project_not_found(self, wm):
        """Updating non-existent project returns None."""
        assert wm.update_project("nonexistent", name="X") is None

    def test_get_projects_by_status(self, wm):
        """Can filter projects by status."""
        wm.add_project("Active1")
        wm.add_project("Active2")
        p3 = wm.add_project("Paused1", status=ProjectStatus.PAUSED)

        active = wm.get_projects_by_status(ProjectStatus.ACTIVE)
        assert len(active) == 2

        paused = wm.get_projects_by_status(ProjectStatus.PAUSED)
        assert len(paused) == 1
        assert paused[0].name == "Paused1"

    def test_get_all_projects(self, wm):
        """get_all_projects returns all projects regardless of status."""
        wm.add_project("A")
        wm.add_project("B", status=ProjectStatus.COMPLETED)
        assert len(wm.get_all_projects()) == 2

    def test_id_uniqueness(self, wm):
        """Each project gets a unique ID."""
        p1 = wm.add_project("A")
        p2 = wm.add_project("B")
        assert p1.id != p2.id

    def test_technologies_stored(self, wm):
        """Technologies list is persisted."""
        proj = wm.add_project("App", technologies=["python", "react"])
        assert proj.technologies == ["python", "react"]

        retrieved = wm.get_project(proj.id)
        assert retrieved.technologies == ["python", "react"]


# ============================================================================
# TestBlockerCRUD
# ============================================================================


class TestBlockerCRUD:
    """Tests for project blocker operations."""

    def test_add_blocker(self, wm):
        """Can add a blocker to a project."""
        proj = wm.add_project("App")
        bid = wm.add_blocker(proj.id, "VRAM limit", severity="high")
        assert isinstance(bid, int)

        blockers = wm.get_project_blockers(proj.id)
        assert len(blockers) == 1
        assert blockers[0]["description"] == "VRAM limit"
        assert blockers[0]["severity"] == "high"
        assert blockers[0]["status"] == "ongoing"

    def test_resolve_blocker(self, wm):
        """Resolving a blocker sets resolved_at and status."""
        proj = wm.add_project("App")
        bid = wm.add_blocker(proj.id, "API rate limit")

        result = wm.resolve_blocker(bid, resolution="Upgraded plan")
        assert result is True

        blockers = wm.get_project_blockers(proj.id)
        assert blockers[0]["status"] == "resolved"
        assert blockers[0]["resolution"] == "Upgraded plan"

    def test_get_project_blockers_empty(self, wm):
        """Project with no blockers returns empty list."""
        proj = wm.add_project("App")
        assert wm.get_project_blockers(proj.id) == []


# ============================================================================
# TestGoalCRUD
# ============================================================================


class TestGoalCRUD:
    """Tests for goal create, read, update operations."""

    def test_add_goal(self, wm):
        """Adding a goal returns Goal with generated ID."""
        goal = wm.add_goal("Ship MVP", horizon=GoalHorizon.MEDIUM_TERM)
        assert goal.id.startswith("goal_")
        assert goal.description == "Ship MVP"
        assert goal.horizon == GoalHorizon.MEDIUM_TERM
        assert goal.progress == 0.0
        assert goal.status == "active"

    def test_update_goal_progress(self, wm):
        """Can update goal progress."""
        goal = wm.add_goal("Ship MVP")
        updated = wm.update_goal(goal.id, progress=0.5)
        assert updated is not None
        assert updated.progress == 0.5

    def test_get_active_goals_by_horizon(self, wm):
        """Can filter active goals by horizon."""
        wm.add_goal("Quick fix", horizon=GoalHorizon.SHORT_TERM)
        wm.add_goal("Ship MVP", horizon=GoalHorizon.MEDIUM_TERM)
        wm.add_goal("IPO", horizon=GoalHorizon.LONG_TERM)

        short = wm.get_active_goals(horizon=GoalHorizon.SHORT_TERM)
        assert len(short) == 1
        assert short[0].description == "Quick fix"

        all_active = wm.get_active_goals()
        assert len(all_active) == 3


# ============================================================================
# TestBeliefCRUD
# ============================================================================


class TestBeliefCRUD:
    """Tests for belief create, reinforce, supersede, query operations."""

    def test_add_belief(self, wm):
        """Adding a belief returns Belief with valid_from set, valid_to None."""
        belief = wm.add_belief(
            "User prefers detailed responses",
            category=BeliefCategory.PREFERENCE,
            confidence=0.8,
            evidence=["asked for more detail twice"],
        )
        assert belief.id.startswith("belief_")
        assert belief.valid_from != ""
        assert belief.valid_to is None
        assert belief.confidence == 0.8
        assert len(belief.evidence) == 1

    def test_reinforce_belief(self, wm):
        """Reinforcing a belief boosts confidence and appends evidence."""
        belief = wm.add_belief("User likes Python", confidence=0.6)
        original_conf = belief.confidence

        reinforced = wm.reinforce_belief(belief.id, "used python again", 0.1)
        assert reinforced is not None
        assert reinforced.confidence == original_conf + 0.1
        assert "used python again" in reinforced.evidence

    def test_reinforce_caps_at_1(self, wm):
        """Reinforcing cannot push confidence above 1.0."""
        belief = wm.add_belief("High conf", confidence=0.95)
        reinforced = wm.reinforce_belief(belief.id, confidence_boost=0.2)
        assert reinforced.confidence == 1.0

    def test_supersede_belief(self, wm):
        """Superseding marks old belief and creates new one."""
        old = wm.add_belief("User prefers React", category=BeliefCategory.PREFERENCE)

        new = wm.supersede_belief(
            old.id,
            "User prefers Svelte",
            new_confidence=0.85,
            new_evidence=["mentioned switching to Svelte"],
        )
        assert new is not None
        assert new.statement == "User prefers Svelte"

        # Old belief should not be in current beliefs
        current = wm.get_current_beliefs()
        current_ids = [b.id for b in current]
        assert old.id not in current_ids
        assert new.id in current_ids

    def test_get_current_excludes_superseded(self, wm):
        """get_current_beliefs only returns valid_to=None beliefs."""
        b1 = wm.add_belief("Belief A")
        b2 = wm.add_belief("Belief B")
        wm.supersede_belief(b1.id, "Belief A revised")

        current = wm.get_current_beliefs()
        statements = [b.statement for b in current]
        assert "Belief A" not in statements
        assert "Belief A revised" in statements
        assert "Belief B" in statements

    def test_filter_by_category(self, wm):
        """Can filter beliefs by category."""
        wm.add_belief("Intent belief", category=BeliefCategory.USER_INTENT)
        wm.add_belief("Preference belief", category=BeliefCategory.PREFERENCE)
        wm.add_belief("Another intent", category=BeliefCategory.USER_INTENT)

        intents = wm.get_current_beliefs(category=BeliefCategory.USER_INTENT)
        assert len(intents) == 2
        assert all(b.category == BeliefCategory.USER_INTENT for b in intents)


# ============================================================================
# TestRelationshipCRUD
# ============================================================================


class TestRelationshipCRUD:
    """Tests for relationship operations."""

    def test_add_relationship(self, wm):
        """Adding a relationship returns Relationship with ID."""
        rel = wm.add_relationship("Alice", role="collaborator", context="works on design")
        assert rel.id.startswith("rel_")
        assert rel.name == "Alice"
        assert rel.role == "collaborator"
        assert rel.context_notes == ["works on design"]

    def test_update_appends_context(self, wm):
        """Updating a relationship appends context notes and bumps count."""
        rel = wm.add_relationship("Bob", role="manager")
        initial_count = rel.mention_count

        updated = wm.update_relationship(rel.id, context_note="discussed project timeline")
        assert updated is not None
        assert updated.mention_count == initial_count + 1
        assert "discussed project timeline" in updated.context_notes

    def test_case_insensitive_lookup(self, wm):
        """get_relationship does case-insensitive name matching."""
        wm.add_relationship("Alice", role="collaborator")

        found = wm.get_relationship("alice")
        assert found is not None
        assert found.name == "Alice"

        found_upper = wm.get_relationship("ALICE")
        assert found_upper is not None


# ============================================================================
# TestEnvironmentCRUD
# ============================================================================


class TestEnvironmentCRUD:
    """Tests for environment observation operations."""

    def test_set_environment(self, wm):
        """Can set an environment observation."""
        wm.set_environment("gpu", "hardware", "RTX 4060")
        result = wm.get_environment("gpu")
        assert result is not None
        assert result["value"] == "RTX 4060"
        assert result["category"] == "hardware"
        assert result["observation_count"] == 1

    def test_update_existing(self, wm):
        """Setting same key again increments observation_count."""
        wm.set_environment("gpu", "hardware", "RTX 4060")
        wm.set_environment("gpu", "hardware", "RTX 4060 8GB")

        result = wm.get_environment("gpu")
        assert result["value"] == "RTX 4060 8GB"
        assert result["observation_count"] == 2

    def test_get_by_category(self, wm):
        """Can filter environment entries by category."""
        wm.set_environment("gpu", "hardware", "RTX 4060")
        wm.set_environment("ram", "hardware", "32GB")
        wm.set_environment("ide", "tool", "VSCode")

        hw = wm.get_environment_by_category("hardware")
        assert len(hw) == 2
        assert "gpu" in hw
        assert "ram" in hw

        tools = wm.get_environment_by_category("tool")
        assert len(tools) == 1


# ============================================================================
# TestContradictions
# ============================================================================


class TestContradictions:
    """Tests for contradiction detection and resolution."""

    def test_add_contradiction(self, wm):
        """Can add a contradiction between two beliefs."""
        b1 = wm.add_belief("Deploy to Vercel")
        b2 = wm.add_belief("Self-host on home server")

        c = wm.add_contradiction(b1.id, b2.id, "Conflicting deployment targets")
        assert c.id.startswith("contra_")
        assert c.resolved_at is None

    def test_resolve_contradiction(self, wm):
        """Resolving removes from unresolved list."""
        b1 = wm.add_belief("A")
        b2 = wm.add_belief("B")
        c = wm.add_contradiction(b1.id, b2.id, "Conflict")

        assert len(wm.get_unresolved_contradictions()) == 1

        result = wm.resolve_contradiction(c.id, "resolved_b", "B is correct")
        assert result is True
        assert len(wm.get_unresolved_contradictions()) == 0

    def test_get_unresolved(self, wm):
        """Only unresolved contradictions are returned."""
        b1 = wm.add_belief("X")
        b2 = wm.add_belief("Y")
        b3 = wm.add_belief("Z")

        c1 = wm.add_contradiction(b1.id, b2.id, "C1")
        c2 = wm.add_contradiction(b2.id, b3.id, "C2")
        wm.resolve_contradiction(c1.id, "merged")

        unresolved = wm.get_unresolved_contradictions()
        assert len(unresolved) == 1
        assert unresolved[0].id == c2.id


# ============================================================================
# TestStateChangeLogging
# ============================================================================


class TestStateChangeLogging:
    """Tests for state change audit logging."""

    def test_crud_logs_changes(self, wm):
        """CRUD operations generate state change records."""
        wm.add_project("TestApp")
        wm.add_belief("Some belief")

        changes = wm.get_recent_changes(limit=10)
        assert len(changes) >= 2

        change_types = [c.change_type for c in changes]
        assert ChangeType.PROJECT_UPDATE in change_types
        assert ChangeType.BELIEF_FORMED in change_types

    def test_get_recent_respects_limit(self, wm):
        """get_recent_changes respects the limit parameter."""
        for i in range(5):
            wm.add_belief(f"Belief {i}")

        changes = wm.get_recent_changes(limit=3)
        assert len(changes) == 3


# ============================================================================
# TestContextSummary
# ============================================================================


class TestContextSummary:
    """Tests for context summary generation."""

    def test_empty_returns_empty(self, wm):
        """Empty world model returns empty string."""
        assert wm.get_context_summary() == ""

    def test_disabled_returns_empty(self, disabled_wm):
        """Disabled world model returns empty string."""
        assert disabled_wm.get_context_summary() == ""

    def test_includes_projects(self, wm):
        """Context summary includes active projects."""
        wm.add_project("AURA")
        summary = wm.get_context_summary()
        assert "AURA" in summary
        assert "[World State]" in summary

    def test_includes_goals_and_beliefs(self, wm):
        """Context summary includes goals and beliefs."""
        wm.add_project("App")
        wm.add_goal("Ship MVP", horizon=GoalHorizon.MEDIUM_TERM)
        wm.add_belief("User prefers Python")

        summary = wm.get_context_summary()
        assert "Ship MVP" in summary
        assert "User prefers Python" in summary

    def test_header_format(self, wm):
        """Summary starts with [World State] header."""
        wm.add_project("App")
        summary = wm.get_context_summary()
        assert summary.startswith("[World State]")


# ============================================================================
# TestSnapshotPersistence
# ============================================================================


class TestSnapshotPersistence:
    """Tests for JSON snapshot file persistence."""

    def test_creates_file(self, wm, temp_dir):
        """Adding data creates a snapshot JSON file."""
        wm.add_project("TestApp")
        snapshot_path = Path(wm._snapshot_path)
        assert snapshot_path.exists()

    def test_load_returns_dict(self, wm):
        """_load_snapshot returns a dict after data is written."""
        wm.add_project("TestApp")
        data = wm._load_snapshot()
        assert isinstance(data, dict)
        assert "projects" in data
        assert len(data["projects"]) == 1

    def test_survives_restart(self, temp_dir):
        """Snapshot file is readable after creating a new WorldModel instance."""
        db_path = str(temp_dir / "persist.db")
        snap_path = str(temp_dir / "persist.json")

        wm1 = WorldModel(db_path=db_path, snapshot_path=snap_path)
        wm1.add_project("Persistent Project")
        del wm1

        wm2 = WorldModel(db_path=db_path, snapshot_path=snap_path)
        data = wm2._load_snapshot()
        assert data is not None
        assert any(p["name"] == "Persistent Project" for p in data["projects"])


# ============================================================================
# TestMaintenance
# ============================================================================


class TestMaintenance:
    """Tests for maintenance routines."""

    def test_decay_beliefs(self, wm):
        """Beliefs with old last_reinforced get confidence decayed."""
        belief = wm.add_belief("Old belief", confidence=0.9)
        # Manually set last_reinforced to 30 days ago
        old_time = (
            __import__("datetime").datetime.now(__import__("datetime").timezone.utc)
            - __import__("datetime").timedelta(days=30)
        ).isoformat()
        belief.last_reinforced = old_time

        import sqlite3
        conn = sqlite3.connect(wm._db_path)
        conn.execute(
            "UPDATE beliefs SET last_reinforced=? WHERE id=?",
            (old_time, belief.id),
        )
        conn.commit()
        conn.close()

        original_conf = belief.confidence
        decayed = wm.decay_beliefs()
        assert decayed >= 1
        assert belief.confidence < original_conf

    def test_update_project_health(self, wm):
        """Projects with old last_activity get health downgraded."""
        proj = wm.add_project("StaleApp")

        # Manually set last_activity to 10 days ago
        old_time = (
            __import__("datetime").datetime.now(__import__("datetime").timezone.utc)
            - __import__("datetime").timedelta(days=10)
        ).isoformat()
        proj.last_activity = old_time

        import sqlite3
        conn = sqlite3.connect(wm._db_path)
        conn.execute(
            "UPDATE projects SET last_activity=? WHERE id=?",
            (old_time, proj.id),
        )
        conn.commit()
        conn.close()

        changed = wm.update_project_health()
        assert changed >= 1
        assert proj.health == ProjectHealth.RED

    def test_compute_project_priority(self, wm):
        """Priority computation returns value between 0 and 1."""
        proj = wm.add_project("App")
        priority = wm.compute_project_priority(proj)
        assert 0.0 <= priority <= 1.0

    def test_run_maintenance(self, wm):
        """run_maintenance returns a results dict with expected keys."""
        wm.add_project("App")
        wm.add_belief("Something", confidence=0.8)

        results = wm.run_maintenance()
        assert "beliefs_decayed" in results
        assert "health_changes" in results
        assert "priority_updates" in results
        assert "state_changes_cleaned" in results


# ============================================================================
# TestPersistence
# ============================================================================


class TestPersistence:
    """Tests for data persistence across WorldModel instances."""

    def test_data_survives_restart(self, temp_dir):
        """Data written to SQLite survives instance destruction and recreation."""
        db_path = str(temp_dir / "restart.db")
        snap_path = str(temp_dir / "restart.json")

        # First instance: add data
        wm1 = WorldModel(db_path=db_path, snapshot_path=snap_path)
        proj = wm1.add_project("Survivor", technologies=["python"])
        goal = wm1.add_goal("Persist data", horizon=GoalHorizon.SHORT_TERM)
        belief = wm1.add_belief("Test belief", category=BeliefCategory.PREFERENCE)
        rel = wm1.add_relationship("TestPerson", role="tester")
        wm1.set_environment("os", "tool", "Windows")

        proj_id = proj.id
        goal_id = goal.id
        belief_id = belief.id
        del wm1

        # Second instance: verify data
        wm2 = WorldModel(db_path=db_path, snapshot_path=snap_path)

        assert wm2.get_project(proj_id) is not None
        assert wm2.get_project(proj_id).name == "Survivor"
        assert wm2.get_project(proj_id).technologies == ["python"]

        active_goals = wm2.get_active_goals()
        assert any(g.id == goal_id for g in active_goals)

        beliefs = wm2.get_current_beliefs()
        assert any(b.id == belief_id for b in beliefs)

        assert wm2.get_relationship("TestPerson") is not None
        assert wm2.get_environment("os") is not None


# ============================================================================
# TestThreadSafety
# ============================================================================


class TestThreadSafety:
    """Tests for thread safety of concurrent operations."""

    def test_concurrent_crud(self, wm):
        """Concurrent CRUD operations don't corrupt state."""
        errors = []

        def add_projects(prefix, count):
            try:
                for i in range(count):
                    wm.add_project(f"{prefix}_{i}")
            except Exception as e:
                errors.append(str(e))

        def add_beliefs(prefix, count):
            try:
                for i in range(count):
                    wm.add_belief(f"{prefix}_belief_{i}")
            except Exception as e:
                errors.append(str(e))

        threads = [
            threading.Thread(target=add_projects, args=("t1", 10)),
            threading.Thread(target=add_projects, args=("t2", 10)),
            threading.Thread(target=add_beliefs, args=("t1", 10)),
            threading.Thread(target=add_beliefs, args=("t2", 10)),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert errors == [], f"Thread errors: {errors}"
        assert len(wm.get_all_projects()) == 20
        assert len(wm.get_current_beliefs()) == 20


# ============================================================================
# TestInsightEngagementRates — ADV-02 Phase 5
# ============================================================================


class TestInsightEngagementRates:
    """Tests for get_insight_engagement_rates()."""

    def test_empty_returns_empty(self, wm):
        """No insights = empty engagement rates."""
        result = wm.get_insight_engagement_rates(days=30)
        assert result == {}

    def test_mixed_feedback(self, wm):
        """Mixed engaged/dismissed insights compute correct rates."""
        from aura.consciousness.proactive_awareness import (
            InsightType,
            ProactiveInsight,
        )
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc).isoformat()
        # 3 staleness insights: 2 engaged, 1 dismissed
        for i in range(3):
            insight = ProactiveInsight(
                id=f"eng_{i}",
                insight_type=InsightType.STALENESS_ALERT,
                title=f"Stale {i}",
                description="Desc",
                urgency=0.7,
                confidence=0.7,
                generated_at=now,
                related_entity_type="project",
                related_entity_id=f"proj_{i}",
            )
            wm.store_insight(insight)

        wm.update_insight_feedback("eng_0", "engaged")
        wm.update_insight_feedback("eng_1", "engaged")
        wm.update_insight_feedback("eng_2", "dismissed")

        rates = wm.get_insight_engagement_rates(days=30)
        assert "staleness_alert" in rates
        assert rates["staleness_alert"]["total"] == 3
        assert rates["staleness_alert"]["engaged"] == 2
        assert rates["staleness_alert"]["dismissed"] == 1
        # engagement_rate = 2 / max(1, 2 + 1) = 2/3
        assert abs(rates["staleness_alert"]["engagement_rate"] - 2 / 3) < 0.01

    def test_single_type(self, wm):
        """Single insight type with no feedback returns 0 engagement rate."""
        from aura.consciousness.proactive_awareness import (
            InsightType,
            ProactiveInsight,
        )
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc).isoformat()
        insight = ProactiveInsight(
            id="single_1",
            insight_type=InsightType.DEADLINE_APPROACHING,
            title="Due soon",
            description="Desc",
            urgency=0.8,
            confidence=0.7,
            generated_at=now,
        )
        wm.store_insight(insight)

        rates = wm.get_insight_engagement_rates(days=30)
        assert "deadline_approaching" in rates
        assert rates["deadline_approaching"]["total"] == 1
        assert rates["deadline_approaching"]["engagement_rate"] == 0.0


# ============================================================================
# TestAdaptiveHalfLife — ADV-02 Phase 5
# ============================================================================


class TestAdaptiveHalfLife:
    """Tests for compute_adaptive_half_life()."""

    def test_insufficient_data_returns_default(self, wm):
        """With fewer than 5 reinforced beliefs, return default half-life."""
        # Add 3 beliefs (not reinforced, so first_formed == last_reinforced)
        for i in range(3):
            wm.add_belief(f"Belief {i}")

        result = wm.compute_adaptive_half_life()
        assert result == wm.BELIEF_DECAY_HALF_LIFE  # 336

    def test_sufficient_data_computes_value(self, wm):
        """With 5+ reinforced beliefs, compute adaptive value from median interval."""
        import sqlite3
        from datetime import datetime, timedelta, timezone

        now = datetime.now(timezone.utc)

        # Add 6 beliefs with varying reinforcement intervals
        intervals_hours = [48, 100, 200, 300, 400, 500]
        for i, hours in enumerate(intervals_hours):
            belief = wm.add_belief(f"Adaptive belief {i}")
            formed = (now - timedelta(hours=hours)).isoformat()
            reinforced = now.isoformat()
            # Manually set first_formed and last_reinforced so they differ
            conn = sqlite3.connect(wm._db_path)
            conn.execute(
                "UPDATE beliefs SET first_formed=?, last_reinforced=? WHERE id=?",
                (formed, reinforced, belief.id),
            )
            conn.commit()
            conn.close()
            wm._beliefs[belief.id].first_formed = formed
            wm._beliefs[belief.id].last_reinforced = reinforced

        # Clear cache to force recomputation
        if hasattr(wm, "_half_life_computed_at"):
            wm._half_life_computed_at = 0

        result = wm.compute_adaptive_half_life()

        # Sorted intervals: [48, 100, 200, 300, 400, 500]
        # Median (index 3 of 6): 300
        # Adaptive = max(168, min(672, 300 * 2)) = max(168, min(672, 600)) = 600
        assert result != wm.BELIEF_DECAY_HALF_LIFE  # Not default
        assert 168 <= result <= 672  # Within clamp range
