"""Tests for ProactiveAwarenessEngine — ADV-02 Phase 3 + Phase 4."""

import shutil
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aura.consciousness.world_model import (
    ChangeType,
    GoalHorizon,
    ProjectStatus,
    WorldModel,
)
from aura.consciousness.proactive_awareness import (
    InsightType,
    ProactiveAwarenessEngine,
    ProactiveInsight,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test databases."""
    tmpdir = tempfile.mkdtemp(prefix="aura_test_pa_")
    yield Path(tmpdir)
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def wm(temp_dir):
    """Create a WorldModel with test database."""
    db_path = str(temp_dir / "test_pa.db")
    snapshot_path = str(temp_dir / "test_pa_state.json")
    return WorldModel(db_path=db_path, snapshot_path=snapshot_path, enabled=True)


@pytest.fixture
def engine(wm):
    """Create a ProactiveAwarenessEngine with the test WorldModel."""
    return ProactiveAwarenessEngine(world_model=wm, enabled=True)


@pytest.fixture
def disabled_engine(wm):
    """Create a disabled ProactiveAwarenessEngine."""
    return ProactiveAwarenessEngine(world_model=wm, enabled=False)


def _past_iso(days: int = 0, hours: int = 0) -> str:
    """Return ISO timestamp N days/hours in the past."""
    dt = datetime.now(timezone.utc) - timedelta(days=days, hours=hours)
    return dt.isoformat()


def _future_iso(days: int = 0) -> str:
    """Return ISO timestamp N days in the future."""
    dt = datetime.now(timezone.utc) + timedelta(days=days)
    return dt.isoformat()


# ============================================================================
# TestInsightDataclass
# ============================================================================


class TestInsightDataclass:
    def test_delivery_score(self):
        insight = ProactiveInsight(
            id="test1",
            insight_type=InsightType.STALENESS_ALERT,
            title="Test",
            description="Test description",
            urgency=0.8,
            confidence=0.6,
        )
        # delivery_score = urgency * 0.6 + confidence * 0.4
        expected = 0.8 * 0.6 + 0.6 * 0.4
        assert abs(insight.delivery_score - expected) < 0.001

    def test_defaults(self):
        insight = ProactiveInsight(
            id="test2",
            insight_type=InsightType.DEADLINE_APPROACHING,
            title="Test",
            description="Desc",
        )
        assert insight.urgency == 0.5
        assert insight.confidence == 0.5
        assert insight.delivered_at is None
        assert insight.dismissed_at is None
        assert insight.acted_on_at is None
        assert insight.related_entity_type is None
        assert insight.related_entity_id is None


# ============================================================================
# TestCheckStaleness
# ============================================================================


class TestCheckStaleness:
    def test_empty_projects(self, engine):
        """No projects = no staleness alerts."""
        result = engine.check_staleness()
        assert result == []

    def test_fresh_project(self, engine, wm):
        """Recently active project should not generate alert."""
        wm.add_project("Fresh Project", description="Just started")
        result = engine.check_staleness()
        assert result == []

    def test_stale_project(self, engine, wm):
        """Project inactive > 7 days should generate alert."""
        proj = wm.add_project("Stale Project")
        # Manually set last_activity to 10 days ago
        old_date = _past_iso(days=10)
        conn = wm._connect()
        try:
            conn.execute(
                "UPDATE projects SET last_activity=? WHERE id=?",
                (old_date, proj.id),
            )
            conn.commit()
        finally:
            conn.close()
        wm._projects[proj.id].last_activity = old_date

        result = engine.check_staleness()
        assert len(result) == 1
        assert result[0].insight_type == InsightType.STALENESS_ALERT
        assert "Stale Project" in result[0].title
        assert result[0].related_entity_id == proj.id

    def test_completed_project_ignored(self, engine, wm):
        """Completed projects should not generate staleness alerts."""
        proj = wm.add_project("Done Project")
        wm.update_project(proj.id, status=ProjectStatus.COMPLETED)
        # Set last_activity old
        old_date = _past_iso(days=20)
        conn = wm._connect()
        try:
            conn.execute(
                "UPDATE projects SET last_activity=? WHERE id=?",
                (old_date, proj.id),
            )
            conn.commit()
        finally:
            conn.close()
        wm._projects[proj.id].last_activity = old_date

        result = engine.check_staleness()
        assert result == []


# ============================================================================
# TestCheckDeadlines
# ============================================================================


class TestCheckDeadlines:
    def test_empty_goals(self, engine):
        result = engine.check_deadlines()
        assert result == []

    def test_no_target_date(self, engine, wm):
        """Goals without target_date should not generate deadline alerts."""
        wm.add_goal("Learn Python", horizon=GoalHorizon.SHORT_TERM)
        result = engine.check_deadlines()
        assert result == []

    def test_approaching_deadline(self, engine, wm):
        """Goal due in 5 days should trigger alert."""
        target = _future_iso(days=5)
        wm.add_goal(
            "Finish report",
            horizon=GoalHorizon.SHORT_TERM,
            target_date=target,
        )
        result = engine.check_deadlines()
        assert len(result) == 1
        assert result[0].insight_type == InsightType.DEADLINE_APPROACHING
        assert result[0].urgency >= 0.7

    def test_distant_deadline(self, engine, wm):
        """Goal due in 60 days should not trigger alert."""
        target = _future_iso(days=60)
        wm.add_goal(
            "Long term plan",
            horizon=GoalHorizon.LONG_TERM,
            target_date=target,
        )
        result = engine.check_deadlines()
        assert result == []


# ============================================================================
# TestCheckContradictions
# ============================================================================


class TestCheckContradictions:
    def test_empty_contradictions(self, engine):
        result = engine.check_contradictions()
        assert result == []

    def test_fresh_contradiction(self, engine, wm):
        """Contradiction < 24h old should not generate alert."""
        b1 = wm.add_belief("Python is best")
        b2 = wm.add_belief("JavaScript is best")
        wm.add_contradiction(b1.id, b2.id, "Language preference conflict")
        result = engine.check_contradictions()
        assert result == []

    def test_old_contradiction(self, engine, wm):
        """Contradiction > 24h old should generate alert."""
        b1 = wm.add_belief("Python is best")
        b2 = wm.add_belief("JavaScript is best")
        contra = wm.add_contradiction(b1.id, b2.id, "Language preference conflict")
        # Set detected_at to 48 hours ago
        old_date = _past_iso(hours=48)
        conn = wm._connect()
        try:
            conn.execute(
                "UPDATE contradictions SET detected_at=? WHERE id=?",
                (old_date, contra.id),
            )
            conn.commit()
        finally:
            conn.close()
        wm._contradictions[contra.id].detected_at = old_date

        result = engine.check_contradictions()
        assert len(result) == 1
        assert result[0].insight_type == InsightType.CONTRADICTION_ALERT


# ============================================================================
# TestCheckRelationshipGaps
# ============================================================================


class TestCheckRelationshipGaps:
    def test_empty_relationships(self, engine):
        result = engine.check_relationship_gaps()
        assert result == []

    def test_casual_contact_skipped(self, engine, wm):
        """Casual contact (1 mention, no important role) should be skipped."""
        rel = wm.add_relationship("Random Person", role="acquaintance")
        # Set last_mentioned old
        old_date = _past_iso(days=30)
        conn = wm._connect()
        try:
            conn.execute(
                "UPDATE relationships SET last_mentioned=? WHERE id=?",
                (old_date, rel.id),
            )
            conn.commit()
        finally:
            conn.close()
        wm._relationships[rel.id].last_mentioned = old_date

        result = engine.check_relationship_gaps()
        assert result == []

    def test_collaborator_gap(self, engine, wm):
        """Important collaborator with 14+ day gap should trigger alert."""
        rel = wm.add_relationship("Alice", role="collaborator")
        # Bump mention count
        wm._relationships[rel.id].mention_count = 5
        conn = wm._connect()
        try:
            conn.execute(
                "UPDATE relationships SET mention_count=5 WHERE id=?",
                (rel.id,),
            )
            conn.commit()
        finally:
            conn.close()
        # Set last_mentioned old
        old_date = _past_iso(days=20)
        conn = wm._connect()
        try:
            conn.execute(
                "UPDATE relationships SET last_mentioned=? WHERE id=?",
                (old_date, rel.id),
            )
            conn.commit()
        finally:
            conn.close()
        wm._relationships[rel.id].last_mentioned = old_date

        result = engine.check_relationship_gaps()
        assert len(result) == 1
        assert result[0].insight_type == InsightType.RELATIONSHIP_GAP
        assert "Alice" in result[0].title


# ============================================================================
# TestCheckGoalBlockers
# ============================================================================


class TestCheckGoalBlockers:
    def test_no_blockers(self, engine, wm):
        """Goal with clean project should not trigger."""
        proj = wm.add_project("Clean Project")
        wm.add_goal("Ship it", related_project_ids=[proj.id])
        result = engine.check_goal_blockers()
        assert result == []

    def test_blocked_goal(self, engine, wm):
        """Goal with blocked project should trigger."""
        proj = wm.add_project("Blocked Project")
        wm.add_blocker(proj.id, "CI pipeline broken", severity="high")
        wm.add_goal("Ship it", related_project_ids=[proj.id])
        result = engine.check_goal_blockers()
        assert len(result) == 1
        assert result[0].insight_type == InsightType.GOAL_RISK
        assert "CI pipeline broken" in result[0].description


# ============================================================================
# TestFilterAndDedup
# ============================================================================


class TestFilterAndDedup:
    def test_low_confidence_dropped(self, engine):
        """Insights below MIN_CONFIDENCE should be filtered out."""
        insights = [
            ProactiveInsight(
                id="low1",
                insight_type=InsightType.STALENESS_ALERT,
                title="Low confidence",
                description="Desc",
                urgency=0.8,
                confidence=0.2,  # Below MIN_CONFIDENCE (0.4)
                generated_at=datetime.now(timezone.utc).isoformat(),
            ),
        ]
        result = engine._filter_and_dedup(insights)
        assert len(result) == 0

    def test_dedup_same_entity(self, engine, wm):
        """Same (type, entity_id) within cooldown should be deduped."""
        # Pre-store an insight in the DB for the same entity
        existing = ProactiveInsight(
            id="existing1",
            insight_type=InsightType.STALENESS_ALERT,
            title="Old alert",
            description="Desc",
            urgency=0.7,
            confidence=0.7,
            generated_at=datetime.now(timezone.utc).isoformat(),
            related_entity_type="project",
            related_entity_id="proj_123",
        )
        wm.store_insight(existing)

        # New insight for same entity/type
        new_insight = ProactiveInsight(
            id="new1",
            insight_type=InsightType.STALENESS_ALERT,
            title="New alert",
            description="Desc",
            urgency=0.8,
            confidence=0.8,
            generated_at=datetime.now(timezone.utc).isoformat(),
            related_entity_type="project",
            related_entity_id="proj_123",
        )
        result = engine._filter_and_dedup([new_insight])
        assert len(result) == 0

    def test_cap_at_max(self, engine):
        """Should cap at MAX_INSIGHTS_PER_RUN."""
        insights = [
            ProactiveInsight(
                id=f"cap_{i}",
                insight_type=InsightType.STALENESS_ALERT,
                title=f"Alert {i}",
                description="Desc",
                urgency=0.9 - i * 0.05,
                confidence=0.8,
                generated_at=datetime.now(timezone.utc).isoformat(),
                related_entity_type="project",
                related_entity_id=f"proj_{i}",
            )
            for i in range(5)
        ]
        result = engine._filter_and_dedup(insights)
        assert len(result) == engine.MAX_INSIGHTS_PER_RUN


# ============================================================================
# TestInsightCRUD
# ============================================================================


class TestInsightCRUD:
    def test_store_and_retrieve(self, wm):
        """Stored insight should be retrievable as pending."""
        insight = ProactiveInsight(
            id="crud1",
            insight_type=InsightType.DEADLINE_APPROACHING,
            title="Due soon",
            description="Goal is due",
            urgency=0.8,
            confidence=0.7,
            generated_at=datetime.now(timezone.utc).isoformat(),
        )
        wm.store_insight(insight)
        pending = wm.get_pending_insights(max_count=5)
        assert len(pending) == 1
        assert pending[0]["id"] == "crud1"
        assert pending[0]["title"] == "Due soon"

    def test_get_pending_excludes_dismissed(self, wm):
        """Dismissed insights should not appear in pending."""
        insight = ProactiveInsight(
            id="crud2",
            insight_type=InsightType.STALENESS_ALERT,
            title="Old",
            description="Desc",
            urgency=0.7,
            confidence=0.7,
            generated_at=datetime.now(timezone.utc).isoformat(),
        )
        wm.store_insight(insight)
        wm.update_insight_feedback("crud2", "dismissed")
        pending = wm.get_pending_insights()
        assert len(pending) == 0

    def test_feedback_engaged(self, wm):
        """Engaged feedback should set acted_on_at."""
        insight = ProactiveInsight(
            id="crud3",
            insight_type=InsightType.CONTRADICTION_ALERT,
            title="Conflict",
            description="Desc",
            urgency=0.6,
            confidence=0.6,
            generated_at=datetime.now(timezone.utc).isoformat(),
        )
        wm.store_insight(insight)
        result = wm.update_insight_feedback("crud3", "engaged")
        assert result is True
        # Verify acted_on_at is set
        conn = wm._connect()
        try:
            row = conn.execute(
                "SELECT acted_on_at FROM proactive_insights WHERE id=?",
                ("crud3",),
            ).fetchone()
            assert row["acted_on_at"] is not None
        finally:
            conn.close()


# ============================================================================
# TestAwarenessContext
# ============================================================================


class TestAwarenessContext:
    def test_empty_world(self, engine):
        """Empty world model should return empty context."""
        context = engine.get_awareness_context()
        assert context == ""

    def test_with_pending_insights(self, engine, wm):
        """Should include pending insights in context."""
        insight = ProactiveInsight(
            id="ctx1",
            insight_type=InsightType.STALENESS_ALERT,
            title="Project X is stale",
            description="Desc",
            urgency=0.7,
            confidence=0.7,
            generated_at=datetime.now(timezone.utc).isoformat(),
        )
        wm.store_insight(insight)
        context = engine.get_awareness_context()
        assert "[Proactive Awareness]" in context
        assert "Project X is stale" in context


# ============================================================================
# TestRunFullAnalysis
# ============================================================================


class TestRunFullAnalysis:
    def test_with_populated_wm(self, engine, wm):
        """Full analysis on populated world model should produce insights."""
        # Create a stale project
        proj = wm.add_project("Old Project")
        old_date = _past_iso(days=10)
        conn = wm._connect()
        try:
            conn.execute(
                "UPDATE projects SET last_activity=? WHERE id=?",
                (old_date, proj.id),
            )
            conn.commit()
        finally:
            conn.close()
        wm._projects[proj.id].last_activity = old_date

        # Create a goal with approaching deadline
        target = _future_iso(days=3)
        wm.add_goal("Ship feature", target_date=target)

        results = engine.run_full_analysis()
        assert len(results) >= 1  # At least staleness or deadline

        # Verify insights were stored
        pending = wm.get_pending_insights(max_count=10)
        assert len(pending) >= 1

    def test_disabled_returns_empty(self, disabled_engine):
        """Disabled engine should return empty list."""
        results = disabled_engine.run_full_analysis()
        assert results == []


# ============================================================================
# TestRunQuickAnalysis
# ============================================================================


class TestRunQuickAnalysis:
    def test_only_fast_checks(self, engine, wm):
        """Quick analysis should only run staleness, deadlines, contradictions."""
        # Create a stale project
        proj = wm.add_project("Stale Quick")
        old_date = _past_iso(days=15)
        conn = wm._connect()
        try:
            conn.execute(
                "UPDATE projects SET last_activity=? WHERE id=?",
                (old_date, proj.id),
            )
            conn.commit()
        finally:
            conn.close()
        wm._projects[proj.id].last_activity = old_date

        # Create a relationship gap (should NOT be caught by quick analysis)
        rel = wm.add_relationship("Important Person", role="collaborator")
        wm._relationships[rel.id].mention_count = 5
        old_rel_date = _past_iso(days=30)
        conn = wm._connect()
        try:
            conn.execute(
                "UPDATE relationships SET last_mentioned=?, mention_count=5 WHERE id=?",
                (old_rel_date, rel.id),
            )
            conn.commit()
        finally:
            conn.close()
        wm._relationships[rel.id].last_mentioned = old_rel_date

        results = engine.run_quick_analysis()
        # Should have staleness alert but NOT relationship gap
        types = [r.insight_type for r in results]
        assert InsightType.STALENESS_ALERT in types
        assert InsightType.RELATIONSHIP_GAP not in types


# ============================================================================
# Phase 4 Tests — check_patterns
# ============================================================================


class TestCheckPatterns:
    def test_empty_patterns(self, engine):
        """No patterns available = no alerts."""
        mock_prophet = MagicMock()
        mock_prophet.patterns = {}
        with patch(
            "aura.patterns.get_pattern_prophet",
            return_value=mock_prophet,
        ):
            result = engine.check_patterns()
        assert result == []

    def test_high_confidence_pattern(self, engine):
        """Pattern with confidence >= 0.5 and occurrences >= 3 should generate alert."""
        mock_pattern = MagicMock()
        mock_pattern.name = "morning_coding"
        mock_pattern.confidence = 0.75
        mock_pattern.occurrences = 5
        mock_pattern.last_seen = _past_iso(days=2)
        mock_pattern.description = "User codes every morning"

        mock_prophet = MagicMock()
        mock_prophet.patterns = {"morning_coding": mock_pattern}

        with patch(
            "aura.patterns.get_pattern_prophet",
            return_value=mock_prophet,
        ):
            result = engine.check_patterns()

        assert len(result) == 1
        assert result[0].insight_type == InsightType.PATTERN_ALERT
        assert "morning_coding" in result[0].title
        assert result[0].confidence == 0.75

    def test_low_confidence_filtered(self, engine):
        """Pattern with confidence < 0.5 should be filtered out."""
        mock_pattern = MagicMock()
        mock_pattern.name = "weak_signal"
        mock_pattern.confidence = 0.3
        mock_pattern.occurrences = 5
        mock_pattern.last_seen = _past_iso(days=1)
        mock_pattern.description = "Weak"

        mock_prophet = MagicMock()
        mock_prophet.patterns = {"weak_signal": mock_pattern}

        with patch(
            "aura.patterns.get_pattern_prophet",
            return_value=mock_prophet,
        ):
            result = engine.check_patterns()

        assert result == []


# ============================================================================
# Phase 4 Tests — check_priority_shifts
# ============================================================================


class TestCheckPriorityShifts:
    def test_empty_state_changes(self, engine):
        """No state changes = no priority shifts."""
        result = engine.check_priority_shifts()
        assert result == []

    def test_surge_detected(self, engine, wm):
        """Project with recent surge (>=5 recent, <=1 prior) should trigger."""
        proj = wm.add_project("Surging Project")

        # Add 6 recent state changes (within last 7 days)
        now = datetime.now(timezone.utc)
        for i in range(6):
            ts = (now - timedelta(days=i % 5, hours=i)).isoformat()
            conn = wm._connect()
            try:
                conn.execute(
                    """INSERT INTO state_changes
                       (timestamp, change_type, entity_type, entity_id, reasoning)
                       VALUES (?, ?, ?, ?, ?)""",
                    (ts, "project_update", "project", proj.id, f"change_{i}"),
                )
                conn.commit()
            finally:
                conn.close()

        result = engine.check_priority_shifts()
        assert len(result) == 1
        assert result[0].insight_type == InsightType.PRIORITY_SHIFT
        assert "surging" in result[0].title.lower()
        assert result[0].related_entity_id == proj.id

    def test_drop_detected(self, engine, wm):
        """Project with activity drop (>=5 prior, <=1 recent) should trigger."""
        proj = wm.add_project("Dropping Project")

        # Add 6 state changes in prior window (7-14 days ago)
        now = datetime.now(timezone.utc)
        for i in range(6):
            ts = (now - timedelta(days=8 + i % 5, hours=i)).isoformat()
            conn = wm._connect()
            try:
                conn.execute(
                    """INSERT INTO state_changes
                       (timestamp, change_type, entity_type, entity_id, reasoning)
                       VALUES (?, ?, ?, ?, ?)""",
                    (ts, "project_update", "project", proj.id, f"change_{i}"),
                )
                conn.commit()
            finally:
                conn.close()

        result = engine.check_priority_shifts()
        assert len(result) == 1
        assert result[0].insight_type == InsightType.PRIORITY_SHIFT
        assert "dropped off" in result[0].title.lower()


# ============================================================================
# Phase 4 Tests — check_stress_correlations
# ============================================================================


class TestCheckStressCorrelations:
    def test_no_stress(self, engine):
        """When both ALMA and ToM report no stress, no insight generated."""
        mock_alma_state = {
            "pad": {"arousal": 0.3, "pleasure": 0.2, "dominance": 0.5},
        }
        mock_emo = MagicMock()
        mock_emo.frustration = 0.2

        mock_tom = MagicMock()
        mock_tom.get_emotional_state.return_value = mock_emo

        with patch(
            "aura.emotion.alma_engine.get_emotional_state",
            return_value=mock_alma_state,
        ), patch(
            "aura.proactive.theory_of_mind.get_theory_of_mind",
            return_value=mock_tom,
        ):
            result = engine.check_stress_correlations()

        assert result == []

    def test_stress_with_workload(self, engine, wm):
        """Combined stress from ALMA + ToM with workload should trigger insight."""
        # Set up workload context
        proj = wm.add_project("Stressful Project")
        wm.add_blocker(proj.id, "CI broken", severity="high")
        wm.add_goal(
            "Ship feature",
            target_date=_future_iso(days=3),
            related_project_ids=[proj.id],
        )

        # ALMA: high arousal + strong negative pleasure
        # stress = (0.9 - 0.5) * abs(-0.8) = 0.4 * 0.8 = 0.32
        mock_alma_state = {
            "pad": {"arousal": 0.9, "pleasure": -0.8, "dominance": 0.3},
        }

        # ToM: high frustration
        # stress = (0.8 - 0.5) * 0.5 = 0.15
        # Combined = 0.32 + 0.15 = 0.47, above 0.3 threshold
        mock_emo = MagicMock()
        mock_emo.frustration = 0.8

        mock_tom = MagicMock()
        mock_tom.get_emotional_state.return_value = mock_emo

        with patch(
            "aura.emotion.alma_engine.get_emotional_state",
            return_value=mock_alma_state,
        ), patch(
            "aura.proactive.theory_of_mind.get_theory_of_mind",
            return_value=mock_tom,
        ):
            result = engine.check_stress_correlations()

        assert len(result) == 1
        assert result[0].insight_type == InsightType.STRESS_CORRELATION
        assert result[0].related_entity_id == "global"
        assert "ALMA" in result[0].description
        assert "ToM" in result[0].description


# ============================================================================
# Phase 4 Tests — get_state_change_counts
# ============================================================================


class TestStateChangeCounts:
    def test_empty(self, wm):
        """No state changes returns empty dict."""
        result = wm.get_state_change_counts(
            "project", since=_past_iso(days=7)
        )
        assert result == {}

    def test_with_data(self, wm):
        """State changes within window are counted correctly."""
        proj1 = wm.add_project("P1")
        proj2 = wm.add_project("P2")
        # add_project already logs 1 state_change per project

        now = datetime.now(timezone.utc)
        # Add 3 more for proj1, 2 more for proj2
        for i in range(3):
            ts = (now - timedelta(days=i + 1)).isoformat()
            conn = wm._connect()
            try:
                conn.execute(
                    """INSERT INTO state_changes
                       (timestamp, change_type, entity_type, entity_id, reasoning)
                       VALUES (?, ?, ?, ?, ?)""",
                    (ts, "project_update", "project", proj1.id, f"c{i}"),
                )
                conn.commit()
            finally:
                conn.close()

        for i in range(2):
            ts = (now - timedelta(days=i + 1)).isoformat()
            conn = wm._connect()
            try:
                conn.execute(
                    """INSERT INTO state_changes
                       (timestamp, change_type, entity_type, entity_id, reasoning)
                       VALUES (?, ?, ?, ?, ?)""",
                    (ts, "project_update", "project", proj2.id, f"c{i}"),
                )
                conn.commit()
            finally:
                conn.close()

        result = wm.get_state_change_counts(
            "project", since=_past_iso(days=7)
        )
        # 1 (from add_project) + 3 manual = 4
        assert result[proj1.id] == 4
        # 1 (from add_project) + 2 manual = 3
        assert result[proj2.id] == 3


# ============================================================================
# Phase 4 Tests — Drive Signals + Motivation Wiring
# ============================================================================


class TestDriveSignals:
    def test_neutral_signals(self, engine):
        """Empty world model should return neutral drive signals."""
        signals = engine.get_drive_signals()
        assert signals["curiosity"] == 0.5
        assert signals["coherence"] == 0.5
        assert signals["social"] == 0.5
        assert signals["competence"] == 0.5

    def test_signals_with_contradictions(self, engine, wm):
        """Unresolved contradictions should lower coherence signal."""
        b1 = wm.add_belief("A is true")
        b2 = wm.add_belief("A is false")
        wm.add_contradiction(b1.id, b2.id, "Contradiction about A")

        signals = engine.get_drive_signals()
        assert signals["coherence"] < 0.5  # Reduced by contradiction


# ============================================================================
# Phase 5 Tests — Threshold Tuning
# ============================================================================


class TestTuneThresholds:
    """Tests for _tune_thresholds() feedback loop."""

    def test_no_data_no_change(self, engine, wm):
        """No insight history = no threshold changes."""
        engine._tune_thresholds()
        assert engine._confidence_overrides == {}

    def test_high_engagement_lowers_threshold(self, engine, wm):
        """Types with engagement_rate > 0.6 should get lowered threshold."""
        # Create 6 staleness insights, all engaged (engagement_rate = 1.0)
        now = datetime.now(timezone.utc).isoformat()
        for i in range(6):
            insight = ProactiveInsight(
                id=f"tune_hi_{i}",
                insight_type=InsightType.STALENESS_ALERT,
                title=f"Stale {i}",
                description="Desc",
                urgency=0.7,
                confidence=0.7,
                generated_at=now,
                related_entity_type="project",
                related_entity_id=f"proj_tune_{i}",
            )
            wm.store_insight(insight)
            wm.update_insight_feedback(f"tune_hi_{i}", "engaged")

        # Force tune (bypass rate limit — set far in the past so monotonic() - it > 3600)
        import time as _time
        engine._last_tune_time = _time.monotonic() - 7200
        engine._tune_thresholds()

        assert "staleness_alert" in engine._confidence_overrides
        # Default MIN_CONFIDENCE is 0.4, should be lowered by 0.02 to 0.38
        assert engine._confidence_overrides["staleness_alert"] < engine.MIN_CONFIDENCE

    def test_low_engagement_raises_threshold(self, engine, wm):
        """Types with engagement_rate < 0.2 should get raised threshold."""
        now = datetime.now(timezone.utc).isoformat()
        for i in range(6):
            insight = ProactiveInsight(
                id=f"tune_lo_{i}",
                insight_type=InsightType.CONTRADICTION_ALERT,
                title=f"Contradiction {i}",
                description="Desc",
                urgency=0.6,
                confidence=0.6,
                generated_at=now,
                related_entity_type="contradiction",
                related_entity_id=f"contra_tune_{i}",
            )
            wm.store_insight(insight)
            wm.update_insight_feedback(f"tune_lo_{i}", "dismissed")

        import time as _time
        engine._last_tune_time = _time.monotonic() - 7200
        engine._tune_thresholds()

        assert "contradiction_alert" in engine._confidence_overrides
        # Default MIN_CONFIDENCE is 0.4, should be raised by 0.02 to 0.42
        assert engine._confidence_overrides["contradiction_alert"] > engine.MIN_CONFIDENCE


# ============================================================================
# Phase 5 Tests — Confidence Overrides in Filter
# ============================================================================


class TestConfidenceOverrides:
    """Tests for per-type confidence overrides in _filter_and_dedup."""

    def test_override_applied_in_filter(self, engine):
        """Insight below MIN_CONFIDENCE but above override should pass."""
        # Set a low override for staleness
        engine._confidence_overrides["staleness_alert"] = 0.25

        insight = ProactiveInsight(
            id="override_pass",
            insight_type=InsightType.STALENESS_ALERT,
            title="Low conf stale",
            description="Desc",
            urgency=0.7,
            confidence=0.3,  # Below MIN_CONFIDENCE (0.4), above override (0.25)
            generated_at=datetime.now(timezone.utc).isoformat(),
            related_entity_type="project",
            related_entity_id="proj_override",
        )
        result = engine._filter_and_dedup([insight])
        assert len(result) == 1

    def test_default_used_when_no_override(self, engine):
        """Insight type without override uses default MIN_CONFIDENCE."""
        insight = ProactiveInsight(
            id="no_override",
            insight_type=InsightType.OPPORTUNITY,
            title="An opportunity",
            description="Desc",
            urgency=0.7,
            confidence=0.3,  # Below MIN_CONFIDENCE (0.4), no override for OPPORTUNITY
            generated_at=datetime.now(timezone.utc).isoformat(),
            related_entity_type="global",
            related_entity_id="global",
        )
        result = engine._filter_and_dedup([insight])
        assert len(result) == 0
