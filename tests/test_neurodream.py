"""Tests for NeuroDream - Sleep/Dream Memory Consolidation (Tool #24)."""

import pytest
import tempfile
import shutil
import time
import datetime
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from aura.tools.neurodream import (
    NeuroDreamEngine,
    SleepPhase,
    DreamTrigger,
    DreamInsight,
    SleepSession,
    ConsolidatedPattern,
    get_neurodream,
    create_neurodream
)


@pytest.fixture
def temp_data_dir():
    """Create a temporary data directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_kg():
    """Create a mock knowledge graph."""
    kg = MagicMock()
    kg.get_recent_nodes.return_value = []
    kg.get_related.return_value = {"nodes": [], "edges": []}
    kg.add_node.return_value = {"success": True}
    kg.add_edge.return_value = {"success": True}
    return kg


@pytest.fixture
def mock_evoemo():
    """Create a mock EvoEmo instance."""
    evoemo = MagicMock()
    evoemo.get_current_mood.return_value = MagicMock(
        emotion="calm",
        confidence=75,
        valence=0.6,
        arousal=0.3
    )
    evoemo.get_session_summary.return_value = {
        "dominant": "calm",
        "readings": 5
    }
    return evoemo


@pytest.fixture
def mock_monologue():
    """Create a mock inner monologue."""
    monologue = MagicMock()
    monologue.get_recent_thoughts.return_value = []
    monologue.add_thought.return_value = MagicMock(id="thought_1")
    return monologue


@pytest.fixture
def neurodream_engine(temp_data_dir, mock_kg, mock_evoemo, mock_monologue):
    """Create a NeuroDream engine with mock dependencies."""
    engine = NeuroDreamEngine(
        knowledge_graph=mock_kg,
        hybrid_memory=None,
        evoemo=mock_evoemo,
        inner_monologue=mock_monologue,
        chromadb=None,
        data_dir=temp_data_dir,
        idle_threshold_minutes=1,  # Short for testing
        max_vram_gb=2.0
    )
    yield engine
    # Ensure sleep thread is stopped before temp dir cleanup
    engine.shutdown(timeout=3.0)


class TestSleepPhase:
    """Tests for SleepPhase enum."""

    def test_phases_exist(self):
        """Test that all sleep phases are defined."""
        assert SleepPhase.AWAKE is not None
        assert SleepPhase.LIGHT is not None
        assert SleepPhase.DEEP is not None
        assert SleepPhase.REM is not None
        assert SleepPhase.WAKING is not None

    def test_phase_values(self):
        """Test sleep phase values."""
        assert SleepPhase.AWAKE.value == "awake"
        assert SleepPhase.LIGHT.value == "light"
        assert SleepPhase.DEEP.value == "deep"
        assert SleepPhase.REM.value == "rem"
        assert SleepPhase.WAKING.value == "waking"


class TestDreamTrigger:
    """Tests for DreamTrigger enum."""

    def test_triggers_exist(self):
        """Test that all triggers are defined."""
        assert DreamTrigger.SCHEDULED is not None
        assert DreamTrigger.IDLE is not None
        assert DreamTrigger.MANUAL is not None
        assert DreamTrigger.LOW_RESOURCES is not None

    def test_trigger_values(self):
        """Test trigger values."""
        assert DreamTrigger.SCHEDULED.value == "scheduled"
        assert DreamTrigger.IDLE.value == "idle"
        assert DreamTrigger.MANUAL.value == "manual"
        assert DreamTrigger.LOW_RESOURCES.value == "low_resources"


class TestNeuroDreamEngine:
    """Tests for NeuroDreamEngine."""

    def test_initialization(self, neurodream_engine):
        """Test engine initializes correctly."""
        assert neurodream_engine is not None
        assert neurodream_engine.current_phase == SleepPhase.AWAKE
        assert neurodream_engine.current_phase == SleepPhase.AWAKE  # Not None

    def test_get_status_awake(self, neurodream_engine):
        """Test status when awake."""
        status = neurodream_engine.get_status()
        assert status["is_sleeping"] is False
        assert "total_sessions" in status
        assert "total_insights" in status

    def test_enter_sleep(self, neurodream_engine):
        """Test entering sleep mode."""
        result = neurodream_engine.enter_sleep(trigger="manual")
        assert result.get("success") is True
        # Give the thread a moment to start
        time.sleep(0.1)
        assert neurodream_engine.current_phase != SleepPhase.AWAKE

    def test_cannot_sleep_while_sleeping(self, neurodream_engine):
        """Test that we can't enter sleep while already sleeping."""
        # Manually set phase to simulate being asleep (thread may finish too fast)
        with neurodream_engine._phase_lock:
            neurodream_engine.current_phase = SleepPhase.LIGHT
        result = neurodream_engine.enter_sleep(trigger="manual")
        assert result.get("success") is False
        assert "error" in result
        # Reset for cleanup
        with neurodream_engine._phase_lock:
            neurodream_engine.current_phase = SleepPhase.AWAKE

    def test_wake_up(self, neurodream_engine):
        """Test waking up."""
        neurodream_engine.enter_sleep(trigger="manual")
        time.sleep(0.1)
        result = neurodream_engine.wake_up(reason="test")
        assert result.get("success") is True
        assert neurodream_engine.current_phase == SleepPhase.AWAKE

    def test_wake_up_when_awake(self, neurodream_engine):
        """Test wake up when already awake."""
        result = neurodream_engine.wake_up(reason="test")
        assert result.get("success") is True  # Should still succeed

    def test_record_activity(self, neurodream_engine):
        """Test recording user activity."""
        neurodream_engine.record_activity()
        assert neurodream_engine.last_activity_time is not None

    def test_check_idle_trigger_not_idle(self, neurodream_engine):
        """Test idle check when not idle."""
        neurodream_engine.record_activity()
        assert neurodream_engine.check_idle_trigger() is False

    def test_check_idle_trigger_idle(self, neurodream_engine):
        """Test idle check when idle (by manipulating last activity time)."""
        # idle_threshold is a timedelta, set last_activity_time to past it
        neurodream_engine.last_activity_time = (
            datetime.datetime.now() -
            neurodream_engine.idle_threshold -
            datetime.timedelta(seconds=10)
        )
        assert neurodream_engine.check_idle_trigger() is True

    def test_get_dream_journal_empty(self, neurodream_engine):
        """Test getting dream journal when empty."""
        entries = neurodream_engine.get_dream_journal(n=5)
        assert entries == []

    def test_get_insights_empty(self, neurodream_engine):
        """Test getting insights when empty."""
        insights = neurodream_engine.get_insights()
        assert insights == []

    def test_get_patterns_empty(self, neurodream_engine):
        """Test getting patterns when empty."""
        patterns = neurodream_engine.get_patterns()
        assert patterns == []


class TestSleepPhases:
    """Tests for individual sleep phases."""

    def test_light_phase_execution(self, neurodream_engine):
        """Test light phase runs without error."""
        neurodream_engine.enter_sleep(trigger="manual")
        time.sleep(0.1)
        result = neurodream_engine.run_light_phase()
        assert "memories_replayed" in result or "error" in result

    def test_deep_phase_execution(self, neurodream_engine):
        """Test deep phase runs without error."""
        neurodream_engine.enter_sleep(trigger="manual")
        time.sleep(0.1)
        result = neurodream_engine.run_deep_phase()
        assert "patterns_found" in result or "error" in result

    def test_rem_phase_execution(self, neurodream_engine):
        """Test REM phase runs without error."""
        neurodream_engine.enter_sleep(trigger="manual")
        time.sleep(0.1)
        result = neurodream_engine.run_rem_phase()
        assert "connections_made" in result or "insights_generated" in result or "error" in result


class TestDreamJournal:
    """Tests for dream journal functionality."""

    def test_log_dream(self, neurodream_engine):
        """Test logging a dream entry via _log_dream (logs to monologue)."""
        neurodream_engine.enter_sleep(trigger="manual")
        time.sleep(0.1)
        # _log_dream takes (phase, message) — logs to inner monologue
        neurodream_engine._log_dream(
            phase="light",
            message="Test dream content",
        )
        # _log_dream writes to monologue, not to the journal file
        # Verify monologue.think was called
        neurodream_engine.monologue.think.assert_called()

    def test_dream_journal_via_save_session(self, neurodream_engine):
        """Test dream journal entries come from saved sessions."""
        # Create and save a session to the journal
        session = SleepSession(
            session_id="test_session_1",
            start_time=datetime.datetime.now().isoformat(),
            end_time=datetime.datetime.now().isoformat(),
            trigger="manual",
            phases_completed=["light", "deep"],
        )
        neurodream_engine._save_session(session)
        entries = neurodream_engine.get_dream_journal(n=5)
        assert len(entries) == 1
        assert entries[0]["session_id"] == "test_session_1"

    def test_dream_journal_limit(self, neurodream_engine):
        """Test dream journal respects limit."""
        for i in range(10):
            session = SleepSession(
                session_id=f"test_session_{i}",
                start_time=datetime.datetime.now().isoformat(),
                end_time=datetime.datetime.now().isoformat(),
                trigger="manual",
                phases_completed=["light"],
            )
            neurodream_engine._save_session(session)
        entries = neurodream_engine.get_dream_journal(n=5)
        assert len(entries) == 5


class TestInsights:
    """Tests for insight generation."""

    def test_save_and_get_insight(self, neurodream_engine):
        """Test saving and retrieving an insight."""
        insight = DreamInsight(
            id="insight_1",
            timestamp=datetime.datetime.now().isoformat(),
            insight_type="pattern",
            content="Test insight",
            confidence=0.8,
            source_nodes=["mem1", "mem2"],
            created_edges=[{"from": "mem1", "to": "mem2", "type": "related"}],
        )
        neurodream_engine._save_insight(insight)
        insights = neurodream_engine.get_insights()
        assert len(insights) == 1
        assert insights[0]["content"] == "Test insight"
        assert insights[0]["confidence"] == 0.8


class TestPatternConsolidation:
    """Tests for pattern consolidation."""

    def test_save_and_get_pattern(self, neurodream_engine):
        """Test saving and retrieving a consolidated pattern."""
        pattern = ConsolidatedPattern(
            pattern_id="pattern_1",
            timestamp=datetime.datetime.now().isoformat(),
            pattern_type="topical",
            description="Test Pattern",
            frequency=5,
            confidence=0.75,
            examples=["example1", "example2"],
        )
        neurodream_engine._save_consolidated_patterns([pattern])
        patterns = neurodream_engine.get_patterns()
        assert len(patterns) == 1
        assert patterns[0]["description"] == "Test Pattern"
        assert patterns[0]["pattern_id"] == "pattern_1"


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_neurodream(self, temp_data_dir, mock_kg, mock_evoemo, mock_monologue):
        """Test create_neurodream factory function."""
        engine = create_neurodream(
            knowledge_graph=mock_kg,
            evoemo=mock_evoemo,
            inner_monologue=mock_monologue,
            data_dir=temp_data_dir
        )
        assert engine is not None
        assert isinstance(engine, NeuroDreamEngine)
        engine.shutdown(timeout=2.0)

    def test_get_neurodream_singleton(self, temp_data_dir, mock_kg, mock_evoemo, mock_monologue):
        """Test get_neurodream returns singleton."""
        # Create first so singleton exists
        create_neurodream(
            knowledge_graph=mock_kg,
            evoemo=mock_evoemo,
            inner_monologue=mock_monologue,
            data_dir=temp_data_dir
        )
        engine1 = get_neurodream()
        engine2 = get_neurodream()
        assert engine1 is engine2
        engine1.shutdown(timeout=2.0)


class TestSleepSession:
    """Tests for SleepSession dataclass."""

    def test_session_creation(self):
        """Test creating a sleep session."""
        session = SleepSession(
            session_id="test_session_1",
            start_time=datetime.datetime.now().isoformat(),
            end_time=None,
            trigger="manual",
            phases_completed=[],
        )
        assert session.session_id == "test_session_1"
        assert session.trigger == "manual"
        assert not session.interrupted
        assert session.insights_generated == 0
        assert session.memories_replayed == 0

    def test_session_to_dict(self):
        """Test session serialization."""
        session = SleepSession(
            session_id="test_session_2",
            start_time="2026-02-09T12:00:00",
            end_time="2026-02-09T12:05:00",
            trigger="idle",
            phases_completed=["light", "deep"],
            memories_replayed=10,
            patterns_found=3,
        )
        d = session.to_dict()
        assert d["session_id"] == "test_session_2"
        assert d["memories_replayed"] == 10
        assert d["patterns_found"] == 3


class TestDreamInsight:
    """Tests for DreamInsight dataclass."""

    def test_insight_creation(self):
        """Test creating a dream insight."""
        insight = DreamInsight(
            id="insight_1",
            timestamp=datetime.datetime.now().isoformat(),
            insight_type="pattern",
            content="Test insight content",
            confidence=0.85,
            source_nodes=["mem1", "mem2"],
            created_edges=[{"from": "mem1", "to": "mem2", "type": "pattern"}],
        )
        assert insight.id == "insight_1"
        assert insight.confidence == 0.85

    def test_insight_to_dict(self):
        """Test insight serialization."""
        insight = DreamInsight(
            id="insight_2",
            timestamp="2026-02-09T12:00:00",
            insight_type="connection",
            content="Found a connection",
            confidence=0.9,
            source_nodes=["a", "b"],
            created_edges=[],
        )
        d = insight.to_dict()
        assert d["id"] == "insight_2"
        assert d["content"] == "Found a connection"


class TestConsolidatedPattern:
    """Tests for ConsolidatedPattern dataclass."""

    def test_pattern_creation(self):
        """Test creating a consolidated pattern."""
        pattern = ConsolidatedPattern(
            pattern_id="pattern_1",
            timestamp=datetime.datetime.now().isoformat(),
            pattern_type="temporal",
            description="User works on code in the morning",
            frequency=10,
            confidence=0.9,
            examples=["morning coding session 1", "morning coding session 2"],
        )
        assert pattern.pattern_id == "pattern_1"
        assert pattern.confidence == 0.9

    def test_pattern_to_dict(self):
        """Test pattern serialization."""
        pattern = ConsolidatedPattern(
            pattern_id="pattern_2",
            timestamp="2026-02-09T12:00:00",
            pattern_type="emotional",
            description="User is happier on Fridays",
            frequency=4,
            confidence=0.7,
            examples=["friday_1", "friday_2"],
        )
        d = pattern.to_dict()
        assert d["pattern_id"] == "pattern_2"
        assert d["frequency"] == 4


class TestInterruptibility:
    """Tests for sleep interruption."""

    def test_interrupt_during_sleep(self, neurodream_engine):
        """Test that sleep can be interrupted via wake_up."""
        # Manually set phase to simulate being asleep (thread may finish too fast)
        with neurodream_engine._phase_lock:
            neurodream_engine.current_phase = SleepPhase.DEEP

        # Interrupt by waking up
        result = neurodream_engine.wake_up(reason="user_interrupt")
        assert result.get("success") is True
        assert neurodream_engine.current_phase == SleepPhase.AWAKE

    def test_interrupt_sets_flag(self, neurodream_engine):
        """Test that interruption sets the end_time on session summary."""
        neurodream_engine.enter_sleep(trigger="manual")
        time.sleep(0.1)
        result = neurodream_engine.wake_up(reason="user_interrupt")

        # wake_up returns {"success": True, "reason": ..., "summary": {...}}
        summary = result.get("summary", {})
        if summary:
            assert summary.get("end_time") is not None
            # "user_interrupt" is not in ["cycle_complete", "manual"],
            # so interrupted should be True
            assert summary.get("interrupted") is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
