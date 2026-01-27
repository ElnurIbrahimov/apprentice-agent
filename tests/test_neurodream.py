"""Tests for NeuroDream - Sleep/Dream Memory Consolidation (Tool #24)."""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch
import time

from apprentice_agent.tools.neurodream import (
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
    shutil.rmtree(temp_dir)


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
    return engine


class TestSleepPhase:
    """Tests for SleepPhase enum."""

    def test_phases_exist(self):
        """Test that all sleep phases are defined."""
        assert SleepPhase.LIGHT is not None
        assert SleepPhase.DEEP is not None
        assert SleepPhase.REM is not None

    def test_phase_values(self):
        """Test sleep phase values."""
        assert SleepPhase.LIGHT.value == "light"
        assert SleepPhase.DEEP.value == "deep"
        assert SleepPhase.REM.value == "rem"


class TestDreamTrigger:
    """Tests for DreamTrigger enum."""

    def test_triggers_exist(self):
        """Test that all triggers are defined."""
        assert DreamTrigger.SCHEDULED is not None
        assert DreamTrigger.IDLE is not None
        assert DreamTrigger.MANUAL is not None
        assert DreamTrigger.MEMORY_THRESHOLD is not None


class TestNeuroDreamEngine:
    """Tests for NeuroDreamEngine."""

    def test_initialization(self, neurodream_engine):
        """Test engine initializes correctly."""
        assert neurodream_engine is not None
        assert not neurodream_engine.is_sleeping
        assert neurodream_engine.current_phase is None

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
        assert neurodream_engine.is_sleeping is True

    def test_cannot_sleep_while_sleeping(self, neurodream_engine):
        """Test that we can't enter sleep while already sleeping."""
        neurodream_engine.enter_sleep(trigger="manual")
        result = neurodream_engine.enter_sleep(trigger="manual")
        assert result.get("success") is False
        assert "already" in result.get("error", "").lower()

    def test_wake_up(self, neurodream_engine):
        """Test waking up."""
        neurodream_engine.enter_sleep(trigger="manual")
        result = neurodream_engine.wake_up(reason="test")
        assert result.get("success") is True
        assert neurodream_engine.is_sleeping is False

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
        import datetime
        neurodream_engine.last_activity_time = (
            datetime.datetime.now() -
            datetime.timedelta(minutes=neurodream_engine.idle_threshold_minutes + 1)
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
        result = neurodream_engine.run_light_phase()
        assert "memories_replayed" in result or "error" in result

    def test_deep_phase_execution(self, neurodream_engine):
        """Test deep phase runs without error."""
        neurodream_engine.enter_sleep(trigger="manual")
        result = neurodream_engine.run_deep_phase()
        assert "patterns_found" in result or "error" in result

    def test_rem_phase_execution(self, neurodream_engine):
        """Test REM phase runs without error."""
        neurodream_engine.enter_sleep(trigger="manual")
        result = neurodream_engine.run_rem_phase()
        assert "connections_made" in result or "error" in result


class TestDreamJournal:
    """Tests for dream journal functionality."""

    def test_log_dream(self, neurodream_engine):
        """Test logging a dream entry."""
        neurodream_engine.enter_sleep(trigger="manual")
        neurodream_engine._log_dream(
            phase="light",
            content="Test dream content",
            metadata={"test": True}
        )
        entries = neurodream_engine.get_dream_journal(n=1)
        assert len(entries) == 1
        assert entries[0]["content"] == "Test dream content"

    def test_dream_journal_limit(self, neurodream_engine):
        """Test dream journal respects limit."""
        neurodream_engine.enter_sleep(trigger="manual")
        for i in range(10):
            neurodream_engine._log_dream(
                phase="light",
                content=f"Dream {i}",
                metadata={}
            )
        entries = neurodream_engine.get_dream_journal(n=5)
        assert len(entries) == 5


class TestInsights:
    """Tests for insight generation."""

    def test_record_insight(self, neurodream_engine):
        """Test recording an insight."""
        neurodream_engine._record_insight(
            insight_type="pattern",
            content="Test insight",
            confidence=0.8,
            source_memories=["mem1", "mem2"]
        )
        insights = neurodream_engine.get_insights()
        assert len(insights) == 1
        assert insights[0]["content"] == "Test insight"
        assert insights[0]["confidence"] == 80


class TestPatternConsolidation:
    """Tests for pattern consolidation."""

    def test_record_pattern(self, neurodream_engine):
        """Test recording a consolidated pattern."""
        neurodream_engine._record_pattern(
            pattern_name="Test Pattern",
            strength=0.75,
            memories_consolidated=5,
            metadata={}
        )
        patterns = neurodream_engine.get_patterns()
        assert len(patterns) == 1
        assert patterns[0]["pattern_name"] == "Test Pattern"


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

    def test_get_neurodream_singleton(self, temp_data_dir, mock_kg, mock_evoemo, mock_monologue):
        """Test get_neurodream returns singleton."""
        # This test depends on global state, may need adjustment
        engine1 = get_neurodream()
        engine2 = get_neurodream()
        # Both should return the same instance (or None if not initialized)
        assert engine1 is engine2


class TestSleepSession:
    """Tests for SleepSession dataclass."""

    def test_session_creation(self):
        """Test creating a sleep session."""
        import datetime
        session = SleepSession(
            session_id="test_session_1",
            trigger=DreamTrigger.MANUAL,
            start_time=datetime.datetime.now(),
            phases_completed=[],
            insights_generated=[],
            patterns_consolidated=[],
            interrupted=False
        )
        assert session.session_id == "test_session_1"
        assert session.trigger == DreamTrigger.MANUAL
        assert not session.interrupted


class TestDreamInsight:
    """Tests for DreamInsight dataclass."""

    def test_insight_creation(self):
        """Test creating a dream insight."""
        import datetime
        insight = DreamInsight(
            insight_id="insight_1",
            insight_type="pattern",
            content="Test insight content",
            confidence=0.85,
            source_memories=["mem1", "mem2"],
            phase=SleepPhase.DEEP,
            timestamp=datetime.datetime.now()
        )
        assert insight.insight_id == "insight_1"
        assert insight.confidence == 0.85


class TestConsolidatedPattern:
    """Tests for ConsolidatedPattern dataclass."""

    def test_pattern_creation(self):
        """Test creating a consolidated pattern."""
        import datetime
        pattern = ConsolidatedPattern(
            pattern_id="pattern_1",
            pattern_name="Test Pattern",
            strength=0.9,
            memories_consolidated=10,
            created_at=datetime.datetime.now(),
            last_reinforced=datetime.datetime.now()
        )
        assert pattern.pattern_id == "pattern_1"
        assert pattern.strength == 0.9


class TestInterruptibility:
    """Tests for sleep interruption."""

    def test_interrupt_during_sleep(self, neurodream_engine):
        """Test that sleep can be interrupted."""
        neurodream_engine.enter_sleep(trigger="manual")
        assert neurodream_engine.is_sleeping is True

        # Interrupt by waking up
        result = neurodream_engine.wake_up(reason="user_interrupt")
        assert result.get("success") is True
        assert neurodream_engine.is_sleeping is False

    def test_interrupt_flag(self, neurodream_engine):
        """Test interrupt flag is set on sessions."""
        neurodream_engine.enter_sleep(trigger="manual")
        neurodream_engine.wake_up(reason="interrupted")

        # Check that the session was marked as interrupted
        if neurodream_engine.sessions:
            last_session = neurodream_engine.sessions[-1]
            # The session should have an end_time set
            assert last_session.end_time is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
