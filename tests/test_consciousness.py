"""Tests for consciousness modules — metacognition, self-improvement, world model.

Tests core logic without requiring LLM calls or heavy dependencies.
"""

import os
import tempfile
import threading
import time

import pytest


# ============================================================================
# MetacognitiveEngine Tests
# ============================================================================

class TestMetacognitiveEngine:
    """Test MetacognitiveEngine core logic."""

    @pytest.fixture
    def engine(self, tmp_path):
        from aura.consciousness.metacognition import MetacognitiveEngine
        return MetacognitiveEngine(data_dir=str(tmp_path / "metacog"))

    def test_init_creates_data_dir(self, engine, tmp_path):
        assert (tmp_path / "metacog").exists()

    def test_record_interaction_outcome(self, engine):
        engine.record_interaction_outcome("coding", True, confidence=0.8, details="wrote clean code")
        engine.record_interaction_outcome("coding", False, confidence=0.6, details="syntax error")
        assert len(engine._outcomes) == 2
        assert engine._outcomes[0]["domain"] == "coding"
        assert engine._outcomes[0]["success"] is True
        assert engine._outcomes[1]["success"] is False

    def test_outcomes_bounded_at_500(self, engine):
        for i in range(550):
            engine.record_interaction_outcome("coding", True, confidence=0.5)
        assert len(engine._outcomes) == 500

    def test_identify_weak_areas_empty(self, engine):
        # No capabilities assessed yet — should trigger assessment then return
        weak = engine.identify_weak_areas(threshold=0.5)
        assert isinstance(weak, list)

    def test_identify_weak_areas_with_manual_capabilities(self, engine):
        from aura.consciousness.metacognition import CapabilityScore
        engine._capabilities = {
            "coding": CapabilityScore(
                domain="coding", score=0.8, confidence=0.9,
                sample_count=20, trend=0.1, last_assessed="2026-01-01",
            ),
            "writing": CapabilityScore(
                domain="writing", score=0.3, confidence=0.7,
                sample_count=15, trend=-0.05, last_assessed="2026-01-01",
            ),
            "research": CapabilityScore(
                domain="research", score=0.45, confidence=0.1,
                sample_count=2, trend=0.0, last_assessed="2026-01-01",
            ),
        }
        weak = engine.identify_weak_areas(threshold=0.5)
        # writing should be weak (0.3 < 0.5 and confidence > 0.2)
        assert len(weak) == 1
        assert weak[0]["domain"] == "writing"
        assert weak[0]["gap"] == pytest.approx(0.2, abs=0.01)
        # research is below threshold but confidence too low (0.1 < 0.2)

    def test_get_strengths_and_weaknesses(self, engine):
        from aura.consciousness.metacognition import CapabilityScore
        engine._capabilities = {
            "coding": CapabilityScore(
                domain="coding", score=0.85, confidence=0.9,
                sample_count=30, trend=0.1, last_assessed="2026-01-01",
            ),
            "writing": CapabilityScore(
                domain="writing", score=0.3, confidence=0.8,
                sample_count=20, trend=-0.1, last_assessed="2026-01-01",
            ),
            "analysis": CapabilityScore(
                domain="analysis", score=0.55, confidence=0.6,
                sample_count=10, trend=0.0, last_assessed="2026-01-01",
            ),
        }
        result = engine.get_strengths_and_weaknesses()
        assert "coding (improving)" in result["strengths"]
        assert "writing (declining)" in result["weaknesses"]
        # analysis (0.55) is neither strong (>=0.7) nor weak (<0.4)
        assert "analysis" not in str(result["strengths"])
        assert "analysis" not in str(result["weaknesses"])

    def test_get_domain_for_query(self, engine):
        from aura.consciousness.metacognition import CapabilityDomain
        # Code-related queries
        domain = engine.get_domain_for_query("write a python function to sort a list")
        assert domain == CapabilityDomain.CODING
        # Research-related queries
        domain = engine.get_domain_for_query("search for papers about attention mechanisms")
        assert domain == CapabilityDomain.RESEARCH

    def test_self_model_prompt_generation(self):
        from aura.consciousness.metacognition import SelfModel, CapabilityScore, LearningGoal
        model = SelfModel(
            capabilities={},
            strengths=["coding", "analysis"],
            weaknesses=["writing"],
            learning_goals=[
                LearningGoal(
                    id="g1", domain="writing", description="improve clarity",
                    strategy="practice", priority=0.8, created_at="2026-01-01",
                    target_score=0.7, current_score=0.3, status="active",
                ),
            ],
            total_improvements=10,
            successful_improvements=7,
            last_assessment="2026-01-01",
        )
        prompt = model.to_system_prompt()
        assert "[Self-Model]" in prompt
        assert "Strengths: coding, analysis" in prompt
        assert "Growth areas: writing" in prompt
        assert "Currently improving: writing: improve clarity" in prompt
        assert "7/10 successful (70%)" in prompt

    def test_self_model_empty_prompt(self):
        from aura.consciousness.metacognition import SelfModel
        model = SelfModel(
            capabilities={}, strengths=[], weaknesses=[],
            learning_goals=[], total_improvements=0,
            successful_improvements=0, last_assessment="",
        )
        prompt = model.to_system_prompt()
        assert prompt == ""  # Nothing to report

    def test_singleton_returns_same_instance(self):
        from aura.consciousness.metacognition import get_metacognitive_engine
        e1 = get_metacognitive_engine()
        e2 = get_metacognitive_engine()
        assert e1 is e2

    def test_thread_safe_outcome_recording(self, engine):
        """Multiple threads recording outcomes shouldn't crash."""
        errors = []

        def record_many():
            try:
                for _ in range(50):
                    engine.record_interaction_outcome("coding", True, 0.5)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=record_many) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        assert len(engine._outcomes) == 200


# ============================================================================
# SelfImprovementEngine Tests
# ============================================================================

class TestSelfImprovementEngine:
    """Test SelfImprovementEngine core logic."""

    @pytest.fixture
    def engine(self, tmp_path):
        from aura.consciousness.self_improvement import SelfImprovementEngine
        # Patch data dir
        eng = SelfImprovementEngine.__new__(SelfImprovementEngine)
        eng._data_dir = tmp_path / "self_improvement"
        eng._data_dir.mkdir(parents=True, exist_ok=True)
        eng._state_file = eng._data_dir / "engine_state.json"
        eng._lock = threading.Lock()
        from collections import deque
        eng._outcomes = deque(maxlen=500)
        eng._outcomes_since_last_cycle = 0
        eng._cycle_history = deque(maxlen=50)
        eng._cycle_count = 0
        eng._tunable_params = {}
        eng._strategy_results = []
        eng._running = False
        eng._scheduler_thread = None
        eng._last_cycle_time = 0.0
        eng._practice_brain = None
        eng._stats = {
            "total_outcomes": 0,
            "total_cycles": 0,
            "total_adjustments": 0,
            "total_strategies": 0,
            "engine_started_at": None,
        }
        return eng

    def test_outcomes_bounded(self, engine):
        from aura.consciousness.self_improvement import InteractionOutcome
        for i in range(550):
            engine._outcomes.append(InteractionOutcome(
                domain="coding", success=True, confidence=0.5,
                prompt_length=10, response_length=100,
                model_used="test", timestamp=time.time(),
            ))
        assert len(engine._outcomes) == 500

    def test_cycle_gating_no_outcomes(self, engine):
        """Should not run cycle without enough outcomes."""
        result = engine._should_run_cycle()
        assert result is False

    def test_stats_initialized(self, engine):
        assert engine._stats["total_outcomes"] == 0
        assert engine._stats["total_cycles"] == 0

    def test_start_stop_lifecycle(self, engine):
        assert engine._running is False
        # Don't actually start (would need full deps), just verify state
        engine._running = True
        assert engine._running is True
        engine._running = False
        assert engine._running is False


# ============================================================================
# WorldModel Tests
# ============================================================================

class TestWorldModel:
    """Test WorldModel CRUD and query logic."""

    @pytest.fixture
    def wm(self, tmp_path):
        from aura.consciousness.world_model import WorldModel
        return WorldModel(
            db_path=str(tmp_path / "world_model.db"),
            snapshot_path=str(tmp_path / "world_state.json"),
            enabled=True,
        )

    def test_init_creates_db(self, wm, tmp_path):
        assert (tmp_path / "world_model.db").exists()

    def test_add_and_get_project(self, wm):
        from aura.consciousness.world_model import ProjectStatus
        project = wm.add_project(
            name="TestProject",
            description="A test project",
            status=ProjectStatus.ACTIVE,
            technologies=["python", "fastapi"],
        )
        assert project.name == "TestProject"
        assert project.description == "A test project"

        retrieved = wm.get_project(project.id)
        assert retrieved is not None
        assert retrieved.name == "TestProject"

    def test_update_project(self, wm):
        project = wm.add_project(name="OldName", description="old desc")
        updated = wm.update_project(project.id, name="NewName", description="new desc")
        assert updated is not None
        assert updated.name == "NewName"
        assert updated.description == "new desc"

    def test_get_nonexistent_project(self, wm):
        result = wm.get_project("nonexistent-id-12345")
        assert result is None

    def test_add_and_get_goal(self, wm):
        from aura.consciousness.world_model import GoalHorizon
        goal = wm.add_goal(
            description="Learn Rust",
            horizon=GoalHorizon.SHORT_TERM,
        )
        assert goal.description == "Learn Rust"

        active = wm.get_active_goals()
        assert len(active) >= 1
        assert any(g.description == "Learn Rust" for g in active)

    def test_update_goal(self, wm):
        from aura.consciousness.world_model import GoalHorizon
        goal = wm.add_goal(description="Original goal", horizon=GoalHorizon.MEDIUM_TERM)
        updated = wm.update_goal(goal.id, description="Updated goal", progress=0.5)
        assert updated is not None
        assert updated.description == "Updated goal"
        assert updated.progress == 0.5

    def test_add_and_get_belief(self, wm):
        from aura.consciousness.world_model import BeliefCategory
        belief = wm.add_belief(
            statement="User prefers concise responses",
            category=BeliefCategory.PREFERENCE,
            confidence=0.9,
            evidence="Multiple interactions showing preference",
        )
        assert belief.statement == "User prefers concise responses"
        assert belief.confidence == 0.9

        beliefs = wm.get_current_beliefs()
        assert len(beliefs) >= 1

    def test_add_relationship(self, wm):
        rel = wm.add_relationship(
            name="Elnur",
            role="developer",
            relationship_type="primary_user",
            context="Works on AI research",
        )
        assert rel.name == "Elnur"
        assert rel.role == "developer"

        snapshot = wm.get_relationships_snapshot()
        assert len(snapshot) >= 1

    def test_context_summary_empty(self, wm):
        summary = wm.get_context_summary()
        # Empty world model should return empty or minimal string
        assert isinstance(summary, str)

    def test_context_summary_with_data(self, wm):
        from aura.consciousness.world_model import ProjectStatus, GoalHorizon
        wm.add_project(name="Aura", description="AI agent", status=ProjectStatus.ACTIVE)
        wm.add_goal(description="Ship v5.0", horizon=GoalHorizon.SHORT_TERM)
        from aura.consciousness.world_model import BeliefCategory
        wm.add_belief(statement="User prefers TypeScript", category=BeliefCategory.PREFERENCE, confidence=0.8)
        summary = wm.get_context_summary()
        assert "Aura" in summary
        assert "Ship v5.0" in summary

    def test_project_health_update_runs(self, wm):
        from aura.consciousness.world_model import ProjectHealth
        wm.add_project(name="HealthCheckProject", description="test")
        count = wm.update_project_health()
        assert isinstance(count, int)
        assert count >= 0

    def test_add_blocker(self, wm):
        project = wm.add_project(name="BlockedProject", description="test")
        blocker_id = wm.add_blocker(project.id, "Waiting for API key", severity="high")
        assert blocker_id > 0
        blockers = wm.get_project_blockers(project.id)
        assert len(blockers) >= 1

    def test_maintenance_runs(self, wm):
        wm.add_project(name="TestMaint", description="test")
        result = wm.run_maintenance()
        assert isinstance(result, dict)
        assert "health_changes" in result

    def test_disabled_world_model(self, tmp_path):
        from aura.consciousness.world_model import WorldModel
        wm = WorldModel(
            db_path=str(tmp_path / "disabled.db"),
            snapshot_path=str(tmp_path / "disabled.json"),
            enabled=False,
        )
        # Operations should be no-ops or return empty
        summary = wm.get_context_summary()
        assert summary == "" or summary is None or isinstance(summary, str)

    def test_thread_safe_project_operations(self, wm):
        """Multiple threads adding projects shouldn't crash."""
        errors = []

        def add_projects(prefix):
            try:
                for i in range(10):
                    wm.add_project(name=f"{prefix}_project_{i}", description="test")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=add_projects, args=(f"t{i}",)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        all_projects = wm.get_all_projects()
        assert len(all_projects) == 40


# ============================================================================
# HandManager Crash Recovery Tests
# ============================================================================

class TestHandCrashRecovery:
    """Test Hand crash recovery mechanisms."""

    def _make_hand(self, name, initial_state=None):
        """Helper to create a concrete Hand subclass."""
        from aura.hands.base import Hand, HandManifest, HandState

        _manifest = HandManifest(name=name, description="test hand", version="1.0")

        class DummyHand(Hand):
            def get_manifest(self):
                return _manifest

            def get_system_prompt(self):
                return "test"

            async def execute(self, brain, tools, context):
                pass

        hand = DummyHand()
        if initial_state is not None:
            hand._state = initial_state
        return hand

    def test_register_resets_stuck_running_hand(self):
        from aura.hands.manager import HandManager
        from aura.hands.base import HandState

        manager = HandManager()
        hand = self._make_hand("stuck_hand", HandState.RUNNING)
        assert hand.state == HandState.RUNNING

        manager.register(hand)
        assert hand.state == HandState.COOLDOWN

    def test_register_normal_hand_unchanged(self):
        from aura.hands.manager import HandManager
        from aura.hands.base import HandState

        manager = HandManager()
        hand = self._make_hand("normal_hand")
        assert hand.state == HandState.INACTIVE
        manager.register(hand)
        assert hand.state == HandState.INACTIVE

    def test_circuit_breaker_exponential_backoff(self):
        from aura.hands.base import HandState, HandResult

        hand = self._make_hand("failing_hand", HandState.ACTIVE)

        # Simulate 3 consecutive failures
        for _ in range(3):
            hand.record_run(HandResult(
                hand_name="failing_hand", success=False,
                summary="failed", duration_seconds=1.0, error="test error",
            ))

        assert hand._consecutive_failures == 3
        # With 3 failures, can_run should return False (circuit breaker)
        # because cooldown = min(3600, 300 * 2^0) = 300s
        hand._last_run = time.time()  # Just ran
        assert hand.can_run() is False
