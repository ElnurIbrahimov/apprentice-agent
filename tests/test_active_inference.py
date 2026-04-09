"""Tests for Active Inference Engine — pymdp integration (Phase 6A)."""

import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch

from aura.proactive.active_inference import (
    ActiveInferenceEngine,
    SimplifiedActiveInference,
    BeliefState,
    ProactiveAction,
    ProactiveDecision,
    PYMDP_AVAILABLE,
)


# ── TestBeliefState ──────────────────────────────────────────────────


class TestBeliefState:
    """Tests for BeliefState dataclass."""

    def test_default_values(self):
        bs = BeliefState()
        assert bs.user_busy == 0.5
        assert bs.user_receptive == 0.5
        assert bs.task_urgent == 0.0
        assert bs.context_stable == 0.5
        assert bs.uncertainty == 0.5

    def test_to_array(self):
        bs = BeliefState(user_busy=0.1, user_receptive=0.2, task_urgent=0.3,
                         context_stable=0.4, uncertainty=0.5)
        arr = bs.to_array()
        assert arr.shape == (5,)
        np.testing.assert_allclose(arr, [0.1, 0.2, 0.3, 0.4, 0.5])

    def test_from_array(self):
        arr = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        bs = BeliefState.from_array(arr)
        assert bs.user_busy == pytest.approx(0.1)
        assert bs.user_receptive == pytest.approx(0.2)
        assert bs.task_urgent == pytest.approx(0.3)
        assert bs.context_stable == pytest.approx(0.4)
        assert bs.uncertainty == pytest.approx(0.5)

    def test_roundtrip(self):
        original = BeliefState(user_busy=0.8, user_receptive=0.3,
                               task_urgent=0.9, context_stable=0.1,
                               uncertainty=0.7)
        restored = BeliefState.from_array(original.to_array())
        assert restored.user_busy == pytest.approx(original.user_busy)
        assert restored.user_receptive == pytest.approx(original.user_receptive)
        assert restored.task_urgent == pytest.approx(original.task_urgent)
        assert restored.context_stable == pytest.approx(original.context_stable)
        assert restored.uncertainty == pytest.approx(original.uncertainty)


# ── TestSimplifiedActiveInference ────────────────────────────────────


class TestSimplifiedActiveInference:
    """Tests for SimplifiedActiveInference engine."""

    def test_init(self):
        engine = SimplifiedActiveInference()
        assert isinstance(engine.beliefs, BeliefState)
        assert engine.action_history == []
        assert engine.last_action_time is None

    def test_update_beliefs(self):
        engine = SimplifiedActiveInference()
        obs = {"user_activity": 0.9, "urgent_events": 0.8}
        beliefs = engine.update_beliefs(obs)
        assert beliefs.user_busy > 0.5  # Should increase toward 0.9
        assert beliefs.task_urgent > 0.0  # Should increase toward 0.8

    def test_select_action_default(self):
        engine = SimplifiedActiveInference()
        decision = engine.select_action()
        assert isinstance(decision, ProactiveDecision)
        assert isinstance(decision.action, ProactiveAction)
        assert 0.0 <= decision.confidence <= 1.0

    def test_urgent_leads_to_notify_or_intervene(self):
        engine = SimplifiedActiveInference()
        engine.beliefs.task_urgent = 0.9
        engine.beliefs.user_busy = 0.2
        engine.beliefs.user_receptive = 0.8
        decision = engine.select_action()
        assert decision.action in (
            ProactiveAction.NOTIFY,
            ProactiveAction.INTERVENE,
            ProactiveAction.REMIND,
        )

    def test_high_uncertainty_leads_to_ask(self):
        engine = SimplifiedActiveInference()
        engine.beliefs.uncertainty = 0.9
        engine.beliefs.user_busy = 0.2
        engine.beliefs.user_receptive = 0.7
        decision = engine.select_action()
        assert decision.action == ProactiveAction.ASK

    def test_cooldowns_enforced(self):
        engine = SimplifiedActiveInference()
        engine.beliefs.task_urgent = 0.9
        # Take an action
        d1 = engine.select_action()
        action_taken = d1.action
        if action_taken != ProactiveAction.WAIT:
            # Immediately, same action should be on cooldown
            assert not engine._can_take_action(action_taken)

    def test_drift_beliefs(self):
        engine = SimplifiedActiveInference()
        engine.beliefs.user_busy = 0.8
        engine.beliefs.user_receptive = 0.3
        for _ in range(50):
            engine.drift_beliefs_toward_idle(drift_rate=0.05)
        # After drift, user_busy should decrease, receptive should increase
        assert engine.beliefs.user_busy < 0.6
        assert engine.beliefs.user_receptive > 0.5

    def test_should_act(self):
        engine = SimplifiedActiveInference()
        should, reason = engine.should_act_proactively()
        assert isinstance(should, bool)
        assert isinstance(reason, str)

    def test_restore_action_history(self):
        engine = SimplifiedActiveInference()
        history = [
            ("notify", datetime.now() - timedelta(minutes=10)),
            ("suggest", datetime.now() - timedelta(minutes=5)),
        ]
        engine.restore_action_history(history)
        assert len(engine.action_history) == 2
        assert engine.last_action_time is not None

    def test_all_actions_on_cooldown(self):
        engine = SimplifiedActiveInference()
        # Set all non-WAIT actions as recently taken
        now = datetime.now()
        for action in ProactiveAction:
            if action != ProactiveAction.WAIT:
                engine._last_action_times[action] = now
        decision = engine.select_action()
        # WAIT has 0 cooldown so it's always available — should be chosen
        assert decision.action == ProactiveAction.WAIT


# ── TestActiveInferenceEngine (simplified path) ─────────────────────


class TestActiveInferenceEngine:
    """Tests for ActiveInferenceEngine via simplified path."""

    def test_init_without_pymdp(self):
        engine = ActiveInferenceEngine(use_pymdp=False)
        assert engine.use_pymdp is False
        assert isinstance(engine._simple_engine, SimplifiedActiveInference)

    def test_update_beliefs(self):
        engine = ActiveInferenceEngine(use_pymdp=False)
        obs = {"user_activity": 0.8, "urgent_events": 0.5}
        beliefs = engine.update_beliefs(obs)
        assert isinstance(beliefs, BeliefState)
        assert beliefs.user_busy > 0.5

    def test_select_action(self):
        engine = ActiveInferenceEngine(use_pymdp=False)
        decision = engine.select_action()
        assert isinstance(decision, ProactiveDecision)

    def test_get_beliefs(self):
        engine = ActiveInferenceEngine(use_pymdp=False)
        beliefs = engine.get_beliefs()
        assert isinstance(beliefs, BeliefState)

    def test_restore_beliefs(self):
        engine = ActiveInferenceEngine(use_pymdp=False)
        custom = BeliefState(user_busy=0.9, task_urgent=0.8)
        engine.restore_beliefs(custom)
        assert engine.get_beliefs().user_busy == 0.9
        assert engine.get_beliefs().task_urgent == 0.8

    def test_set_preferences_simplified(self):
        engine = ActiveInferenceEngine(use_pymdp=False)
        engine.set_intrinsic_preferences({"curiosity": 1.0})
        assert "curiosity" in engine._simple_engine.preferences

    def test_should_act(self):
        engine = ActiveInferenceEngine(use_pymdp=False)
        should, reason = engine.should_act_proactively()
        assert isinstance(should, bool)
        assert isinstance(reason, str)

    def test_drift(self):
        engine = ActiveInferenceEngine(use_pymdp=False)
        engine._simple_engine.beliefs.user_busy = 0.9
        engine.drift_beliefs_toward_idle(drift_rate=0.1)
        assert engine.get_beliefs().user_busy < 0.9


# ── TestActiveInferencePyMDP ────────────────────────────────────────


@pytest.mark.skipif(not PYMDP_AVAILABLE, reason="pymdp not installed")
class TestActiveInferencePyMDP:
    """Tests for ActiveInferenceEngine with pymdp enabled."""

    def test_init_with_pymdp(self):
        engine = ActiveInferenceEngine(use_pymdp=True)
        assert engine.use_pymdp is True
        assert hasattr(engine, '_pymdp_agent')

    def test_has_base_c(self):
        engine = ActiveInferenceEngine(use_pymdp=True)
        assert hasattr(engine, '_base_C')
        assert engine._base_C is not None
        assert len(engine._base_C) == 4

    def test_belief_sync_after_update(self):
        engine = ActiveInferenceEngine(use_pymdp=True)
        obs = {"user_activity": 0.9, "urgent_events": 0.8,
               "emotional_valence": 0.5, "context_changes": 0.1}
        beliefs = engine.update_beliefs(obs)
        # After pymdp update + sync, beliefs should be within valid range
        assert 0.0 <= beliefs.user_busy <= 1.0
        assert 0.0 <= beliefs.user_receptive <= 1.0
        assert 0.0 <= beliefs.task_urgent <= 1.0
        assert 0.0 <= beliefs.context_stable <= 1.0
        assert 0.0 <= beliefs.uncertainty <= 1.0

    def test_pymdp_reasoning_prefix(self):
        engine = ActiveInferenceEngine(use_pymdp=True)
        # Need to feed observations first so _last_obs is set
        engine.update_beliefs({
            "user_activity": 0.5, "urgent_events": 0.5,
            "emotional_valence": 0.5, "context_changes": 0.5,
        })
        decision = engine.select_action()
        assert "[pymdp]" in decision.reasoning

    def test_c_vector_no_drift(self):
        engine = ActiveInferenceEngine(use_pymdp=True)
        # Record original C values
        engine._pymdp_agent.C[0].copy()

        # Set preferences multiple times
        for _ in range(10):
            engine.set_intrinsic_preferences({"curiosity": 1.5, "social": 0.8})

        # C should NOT have accumulated — it should be base + single offset
        expected_c0 = engine._base_C[0].copy()
        expected_c0[1] += 0.8 * 0.2
        expected_c0[2] += 0.8 * 0.3
        np.testing.assert_allclose(engine._pymdp_agent.C[0], expected_c0, atol=1e-10)

    def test_idle_drift_feeds_pymdp(self):
        engine = ActiveInferenceEngine(use_pymdp=True)
        engine.update_beliefs({
            "user_activity": 0.9, "urgent_events": 0.8,
            "emotional_valence": 0.5, "context_changes": 0.5,
        })
        busy_before = engine.get_beliefs().user_busy
        for _ in range(20):
            engine.drift_beliefs_toward_idle(drift_rate=0.05)
        busy_after = engine.get_beliefs().user_busy
        # After idle drift, user_busy should decrease
        assert busy_after < busy_before

    def test_should_act_uses_pymdp(self):
        engine = ActiveInferenceEngine(use_pymdp=True)
        engine.update_beliefs({
            "user_activity": 0.2, "urgent_events": 0.9,
            "emotional_valence": 0.5, "context_changes": 0.1,
        })
        should, reason = engine.should_act_proactively()
        assert isinstance(should, bool)
        assert "[pymdp]" in reason

    def test_record_outcome(self):
        engine = ActiveInferenceEngine(use_pymdp=True)
        # Full cycle: update beliefs → select action (triggers step_time)
        engine.update_beliefs({
            "user_activity": 0.5, "urgent_events": 0.5,
            "emotional_valence": 0.5, "context_changes": 0.5,
        })
        engine.select_action()  # Runs infer_policies + sample_action + step_time
        initial_steps = engine._learning_steps
        engine.record_outcome({
            "user_activity": 0.8, "urgent_events": 0.3,
            "emotional_valence": 0.7, "context_changes": 0.1,
        })
        # update_A exists and pA is initialized, so learning_steps should increase
        assert engine._learning_steps == initial_steps + 1

    def test_state_persistence_roundtrip(self):
        engine = ActiveInferenceEngine(use_pymdp=True)
        # Do some updates to create learned state
        engine.update_beliefs({
            "user_activity": 0.9, "urgent_events": 0.1,
            "emotional_valence": 0.8, "context_changes": 0.2,
        })
        engine._learning_steps = 42

        # Serialize
        state = engine.get_pymdp_state()
        assert state is not None
        assert state["learning_steps"] == 42

        # Create new engine and restore
        engine2 = ActiveInferenceEngine(use_pymdp=True)
        engine2.restore_pymdp_state(state)
        assert engine2._learning_steps == 42

    def test_get_pymdp_state_returns_none_when_disabled(self):
        engine = ActiveInferenceEngine(use_pymdp=False)
        assert engine.get_pymdp_state() is None

    def test_record_outcome_noop_when_disabled(self):
        engine = ActiveInferenceEngine(use_pymdp=False)
        # Should not raise
        engine.record_outcome({"user_activity": 0.5})

    def test_multiple_update_cycles(self):
        """Test that multiple update/select/step cycles work without error."""
        engine = ActiveInferenceEngine(use_pymdp=True)
        for i in range(5):
            engine.update_beliefs({
                "user_activity": 0.3 + i * 0.1,
                "urgent_events": 0.1 * i,
                "emotional_valence": 0.5,
                "context_changes": 0.1 * i,
            })
            decision = engine.select_action()
            assert isinstance(decision, ProactiveDecision)
            engine.drift_beliefs_toward_idle(drift_rate=0.01)
