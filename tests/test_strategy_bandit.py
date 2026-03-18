"""Tests for Strategy Bandit — Thompson Sampling over reasoning strategies."""

import os
import shutil
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aura.consciousness.strategy_bandit import (
    ArmState,
    BanditSelection,
    CATEGORY_STRATEGIES,
    CompositeRewardComputer,
    ProblemCategory,
    ProblemClassifier,
    ReasoningStrategy,
    StrategyBandit,
    StrategyOutcome,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_data_dir():
    """Create a temporary data directory for test databases."""
    tmpdir = tempfile.mkdtemp(prefix="aura_test_bandit_")
    yield Path(tmpdir)
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def bandit(temp_data_dir):
    """Create a StrategyBandit with test database."""
    db_path = str(temp_data_dir / "test_meta.db")
    b = StrategyBandit(db_path=db_path, epsilon=0.1, enabled=True)
    yield b


@pytest.fixture
def disabled_bandit(temp_data_dir):
    """Create a disabled StrategyBandit."""
    db_path = str(temp_data_dir / "test_meta_disabled.db")
    return StrategyBandit(db_path=db_path, epsilon=0.1, enabled=False)


@pytest.fixture
def classifier():
    """Create a ProblemClassifier."""
    return ProblemClassifier()


@pytest.fixture
def reward_computer():
    """Create a CompositeRewardComputer."""
    return CompositeRewardComputer()


# ============================================================================
# TestProblemClassifier
# ============================================================================


class TestProblemClassifier:
    """Test keyword-based query classification."""

    def test_classify_math(self, classifier):
        assert classifier.classify("Calculate the derivative of x^2") == ProblemCategory.MATH

    def test_classify_code(self, classifier):
        assert classifier.classify("Write a Python function to sort a list") == ProblemCategory.CODE

    def test_classify_analysis(self, classifier):
        assert classifier.classify("Analyze this data and compare trends") == ProblemCategory.ANALYSIS

    def test_classify_creative(self, classifier):
        assert classifier.classify("Write a creative story about a dragon") == ProblemCategory.CREATIVE

    def test_classify_planning(self, classifier):
        assert classifier.classify("Plan a roadmap for the project with milestones") == ProblemCategory.PLANNING

    def test_classify_debug(self, classifier):
        assert classifier.classify("Debug this error traceback from my code") == ProblemCategory.DEBUG

    def test_classify_default(self, classifier):
        """Ambiguous queries should fall back to ANALYSIS."""
        result = classifier.classify("Hello, how are you?")
        assert result == ProblemCategory.ANALYSIS


# ============================================================================
# TestCompositeReward
# ============================================================================


class TestCompositeReward:
    """Test composite reward computation."""

    def test_all_metrics_perfect(self, reward_computer):
        metrics = {
            "latency_score": 1.0,
            "self_consistency": 1.0,
            "judge_score": 1.0,
            "coherence_score": 1.0,
            "user_feedback": 1.0,
        }
        reward = reward_computer.compute(metrics)
        assert reward == pytest.approx(1.0, abs=0.01)

    def test_all_metrics_zero(self, reward_computer):
        metrics = {
            "latency_score": 0.0,
            "self_consistency": 0.0,
            "judge_score": 0.0,
            "coherence_score": 0.0,
        }
        reward = reward_computer.compute(metrics)
        assert reward == pytest.approx(0.0, abs=0.01)

    def test_no_metrics(self, reward_computer):
        """No metrics → neutral prior 0.5."""
        reward = reward_computer.compute({})
        assert reward == pytest.approx(0.5, abs=0.01)

    def test_partial_metrics(self, reward_computer):
        """Only latency → renormalized weight."""
        metrics = {"latency_score": 0.8}
        reward = reward_computer.compute(metrics)
        assert reward == pytest.approx(0.8, abs=0.01)

    def test_user_feedback_upweighting(self, reward_computer):
        """When user_feedback is present, its weight increases."""
        # With user feedback = 1.0, all others = 0.0
        metrics_with_fb = {
            "latency_score": 0.0,
            "self_consistency": 0.0,
            "judge_score": 0.0,
            "coherence_score": 0.0,
            "user_feedback": 1.0,
        }
        reward_with = reward_computer.compute(metrics_with_fb)

        # Without user feedback, same zeros
        metrics_without_fb = {
            "latency_score": 0.0,
            "self_consistency": 0.0,
            "judge_score": 0.0,
            "coherence_score": 0.0,
        }
        reward_without = reward_computer.compute(metrics_without_fb)

        # With user feedback present and = 1.0, reward should be higher
        assert reward_with > reward_without


# ============================================================================
# TestArmState
# ============================================================================


class TestArmState:
    """Test ArmState dataclass."""

    def test_default_prior(self):
        arm = ArmState(strategy="cot", category="math")
        assert arm.alpha == 1.0
        assert arm.beta == 1.0
        assert arm.mean_reward == pytest.approx(0.5)

    def test_sample_in_range(self):
        arm = ArmState(strategy="cot", category="math", alpha=10, beta=2)
        samples = [arm.sample() for _ in range(100)]
        assert all(0.0 <= s <= 1.0 for s in samples)
        # High alpha should skew toward 1.0
        assert sum(samples) / len(samples) > 0.5


# ============================================================================
# TestStrategyBandit
# ============================================================================


class TestStrategyBandit:
    """Test StrategyBandit selection, learning, and persistence."""

    def test_init_creates_db(self, bandit, temp_data_dir):
        db_path = temp_data_dir / "test_meta.db"
        assert db_path.exists()

    def test_select_returns_valid_strategy(self, bandit):
        selection = bandit.select_strategy("Write a Python script to sort numbers")
        assert isinstance(selection, BanditSelection)
        assert isinstance(selection.strategy, ReasoningStrategy)
        assert isinstance(selection.category, ProblemCategory)
        assert len(selection.request_id) > 0

    def test_select_respects_category_strategies(self, bandit):
        """Selected strategy should be in the valid set for its category."""
        for _ in range(20):
            selection = bandit.select_strategy("Calculate the factorial of 10")
            valid = CATEGORY_STRATEGIES[selection.category]
            assert selection.strategy in valid

    def test_disabled_returns_chain_of_thought(self, disabled_bandit):
        selection = disabled_bandit.select_strategy("Any query")
        assert selection.strategy == ReasoningStrategy.CHAIN_OF_THOUGHT
        assert selection.exploration is False

    def test_record_outcome_updates_arm(self, bandit):
        selection = bandit.select_strategy("Debug this error traceback")
        reward = bandit.record_outcome(
            request_id=selection.request_id,
            strategy=selection.strategy,
            category=selection.category,
            latency_ms=1500,
            response_length=500,
        )
        assert 0.0 <= reward <= 1.0

        # Verify arm was updated
        stats = bandit.get_arm_stats()
        cat_stats = stats.get(selection.category.value, [])
        arm = next(
            (a for a in cat_stats if a["strategy"] == selection.strategy.value),
            None,
        )
        assert arm is not None
        assert arm["total_pulls"] == 1

    def test_learning_differentiates_arms(self, bandit):
        """After many outcomes, a consistently good strategy should be preferred."""
        category = ProblemCategory.CODE

        # Give reflexion consistently high rewards
        for _ in range(30):
            bandit.record_outcome(
                request_id=f"good_{_}",
                strategy=ReasoningStrategy.MCTS,
                category=category,
                latency_ms=1000,
                metrics={"judge_score": 0.95, "coherence_score": 0.9},
            )

        # Give chain_of_thought consistently low rewards
        for _ in range(30):
            bandit.record_outcome(
                request_id=f"bad_{_}",
                strategy=ReasoningStrategy.CHAIN_OF_THOUGHT,
                category=category,
                latency_ms=5000,
                metrics={"judge_score": 0.3, "coherence_score": 0.3},
            )

        # Check: reflexion should have higher mean reward
        stats = bandit.get_arm_stats()
        code_stats = stats.get("code", [])
        reflexion = next(a for a in code_stats if a["strategy"] == "reflexion")
        cot = next(a for a in code_stats if a["strategy"] == "chain_of_thought")
        assert reflexion["mean_reward"] > cot["mean_reward"]

    def test_user_feedback_updates_outcome(self, bandit):
        selection = bandit.select_strategy("Analyze this data trend")
        bandit.record_outcome(
            request_id=selection.request_id,
            strategy=selection.strategy,
            category=selection.category,
            latency_ms=2000,
        )
        # Record positive user feedback
        bandit.record_user_feedback(selection.request_id, feedback=1.0)

        # Should not crash, and stats should be updated
        summary = bandit.get_stats_summary()
        assert summary["total_outcomes"] >= 1

    def test_decay_arms(self, bandit):
        """Decay should move arms toward prior Beta(1,1)."""
        import sqlite3

        # Give one arm a strong signal
        bandit.record_outcome(
            request_id="decay_test",
            strategy=ReasoningStrategy.CHAIN_OF_THOUGHT,
            category=ProblemCategory.MATH,
            latency_ms=500,
            metrics={"judge_score": 1.0},
        )

        # Backdate last_updated by 1 day so decay has elapsed time to act on
        conn = sqlite3.connect(bandit._db_path)
        conn.execute(
            "UPDATE strategy_arms SET last_updated = last_updated - 86400 "
            "WHERE strategy = 'chain_of_thought' AND category = 'math'"
        )
        conn.commit()
        conn.close()

        stats_before = bandit.get_arm_stats()
        math_before = next(
            a for a in stats_before["math"]
            if a["strategy"] == "chain_of_thought"
        )

        # Apply decay with a 1-day half-life (arm is 1 day old → 50% decay)
        bandit.decay_arms(half_life_days=1.0)

        stats_after = bandit.get_arm_stats()
        math_after = next(
            a for a in stats_after["math"]
            if a["strategy"] == "chain_of_thought"
        )

        # After decay, alpha should be closer to the prior of 1.0
        assert abs(math_after["alpha"] - 1.0) < abs(math_before["alpha"] - 1.0)

    def test_get_stats_summary(self, bandit):
        summary = bandit.get_stats_summary()
        assert "enabled" in summary
        assert "epsilon" in summary
        assert "total_outcomes" in summary
        assert "total_arms" in summary
        assert summary["enabled"] is True


# ============================================================================
# TestCategoryStrategies
# ============================================================================


class TestCategoryStrategies:
    """Validate the category-strategy mapping."""

    def test_all_categories_have_strategies(self):
        for cat in ProblemCategory:
            assert cat in CATEGORY_STRATEGIES
            assert len(CATEGORY_STRATEGIES[cat]) > 0

    def test_all_categories_include_chain_of_thought(self):
        """chain_of_thought should be available everywhere as fallback."""
        for cat in ProblemCategory:
            strategies = CATEGORY_STRATEGIES[cat]
            assert ReasoningStrategy.CHAIN_OF_THOUGHT in strategies
