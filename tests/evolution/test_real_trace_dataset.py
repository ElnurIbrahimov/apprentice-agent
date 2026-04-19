"""
Tests for the real-trace dataset producer.

Covers:
- Real episodes converted to EvalExample objects with correct source tags
- Rubric shape depends on composite outcome score (match / avoid / ambiguous)
- Fallback to synthetic when < min_episodes labeled episodes exist
- load_mixed_dataset composes real + synthetic correctly across skills
"""

import os
import tempfile

import pytest

from aura.evolution.episode_log import SkillEpisodeLog
from aura.evolution.real_trace_dataset import (
    DEFAULT_MIN_EPISODES,
    DatasetMix,
    load_mixed_dataset,
    load_real_episodes,
)
from aura.evolution.types import EvalExample


@pytest.fixture
def log():
    with tempfile.TemporaryDirectory() as td:
        yield SkillEpisodeLog(db_path=os.path.join(td, "dataset_test.db"))


def _seed(log, skill_id, request_id, score, confidence=0.9):
    ids = log.log(
        request_id=request_id,
        skill_ids=[skill_id],
        user_input=f"input for {request_id}",
        response=f"response for {request_id}",
        procedures={skill_id: "p"},
    )
    log.add_outcome(
        episode_id=ids[0],
        signal_kind="reaction",
        score=score,
        confidence=confidence,
    )
    return ids[0]


def test_load_returns_empty_when_below_min_episodes(log):
    _seed(log, "sparse", "r1", 1.0)
    _seed(log, "sparse", "r2", 1.0)
    examples = load_real_episodes("sparse", log=log, min_episodes=5)
    assert examples == []


def test_load_returns_examples_at_or_above_threshold(log):
    for i in range(DEFAULT_MIN_EPISODES):
        _seed(log, "dense", f"r{i}", 1.0)
    examples = load_real_episodes("dense", log=log)
    assert len(examples) >= DEFAULT_MIN_EPISODES
    assert all(isinstance(e, EvalExample) for e in examples)
    assert all(e.source == "real" for e in examples)


def test_high_score_gets_match_rubric(log):
    for i in range(DEFAULT_MIN_EPISODES):
        _seed(log, "good", f"g{i}", 1.0)
    examples = load_real_episodes("good", log=log)
    assert examples
    assert "match" in examples[0].expected_behavior.lower()


def test_low_score_gets_avoid_rubric(log):
    for i in range(DEFAULT_MIN_EPISODES):
        _seed(log, "bad", f"b{i}", 0.0)
    examples = load_real_episodes("bad", log=log)
    assert examples
    assert "avoid" in examples[0].expected_behavior.lower()


def test_mid_score_gets_improvement_rubric(log):
    for i in range(DEFAULT_MIN_EPISODES):
        _seed(log, "mid", f"m{i}", 0.5)
    examples = load_real_episodes("mid", log=log)
    assert examples
    rubric = examples[0].expected_behavior.lower()
    assert "improvement" in rubric or "middling" in rubric


def test_load_respects_limit(log):
    for i in range(15):
        _seed(log, "many", f"r{i}", 1.0)
    examples = load_real_episodes("many", log=log, limit=5)
    assert len(examples) == 5


def test_load_respects_window(log):
    # Seed one recent + one old; only recent should appear
    import time
    ids_old = log.log(
        request_id="old", skill_ids=["w"], user_input="a", response="b",
        procedures={"w": "p"}, invoked_at=time.time() - 30 * 86400,
    )
    log.add_outcome(episode_id=ids_old[0], signal_kind="reaction", score=1.0, confidence=0.9)
    for i in range(DEFAULT_MIN_EPISODES):
        _seed(log, "w", f"recent{i}", 1.0)
    examples = load_real_episodes("w", log=log, window_days=14)
    # Old is outside window, only the 5 recent ones count — all 5 should appear
    assert len(examples) == DEFAULT_MIN_EPISODES


def test_load_mixed_dataset_uses_real_when_available(log):
    for i in range(DEFAULT_MIN_EPISODES):
        _seed(log, "real_skill", f"r{i}", 1.0)

    def synthetic_noop(sid):
        return [EvalExample(id=f"syn_{sid}_0", task_input="t", expected_behavior="e", source="synthetic")]

    examples, mixes = load_mixed_dataset(
        skill_ids=["real_skill"], synthetic_fallback=synthetic_noop, log=log,
    )
    assert all(e.source == "real" for e in examples)
    assert len(mixes) == 1
    assert mixes[0].real_count > 0
    assert mixes[0].synthetic_count == 0
    assert mixes[0].source == "real"


def test_load_mixed_dataset_falls_back_to_synthetic(log):
    # Only 2 episodes — below min_episodes
    _seed(log, "thin", "r1", 1.0)
    _seed(log, "thin", "r2", 1.0)

    synthetic_called = []
    def synthetic_fallback(sid):
        synthetic_called.append(sid)
        return [EvalExample(id=f"syn_{sid}_0", task_input="t", expected_behavior="e", source="synthetic")]

    examples, mixes = load_mixed_dataset(
        skill_ids=["thin"], synthetic_fallback=synthetic_fallback, log=log,
    )
    assert synthetic_called == ["thin"]
    assert all(e.source == "synthetic" for e in examples)
    assert mixes[0].real_count == 0
    assert mixes[0].synthetic_count == 1
    assert mixes[0].source == "synthetic"


def test_load_mixed_dataset_handles_mix(log):
    for i in range(DEFAULT_MIN_EPISODES):
        _seed(log, "rich", f"rich{i}", 1.0)

    def synthetic_fallback(sid):
        return [EvalExample(id=f"syn_{sid}", task_input="t", expected_behavior="e", source="synthetic")]

    examples, mixes = load_mixed_dataset(
        skill_ids=["rich", "poor"],
        synthetic_fallback=synthetic_fallback,
        log=log,
    )
    by_skill = {m.skill_id: m for m in mixes}
    assert by_skill["rich"].source == "real"
    assert by_skill["poor"].source == "synthetic"
    assert any(e.source == "real" for e in examples)
    assert any(e.source == "synthetic" for e in examples)


def test_dataset_mix_source_property():
    assert DatasetMix("s", real_count=5, synthetic_count=0).source == "real"
    assert DatasetMix("s", real_count=0, synthetic_count=5).source == "synthetic"
    assert DatasetMix("s", real_count=3, synthetic_count=2).source == "mixed"
