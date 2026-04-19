"""
Tests for the SkillEpisodeLog.

All tests use an isolated temp DB so they never touch the real aura_meta.db.
"""

import os
import tempfile
import time

import pytest

from aura.evolution.episode_log import (
    SkillEpisodeLog,
    procedure_hash,
)


@pytest.fixture
def log():
    with tempfile.TemporaryDirectory() as td:
        db_path = os.path.join(td, "episodes_test.db")
        yield SkillEpisodeLog(db_path=db_path)


def test_procedure_hash_is_stable_and_short():
    h1 = procedure_hash("do X then Y")
    h2 = procedure_hash("do X then Y")
    h3 = procedure_hash("do X then Z")
    assert h1 == h2
    assert h1 != h3
    assert len(h1) == 16


def test_log_writes_one_row_per_skill(log):
    ids = log.log(
        request_id="req-1",
        skill_ids=["skill_a", "skill_b"],
        user_input="hello",
        response="world",
        procedures={"skill_a": "proc A", "skill_b": "proc B"},
        tools_called=["search"],
    )
    assert len(ids) == 2
    assert ids[0] != ids[1]


def test_log_handles_missing_request_id(log):
    ids = log.log(
        request_id="",
        skill_ids=["s"],
        user_input="hi",
        response="ok",
        procedures={"s": "proc"},
    )
    assert ids == []


def test_log_handles_empty_skill_list(log):
    ids = log.log(
        request_id="req",
        skill_ids=[],
        user_input="hi",
        response="ok",
        procedures={},
    )
    assert ids == []


def test_episodes_for_request_returns_all_skills(log):
    log.log(
        request_id="req-42",
        skill_ids=["x", "y", "z"],
        user_input="a",
        response="b",
        procedures={"x": "px", "y": "py", "z": "pz"},
    )
    rows = log.episodes_for_request("req-42")
    assert len(rows) == 3
    assert {r.skill_id for r in rows} == {"x", "y", "z"}


def test_add_outcome_attaches_to_episode(log):
    ids = log.log(
        request_id="r",
        skill_ids=["s"],
        user_input="i",
        response="o",
        procedures={"s": "p"},
    )
    assert log.add_outcome(
        episode_id=ids[0],
        signal_kind="reaction",
        score=1.0,
        confidence=0.9,
    )


def test_score_is_clamped_to_01(log):
    ids = log.log(
        request_id="r",
        skill_ids=["s"],
        user_input="i",
        response="o",
        procedures={"s": "p"},
    )
    log.add_outcome(episode_id=ids[0], signal_kind="test", score=5.0, confidence=2.0)
    labeled = log.labeled_episodes_for_skill("s", window_days=30)
    assert len(labeled) == 1
    out = labeled[0].outcomes[0]
    assert out.score == 1.0
    assert out.confidence == 1.0


def test_labeled_episodes_excludes_unlabeled(log):
    ids1 = log.log(
        request_id="r1", skill_ids=["s"], user_input="a", response="b",
        procedures={"s": "p"},
    )
    log.log(
        request_id="r2", skill_ids=["s"], user_input="c", response="d",
        procedures={"s": "p"},
    )
    log.add_outcome(
        episode_id=ids1[0], signal_kind="reaction", score=1.0, confidence=0.9,
    )
    labeled = log.labeled_episodes_for_skill("s", window_days=30)
    assert len(labeled) == 1
    assert labeled[0].episode.request_id == "r1"


def test_composite_score_weights_by_confidence(log):
    ids = log.log(
        request_id="r", skill_ids=["s"], user_input="i", response="o",
        procedures={"s": "p"},
    )
    # Strong positive with high confidence + weak negative with low confidence
    # should lean heavily positive.
    log.add_outcome(episode_id=ids[0], signal_kind="reaction", score=1.0, confidence=0.9)
    log.add_outcome(episode_id=ids[0], signal_kind="action_deeper", score=0.3, confidence=0.1)
    labeled = log.labeled_episodes_for_skill("s", window_days=30)
    composite = labeled[0].composite_score()
    assert composite is not None
    assert composite > 0.85


def test_window_filter_excludes_old_episodes(log):
    ids = log.log(
        request_id="r", skill_ids=["s"], user_input="i", response="o",
        procedures={"s": "p"},
        invoked_at=time.time() - 30 * 86400,  # 30 days old
    )
    log.add_outcome(episode_id=ids[0], signal_kind="reaction", score=1.0, confidence=0.9)
    recent = log.labeled_episodes_for_skill("s", window_days=14)
    assert recent == []
    everything = log.labeled_episodes_for_skill("s", window_days=60)
    assert len(everything) == 1


def test_count_for_skill(log):
    for i in range(3):
        ids = log.log(
            request_id=f"r{i}", skill_ids=["s"],
            user_input="i", response="o",
            procedures={"s": "p"},
        )
        log.add_outcome(
            episode_id=ids[0], signal_kind="reaction", score=0.5, confidence=0.5,
        )
    # One unlabeled
    log.log(
        request_id="r-unlabeled", skill_ids=["s"],
        user_input="i", response="o", procedures={"s": "p"},
    )
    assert log.count_for_skill("s", window_days=30) == 3
