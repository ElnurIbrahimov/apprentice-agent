"""
Tests for OutcomeScorer — the mapping from user signals to episode outcomes.
"""

import os
import tempfile

import pytest

from aura.evolution.episode_log import SkillEpisodeLog
from aura.evolution.outcome_scorer import OutcomeScorer


@pytest.fixture
def log():
    with tempfile.TemporaryDirectory() as td:
        yield SkillEpisodeLog(db_path=os.path.join(td, "scorer_test.db"))


@pytest.fixture
def scorer(log):
    return OutcomeScorer(episode_log=log)


def _seed_episode(log, request_id="r", skill_ids=("s",)):
    return log.log(
        request_id=request_id,
        skill_ids=list(skill_ids),
        user_input="i",
        response="o",
        procedures={sid: "p" for sid in skill_ids},
    )


def test_positive_emoji_maps_to_high_score(scorer, log):
    _seed_episode(log, request_id="r1")
    written = scorer.score_from_reaction(request_id="r1", emoji="👍")
    assert written == 1
    labeled = log.labeled_episodes_for_skill("s", window_days=30)
    assert len(labeled) == 1
    assert labeled[0].outcomes[0].score == 1.0
    assert labeled[0].outcomes[0].signal_kind == "reaction"


def test_negative_emoji_maps_to_zero_score(scorer, log):
    _seed_episode(log, request_id="r2")
    scorer.score_from_reaction(request_id="r2", emoji="👎")
    labeled = log.labeled_episodes_for_skill("s", window_days=30)
    assert labeled[0].outcomes[0].score == 0.0


def test_unknown_emoji_falls_to_weak_neutral(scorer, log):
    _seed_episode(log, request_id="r3")
    scorer.score_from_reaction(request_id="r3", emoji="🧩")
    labeled = log.labeled_episodes_for_skill("s", window_days=30)
    out = labeled[0].outcomes[0]
    assert out.score == 0.5
    assert out.confidence == 0.3


def test_sentiment_string_fallback(scorer, log):
    _seed_episode(log, request_id="r4")
    scorer.score_from_reaction(request_id="r4", sentiment="positive")
    labeled = log.labeled_episodes_for_skill("s", window_days=30)
    assert labeled[0].outcomes[0].score == 1.0


def test_emoji_takes_priority_over_sentiment(scorer, log):
    _seed_episode(log, request_id="r5")
    scorer.score_from_reaction(request_id="r5", sentiment="positive", emoji="👎")
    labeled = log.labeled_episodes_for_skill("s", window_days=30)
    assert labeled[0].outcomes[0].score == 0.0


def test_action_save_scores_strong_positive(scorer, log):
    _seed_episode(log, request_id="rA")
    scorer.score_from_action(request_id="rA", action="save")
    labeled = log.labeled_episodes_for_skill("s", window_days=30)
    out = labeled[0].outcomes[0]
    assert out.score == 0.9
    assert out.signal_kind == "action_save"


def test_action_regenerate_scores_negative(scorer, log):
    _seed_episode(log, request_id="rR")
    scorer.score_from_action(request_id="rR", action="regenerate")
    labeled = log.labeled_episodes_for_skill("s", window_days=30)
    assert labeled[0].outcomes[0].score == 0.25


def test_unknown_action_is_ignored(scorer, log):
    _seed_episode(log, request_id="rX")
    written = scorer.score_from_action(request_id="rX", action="nonsense")
    assert written == 0


def test_explicit_verdict_has_full_confidence(scorer, log):
    _seed_episode(log, request_id="rE")
    scorer.score_from_explicit(request_id="rE", verdict="good")
    labeled = log.labeled_episodes_for_skill("s", window_days=30)
    out = labeled[0].outcomes[0]
    assert out.confidence == 1.0
    assert out.score == 1.0


def test_signal_fans_out_across_all_skills_in_request(scorer, log):
    _seed_episode(log, request_id="multi", skill_ids=("x", "y", "z"))
    written = scorer.score_from_reaction(request_id="multi", emoji="👍")
    assert written == 3


def test_scoring_unknown_request_is_noop(scorer):
    # No episode seeded for this request_id
    written = scorer.score_from_reaction(request_id="ghost", emoji="👍")
    assert written == 0


def test_empty_request_id_scored_as_noop(scorer):
    assert scorer.score_from_reaction(request_id="", emoji="👍") == 0
    assert scorer.score_from_action(request_id="", action="save") == 0
    assert scorer.score_from_explicit(request_id="", verdict="good") == 0
