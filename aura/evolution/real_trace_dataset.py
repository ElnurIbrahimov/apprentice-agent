"""
Real-Trace Dataset — turns labeled skill episodes into EvalExample objects.

Replaces the LLM-vs-LLM echo chamber where `generate_eval_dataset()` asked an
LLM to invent tasks from a skill's own procedure and grade against rubrics a
second LLM wrote. Here, the "tasks" are real user inputs, and the "rubric" is
derived from the outcome signals the user already gave (reactions, buttons,
explicit verdicts).

A skill must have at least `min_episodes` labeled episodes in the window for
real-trace eval to fire; below that, callers fall back to synthetic.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

from .episode_log import LabeledEpisode, SkillEpisodeLog, get_episode_log
from .types import EvalExample

logger = logging.getLogger(__name__)

# When too few labeled episodes exist, skip real-trace and let caller fall back.
DEFAULT_MIN_EPISODES = 5
DEFAULT_WINDOW_DAYS = 14
DEFAULT_EPISODES_PER_SKILL = 10


@dataclass
class DatasetMix:
    """Reports how a skill's eval dataset was assembled. Useful in result.json."""
    skill_id: str
    real_count: int
    synthetic_count: int

    @property
    def source(self) -> str:
        if self.real_count > 0 and self.synthetic_count == 0:
            return "real"
        if self.real_count == 0:
            return "synthetic"
        return "mixed"


def _episode_to_eval_example(labeled: LabeledEpisode) -> Optional[EvalExample]:
    """Convert one labeled episode to an EvalExample.

    The expected_behavior rubric is shaped by the episode's composite score:
      score ≥ 0.7  → "match the successful pattern shown in this exemplar"
      score ≤ 0.3  → "avoid the failure pattern shown here"
      mid          → weaker signal, still include but mark ambiguous

    Returns None if the episode has no usable signal.
    """
    score = labeled.composite_score()
    if score is None:
        return None

    ep = labeled.episode
    if score >= 0.7:
        rubric = (
            f"Produce a response that matches the quality of this successful "
            f"exemplar: {ep.response[:400]}"
        )
    elif score <= 0.3:
        rubric = (
            f"Avoid producing a response like this one, which the user rejected: "
            f"{ep.response[:400]}"
        )
    else:
        rubric = (
            f"Aim for an improvement over this middling exemplar: {ep.response[:400]}"
        )

    return EvalExample(
        id=f"real_{ep.episode_id}",
        task_input=ep.user_input,
        expected_behavior=rubric,
        source="real",
    )


def load_real_episodes(
    skill_id: str,
    *,
    log: Optional[SkillEpisodeLog] = None,
    window_days: int = DEFAULT_WINDOW_DAYS,
    min_episodes: int = DEFAULT_MIN_EPISODES,
    limit: int = DEFAULT_EPISODES_PER_SKILL,
) -> List[EvalExample]:
    """Produce eval examples from labeled live episodes for one skill.

    Returns an empty list when fewer than `min_episodes` labeled episodes exist
    — the caller treats that as "fall back to synthetic".
    """
    log = log or get_episode_log()

    labeled = log.labeled_episodes_for_skill(
        skill_id=skill_id,
        window_days=window_days,
        limit=max(limit * 2, min_episodes * 2),  # oversample; some may be Nones
    )

    if len(labeled) < min_episodes:
        logger.info(
            "skill '%s' has %d labeled episodes (< %d) — falling back to synthetic",
            skill_id, len(labeled), min_episodes,
        )
        return []

    examples: List[EvalExample] = []
    for le in labeled:
        ex = _episode_to_eval_example(le)
        if ex is not None:
            examples.append(ex)
        if len(examples) >= limit:
            break

    logger.info(
        "skill '%s' contributed %d real-trace eval examples",
        skill_id, len(examples),
    )
    return examples


def load_mixed_dataset(
    skill_ids: List[str],
    *,
    synthetic_fallback,  # Callable[[str], List[EvalExample]]
    log: Optional[SkillEpisodeLog] = None,
    window_days: int = DEFAULT_WINDOW_DAYS,
    min_episodes: int = DEFAULT_MIN_EPISODES,
    per_skill_limit: int = DEFAULT_EPISODES_PER_SKILL,
) -> tuple[List[EvalExample], List[DatasetMix]]:
    """Compose the full eval dataset across multiple skills.

    Real episodes win when available; synthetic fills the gap. Returns both
    the flattened example list and a per-skill breakdown so callers can report
    a `source_mix` in the run's `result.json`.
    """
    log = log or get_episode_log()
    all_examples: List[EvalExample] = []
    mixes: List[DatasetMix] = []

    for sid in skill_ids:
        real = load_real_episodes(
            skill_id=sid,
            log=log,
            window_days=window_days,
            min_episodes=min_episodes,
            limit=per_skill_limit,
        )
        if real:
            all_examples.extend(real)
            mixes.append(DatasetMix(skill_id=sid, real_count=len(real), synthetic_count=0))
            continue

        # Fall back — synthetic_fallback gets this one skill_id and returns its
        # own list of EvalExamples. It has the full signature of
        # AuraSkillAdapter.generate_eval_dataset_synthetic_for_skill.
        try:
            syn = synthetic_fallback(sid)
        except Exception as e:
            logger.warning("synthetic fallback failed for '%s': %s", sid, e)
            syn = []

        all_examples.extend(syn)
        mixes.append(DatasetMix(skill_id=sid, real_count=0, synthetic_count=len(syn)))

    return all_examples, mixes
