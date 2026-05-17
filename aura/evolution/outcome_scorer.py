"""
Outcome Scorer — maps live user-signals to per-episode [0,1] scores.

Rules are deterministic, not LLM-driven. Every signal has a fixed score and
confidence derived from how directly it indicates user satisfaction:
- Explicit /rate commands are most trustworthy (confidence 1.0).
- Reactions (👍/👎) are high confidence but binary.
- Action buttons are noisier — "save" / "export" clearly positive, "regenerate"
  clearly negative, "deeper" ambiguous.

The scorer does not decide whether to run evolution. It just records signals
into the episode log; a separate consolidation step aggregates them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

from .episode_log import SkillEpisodeLog, get_episode_log

logger = logging.getLogger(__name__)


# ─── signal → (score, confidence) table ─────────────────────────────────────
# Telegram reaction emojis. Lists chosen to match the capture set already used
# by agent_core._handle_reaction_update (positive/negative emoji families).
_POSITIVE_EMOJI = {"👍", "❤️", "🔥", "⭐", "🎉", "💯", "👏", "🤩", "💪"}
_NEGATIVE_EMOJI = {"👎", "😢", "😡", "🤮", "💩"}

# Fallback when Telegram only provides 'positive'/'negative'/'neutral' string.
_REACTION_SENTIMENT_TABLE = {
    "positive": (1.0, 0.9),
    "negative": (0.0, 0.9),
    "neutral":  (0.5, 0.4),
}

# Action-button kinds. Keys match the callback-data prefix used in
# messaging/telegram/mixins/misc.py (e.g. `act_save_*`, `act_regenerate_*`).
_ACTION_TABLE = {
    "save":       (0.9, 0.7),
    "export":     (0.9, 0.7),
    "deeper":     (0.7, 0.5),
    "regenerate": (0.25, 0.7),
    "shorter":    (0.30, 0.6),
    "translate":  (0.55, 0.3),  # ambiguous — user wants different form, not a verdict
}

# Explicit /rate command. Binary + fully trusted.
_EXPLICIT_TABLE = {
    "good":  (1.0, 1.0),
    "bad":   (0.0, 1.0),
}


@dataclass
class SignalRecord:
    """One stored outcome signal bound to a request_id."""
    request_id: str
    signal_kind: str
    score: float
    confidence: float


class OutcomeScorer:
    """Routes user-emitted signals into the skill episode log.

    Each entry point:
      1. maps (signal) → (score, confidence) via the module-level tables
      2. queries episode_log.episodes_for_request(request_id)
      3. writes one outcome row per matched episode

    Signals that can't be mapped are silently dropped with a debug log.
    """

    def __init__(self, episode_log: Optional[SkillEpisodeLog] = None):
        self._log = episode_log or get_episode_log()

    # ── Public scoring entry points ────────────────────────────────────────

    def score_from_reaction(
        self,
        *,
        request_id: str,
        sentiment: Optional[str] = None,
        emoji: Optional[str] = None,
    ) -> int:
        """Score from a Telegram-style reaction.

        Caller may pass either `sentiment` ('positive'/'negative'/'neutral')
        from the reaction handler's sentiment classifier, OR a specific
        `emoji` character. Emoji takes priority if both are present because
        it's higher-resolution.
        """
        score_conf = self._resolve_reaction(emoji=emoji, sentiment=sentiment)
        if score_conf is None:
            return 0
        score, conf = score_conf
        return self._write("reaction", request_id, score, conf)

    def score_from_action(self, *, request_id: str, action: str) -> int:
        """Score from an action-button click ('save', 'regenerate', etc)."""
        pair = _ACTION_TABLE.get(action.lower())
        if pair is None:
            logger.debug("unknown action kind '%s' — skipping scoring", action)
            return 0
        score, conf = pair
        return self._write(f"action_{action.lower()}", request_id, score, conf)

    def score_from_explicit(self, *, request_id: str, verdict: str) -> int:
        """Score from an explicit `/rate good` / `/rate bad` command."""
        pair = _EXPLICIT_TABLE.get(verdict.lower())
        if pair is None:
            logger.debug("unknown explicit verdict '%s' — skipping", verdict)
            return 0
        score, conf = pair
        return self._write("explicit", request_id, score, conf)

    # ── Internals ──────────────────────────────────────────────────────────

    def _resolve_reaction(
        self,
        emoji: Optional[str],
        sentiment: Optional[str],
    ) -> Optional[Tuple[float, float]]:
        if emoji:
            if emoji in _POSITIVE_EMOJI:
                return (1.0, 0.9)
            if emoji in _NEGATIVE_EMOJI:
                return (0.0, 0.9)
            return (0.5, 0.3)  # unknown emoji — weak neutral signal
        if sentiment is not None:
            return _REACTION_SENTIMENT_TABLE.get(sentiment.lower())
        return None

    def _write(
        self,
        signal_kind: str,
        request_id: str,
        score: float,
        confidence: float,
    ) -> int:
        """Attach the signal to every episode logged under this request_id.

        Returns the number of outcome rows written (= number of skills invoked
        on this request that got a signal).
        """
        if not request_id:
            return 0

        episodes = self._log.episodes_for_request(request_id)
        if not episodes:
            # Reaction fired for a message with no tracked skills — not a bug,
            # just means this message didn't invoke the skill library.
            logger.debug(
                "scorer: request_id=%s has no episodes (no skills invoked)",
                request_id,
            )
            return 0

        written = 0
        for ep in episodes:
            if self._log.add_outcome(
                episode_id=ep.episode_id,
                signal_kind=signal_kind,
                score=score,
                confidence=confidence,
            ):
                written += 1
        return written


# Module-level singleton — same pattern as episode_log.
_scorer_singleton: Optional[OutcomeScorer] = None


def get_outcome_scorer() -> OutcomeScorer:
    """Return the process-wide outcome scorer."""
    global _scorer_singleton
    if _scorer_singleton is None:
        _scorer_singleton = OutcomeScorer()
    return _scorer_singleton


def reset_for_tests() -> None:
    """Drop module-level singletons — tests use this for isolation."""
    global _scorer_singleton
    _scorer_singleton = None
