"""
Mood-Congruent Memory Retrieval

Implements the psychological phenomenon where current emotional state
biases which memories are more easily recalled. Positive moods boost
retrieval of positively-valenced memories, and vice versa.

Based on: Bower (1981) mood-congruent memory effect.

Author: Aura Development Team
Created: 2026-02-07
"""

import logging
from typing import List, Tuple, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Word lists for valence estimation (kept small and focused)
POSITIVE_WORDS = frozenset({
    "good", "great", "happy", "love", "like", "enjoy", "success", "win",
    "excellent", "amazing", "wonderful", "helpful", "thank", "thanks",
    "appreciate", "awesome", "perfect", "beautiful", "fun", "exciting",
    "glad", "pleased", "proud", "confident", "brilliant", "creative",
    "kind", "warm", "friendly", "calm", "peaceful", "satisfied",
    "learned", "improved", "solved", "fixed", "completed", "achieved",
    "progress", "growth", "insight", "discovery", "breakthrough",
})

NEGATIVE_WORDS = frozenset({
    "bad", "error", "fail", "failed", "wrong", "broken", "bug", "issue",
    "problem", "hate", "dislike", "frustrat", "angry", "sad", "fear",
    "worried", "anxious", "terrible", "awful", "horrible", "difficult",
    "stuck", "confused", "lost", "crash", "missing", "broken",
    "annoyed", "stressed", "tired", "bored", "painful", "struggle",
    "rejected", "denied", "timeout", "exception", "critical",
})

# Mood-congruent bias strength (0 = no bias, 1 = full bias)
MOOD_CONGRUENCE_STRENGTH = 0.15


def get_current_mood_pad() -> Optional[Dict[str, float]]:
    """
    Safely get ALMA's current PAD state.

    Returns:
        Dict with pleasure, arousal, dominance or None if unavailable
    """
    try:
        from aura.emotion.alma_engine import alma_engine
        state = alma_engine.get_emotional_state()
        return state.get("pad", None)
    except Exception:
        return None


def estimate_memory_valence(
    content: str,
    keywords: Optional[List[str]] = None,
    tags: Optional[List[str]] = None
) -> float:
    """
    Estimate the emotional valence of a memory.

    Returns a value from -1.0 (very negative) to 1.0 (very positive).
    0.0 means neutral or unknown.

    Uses word-level heuristics since memories don't have PAD tags yet.
    """
    if not content:
        return 0.0

    words = set(content.lower().split())

    # Add keywords and tags to word set
    if keywords:
        words.update(w.lower() for w in keywords)
    if tags:
        words.update(t.lower() for t in tags)

    # Count positive and negative word matches
    pos_count = sum(1 for w in words if w in POSITIVE_WORDS)
    neg_count = sum(1 for w in words if w in NEGATIVE_WORDS)

    # Also check for partial matches (e.g., "frustrated" matches "frustrat")
    content_lower = content.lower()
    for pw in POSITIVE_WORDS:
        if len(pw) >= 5 and pw in content_lower:
            pos_count += 0.5
    for nw in NEGATIVE_WORDS:
        if len(nw) >= 5 and nw in content_lower:
            neg_count += 0.5

    total = pos_count + neg_count
    if total == 0:
        return 0.0

    # Normalize to [-1, 1]
    valence = (pos_count - neg_count) / (total + 2)  # +2 for smoothing
    return max(-1.0, min(1.0, valence))


def mood_congruent_score_adjustment(
    base_score: float,
    memory_valence: float,
    mood_pleasure: float,
    strength: float = MOOD_CONGRUENCE_STRENGTH
) -> float:
    """
    Adjust a memory's retrieval score based on mood-congruence.

    When mood pleasure is positive, positively-valenced memories get boosted.
    When mood pleasure is negative, negatively-valenced memories get boosted.
    Neutral memories are unaffected.

    Args:
        base_score: Original retrieval score
        memory_valence: Estimated valence of the memory (-1 to 1)
        mood_pleasure: Current mood pleasure dimension (-1 to 1)
        strength: How strongly mood affects retrieval (0 to 1)

    Returns:
        Adjusted score (always >= 0)
    """
    if abs(mood_pleasure) < 0.05 or abs(memory_valence) < 0.05:
        return base_score

    # Congruence: same-sign valence and mood = positive alignment
    alignment = mood_pleasure * memory_valence  # ranges -1 to 1

    # Apply adjustment: congruent memories boosted, incongruent slightly dampened
    adjustment = 1.0 + (alignment * strength)

    return max(0.0, base_score * adjustment)


def apply_mood_congruent_bias(
    results: List[Dict[str, Any]],
    mood_pad: Optional[Dict[str, float]] = None,
    content_key: str = "content",
    score_key: str = "score",
    keywords_key: str = "keywords",
    tags_key: str = "tags"
) -> List[Dict[str, Any]]:
    """
    Apply mood-congruent bias to a list of memory search results.

    Modifies scores in-place and returns the list (for chaining).

    Args:
        results: List of result dicts with content and score
        mood_pad: PAD state dict, or None to auto-fetch
        content_key: Key for content in result dicts
        score_key: Key for score in result dicts
    """
    if not results:
        return results

    # Get mood if not provided
    if mood_pad is None:
        mood_pad = get_current_mood_pad()

    if mood_pad is None:
        return results

    pleasure = mood_pad.get("pleasure", 0.0)

    # Only apply bias if mood has meaningful pleasure component
    if abs(pleasure) < 0.1:
        return results

    adjusted = False
    for result in results:
        content = result.get(content_key, "")
        keywords = result.get(keywords_key, [])
        tags = result.get(tags_key, [])

        valence = estimate_memory_valence(content, keywords, tags)
        if abs(valence) > 0.05:
            old_score = result.get(score_key, 0.0)
            new_score = mood_congruent_score_adjustment(
                old_score, valence, pleasure
            )
            result[score_key] = new_score
            adjusted = True

    if adjusted:
        logger.debug(
            f"Applied mood-congruent bias (pleasure={pleasure:.2f}) "
            f"to {len(results)} results"
        )

    return results


def get_note_valence(note: Any) -> float:
    """
    Get the emotional valence of a MemoryNote.

    Prefers stored emotional_pad (from 2C emotional tagging) over
    word-heuristic estimation (from 2A).
    """
    # Prefer stored PAD data if available
    pad = getattr(note, "emotional_pad", None)
    if pad and isinstance(pad, dict):
        pleasure = pad.get("pleasure", 0.0)
        if abs(pleasure) > 0.01:
            return max(-1.0, min(1.0, pleasure))

    # Fallback: word-level heuristic
    return estimate_memory_valence(
        getattr(note, "content", ""),
        getattr(note, "keywords", []),
        getattr(note, "tags", [])
    )


def apply_mood_bias_to_tuples(
    results: List[Tuple[Any, float]],
    mood_pad: Optional[Dict[str, float]] = None
) -> List[Tuple[Any, float]]:
    """
    Apply mood-congruent bias to (MemoryNote, score) tuple results.

    Used by amem.py search() which returns tuples.
    """
    if not results:
        return results

    if mood_pad is None:
        mood_pad = get_current_mood_pad()

    if mood_pad is None:
        return results

    pleasure = mood_pad.get("pleasure", 0.0)
    if abs(pleasure) < 0.1:
        return results

    adjusted = []
    for note, score in results:
        valence = get_note_valence(note)
        new_score = mood_congruent_score_adjustment(score, valence, pleasure)
        adjusted.append((note, new_score))

    return adjusted
