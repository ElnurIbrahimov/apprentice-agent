"""Temporal Grounding — Phase 3.7: Session continuity.

At session start, creates the experience of reconnecting with a being
that has continuity by:
1. Loading the narrative self-model
2. Calculating time elapsed since last interaction
3. Running a "since last time" summary from memory
4. Loading dream insights marked for proactive delivery
5. Adjusting mood based on time elapsed (arousal decay, curiosity buildup)

This module produces a short context block (~200 tokens) injected into
the system prompt at session start.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


def compute_session_greeting(last_interaction_ts: Optional[float] = None) -> Dict[str, Any]:
    """Compute temporal grounding context for session start.

    Returns a dict with:
        elapsed_hours: float
        greeting_hint: str (time-aware)
        mood_adjustments: dict (PAD deltas)
        dream_insights: list of str
        since_last_summary: str
    """
    now = time.time()
    elapsed_hours = 0.0
    if last_interaction_ts and last_interaction_ts > 0:
        elapsed_hours = (now - last_interaction_ts) / 3600.0

    result: Dict[str, Any] = {
        "elapsed_hours": round(elapsed_hours, 2),
        "greeting_hint": _time_greeting(elapsed_hours),
        "mood_adjustments": _compute_mood_adjustments(elapsed_hours),
        "dream_insights": _load_dream_insights(),
        "since_last_summary": "",
    }

    # Build a "since last time" summary from recent memories
    result["since_last_summary"] = _build_since_last(elapsed_hours)

    return result


def build_temporal_prompt(grounding: Dict[str, Any]) -> str:
    """Build a system prompt injection from temporal grounding context.

    Returns a concise string (~150 tokens) or empty string if nothing notable.
    """
    parts: List[str] = []
    hours = grounding.get("elapsed_hours", 0)

    if hours > 0.5:
        hint = grounding.get("greeting_hint", "")
        if hint:
            parts.append(hint)

    summary = grounding.get("since_last_summary", "")
    if summary:
        parts.append(summary)

    insights = grounding.get("dream_insights", [])
    if insights:
        insight_text = "; ".join(insights[:3])
        parts.append(f"Dream insights to share: {insight_text}")

    if not parts:
        return ""

    return "[Temporal Context]\n" + "\n".join(parts)


def _time_greeting(elapsed_hours: float) -> str:
    """Generate a time-aware greeting hint based on elapsed time."""
    if elapsed_hours < 0.1:
        return ""  # Basically continuous conversation
    elif elapsed_hours < 1:
        return f"It's been about {int(elapsed_hours * 60)} minutes since you last spoke."
    elif elapsed_hours < 24:
        return f"It's been about {elapsed_hours:.1f} hours since your last session."
    elif elapsed_hours < 168:
        days = elapsed_hours / 24
        return f"It's been {days:.1f} days since you were last here."
    else:
        days = elapsed_hours / 24
        return f"It's been {int(days)} days — quite a while. Welcome back."


def _compute_mood_adjustments(elapsed_hours: float) -> Dict[str, float]:
    """Compute PAD adjustments based on time elapsed.

    Long absence -> arousal decays toward neutral, curiosity builds up.
    Short gap -> minimal adjustment.
    """
    adjustments: Dict[str, float] = {}

    if elapsed_hours < 0.5:
        return adjustments  # No adjustment needed

    # Arousal decays toward 0 (neutral) with time
    # After 8 hours, arousal should be near baseline
    arousal_decay = min(0.3, elapsed_hours * 0.04)
    adjustments["arousal_decay"] = -arousal_decay

    # Curiosity builds with absence (what has the user been up to?)
    curiosity_boost = min(0.2, elapsed_hours * 0.02)
    adjustments["curiosity_boost"] = curiosity_boost

    # Pleasure gently rises on reconnection (glad to see you)
    if elapsed_hours > 2:
        adjustments["pleasure_boost"] = min(0.15, elapsed_hours * 0.02)

    return adjustments


def _load_dream_insights() -> List[str]:
    """Load dream insights marked for proactive delivery."""
    try:
        from aura.tools.neurodream import get_dream_insights_for_delivery
        return get_dream_insights_for_delivery()
    except (ImportError, AttributeError):
        pass
    return []


def _build_since_last(elapsed_hours: float) -> str:
    """Build a brief summary of what happened since last interaction."""
    if elapsed_hours < 1:
        return ""

    try:
        from aura.memory.unified_memory import get_unified_memory
        um = get_unified_memory()
        stats = um.get_stats() if hasattr(um, "get_stats") else {}
        total = stats.get("total_memories", 0)
        if total > 0:
            return f"You have {total} memories stored. Let me check what's relevant to pick up where we left off."
    except Exception as e:
        logger.debug(f"[TemporalGrounding] Memory summary error: {e}")

    return ""


def apply_mood_adjustments(adjustments: Dict[str, float]) -> None:
    """Apply temporal mood adjustments to the ALMA engine."""
    if not adjustments:
        return

    try:
        from aura.emotion.alma_engine import alma_engine

        with alma_engine._lock:
            pad = alma_engine.mood.pad

            decay = adjustments.get("arousal_decay", 0)
            if decay:
                pad.arousal = max(-1.0, min(1.0, pad.arousal + decay))

            p_boost = adjustments.get("pleasure_boost", 0)
            if p_boost:
                pad.pleasure = max(-1.0, min(1.0, pad.pleasure + p_boost))

            # Boost curiosity via dopamine
            c_boost = adjustments.get("curiosity_boost", 0)
            if c_boost:
                alma_engine.neuromodulators.dopamine = min(
                    1.0, alma_engine.neuromodulators.dopamine + c_boost
                )

        logger.info(f"[TemporalGrounding] Applied mood adjustments: {adjustments}")
    except Exception as e:
        logger.debug(f"[TemporalGrounding] Mood adjustment error: {e}")
