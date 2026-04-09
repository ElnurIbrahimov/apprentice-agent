"""
AURA Emotion Module - ALMA-based Emotional Intelligence System

This module provides AURA with genuine emotional depth through:
- PAD (Pleasure-Arousal-Dominance) emotional space
- Three-layer affect model (Emotion/Mood/Personality)
- Neuromodulator-inspired response modulation
- OCC appraisal-based emotion generation
"""

from .alma_engine import (
    ALMAEngine,
    EmotionState,
    MoodState,
    PADState,
    PersonalityProfile,
    get_alma_engine,
    get_emotional_state,
    get_response_modulation,
    trigger_emotion,
    update_from_interaction,
)


def __getattr__(name: str):
    """Lazy access to ``alma_engine`` singleton (avoids import-time instantiation)."""
    if name == "alma_engine":
        return get_alma_engine()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ALMAEngine",
    "EmotionState",
    "MoodState",
    "PADState",
    "PersonalityProfile",
    "alma_engine",
    "get_alma_engine",
    "get_emotional_state",
    "get_response_modulation",
    "trigger_emotion",
    "update_from_interaction",
]
