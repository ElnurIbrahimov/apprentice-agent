"""AURA Consciousness modules - Higher-order cognitive functions."""

from .metacognition import MetacognitiveEngine, get_metacognitive_engine
from .idle_presence import IdlePresenceEngine, get_idle_presence_engine
from .intrinsic_motivation import IntrinsicMotivationEngine, get_intrinsic_motivation

__all__ = [
    "MetacognitiveEngine", "get_metacognitive_engine",
    "IdlePresenceEngine", "get_idle_presence_engine",
    "IntrinsicMotivationEngine", "get_intrinsic_motivation",
]
