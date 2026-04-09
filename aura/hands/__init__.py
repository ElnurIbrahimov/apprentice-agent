"""Aura Hands — autonomous task packages that run without prompting.

Stolen from OpenFang's "Hands" concept: self-contained capability packages
with manifests, schedules, guardrails, and budgets. Adapted to work with
Aura's consciousness stack (intrinsic motivation drives Hand priority,
idle presence triggers Hand activation, metacognition evaluates Hand performance).
"""

from aura.hands.base import Hand, HandManifest, HandResult, HandState
from aura.hands.manager import HandManager, get_hand_manager

__all__ = [
    "Hand",
    "HandManager",
    "HandManifest",
    "HandResult",
    "HandState",
    "get_hand_manager",
]
