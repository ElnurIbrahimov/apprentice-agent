"""AURA Consciousness modules - Higher-order cognitive functions."""

from .metacognition import MetacognitiveEngine, get_metacognitive_engine
from .idle_presence import IdlePresenceEngine, get_idle_presence_engine
from .intrinsic_motivation import IntrinsicMotivationEngine, get_intrinsic_motivation
from .global_workspace import GlobalWorkspaceEngine, get_global_workspace
from .self_improvement import SelfImprovementEngine, get_self_improvement_engine
from .strategy_bandit import StrategyBandit, get_strategy_bandit

__all__ = [
    "MetacognitiveEngine", "get_metacognitive_engine",
    "IdlePresenceEngine", "get_idle_presence_engine",
    "IntrinsicMotivationEngine", "get_intrinsic_motivation",
    "GlobalWorkspaceEngine", "get_global_workspace",
    "SelfImprovementEngine", "get_self_improvement_engine",
    "StrategyBandit", "get_strategy_bandit",
]
