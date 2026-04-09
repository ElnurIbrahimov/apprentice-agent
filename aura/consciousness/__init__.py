"""AURA Consciousness modules - Higher-order cognitive functions."""

from .idle_presence import IdlePresenceEngine, get_idle_presence_engine
from .intrinsic_motivation import IntrinsicMotivationEngine, get_intrinsic_motivation
from .metacognition import MetacognitiveEngine, get_metacognitive_engine
from .proactive_awareness import ProactiveAwarenessEngine, get_proactive_awareness_engine
from .reasoning_templates import ReasoningTemplateLibrary, get_template_library

# GlobalWorkspaceEngine removed — module still exists but no longer imported at init
# from .global_workspace import GlobalWorkspaceEngine, get_global_workspace
from .self_improvement import SelfImprovementEngine, get_self_improvement_engine
from .state_extractor import StateExtractor, get_state_extractor
from .strategy_bandit import StrategyBandit, get_strategy_bandit
from .world_model import WorldModel, get_world_model

__all__ = [
    "IdlePresenceEngine",
    "IntrinsicMotivationEngine",
    "MetacognitiveEngine",
    "ProactiveAwarenessEngine",
    "ReasoningTemplateLibrary",
    "SelfImprovementEngine",
    "StateExtractor",
    "StrategyBandit",
    "WorldModel",
    "get_idle_presence_engine",
    "get_intrinsic_motivation",
    "get_metacognitive_engine",
    "get_proactive_awareness_engine",
    "get_self_improvement_engine",
    "get_state_extractor",
    "get_strategy_bandit",
    "get_template_library",
    "get_world_model",
]
