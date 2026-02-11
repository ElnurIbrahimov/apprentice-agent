"""AURA Consciousness modules - Higher-order cognitive functions."""

from .metacognition import MetacognitiveEngine, get_metacognitive_engine
from .idle_presence import IdlePresenceEngine, get_idle_presence_engine
from .intrinsic_motivation import IntrinsicMotivationEngine, get_intrinsic_motivation
from .global_workspace import GlobalWorkspaceEngine, get_global_workspace
from .self_improvement import SelfImprovementEngine, get_self_improvement_engine
from .strategy_bandit import StrategyBandit, get_strategy_bandit
from .reasoning_templates import ReasoningTemplateLibrary, get_template_library
from .prompt_evolution import PromptEvolutionEngine, get_prompt_evolution_engine
from .world_model import WorldModel, get_world_model
from .state_extractor import StateExtractor, get_state_extractor
from .proactive_awareness import ProactiveAwarenessEngine, get_proactive_awareness_engine

__all__ = [
    "MetacognitiveEngine", "get_metacognitive_engine",
    "IdlePresenceEngine", "get_idle_presence_engine",
    "IntrinsicMotivationEngine", "get_intrinsic_motivation",
    "GlobalWorkspaceEngine", "get_global_workspace",
    "SelfImprovementEngine", "get_self_improvement_engine",
    "StrategyBandit", "get_strategy_bandit",
    "ReasoningTemplateLibrary", "get_template_library",
    "PromptEvolutionEngine", "get_prompt_evolution_engine",
    "WorldModel", "get_world_model",
    "StateExtractor", "get_state_extractor",
    "ProactiveAwarenessEngine", "get_proactive_awareness_engine",
]
