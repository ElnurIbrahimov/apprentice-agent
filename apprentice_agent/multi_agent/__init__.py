"""AURA Multi-Agent System.

A collaborative multi-agent architecture where specialized agents work together
to handle complex tasks. All agents share the same Ollama model but have different
system prompts and tool access, making it VRAM-efficient.

Components:
- Orchestrator: Routes tasks to appropriate specialists
- Specialists: Domain-specific agents (Research, Coder, Analyst, Creative)
- Protocol: Inter-agent communication structures
- Router: Intent classification and agent selection
"""

from .protocol import AgentMessage, AgentResult, CollaborationMode
from .base_agent import BaseSpecialist
from .orchestrator import MultiAgentOrchestrator
from .router import IntentRouter

__all__ = [
    "AgentMessage",
    "AgentResult",
    "CollaborationMode",
    "BaseSpecialist",
    "MultiAgentOrchestrator",
    "IntentRouter",
]
