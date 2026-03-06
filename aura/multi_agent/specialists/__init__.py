"""Specialist agents for the multi-agent system.

Each specialist is optimized for a specific domain:
- ResearchAgent: Information gathering (web, papers, documents)
- CoderAgent: Code writing, debugging, and execution
- AnalystAgent: Data analysis and reasoning
- CreativeAgent: Creative content generation
"""

from .research import ResearchAgent
from .coder import CoderAgent
from .analyst import AnalystAgent
from .creative import CreativeAgent

__all__ = [
    "ResearchAgent",
    "CoderAgent",
    "AnalystAgent",
    "CreativeAgent",
]

# Default specialist configurations
DEFAULT_SPECIALISTS = {
    "research": ResearchAgent,
    "coder": CoderAgent,
    "analyst": AnalystAgent,
    "creative": CreativeAgent,
}
