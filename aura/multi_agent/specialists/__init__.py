"""Specialist agents for the multi-agent system.

Each specialist is optimized for a specific domain:
- ResearchAgent: Information gathering (web, papers, documents)
- CoderAgent: Code writing, debugging, and execution
- AnalystAgent: Data analysis and reasoning
- CreativeAgent: Creative content generation
- SearcherAgent: Code search, definitions, references, project structure
"""

from .research import ResearchAgent
from .coder import CoderAgent
from .analyst import AnalystAgent
from .creative import CreativeAgent
from .searcher import SearcherAgent

__all__ = [
    "ResearchAgent",
    "CoderAgent",
    "AnalystAgent",
    "CreativeAgent",
    "SearcherAgent",
]

# Default specialist configurations
DEFAULT_SPECIALISTS = {
    "research": ResearchAgent,
    "coder": CoderAgent,
    "analyst": AnalystAgent,
    "creative": CreativeAgent,
    "searcher": SearcherAgent,
}
