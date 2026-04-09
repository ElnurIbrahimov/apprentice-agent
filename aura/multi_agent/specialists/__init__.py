"""Specialist agents for the multi-agent system.

Each specialist is optimized for a specific domain:
- ResearchAgent: Information gathering (web, papers, documents)
- CoderAgent: Code writing, debugging, and execution
- AnalystAgent: Data analysis and reasoning
- CreativeAgent: Creative content generation
- SearcherAgent: Code search, definitions, references, project structure
"""

from .analyst import AnalystAgent
from .coder import CoderAgent
from .creative import CreativeAgent
from .research import ResearchAgent
from .searcher import SearcherAgent

__all__ = [
    "AnalystAgent",
    "CoderAgent",
    "CreativeAgent",
    "ResearchAgent",
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
