"""
AURA Episodic Time-Travel Memory

A temporal autobiographical memory system for storing and retrieving
episodic memories with rich temporal context and natural language queries.

Key Components:
- Episode: Core data model for episodic memories
- EpisodicMemoryStore: Qdrant-based vector storage
- TemporalParser: Natural language time parsing
- MemoryScorer: Multi-factor retrieval scoring
- TimelineEngine: Temporal navigation and story mode
- TitansEpisodicBridge: Integration with Titans Memory
- MemoryConsolidator: Memory maintenance operations
- MCP Tools: Agent-accessible memory operations

Usage:
```python
from aura_episodic_memory import (
    EpisodicMemoryStore,
    Episode,
    EpisodeType,
    TitansEpisodicBridge,
    TitansEpisodicConfig,
    TimelineEngine,
    create_episodic_tools
)

# Initialize memory store
memory = EpisodicMemoryStore("./aura_data/episodic_memory")

# Store an episode
episode = Episode(
    content="User asked about Python decorators",
    episode_type=EpisodeType.CONVERSATION,
    temporal_context=TemporalContext(timestamp=datetime.now()),
    importance=0.7
)
memory.store_episode(episode)

# Search memories
from aura_episodic_memory import EpisodeQuery
results = memory.search(EpisodeQuery(
    query_text="Python decorators",
    limit=5
))

# Time travel
timeline = TimelineEngine(memory)
episodes, narrative = timeline.time_travel("yesterday afternoon")

# Initialize Titans bridge
bridge = TitansEpisodicBridge(memory)
context = bridge.get_context_for_query("What did we discuss about Python?")

# Register MCP tools
tools = create_episodic_tools(memory)
```
"""

from .episode import (
    Episode,
    EpisodeType,
    EpisodeQuery,
    EpisodeSearchResult,
    TemporalContext,
    EmotionalValence
)

from .memory_store import (
    EpisodicMemoryStore,
    EmbeddingModel,
    QDRANT_AVAILABLE,
)

# Compatibility shim — was removed from memory_store but still referenced in tests
try:
    from .memory_store import SENTENCE_TRANSFORMERS_AVAILABLE  # type: ignore
except ImportError:
    try:
        import sentence_transformers  # noqa: F401
        SENTENCE_TRANSFORMERS_AVAILABLE = True
    except ImportError:
        SENTENCE_TRANSFORMERS_AVAILABLE = False

from .temporal_parser import (
    TemporalParser,
    TemporalRange,
    DATEPARSER_AVAILABLE
)

from .memory_scorer import (
    MemoryScorer,
    ScoringConfig,
    AdaptiveScorer,
    ContextualScorer
)

from .timeline import (
    TimelineEngine,
    TimelineView,
    TimelineSegment
)

from .titans_integration import (
    TitansEpisodicBridge,
    TitansEpisodicConfig
)

from .consolidation import (
    MemoryConsolidator,
    ConsolidationConfig,
    ConsolidationResult
)

from .mcp_tools import (
    create_episodic_tools,
    register_episodic_tools_with_agent,
    MCPTool,
    QuickEpisodicMemory,
)

__all__ = [
    # Episode data model
    "Episode",
    "EpisodeType",
    "EpisodeQuery",
    "EpisodeSearchResult",
    "TemporalContext",
    "EmotionalValence",

    # Memory store
    "EpisodicMemoryStore",
    "EmbeddingModel",
    "QDRANT_AVAILABLE",
    "SENTENCE_TRANSFORMERS_AVAILABLE",

    # Temporal parsing
    "TemporalParser",
    "TemporalRange",
    "DATEPARSER_AVAILABLE",

    # Scoring
    "MemoryScorer",
    "ScoringConfig",
    "AdaptiveScorer",
    "ContextualScorer",

    # Timeline
    "TimelineEngine",
    "TimelineView",
    "TimelineSegment",

    # Titans integration
    "TitansEpisodicBridge",
    "TitansEpisodicConfig",

    # Consolidation
    "MemoryConsolidator",
    "ConsolidationConfig",
    "ConsolidationResult",

    # MCP Tools
    "create_episodic_tools",
    "register_episodic_tools_with_agent",
    "MCPTool",
    "QuickEpisodicMemory",
]

__version__ = "1.0.0"
