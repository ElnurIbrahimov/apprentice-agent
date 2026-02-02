"""
AURA Knowledge Graph Brain

A structured long-term memory system that complements Titans Neural Memory.
Uses Kùzu embedded graph database for zero-server, zero-VRAM operation.

Key Components:
- AURAKnowledgeGraph: Kùzu-based graph database wrapper
- EntityExtractor: LLM-based entity/relationship extraction
- TitansKGBridge: Integration with Titans Memory
- KGQueryEngine: Multi-strategy graph retrieval
- MCP Tools: Agent-accessible graph operations

Usage:
```python
from aura_knowledge_graph import (
    AURAKnowledgeGraph,
    TitansKGBridge,
    BridgeConfig,
    KGQueryEngine,
    create_kg_tools
)

# Initialize graph
kg = AURAKnowledgeGraph("./aura_data/knowledge_graph")

# Initialize bridge for Titans integration
bridge = TitansKGBridge(
    knowledge_graph=kg,
    llm_func=your_llm_function,
    config=BridgeConfig(surprise_threshold=0.5)
)

# Query the graph
query_engine = KGQueryEngine(kg)
result = query_engine.query("AURA", mode=QueryMode.HYBRID)
print(result.context_string)

# Register MCP tools
tools = create_kg_tools(kg)
```
"""

from .schema import (
    EntityType,
    ALLOWED_RELATIONSHIPS,
    get_schema_prompt,
    validate_relationship
)

from .graph_database import (
    AURAKnowledgeGraph,
    Entity,
    Relationship,
    KUZU_AVAILABLE
)

from .entity_extractor import (
    EntityExtractor,
    ExtractionResult
)

from .titans_bridge import (
    TitansKGBridge,
    BridgeConfig
)

from .query_engine import (
    KGQueryEngine,
    QueryMode,
    QueryResult
)

from .mcp_tools import (
    create_kg_tools,
    MCPTool,
    register_kg_tools_with_agent
)

__all__ = [
    # Schema
    "EntityType",
    "ALLOWED_RELATIONSHIPS",
    "get_schema_prompt",
    "validate_relationship",

    # Database
    "AURAKnowledgeGraph",
    "Entity",
    "Relationship",
    "KUZU_AVAILABLE",

    # Extraction
    "EntityExtractor",
    "ExtractionResult",

    # Titans Integration
    "TitansKGBridge",
    "BridgeConfig",

    # Query Engine
    "KGQueryEngine",
    "QueryMode",
    "QueryResult",

    # MCP Tools
    "create_kg_tools",
    "MCPTool",
    "register_kg_tools_with_agent",
]

__version__ = "1.0.0"
