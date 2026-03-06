# Knowledge Graph Memory System

## Overview
Graph-based memory storing entities (nodes) and relationships (edges). Enables semantic search, relationship traversal, and knowledge discovery.

## Architecture
- **Storage**: SQLite database (`aura_data/knowledge_graph_brain`)
- **Nodes**: Entities with type, properties, embeddings
- **Edges**: Typed relationships between nodes
- **Query Engine**: Natural language to graph queries

## Node Types
- person, concept, tool, project, file, event, location, organization, skill, topic

## Edge Types
- relates_to, part_of, created_by, uses, depends_on, similar_to, causes, enables

## Key Operations
- `add_node(name, type, properties)` - Add entity
- `add_edge(source, target, type)` - Add relationship
- `query(natural_language)` - Search with NL
- `get_neighbors(node_id)` - Traverse relationships
- `find_path(source, target)` - Find connection path

## Integration
- Fed by `KnowledgeExtractor` (auto-extracts entities from conversations)
- Used by `HybridMemory` and `HybridAMEM` for semantic retrieval
- Connected to `NeuroDream` for relationship discovery during sleep
- Exposed via MCP tools in `aura_knowledge_graph/mcp_tools.py`

## Files
- `aura/tools/knowledge_graph.py` - Core KG tool
- `aura/tools/kg_extractor.py` - Auto-extraction from text
- `aura_knowledge_graph/` - Extended KG module with MCP integration
