"""
MCP (Model Context Protocol) tools for Knowledge Graph operations.

These tools allow AURA to:
- Query the knowledge graph
- Add entities manually
- View graph statistics
- Export/import knowledge
"""

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict

from .graph_database import AURAKnowledgeGraph, Entity, Relationship
from .query_engine import KGQueryEngine, QueryMode
from .schema import EntityType

logger = logging.getLogger(__name__)


@dataclass
class MCPTool:
    """MCP Tool definition."""
    name: str
    description: str
    input_schema: Dict
    execute_func: Callable
    category: str = "knowledge_graph"


def create_kg_tools(kg: AURAKnowledgeGraph) -> Dict[str, MCPTool]:
    """
    Create MCP tools for Knowledge Graph operations.

    Usage in AURA:
    ```python
    kg_tools = create_kg_tools(self.knowledge_graph)
    for tool in kg_tools.values():
        self.tools.register(tool)
    ```
    """
    query_engine = KGQueryEngine(kg)

    # Tool: Query Knowledge Graph
    def query_kg(params: Dict) -> Dict:
        query = params.get("query", "")
        mode = params.get("mode", "hybrid")
        max_results = params.get("max_results", 10)

        if not query:
            return {"success": False, "error": "Query is required"}

        try:
            query_mode = QueryMode(mode)
        except ValueError:
            query_mode = QueryMode.HYBRID

        result = query_engine.query(query, mode=query_mode, max_entities=max_results)

        return {
            "success": True,
            "entities": result.entities,
            "relationships": result.relationships,
            "context": result.context_string,
            "metadata": result.metadata
        }

    # Tool: Add Entity
    def add_entity(params: Dict) -> Dict:
        name = params.get("name", "")
        entity_type = params.get("type", "Concept")
        description = params.get("description", "")
        importance = params.get("importance", 0.5)

        if not name:
            return {"success": False, "error": "Entity name is required"}

        try:
            etype = EntityType(entity_type)
        except ValueError:
            logger.warning(f"Unknown entity type: {entity_type}, using Concept")
            etype = EntityType.CONCEPT

        entity = Entity(
            name=name,
            entity_type=etype,
            description=description,
            importance=importance
        )

        entity_id = kg.add_entity(entity)

        return {
            "success": True,
            "entity_id": entity_id,
            "message": f"Added entity: {name} ({entity_type})"
        }

    # Tool: Add Relationship
    def add_relationship(params: Dict) -> Dict:
        source = params.get("source", "")
        target = params.get("target", "")
        relationship = params.get("relationship", "RELATES_TO")
        evidence = params.get("evidence", "")

        if not source or not target:
            return {"success": False, "error": "Source and target are required"}

        # Find entity IDs by name
        source_entities = kg.query_entities(source, limit=1)
        target_entities = kg.query_entities(target, limit=1)

        if not source_entities:
            return {"success": False, "error": f"Source entity not found: {source}"}
        if not target_entities:
            return {"success": False, "error": f"Target entity not found: {target}"}

        rel = Relationship(
            source_id=source_entities[0]["id"],
            target_id=target_entities[0]["id"],
            relationship_type=relationship,
            evidence=evidence
        )

        success = kg.add_relationship(rel)

        return {
            "success": success,
            "message": f"Added relationship: {source} --[{relationship}]--> {target}"
        }

    # Tool: Get Graph Statistics
    def get_kg_stats(params: Dict) -> Dict:
        stats = kg.get_statistics()
        return {
            "success": True,
            **stats
        }

    # Tool: Find Related Entities
    def find_related(params: Dict) -> Dict:
        entity_name = params.get("entity", "")
        hops = params.get("hops", 2)

        if not entity_name:
            return {"success": False, "error": "Entity name is required"}

        # Find the entity
        entities = kg.query_entities(entity_name, limit=1)
        if not entities:
            return {"success": False, "error": f"Entity not found: {entity_name}"}

        entity_id = entities[0]["id"]
        related = kg.get_related_entities(entity_id, hops=hops)

        return {
            "success": True,
            "entity": entities[0],
            "related": related,
            "count": len(related)
        }

    # Tool: Get Entity Details
    def get_entity(params: Dict) -> Dict:
        entity_name = params.get("name", "")

        if not entity_name:
            return {"success": False, "error": "Entity name is required"}

        entity = kg.get_entity_by_name(entity_name)
        if not entity:
            return {"success": False, "error": f"Entity not found: {entity_name}"}

        # Get relationships
        relationships = kg.get_relationships(entity["id"])

        return {
            "success": True,
            "entity": entity,
            "relationships": relationships,
            "summary": query_engine.get_entity_summary(entity_name)
        }

    # Tool: Execute Cypher Query (advanced)
    def execute_cypher(params: Dict) -> Dict:
        query = params.get("query", "")

        if not query:
            return {"success": False, "error": "Query is required"}

        # Safety check - no destructive operations
        query_upper = query.upper()
        if any(word in query_upper for word in ["DELETE", "DROP", "REMOVE", "DETACH", "CREATE", "SET"]):
            return {"success": False, "error": "Destructive/write operations not allowed via this tool"}

        try:
            result = kg.execute_cypher(query)
            return {
                "success": True,
                "result": result,
                "count": len(result)
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    # Tool: Maintenance - Decay and Prune
    def maintenance(params: Dict) -> Dict:
        action = params.get("action", "decay")

        if action == "decay":
            decay_rate = params.get("decay_rate", 0.01)
            kg.decay_importance(decay_rate)
            return {"success": True, "message": f"Applied decay rate: {decay_rate}"}

        elif action == "prune":
            threshold = params.get("threshold", 0.05)
            kg.prune_low_importance(threshold)
            return {"success": True, "message": f"Pruned entities below importance: {threshold}"}

        elif action == "stats":
            return get_kg_stats({})

        else:
            return {"success": False, "error": f"Unknown action: {action}"}

    # Create tool definitions
    tools = {
        "query_knowledge_graph": MCPTool(
            name="query_knowledge_graph",
            description="Query the knowledge graph for entities and relationships. Use for questions like 'What do I know about X?' or 'How is X related to Y?'",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (entity name, concept, or natural language)"
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["entity", "traversal", "global", "hybrid"],
                        "description": "Query mode: entity (direct search), traversal (graph walk), global (overview), hybrid (combined)",
                        "default": "hybrid"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of entities to return",
                        "default": 10
                    }
                },
                "required": ["query"]
            },
            execute_func=query_kg
        ),

        "add_entity": MCPTool(
            name="add_entity",
            description="Add a new entity to the knowledge graph. Use this to store important facts about people, projects, technologies, etc.",
            input_schema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Entity name"
                    },
                    "type": {
                        "type": "string",
                        "enum": ["Person", "Project", "Technology", "Company", "Concept", "Task", "Location", "Event", "Document", "Skill"],
                        "description": "Entity type"
                    },
                    "description": {
                        "type": "string",
                        "description": "Brief description of the entity"
                    },
                    "importance": {
                        "type": "number",
                        "description": "Initial importance score (0-1)",
                        "default": 0.5
                    }
                },
                "required": ["name", "type"]
            },
            execute_func=add_entity
        ),

        "add_relationship": MCPTool(
            name="add_relationship",
            description="Add a relationship between two entities in the knowledge graph",
            input_schema={
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": "Source entity name"
                    },
                    "target": {
                        "type": "string",
                        "description": "Target entity name"
                    },
                    "relationship": {
                        "type": "string",
                        "description": "Relationship type (e.g., WORKS_ON, USES, RELATES_TO, KNOWS)"
                    },
                    "evidence": {
                        "type": "string",
                        "description": "Evidence or context for this relationship"
                    }
                },
                "required": ["source", "target", "relationship"]
            },
            execute_func=add_relationship
        ),

        "kg_statistics": MCPTool(
            name="kg_statistics",
            description="Get statistics about the knowledge graph (entity count, relationship count, type distribution, etc.)",
            input_schema={
                "type": "object",
                "properties": {}
            },
            execute_func=get_kg_stats
        ),

        "find_related_entities": MCPTool(
            name="find_related_entities",
            description="Find entities related to a given entity within N relationship hops",
            input_schema={
                "type": "object",
                "properties": {
                    "entity": {
                        "type": "string",
                        "description": "Entity name to find relationships for"
                    },
                    "hops": {
                        "type": "integer",
                        "description": "Number of relationship hops to traverse (1-4)",
                        "default": 2
                    }
                },
                "required": ["entity"]
            },
            execute_func=find_related
        ),

        "get_entity_details": MCPTool(
            name="get_entity_details",
            description="Get detailed information about a specific entity including all its relationships",
            input_schema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Entity name"
                    }
                },
                "required": ["name"]
            },
            execute_func=get_entity
        ),

        "cypher_query": MCPTool(
            name="cypher_query",
            description="Execute a custom Cypher query on the knowledge graph (read-only). For advanced graph queries.",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Cypher query to execute (e.g., MATCH (n:Entity) RETURN n.name LIMIT 10)"
                    }
                },
                "required": ["query"]
            },
            execute_func=execute_cypher
        ),

        "kg_maintenance": MCPTool(
            name="kg_maintenance",
            description="Perform maintenance on the knowledge graph (decay importance, prune low-importance entities)",
            input_schema={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["decay", "prune", "stats"],
                        "description": "Maintenance action to perform"
                    },
                    "decay_rate": {
                        "type": "number",
                        "description": "Decay rate for importance (default: 0.01)",
                        "default": 0.01
                    },
                    "threshold": {
                        "type": "number",
                        "description": "Importance threshold for pruning (default: 0.05)",
                        "default": 0.05
                    }
                },
                "required": ["action"]
            },
            execute_func=maintenance
        )
    }

    return tools


def register_kg_tools_with_agent(agent: Any, kg: AURAKnowledgeGraph):
    """
    Helper to register all KG tools with an agent.

    Args:
        agent: The agent instance (must have a tools dict)
        kg: The knowledge graph instance
    """
    tools = create_kg_tools(kg)

    for name, tool in tools.items():
        # Create a tool wrapper compatible with agent's tool interface
        class KGToolWrapper:
            def __init__(self, mcp_tool: MCPTool):
                self.name = mcp_tool.name
                self.description = mcp_tool.description
                self._execute = mcp_tool.execute_func

            def execute(self, **kwargs) -> Dict:
                return self._execute(kwargs)

        agent.tools[f"kg_{name}"] = KGToolWrapper(tool)
        logger.info(f"[KG] Registered tool: kg_{name}")
