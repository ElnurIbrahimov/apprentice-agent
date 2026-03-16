"""
Query engine for Knowledge Graph retrieval.

Provides multiple retrieval strategies:
1. Entity search (text matching)
2. Relationship traversal (graph walks)
3. Community detection (entity clusters)
4. Hybrid search (combine with vector similarity)
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

from .graph_database import AURAKnowledgeGraph
from .schema import EntityType

logger = logging.getLogger(__name__)


class QueryMode(Enum):
    """Query modes for different retrieval strategies."""
    ENTITY = "entity"  # Direct entity lookup
    TRAVERSAL = "traversal"  # Graph walk from seed entities
    GLOBAL = "global"  # High-level overview
    HYBRID = "hybrid"  # Combine multiple strategies
    TEMPORAL = "temporal"  # Point-in-time query


@dataclass
class QueryResult:
    """Result from a knowledge graph query."""
    entities: List[Dict]
    relationships: List[Dict]
    context_string: str
    query_mode: QueryMode
    metadata: Dict


class KGQueryEngine:
    """
    Query engine for Knowledge Graph retrieval.

    Designed to provide context for AURA's LLM responses.
    """

    def __init__(self, knowledge_graph: AURAKnowledgeGraph):
        self.kg = knowledge_graph

    def query(
        self,
        query_text: str,
        mode: QueryMode = QueryMode.HYBRID,
        max_entities: int = 10,
        max_hops: int = 2,
        entity_type: Optional[EntityType] = None,
        at_time: Optional[int] = None
    ) -> QueryResult:
        """
        Query the knowledge graph.

        Args:
            query_text: Natural language query
            mode: Query mode (entity, traversal, global, hybrid, temporal)
            max_entities: Maximum entities to return
            max_hops: Maximum relationship hops for traversal
            entity_type: Optional filter by entity type
            at_time: Epoch timestamp for temporal/time-travel queries

        Returns:
            QueryResult with entities, relationships, and formatted context
        """
        if mode == QueryMode.TEMPORAL:
            return self._temporal_query(query_text, at_time, max_entities, entity_type)
        elif mode == QueryMode.ENTITY:
            return self._entity_query(query_text, max_entities, entity_type)
        elif mode == QueryMode.TRAVERSAL:
            return self._traversal_query(query_text, max_entities, max_hops, entity_type)
        elif mode == QueryMode.GLOBAL:
            return self._global_query(max_entities)
        else:  # HYBRID
            return self._hybrid_query(query_text, max_entities, max_hops, entity_type)

    def _entity_query(
        self,
        query_text: str,
        max_entities: int,
        entity_type: Optional[EntityType] = None
    ) -> QueryResult:
        """Direct entity search."""
        entities = self.kg.query_entities(query_text, entity_type=entity_type, limit=max_entities)

        return QueryResult(
            entities=entities,
            relationships=[],
            context_string=self._format_entities(entities),
            query_mode=QueryMode.ENTITY,
            metadata={"query": query_text, "matches": len(entities)}
        )

    def _traversal_query(
        self,
        query_text: str,
        max_entities: int,
        max_hops: int,
        entity_type: Optional[EntityType] = None
    ) -> QueryResult:
        """Graph traversal from seed entities."""
        # Find seed entities
        seeds = self.kg.query_entities(query_text, entity_type=entity_type, limit=3)

        all_entities = list(seeds)
        all_relationships = []
        seen_ids = {e["id"] for e in seeds}

        # Traverse from each seed
        for seed in seeds:
            related = self.kg.get_related_entities(
                seed["id"],
                hops=max_hops,
                limit=max_entities // max(len(seeds), 1)
            )

            for entity in related:
                if entity["id"] not in seen_ids:
                    all_entities.append(entity)
                    seen_ids.add(entity["id"])

            # Get relationships
            rels = self.kg.get_relationships(seed["id"])
            all_relationships.extend(rels)

        return QueryResult(
            entities=all_entities[:max_entities],
            relationships=all_relationships,
            context_string=self._format_graph(all_entities[:max_entities], all_relationships),
            query_mode=QueryMode.TRAVERSAL,
            metadata={"seeds": [s["name"] for s in seeds], "total_related": len(all_entities)}
        )

    def _global_query(self, max_entities: int) -> QueryResult:
        """High-level overview of important entities."""
        # Get most important entities
        result = self.kg.execute_cypher(f"""
            MATCH (e:Entity)
            RETURN e.id, e.name, e.entity_type, e.description, e.importance
            ORDER BY e.importance DESC
            LIMIT {int(max(1, min(100, max_entities)))}
        """)

        entities = []
        for row in result:
            entities.append({
                "id": row[0],
                "name": row[1],
                "entity_type": row[2],
                "description": row[3],
                "importance": row[4]
            })

        # Get entity type distribution
        type_dist = self.kg.execute_cypher("""
            MATCH (e:Entity)
            RETURN e.entity_type, COUNT(e) as count
            ORDER BY count DESC
        """)

        return QueryResult(
            entities=entities,
            relationships=[],
            context_string=self._format_global(entities, type_dist),
            query_mode=QueryMode.GLOBAL,
            metadata={"type_distribution": {row[0]: row[1] for row in type_dist}}
        )

    def _hybrid_query(
        self,
        query_text: str,
        max_entities: int,
        max_hops: int,
        entity_type: Optional[EntityType] = None
    ) -> QueryResult:
        """Combine entity search and traversal."""
        # Entity search
        entity_result = self._entity_query(query_text, max_entities // 2, entity_type)

        # Traversal from found entities
        if entity_result.entities:
            traversal_result = self._traversal_query(
                query_text,
                max_entities // 2,
                max_hops,
                entity_type
            )
        else:
            traversal_result = QueryResult([], [], "", QueryMode.TRAVERSAL, {})

        # Merge results, prioritizing direct matches
        all_entities = entity_result.entities + traversal_result.entities
        seen_ids = set()
        unique_entities = []
        for e in all_entities:
            if e["id"] not in seen_ids:
                unique_entities.append(e)
                seen_ids.add(e["id"])

        return QueryResult(
            entities=unique_entities[:max_entities],
            relationships=traversal_result.relationships,
            context_string=self._format_graph(
                unique_entities[:max_entities],
                traversal_result.relationships
            ),
            query_mode=QueryMode.HYBRID,
            metadata={
                "entity_matches": len(entity_result.entities),
                "traversal_matches": len(traversal_result.entities)
            }
        )

    def _temporal_query(
        self,
        query_text: str,
        at_time: Optional[int],
        max_entities: int,
        entity_type: Optional[EntityType] = None,
    ) -> QueryResult:
        """Point-in-time query: find entities and their relationships at a specific time."""
        import time as _time

        if at_time is None:
            at_time = int(_time.time())

        # Find seed entities matching the text
        entities = self.kg.query_entities(query_text, entity_type=entity_type, limit=max_entities)
        all_relationships = []

        for entity in entities:
            rels = self.kg.get_relationships_at_time(entity["id"], at_time)
            all_relationships.extend(rels)

        return QueryResult(
            entities=entities,
            relationships=all_relationships,
            context_string=self._format_temporal(entities, all_relationships, at_time),
            query_mode=QueryMode.TEMPORAL,
            metadata={"at_time": at_time, "matches": len(entities), "relationships": len(all_relationships)},
        )

    def _format_temporal(self, entities: List[Dict], relationships: List[Dict], at_time: int) -> str:
        """Format temporal query results."""
        from datetime import datetime

        time_str = datetime.fromtimestamp(at_time).strftime("%Y-%m-%d %H:%M:%S")
        lines = [f"KNOWLEDGE GRAPH (point-in-time: {time_str}):"]

        if entities:
            lines.append("\nEntities:")
            for e in entities:
                desc = e.get("description", "") or ""
                lines.append(f"  [{e['entity_type']}] {e['name']}")
                if desc:
                    lines.append(f"    {desc}")

        if relationships:
            lines.append("\nRelationships (active at that time):")
            seen = set()
            for r in relationships:
                key = f"{r['source']}-{r['relationship']}-{r['target']}"
                if key not in seen:
                    seen.add(key)
                    lines.append(f"  {r['source']} --[{r['relationship']}]--> {r['target']}")

        return "\n".join(lines)

    def _format_entities(self, entities: List[Dict]) -> str:
        """Format entities as context string."""
        if not entities:
            return ""

        lines = ["KNOWLEDGE GRAPH - Relevant Entities:"]
        for e in entities:
            desc = e.get('description', '') or 'No description'
            importance = e.get('importance', 0)
            lines.append(f"  [{e['entity_type']}] {e['name']} (importance: {importance:.2f})")
            if desc and desc != 'No description':
                lines.append(f"    └─ {desc}")

        return "\n".join(lines)

    def _format_graph(self, entities: List[Dict], relationships: List[Dict]) -> str:
        """Format entities and relationships as context string."""
        lines = ["KNOWLEDGE GRAPH CONTEXT:"]

        if entities:
            lines.append("\nEntities:")
            for e in entities:
                desc = e.get('description', '') or ''
                lines.append(f"  [{e['entity_type']}] {e['name']}")
                if desc:
                    lines.append(f"    └─ {desc}")

        if relationships:
            lines.append("\nRelationships:")
            seen_rels = set()
            for r in relationships:
                rel_key = f"{r['source']}-{r['relationship']}-{r['target']}"
                if rel_key not in seen_rels:
                    seen_rels.add(rel_key)
                    lines.append(f"  {r['source']} --[{r['relationship']}]--> {r['target']}")

        return "\n".join(lines)

    def _format_global(self, entities: List[Dict], type_distribution: List) -> str:
        """Format global overview."""
        lines = ["KNOWLEDGE GRAPH OVERVIEW:"]

        if type_distribution:
            lines.append("\nEntity Types:")
            for row in type_distribution:
                lines.append(f"  {row[0]}: {row[1]} entities")

        if entities:
            lines.append("\nMost Important Entities:")
            for e in entities[:5]:
                importance = e.get('importance', 0)
                lines.append(f"  [{e['entity_type']}] {e['name']} (importance: {importance:.2f})")

        return "\n".join(lines)

    def answer_graph_question(self, question: str) -> str:
        """
        Answer a question using only the knowledge graph.
        Useful for "What do I know about X?" type questions.
        """
        # Extract keywords by stripping common question/stop words
        stop_words = {
            "what", "who", "where", "when", "why", "how", "which",
            "is", "are", "was", "were", "do", "does", "did",
            "the", "a", "an", "of", "in", "on", "at", "to", "for",
            "and", "or", "but", "not", "with", "about", "can", "could",
            "will", "would", "should", "have", "has", "had",
            "tell", "me", "know", "think", "my", "your", "i", "you",
        }
        words = question.lower().strip().rstrip("?!.").split()
        keywords = [w for w in words if w not in stop_words]
        search_query = " ".join(keywords) if keywords else question

        result = self.query(search_query, mode=QueryMode.HYBRID, max_entities=10)

        if not result.entities:
            return "I don't have any information about that in my knowledge graph."

        return result.context_string

    def find_path(self, source_name: str, target_name: str, max_hops: int = 4) -> Optional[List[Dict]]:
        """
        Find shortest path between two entities.

        Returns list of entities in path, or None if no path exists.
        """
        # Find source entity
        source_entities = self.kg.query_entities(source_name, limit=1)
        if not source_entities:
            return None

        # Find target entity
        target_entities = self.kg.query_entities(target_name, limit=1)
        if not target_entities:
            return None

        source_id = source_entities[0]["id"]
        target_id = target_entities[0]["id"]

        # Try to find path using Cypher (sanitize IDs to prevent injection)
        import re
        _safe_id = re.compile(r'^[a-zA-Z0-9_-]+$')
        if not _safe_id.match(source_id) or not _safe_id.match(target_id):
            logger.warning(f"[QueryEngine] Invalid entity IDs for path query: {source_id}, {target_id}")
            return None
        max_hops = int(max(1, min(10, max_hops)))
        try:
            result = self.kg.execute_cypher(f"""
                MATCH path = shortestPath(
                    (s:Entity {{id: '{source_id}'}})-[*1..{max_hops}]-(t:Entity {{id: '{target_id}'}})
                )
                RETURN nodes(path)
            """)

            if result:
                # Parse path nodes
                return result[0][0] if result[0] else None
        except Exception as e:
            logger.debug(f"Path finding error: {e}")

        return None

    def get_entity_summary(self, entity_name: str) -> str:
        """
        Get a comprehensive summary of an entity.
        """
        entities = self.kg.query_entities(entity_name, limit=1)
        if not entities:
            return f"No entity found matching '{entity_name}'"

        entity = entities[0]
        entity_id = entity["id"]

        # Get relationships
        outgoing = self.kg.get_relationships(entity_id, direction="outgoing")
        incoming = self.kg.get_relationships(entity_id, direction="incoming")

        lines = [
            f"=== {entity['name']} ===",
            f"Type: {entity['entity_type']}",
            f"Importance: {entity.get('importance', 0):.2f}",
            f"Access Count: {entity.get('access_count', 0)}",
        ]

        if entity.get('description'):
            lines.append(f"Description: {entity['description']}")

        if outgoing:
            lines.append("\nOutgoing Relationships:")
            for r in outgoing[:10]:
                lines.append(f"  --[{r['relationship']}]--> {r['target']}")

        if incoming:
            lines.append("\nIncoming Relationships:")
            for r in incoming[:10]:
                lines.append(f"  {r['source']} --[{r['relationship']}]-->")

        return "\n".join(lines)
