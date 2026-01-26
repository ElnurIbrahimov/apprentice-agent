"""
Hybrid Memory System for Aura
Combines ChromaDB vector search with Knowledge Graph relationships.

Author: Aura Development Team
Created: 2025-01-26
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# Support both relative and absolute imports
try:
    from .knowledge_graph import KnowledgeGraphTool, get_knowledge_graph, Node
    from .kg_extractor import KnowledgeExtractor
except ImportError:
    from knowledge_graph import KnowledgeGraphTool, get_knowledge_graph, Node
    from kg_extractor import KnowledgeExtractor


@dataclass
class MemoryResult:
    """Represents a memory retrieval result."""
    content: str
    source: str           # "vector" or "graph"
    relevance: float      # 0-1 score
    node: Optional[Node] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class HybridMemory:
    """
    Hybrid memory system combining vector and graph-based retrieval.

    Uses ChromaDB for semantic similarity search and
    Knowledge Graph for relationship-based retrieval.
    """

    def __init__(self, chromadb=None, knowledge_graph: Optional[KnowledgeGraphTool] = None):
        """
        Initialize hybrid memory.

        Args:
            chromadb: ChromaDB instance for vector search
            knowledge_graph: Knowledge graph instance
        """
        self.vector_db = chromadb
        self.graph = knowledge_graph or get_knowledge_graph()
        self.extractor = KnowledgeExtractor()

    def remember(
        self,
        content: str,
        node_type: str = "concept",
        relations: Optional[List[Dict]] = None,
        metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Store content in both vector and graph systems.

        Args:
            content: Text content to remember
            node_type: Type of node to create
            relations: List of relationships to create
                       [{"target": "label", "type": "edge_type"}]
            metadata: Additional metadata

        Returns:
            Dict with created node and any vector IDs
        """
        result = {
            "vector_id": None,
            "node": None,
            "edges_created": 0
        }

        # 1. Extract label from content
        label = self._extract_label(content)

        # 2. Add to graph
        node = self.graph.add_node(
            node_type=node_type,
            label=label,
            properties={
                "full_content": content,
                **(metadata or {})
            },
            source="hybrid_memory"
        )
        result["node"] = node

        # 3. Add to vector DB if available
        if self.vector_db:
            try:
                # Use ChromaDB's add method
                vector_id = self.vector_db.add(content, metadata={
                    "node_id": node.id,
                    "node_type": node_type,
                    "label": label
                })
                result["vector_id"] = vector_id
            except Exception as e:
                print(f"[HybridMemory] Vector store error: {e}")

        # 4. Add relationships if provided
        if relations:
            for rel in relations:
                target = rel.get("target")
                edge_type = rel.get("type", "relates_to")

                if target:
                    # Find or create target node
                    target_node = self.graph.get_node_by_label(target)
                    if not target_node:
                        target_node = self.graph.add_node(
                            node_type="concept",
                            label=target,
                            source="inferred"
                        )

                    # Create edge
                    edge = self.graph.add_edge(
                        node.id,
                        target_node.id,
                        edge_type
                    )
                    if edge:
                        result["edges_created"] += 1

        return result

    def recall(
        self,
        query: str,
        use_graph: bool = True,
        use_vectors: bool = True,
        limit: int = 10,
        min_relevance: float = 0.3
    ) -> List[MemoryResult]:
        """
        Multi-strategy retrieval from memory.

        Args:
            query: Search query
            use_graph: Whether to search knowledge graph
            use_vectors: Whether to search vector DB
            limit: Maximum results to return
            min_relevance: Minimum relevance score

        Returns:
            List of MemoryResult objects sorted by relevance
        """
        results = []

        # 1. Vector search (semantic similarity)
        if use_vectors and self.vector_db:
            try:
                vector_results = self.vector_db.search(query, k=limit)
                for doc in vector_results:
                    results.append(MemoryResult(
                        content=doc.get("content", ""),
                        source="vector",
                        relevance=doc.get("score", 0.5),
                        metadata=doc.get("metadata", {})
                    ))
            except Exception as e:
                print(f"[HybridMemory] Vector search error: {e}")

        # 2. Graph search (relationship traversal)
        if use_graph:
            # Find seed nodes matching query
            seed_nodes = self.graph.find_nodes(query, limit=5)

            for seed in seed_nodes:
                # Add seed node
                results.append(MemoryResult(
                    content=seed.properties.get("full_content", seed.label),
                    source="graph",
                    relevance=seed.confidence,
                    node=seed,
                    metadata=seed.properties
                ))

                # Get related nodes (1-2 hops)
                related = self.graph.get_related(seed.id, depth=2, min_weight=0.4)

                for node in related.get("nodes", []):
                    if node.id != seed.id:
                        results.append(MemoryResult(
                            content=node.properties.get("full_content", node.label),
                            source="graph",
                            relevance=node.confidence * 0.8,  # Slightly lower for related
                            node=node,
                            metadata=node.properties
                        ))

        # 3. Merge and rank results
        return self._rank_results(results, query, limit, min_relevance)

    def _rank_results(
        self,
        results: List[MemoryResult],
        query: str,
        limit: int,
        min_relevance: float
    ) -> List[MemoryResult]:
        """Merge, deduplicate, and rank results."""
        # Deduplicate by content
        seen_content = set()
        unique_results = []

        for result in results:
            content_key = result.content[:100].lower()
            if content_key not in seen_content:
                seen_content.add(content_key)
                unique_results.append(result)

        # Filter by minimum relevance
        filtered = [r for r in unique_results if r.relevance >= min_relevance]

        # Sort by relevance
        filtered.sort(key=lambda r: r.relevance, reverse=True)

        # Boost results that match query more directly
        query_lower = query.lower()
        for result in filtered:
            if query_lower in result.content.lower():
                result.relevance = min(1.0, result.relevance + 0.2)

        # Re-sort and return top results
        filtered.sort(key=lambda r: r.relevance, reverse=True)
        return filtered[:limit]

    def _extract_label(self, content: str) -> str:
        """Extract a short label from content."""
        # Take first sentence or first N words
        first_sentence = content.split('.')[0].strip()

        if len(first_sentence) <= 50:
            return first_sentence

        # Take first few words
        words = content.split()[:5]
        return " ".join(words)

    def learn_from_interaction(
        self,
        user_msg: str,
        aura_response: str,
        tools_used: Optional[List[str]] = None,
        success: bool = True
    ) -> Dict[str, Any]:
        """
        Learn from a conversation interaction.

        Extracts knowledge and stores in both systems.

        Args:
            user_msg: User's message
            aura_response: Aura's response
            tools_used: List of tools that were used
            success: Whether the interaction was successful

        Returns:
            Summary of what was learned
        """
        learned = {
            "entities_added": 0,
            "relationships_added": 0,
            "tool_patterns": 0
        }

        # 1. Extract entities from conversation
        extracted = self.extractor.extract_from_dialogue(user_msg, aura_response)

        # 2. Add entities to graph
        entity_nodes = {}
        for entity in extracted.get("entities", []):
            node = self.graph.add_node(
                node_type=entity.get("type", "concept"),
                label=entity.get("label"),
                properties=entity.get("properties", {}),
                source="conversation"
            )
            entity_nodes[entity.get("label", "").lower()] = node
            learned["entities_added"] += 1

        # 3. Add relationships
        for rel in extracted.get("relationships", []):
            source_label = rel.get("source", "").lower()
            target_label = rel.get("target", "").lower()

            source_node = entity_nodes.get(source_label) or \
                          self.graph.get_node_by_label(source_label)
            target_node = entity_nodes.get(target_label) or \
                          self.graph.get_node_by_label(target_label)

            if source_node and target_node:
                self.graph.add_edge(
                    source_node.id,
                    target_node.id,
                    rel.get("type", "relates_to"),
                    properties={"evidence": rel.get("evidence", "")}
                )
                learned["relationships_added"] += 1

        # 4. Learn tool usage patterns
        if tools_used:
            for tool_name in tools_used:
                result = self.graph.learn_from_tool_use(
                    tool_name,
                    user_msg,
                    aura_response,
                    success
                )
                learned["tool_patterns"] += 1

        return learned

    def get_context_for_query(self, query: str, max_tokens: int = 500) -> str:
        """
        Get relevant context from memory for a query.

        Returns formatted string suitable for LLM context.
        """
        # Extract topics
        topics = self.extractor.extract_topics(query)

        context_parts = []
        total_chars = 0

        # Search for each topic
        for topic in topics[:3]:  # Limit topics
            results = self.recall(topic, limit=3)

            for result in results:
                # Estimate tokens (rough: 4 chars per token)
                content = result.content
                if total_chars + len(content) > max_tokens * 4:
                    break

                if result.node:
                    context_parts.append(f"- {result.node.format_display()}: {content[:200]}")
                else:
                    context_parts.append(f"- {content[:200]}")

                total_chars += len(content)

        if not context_parts:
            return ""

        return "Relevant knowledge:\n" + "\n".join(context_parts)

    def consolidate(self) -> Dict[str, Any]:
        """
        Run memory consolidation.

        Merges similar nodes, prunes weak edges,
        and strengthens frequently used connections.
        """
        # Consolidate graph
        graph_result = self.graph.consolidate()

        # Save graph
        self.graph.save()

        return {
            "graph": graph_result
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        stats = {
            "graph": self.graph.get_stats(),
            "vector_db": None
        }

        if self.vector_db:
            try:
                stats["vector_db"] = {
                    "count": self.vector_db.count() if hasattr(self.vector_db, 'count') else "unknown"
                }
            except:
                pass

        return stats


def create_hybrid_memory(chromadb=None, knowledge_graph=None) -> HybridMemory:
    """Create a hybrid memory instance."""
    return HybridMemory(chromadb, knowledge_graph)


# Export
__all__ = ["HybridMemory", "create_hybrid_memory", "MemoryResult"]
