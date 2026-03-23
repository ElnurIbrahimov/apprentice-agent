"""Knowledge Graph Brain mixin — KG commands, stats, queries, consolidation.

Extracted from agent.py (2026-03-23) to reduce class size.
All methods assume self has: kg_brain, kg_bridge, kg_query_engine, tools, _kg_queue_lock.
"""

import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

# Safe imports — mirrors agent.py top-level
try:
    from aura_knowledge_graph import QueryMode
except ImportError:
    QueryMode = None


class KGBrainMixin:
    """Mixin providing Knowledge Graph Brain methods for ApprenticeAgent."""

    # ------------------------------------------------------------------
    # Direct command handler (called from run() and _prepare_chat)
    # ------------------------------------------------------------------

    def _handle_knowledge_graph_command(self, message: str) -> Optional[str]:
        """Handle knowledge graph commands directly, bypassing the LLM.

        Supports both the legacy KnowledgeGraphTool and the new KG Brain.

        Args:
            message: The user's message

        Returns:
            Formatted result string if KG command, None otherwise
        """
        msg_lower = message.lower()

        kg_keywords = [
            'what do you know about', 'knowledge graph', 'show graph',
            'how is', 'related to', 'connected to', 'find path between',
            'what have you learned', 'consolidate memory', 'graph stats',
            'kg brain', 'kg stats', 'add to knowledge', 'remember that',
            'learn that', 'extract entities', 'kg query'
        ]

        if not any(kw in msg_lower for kw in kg_keywords):
            return None

        # Try KG Brain first (new system)
        if self.kg_brain is not None and self.kg_query_engine is not None:
            # Handle KG Brain specific commands
            if "kg brain" in msg_lower or "kg stats" in msg_lower:
                stats = self.kg_brain.get_statistics()
                bridge_stats = self.kg_bridge.get_statistics() if self.kg_bridge else {}
                return (
                    f"**Knowledge Graph Brain Statistics**\n"
                    f"- Total Entities: {stats.get('total_entities', 0)}\n"
                    f"- Total Relationships: {stats.get('total_relationships', 0)}\n"
                    f"- Average Importance: {stats.get('average_importance', 0):.2f}\n"
                    f"- Entity Types: {stats.get('entity_type_distribution', {})}\n"
                    f"- Entities Extracted: {bridge_stats.get('total_entities_extracted', 0)}\n"
                    f"- Extractions Triggered: {bridge_stats.get('total_extractions_triggered', 0)}"
                )

            if "what do you know about" in msg_lower:
                topic = msg_lower.split("what do you know about")[-1].strip().rstrip("?")
                # Query KG Brain
                result = self.kg_query_engine.query(topic, mode=QueryMode.HYBRID, max_entities=10)
                if result.entities:
                    return result.context_string
                # Fall through to legacy KG if no results

            if "how is" in msg_lower and "related to" in msg_lower:
                parts = msg_lower.replace("?", "").split("related to")
                if len(parts) == 2:
                    source = parts[0].replace("how is", "").strip()
                    target = parts[1].strip()
                    # Try to find path in KG Brain
                    path = self.kg_query_engine.find_path(source, target)
                    if path:
                        return f"Connection found: {path}"

            if "extract entities" in msg_lower or "learn that" in msg_lower or "remember that" in msg_lower:
                # Force extraction from message
                text_to_extract = message.split("that", 1)[-1].strip() if "that" in message else message
                if self.kg_bridge:
                    entity_ids = self.kg_bridge.force_extract(text_to_extract, context="user command")
                    if entity_ids:
                        return f"Extracted and stored {len(entity_ids)} entities in knowledge graph."
                    return "No entities could be extracted from that text."

            if "consolidate memory" in msg_lower:
                # Apply decay and prune
                self.kg_brain.decay_importance(decay_rate=0.05)
                self.kg_brain.prune_low_importance(threshold=0.03)
                if self.kg_bridge:
                    self.kg_bridge.flush()
                stats = self.kg_brain.get_statistics()
                return f"Memory consolidated. Current state: {stats['total_entities']} entities, {stats['total_relationships']} relationships."

            # Generic KG Brain query
            if "kg query" in msg_lower:
                query = msg_lower.replace("kg query", "").strip()
                result = self.kg_query_engine.query(query, mode=QueryMode.HYBRID)
                return result.context_string if result.entities else "No matching entities found."

        # Fall back to legacy knowledge_graph tool
        if "knowledge_graph" not in self.tools:
            if self.kg_brain is None:
                return "Knowledge graph not available. Install kuzu: pip install kuzu"
            return "No results found in knowledge graph."

        kg = self.tools["knowledge_graph"]

        # Handle specific patterns with legacy tool
        if "what do you know about" in msg_lower:
            topic = msg_lower.split("what do you know about")[-1].strip().rstrip("?")
            result = kg.execute(f"query {topic}")
            if result.get("success") and result.get("results"):
                return "Here's what I know:\n" + "\n".join(result["results"])
            return f"I don't have much knowledge about '{topic}' yet."

        if "how is" in msg_lower and "related to" in msg_lower:
            parts = msg_lower.replace("?", "").split("related to")
            if len(parts) == 2:
                source = parts[0].replace("how is", "").strip()
                target = parts[1].strip()
                result = kg.execute(f"path {source} to {target}")
                if result.get("success") and result.get("path"):
                    return f"Connection: {result['path']}"
                return f"No direct connection found between '{source}' and '{target}'."

        if "consolidate memory" in msg_lower:
            result = kg.execute("consolidate")
            return f"Memory consolidated: {result.get('merged_nodes', 0)} nodes merged, {result.get('pruned_edges', 0)} edges pruned."

        if "graph stats" in msg_lower or "knowledge graph" in msg_lower:
            result = kg.execute("stats")
            if result.get("success"):
                return f"Knowledge Graph: {result['total_nodes']} nodes, {result['total_edges']} edges, {result['clusters']} clusters"

        # Generic query
        result = kg.execute(message)
        if result.get("success"):
            if result.get("results"):
                return "\n".join(result["results"])
            return str(result)

        return result.get("error", "Unknown error")

    # ------------------------------------------------------------------
    # Public API methods
    # ------------------------------------------------------------------

    def get_kg_brain_stats(self) -> dict:
        """Get Knowledge Graph Brain statistics.

        Returns:
            dict with KG Brain stats, or empty dict if not available
        """
        if self.kg_brain is None:
            return {"available": False, "reason": "KG Brain not initialized"}

        try:
            kg_stats = self.kg_brain.get_statistics()
            bridge_stats = self.kg_bridge.get_statistics() if self.kg_bridge else {}

            return {
                "available": True,
                "total_entities": kg_stats.get("total_entities", 0),
                "total_relationships": kg_stats.get("total_relationships", 0),
                "entity_types": kg_stats.get("entity_type_distribution", {}),
                "average_importance": kg_stats.get("average_importance", 0),
                "entities_extracted": bridge_stats.get("total_entities_extracted", 0),
                "extractions_triggered": bridge_stats.get("total_extractions_triggered", 0),
                "queue_size": bridge_stats.get("queue_size", 0)
            }
        except (AttributeError, KeyError, TypeError, OSError) as e:
            return {"available": False, "error": str(e)}

    def kg_brain_query(self, query: str, max_entities: int = 10) -> str:
        """Query the Knowledge Graph Brain.

        Args:
            query: Search query
            max_entities: Maximum entities to return

        Returns:
            Formatted context string with matching entities
        """
        if self.kg_query_engine is None:
            return "Knowledge Graph Brain not available."

        try:
            result = self.kg_query_engine.query(query, mode=QueryMode.HYBRID, max_entities=max_entities)
            return result.context_string if result.entities else "No matching entities found."
        except (AttributeError, KeyError, TypeError, ValueError, OSError) as e:
            return f"Query error: {e}"

    def kg_brain_add_knowledge(self, text: str, context: str = "manual") -> dict:
        """Manually add knowledge to the KG Brain.

        Args:
            text: Text to extract entities from
            context: Context for the extraction

        Returns:
            dict with extraction results
        """
        if self.kg_bridge is None:
            return {"success": False, "error": "KG Brain not available"}

        try:
            entity_ids = self.kg_bridge.force_extract(text, context=context)
            return {
                "success": True,
                "entities_extracted": len(entity_ids),
                "entity_ids": entity_ids
            }
        except (AttributeError, KeyError, TypeError, ValueError, OSError) as e:
            return {"success": False, "error": str(e)}

    def kg_brain_consolidate(self, decay_rate: float = 0.01, prune_threshold: float = 0.05) -> dict:
        """Consolidate KG Brain memory (decay importance, prune low entities).

        Args:
            decay_rate: Rate of importance decay
            prune_threshold: Threshold below which to prune entities

        Returns:
            dict with consolidation results
        """
        if self.kg_brain is None:
            return {"success": False, "error": "KG Brain not available"}

        try:
            # Flush any pending extractions
            if self.kg_bridge:
                self.kg_bridge.flush()

            # Get stats before
            stats_before = self.kg_brain.get_statistics()

            # Apply decay
            self.kg_brain.decay_importance(decay_rate)

            # Prune low importance
            self.kg_brain.prune_low_importance(prune_threshold)

            # Get stats after
            stats_after = self.kg_brain.get_statistics()

            return {
                "success": True,
                "entities_before": stats_before.get("total_entities", 0),
                "entities_after": stats_after.get("total_entities", 0),
                "entities_pruned": stats_before.get("total_entities", 0) - stats_after.get("total_entities", 0),
                "decay_rate": decay_rate,
                "prune_threshold": prune_threshold
            }
        except (AttributeError, KeyError, TypeError, ValueError, OSError) as e:
            return {"success": False, "error": str(e)}

    # ------------------------------------------------------------------
    # Episodic memory stubs (redirected to UnifiedMemory)
    # ------------------------------------------------------------------

    def get_episodic_memory_stats(self) -> dict:
        """Get memory stats (redirected to UnifiedMemory)."""
        try:
            from aura.memory.unified_memory import get_unified_memory
            um = get_unified_memory()
            stats = um.get_stats() if hasattr(um, 'get_stats') else {}
            return {"available": True, **stats}
        except ImportError as e:
            return {"available": False, "error": str(e)}
        except (AttributeError, KeyError, TypeError, OSError) as e:
            return {"available": False, "error": str(e)}

    def episodic_recall(self, query: str, limit: int = 5, time_filter: str = None) -> list:
        """Recall memories (redirected to UnifiedMemory)."""
        try:
            from aura.memory.unified_memory import get_unified_memory
            results = get_unified_memory().query(query, k=limit)
            return [
                {"id": r.source_id, "content": r.content[:300],
                 "type": r.metadata.get("memory_type", "conversation"),
                 "importance": r.importance, "score": r.score, "entities": []}
                for r in results
            ]
        except (ImportError, AttributeError, KeyError, TypeError, OSError) as e:
            logger.debug(f"[EpisodicRecall] Memory query failed: {e}")
            return []

    def episodic_time_travel(self, time_reference: str) -> dict:
        return {"success": False, "error": "Time travel consolidated — use memory query"}

    def episodic_record(self, content: str, episode_type: str = "conversation",
                        importance: float = 0.5, entities: list = None,
                        tools_used: list = None) -> dict:
        """Record a memory (redirected to UnifiedMemory)."""
        try:
            from aura.memory.unified_memory import get_unified_memory
            um = get_unified_memory()
            ids = um.store(content=content, source="episodic_record", importance=importance,
                           tags=entities or [], episode_type=episode_type)
            return {"success": True, "episode_id": ids.get("store", ""), "type": episode_type}
        except ImportError as e:
            return {"success": False, "error": str(e)}
        except (AttributeError, KeyError, TypeError, OSError) as e:
            return {"success": False, "error": str(e)}

    def episodic_get_context(self, query: str, include_timeline: bool = False) -> str:
        try:
            from aura.memory.unified_memory import get_unified_memory
            results = get_unified_memory().query(query, k=3)
            return "\n".join(f"- {r.content[:120]}" for r in results) if results else ""
        except (ImportError, AttributeError, KeyError, TypeError, OSError) as e:
            logger.debug(f"[EpisodicContext] Memory query failed: {e}")
            return ""

    def episodic_consolidate(self) -> dict:
        try:
            from aura.memory.store import get_memory_store
            store = get_memory_store()
            return {"success": True, "message": "Consolidation via UnifiedMemory"}
        except ImportError as e:
            return {"success": False, "error": str(e)}
        except (AttributeError, OSError) as e:
            return {"success": False, "error": str(e)}

    def episodic_get_health(self) -> dict:
        return {"status": "consolidated_into_unified_memory"}
