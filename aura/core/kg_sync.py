"""KG Sync Bridge — keeps NetworkX runtime KG and Kuzu persistent KG in sync.

On init, loads all Kuzu entities/relations into NetworkX so the agent starts
with the full persistent knowledge.  When the agent adds entities or relations
to NetworkX during a conversation, the bridge writes them through to Kuzu in
the background so they persist across restarts.

All Kuzu writes are non-blocking (bg_pool) and failure-safe (try/except).
"""

import json
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Map Kuzu EntityType values -> NetworkX NODE_TYPES keys
_KUZU_TO_NX_TYPE = {
    "Person": "person",
    "Project": "project",
    "Technology": "concept",
    "Company": "entity",
    "Concept": "concept",
    "Task": "entity",
    "Location": "location",
    "Event": "event",
    "Document": "file",
    "Skill": "skill",
}

# Reverse: NetworkX node type -> closest Kuzu EntityType value
_NX_TO_KUZU_TYPE = {
    "concept": "Concept",
    "entity": "Concept",
    "person": "Person",
    "project": "Project",
    "tool": "Technology",
    "event": "Event",
    "emotion": "Concept",
    "skill": "Skill",
    "location": "Location",
    "file": "Document",
}

# Map NetworkX edge types -> Kuzu relationship_type strings
_NX_TO_KUZU_REL = {
    "relates_to": "RELATES_TO",
    "is_a": "RELATES_TO",
    "part_of": "RELATES_TO",
    "causes": "RELATES_TO",
    "solves": "RELATES_TO",
    "created_by": "RELATES_TO",
    "uses": "USES",
    "triggers": "RELATES_TO",
    "learned_from": "RELATES_TO",
    "preceded_by": "RELATES_TO",
    "followed_by": "RELATES_TO",
    "conflicts_with": "RELATES_TO",
    "strengthens": "RELATES_TO",
    "weakens": "RELATES_TO",
    "knows": "KNOWS",
    "works_on": "WORKS_ON",
    "located_at": "LOCATED_AT",
}


class KGSyncBridge:
    """Bidirectional sync bridge between NetworkX (runtime) and Kuzu (persistent) KGs."""

    def __init__(self, networkx_kg, kuzu_kg):
        """
        Args:
            networkx_kg: KnowledgeGraphTool instance (aura.tools.knowledge_graph)
            kuzu_kg: AURAKnowledgeGraph instance (aura_knowledge_graph.graph_database)
        """
        self._nx = networkx_kg
        self._kuzu = kuzu_kg
        self._synced_from_kuzu = False

    # ------------------------------------------------------------------
    # Kuzu -> NetworkX  (call once at init)
    # ------------------------------------------------------------------

    def sync_from_kuzu(self) -> int:
        """Load all Kuzu entities and active relations into NetworkX.

        Returns the number of entities loaded.
        """
        if self._synced_from_kuzu:
            return 0

        loaded_entities = 0
        loaded_relations = 0

        try:
            # Load all entities
            rows = self._kuzu.execute_cypher(
                "MATCH (e:Entity) "
                "RETURN e.id, e.name, e.entity_type, e.description, "
                "       e.importance, e.properties "
                "ORDER BY e.importance DESC"
            )

            for row in rows:
                kuzu_id, name, entity_type, description, importance, props_json = row
                if not name:
                    continue

                nx_type = _KUZU_TO_NX_TYPE.get(entity_type, "concept")
                properties = {}
                if props_json:
                    try:
                        properties = json.loads(props_json)
                    except (json.JSONDecodeError, TypeError):
                        pass
                if description:
                    properties["description"] = description
                properties["_kuzu_id"] = kuzu_id

                try:
                    self._nx.add_node(
                        node_type=nx_type,
                        label=name,
                        properties=properties,
                        confidence=min(1.0, max(0.1, importance or 0.5)),
                        source="kuzu_sync",
                    )
                    loaded_entities += 1
                except (ValueError, TypeError) as e:
                    logger.debug("[KGSync] Skipped entity %s: %s", name, e)

            # Load all active relations
            rel_rows = self._kuzu.execute_cypher(
                "MATCH (s:Entity)-[r:RELATES_TO]->(t:Entity) "
                "WHERE r.is_active = true "
                "RETURN s.name, t.name, r.relationship_type, r.weight"
            )

            for rel_row in rel_rows:
                src_name, tgt_name, rel_type, weight = rel_row
                if not src_name or not tgt_name:
                    continue

                # Map Kuzu rel type to NX edge type (lowercase)
                nx_edge_type = (rel_type or "RELATES_TO").lower()

                try:
                    self._nx.add_edge(
                        source_id=src_name,  # add_edge resolves labels
                        target_id=tgt_name,
                        edge_type=nx_edge_type,
                        weight=min(1.0, max(0.1, weight or 0.5)),
                        properties={"_from_kuzu": True},
                    )
                    loaded_relations += 1
                except (ValueError, TypeError) as e:
                    logger.debug("[KGSync] Skipped relation %s->%s: %s", src_name, tgt_name, e)

            self._synced_from_kuzu = True
            logger.info(
                "[KGSync] Loaded %d entities and %d relations from Kuzu into NetworkX",
                loaded_entities, loaded_relations,
            )

        except Exception as e:
            logger.warning("[KGSync] sync_from_kuzu failed: %s", e)

        return loaded_entities

    # ------------------------------------------------------------------
    # NetworkX -> Kuzu  (called on every add, runs in background)
    # ------------------------------------------------------------------

    def sync_entity_to_kuzu(
        self,
        name: str,
        entity_type: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Write an entity to Kuzu in the background. Non-blocking, failure-safe."""
        try:
            from aura.pools import bg_pool
            bg_pool().submit(self._write_entity_to_kuzu, name, entity_type, properties or {})
        except Exception as e:
            logger.debug("[KGSync] Failed to submit entity sync: %s", e)

    def sync_relation_to_kuzu(
        self,
        source_label: str,
        target_label: str,
        relation_type: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Write a relation to Kuzu in the background. Non-blocking, failure-safe."""
        try:
            from aura.pools import bg_pool
            bg_pool().submit(
                self._write_relation_to_kuzu,
                source_label, target_label, relation_type, properties or {},
            )
        except Exception as e:
            logger.debug("[KGSync] Failed to submit relation sync: %s", e)

    # ------------------------------------------------------------------
    # Private helpers (run on bg_pool threads)
    # ------------------------------------------------------------------

    def _write_entity_to_kuzu(
        self, name: str, nx_type: str, properties: Dict[str, Any]
    ) -> None:
        """Actual Kuzu write — runs on background thread."""
        try:
            from aura_knowledge_graph.graph_database import Entity
            from aura_knowledge_graph.schema import EntityType

            kuzu_type_str = _NX_TO_KUZU_TYPE.get(nx_type, "Concept")
            try:
                kuzu_etype = EntityType(kuzu_type_str)
            except ValueError:
                kuzu_etype = EntityType.CONCEPT

            # Strip internal props before storing
            clean_props = {
                k: v for k, v in properties.items()
                if not k.startswith("_") and k != "description"
            }

            entity = Entity(
                name=name,
                entity_type=kuzu_etype,
                description=properties.get("description", ""),
                properties=clean_props,
                importance=0.5,
            )
            self._kuzu.add_entity(entity)
            logger.debug("[KGSync] Synced entity to Kuzu: %s", name)

        except Exception as e:
            logger.debug("[KGSync] Entity write to Kuzu failed (%s): %s", name, e)

    def _write_relation_to_kuzu(
        self,
        source_label: str,
        target_label: str,
        nx_rel_type: str,
        properties: Dict[str, Any],
    ) -> None:
        """Actual Kuzu relation write — runs on background thread."""
        try:
            from aura_knowledge_graph.graph_database import Entity, Relationship
            from aura_knowledge_graph.schema import EntityType

            # Resolve source and target entity IDs in Kuzu
            src_entity = self._kuzu.get_entity_by_name(source_label)
            tgt_entity = self._kuzu.get_entity_by_name(target_label)

            if not src_entity or not tgt_entity:
                logger.debug(
                    "[KGSync] Skipping relation sync — entity not found in Kuzu: %s -> %s",
                    source_label, target_label,
                )
                return

            kuzu_rel_type = _NX_TO_KUZU_REL.get(nx_rel_type, "RELATES_TO")

            rel = Relationship(
                source_id=src_entity["id"],
                target_id=tgt_entity["id"],
                relationship_type=kuzu_rel_type,
                weight=properties.get("weight", 0.5) if isinstance(properties.get("weight"), (int, float)) else 0.5,
                evidence=properties.get("context", ""),
            )
            self._kuzu.add_relationship(rel)
            logger.debug(
                "[KGSync] Synced relation to Kuzu: %s -[%s]-> %s",
                source_label, kuzu_rel_type, target_label,
            )

        except Exception as e:
            logger.debug(
                "[KGSync] Relation write to Kuzu failed (%s->%s): %s",
                source_label, target_label, e,
            )
