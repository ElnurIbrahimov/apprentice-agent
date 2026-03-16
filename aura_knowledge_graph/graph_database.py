"""Kuzu-based persistent Knowledge Graph.

Long-term persistent graph storage using Kuzu embedded database.
Exposed via MCP server for external tool integration.

Note: The runtime KG (aura/tools/knowledge_graph.py) operates independently
using NetworkX. Changes made here are NOT automatically reflected there
and vice versa. See sync utilities in the runtime KG module.

Key features:
- Embedded database (no server)
- Cypher query support
- Automatic schema initialization
- Importance decay (like Titans forgetting curve)

ARCHITECTURE NOTE — Dual KG design:
  - THIS FILE (aura_knowledge_graph/graph_database.py): Kuzu-backed persistent graph
    exposed as an MCP server (see aura_knowledge_graph/server.py). Intended for
    large-scale persistent storage and external clients via the MCP protocol.
  - aura/tools/knowledge_graph.py: In-process NetworkX graph used as a runtime tool
    by ApprenticeAgent. These are independent backends — not the same data store.
"""

import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import kuzu
    KUZU_AVAILABLE = True
except ImportError:
    KUZU_AVAILABLE = False
    kuzu = None

from .schema import EntityType, ALLOWED_RELATIONSHIPS

logger = logging.getLogger(__name__)


@dataclass
class Entity:
    """Represents a node in the knowledge graph."""
    name: str
    entity_type: EntityType
    description: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)
    importance: float = 0.5
    id: Optional[str] = None

    def __post_init__(self):
        if self.id is None:
            # Generate deterministic ID from name + type
            self.id = hashlib.md5(
                f"{self.name}:{self.entity_type.value}".lower().encode()
            ).hexdigest()[:12]


@dataclass
class Relationship:
    """Represents an edge in the knowledge graph."""
    source_id: str
    target_id: str
    relationship_type: str
    weight: float = 1.0
    evidence: str = ""  # Source text that supports this relationship
    # Bi-temporal fields
    valid_from: Optional[int] = None     # When this fact became true (epoch seconds)
    valid_to: Optional[int] = None       # When it stopped being true (None = current)
    ingested_at: Optional[int] = None    # When recorded (immutable)
    updated_at: Optional[int] = None     # Last modification time
    is_active: bool = True               # Soft-delete flag


class AURAKnowledgeGraph:
    """
    Kùzu-based Knowledge Graph for AURA.

    Design principles:
    1. Embedded database — no server required
    2. Disk-based storage — zero VRAM usage
    3. Importance decay — old unused entities fade
    4. Cypher queries — standard graph query language
    """

    def __init__(self, db_path: str = "./aura_data/knowledge_graph"):
        """Initialize the knowledge graph database."""
        if not KUZU_AVAILABLE:
            raise ImportError(
                "Kùzu is not installed. Install with: pip install kuzu"
            )

        self.db_path = Path(db_path)
        # Ensure parent directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Kuzu 0.4+ expects a database path (will create directory structure)
        # For newer versions, we pass the path directly
        try:
            self.db = kuzu.Database(str(self.db_path))
        except Exception as e:
            raise RuntimeError(
                f"AURAKnowledgeGraph init failed: {e}. "
                f"Delete DB dir '{self.db_path}' to reset."
            ) from e
        try:
            self.conn = kuzu.Connection(self.db)
        except Exception as e:
            raise RuntimeError(
                f"AURAKnowledgeGraph connection failed: {e}. "
                f"Delete DB dir '{self.db_path}' to reset."
            ) from e

        self._lock = threading.RLock()

        self._init_schema()
        self._migrate_temporal_schema()

        # Statistics
        self.total_entities_added = 0
        self.total_relationships_added = 0
        self.total_queries = 0

    def _init_schema(self):
        """Initialize graph schema with all entity types and relationships."""
        # Create Entity node table with all properties
        try:
            with self._lock:
                self.conn.execute("""
                    CREATE NODE TABLE IF NOT EXISTS Entity(
                        id STRING PRIMARY KEY,
                        name STRING,
                        entity_type STRING,
                        description STRING,
                        importance DOUBLE,
                        access_count INT64,
                        created_at INT64,
                        last_accessed INT64,
                        properties STRING
                    )
                """)
        except Exception as e:
            logger.debug(f"Entity table may exist: {e}")

        # Create relationship table
        try:
            with self._lock:
                self.conn.execute("""
                    CREATE REL TABLE IF NOT EXISTS RELATES_TO(
                        FROM Entity TO Entity,
                        relationship_type STRING,
                        weight DOUBLE,
                        evidence STRING,
                        created_at INT64
                    )
                """)
        except Exception as e:
            logger.debug(f"RELATES_TO table may exist: {e}")

        # Create Document node for source tracking
        try:
            with self._lock:
                self.conn.execute("""
                    CREATE NODE TABLE IF NOT EXISTS SourceDocument(
                        id STRING PRIMARY KEY,
                        content STRING,
                        source_type STRING,
                        timestamp INT64
                    )
                """)
        except Exception as e:
            logger.debug(f"SourceDocument table may exist: {e}")

        # Create MENTIONED_IN relationship
        try:
            with self._lock:
                self.conn.execute("""
                    CREATE REL TABLE IF NOT EXISTS MENTIONED_IN(
                        FROM Entity TO SourceDocument,
                        context STRING
                    )
                """)
        except Exception as e:
            logger.debug(f"MENTIONED_IN table may exist: {e}")

    def _migrate_temporal_schema(self):
        """Add bi-temporal columns to RELATES_TO if they don't exist yet (idempotent)."""
        columns = [
            ("valid_from", "INT64"),
            ("valid_to", "INT64"),
            ("ingested_at", "INT64"),
            ("updated_at", "INT64"),
            ("is_active", "BOOLEAN"),
        ]
        for col_name, col_type in columns:
            try:
                with self._lock:
                    self.conn.execute(
                        f"ALTER TABLE RELATES_TO ADD {col_name} {col_type}"
                    )
                logger.info(f"[KG] Migrated: added RELATES_TO.{col_name}")
            except Exception:
                pass  # Column already exists

        # Back-fill existing edges that have NULL temporal fields
        self._migrate_existing_edges()

    def _migrate_existing_edges(self):
        """Set default temporal values on pre-existing edges."""
        # Guard: check if migration has already been done by counting edges
        # that already have is_active set. If any exist, skip the full scan.
        try:
            with self._lock:
                check_result = self.conn.execute(
                    "MATCH ()-[r:RELATES_TO]->() WHERE r.is_active IS NOT NULL RETURN COUNT(r) LIMIT 1"
                )
                if check_result.has_next():
                    already_migrated = check_result.get_next()[0]
                    if already_migrated > 0:
                        logger.debug(f"[KG] Edge migration already done ({already_migrated} edges have is_active set), skipping")
                        return
        except Exception as e:
            logger.debug(f"[KG] Migration guard check failed (proceeding with migration): {e}")

        try:
            with self._lock:
                self.conn.execute("""
                    MATCH ()-[r:RELATES_TO]->()
                    WHERE r.is_active IS NULL
                    SET r.valid_from = r.created_at,
                        r.ingested_at = r.created_at,
                        r.updated_at = r.created_at,
                        r.is_active = true
                """)
        except Exception as e:
            logger.debug(f"[KG] Edge migration (may be no-op): {e}")

    def _escape_string(self, s: str) -> str:
        """Escape special characters in strings for Kuzu Cypher queries."""
        if s is None:
            return ""
        return s.replace("\\", "\\\\").replace("'", "\\'").replace('"', '\\"')

    def add_entity(self, entity: Entity) -> str:
        """
        Add or update an entity in the graph.
        Uses MERGE to update existing entities.
        Returns the entity ID.
        """
        now = int(time.time())
        props_json = json.dumps(entity.properties)
        importance_boost = entity.importance * 0.1

        try:
            with self._lock:
                # Check if entity exists
                result = self.conn.execute(
                    "MATCH (e:Entity {id: $id}) RETURN e.id",
                    parameters={"id": entity.id}
                )

                if result.has_next():
                    # Update existing entity - boost importance, capped at 1.0
                    self.conn.execute(
                        "MATCH (e:Entity {id: $id}) "
                        "SET e.importance = CASE WHEN e.importance + $boost > 1.0 THEN 1.0 ELSE e.importance + $boost END, "
                        "    e.access_count = e.access_count + 1, "
                        "    e.last_accessed = $now",
                        parameters={"id": entity.id, "boost": importance_boost, "now": now}
                    )

                    # Update description if current is empty and new one provided
                    if entity.description:
                        self.conn.execute(
                            "MATCH (e:Entity {id: $id}) "
                            "WHERE e.description = '' OR e.description IS NULL "
                            "SET e.description = $desc",
                            parameters={"id": entity.id, "desc": entity.description}
                        )
                else:
                    # Create new entity
                    self.conn.execute(
                        "CREATE (e:Entity {"
                        "    id: $id, name: $name, entity_type: $entity_type,"
                        "    description: $desc, importance: $importance,"
                        "    access_count: 1, created_at: $now,"
                        "    last_accessed: $now, properties: $props"
                        "})",
                        parameters={
                            "id": entity.id,
                            "name": entity.name,
                            "entity_type": entity.entity_type.value,
                            "desc": entity.description,
                            "importance": entity.importance,
                            "now": now,
                            "props": props_json,
                        }
                    )
                    self.total_entities_added += 1
                    logger.info(f"[KG] Added entity: {entity.name} ({entity.entity_type.value})")

        except Exception as e:
            logger.error(f"[KG] Error adding entity: {e}")

        return entity.id

    def add_relationship(self, rel: Relationship) -> bool:
        """
        Add or strengthen a relationship between entities.
        Returns True if successful.
        """
        now = int(time.time())
        weight_boost = rel.weight * 0.1

        try:
            with self._lock:
                # Check if an active relationship of this type already exists
                result = self.conn.execute(
                    "MATCH (s:Entity {id: $src})-[r:RELATES_TO]->(t:Entity {id: $tgt}) "
                    "WHERE r.relationship_type = $rel_type AND r.is_active = true "
                    "RETURN r.weight",
                    parameters={"src": rel.source_id, "tgt": rel.target_id, "rel_type": rel.relationship_type}
                )

                if result.has_next():
                    # Strengthen existing active relationship
                    self.conn.execute(
                        "MATCH (s:Entity {id: $src})-[r:RELATES_TO]->(t:Entity {id: $tgt}) "
                        "WHERE r.relationship_type = $rel_type AND r.is_active = true "
                        "SET r.weight = r.weight + $boost, r.updated_at = $now",
                        parameters={
                            "src": rel.source_id, "tgt": rel.target_id,
                            "rel_type": rel.relationship_type, "boost": weight_boost, "now": now
                        }
                    )
                else:
                    # Create new relationship with temporal fields
                    valid_from = rel.valid_from if rel.valid_from else now
                    self.conn.execute(
                        "MATCH (s:Entity {id: $src}), (t:Entity {id: $tgt}) "
                        "CREATE (s)-[:RELATES_TO {"
                        "    relationship_type: $rel_type, weight: $weight,"
                        "    evidence: $evidence, created_at: $now,"
                        "    valid_from: $valid_from, ingested_at: $now,"
                        "    updated_at: $now, is_active: true"
                        "}]->(t)",
                        parameters={
                            "src": rel.source_id, "tgt": rel.target_id,
                            "rel_type": rel.relationship_type, "weight": rel.weight,
                            "evidence": rel.evidence, "now": now, "valid_from": valid_from,
                        }
                    )
                    self.total_relationships_added += 1
                    logger.info(f"[KG] Added relationship: {rel.source_id} --[{rel.relationship_type}]--> {rel.target_id}")

            return True

        except Exception as e:
            logger.error(f"[KG] Error adding relationship: {e}")
            return False

    def query_entities(
        self,
        query: str,
        entity_type: Optional[EntityType] = None,
        limit: int = 10
    ) -> List[Dict]:
        """
        Query entities by name/description text match.
        For semantic search, use query_entities_semantic().
        """
        self.total_queries += 1

        query_lower = query.lower() if query else ""

        # Build WHERE clause with parameterized values
        conditions = []
        params: Dict[str, Any] = {"limit": limit}

        if query_lower:
            conditions.append(
                "(toLower(e.name) CONTAINS $query"
                " OR toLower(e.description) CONTAINS $query)"
            )
            params["query"] = query_lower
        if entity_type:
            conditions.append("e.entity_type = $entity_type")
            params["entity_type"] = entity_type.value

        where_clause = ""
        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)

        try:
            with self._lock:
                result = self.conn.execute(
                    f"MATCH (e:Entity) "
                    f"{where_clause} "
                    f"RETURN e.id, e.name, e.entity_type, e.description, "
                    f"       e.importance, e.access_count "
                    f"ORDER BY e.importance DESC "
                    f"LIMIT $limit",
                    parameters=params
                )

                entities = []
                while result.has_next():
                    row = result.get_next()
                    entities.append({
                        "id": row[0],
                        "name": row[1],
                        "entity_type": row[2],
                        "description": row[3],
                        "importance": row[4],
                        "access_count": row[5]
                    })
            return entities

        except Exception as e:
            logger.error(f"[KG] Query error: {e}")
            return []

    def get_entity_by_id(self, entity_id: str) -> Optional[Dict]:
        """Get a specific entity by ID."""
        try:
            with self._lock:
                result = self.conn.execute(
                    "MATCH (e:Entity {id: $id}) "
                    "RETURN e.id, e.name, e.entity_type, e.description, "
                    "       e.importance, e.access_count, e.properties",
                    parameters={"id": entity_id}
                )

                if result.has_next():
                    row = result.get_next()
                    return {
                        "id": row[0],
                        "name": row[1],
                        "entity_type": row[2],
                        "description": row[3],
                        "importance": row[4],
                        "access_count": row[5],
                        "properties": row[6]
                    }
            return None

        except Exception as e:
            logger.error(f"[KG] Get entity error: {e}")
            return None

    def get_entity_by_name(self, name: str, entity_type: Optional[EntityType] = None) -> Optional[Dict]:
        """Get entity by exact name match."""
        params: Dict[str, Any] = {"name": name}
        type_filter = ""
        if entity_type:
            type_filter = "AND e.entity_type = $entity_type"
            params["entity_type"] = entity_type.value

        try:
            with self._lock:
                result = self.conn.execute(
                    f"MATCH (e:Entity) "
                    f"WHERE toLower(e.name) = toLower($name) "
                    f"{type_filter} "
                    f"RETURN e.id, e.name, e.entity_type, e.description, "
                    f"       e.importance, e.access_count "
                    f"LIMIT 1",
                    parameters=params
                )

                if result.has_next():
                    row = result.get_next()
                    return {
                        "id": row[0],
                        "name": row[1],
                        "entity_type": row[2],
                        "description": row[3],
                        "importance": row[4],
                        "access_count": row[5]
                    }
            return None

        except Exception as e:
            logger.error(f"[KG] Get entity by name error: {e}")
            return None

    def get_related_entities(
        self,
        entity_id: str,
        hops: int = 2,
        limit: int = 20
    ) -> List[Dict]:
        """
        Get entities within N hops of a given entity.
        Returns entities with their relationship path.
        Only follows active edges.

        Note: `hops` is used in the variable-length path pattern (e.g. *1..2)
        and cannot be a query parameter in Kuzu, so it is interpolated directly.
        The value is always an int, so this is safe.
        """
        try:
            with self._lock:
                result = self.conn.execute(
                    f"MATCH (start:Entity {{id: $id}})-[r:RELATES_TO*1..{int(hops)}]-(related:Entity) "
                    f"WHERE ALL(rel IN r WHERE rel.is_active = true) "
                    f"AND related.id <> $id "
                    f"RETURN DISTINCT related.id, related.name, related.entity_type, "
                    f"       related.description, related.importance "
                    f"ORDER BY related.importance DESC "
                    f"LIMIT $limit",
                    parameters={"id": entity_id, "limit": limit}
                )

                entities = []
                while result.has_next():
                    row = result.get_next()
                    entities.append({
                        "id": row[0],
                        "name": row[1],
                        "entity_type": row[2],
                        "description": row[3],
                        "importance": row[4],
                        "relationship_path": []  # Simplified - Kùzu path handling differs
                    })
            return entities

        except Exception as e:
            logger.error(f"[KG] Related entities error: {e}")
            return []

    def get_relationships(
        self,
        entity_id: str,
        direction: str = "both",  # "outgoing", "incoming", "both"
        include_inactive: bool = False
    ) -> List[Dict]:
        """Get all relationships for an entity.

        Args:
            entity_id: Entity to query.
            direction: "outgoing", "incoming", or "both".
            include_inactive: If False (default), only return active edges.
        """
        active_filter = "" if include_inactive else "AND r.is_active = true "
        params = {"id": entity_id}
        try:
            if direction == "outgoing":
                query = (
                    "MATCH (e:Entity {id: $id})-[r:RELATES_TO]->(t:Entity) "
                    f"WHERE true {active_filter}"
                    "RETURN e.name, r.relationship_type, t.name, t.id, r.weight"
                )
            elif direction == "incoming":
                query = (
                    "MATCH (s:Entity)-[r:RELATES_TO]->(e:Entity {id: $id}) "
                    f"WHERE true {active_filter}"
                    "RETURN s.name, r.relationship_type, e.name, s.id, r.weight"
                )
            else:
                query = (
                    "MATCH (e:Entity {id: $id})-[r:RELATES_TO]-(other:Entity) "
                    f"WHERE true {active_filter}"
                    "RETURN e.name, r.relationship_type, other.name, other.id, r.weight"
                )

            with self._lock:
                result = self.conn.execute(query, parameters=params)

                relationships = []
                while result.has_next():
                    row = result.get_next()
                    relationships.append({
                        "source": row[0],
                        "relationship": row[1],
                        "target": row[2],
                        "target_id": row[3],
                        "weight": row[4]
                    })
            return relationships

        except Exception as e:
            logger.error(f"[KG] Get relationships error: {e}")
            return []

    def decay_importance(self, decay_rate: float = 0.01):
        """
        Apply forgetting curve to all entities.
        Call this during memory consolidation.
        """
        try:
            decay_factor = 1 - decay_rate
            with self._lock:
                self.conn.execute(
                    "MATCH (e:Entity) "
                    "WHERE e.importance > 0.01 "
                    "SET e.importance = e.importance * $factor",
                    parameters={"factor": decay_factor}
                )
            logger.info(f"[KG] Applied importance decay: {decay_rate}")
        except Exception as e:
            logger.error(f"[KG] Decay error: {e}")

    def boost_importance(self, entity_id: str, boost: float = 0.1):
        """Boost importance of a specific entity (e.g., when accessed). Capped at 1.0."""
        try:
            now = int(time.time())
            with self._lock:
                self.conn.execute(
                    "MATCH (e:Entity {id: $id}) "
                    "SET e.importance = CASE WHEN e.importance + $boost > 1.0 THEN 1.0 ELSE e.importance + $boost END, "
                    "    e.access_count = e.access_count + 1, "
                    "    e.last_accessed = $now",
                    parameters={"id": entity_id, "boost": boost, "now": now}
                )
        except Exception as e:
            logger.error(f"[KG] Boost importance error: {e}")

    def prune_low_importance(self, threshold: float = 0.05):
        """Invalidate edges of low-importance entities and delete them."""
        now = int(time.time())
        params = {"threshold": threshold, "now": now}
        try:
            with self._lock:
                # Invalidate relationships to/from low importance entities
                self.conn.execute(
                    "MATCH (e:Entity)-[r:RELATES_TO]-() "
                    "WHERE e.importance < $threshold AND r.is_active = true "
                    "SET r.is_active = false, r.valid_to = $now, r.updated_at = $now",
                    parameters=params
                )

                # Count affected entities
                result = self.conn.execute(
                    "MATCH (e:Entity) WHERE e.importance < $threshold RETURN COUNT(e)",
                    parameters={"threshold": threshold}
                )

                count = 0
                if result.has_next():
                    count = result.get_next()[0]

                # DETACH DELETE removes the node AND all its remaining edges
                self.conn.execute(
                    "MATCH (e:Entity) WHERE e.importance < $threshold DETACH DELETE e",
                    parameters={"threshold": threshold}
                )

            if count > 0:
                logger.info(f"[KG] Pruned {count} low-importance entities")

        except Exception as e:
            logger.error(f"[KG] Prune error: {e}")

    def get_statistics(self) -> Dict:
        """Get knowledge graph statistics."""
        try:
            with self._lock:
                entity_result = self.conn.execute(
                    "MATCH (e:Entity) RETURN COUNT(e)"
                )
                entity_count = entity_result.get_next()[0] if entity_result.has_next() else 0

                rel_result = self.conn.execute(
                    "MATCH ()-[r:RELATES_TO]->() RETURN COUNT(r)"
                )
                rel_count = rel_result.get_next()[0] if rel_result.has_next() else 0

                # Active vs inactive relationship counts
                active_rel_result = self.conn.execute(
                    "MATCH ()-[r:RELATES_TO]->() WHERE r.is_active = true RETURN COUNT(r)"
                )
                active_rel_count = active_rel_result.get_next()[0] if active_rel_result.has_next() else rel_count

                # Entity type distribution
                type_result = self.conn.execute("""
                    MATCH (e:Entity)
                    RETURN e.entity_type, COUNT(e) as count
                    ORDER BY count DESC
                """)

                type_distribution = {}
                while type_result.has_next():
                    row = type_result.get_next()
                    type_distribution[row[0]] = row[1]

                # Average importance
                avg_result = self.conn.execute(
                    "MATCH (e:Entity) RETURN AVG(e.importance)"
                )
                avg_importance = avg_result.get_next()[0] if avg_result.has_next() else 0

            return {
                "total_entities": entity_count,
                "total_relationships": rel_count,
                "active_relationships": active_rel_count,
                "inactive_relationships": rel_count - active_rel_count,
                "entity_type_distribution": type_distribution,
                "average_importance": avg_importance,
                "total_entities_added": self.total_entities_added,
                "total_relationships_added": self.total_relationships_added,
                "total_queries": self.total_queries
            }

        except Exception as e:
            logger.error(f"[KG] Stats error: {e}")
            return {
                "total_entities": 0,
                "total_relationships": 0,
                "error": str(e)
            }

    def get_all_entity_names(self, limit: int = 1000) -> List[str]:
        """Get all entity names for deduplication."""
        try:
            with self._lock:
                result = self.conn.execute(
                    "MATCH (e:Entity) RETURN e.name ORDER BY e.importance DESC LIMIT $limit",
                    parameters={"limit": limit}
                )

                names = []
                while result.has_next():
                    names.append(result.get_next()[0])
            return names

        except Exception as e:
            logger.error(f"[KG] Get all names error: {e}")
            return []

    def execute_cypher(self, query: str) -> List[Any]:
        """Execute arbitrary Cypher query. Use with caution."""
        try:
            with self._lock:
                result = self.conn.execute(query)

                rows = []
                while result.has_next():
                    rows.append(result.get_next())
            return rows

        except Exception as e:
            logger.error(f"[KG] Cypher error: {e}")
            return []

    # ------------------------------------------------------------------
    # Temporal edge methods
    # ------------------------------------------------------------------

    def invalidate_relationship(
        self, source_id: str, target_id: str, rel_type: str
    ) -> bool:
        """Soft-invalidate an active relationship (sets is_active=false, valid_to=now)."""
        now = int(time.time())
        try:
            with self._lock:
                self.conn.execute(
                    "MATCH (s:Entity {id: $src})-[r:RELATES_TO]->(t:Entity {id: $tgt}) "
                    "WHERE r.relationship_type = $rel_type AND r.is_active = true "
                    "SET r.is_active = false, r.valid_to = $now, r.updated_at = $now",
                    parameters={"src": source_id, "tgt": target_id, "rel_type": rel_type, "now": now}
                )
            logger.info(f"[KG] Invalidated: {source_id} --[{rel_type}]--> {target_id}")
            return True
        except Exception as e:
            logger.error(f"[KG] Invalidate relationship error: {e}")
            return False

    def supersede_relationship(
        self,
        source_id: str,
        target_id: str,
        rel_type: str,
        new_evidence: str = "",
        new_weight: float = 1.0,
    ) -> bool:
        """Invalidate old relationship and create a new version atomically."""
        if not self.invalidate_relationship(source_id, target_id, rel_type):
            return False
        return self.add_relationship(Relationship(
            source_id=source_id,
            target_id=target_id,
            relationship_type=rel_type,
            weight=new_weight,
            evidence=new_evidence,
        ))

    def get_relationships_at_time(
        self, entity_id: str, at_time: int, direction: str = "both"
    ) -> List[Dict]:
        """Point-in-time query: return relationships that were active at ``at_time``."""
        time_filter = (
            "AND r.valid_from <= $at_time "
            "AND (r.valid_to IS NULL OR r.valid_to = 0 OR r.valid_to > $at_time)"
        )
        params = {"id": entity_id, "at_time": at_time}
        try:
            if direction == "outgoing":
                query = (
                    "MATCH (e:Entity {id: $id})-[r:RELATES_TO]->(t:Entity) "
                    f"WHERE true {time_filter} "
                    "RETURN e.name, r.relationship_type, t.name, t.id, r.weight, "
                    "       r.valid_from, r.valid_to, r.is_active"
                )
            elif direction == "incoming":
                query = (
                    "MATCH (s:Entity)-[r:RELATES_TO]->(e:Entity {id: $id}) "
                    f"WHERE true {time_filter} "
                    "RETURN s.name, r.relationship_type, e.name, s.id, r.weight, "
                    "       r.valid_from, r.valid_to, r.is_active"
                )
            else:
                query = (
                    "MATCH (e:Entity {id: $id})-[r:RELATES_TO]-(other:Entity) "
                    f"WHERE true {time_filter} "
                    "RETURN e.name, r.relationship_type, other.name, other.id, r.weight, "
                    "       r.valid_from, r.valid_to, r.is_active"
                )

            with self._lock:
                result = self.conn.execute(query, parameters=params)
                rows = []
                while result.has_next():
                    row = result.get_next()
                    rows.append({
                        "source": row[0],
                        "relationship": row[1],
                        "target": row[2],
                        "target_id": row[3],
                        "weight": row[4],
                        "valid_from": row[5],
                        "valid_to": row[6],
                        "is_active": row[7],
                    })
            return rows
        except Exception as e:
            logger.error(f"[KG] Time-travel query error: {e}")
            return []

    def get_relationship_history(
        self, source_id: str, target_id: str
    ) -> List[Dict]:
        """Return all versions of a relationship sorted by valid_from."""
        try:
            with self._lock:
                result = self.conn.execute(
                    "MATCH (s:Entity {id: $src})-[r:RELATES_TO]->(t:Entity {id: $tgt}) "
                    "RETURN r.relationship_type, r.weight, r.evidence, "
                    "       r.valid_from, r.valid_to, r.is_active, r.created_at "
                    "ORDER BY r.valid_from",
                    parameters={"src": source_id, "tgt": target_id}
                )
                rows = []
                while result.has_next():
                    row = result.get_next()
                    rows.append({
                        "relationship_type": row[0],
                        "weight": row[1],
                        "evidence": row[2],
                        "valid_from": row[3],
                        "valid_to": row[4],
                        "is_active": row[5],
                        "created_at": row[6],
                    })
            return rows
        except Exception as e:
            logger.error(f"[KG] Relationship history error: {e}")
            return []

    def get_active_relationships(self, entity_id: str) -> List[Dict]:
        """Convenience: get only active relationships for an entity."""
        return self.get_relationships(entity_id, include_inactive=False)

    def close(self):
        """Close database connection."""
        # Kùzu handles cleanup automatically
        logger.info("[KG] Knowledge graph closed")
