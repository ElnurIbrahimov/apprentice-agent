"""
Kùzu graph database wrapper for AURA Knowledge Graph.

Key features:
- Embedded database (no server)
- Cypher query support
- Automatic schema initialization
- Importance decay (like Titans forgetting curve)
"""

import hashlib
import json
import logging
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
        self.db = kuzu.Database(str(self.db_path))
        self.conn = kuzu.Connection(self.db)

        self._init_schema()

        # Statistics
        self.total_entities_added = 0
        self.total_relationships_added = 0
        self.total_queries = 0

    def _init_schema(self):
        """Initialize graph schema with all entity types and relationships."""
        # Create Entity node table with all properties
        try:
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
            self.conn.execute("""
                CREATE REL TABLE IF NOT EXISTS MENTIONED_IN(
                    FROM Entity TO SourceDocument,
                    context STRING
                )
            """)
        except Exception as e:
            logger.debug(f"MENTIONED_IN table may exist: {e}")

    def _escape_string(self, s: str) -> str:
        """Escape single quotes in strings for Cypher queries."""
        if s is None:
            return ""
        return s.replace("'", "''").replace("\\", "\\\\")

    def add_entity(self, entity: Entity) -> str:
        """
        Add or update an entity in the graph.
        Uses MERGE to update existing entities.
        Returns the entity ID.
        """
        now = int(time.time())
        props_json = json.dumps(entity.properties)

        # Escape strings
        name_escaped = self._escape_string(entity.name)
        desc_escaped = self._escape_string(entity.description)
        props_escaped = self._escape_string(props_json)

        try:
            # Check if entity exists
            result = self.conn.execute(f"""
                MATCH (e:Entity {{id: '{entity.id}'}})
                RETURN e.id
            """)

            if result.has_next():
                # Update existing entity - boost importance
                self.conn.execute(f"""
                    MATCH (e:Entity {{id: '{entity.id}'}})
                    SET e.importance = e.importance + {entity.importance * 0.1},
                        e.access_count = e.access_count + 1,
                        e.last_accessed = {now}
                """)

                # Update description if current is empty and new one provided
                if entity.description:
                    self.conn.execute(f"""
                        MATCH (e:Entity {{id: '{entity.id}'}})
                        WHERE e.description = '' OR e.description IS NULL
                        SET e.description = '{desc_escaped}'
                    """)
            else:
                # Create new entity
                self.conn.execute(f"""
                    CREATE (e:Entity {{
                        id: '{entity.id}',
                        name: '{name_escaped}',
                        entity_type: '{entity.entity_type.value}',
                        description: '{desc_escaped}',
                        importance: {entity.importance},
                        access_count: 1,
                        created_at: {now},
                        last_accessed: {now},
                        properties: '{props_escaped}'
                    }})
                """)
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
        evidence_escaped = self._escape_string(rel.evidence)
        rel_type_escaped = self._escape_string(rel.relationship_type)

        try:
            # Check if relationship exists
            result = self.conn.execute(f"""
                MATCH (s:Entity {{id: '{rel.source_id}'}})-[r:RELATES_TO]->(t:Entity {{id: '{rel.target_id}'}})
                WHERE r.relationship_type = '{rel_type_escaped}'
                RETURN r.weight
            """)

            if result.has_next():
                # Strengthen existing relationship
                self.conn.execute(f"""
                    MATCH (s:Entity {{id: '{rel.source_id}'}})-[r:RELATES_TO]->(t:Entity {{id: '{rel.target_id}'}})
                    WHERE r.relationship_type = '{rel_type_escaped}'
                    SET r.weight = r.weight + {rel.weight * 0.1}
                """)
            else:
                # Create new relationship
                self.conn.execute(f"""
                    MATCH (s:Entity {{id: '{rel.source_id}'}}), (t:Entity {{id: '{rel.target_id}'}})
                    CREATE (s)-[:RELATES_TO {{
                        relationship_type: '{rel_type_escaped}',
                        weight: {rel.weight},
                        evidence: '{evidence_escaped}',
                        created_at: {now}
                    }}]->(t)
                """)
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

        type_filter = ""
        if entity_type:
            type_filter = f"AND e.entity_type = '{entity_type.value}'"

        query_escaped = self._escape_string(query).lower()

        try:
            result = self.conn.execute(f"""
                MATCH (e:Entity)
                WHERE toLower(e.name) CONTAINS '{query_escaped}'
                   OR toLower(e.description) CONTAINS '{query_escaped}'
                {type_filter}
                RETURN e.id, e.name, e.entity_type, e.description,
                       e.importance, e.access_count
                ORDER BY e.importance DESC
                LIMIT {limit}
            """)

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
            result = self.conn.execute(f"""
                MATCH (e:Entity {{id: '{entity_id}'}})
                RETURN e.id, e.name, e.entity_type, e.description,
                       e.importance, e.access_count, e.properties
            """)

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
        name_escaped = self._escape_string(name)
        type_filter = ""
        if entity_type:
            type_filter = f"AND e.entity_type = '{entity_type.value}'"

        try:
            result = self.conn.execute(f"""
                MATCH (e:Entity)
                WHERE toLower(e.name) = toLower('{name_escaped}')
                {type_filter}
                RETURN e.id, e.name, e.entity_type, e.description,
                       e.importance, e.access_count
                LIMIT 1
            """)

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
        """
        try:
            result = self.conn.execute(f"""
                MATCH (start:Entity {{id: '{entity_id}'}})-[r:RELATES_TO*1..{hops}]-(related:Entity)
                WHERE related.id <> '{entity_id}'
                RETURN DISTINCT related.id, related.name, related.entity_type,
                       related.description, related.importance
                ORDER BY related.importance DESC
                LIMIT {limit}
            """)

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
        direction: str = "both"  # "outgoing", "incoming", "both"
    ) -> List[Dict]:
        """Get all relationships for an entity."""
        try:
            if direction == "outgoing":
                query = f"""
                    MATCH (e:Entity {{id: '{entity_id}'}})-[r:RELATES_TO]->(t:Entity)
                    RETURN e.name, r.relationship_type, t.name, t.id, r.weight
                """
            elif direction == "incoming":
                query = f"""
                    MATCH (s:Entity)-[r:RELATES_TO]->(e:Entity {{id: '{entity_id}'}})
                    RETURN s.name, r.relationship_type, e.name, s.id, r.weight
                """
            else:
                query = f"""
                    MATCH (e:Entity {{id: '{entity_id}'}})-[r:RELATES_TO]-(other:Entity)
                    RETURN e.name, r.relationship_type, other.name, other.id, r.weight
                """

            result = self.conn.execute(query)

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
            self.conn.execute(f"""
                MATCH (e:Entity)
                WHERE e.importance > 0.01
                SET e.importance = e.importance * {1 - decay_rate}
            """)
            logger.info(f"[KG] Applied importance decay: {decay_rate}")
        except Exception as e:
            logger.error(f"[KG] Decay error: {e}")

    def boost_importance(self, entity_id: str, boost: float = 0.1):
        """Boost importance of a specific entity (e.g., when accessed)."""
        try:
            now = int(time.time())
            self.conn.execute(f"""
                MATCH (e:Entity {{id: '{entity_id}'}})
                SET e.importance = e.importance + {boost},
                    e.access_count = e.access_count + 1,
                    e.last_accessed = {now}
            """)
        except Exception as e:
            logger.error(f"[KG] Boost importance error: {e}")

    def prune_low_importance(self, threshold: float = 0.05):
        """Remove entities below importance threshold."""
        try:
            # First remove relationships to/from low importance entities
            self.conn.execute(f"""
                MATCH (e:Entity)-[r:RELATES_TO]-()
                WHERE e.importance < {threshold}
                DELETE r
            """)

            # Then remove entities
            result = self.conn.execute(f"""
                MATCH (e:Entity)
                WHERE e.importance < {threshold}
                RETURN COUNT(e)
            """)

            count = 0
            if result.has_next():
                count = result.get_next()[0]

            self.conn.execute(f"""
                MATCH (e:Entity)
                WHERE e.importance < {threshold}
                DELETE e
            """)

            if count > 0:
                logger.info(f"[KG] Pruned {count} low-importance entities")

        except Exception as e:
            logger.error(f"[KG] Prune error: {e}")

    def get_statistics(self) -> Dict:
        """Get knowledge graph statistics."""
        try:
            entity_result = self.conn.execute(
                "MATCH (e:Entity) RETURN COUNT(e)"
            )
            entity_count = entity_result.get_next()[0] if entity_result.has_next() else 0

            rel_result = self.conn.execute(
                "MATCH ()-[r:RELATES_TO]->() RETURN COUNT(r)"
            )
            rel_count = rel_result.get_next()[0] if rel_result.has_next() else 0

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
            result = self.conn.execute(f"""
                MATCH (e:Entity)
                RETURN e.name
                ORDER BY e.importance DESC
                LIMIT {limit}
            """)

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
            result = self.conn.execute(query)

            rows = []
            while result.has_next():
                rows.append(result.get_next())
            return rows

        except Exception as e:
            logger.error(f"[KG] Cypher error: {e}")
            return []

    def close(self):
        """Close database connection."""
        # Kùzu handles cleanup automatically
        logger.info("[KG] Knowledge graph closed")
