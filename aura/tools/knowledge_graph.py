"""
Knowledge Graph Memory System for Aura
Relationship-based memory with semantic understanding.

Author: Aura Development Team
Created: 2025-01-26

ARCHITECTURE NOTE — Dual KG design:
  - THIS FILE (aura/tools/knowledge_graph.py): In-process NetworkX graph used as a
    runtime tool by the agent. Lives in RAM, serialized to JSON. Fast for moderate
    graphs (~100K edges). Used directly by ApprenticeAgent.tools["knowledge_graph"].
  - aura_knowledge_graph/graph_database.py: Kuzu-backed persistent graph exposed as
    an MCP server. Intended for large-scale persistent storage and external clients.
    Not the same data store — the two are independent backends.
"""

import json
import logging
import math
import uuid
import threading
import tempfile
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, asdict, field
import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


# Node types in Aura's mind
NODE_TYPES = {
    "concept": "\U0001F4A1",      # Ideas, topics, domains (e.g., "Python", "debugging")
    "entity": "\U0001F4CC",       # Specific things (e.g., "FluxMind", "RTX 4060")
    "person": "\U0001F464",       # People (e.g., "Elnur", "dad")
    "project": "\U0001F4C1",      # Projects (e.g., "MetaFluxMind", "Aura")
    "tool": "\U0001F527",         # Aura's tools (e.g., "web_search", "fluxmind")
    "event": "\U0001F4C5",        # Things that happened (e.g., "debugged CUDA error")
    "emotion": "\U0001F49A",      # Emotional associations
    "skill": "\u26A1",            # Learned capabilities
    "location": "\U0001F4CD",     # Places
    "file": "\U0001F4C4",         # Files user works with
}

# Edge types (relationships)
EDGE_TYPES = {
    "relates_to": "\u2194\uFE0F",       # Generic association
    "is_a": "\u2282",                   # Category membership
    "part_of": "\u2208",                # Composition
    "causes": "\u2192",                 # Causation
    "solves": "\u2713",                 # Solution relationship
    "created_by": "\U0001F464\u2192",   # Authorship
    "uses": "\U0001F527\u2192",         # Usage relationship
    "triggers": "\u26A1\u2192",         # Emotional/behavioral triggers
    "learned_from": "\U0001F4DA\u2192", # Knowledge source
    "preceded_by": "\u23EE\uFE0F",      # Temporal sequence
    "followed_by": "\u23ED\uFE0F",      # Temporal sequence
    "conflicts_with": "\u2694\uFE0F",   # Contradiction/tension
    "strengthens": "\U0001F4AA",        # Reinforcement
    "weakens": "\U0001F4C9",            # Diminishment
    "knows": "\U0001F9E0",              # Knowledge relationship
    "works_on": "\U0001F4BC",           # Work relationship
    "located_at": "\U0001F4CD",         # Location relationship
}


@dataclass
class Node:
    """Represents a node in the knowledge graph.

    Bi-temporal tracking (Phase 4A):
    - valid_from: when this entity became relevant
    - valid_to: when this entity stopped being relevant (None = still valid)
    """
    id: str
    type: str
    label: str
    properties: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    created_at: str = ""
    updated_at: str = ""
    access_count: int = 0
    last_accessed: str = ""
    confidence: float = 0.8
    source: str = "inference"
    # Bi-temporal fields (Phase 4A)
    valid_from: str = ""
    valid_to: Optional[str] = None  # None = still valid

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if not self.updated_at:
            self.updated_at = self.created_at
        if not self.last_accessed:
            self.last_accessed = self.created_at
        if not self.valid_from:
            self.valid_from = self.created_at

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        d = asdict(self)
        # Don't serialize large embeddings to JSONL
        if d.get("embedding"):
            d["has_embedding"] = True
            del d["embedding"]
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Node':
        """Create Node from dictionary."""
        # Handle has_embedding flag
        if data.get("has_embedding"):
            del data["has_embedding"]
            data["embedding"] = None
        return cls(**data)

    def format_display(self) -> str:
        """Format for display."""
        # Handle case where type might be corrupted (dict instead of str)
        node_type = self.type if isinstance(self.type, str) else "unknown"
        icon = NODE_TYPES.get(node_type, "\U0001F4AD")
        conf = f" [{int(self.confidence * 100)}%]" if self.confidence < 1.0 else ""
        return f"{icon} {self.label}{conf}"


@dataclass
class Edge:
    """Represents an edge (relationship) in the knowledge graph.

    Bi-temporal tracking (Phase 4A):
    - transaction_time: when this edge was recorded in the KG (immutable)
    - valid_from: when this fact became true in the real world
    - valid_to: when this fact stopped being true (None = still valid)
    - superseded_by: ID of the edge that replaced this one
    """
    id: str
    type: str
    source_id: str
    target_id: str
    weight: float = 0.5
    properties: Dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    last_reinforced: str = ""
    # Bi-temporal fields (Phase 4A)
    transaction_time: str = ""
    valid_from: str = ""
    valid_to: Optional[str] = None  # None = still valid
    superseded_by: Optional[str] = None  # Edge ID that replaced this

    def __post_init__(self):
        now = datetime.now().isoformat()
        if not self.created_at:
            self.created_at = now
        if not self.last_reinforced:
            self.last_reinforced = self.created_at
        if not self.transaction_time:
            self.transaction_time = self.created_at
        if not self.valid_from:
            self.valid_from = self.created_at

    @property
    def is_valid(self) -> bool:
        """Check if this edge is currently valid (not superseded)."""
        return self.valid_to is None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Edge':
        """Create Edge from dictionary."""
        return cls(**data)

    def format_display(self, source_label: str = "", target_label: str = "") -> str:
        """Format for display."""
        icon = EDGE_TYPES.get(self.type, "\u2192")
        weight_str = f" ({int(self.weight * 100)}%)" if self.weight < 1.0 else ""
        return f"{source_label} {icon} {self.type} {icon} {target_label}{weight_str}"


class KnowledgeGraphTool:
    """
    Knowledge Graph Memory for Aura.

    Provides relationship-based memory storage and retrieval
    using a directed graph structure.
    """

    name = "knowledge_graph"
    description = "Query and manage Aura's knowledge graph memory"

    # Quota limits to prevent unbounded memory growth
    MAX_NODES = 50000  # Maximum number of nodes allowed
    MAX_EDGES = 100000  # Maximum number of edges allowed

    def __init__(self, db_path: str = "data/knowledge_graph/"):
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)

        # NetworkX directed graph
        self.graph = nx.DiGraph()

        # Node and edge storage
        self._nodes: Dict[str, Node] = {}
        self._edges: Dict[str, Edge] = {}

        # Label to ID index for fast lookups
        self._label_index: Dict[str, str] = {}

        # Word-based index for fast find_nodes queries (word -> list of node IDs)
        self._word_index: Dict[str, List[str]] = {}

        # Edge index for O(1) duplicate detection: (source_id, target_id, type) -> edge_id
        self._edge_index: Dict[Tuple[str, str, str], str] = {}

        # Thread safety (RLock allows reentrant locking for nested method calls)
        self._lock = threading.RLock()

        # File paths
        self.nodes_file = self.db_path / "nodes.jsonl"
        self.edges_file = self.db_path / "edges.jsonl"
        self.embeddings_file = self.db_path / "embeddings.npy"
        self.stats_file = self.db_path / "stats.json"

        # Load existing graph
        self.load()

    # =========================================================================
    # NODE OPERATIONS
    # =========================================================================

    def add_node(
        self,
        node_type: str,
        label: str,
        properties: Optional[Dict] = None,
        confidence: float = 0.8,
        source: str = "inference",
        embedding: Optional[List[float]] = None
    ) -> Node:
        """
        Add a new node to the knowledge graph.

        Args:
            node_type: Type from NODE_TYPES
            label: Human-readable name
            properties: Additional attributes
            confidence: How certain is this knowledge (0-1)
            source: Where did this come from
            embedding: Optional vector embedding

        Returns:
            Created Node object

        Raises:
            ValueError: If node quota exceeded
        """
        # Validate node_type is a string
        if not isinstance(node_type, str):
            logger.warning(f"Invalid node_type {type(node_type)}, defaulting to 'concept'")
            node_type = "concept"

        with self._lock:
            # Check quota before adding new node
            if len(self._nodes) >= self.MAX_NODES:
                # Try to prune old, low-confidence nodes first
                self._prune_lowest_confidence_nodes(count=100)
                if len(self._nodes) >= self.MAX_NODES:
                    raise ValueError(f"Node quota exceeded ({self.MAX_NODES}). Run consolidate() to free space.")

            # Check if node with same label exists
            existing_id = self._label_index.get(label.lower())
            if existing_id and existing_id in self._nodes:
                # Update existing node
                existing = self._nodes[existing_id]
                existing.access_count += 1
                existing.last_accessed = datetime.now().isoformat()
                existing.updated_at = datetime.now().isoformat()
                if properties:
                    existing.properties.update(properties)
                if confidence > existing.confidence:
                    existing.confidence = confidence
                return existing

            # Create new node
            node_id = f"node_{uuid.uuid4().hex[:12]}"
            node = Node(
                id=node_id,
                type=node_type,
                label=label,
                properties=properties or {},
                embedding=embedding,
                confidence=confidence,
                source=source
            )

            # Add to storage
            self._nodes[node_id] = node
            self._label_index[label.lower()] = node_id
            self._index_node(node_id, label)
            self.graph.add_node(node_id, **node.to_dict())

            # Persist
            self._append_node(node)

            return node

    def get_node(self, node_id: str) -> Optional[Node]:
        """Retrieve node by ID."""
        with self._lock:
            node = self._nodes.get(node_id)
            if node:
                node.access_count += 1
                node.last_accessed = datetime.now().isoformat()
                self.save()
            return node

    def get_node_by_label(self, label: str) -> Optional[Node]:
        """Retrieve node by label."""
        with self._lock:
            node_id = self._label_index.get(label.lower())
            if node_id:
                return self.get_node(node_id)
            return None

    def find_nodes(
        self,
        query: str,
        node_type: Optional[str] = None,
        limit: int = 10
    ) -> List[Node]:
        """
        Find nodes matching a query.

        Uses word index for fast candidate lookup, then scores candidates.
        Falls back to full scan for empty queries or property-based matches.
        """
        with self._lock:
            query_lower = query.lower()
            query_words = query_lower.split()

            # Use word index for fast candidate lookup
            candidate_ids: set = set()
            for qword in query_words:
                for idx_word, node_ids in self._word_index.items():
                    if qword in idx_word or idx_word in qword:
                        candidate_ids.update(node_ids)

            # If no query words or no candidates from index, fall back to full scan
            # (handles empty query and property-only matches)
            if not query_words:
                candidate_ids = set(self._nodes.keys())

            matches = []
            for node_id in candidate_ids:
                node = self._nodes.get(node_id)
                if not node:
                    continue

                # Filter by type if specified
                if node_type and node.type != node_type:
                    continue

                # Score based on label match
                score = 0.0
                label_lower = node.label.lower()

                if query_lower == label_lower:
                    score = 1.0
                elif query_lower in label_lower:
                    score = 0.8
                elif label_lower in query_lower:
                    score = 0.6
                else:
                    # Check properties
                    for key, value in node.properties.items():
                        if query_lower in str(value).lower():
                            score = 0.4
                            break

                if score > 0:
                    matches.append((score * node.confidence, node))

            # Sort by score and return top matches
            matches.sort(key=lambda x: x[0], reverse=True)
            final_nodes = [node for _, node in matches[:limit]]
            try:
                from api.routes.memory import record_memory_recall
                if final_nodes:
                    record_memory_recall("kg", len(final_nodes), query, [n.label for n in final_nodes[:5]])
            except Exception:
                pass
            return final_nodes

    def update_node(self, node_id: str, properties: Dict[str, Any]) -> bool:
        """Update node properties."""
        with self._lock:
            node = self._nodes.get(node_id)
            if not node:
                return False

            node.properties.update(properties)
            node.updated_at = datetime.now().isoformat()

            # Update graph
            self.graph.nodes[node_id].update(node.to_dict())

            self.save()
            return True

    def delete_node(self, node_id: str) -> bool:
        """Remove node and all connected edges."""
        with self._lock:
            node = self._nodes.get(node_id)
            if not node:
                return False

            # Remove from label index and word index
            self._label_index.pop(node.label.lower(), None)
            self._deindex_node(node_id, node.label)

            # Remove connected edges
            edges_to_remove = []
            for edge_id, edge in self._edges.items():
                if edge.source_id == node_id or edge.target_id == node_id:
                    edges_to_remove.append(edge_id)

            for edge_id in edges_to_remove:
                del self._edges[edge_id]

            # Remove from NetworkX
            self.graph.remove_node(node_id)

            # Remove from storage
            del self._nodes[node_id]

            self.save()
            return True

    # =========================================================================
    # EDGE OPERATIONS
    # =========================================================================

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: str,
        weight: float = 0.5,
        properties: Optional[Dict] = None
    ) -> Optional[Edge]:
        """
        Create a relationship between two nodes.

        Args:
            source_id: Source node ID or label
            target_id: Target node ID or label
            edge_type: Type from EDGE_TYPES
            weight: Strength of relationship (0-1)
            properties: Additional context

        Returns:
            Created Edge object or None if nodes don't exist or quota exceeded
        """
        with self._lock:
            # Check edge quota before adding
            if len(self._edges) >= self.MAX_EDGES:
                # Try to prune weak edges first
                self._prune_weak_edges(min_weight=0.2, count=200)
                if len(self._edges) >= self.MAX_EDGES:
                    logger.debug(f"[KG] Edge quota exceeded ({self.MAX_EDGES}). Run consolidate().")
                    return None

            # Resolve labels to IDs if needed
            if not source_id.startswith("node_"):
                source_id = self._label_index.get(source_id.lower(), source_id)
            if not target_id.startswith("node_"):
                target_id = self._label_index.get(target_id.lower(), target_id)

            # Verify nodes exist
            if source_id not in self._nodes or target_id not in self._nodes:
                return None

            # Check if edge already exists (O(1) via index)
            edge_key = (source_id, target_id, edge_type)
            existing_id = self._edge_index.get(edge_key)
            if existing_id:
                existing = self._edges.get(existing_id)
                if existing:
                    # Strengthen existing edge
                    existing.weight = min(1.0, existing.weight + 0.1)
                    existing.last_reinforced = datetime.now().isoformat()
                    return existing

            # Create new edge
            edge_id = f"edge_{uuid.uuid4().hex[:12]}"
            edge = Edge(
                id=edge_id,
                type=edge_type,
                source_id=source_id,
                target_id=target_id,
                weight=weight,
                properties=properties or {}
            )

            # Add to storage and index
            self._edges[edge_id] = edge
            self._edge_index[edge_key] = edge_id
            # Note: edge.to_dict() already contains id, type, weight
            edge_attrs = edge.to_dict()
            self.graph.add_edge(source_id, target_id, **edge_attrs)

            # Persist
            self._append_edge(edge)

            return edge

    def get_edges(
        self,
        node_id: str,
        direction: str = "both",
        edge_type: Optional[str] = None,
        include_invalidated: bool = False,
    ) -> List[Edge]:
        """
        Get all edges connected to a node.

        Args:
            node_id: Node ID or label
            direction: "in", "out", or "both"
            edge_type: Filter by edge type
            include_invalidated: If False (default), exclude edges where valid_to is set
        """
        with self._lock:
            # Resolve label to ID
            if not node_id.startswith("node_"):
                node_id = self._label_index.get(node_id.lower(), node_id)

            edges = []
            for edge in self._edges.values():
                if not include_invalidated and not edge.is_valid:
                    continue
                match = False
                if direction in ("out", "both") and edge.source_id == node_id:
                    match = True
                if direction in ("in", "both") and edge.target_id == node_id:
                    match = True

                if match and (edge_type is None or edge.type == edge_type):
                    edges.append(edge)

            return edges

    def strengthen_edge(self, edge_id: str, amount: float = 0.1) -> bool:
        """Reinforce a relationship (learning)."""
        with self._lock:
            edge = self._edges.get(edge_id)
            if not edge:
                return False

            edge.weight = min(1.0, edge.weight + amount)
            edge.last_reinforced = datetime.now().isoformat()
            self.save()
            return True

    def weaken_edge(self, edge_id: str, amount: float = 0.05) -> bool:
        """Decay unused relationships (forgetting)."""
        with self._lock:
            edge = self._edges.get(edge_id)
            if not edge:
                return False

            edge.weight = max(0.0, edge.weight - amount)
            self.save()
            return True

    def delete_edge(self, edge_id: str) -> bool:
        """Remove an edge."""
        with self._lock:
            edge = self._edges.get(edge_id)
            if not edge:
                return False

            # Remove from NetworkX
            if self.graph.has_edge(edge.source_id, edge.target_id):
                self.graph.remove_edge(edge.source_id, edge.target_id)

            # Remove from edge index
            edge_key = (edge.source_id, edge.target_id, edge.type)
            self._edge_index.pop(edge_key, None)

            del self._edges[edge_id]
            self.save()
            return True

    # =========================================================================
    # BI-TEMPORAL OPERATIONS (Phase 4A)
    # =========================================================================

    def invalidate_edge(self, edge_id: str, reason: str = "") -> bool:
        """
        Mark an edge as no longer valid (Zep-style invalidation).

        Does NOT delete the edge — preserves history.
        Sets valid_to = now so time-travel queries still see it.

        Args:
            edge_id: Edge to invalidate
            reason: Why this fact is no longer true

        Returns:
            True if invalidated successfully
        """
        with self._lock:
            edge = self._edges.get(edge_id)
            if not edge or not edge.is_valid:
                return False

            edge.valid_to = datetime.now().isoformat()
            if reason:
                edge.properties["invalidation_reason"] = reason
            self.save()
            return True

    def supersede_edge(
        self,
        old_edge_id: str,
        new_edge_type: str,
        new_weight: float = 0.5,
        new_properties: Optional[Dict] = None,
        reason: str = ""
    ) -> Optional[Edge]:
        """
        Replace an edge with a new one (supersession tracking).

        Invalidates the old edge and creates a new one between the same nodes.
        The old edge's superseded_by points to the new edge.

        Args:
            old_edge_id: Edge being replaced
            new_edge_type: Type for the replacement edge
            new_weight: Weight for the new edge
            new_properties: Properties for the new edge
            reason: Why the old fact was superseded

        Returns:
            The new Edge, or None if old edge not found
        """
        with self._lock:
            old_edge = self._edges.get(old_edge_id)
            if not old_edge:
                return None

            # Create new edge between same nodes
            new_edge = self.add_edge(
                old_edge.source_id,
                old_edge.target_id,
                new_edge_type,
                weight=new_weight,
                properties=new_properties or {}
            )

            if new_edge:
                # Invalidate old edge and link to new
                old_edge.valid_to = datetime.now().isoformat()
                old_edge.superseded_by = new_edge.id
                if reason:
                    old_edge.properties["supersession_reason"] = reason
                self.save()

            return new_edge

    def query_at_time(
        self,
        query: str,
        at_time: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Time-travel query: what did AURA believe at a given time?

        Returns nodes and their edges that were valid at the specified time.

        Args:
            query: Search query (same as regular query)
            at_time: ISO timestamp to query at
            limit: Max results

        Returns:
            List of dicts with node and its valid edges at that time
        """
        with self._lock:
            try:
                target_time = datetime.fromisoformat(at_time)
            except (ValueError, TypeError):
                return []

            # Find matching nodes that existed at that time
            nodes = self.find_nodes(query, limit=limit * 2)
            results = []

            for node in nodes:
                try:
                    node_created = datetime.fromisoformat(node.created_at)
                except (ValueError, TypeError):
                    continue

                # Node must have existed by target_time
                if node_created > target_time:
                    continue

                # Check if node was still valid at target_time
                if node.valid_to:
                    try:
                        node_expired = datetime.fromisoformat(node.valid_to)
                        if node_expired < target_time:
                            continue
                    except (ValueError, TypeError):
                        pass

                # Get edges that were valid at target_time
                valid_edges = []
                for edge in self._edges.values():
                    if edge.source_id != node.id and edge.target_id != node.id:
                        continue

                    try:
                        edge_from = datetime.fromisoformat(edge.valid_from)
                    except (ValueError, TypeError):
                        continue

                    if edge_from > target_time:
                        continue

                    if edge.valid_to:
                        try:
                            edge_to = datetime.fromisoformat(edge.valid_to)
                            if edge_to < target_time:
                                continue
                        except (ValueError, TypeError):
                            pass

                    valid_edges.append(edge)

                results.append({
                    "node": node,
                    "edges": valid_edges,
                    "edge_count": len(valid_edges),
                })

                if len(results) >= limit:
                    break

            return results

    def get_edge_history(
        self,
        source_label: str,
        target_label: str
    ) -> List[Edge]:
        """
        Get all versions of a relationship between two nodes.

        Returns edges sorted by valid_from (oldest first), including
        invalidated/superseded edges.

        Args:
            source_label: Source node label
            target_label: Target node label

        Returns:
            List of all edges (current and historical) between the nodes
        """
        with self._lock:
            source_id = self._label_index.get(source_label.lower())
            target_id = self._label_index.get(target_label.lower())

            if not source_id or not target_id:
                return []

            history = []
            for edge in self._edges.values():
                if ((edge.source_id == source_id and edge.target_id == target_id) or
                        (edge.source_id == target_id and edge.target_id == source_id)):
                    history.append(edge)

            # Sort by valid_from
            history.sort(key=lambda e: e.valid_from)
            return history

    def get_valid_edges(
        self,
        node_id: str,
        direction: str = "both",
        edge_type: Optional[str] = None
    ) -> List[Edge]:
        """
        Get only currently valid edges for a node (filters out invalidated).

        Same as get_edges() but excludes edges where valid_to is set.
        """
        all_edges = self.get_edges(node_id, direction, edge_type)
        return [e for e in all_edges if e.is_valid]

    # =========================================================================
    # QUERY OPERATIONS
    # =========================================================================

    def query(self, question: str) -> List[Node]:
        """
        Natural language query converted to graph traversal.

        Interprets common question patterns and returns relevant nodes.
        """
        question_lower = question.lower()

        # Pattern: "what do you know about X?"
        if "know about" in question_lower or "tell me about" in question_lower:
            # Extract topic
            for phrase in ["know about", "tell me about"]:
                if phrase in question_lower:
                    topic = question_lower.split(phrase)[-1].strip().rstrip("?")
                    return self.find_nodes(topic, limit=10)

        # Pattern: "how is X related to Y?"
        if "related to" in question_lower or "connected to" in question_lower:
            parts = question_lower.replace("?", "").split()
            # Try to find two entities
            nodes = self.find_nodes(question_lower, limit=5)
            return nodes

        # Pattern: "what tools/projects/etc..."
        for node_type in NODE_TYPES:
            if node_type in question_lower:
                return self.find_nodes("", node_type=node_type, limit=20)

        # Fallback: general search
        results = self.find_nodes(question, limit=10)
        # === PHASE 1: Track memory recall ===
        try:
            from api.routes.memory import record_memory_recall
            if results:
                record_memory_recall("kg", len(results), question, [n.label for n in results[:5]])
        except Exception:
            pass
        try:
            from api.routes.context import track_context_from_memory
            if results:
                track_context_from_memory([n.label for n in results[:5]])
        except Exception:
            pass
        return results

    def get_related(
        self,
        node_id: str,
        depth: int = 2,
        min_weight: float = 0.3
    ) -> Dict[str, Any]:
        """
        Get neighborhood of a node up to N hops.

        Returns dict with 'nodes' and 'edges' keys.
        Only follows currently valid edges (Phase 4A).
        """
        with self._lock:
            # Resolve label to ID
            if not node_id.startswith("node_"):
                node_id = self._label_index.get(node_id.lower(), node_id)

            if node_id not in self._nodes:
                return {"nodes": [], "edges": []}

            # BFS traversal
            visited_nodes = {node_id}
            visited_edges = set()
            frontier = [node_id]

            for _ in range(depth):
                next_frontier = []
                for current_id in frontier:
                    # Get connected edges (only valid ones)
                    for edge in self._edges.values():
                        if edge.weight < min_weight:
                            continue
                        if not edge.is_valid:
                            continue  # Skip invalidated edges (Phase 4A)

                        neighbor_id = None
                        if edge.source_id == current_id:
                            neighbor_id = edge.target_id
                        elif edge.target_id == current_id:
                            neighbor_id = edge.source_id

                        if neighbor_id and neighbor_id not in visited_nodes:
                            visited_nodes.add(neighbor_id)
                            visited_edges.add(edge.id)
                            next_frontier.append(neighbor_id)

                frontier = next_frontier

            # Collect results
            nodes = [self._nodes[nid] for nid in visited_nodes if nid in self._nodes]
            edges = [self._edges[eid] for eid in visited_edges if eid in self._edges]

            return {"nodes": nodes, "edges": edges}

    def find_path(
        self,
        source_id: str,
        target_id: str
    ) -> List[Tuple[Node, Edge, Node]]:
        """
        Find connection path between two concepts.

        Returns list of (node, edge, node) tuples representing the path.
        """
        with self._lock:
            # Resolve labels to IDs
            if not source_id.startswith("node_"):
                source_id = self._label_index.get(source_id.lower(), source_id)
            if not target_id.startswith("node_"):
                target_id = self._label_index.get(target_id.lower(), target_id)

            if source_id not in self._nodes or target_id not in self._nodes:
                return []

            try:
                # Use NetworkX shortest path
                path_ids = nx.shortest_path(
                    self.graph.to_undirected(),
                    source_id,
                    target_id
                )
            except nx.NetworkXNoPath:
                return []

            # Build path with edges
            result = []
            for i in range(len(path_ids) - 1):
                src_id = path_ids[i]
                tgt_id = path_ids[i + 1]

                # Find connecting edge
                edge = None
                for e in self._edges.values():
                    if (e.source_id == src_id and e.target_id == tgt_id) or \
                       (e.source_id == tgt_id and e.target_id == src_id):
                        edge = e
                        break

                if edge:
                    result.append((
                        self._nodes.get(src_id),
                        edge,
                        self._nodes.get(tgt_id)
                    ))

            return result

    def get_clusters(self) -> List[List[Node]]:
        """Identify strongly connected concept groups."""
        with self._lock:
            # Get connected components
            undirected = self.graph.to_undirected()
            components = list(nx.connected_components(undirected))

            clusters = []
            for component in components:
                nodes = [self._nodes[nid] for nid in component if nid in self._nodes]
                if nodes:
                    clusters.append(nodes)

            # Sort by size
            clusters.sort(key=len, reverse=True)
            return clusters

    # =========================================================================
    # LEARNING OPERATIONS
    # =========================================================================

    def learn_from_conversation(
        self,
        user_msg: str,
        aura_response: str,
        entities: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Extract and store knowledge from dialogue.

        Args:
            user_msg: What the user said
            aura_response: Aura's response
            entities: Pre-extracted entities (optional)

        Returns:
            Dict with added nodes and edges
        """
        added_nodes = []
        added_edges = []

        # If entities provided, add them
        if entities:
            for entity in entities:
                node = self.add_node(
                    node_type=entity.get("type", "concept"),
                    label=entity.get("label", ""),
                    properties=entity.get("properties", {}),
                    source="conversation"
                )
                added_nodes.append(node)

        return {
            "nodes_added": len(added_nodes),
            "edges_added": len(added_edges),
            "nodes": added_nodes,
            "edges": added_edges
        }

    def learn_from_tool_use(
        self,
        tool_name: str,
        tool_input: str,
        tool_output: str,
        success: bool
    ) -> Dict[str, Any]:
        """
        Learn from tool execution patterns.

        Creates/strengthens relationships between tools and concepts.
        """
        # Find or create tool node
        tool_node = self.get_node_by_label(tool_name)
        if not tool_node:
            tool_node = self.add_node(
                node_type="tool",
                label=tool_name,
                properties={"type": "aura_tool"},
                confidence=1.0,
                source="system"
            )

        # Extract concepts from input
        input_nodes = self.find_nodes(tool_input, limit=3)

        # Create relationships
        edge_type = "solves" if success else "relates_to"
        for input_node in input_nodes:
            self.add_edge(
                tool_node.id,
                input_node.id,
                edge_type,
                weight=0.6 if success else 0.3,
                properties={"context": tool_input[:100]}
            )

        return {
            "tool_node": tool_node,
            "related_concepts": len(input_nodes),
            "success": success
        }

    def consolidate(self) -> Dict[str, Any]:
        """
        Dream-mode: merge similar nodes, prune weak edges.

        Returns summary of consolidation actions.
        """
        merged = 0
        pruned = 0
        strengthened = 0

        with self._lock:
            # 1. Prune weak, old edges
            edges_to_prune = []
            now = datetime.now()

            for edge_id, edge in self._edges.items():
                # Parse last_reinforced
                try:
                    last_reinforced = datetime.fromisoformat(edge.last_reinforced)
                    age_days = (now - last_reinforced).days
                except (ValueError, TypeError, AttributeError):
                    age_days = 0  # Default to 0 if timestamp is invalid

                # Prune weak edges older than 7 days
                if edge.weight < 0.2 and age_days > 7:
                    edges_to_prune.append(edge_id)

            for edge_id in edges_to_prune:
                self.delete_edge(edge_id)
                pruned += 1

            # 2. Find and merge very similar nodes
            # (Simple approach: exact label match after normalization)
            label_groups: Dict[str, List[str]] = {}
            for node_id, node in self._nodes.items():
                normalized = node.label.lower().strip()
                if normalized not in label_groups:
                    label_groups[normalized] = []
                label_groups[normalized].append(node_id)

            for label, node_ids in label_groups.items():
                if len(node_ids) > 1:
                    # Keep the one with highest confidence
                    nodes = [(self._nodes[nid], nid) for nid in node_ids]
                    nodes.sort(key=lambda x: x[0].confidence, reverse=True)

                    # Merge into first
                    keeper = nodes[0][1]
                    for _, to_remove in nodes[1:]:
                        self._merge_nodes(keeper, to_remove)
                        merged += 1

        return {
            "merged_nodes": merged,
            "pruned_edges": pruned,
            "strengthened_edges": strengthened
        }

    def _prune_lowest_confidence_nodes(self, count: int = 100):
        """Remove lowest confidence nodes to free quota space.

        Args:
            count: Number of nodes to prune
        """
        if not self._nodes:
            return

        # Sort by confidence (ascending) and access_count
        sorted_nodes = sorted(
            self._nodes.values(),
            key=lambda n: (n.confidence, n.access_count)
        )

        # Prune the lowest confidence nodes
        pruned = 0
        for node in sorted_nodes[:count]:
            if node.confidence < 0.5 and node.access_count < 3:
                self.delete_node(node.id)
                pruned += 1

        if pruned > 0:
            logger.debug(f"[KG] Auto-pruned {pruned} low-confidence nodes")

    def _prune_weak_edges(self, min_weight: float = 0.2, count: int = 200):
        """Remove weak edges to free quota space.

        Args:
            min_weight: Edges below this weight are candidates for pruning
            count: Maximum number of edges to prune
        """
        if not self._edges:
            return

        # Find weak edges
        weak_edges = [
            edge for edge in self._edges.values()
            if edge.weight < min_weight
        ]

        # Sort by weight (ascending)
        weak_edges.sort(key=lambda e: e.weight)

        # Prune weakest edges
        pruned = 0
        for edge in weak_edges[:count]:
            self.delete_edge(edge.id)
            pruned += 1

        if pruned > 0:
            logger.debug(f"[KG] Auto-pruned {pruned} weak edges")

    def _merge_nodes(self, keeper_id: str, remove_id: str):
        """Merge remove_id node into keeper_id."""
        keeper = self._nodes.get(keeper_id)
        remove = self._nodes.get(remove_id)

        if not keeper or not remove:
            return

        # Merge properties
        keeper.properties.update(remove.properties)
        keeper.access_count += remove.access_count
        keeper.confidence = max(keeper.confidence, remove.confidence)

        # Redirect edges (both in-memory dict and NetworkX graph)
        for edge in list(self._edges.values()):
            old_source = edge.source_id
            old_target = edge.target_id
            changed = False
            if edge.source_id == remove_id:
                edge.source_id = keeper_id
                changed = True
            if edge.target_id == remove_id:
                edge.target_id = keeper_id
                changed = True
            if changed and self.graph.has_edge(old_source, old_target):
                data = self.graph.get_edge_data(old_source, old_target)
                self.graph.remove_edge(old_source, old_target)
                self.graph.add_edge(edge.source_id, edge.target_id, **(data or {}))

        # Remove the merged node
        self.delete_node(remove_id)

    # Edge half-life in hours (Phase 4B): unreinforced edges decay to 50% weight in 2 weeks
    EDGE_HALF_LIFE_HOURS: float = 336.0  # 14 days
    # Node confidence half-life: unreferenced nodes lose confidence over 4 weeks
    NODE_HALF_LIFE_HOURS: float = 672.0  # 28 days

    def decay(self, hours_passed: float = 24) -> int:
        """
        Apply Ebbinghaus exponential forgetting curve to knowledge (Phase 4B).

        Uses: weight *= e^(-ln(2)/half_life * hours_since_last_access)
        Replaces the old linear 1%/day decay.

        Spaced repetition: accessing a node/edge resets its decay timer
        (already handled by last_reinforced / last_accessed updates).

        Returns number of edges weakened.
        """
        now = datetime.now()
        decay_rate_edge = math.log(2) / self.EDGE_HALF_LIFE_HOURS
        decay_rate_node = math.log(2) / self.NODE_HALF_LIFE_HOURS
        weakened = 0

        with self._lock:
            # Decay edges
            for edge in self._edges.values():
                if not edge.is_valid:
                    continue  # Don't decay already-invalidated edges
                try:
                    last_accessed = datetime.fromisoformat(edge.last_reinforced)
                    hours_since = (now - last_accessed).total_seconds() / 3600

                    if hours_since > hours_passed:
                        # Ebbinghaus exponential decay
                        decay_factor = math.exp(-decay_rate_edge * hours_since)
                        # Apply decay relative to original weight, with floor
                        new_weight = edge.weight * decay_factor
                        if new_weight < edge.weight:
                            edge.weight = max(0.01, new_weight)
                            weakened += 1
                except (ValueError, TypeError, AttributeError):
                    pass

            # Decay node confidence (Phase 4B)
            for node in self._nodes.values():
                if node.valid_to is not None:
                    continue  # Don't decay invalidated nodes
                try:
                    last_accessed = datetime.fromisoformat(node.last_accessed)
                    hours_since = (now - last_accessed).total_seconds() / 3600

                    if hours_since > hours_passed:
                        decay_factor = math.exp(-decay_rate_node * hours_since)
                        new_conf = node.confidence * decay_factor
                        if new_conf < node.confidence:
                            node.confidence = max(0.05, new_conf)
                except (ValueError, TypeError, AttributeError):
                    pass

        return weakened

    def reinforce_node(self, node_id: str) -> bool:
        """
        Spaced repetition: reinforce a node by resetting its decay timer (Phase 4B).

        Call this whenever a node is accessed or referenced.
        """
        with self._lock:
            node = self._nodes.get(node_id)
            if not node:
                return False
            node.last_accessed = datetime.now().isoformat()
            node.access_count += 1
            # Slight confidence boost on access (spaced repetition effect)
            node.confidence = min(1.0, node.confidence + 0.02)
            return True

    # =========================================================================
    # INDEXING HELPERS
    # =========================================================================

    def _index_node(self, node_id: str, label: str) -> None:
        """Add a node to the word index for fast find_nodes lookups."""
        for word in label.lower().split():
            self._word_index.setdefault(word, [])
            if node_id not in self._word_index[word]:
                self._word_index[word].append(node_id)

    def _deindex_node(self, node_id: str, label: str) -> None:
        """Remove a node from the word index."""
        for word in label.lower().split():
            if word in self._word_index:
                try:
                    self._word_index[word].remove(node_id)
                except ValueError:
                    pass

    # =========================================================================
    # PERSISTENCE
    # =========================================================================

    def _atomic_write_json(self, path, data) -> None:
        """Write JSON data atomically using temp file + rename."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.tmp.")
        try:
            with os.fdopen(tmp_fd, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            os.replace(tmp_path, str(path))
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def save(self):
        """Persist entire graph to disk (atomic writes to prevent data loss on crash)."""
        with self._lock:
            # Save nodes (atomic temp+rename)
            fd, tmp_path = tempfile.mkstemp(dir=self.nodes_file.parent, suffix='.tmp')
            try:
                with os.fdopen(fd, 'w', encoding='utf-8') as f:
                    for node in self._nodes.values():
                        f.write(json.dumps(node.to_dict()) + '\n')
                os.replace(tmp_path, str(self.nodes_file))
            except Exception:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise

            # Save edges (atomic temp+rename)
            fd, tmp_path = tempfile.mkstemp(dir=self.edges_file.parent, suffix='.tmp')
            try:
                with os.fdopen(fd, 'w', encoding='utf-8') as f:
                    for edge in self._edges.values():
                        f.write(json.dumps(edge.to_dict()) + '\n')
                os.replace(tmp_path, str(self.edges_file))
            except Exception:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise

            # Save stats (atomic JSON write)
            stats = {
                "node_count": len(self._nodes),
                "edge_count": len(self._edges),
                "last_saved": datetime.now().isoformat(),
                "node_types": {},
                "edge_types": {}
            }

            for node in self._nodes.values():
                stats["node_types"][node.type] = stats["node_types"].get(node.type, 0) + 1
            for edge in self._edges.values():
                stats["edge_types"][edge.type] = stats["edge_types"].get(edge.type, 0) + 1

            self._atomic_write_json(self.stats_file, stats)

    def load(self):
        """Load graph from disk."""
        with self._lock:
            # Load nodes
            if self.nodes_file.exists():
                with open(self.nodes_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                data = json.loads(line)
                                node = Node.from_dict(data)
                                self._nodes[node.id] = node
                                self._label_index[node.label.lower()] = node.id
                                self.graph.add_node(node.id, **node.to_dict())
                            except Exception as e:
                                logger.debug(f"[KG] Error loading node: {e}")

            # Load edges
            if self.edges_file.exists():
                with open(self.edges_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                data = json.loads(line)
                                edge = Edge.from_dict(data)
                                self._edges[edge.id] = edge
                                self.graph.add_edge(
                                    edge.source_id, edge.target_id,
                                    **edge.to_dict()
                                )
                            except Exception as e:
                                logger.debug(f"[KG] Error loading edge: {e}")

            # Rebuild word index for fast find_nodes queries
            self._word_index.clear()
            for node in self._nodes.values():
                self._index_node(node.id, node.label)

            # Rebuild edge index for O(1) duplicate detection
            self._edge_index = {}
            for edge in self._edges.values():
                if edge.is_valid:
                    self._edge_index[(edge.source_id, edge.target_id, edge.type)] = edge.id

            logger.debug(f"[KG] Loaded {len(self._nodes)} nodes, {len(self._edges)} edges")

    def _append_node(self, node: Node):
        """Append a single node to the JSONL file."""
        with open(self.nodes_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(node.to_dict()) + '\n')

    def _append_edge(self, edge: Edge):
        """Append a single edge to the JSONL file."""
        with open(self.edges_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(edge.to_dict()) + '\n')

    # =========================================================================
    # STATISTICS
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get graph statistics."""
        with self._lock:
            node_types = {}
            edge_types = {}

            for node in self._nodes.values():
                # Handle corrupted type (dict instead of str)
                ntype = node.type if isinstance(node.type, str) else "corrupted"
                node_types[ntype] = node_types.get(ntype, 0) + 1

            for edge in self._edges.values():
                etype = edge.type if isinstance(edge.type, str) else "corrupted"
                edge_types[etype] = edge_types.get(etype, 0) + 1

            # Calculate average confidence
            avg_confidence = 0.0
            if self._nodes:
                avg_confidence = sum(n.confidence for n in self._nodes.values()) / len(self._nodes)

            return {
                "total_nodes": len(self._nodes),
                "total_edges": len(self._edges),
                "node_types": node_types,
                "edge_types": edge_types,
                "clusters": len(self.get_clusters()),
                "avg_confidence": round(avg_confidence, 2)
            }

    def get_recent_nodes(self, limit: int = 20) -> List[Node]:
        """Get most recently accessed nodes."""
        with self._lock:
            nodes = list(self._nodes.values())
            nodes.sort(key=lambda n: n.last_accessed, reverse=True)
            return nodes[:limit]

    # =========================================================================
    # TOOL INTERFACE
    # =========================================================================

    def execute(self, action: str, **kwargs) -> Dict[str, Any]:
        """
        Execute knowledge graph actions (tool interface).

        Actions:
            - "query <question>": Natural language query
            - "add <type> <label>": Add a new node
            - "relate <source> <type> <target>": Add relationship
            - "show <label>": Show node and relationships
            - "path <source> to <target>": Find connection path
            - "stats": Show graph statistics
            - "consolidate": Run memory consolidation
        """
        action_lower = action.lower().strip()

        # Query
        if action_lower.startswith("query "):
            question = action[6:].strip()
            nodes = self.query(question)
            return {
                "success": True,
                "count": len(nodes),
                "results": [n.format_display() for n in nodes],
                "nodes": nodes
            }

        # Add node
        if action_lower.startswith("add "):
            parts = action[4:].strip().split(None, 1)
            if len(parts) >= 2:
                node_type, label = parts
                if node_type in NODE_TYPES:
                    node = self.add_node(node_type, label)
                    return {
                        "success": True,
                        "message": f"Added {node.format_display()}",
                        "node": node
                    }
            return {"success": False, "error": "Usage: add <type> <label>"}

        # Add relationship
        if action_lower.startswith("relate "):
            # Parse: relate <source> <type> <target>
            parts = action[7:].strip().split()
            if len(parts) >= 3:
                source = parts[0]
                edge_type = parts[1]
                target = " ".join(parts[2:])

                if edge_type in EDGE_TYPES:
                    edge = self.add_edge(source, target, edge_type)
                    if edge:
                        return {
                            "success": True,
                            "message": f"Created relationship: {source} --{edge_type}--> {target}",
                            "edge": edge
                        }
            return {"success": False, "error": "Usage: relate <source> <type> <target>"}

        # Show node
        if action_lower.startswith("show "):
            label = action[5:].strip()
            node = self.get_node_by_label(label)
            if node:
                related = self.get_related(node.id, depth=1)
                edges = self.get_edges(node.id)
                return {
                    "success": True,
                    "node": node,
                    "properties": node.properties,
                    "related_count": len(related["nodes"]) - 1,
                    "edges": [e.format_display() for e in edges]
                }
            return {"success": False, "error": f"Node '{label}' not found"}

        # Find path
        if " to " in action_lower and action_lower.startswith("path "):
            parts = action[5:].split(" to ")
            if len(parts) == 2:
                source, target = parts[0].strip(), parts[1].strip()
                path = self.find_path(source, target)
                if path:
                    path_str = " -> ".join([
                        f"{n1.label} --{e.type}--> {n2.label}"
                        for n1, e, n2 in path
                    ])
                    return {
                        "success": True,
                        "path_length": len(path),
                        "path": path_str
                    }
                return {"success": False, "error": "No path found"}

        # Stats
        if action_lower == "stats" or action_lower == "status":
            return {"success": True, **self.get_stats()}

        # Consolidate
        if action_lower == "consolidate":
            result = self.consolidate()
            self.save()
            return {"success": True, **result}

        # Default: treat as query
        nodes = self.query(action)
        return {
            "success": True,
            "count": len(nodes),
            "results": [n.format_display() for n in nodes]
        }


# Singleton instance
_kg_instance: Optional[KnowledgeGraphTool] = None


def seed_initial_knowledge(kg: 'KnowledgeGraphTool') -> Dict[str, int]:
    """
    Seed the knowledge graph with initial foundational knowledge.

    Call this once to bootstrap Aura's core knowledge.
    Returns count of nodes and edges created.
    """
    if kg.get_stats()["total_nodes"] > 5:
        # Already seeded
        return {"nodes_created": 0, "edges_created": 0, "status": "already_seeded"}

    nodes_created = 0
    edges_created = 0

    # Core identity
    aura = kg.add_node("entity", "Aura", {
        "description": "AI assistant, personal apprentice",
        "created_by": "Elnur",
        "purpose": "Help, learn, and grow together"
    }, confidence=1.0, source="core")
    nodes_created += 1

    # Creator
    elnur = kg.add_node("person", "Elnur", {
        "role": "creator",
        "relationship": "creator and friend"
    }, confidence=1.0, source="core")
    nodes_created += 1

    # Core relationships
    kg.add_edge(aura.id, elnur.id, "created_by", weight=1.0)
    kg.add_edge(elnur.id, aura.id, "works_on", weight=1.0)
    edges_created += 2

    # Projects
    apprentice = kg.add_node("project", "AURA", {
        "description": "AURA's codebase and home",
        "status": "active development"
    }, confidence=1.0, source="core")
    nodes_created += 1

    kg.add_edge(aura.id, apprentice.id, "part_of", weight=1.0)
    kg.add_edge(elnur.id, apprentice.id, "created_by", weight=1.0)
    edges_created += 2

    # Core tools
    tool_names = [
        ("web_search", "Search the internet for information"),
        ("code_executor", "Run Python code"),
        ("browser", "Browse and interact with websites"),
        ("vision", "Analyze images and screenshots"),
        ("fluxmind", "Generate images with FLUX"),
        ("filesystem", "Read and write files"),
        ("screenshot", "Capture screen content"),
        ("knowledge_graph", "Memory and relationship storage"),
        ("inner_monologue", "Self-reflection and thinking aloud"),
    ]

    for tool_name, description in tool_names:
        tool_node = kg.add_node("tool", tool_name, {
            "description": description,
            "status": "active"
        }, confidence=1.0, source="core")
        nodes_created += 1

        kg.add_edge(aura.id, tool_node.id, "uses", weight=0.8)
        edges_created += 1

    # Core concepts
    concepts = [
        ("Python", "concept", "Primary programming language"),
        ("AI", "concept", "Artificial Intelligence domain"),
        ("Memory", "concept", "Knowledge storage and retrieval"),
        ("Learning", "skill", "Ability to acquire new knowledge"),
        ("Helping", "skill", "Assisting users with tasks"),
    ]

    for label, node_type, description in concepts:
        concept_node = kg.add_node(node_type, label, {
            "description": description
        }, confidence=0.9, source="core")
        nodes_created += 1

        kg.add_edge(aura.id, concept_node.id, "knows", weight=0.7)
        edges_created += 1

    # Save the seeded graph
    kg.save()

    return {
        "nodes_created": nodes_created,
        "edges_created": edges_created,
        "status": "seeded"
    }


_kg_atexit_registered: bool = False


def _kg_atexit_save():
    """Save KG data on interpreter shutdown to prevent data loss."""
    global _kg_instance
    if _kg_instance is not None:
        try:
            _kg_instance.save()
            logger.info("[KG] atexit: data saved successfully")
        except Exception as e:
            logger.warning(f"[KG] atexit: save failed: {e}")


def get_knowledge_graph() -> KnowledgeGraphTool:
    """Get or create the global KnowledgeGraphTool instance."""
    global _kg_instance, _kg_atexit_registered
    if _kg_instance is None:
        _kg_instance = KnowledgeGraphTool()
        if not _kg_atexit_registered:
            import atexit
            atexit.register(_kg_atexit_save)
            _kg_atexit_registered = True
    return _kg_instance


# Export
__all__ = [
    "KnowledgeGraphTool",
    "get_knowledge_graph",
    "seed_initial_knowledge",
    "Node",
    "Edge",
    "NODE_TYPES",
    "EDGE_TYPES"
]
