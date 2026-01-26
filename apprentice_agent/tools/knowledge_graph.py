"""
Knowledge Graph Memory System for Aura
Relationship-based memory with semantic understanding.

Author: Aura Development Team
Created: 2025-01-26
"""

import json
import uuid
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, asdict, field
import networkx as nx
import numpy as np


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
    """Represents a node in the knowledge graph."""
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

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if not self.updated_at:
            self.updated_at = self.created_at
        if not self.last_accessed:
            self.last_accessed = self.created_at

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
        icon = NODE_TYPES.get(self.type, "\U0001F4AD")
        conf = f" [{int(self.confidence * 100)}%]" if self.confidence < 1.0 else ""
        return f"{icon} {self.label}{conf}"


@dataclass
class Edge:
    """Represents an edge (relationship) in the knowledge graph."""
    id: str
    type: str
    source_id: str
    target_id: str
    weight: float = 0.5
    properties: Dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    last_reinforced: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if not self.last_reinforced:
            self.last_reinforced = self.created_at

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
        """
        with self._lock:
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

        Uses fuzzy string matching on labels and properties.
        """
        with self._lock:
            query_lower = query.lower()
            matches = []

            for node in self._nodes.values():
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
            return [node for _, node in matches[:limit]]

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

            return True

    def delete_node(self, node_id: str) -> bool:
        """Remove node and all connected edges."""
        with self._lock:
            node = self._nodes.get(node_id)
            if not node:
                return False

            # Remove from label index
            self._label_index.pop(node.label.lower(), None)

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
            Created Edge object or None if nodes don't exist
        """
        with self._lock:
            # Resolve labels to IDs if needed
            if not source_id.startswith("node_"):
                source_id = self._label_index.get(source_id.lower(), source_id)
            if not target_id.startswith("node_"):
                target_id = self._label_index.get(target_id.lower(), target_id)

            # Verify nodes exist
            if source_id not in self._nodes or target_id not in self._nodes:
                return None

            # Check if edge already exists
            for edge in self._edges.values():
                if (edge.source_id == source_id and
                    edge.target_id == target_id and
                    edge.type == edge_type):
                    # Strengthen existing edge
                    edge.weight = min(1.0, edge.weight + 0.1)
                    edge.last_reinforced = datetime.now().isoformat()
                    return edge

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

            # Add to storage
            self._edges[edge_id] = edge
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
        edge_type: Optional[str] = None
    ) -> List[Edge]:
        """
        Get all edges connected to a node.

        Args:
            node_id: Node ID or label
            direction: "in", "out", or "both"
            edge_type: Filter by edge type
        """
        with self._lock:
            # Resolve label to ID
            if not node_id.startswith("node_"):
                node_id = self._label_index.get(node_id.lower(), node_id)

            edges = []
            for edge in self._edges.values():
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
            return True

    def weaken_edge(self, edge_id: str, amount: float = 0.05) -> bool:
        """Decay unused relationships (forgetting)."""
        with self._lock:
            edge = self._edges.get(edge_id)
            if not edge:
                return False

            edge.weight = max(0.0, edge.weight - amount)
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

            del self._edges[edge_id]
            return True

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
        return self.find_nodes(question, limit=10)

    def get_related(
        self,
        node_id: str,
        depth: int = 2,
        min_weight: float = 0.3
    ) -> Dict[str, Any]:
        """
        Get neighborhood of a node up to N hops.

        Returns dict with 'nodes' and 'edges' keys.
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
                    # Get connected edges
                    for edge in self._edges.values():
                        if edge.weight < min_weight:
                            continue

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
                except:
                    age_days = 0

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

        # Redirect edges
        for edge in list(self._edges.values()):
            if edge.source_id == remove_id:
                edge.source_id = keeper_id
            if edge.target_id == remove_id:
                edge.target_id = keeper_id

        # Remove the merged node
        self.delete_node(remove_id)

    def decay(self, hours_passed: float = 24) -> int:
        """
        Apply forgetting curve to unused knowledge.

        Returns number of edges weakened.
        """
        decay_amount = 0.01 * (hours_passed / 24)  # ~1% per day
        weakened = 0

        now = datetime.now()

        with self._lock:
            for edge in self._edges.values():
                try:
                    last_accessed = datetime.fromisoformat(edge.last_reinforced)
                    hours_since = (now - last_accessed).total_seconds() / 3600

                    if hours_since > hours_passed:
                        edge.weight = max(0.0, edge.weight - decay_amount)
                        weakened += 1
                except:
                    pass

        return weakened

    # =========================================================================
    # PERSISTENCE
    # =========================================================================

    def save(self):
        """Persist entire graph to disk."""
        with self._lock:
            # Save nodes
            with open(self.nodes_file, 'w', encoding='utf-8') as f:
                for node in self._nodes.values():
                    f.write(json.dumps(node.to_dict()) + '\n')

            # Save edges
            with open(self.edges_file, 'w', encoding='utf-8') as f:
                for edge in self._edges.values():
                    f.write(json.dumps(edge.to_dict()) + '\n')

            # Save stats
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

            with open(self.stats_file, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2)

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
                                print(f"[KG] Error loading node: {e}")

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
                                print(f"[KG] Error loading edge: {e}")

            print(f"[KG] Loaded {len(self._nodes)} nodes, {len(self._edges)} edges")

    def _append_node(self, node: Node):
        """Append a single node to the JSONL file."""
        with open(self.nodes_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(node.to_dict()) + '\n')

    def _append_edge(self, edge: Edge):
        """Append a single edge to the JSONL file."""
        with open(self.edges_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(edge.to_dict()) + '\n')

    def export_graphml(self, path: Optional[str] = None) -> str:
        """Export for visualization in external tools."""
        if path is None:
            path = str(self.db_path / "graph.graphml")

        nx.write_graphml(self.graph, path)
        return path

    # =========================================================================
    # STATISTICS
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get graph statistics."""
        with self._lock:
            node_types = {}
            edge_types = {}

            for node in self._nodes.values():
                node_types[node.type] = node_types.get(node.type, 0) + 1

            for edge in self._edges.values():
                edge_types[edge.type] = edge_types.get(edge.type, 0) + 1

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
    apprentice = kg.add_node("project", "Apprentice Agent", {
        "description": "Aura's codebase and home",
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


def get_knowledge_graph() -> KnowledgeGraphTool:
    """Get or create the global KnowledgeGraphTool instance."""
    global _kg_instance
    if _kg_instance is None:
        _kg_instance = KnowledgeGraphTool()
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
