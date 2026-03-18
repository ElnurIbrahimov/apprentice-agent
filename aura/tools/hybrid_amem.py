"""
Hybrid Memory System: A-MEM + Knowledge Graph

Combines the Zettelkasten-style A-MEM with the relationship-based
Knowledge Graph for comprehensive memory retrieval.

Features:
- Unified query interface
- Cross-system linking (A-MEM notes ↔ KG nodes)
- Combined retrieval scoring
- Automatic entity extraction from A-MEM to KG
- Memory consolidation across both systems

Author: Aura Development Team
Created: 2026-02-03
"""

import json
import logging
import threading
import tempfile
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class HybridResult:
    """A unified result from hybrid memory search."""
    content: str
    source: str  # "amem" or "kg"
    score: float
    id: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    # A-MEM specific
    keywords: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    context: str = ""
    links: int = 0

    # KG specific
    node_type: str = ""
    relationships: List[str] = field(default_factory=list)

    def format_display(self) -> str:
        """Format for display."""
        source_icon = "📝" if self.source == "amem" else "🔗"
        score_str = f"[{self.score:.0%}]"
        return f"{source_icon} {score_str} {self.content[:60]}..."


class HybridAMEMSystem:
    """
    Hybrid Memory System combining A-MEM and Knowledge Graph.

    Provides unified interface for:
    - Storing memories (routes to appropriate system)
    - Searching across both systems
    - Linking between systems
    - Consolidated learning from interactions
    """

    def __init__(
        self,
        amem=None,
        knowledge_graph=None,
        llm_func: Optional[callable] = None,
        auto_extract_entities: bool = True,
        amem_weight: float = 0.6,
        kg_weight: float = 0.4,
        data_dir: Optional[Path] = None
    ):
        """
        Initialize hybrid memory system.

        Args:
            amem: A-MEM instance (or will be created)
            knowledge_graph: KnowledgeGraphTool instance
            llm_func: LLM function for entity extraction
            auto_extract_entities: Auto-extract entities from A-MEM to KG
            amem_weight: Weight for A-MEM results in combined scoring
            kg_weight: Weight for KG results in combined scoring
            data_dir: Directory for persistent storage
        """
        self._amem = amem
        self._kg = knowledge_graph
        self.llm_func = llm_func
        self.auto_extract = auto_extract_entities
        self.amem_weight = amem_weight
        self.kg_weight = kg_weight

        # Resolve data_dir: prefer explicit arg, then amem's dir, then default
        if data_dir is not None:
            self.data_dir = Path(data_dir)
        elif amem is not None and hasattr(amem, 'data_dir'):
            self.data_dir = Path(amem.data_dir)
        else:
            from ..config import Config
            self.data_dir = Config.CHROMADB_PATH.parent / "hybrid_amem"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Cached alma_engine reference (loaded once on first use)
        self._alma_engine = None
        self._alma_engine_loaded = False

        # Cross-system links
        self._amem_to_kg: Dict[str, List[str]] = {}  # note_id -> [node_ids]
        self._kg_to_amem: Dict[str, List[str]] = {}  # node_id -> [note_ids]
        self._links_file = self.data_dir / "cross_system_links.json"
        self._links_lock = threading.Lock()

        if self._links_file.exists():
            try:
                data = json.loads(self._links_file.read_text(encoding="utf-8"))
                self._amem_to_kg = data.get("amem_to_kg", {})
                self._kg_to_amem = data.get("kg_to_amem", {})
            except Exception:
                pass

    @property
    def amem(self):
        """Lazy-load A-MEM."""
        if self._amem is None:
            from .amem import get_amem
            self._amem = get_amem(llm_func=self.llm_func)
        return self._amem

    @property
    def kg(self):
        """Lazy-load Knowledge Graph."""
        if self._kg is None:
            from .knowledge_graph import get_knowledge_graph
            self._kg = get_knowledge_graph()
        return self._kg

    # =========================================================================
    # UNIFIED STORAGE
    # =========================================================================

    def remember(
        self,
        content: str,
        memory_type: str = "auto",
        tags: Optional[List[str]] = None,
        importance: float = 0.5,
        source: str = "user",
        entities: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Store a memory in the appropriate system(s).

        Args:
            content: The memory content
            memory_type: "fact", "episodic", "procedural", "semantic", or "auto"
            tags: Optional tags
            importance: How important (0-1)
            source: Where this came from
            entities: Pre-extracted entities for KG

        Returns:
            Dict with note_id, node_ids created
        """
        result = {
            "note_id": None,
            "node_ids": [],
            "links_created": 0
        }

        # Neuromodulator: Dopamine modulates memory importance (learning rate)
        # High dopamine = reward state = store with higher importance
        try:
            from .mood_memory import get_current_mood_pad
            mood = get_current_mood_pad()
            if mood:
                dopamine_level = 0.5
                try:
                    if not self._alma_engine_loaded:
                        from aura.emotion.alma_engine import alma_engine as _ae
                        self._alma_engine = _ae
                        self._alma_engine_loaded = True
                    if self._alma_engine is not None:
                        neuro = self._alma_engine.neuromodulators
                        dopamine_level = neuro.dopamine
                except Exception as _alma_err:
                    self._alma_engine_loaded = True  # don't retry on import error
                    logger.debug(f"[HybridAMEM] ALMA dopamine scoring unavailable: {_alma_err}")
                # Scale importance: dopamine=0.5 -> no change, 1.0 -> +20%, 0.0 -> -15%
                dopamine_offset = (dopamine_level - 0.5) * 0.4
                importance = max(0.1, min(1.0, importance + dopamine_offset))
        except Exception as _mood_err:
            logger.debug(f"[HybridAMEM] Mood-modulated importance failed: {_mood_err}")

        # 1. Store in A-MEM (always)
        note = self.amem.add(
            content=content,
            tags=tags or [],
            category=memory_type if memory_type != "auto" else "general",
            importance=importance,
            source=source
        )
        result["note_id"] = note.id

        # 2. Extract and store entities in KG
        if self.auto_extract or entities:
            extracted = entities or self._extract_entities(content, note.keywords)

            for entity in extracted:
                try:
                    node = self.kg.add_node(
                        node_type=entity.get("type", "concept"),
                        label=entity.get("label", ""),
                        properties=entity.get("properties", {}),
                        confidence=entity.get("confidence", 0.7),
                        source=source
                    )
                    result["node_ids"].append(node.id)

                    # Create cross-system link
                    self._link_amem_to_kg(note.id, node.id)
                    result["links_created"] += 1

                except Exception as e:
                    logger.warning(f"Failed to add entity to KG: {e}")

        # 3. Create relationships between extracted entities
        if len(result["node_ids"]) > 1:
            for i, node_id in enumerate(result["node_ids"][:-1]):
                self.kg.add_edge(
                    node_id,
                    result["node_ids"][i + 1],
                    "relates_to",
                    weight=0.5,
                    properties={"source_note": note.id}
                )

        return result

    def _extract_entities(
        self,
        content: str,
        keywords: List[str]
    ) -> List[Dict]:
        """
        Extract entities from content for KG storage.

        Uses LLM if available, otherwise uses keyword heuristics.
        """
        entities = []

        if self.llm_func:
            try:
                prompt = f"""Extract named entities from this text for a knowledge graph.

Text: {content}
Keywords: {', '.join(keywords)}

List entities in this format (one per line):
TYPE: label | description

Valid types: person, concept, tool, project, location, event, skill, entity

Example:
person: John Smith | software engineer
concept: Python | programming language
tool: web_search | search the internet"""

                response = self.llm_func(prompt)

                for line in response.strip().split('\n'):
                    if ':' in line and '|' in line:
                        try:
                            type_part, rest = line.split(':', 1)
                            label, desc = rest.split('|', 1)
                            entities.append({
                                "type": type_part.strip().lower(),
                                "label": label.strip(),
                                "properties": {"description": desc.strip()},
                                "confidence": 0.7
                            })
                        except (ValueError, AttributeError):
                            continue

            except Exception as e:
                logger.warning(f"LLM entity extraction failed: {e}")

        # Fallback: use keywords as concepts
        if not entities:
            for keyword in keywords[:5]:
                if len(keyword) > 2:
                    entities.append({
                        "type": "concept",
                        "label": keyword,
                        "properties": {},
                        "confidence": 0.5
                    })

        return entities

    # =========================================================================
    # UNIFIED SEARCH
    # =========================================================================

    def recall(
        self,
        query: str,
        k: int = 10,
        include_amem: bool = True,
        include_kg: bool = True,
        follow_links: bool = True,
        min_score: float = 0.1
    ) -> List[HybridResult]:
        """
        Search across both memory systems.

        Combines results using weighted scoring and cross-system links.

        Args:
            query: Search query
            k: Number of results
            include_amem: Search A-MEM
            include_kg: Search Knowledge Graph
            follow_links: Follow cross-system links
            min_score: Minimum score threshold

        Returns:
            List of HybridResult sorted by combined score
        """
        results = []

        # Record real thought: memory recall happening
        try:
            from api.routes.thinking import record_thought
            record_thought("recalling", f"hybrid memory search: {query[:50]}", 0.6, "memory")
        except Exception:
            pass

        # 1. Search A-MEM
        if include_amem:
            amem_results = self.amem.search_agentic(
                query, k=k, follow_links=follow_links
            )

            for r in amem_results:
                score = r.get("relevance", 0) * self.amem_weight

                results.append(HybridResult(
                    content=r.get("content", ""),
                    source="amem",
                    score=score,
                    id=r.get("id", ""),
                    keywords=r.get("keywords", []),
                    tags=r.get("tags", []),
                    context=r.get("context", ""),
                    links=r.get("hop", 0),
                    metadata={"hop": r.get("hop", 0)}
                ))

        # 2. Search Knowledge Graph
        if include_kg:
            kg_nodes = self.kg.find_nodes(query, limit=k)

            for node in kg_nodes:
                # Calculate score based on match and confidence
                score = node.confidence * self.kg_weight

                # Get relationships
                edges = self.kg.get_edges(node.id, direction="both")
                rel_labels = [
                    f"{e.type} → {self.kg.get_node(e.target_id).label if self.kg.get_node(e.target_id) else '?'}"
                    for e in edges[:3]
                ]

                results.append(HybridResult(
                    content=node.label,
                    source="kg",
                    score=score,
                    id=node.id,
                    node_type=node.type,
                    relationships=rel_labels,
                    metadata={
                        "properties": node.properties,
                        "access_count": node.access_count
                    }
                ))

        # 3. Follow cross-system links for boost
        if follow_links:
            self._boost_linked_results(results)

        # 4. Apply mood-congruent memory bias
        self._apply_mood_congruent_bias(results)

        # 5. Sort and deduplicate
        results.sort(key=lambda r: r.score, reverse=True)

        # Remove low-score results
        results = [r for r in results if r.score >= min_score]

        final = results[:k]

        # Record memory recall for UI indicator
        if final:
            try:
                from api.routes.memory import record_memory_recall
                record_memory_recall(
                    "hybrid_amem",
                    len(final),
                    query,
                    [r.content[:80] for r in final[:5]]
                )
            except Exception:
                pass
            try:
                from api.routes.context import track_context_from_memory
                track_context_from_memory([r.content[:80] for r in final[:5]])
            except Exception:
                pass

            try:
                from aura.activity_logger import record_activity
                record_activity(
                    "memory", "recall",
                    f"Recalled {len(final)} memories: {query[:60]}",
                    {"count": len(final), "query": query,
                     "sources": list({r.source for r in final})},
                )
            except Exception:
                pass

        return final

    def _boost_linked_results(self, results: List[HybridResult]):
        """Boost scores for results that have cross-system links."""
        amem_ids = {r.id for r in results if r.source == "amem"}
        kg_ids = {r.id for r in results if r.source == "kg"}

        for result in results:
            if result.source == "amem":
                # Check if linked to any KG nodes in results
                linked_kg = self._amem_to_kg.get(result.id, [])
                if any(kg_id in kg_ids for kg_id in linked_kg):
                    result.score *= 1.2  # 20% boost

            elif result.source == "kg":
                # Check if linked to any A-MEM notes in results
                linked_amem = self._kg_to_amem.get(result.id, [])
                if any(note_id in amem_ids for note_id in linked_amem):
                    result.score *= 1.2

    def _apply_mood_congruent_bias(self, results: List[HybridResult]):
        """
        Apply mood-congruent memory bias: current emotional state
        biases which memories surface more easily.

        Positive mood -> boost positively-valenced memories
        Negative mood -> boost negatively-valenced memories

        Uses stored emotional_pad from A-MEM notes when available,
        falls back to word-heuristic estimation.
        """
        try:
            from .mood_memory import (
                get_current_mood_pad,
                estimate_memory_valence,
                mood_congruent_score_adjustment,
            )

            mood_pad = get_current_mood_pad()
            if mood_pad is None:
                return

            pleasure = mood_pad.get("pleasure", 0.0)
            if abs(pleasure) < 0.1:
                return

            adjusted_count = 0
            for result in results:
                valence = None
                # Prefer stored emotional PAD from A-MEM note
                if result.source == "amem":
                    note = self.amem.get_note(result.id)
                    if note and getattr(note, "emotional_pad", None):
                        stored_pleasure = note.emotional_pad.get("pleasure", 0.0)
                        if abs(stored_pleasure) > 0.01:
                            valence = stored_pleasure

                # Fallback to word-heuristic estimation
                if valence is None:
                    valence = estimate_memory_valence(
                        result.content, result.keywords, result.tags
                    )

                if abs(valence) > 0.05:
                    result.score = mood_congruent_score_adjustment(
                        result.score, valence, pleasure
                    )
                    adjusted_count += 1

            if adjusted_count > 0:
                try:
                    from api.routes.thinking import record_thought
                    mood_label = "positive" if pleasure > 0 else "negative"
                    record_thought(
                        "observing",
                        f"mood-congruent bias ({mood_label} mood) adjusted {adjusted_count} memories",
                        0.4,
                        "emotion",
                    )
                except Exception:
                    pass

        except Exception as e:
            logger.debug(f"Mood-congruent bias skipped: {e}")

    # =========================================================================
    # CROSS-SYSTEM LINKING
    # =========================================================================

    def _link_amem_to_kg(self, note_id: str, node_id: str):
        """Create bidirectional link between A-MEM note and KG node."""
        with self._links_lock:
            if note_id not in self._amem_to_kg:
                self._amem_to_kg[note_id] = []
            if node_id not in self._amem_to_kg[note_id]:
                self._amem_to_kg[note_id].append(node_id)

            if node_id not in self._kg_to_amem:
                self._kg_to_amem[node_id] = []
            if note_id not in self._kg_to_amem[node_id]:
                self._kg_to_amem[node_id].append(note_id)

            try:
                content = json.dumps({"amem_to_kg": self._amem_to_kg, "kg_to_amem": self._kg_to_amem}, indent=2)
                fd, tmp_path = tempfile.mkstemp(dir=self._links_file.parent, suffix='.tmp')
                try:
                    with os.fdopen(fd, 'w', encoding='utf-8') as f:
                        f.write(content)
                    os.replace(tmp_path, str(self._links_file))
                except Exception:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass
                    raise
            except Exception as e:
                logging.getLogger(__name__).warning(f"[HybridAMEM] Failed to save cross-system links: {e}")

    def get_linked_entities(self, note_id: str) -> List[Any]:
        """Get KG nodes linked to an A-MEM note."""
        node_ids = self._amem_to_kg.get(note_id, [])
        return [self.kg.get_node(nid) for nid in node_ids if self.kg.get_node(nid)]

    def get_linked_notes(self, node_id: str) -> List[Any]:
        """Get A-MEM notes linked to a KG node."""
        note_ids = self._kg_to_amem.get(node_id, [])
        return [self.amem.read(nid) for nid in note_ids if self.amem.read(nid)]

    # =========================================================================
    # LEARNING FROM INTERACTIONS
    # =========================================================================

    def learn_from_conversation(
        self,
        user_message: str,
        aura_response: str,
        tool_uses: Optional[List[Dict]] = None,
        outcome: str = "success"
    ) -> Dict[str, Any]:
        """
        Learn from a conversation turn.

        Stores in A-MEM and extracts entities to KG.
        """
        # Skip trivial interactions
        if len(user_message) < 15 and len(aura_response) < 30:
            return {"stored": False, "reason": "too_short"}

        # Create memory content
        content = f"User: {user_message[:200]}"
        if outcome == "success":
            content += f" | Aura: {aura_response[:150]}"

        # Determine tags
        tags = ["conversation", outcome]
        if tool_uses:
            tags.extend([t.get("tool", "") for t in tool_uses[:3]])

        # Store
        result = self.remember(
            content=content,
            memory_type="episodic",
            tags=tags,
            importance=0.6 if outcome == "success" else 0.4,
            source="conversation"
        )

        # Record tool relationships in KG
        if tool_uses:
            for tool_use in tool_uses:
                tool_name = tool_use.get("tool", "")
                if tool_name:
                    self.kg.learn_from_tool_use(
                        tool_name=tool_name,
                        tool_input=tool_use.get("input", "")[:100],
                        tool_output=tool_use.get("output", "")[:100],
                        success=tool_use.get("success", True)
                    )

        return {
            "stored": True,
            **result
        }

    # =========================================================================
    # CONTEXT GENERATION
    # =========================================================================

    def get_context(
        self,
        query: str,
        max_tokens: int = 800,
        format: str = "prompt"
    ) -> str:
        """
        Get relevant memory context for injection into LLM prompt.

        Args:
            query: The query to find context for
            max_tokens: Approximate max length
            format: "prompt" for LLM injection, "readable" for display

        Returns:
            Formatted context string
        """
        results = self.recall(query, k=5)

        if not results:
            return ""

        if format == "prompt":
            lines = ["[Relevant memories:]"]
        else:
            lines = ["📚 **Relevant Memories:**"]

        total_len = 0
        for r in results:
            if r.source == "amem":
                line = f"- {r.content}"
                if r.context:
                    line += f" ({r.context})"
            else:
                line = f"- {r.content} [{r.node_type}]"
                if r.relationships:
                    line += f" ({', '.join(r.relationships[:2])})"

            if total_len + len(line) > max_tokens:
                break

            lines.append(line)
            total_len += len(line)

        return "\n".join(lines)

    # =========================================================================
    # STATISTICS & CONSOLIDATION
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get combined statistics."""
        amem_stats = self.amem.get_stats()
        kg_stats = self.kg.get_stats()

        return {
            "amem": amem_stats,
            "kg": kg_stats,
            "cross_links": {
                "amem_to_kg": len(self._amem_to_kg),
                "kg_to_amem": len(self._kg_to_amem)
            },
            "total_memories": amem_stats.get("total_notes", 0) + kg_stats.get("total_nodes", 0)
        }

    def consolidate(self) -> Dict[str, Any]:
        """
        Consolidate both memory systems.

        - A-MEM: merge similar notes, prune weak links
        - KG: merge nodes, prune edges, decay unused
        """
        amem_result = self.amem.consolidate()
        kg_result = self.kg.consolidate()

        # Save both
        self.amem.save()
        self.kg.save()

        return {
            "amem": amem_result,
            "kg": kg_result
        }


# Singleton instance (thread-safe double-checked locking)
_hybrid_instance: Optional[HybridAMEMSystem] = None
_hybrid_lock = threading.Lock()


def get_hybrid_memory(
    amem=None,
    knowledge_graph=None,
    llm_func=None
) -> HybridAMEMSystem:
    """Get or create the global hybrid memory instance."""
    global _hybrid_instance
    if _hybrid_instance is None:
        with _hybrid_lock:
            if _hybrid_instance is None:
                _hybrid_instance = HybridAMEMSystem(
                    amem=amem,
                    knowledge_graph=knowledge_graph,
                    llm_func=llm_func
                )
    return _hybrid_instance


# Export
__all__ = [
    "HybridAMEMSystem",
    "HybridResult",
    "get_hybrid_memory"
]
