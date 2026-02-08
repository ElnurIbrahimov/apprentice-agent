"""
Unified Memory Interface — Phase 4C.

Single query fans out to all memory systems (A-MEM, KG, RAG, Episodic)
and returns ranked, deduplicated results with source attribution.

Ranking: relevance × recency × importance × emotional_congruence

Author: Aura Development Team
Created: 2026-02-07
"""

import logging
import math
import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, List, Any

logger = logging.getLogger(__name__)


@dataclass
class UnifiedResult:
    """A single result from the unified memory query."""
    content: str
    source: str  # "amem", "kg", "rag", "episodic", "hybrid"
    score: float  # Unified blended score (0-1)
    source_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Score breakdown
    relevance: float = 0.0
    recency: float = 0.0
    importance: float = 0.0
    emotional_congruence: float = 0.5  # Neutral default

    # Dedup key
    content_hash: str = ""

    def __post_init__(self):
        if not self.content_hash:
            # Hash first 200 chars for dedup
            self.content_hash = hashlib.md5(
                self.content[:200].lower().strip().encode()
            ).hexdigest()[:12]


class UnifiedMemory:
    """
    Unified memory query interface.

    Queries all available memory systems in parallel-like fashion,
    normalizes scores, deduplicates, and returns ranked results.
    """

    # Weights for unified scoring
    W_RELEVANCE = 0.45
    W_RECENCY = 0.25
    W_IMPORTANCE = 0.20
    W_EMOTIONAL = 0.10

    def __init__(self):
        """Initialize with lazy-loaded memory backends."""
        self._amem = None
        self._kg = None
        self._rag = None
        self._episodic = None
        self._hybrid = None
        self._backends_checked = False

    def _init_backends(self):
        """Lazy-load available memory backends."""
        if self._backends_checked:
            return
        self._backends_checked = True

        # A-MEM
        try:
            from apprentice_agent.tools.amem import get_amem
            self._amem = get_amem()
            logger.debug("[UnifiedMemory] A-MEM available")
        except Exception as e:
            logger.debug(f"[UnifiedMemory] A-MEM unavailable: {e}")

        # Knowledge Graph
        try:
            from apprentice_agent.tools.knowledge_graph import get_knowledge_graph
            self._kg = get_knowledge_graph()
            logger.debug("[UnifiedMemory] KG available")
        except Exception as e:
            logger.debug(f"[UnifiedMemory] KG unavailable: {e}")

        # Local RAG
        try:
            from apprentice_agent.tools.local_rag import get_local_rag
            self._rag = get_local_rag()
            logger.debug("[UnifiedMemory] RAG available")
        except Exception as e:
            logger.debug(f"[UnifiedMemory] RAG unavailable: {e}")

        # Episodic Memory
        try:
            from aura_episodic_memory import EpisodicMemoryStore
            self._episodic = EpisodicMemoryStore("./aura_data/episodic_memory")
            logger.debug("[UnifiedMemory] Episodic available")
        except Exception as e:
            logger.debug(f"[UnifiedMemory] Episodic unavailable: {e}")

    def query(
        self,
        query: str,
        k: int = 10,
        sources: Optional[List[str]] = None,
        min_score: float = 0.1,
        emotional_pad: Optional[Dict[str, float]] = None,
    ) -> List[UnifiedResult]:
        """
        Query all memory systems and return unified ranked results.

        Args:
            query: Search query text
            k: Maximum results to return
            sources: Which sources to query (None = all available)
                     Options: "amem", "kg", "rag", "episodic"
            min_score: Minimum unified score threshold
            emotional_pad: Current PAD state for emotional congruence scoring

        Returns:
            Ranked, deduplicated list of UnifiedResult
        """
        self._init_backends()

        all_results: List[UnifiedResult] = []
        query_sources = sources or ["amem", "kg", "rag", "episodic"]

        # Query each backend
        if "amem" in query_sources and self._amem:
            all_results.extend(self._query_amem(query, k, emotional_pad))

        if "kg" in query_sources and self._kg:
            all_results.extend(self._query_kg(query, k))

        if "rag" in query_sources and self._rag:
            all_results.extend(self._query_rag(query, k))

        if "episodic" in query_sources and self._episodic:
            all_results.extend(self._query_episodic(query, k, emotional_pad))

        # Compute unified scores
        for result in all_results:
            result.score = (
                self.W_RELEVANCE * result.relevance
                + self.W_RECENCY * result.recency
                + self.W_IMPORTANCE * result.importance
                + self.W_EMOTIONAL * result.emotional_congruence
            )

        # Deduplicate by content hash
        seen_hashes = set()
        deduped = []
        for r in all_results:
            if r.content_hash not in seen_hashes:
                seen_hashes.add(r.content_hash)
                deduped.append(r)
            else:
                # If duplicate, keep the one with higher score
                for existing in deduped:
                    if existing.content_hash == r.content_hash and r.score > existing.score:
                        deduped.remove(existing)
                        deduped.append(r)
                        break

        # Filter by minimum score
        filtered = [r for r in deduped if r.score >= min_score]

        # Sort by score descending
        filtered.sort(key=lambda r: r.score, reverse=True)

        # Track memory recall
        try:
            from api.routes.memory import record_memory_recall
            if filtered:
                record_memory_recall(
                    "unified",
                    len(filtered),
                    query,
                    [r.content[:60] for r in filtered[:5]]
                )
        except Exception:
            pass

        return filtered[:k]

    def _query_amem(
        self,
        query: str,
        k: int,
        emotional_pad: Optional[Dict[str, float]]
    ) -> List[UnifiedResult]:
        """Query A-MEM and normalize results."""
        results = []
        try:
            raw = self._amem.search(query, k=k)
            for note, sim_score in raw:
                recency = note.get_recency_score()
                emotional = self._emotional_congruence(
                    getattr(note, 'emotional_pad', None), emotional_pad
                )

                results.append(UnifiedResult(
                    content=note.content,
                    source="amem",
                    score=0.0,  # Computed later
                    source_id=note.id,
                    metadata={
                        "category": note.category,
                        "keywords": note.keywords[:5],
                        "tags": note.tags[:5],
                    },
                    relevance=max(0.0, min(1.0, sim_score)),
                    recency=recency,
                    importance=note.importance,
                    emotional_congruence=emotional,
                ))
        except Exception as e:
            logger.debug(f"[UnifiedMemory] A-MEM query error: {e}")
        return results

    def _query_kg(self, query: str, k: int) -> List[UnifiedResult]:
        """Query Knowledge Graph and normalize results."""
        results = []
        try:
            nodes = self._kg.find_nodes(query, limit=k)
            for i, node in enumerate(nodes):
                # KG doesn't return explicit similarity; estimate from position
                position_score = 1.0 - (i / max(1, len(nodes)))
                # Recency from last_accessed
                recency = 0.5
                try:
                    last = datetime.fromisoformat(node.last_accessed)
                    hours = (datetime.now() - last).total_seconds() / 3600
                    recency = math.exp(-math.log(2) / 336 * hours)
                except (ValueError, TypeError):
                    pass

                # Build content string from node
                content = f"{node.label}"
                if node.properties:
                    desc = node.properties.get("description", "")
                    if desc:
                        content += f": {desc}"

                results.append(UnifiedResult(
                    content=content,
                    source="kg",
                    score=0.0,
                    source_id=node.id,
                    metadata={
                        "type": node.type,
                        "confidence": node.confidence,
                        "access_count": node.access_count,
                    },
                    relevance=position_score * node.confidence,
                    recency=recency,
                    importance=node.confidence,
                    emotional_congruence=0.5,  # KG nodes don't have emotional data
                ))
        except Exception as e:
            logger.debug(f"[UnifiedMemory] KG query error: {e}")
        return results

    def _query_rag(self, query: str, k: int) -> List[UnifiedResult]:
        """Query Local RAG and normalize results."""
        results = []
        try:
            raw = self._rag.search(query, top_k=k)
            for sr in raw:
                results.append(UnifiedResult(
                    content=sr.chunk.content if hasattr(sr, 'chunk') else str(sr),
                    source="rag",
                    score=0.0,
                    source_id=sr.chunk.id if hasattr(sr, 'chunk') else "",
                    metadata={
                        "file_source": getattr(sr.chunk, 'source', '') if hasattr(sr, 'chunk') else "",
                    },
                    relevance=max(0.0, min(1.0, sr.score if hasattr(sr, 'score') else 0.5)),
                    recency=0.5,  # RAG doesn't track access time
                    importance=0.5,  # RAG chunks have no importance
                    emotional_congruence=0.5,
                ))
        except Exception as e:
            logger.debug(f"[UnifiedMemory] RAG query error: {e}")
        return results

    def _query_episodic(
        self,
        query: str,
        k: int,
        emotional_pad: Optional[Dict[str, float]]
    ) -> List[UnifiedResult]:
        """Query Episodic Memory and normalize results."""
        results = []
        try:
            from aura_episodic_memory.episode import EpisodeQuery

            eq = EpisodeQuery(
                query_text=query,
                limit=k,
                emotional_pad=emotional_pad,
                emotional_weight=0.15 if emotional_pad else 0.0,
            )
            raw = self._episodic.search(eq)

            for sr in raw:
                ep = sr.episode
                recency = ep.get_recency_score() if hasattr(ep, 'get_recency_score') else 0.5

                results.append(UnifiedResult(
                    content=ep.content,
                    source="episodic",
                    score=0.0,
                    source_id=ep.id,
                    metadata={
                        "title": getattr(ep, 'title', ''),
                        "episode_type": str(getattr(ep, 'episode_type', '')),
                        "emotional_valence": str(getattr(ep, 'emotional_valence', '')),
                    },
                    relevance=max(0.0, min(1.0, sr.score)),
                    recency=recency,
                    importance=getattr(ep, 'importance', 0.5),
                    emotional_congruence=0.5,  # Could be enhanced with PAD matching
                ))
        except Exception as e:
            logger.debug(f"[UnifiedMemory] Episodic query error: {e}")
        return results

    def _emotional_congruence(
        self,
        memory_pad: Optional[Dict[str, float]],
        current_pad: Optional[Dict[str, float]]
    ) -> float:
        """
        Compute emotional congruence between memory's PAD and current PAD.

        Returns 0.0 (opposite) to 1.0 (perfectly congruent).
        """
        if not memory_pad or not current_pad:
            return 0.5  # Neutral if no emotional data

        # Cosine-like similarity in PAD space
        m_p = memory_pad.get("pleasure", 0.0)
        m_a = memory_pad.get("arousal", 0.0)
        m_d = memory_pad.get("dominance", 0.0)
        c_p = current_pad.get("pleasure", 0.0)
        c_a = current_pad.get("arousal", 0.0)
        c_d = current_pad.get("dominance", 0.0)

        # Euclidean distance in PAD space (max distance = sqrt(12) ≈ 3.46)
        dist = math.sqrt((m_p - c_p)**2 + (m_a - c_a)**2 + (m_d - c_d)**2)
        max_dist = math.sqrt(12)  # From (-1,-1,-1) to (1,1,1)

        # Convert to similarity (0-1)
        return 1.0 - (dist / max_dist)

    def get_available_sources(self) -> List[str]:
        """Get list of available memory backends."""
        self._init_backends()
        sources = []
        if self._amem:
            sources.append("amem")
        if self._kg:
            sources.append("kg")
        if self._rag:
            sources.append("rag")
        if self._episodic:
            sources.append("episodic")
        return sources

    def get_stats(self) -> Dict[str, Any]:
        """Get stats from all memory backends."""
        self._init_backends()
        stats = {"available_sources": self.get_available_sources()}

        if self._amem:
            try:
                stats["amem_notes"] = len(self._amem._notes)
            except Exception:
                pass

        if self._kg:
            try:
                kg_stats = self._kg.get_stats()
                stats["kg_nodes"] = kg_stats.get("total_nodes", 0)
                stats["kg_edges"] = kg_stats.get("total_edges", 0)
            except Exception:
                pass

        if self._rag:
            try:
                stats["rag_chunks"] = len(getattr(self._rag, '_chunks', []))
            except Exception:
                pass

        return stats


# Singleton
_unified_instance: Optional[UnifiedMemory] = None


def get_unified_memory() -> UnifiedMemory:
    """Get or create the global UnifiedMemory instance."""
    global _unified_instance
    if _unified_instance is None:
        _unified_instance = UnifiedMemory()
    return _unified_instance


__all__ = ["UnifiedMemory", "UnifiedResult", "get_unified_memory"]
