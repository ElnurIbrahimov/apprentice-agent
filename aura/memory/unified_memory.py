"""Unified Memory Interface — Phase 2 (Memory Consolidation).

Single query routes through the consolidated retrieval pipeline:
  SQLite + FTS5 store  →  BM25 + Semantic + Graph  →  RRF  →  Reranker  →  FadeMem

Same public API as Phase 4C for backward compatibility:
  query(), store(), store_gated(), get_available_sources(), get_stats(), close()

Author: Aura Development Team
Created: 2026-02-07 | Rewritten: 2026-03-16
"""

import atexit
import hashlib
import json
import logging
import math
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class UnifiedResult:
    """A single result from the unified memory query."""
    content: str
    source: str  # "sqlite", "kg_brain", "hybrid" (kept for backward compat)
    score: float  # Unified blended score (0-1)
    source_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Score breakdown (backward compat)
    relevance: float = 0.0
    recency: float = 0.0
    importance: float = 0.0
    emotional_congruence: float = 0.5

    # Dedup key
    content_hash: str = ""

    def __post_init__(self):
        if not self.content_hash:
            self.content_hash = hashlib.md5(
                self.content[:200].lower().strip().encode()
            ).hexdigest()[:12]


class UnifiedMemory:
    """Unified memory query interface backed by consolidated SQLite store.

    Queries the single SQLite store via the multi-channel retrieval pipeline
    (BM25 + semantic + optional Kuzu graph), with RRF fusion, cross-encoder
    reranking, and FadeMem strength weighting.
    """

    def __init__(self):
        self._store = None
        self._kg_brain = None
        self._init_lock = threading.Lock()
        self._store_initialized = False

    def set_kg_brain(self, bridge) -> None:
        """Register the Kuzu KG Brain bridge (called by agent after initialization)."""
        with self._init_lock:
            self._kg_brain = bridge
        logger.debug("[UnifiedMemory] Kuzu KG Brain registered")

    def _ensure_store(self):
        """Lazy-init the MemoryStore singleton."""
        if self._store_initialized:
            return
        with self._init_lock:
            if self._store_initialized:
                return
            try:
                from .store import get_memory_store
                self._store = get_memory_store()
                logger.debug("[UnifiedMemory] SQLite store initialized")
            except Exception as e:
                logger.error("[UnifiedMemory] Failed to init store: %s", e)
            self._store_initialized = True

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(
        self,
        query: str,
        k: int = 10,
        sources: Optional[List[str]] = None,
        min_score: float = 0.0,
        emotional_pad: Optional[Dict[str, float]] = None,
    ) -> List[UnifiedResult]:
        """Query memory and return ranked results.

        Args:
            query: Search query text
            k: Maximum results to return
            sources: Ignored in consolidated mode (kept for backward compat)
            min_score: Minimum score threshold
            emotional_pad: Current PAD state (used for emotional congruence)

        Returns:
            Ranked list of UnifiedResult
        """
        self._ensure_store()
        if not self._store:
            return []

        results: List[UnifiedResult] = []

        # Primary: consolidated retrieval pipeline
        try:
            from .retrieval import retrieve
            retrieval_results = retrieve(
                query=query,
                store=self._store,
                k=k,
                k_candidates=k * 3,
                use_reranker=True,
                use_graph=(self._kg_brain is not None),
            )
            for rr in retrieval_results:
                rec = rr.record
                # Emotional congruence from PAD
                emo_score = 0.5
                if emotional_pad and rec.emotional_pad:
                    emo_score = self._emotional_congruence(
                        self._parse_pad(rec.emotional_pad), emotional_pad
                    )

                # Recency from creation time
                recency = 0.5
                try:
                    created = datetime.fromisoformat(rec.created_at)
                    hours = (datetime.now() - created).total_seconds() / 3600
                    recency = math.exp(-math.log(2) / 336 * hours)  # 2-week half-life
                except (ValueError, TypeError):
                    pass

                results.append(UnifiedResult(
                    content=rec.content,
                    source=rec.source or "sqlite",
                    score=rr.score,
                    source_id=rec.id,
                    metadata={
                        "memory_type": rec.memory_type,
                        "tags": rec.tags,
                        "keywords": rec.keywords,
                        "lifecycle_state": rec.lifecycle_state,
                        "strength": rr.strength,
                        "channels_hit": rr.channels_hit,
                    },
                    relevance=rr.rrf_score,
                    recency=recency,
                    importance=rec.importance,
                    emotional_congruence=emo_score,
                ))
        except Exception as e:
            logger.warning("[UnifiedMemory] Retrieval pipeline error: %s", e, exc_info=True)

        # Deduplicate
        best: dict = {}
        for r in results:
            h = r.content_hash
            if h not in best or r.score > best[h].score:
                best[h] = r
        deduped = list(best.values())

        # Filter and sort
        filtered = [r for r in deduped if r.score >= min_score]
        filtered.sort(key=lambda r: r.score, reverse=True)

        # Track memory recall
        try:
            from api.routes.memory import record_memory_recall
            if filtered:
                record_memory_recall(
                    "unified", len(filtered), query,
                    [r.content[:60] for r in filtered[:5]]
                )
        except Exception:
            pass

        return filtered[:k]

    # ------------------------------------------------------------------
    # Store
    # ------------------------------------------------------------------

    def store(
        self,
        content: str,
        source: str = "conversation",
        importance: float = 0.5,
        tags: Optional[List[str]] = None,
        emotional_pad: Optional[Dict[str, float]] = None,
        episode_type: str = "conversation",
    ) -> Dict[str, str]:
        """Store a memory in the consolidated SQLite store.

        Returns {store: id} for cross-referencing.
        """
        self._ensure_store()
        ids: Dict[str, str] = {}
        if not self._store:
            return ids

        from .store import MemoryRecord

        # Derive emotional valence from PAD
        valence = "neutral"
        if emotional_pad:
            p = emotional_pad.get("pleasure", 0.0)
            if p > 0.2:
                valence = "positive"
            elif p < -0.2:
                valence = "negative"

        record = MemoryRecord(
            content=content,
            title=content[:80],
            source=source,
            memory_type=episode_type,
            importance=importance,
            tags=",".join(tags) if tags else "",
            emotional_valence=valence,
            emotional_pad=json.dumps(emotional_pad) if emotional_pad else "",
            category=episode_type,
        )

        # Get embedding
        embedding = self._get_embedding(content)

        try:
            record_id = self._store.insert(record, embedding=embedding)
            ids["store"] = record_id
        except Exception as e:
            logger.warning("[UnifiedMemory] Store error: %s", e)

        return ids

    def store_gated(
        self,
        content: str,
        source: str = "conversation",
        importance: float = 0.5,
        tags: Optional[List[str]] = None,
        emotional_pad: Optional[Dict[str, float]] = None,
        episode_type: str = "conversation",
        user_id: str = "default_user",
        emotional_salience: float = 0.0,
        explicit_save: bool = False,
        confidence: float = 1.0,
    ) -> Dict[str, Any]:
        """Gated write — runs the MemoryWriteGate before storing.

        Returns dict with store id, decision kind, score, lifecycle state.
        """
        from aura.memory.write_gate import (
            MemoryCandidate, MemoryDecisionKind, get_write_gate
        )

        # Retrieve nearby memories for gate scoring
        try:
            nearby_results = self.query(content, k=5)
            nearby = [
                {"content": r.content, "score": r.relevance,
                 "source_id": r.source_id, "source": r.source}
                for r in nearby_results
            ]
        except Exception:
            nearby = []

        candidate = MemoryCandidate(
            content=content,
            source=source,
            user_id=user_id,
            importance=importance,
            emotional_salience=emotional_salience,
            tags=tags or [],
            explicit_save=explicit_save,
            confidence=confidence,
        )

        gate = get_write_gate()
        decision = gate.evaluate(candidate, nearby=nearby)

        # Telemetry
        try:
            from aura.reliability.telemetry import emit, TelemetryKind
            emit(
                TelemetryKind.MEMORY_DECISION,
                user_id=user_id,
                success=(decision.kind != MemoryDecisionKind.DISCARD),
                memory_writes=(0 if decision.kind == MemoryDecisionKind.DISCARD else 1),
                confidence=decision.score,
                extra=decision.as_log_dict(),
            )
        except Exception:
            pass

        if decision.kind == MemoryDecisionKind.DISCARD:
            return {"decision": decision.kind.value, "score": round(decision.score, 3)}

        # Proceed with actual write
        ids = self.store(
            content=content,
            source=source,
            importance=importance,
            tags=tags,
            emotional_pad=emotional_pad,
            episode_type=episode_type,
        )
        ids["decision"] = decision.kind.value
        ids["score"] = round(decision.score, 3)
        ids["lifecycle"] = decision.lifecycle_state.value

        # Update lifecycle state in store
        if ids.get("store") and decision.lifecycle_state:
            try:
                self._store.update(ids["store"], lifecycle_state=decision.lifecycle_state.value)
            except Exception:
                pass

        return ids

    # ------------------------------------------------------------------
    # Embedding helper
    # ------------------------------------------------------------------

    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding for text via Ollama nomic-embed-text."""
        try:
            import requests
            from aura.config import Config
            url = getattr(Config, 'OLLAMA_HOST', 'http://localhost:11434') + '/api/embeddings'
            r = requests.post(
                url,
                json={"model": "nomic-embed-text:latest", "prompt": text[:1000]},
                timeout=3,
            )
            if r.status_code == 200:
                emb = r.json().get("embedding")
                if emb:
                    return np.array(emb, dtype=np.float32)
        except Exception as e:
            logger.debug("[UnifiedMemory] Embedding failed: %s", e)
        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _parse_pad(self, pad_str: str) -> Optional[Dict[str, float]]:
        """Parse emotional PAD JSON string."""
        if not pad_str:
            return None
        try:
            return json.loads(pad_str)
        except (json.JSONDecodeError, TypeError):
            return None

    def _emotional_congruence(
        self,
        memory_pad: Optional[Dict[str, float]],
        current_pad: Optional[Dict[str, float]],
    ) -> float:
        """Compute emotional congruence between memory's PAD and current PAD."""
        if not memory_pad or not current_pad:
            return 0.5
        m_p = memory_pad.get("pleasure", 0.0)
        m_a = memory_pad.get("arousal", 0.0)
        m_d = memory_pad.get("dominance", 0.0)
        c_p = current_pad.get("pleasure", 0.0)
        c_a = current_pad.get("arousal", 0.0)
        c_d = current_pad.get("dominance", 0.0)
        dist = math.sqrt((m_p - c_p)**2 + (m_a - c_a)**2 + (m_d - c_d)**2)
        max_dist = math.sqrt(12)
        return 1.0 - (dist / max_dist)

    # ------------------------------------------------------------------
    # Info / lifecycle
    # ------------------------------------------------------------------

    def get_available_sources(self) -> List[str]:
        """Get list of available memory backends."""
        self._ensure_store()
        sources = []
        if self._store:
            sources.append("sqlite")
        if self._kg_brain:
            sources.append("kg_brain")
        return sources

    def get_stats(self) -> Dict[str, Any]:
        """Get stats from memory store."""
        self._ensure_store()
        stats: Dict[str, Any] = {"available_sources": self.get_available_sources()}
        if self._store:
            try:
                stats.update(self._store.get_stats())
            except Exception as e:
                logger.debug("[UnifiedMemory] Stats error: %s", e)
        return stats

    def close(self) -> None:
        """Release resources."""
        try:
            if self._store and hasattr(self._store, "close"):
                self._store.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_unified_instance: Optional[UnifiedMemory] = None
_unified_memory_lock = threading.Lock()


def get_unified_memory() -> UnifiedMemory:
    """Get or create the global UnifiedMemory instance."""
    global _unified_instance
    if _unified_instance is None:
        with _unified_memory_lock:
            if _unified_instance is None:
                _unified_instance = UnifiedMemory()
    return _unified_instance


__all__ = ["UnifiedMemory", "UnifiedResult", "get_unified_memory"]
