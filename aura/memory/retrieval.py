"""Multi-channel Retrieval Pipeline — Phase 2 Memory Consolidation.

Three retrieval channels fused via Reciprocal Rank Fusion (RRF):
  1. Semantic — cosine similarity on embedding BLOBs (numpy)
  2. BM25 — FTS5 keyword search
  3. Graph — Kuzu KG multi-hop traversal → memory IDs

Post-fusion:
  - Cross-encoder reranking (ms-marco-MiniLM-L-6-v2, 22MB, CPU)
  - FadeMem strength multiplier
  - Touch accessed memories

Author: Aura Development Team
Created: 2026-03-16
"""

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]

from .store import MemoryStore, MemoryRecord, get_memory_store
from .fade_mem import compute_strength, reinforce

logger = logging.getLogger(__name__)

# RRF constant (standard value from Cormack et al.)
RRF_K = 60

# Cross-encoder singleton (lazy-loaded, thread-safe)
_cross_encoder = None
_cross_encoder_attempted = False
_cross_encoder_lock = threading.Lock()


def _get_cross_encoder():
    """Lazy-load the cross-encoder reranker model (thread-safe)."""
    global _cross_encoder, _cross_encoder_attempted
    if _cross_encoder is not None:
        return _cross_encoder
    if _cross_encoder_attempted:
        return None
    with _cross_encoder_lock:
        if _cross_encoder is not None:
            return _cross_encoder
        if _cross_encoder_attempted:
            return None
        _cross_encoder_attempted = True
        try:
            from sentence_transformers import CrossEncoder

            # Load with a 3-second timeout — if model isn't cached locally, skip it
            # instead of blocking startup with HuggingFace download retries.
            result = [None]
            def _load():
                try:
                    result[0] = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)
                except Exception:
                    pass

            t = threading.Thread(target=_load, daemon=True)
            t.start()
            t.join(timeout=3.0)

            if result[0] is not None:
                _cross_encoder = result[0]
                logger.info("[Retrieval] Cross-encoder loaded successfully")
                return _cross_encoder
            else:
                logger.info("[Retrieval] Cross-encoder skipped (not cached locally or timeout)")
                return None
        except Exception as e:
            logger.warning("[Retrieval] Cross-encoder unavailable (CPU reranking disabled): %s", e)
            return None


def _get_query_embedding(text: str) -> Optional[np.ndarray]:
    """Get embedding for query text via shared Ollama helper."""
    from .embedding import get_embedding
    emb = get_embedding(text, timeout=3.0)
    if emb is not None:
        return np.array(emb, dtype=np.float32)
    return None


# Sentinel for "reranking was not attempted"
_RERANK_NOT_SET = float('-inf')


@dataclass
class RetrievalResult:
    """A single result from the retrieval pipeline."""
    record: MemoryRecord
    score: float = 0.0               # Final fused score
    semantic_rank: int = 0
    bm25_rank: int = 0
    graph_rank: int = 0
    rrf_score: float = 0.0
    rerank_score: float = _RERANK_NOT_SET  # Sentinel: not reranked
    strength: float = 1.0            # FadeMem strength at retrieval time
    channels_hit: int = 0            # How many channels returned this result


def reciprocal_rank_fusion(
    ranked_lists: List[List[Tuple[str, float]]],
    k: int = RRF_K,
) -> Dict[str, float]:
    """Compute RRF scores from multiple ranked lists.

    Each ranked_list is [(memory_id, channel_score), ...] sorted by relevance.
    Returns {memory_id: rrf_score}.
    """
    scores: Dict[str, float] = {}
    for ranked in ranked_lists:
        for rank_idx, (mem_id, _channel_score) in enumerate(ranked):
            scores[mem_id] = scores.get(mem_id, 0.0) + 1.0 / (k + rank_idx + 1)
    return scores


def retrieve(
    query: str,
    store: Optional[MemoryStore] = None,
    k: int = 5,
    k_candidates: int = 20,
    user_id: Optional[str] = None,
    use_reranker: bool = True,
    use_graph: bool = True,
    kg_brain=None,
    lifecycle_states: Optional[List[str]] = None,
    touch_results: bool = True,
) -> List[RetrievalResult]:
    """Multi-channel retrieval with RRF fusion and optional reranking.

    Args:
        query: Search query text
        store: MemoryStore instance (default: global singleton)
        k: Number of final results to return
        k_candidates: Number of candidates per channel before fusion
        user_id: Filter by user (None = all users)
        use_reranker: Apply cross-encoder reranking
        use_graph: Include Kuzu graph channel
        kg_brain: KG brain instance (avoids circular import of unified_memory)
        lifecycle_states: Filter by lifecycle states
        touch_results: Whether to touch/reinforce returned memories

    Returns:
        List of RetrievalResult sorted by final score descending
    """
    # Guard: empty query
    if not query or not query.strip():
        return []

    if store is None:
        store = get_memory_store()

    states = lifecycle_states or ["candidate", "stable", "summary"]
    ranked_lists: List[List[Tuple[str, float]]] = []
    all_records: Dict[str, MemoryRecord] = {}
    channel_hits: Dict[str, int] = {}  # mem_id → count of channels

    # ------------------------------------------------------------------
    # Channel 1: Semantic search
    # ------------------------------------------------------------------
    semantic_ranked: List[Tuple[str, float]] = []
    query_emb = _get_query_embedding(query)
    if query_emb is not None:
        try:
            results = store.search_semantic(
                query_emb, k=k_candidates,
                lifecycle_states=states, user_id=user_id,
            )
            for record, sim in results:
                semantic_ranked.append((record.id, sim))
                all_records[record.id] = record
                channel_hits[record.id] = channel_hits.get(record.id, 0) + 1
        except Exception as e:
            logger.debug("[Retrieval] Semantic channel error: %s", e)
    ranked_lists.append(semantic_ranked)

    # ------------------------------------------------------------------
    # Channel 2: BM25 keyword search
    # ------------------------------------------------------------------
    bm25_ranked: List[Tuple[str, float]] = []
    try:
        results = store.search_bm25(
            query, k=k_candidates,
            lifecycle_states=states, user_id=user_id,
        )
        for record, score in results:
            bm25_ranked.append((record.id, score))
            if record.id not in all_records:
                all_records[record.id] = record
            channel_hits[record.id] = channel_hits.get(record.id, 0) + 1
    except Exception as e:
        logger.debug("[Retrieval] BM25 channel error: %s", e)
    ranked_lists.append(bm25_ranked)

    # ------------------------------------------------------------------
    # Channel 3: Kuzu graph traversal (uses kg_brain passed directly)
    # ------------------------------------------------------------------
    # The KG returns entity names/descriptions, not memory IDs directly.
    # We use entity context as an expanded BM25 query, then deduplicate
    # against the BM25 channel to prevent double-counting in RRF fusion.
    graph_ranked: List[Tuple[str, float]] = []
    bm25_ids = {mid for mid, _ in bm25_ranked}
    if use_graph and kg_brain is not None:
        try:
            context = kg_brain.get_context_for_query(query, max_entities=min(k_candidates, 10))
            if context and len(context) > 20:
                graph_results = store.search_bm25(
                    context[:200], k=k_candidates // 2,
                    lifecycle_states=states, user_id=user_id,
                )
                for record, score in graph_results:
                    # Skip results already returned by the BM25 channel
                    if record.id in bm25_ids:
                        continue
                    graph_ranked.append((record.id, score))
                    if record.id not in all_records:
                        all_records[record.id] = record
                    channel_hits[record.id] = channel_hits.get(record.id, 0) + 1
        except Exception as e:
            logger.debug("[Retrieval] Graph channel error: %s", e)
    if graph_ranked:
        ranked_lists.append(graph_ranked)

    if not all_records:
        return []

    # ------------------------------------------------------------------
    # RRF fusion
    # ------------------------------------------------------------------
    rrf_scores = reciprocal_rank_fusion(ranked_lists)

    # Pre-build rank index dicts (O(1) lookup instead of O(n) scan)
    sem_index = {mid: i for i, (mid, _) in enumerate(semantic_ranked)}
    bm_index = {mid: i for i, (mid, _) in enumerate(bm25_ranked)}
    gr_index = {mid: i for i, (mid, _) in enumerate(graph_ranked)}

    # Build candidate list sorted by RRF score
    candidates: List[RetrievalResult] = []
    for mem_id, rrf_score in sorted(rrf_scores.items(), key=lambda x: -x[1]):
        record = all_records.get(mem_id)
        if not record:
            continue

        candidates.append(RetrievalResult(
            record=record,
            rrf_score=rrf_score,
            semantic_rank=sem_index.get(mem_id, -1) + 1,
            bm25_rank=bm_index.get(mem_id, -1) + 1,
            graph_rank=gr_index.get(mem_id, -1) + 1,
            channels_hit=channel_hits.get(mem_id, 0),
        ))

    # Take top candidates for reranking
    top_candidates = candidates[:k_candidates]

    # ------------------------------------------------------------------
    # Cross-encoder reranking
    # ------------------------------------------------------------------
    reranked = False
    if use_reranker and len(top_candidates) > 1:
        encoder = _get_cross_encoder()
        if encoder is not None:
            try:
                pairs = [(query, c.record.content[:512]) for c in top_candidates]
                scores = encoder.predict(pairs, show_progress_bar=False)
                for i, score in enumerate(scores):
                    top_candidates[i].rerank_score = float(score)
                reranked = True
                top_candidates.sort(key=lambda c: c.rerank_score, reverse=True)
            except Exception as e:
                logger.debug("[Retrieval] Cross-encoder rerank error: %s", e)

    # ------------------------------------------------------------------
    # Normalize rerank scores to [0, 1] (min-max over batch)
    # ------------------------------------------------------------------
    if reranked:
        raw_scores = [c.rerank_score for c in top_candidates
                      if c.rerank_score != _RERANK_NOT_SET]
        if raw_scores:
            lo, hi = min(raw_scores), max(raw_scores)
            span = hi - lo if hi > lo else 1.0
            for c in top_candidates:
                if c.rerank_score != _RERANK_NOT_SET:
                    c.rerank_score = (c.rerank_score - lo) / span

    # ------------------------------------------------------------------
    # FadeMem strength multiplier
    # ------------------------------------------------------------------
    now_ts = datetime.now().timestamp()
    for candidate in top_candidates:
        rec = candidate.record
        try:
            la_ts = datetime.fromisoformat(rec.last_accessed).timestamp()
        except (ValueError, TypeError):
            la_ts = now_ts
        hours = max(0, (now_ts - la_ts) / 3600)
        strength = compute_strength(rec.strength, rec.decay_rate, hours)
        candidate.strength = strength

        # Final score: use rerank score if reranking was performed, else RRF
        if reranked and candidate.rerank_score != _RERANK_NOT_SET:
            candidate.score = candidate.rerank_score * strength
        else:
            candidate.score = candidate.rrf_score * strength

    # Final sort by score
    top_candidates.sort(key=lambda c: c.score, reverse=True)
    final = top_candidates[:k]

    # ------------------------------------------------------------------
    # Touch accessed memories (reinforce via spaced repetition)
    # ------------------------------------------------------------------
    if touch_results and final:
        for result in final:
            try:
                reinforce(store, result.record.id)
            except Exception as e:
                logger.warning("[Retrieval] reinforce failed for %s: %s", result.record.id, e)

    logger.debug(
        "[Retrieval] query=%r → %d semantic, %d bm25, %d graph → %d RRF → %d final",
        query[:50], len(semantic_ranked), len(bm25_ranked), len(graph_ranked),
        len(candidates), len(final),
    )
    return final


__all__ = [
    "RetrievalResult",
    "retrieve",
    "reciprocal_rank_fusion",
]
