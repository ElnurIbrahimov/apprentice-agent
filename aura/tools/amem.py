"""
A-MEM: Agentic Memory System for Aura
Zettelkasten-inspired memory with dynamic linking and evolution.

Based on: "A-MEM: Agentic Memory for LLM Agents" (NeurIPS 2025)
https://arxiv.org/abs/2502.12110

WARNING: This module uses sentence-transformers (all-MiniLM-L6-v2) for embeddings,
while the main memory system (aura.memory) uses Ollama nomic-embed-text.
These produce INCOMPATIBLE vector spaces — memories stored here cannot be
compared with memories in the unified store. This module is being superseded
by aura.memory.unified_memory. New code should use UnifiedMemory instead.

Features:
- Atomic notes with structured attributes (keywords, tags, context)
- Semantic embedding-based linking via Qdrant/sentence-transformers
- Memory evolution: related memories update when new ones are added
- LLM-driven contextual understanding
- Box-based organization (soft clustering)

Author: Aura Development Team
Created: 2026-02-03

ARCHITECTURE NOTE — A-MEM layer chain:
  - THIS FILE (aura/tools/amem.py): Core AMEMSystem — atomic notes, embeddings,
    Qdrant vector search, memory evolution. Used directly as amem_tool's backend.
  - aura/tools/amem_tool.py: Tool wrapper that exposes AMEMSystem to the agent via
    a natural-language action interface. ApprenticeAgent.tools["amem"] → AMEMTool.
  - aura/tools/hybrid_amem.py: HybridAMEM unifies AMEMSystem + KnowledgeGraph into
    a single store — entity extraction populates both backends simultaneously.
"""

from aura.jsonl_utils import rotate_jsonl_if_needed

import json
import hashlib
import logging
import math
import re
import threading
import uuid
import tempfile
import os
from collections import Counter, OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any, Set, Tuple
from dataclasses import dataclass, asdict, field
import numpy as np


def _deterministic_point_id(note_id: str) -> int:
    """Deterministic Qdrant point ID from note ID (survives process restarts)."""
    return int(hashlib.md5(note_id.encode()).hexdigest()[:15], 16)


class _BoundedEmbeddingCache:
    """OrderedDict-backed LRU cache for embeddings with bounded memory usage."""

    def __init__(self, maxsize: int = 10_000):
        self._cache: OrderedDict = OrderedDict()
        self._maxsize = maxsize

    def get(self, key: str):
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def set(self, key: str, value) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self._maxsize:
                self._cache.popitem(last=False)  # Evict oldest
        self._cache[key] = value

    def pop(self, key: str, default=None):
        return self._cache.pop(key, default)

    def __contains__(self, key: str) -> bool:
        return key in self._cache

    def __len__(self) -> int:
        return len(self._cache)

    def items(self):
        return self._cache.items()

    def keys(self):
        return self._cache.keys()

    def clear(self):
        self._cache.clear()

logger = logging.getLogger(__name__)

# sentence_transformers is lazy-loaded via aura.tools._shared_models.get_sentence_transformer()

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance, VectorParams, PointStruct,
        Filter, FieldCondition, MatchValue
    )
    HAS_QDRANT = True
except ImportError:
    HAS_QDRANT = False
    logger.warning("qdrant-client not installed. Using JSON-based storage.")


@dataclass
class MemoryNote:
    """
    Atomic memory note following Zettelkasten principles.

    Each note is self-contained with rich metadata for linking.
    """
    id: str
    content: str

    # LLM-generated attributes
    keywords: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    context: str = ""  # Contextual summary/description

    # Organization
    category: str = "general"  # episodic, semantic, procedural, fact
    boxes: List[str] = field(default_factory=list)  # Soft clusters

    # Linking
    links: List[str] = field(default_factory=list)  # IDs of related notes
    backlinks: List[str] = field(default_factory=list)  # Notes that link to this
    link_strengths: Dict[str, float] = field(default_factory=dict)  # link_id -> strength

    # Metadata
    created_at: str = ""
    updated_at: str = ""
    accessed_at: str = ""
    access_count: int = 0
    importance: float = 0.5  # 0-1 score
    source: str = "user"  # user, conversation, tool, inference

    # Emotional state at time of storage (PAD: pleasure, arousal, dominance)
    emotional_pad: Optional[Dict[str, float]] = None

    # Embedding (stored separately for efficiency)
    has_embedding: bool = False

    def __post_init__(self):
        now = datetime.now().isoformat()
        if not self.created_at:
            self.created_at = now
        if not self.updated_at:
            self.updated_at = now
        if not self.accessed_at:
            self.accessed_at = now

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryNote':
        """Create from dictionary."""
        return cls(**data)

    def touch(self):
        """Mark as accessed (spaced repetition — resets decay timer, Phase 4B)."""
        self.accessed_at = datetime.now().isoformat()
        self.access_count += 1
        # Spaced repetition: slight importance boost on access
        self.importance = min(1.0, self.importance + 0.01)

    def get_recency_score(self, half_life_hours: float = 336.0) -> float:
        """
        Ebbinghaus exponential decay score based on time since last access (Phase 4B).

        Args:
            half_life_hours: Hours for score to decay to 0.5 (default: 2 weeks)

        Returns:
            Score from 0.0 to 1.0
        """
        try:
            last = datetime.fromisoformat(self.accessed_at)
            age_hours = (datetime.now() - last).total_seconds() / 3600
            decay_rate = math.log(2) / half_life_hours
            return math.exp(-decay_rate * age_hours)
        except (ValueError, TypeError):
            return 0.5  # Unknown age, neutral score

    def add_link(self, target_id: str, strength: float = 0.5):
        """Add a link to another note."""
        if target_id not in self.links:
            self.links.append(target_id)
        self.link_strengths[target_id] = min(1.0, strength)
        self.updated_at = datetime.now().isoformat()

    def add_backlink(self, source_id: str):
        """Record that another note links to this one."""
        if source_id not in self.backlinks:
            self.backlinks.append(source_id)

    def strengthen_link(self, target_id: str, amount: float = 0.1):
        """Reinforce a link through use."""
        if target_id in self.link_strengths:
            self.link_strengths[target_id] = min(1.0, self.link_strengths[target_id] + amount)

    def format_display(self) -> str:
        """Format for display."""
        tags_str = ", ".join(self.tags[:3]) if self.tags else "no tags"
        return f"[{self.id[:8]}] {self.content[:50]}... ({tags_str})"


class AMEMSystem:
    """
    A-MEM: Agentic Memory System

    Implements Zettelkasten-inspired memory with:
    - Atomic notes with rich metadata
    - Embedding-based semantic linking
    - LLM-driven attribute extraction
    - Memory evolution on new additions
    """

    def __init__(
        self,
        db_path: str = "data/amem/",
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_func: Optional[callable] = None,
        evolution_enabled: bool = True,
        max_links_per_note: int = 10,
        link_threshold: float = 0.5
    ):
        """
        Initialize A-MEM system.

        Args:
            db_path: Storage directory
            embedding_model: Sentence transformer model name
            llm_func: Function to call LLM for attribute extraction
            evolution_enabled: Whether to update related notes on add
            max_links_per_note: Maximum automatic links per note
            link_threshold: Minimum similarity for auto-linking
        """
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)

        self.llm_func = llm_func
        self.evolution_enabled = evolution_enabled
        self.max_links = max_links_per_note
        self.link_threshold = link_threshold

        # Thread safety
        self._lock = threading.RLock()

        # Storage
        self._notes: Dict[str, MemoryNote] = {}
        self._boxes: Dict[str, Set[str]] = {}  # box_name -> set of note_ids

        # Embedding model
        self._embedder = None
        self._embeddings: _BoundedEmbeddingCache = _BoundedEmbeddingCache(maxsize=10_000)

        self._embedding_model_name = embedding_model  # loaded lazily on first use

        # Qdrant for vector search (optional)
        self._qdrant_client = None
        self._qdrant_collection = "amem_notes"
        self._qdrant_dim = None  # Set when first embedding is computed

        if HAS_QDRANT:
            try:
                qdrant_path = str(self.db_path / "qdrant")
                self._qdrant_client = QdrantClient(path=qdrant_path)
                logger.info("Qdrant initialized for A-MEM")
            except Exception as e:
                logger.warning(f"Failed to initialize Qdrant: {e}")

        # File paths
        self.notes_file = self.db_path / "notes.jsonl"
        self.embeddings_file = self.db_path / "embeddings.npz"
        self.boxes_file = self.db_path / "boxes.json"
        self.stats_file = self.db_path / "stats.json"

        # Load existing data
        self._load()

    # =========================================================================
    # PERSISTENCE HELPERS
    # =========================================================================

    def _atomic_write_json(self, path, data: dict) -> None:
        """Write JSON atomically using temp file + os.replace (crash-safe)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=str(path.parent),
            prefix=f".{path.name}.tmp.",
            suffix=".json"
        )
        try:
            with os.fdopen(tmp_fd, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            os.replace(tmp_path, str(path))
        except Exception as e:
            logger.error(f"[AMEM] Atomic JSON write failed for {path}: {e}")
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def _atomic_write_jsonl(self, path, records: list) -> None:
        """Write JSONL atomically using temp file + os.replace (crash-safe)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=str(path.parent),
            prefix=f".{path.name}.tmp.",
            suffix=".jsonl"
        )
        try:
            with os.fdopen(tmp_fd, 'w', encoding='utf-8') as f:
                for rec in records:
                    f.write(json.dumps(rec, ensure_ascii=False, default=str) + '\n')
            os.replace(tmp_path, str(path))
        except Exception as e:
            logger.error(f"[AMEM] Atomic JSONL write failed for {path}: {e}")
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def _ensure_qdrant_collection(self, dim: int) -> bool:
        """Create Qdrant collection if it doesn't exist. Returns True if ready."""
        if not self._qdrant_client:
            return False
        try:
            collections = [c.name for c in self._qdrant_client.get_collections().collections]
            if self._qdrant_collection not in collections:
                self._qdrant_client.create_collection(
                    collection_name=self._qdrant_collection,
                    vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
                )
            self._qdrant_dim = dim
            return True
        except Exception as e:
            logger.warning(f"[A-MEM] Qdrant collection setup failed: {e}")
            return False

    # =========================================================================
    # CORE OPERATIONS
    # =========================================================================

    def add(
        self,
        content: str,
        tags: Optional[List[str]] = None,
        category: str = "general",
        source: str = "user",
        importance: float = 0.5,
        auto_extract: bool = True,
        auto_link: bool = True,
        auto_evolve: bool = True
    ) -> MemoryNote:
        """
        Add a new memory note.

        This is the main entry point following the Zettelkasten method:
        1. Create atomic note with content
        2. Extract keywords, tags, context via LLM (if enabled)
        3. Generate embedding for semantic search
        4. Find and create links to related notes
        5. Evolve related notes (update their context/links)

        Args:
            content: The memory content (should be atomic/self-contained)
            tags: Optional user-provided tags
            category: episodic, semantic, procedural, fact, general
            source: Where this memory came from
            importance: How important is this memory (0-1)
            auto_extract: Use LLM to extract keywords/context
            auto_link: Automatically link to similar notes
            auto_evolve: Update related notes when this is added

        Returns:
            The created MemoryNote
        """
        with self._lock:
            # Generate unique ID
            note_id = self._generate_id(content)

            # Check for duplicate content
            existing = self._find_exact_match(content)
            if existing:
                existing.touch()
                existing.importance = max(existing.importance, importance)
                return existing

            # Create note
            note = MemoryNote(
                id=note_id,
                content=content,
                tags=tags or [],
                category=category,
                source=source,
                importance=importance
            )

            # Capture current emotional state (PAD) for emotional memory tagging
            # and modulate importance via dopamine (high dopamine = more significant)
            try:
                from aura.emotion.alma_engine import alma_engine
                state = alma_engine.get_emotional_state()
                if state and "pad" in state:
                    note.emotional_pad = state["pad"]
                if state and "neuromodulators" in state:
                    dopamine = state["neuromodulators"].get("dopamine", 0.5)
                    # Scale importance by ±20% based on dopamine
                    dopamine_factor = 1.0 + (dopamine - 0.5) * 0.4
                    note.importance = max(0.0, min(1.0, note.importance * dopamine_factor))
            except Exception as e:
                logger.debug(f"[AMem] non-critical: {e}")
            # Step 1: Extract attributes via LLM
            if auto_extract and self.llm_func:
                self._extract_attributes(note)
            else:
                # Basic keyword extraction without LLM
                note.keywords = self._extract_keywords_simple(content)

            # Step 2: Generate embedding
            embedding = self._embed(content)
            if embedding is not None:
                self._embeddings.set(note_id, embedding)
                note.has_embedding = True

                # Add to Qdrant if available
                if self._qdrant_client is not None:
                    vec = embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)
                    if self._ensure_qdrant_collection(len(vec)):
                        payload = {
                            "note_id": note_id,
                            "content": note.content[:500],
                            "keywords": note.keywords,
                            "tags": note.tags,
                            "category": note.category,
                            "importance": note.importance,
                            "created_at": note.created_at,
                        }
                        if note.emotional_pad:
                            payload["emotion_pleasure"] = note.emotional_pad.get("pleasure", 0.0)
                            payload["emotion_arousal"] = note.emotional_pad.get("arousal", 0.0)
                            payload["emotion_dominance"] = note.emotional_pad.get("dominance", 0.0)
                        try:
                            point_id = _deterministic_point_id(note.id)
                            self._qdrant_client.upsert(
                                collection_name=self._qdrant_collection,
                                points=[PointStruct(id=point_id, vector=vec, payload=payload)]
                            )
                        except Exception as e:
                            logger.warning(f"[A-MEM] Qdrant add failed: {e}")

            # Step 3: Find and create links
            if auto_link:
                related = self._find_related(note, embedding)
                for related_id, similarity in related[:self.max_links]:
                    if similarity >= self.link_threshold:
                        note.add_link(related_id, similarity)
                        # Add backlink
                        if related_id in self._notes:
                            self._notes[related_id].add_backlink(note_id)

            # Step 4: Assign to boxes (soft clustering)
            self._assign_boxes(note)

            # Store note
            self._notes[note_id] = note
            self._append_note(note)

            logger.info(f"Added note {note_id[:8]} with {len(note.links)} links")

        # Step 5: Evolve related notes (outside lock — involves LLM I/O)
        if auto_evolve and self.evolution_enabled:
            self._evolve_related(note)

        return note

    def get_note(self, note_id: str) -> Optional[MemoryNote]:
        """Retrieve a note by ID without touch (non-mutating lookup for internal use)."""
        with self._lock:
            return self._notes.get(note_id)

    def read(self, note_id: str) -> Optional[MemoryNote]:
        """
        Retrieve a note by ID.

        Also strengthens links from recently accessed notes (association).
        """
        with self._lock:
            note = self._notes.get(note_id)
            if note:
                note.touch()
            return note

    def search(
        self,
        query: str,
        k: int = 5,
        category: Optional[str] = None,
        min_importance: float = 0.0,
        mood_pad: Optional[Dict[str, float]] = None
    ) -> List[Tuple[MemoryNote, float]]:
        """
        Semantic search for related notes.

        Args:
            query: Search query
            k: Number of results
            category: Filter by category
            min_importance: Minimum importance threshold

        Returns:
            List of (note, similarity_score) tuples
        """
        with self._lock:
            # Try Qdrant first
            if self._qdrant_client and self._qdrant_dim and self._embedder:
                try:
                    query_embedding = self._embed(query)
                    if query_embedding is not None:
                        query_vec = query_embedding.tolist() if hasattr(query_embedding, 'tolist') else list(query_embedding)
                        qdrant_results = self._qdrant_client.search(
                            collection_name=self._qdrant_collection,
                            query_vector=query_vec,
                            limit=k * 2,
                            with_payload=True
                        )

                        matches = []
                        for hit in qdrant_results:
                            note_id = hit.payload.get("note_id")
                            if note_id and note_id in self._notes:
                                note = self._notes[note_id]
                                if note.importance >= min_importance:
                                    if category and note.category != category:
                                        continue
                                    # Qdrant returns cosine similarity score directly
                                    similarity = hit.score
                                    # Phase 4B: Blend similarity with Ebbinghaus recency
                                    # Serotonin modulates recency bias: high = patient (lower recency weight),
                                    # low = impatient (higher recency weight)
                                    recency = note.get_recency_score()
                                    try:
                                        from aura.emotion.alma_engine import alma_engine
                                        _s = alma_engine.get_emotional_state()
                                        _sero = _s.get("neuromodulators", {}).get("serotonin", 0.5) if _s else 0.5
                                    except (ImportError, Exception) as e:
                                        logger.debug(f"[AMEM] Serotonin fetch failed, using default: {e}")
                                        _sero = 0.5
                                    recency_w = 0.2 + (_sero - 0.5) * -0.1  # 0.25 at low, 0.15 at high
                                    recency_w = max(0.15, min(0.25, recency_w))
                                    sim_w = 0.9 - recency_w  # remaining budget minus importance
                                    blended = sim_w * similarity + recency_w * recency + 0.1 * note.importance
                                    matches.append((note, blended))

                        final = sorted(matches, key=lambda x: x[1], reverse=True)[:k]
                        # Apply mood-congruent reranking
                        try:
                            from aura.tools.mood_memory import apply_mood_bias_to_tuples
                            final = apply_mood_bias_to_tuples(final, mood_pad)
                            final.sort(key=lambda x: x[1], reverse=True)
                        except Exception as e:
                            logger.debug(f"[AMem] non-critical: {e}")
                        # === PHASE 1: Track memory recall ===
                        try:
                            from api.routes.memory import record_memory_recall
                            if final:
                                record_memory_recall("amem", len(final), query, [n.content[:80] for n, _ in final[:5]])
                        except Exception as e:
                            logger.debug(f"[AMem] non-critical: {e}")
                        try:
                            from api.routes.context import track_context_from_memory
                            if final:
                                track_context_from_memory([n.content[:80] for n, _ in final[:5]])
                        except Exception as e:
                            logger.debug(f"[AMem] non-critical: {e}")
                        return final
                except Exception as e:
                    logger.warning(f"[A-MEM] Qdrant search failed: {e}")

            # Fallback: embedding similarity search
            if self._embedder:
                query_embedding = self._embed(query)
                if query_embedding is not None:
                    emb_results = self._search_by_embedding(query_embedding, k, category, min_importance)
                    # Apply mood-congruent reranking
                    try:
                        from aura.tools.mood_memory import apply_mood_bias_to_tuples
                        emb_results = apply_mood_bias_to_tuples(emb_results, mood_pad)
                        emb_results.sort(key=lambda x: x[1], reverse=True)
                    except Exception as e:
                        logger.debug(f"[AMem] non-critical: {e}")
                    # === PHASE 1: Track memory recall ===
                    try:
                        from api.routes.memory import record_memory_recall
                        if emb_results:
                            record_memory_recall("amem", len(emb_results), query, [n.content[:80] for n, _ in emb_results[:5]])
                    except Exception as e:
                        logger.debug(f"[AMem] non-critical: {e}")
                    try:
                        from api.routes.context import track_context_from_memory
                        if emb_results:
                            track_context_from_memory([n.content[:80] for n, _ in emb_results[:5]])
                    except Exception as e:
                        logger.debug(f"[AMem] non-critical: {e}")
                    return emb_results

            # Final fallback: keyword search
            results = self._search_by_keywords(query, k, category, min_importance)
            # === PHASE 1: Track memory recall ===
            try:
                from api.routes.memory import record_memory_recall
                if results:
                    record_memory_recall("amem", len(results), query, [n.content[:80] for n, _ in results[:5]])
            except Exception as e:
                logger.debug(f"[AMem] non-critical: {e}")
            try:
                from api.routes.context import track_context_from_memory
                if results:
                    track_context_from_memory([n.content[:80] for n, _ in results[:5]])
            except Exception as e:
                logger.debug(f"[AMem] non-critical: {e}")
            return results

    def search_agentic(
        self,
        query: str,
        k: int = 5,
        follow_links: bool = True,
        max_hops: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Agentic search with link traversal.

        Finds relevant notes and optionally follows their links
        to discover related knowledge.

        Returns list of dicts with id, content, tags, relevance fields.
        """
        with self._lock:
            # Initial search
            direct_results = self.search(query, k=k)

            if not follow_links:
                return [
                    {
                        "id": note.id,
                        "content": note.content,
                        "tags": note.tags,
                        "relevance": score,
                        "hop": 0
                    }
                    for note, score in direct_results
                ]

            # Follow links
            seen = set()
            results = []

            for note, score in direct_results:
                if note.id not in seen:
                    seen.add(note.id)
                    results.append({
                        "id": note.id,
                        "content": note.content,
                        "tags": note.tags,
                        "keywords": note.keywords,
                        "context": note.context,
                        "relevance": score,
                        "hop": 0
                    })

                # Follow links (BFS)
                frontier = [(note.id, 1)]
                while frontier and len(results) < k * 3:
                    current_id, hop = frontier.pop(0)
                    if hop > max_hops:
                        continue

                    current = self._notes.get(current_id)
                    if not current:
                        continue

                    for linked_id in current.links:
                        if linked_id not in seen and linked_id in self._notes:
                            seen.add(linked_id)
                            linked = self._notes[linked_id]
                            link_strength = current.link_strengths.get(linked_id, 0.5)

                            results.append({
                                "id": linked.id,
                                "content": linked.content,
                                "tags": linked.tags,
                                "keywords": linked.keywords,
                                "context": linked.context,
                                "relevance": score * link_strength * (0.7 ** hop),
                                "hop": hop
                            })

                            if hop < max_hops:
                                frontier.append((linked_id, hop + 1))

            # Sort by relevance
            results.sort(key=lambda x: x["relevance"], reverse=True)
            try:
                from api.routes.memory import record_memory_recall
                from api.routes.context import track_context_from_memory
                record_memory_recall("amem", len(results), query, [r["content"][:80] for r in results[:5]])
                track_context_from_memory([r["content"][:80] for r in results[:5]])
            except Exception as e:
                logger.debug(f"[AMem] non-critical: {e}")
            return results[:k * 2]

    def update(
        self,
        note_id: str,
        content: Optional[str] = None,
        tags: Optional[List[str]] = None,
        importance: Optional[float] = None,
        re_extract: bool = False
    ) -> bool:
        """
        Update a note's content or metadata.

        If content changes significantly, re-links and re-evolves.
        """
        with self._lock:
            note = self._notes.get(note_id)
            if not note:
                return False

            content_changed = False

            if content and content != note.content:
                note.content = content
                content_changed = True

            if tags is not None:
                note.tags = tags

            if importance is not None:
                note.importance = importance

            note.updated_at = datetime.now().isoformat()

            # Re-extract and re-link if content changed
            if content_changed or re_extract:
                if self.llm_func:
                    self._extract_attributes(note)

                # Update embedding
                new_embedding = self._embed(note.content)
                if new_embedding is not None:
                    self._embeddings.set(note_id, new_embedding)

                    if self._qdrant_client is not None:
                        _emb = new_embedding.tolist() if hasattr(new_embedding, 'tolist') else list(new_embedding)
                        if self._ensure_qdrant_collection(len(_emb)):
                            _tags = note.tags
                            if isinstance(_tags, set):
                                _tags = sorted(_tags)
                            elif not isinstance(_tags, list):
                                _tags = list(_tags) if _tags else []
                            _payload = {
                                "note_id": note_id,
                                "content": note.content[:500],
                                "keywords": note.keywords,
                                "tags": _tags,
                                "category": note.category or "",
                                "importance": float(note.importance),
                                "created_at": note.created_at or "",
                            }
                            try:
                                point_id = _deterministic_point_id(note_id)
                                self._qdrant_client.upsert(
                                    collection_name=self._qdrant_collection,
                                    points=[PointStruct(id=point_id, vector=_emb, payload=_payload)]
                                )
                            except Exception as e:
                                logger.warning(f"[A-MEM] Qdrant upsert failed: {e}")

                # Re-link
                note.links = []
                note.link_strengths = {}
                related = self._find_related(note, new_embedding)
                for related_id, similarity in related[:self.max_links]:
                    if similarity >= self.link_threshold:
                        note.add_link(related_id, similarity)

            # Persist updated note to disk
            self.save()
            return True

    def delete(self, note_id: str) -> bool:
        """
        Delete a note and clean up its links.
        """
        with self._lock:
            if note_id not in self._notes:
                return False

            note = self._notes[note_id]

            # Remove backlinks from linked notes
            for linked_id in note.links:
                if linked_id in self._notes:
                    linked = self._notes[linked_id]
                    if note_id in linked.backlinks:
                        linked.backlinks.remove(note_id)

            # Remove from boxes
            for box_name in note.boxes:
                if box_name in self._boxes:
                    self._boxes[box_name].discard(note_id)

            # Remove from Qdrant
            if self._qdrant_client:
                try:
                    point_id = _deterministic_point_id(note_id)
                    self._qdrant_client.delete(
                        collection_name=self._qdrant_collection,
                        points_selector=[point_id]
                    )
                except Exception as e:
                    logger.warning(f"[A-MEM] Qdrant delete failed: {e}")

            # Remove embedding
            self._embeddings.pop(note_id, None)

            # Remove note
            del self._notes[note_id]

            self.save()
            return True

    # =========================================================================
    # LINKING OPERATIONS
    # =========================================================================

    def link(
        self,
        source_id: str,
        target_id: str,
        strength: float = 0.7
    ) -> bool:
        """
        Manually create a link between notes.
        """
        with self._lock:
            source = self._notes.get(source_id)
            target = self._notes.get(target_id)

            if not source or not target:
                return False

            source.add_link(target_id, strength)
            target.add_backlink(source_id)
            return True

    def strengthen_association(
        self,
        note_id1: str,
        note_id2: str,
        amount: float = 0.1
    ):
        """
        Strengthen bidirectional link between notes (learning).

        Call this when two notes are accessed together.
        """
        with self._lock:
            note1 = self._notes.get(note_id1)
            note2 = self._notes.get(note_id2)

            if note1 and note2:
                if note_id2 in note1.links:
                    note1.strengthen_link(note_id2, amount)
                else:
                    note1.add_link(note_id2, amount)
                    note2.add_backlink(note_id1)

                if note_id1 in note2.links:
                    note2.strengthen_link(note_id1, amount)

    def get_linked(
        self,
        note_id: str,
        direction: str = "both",
        min_strength: float = 0.0
    ) -> List[Tuple[MemoryNote, float]]:
        """
        Get notes linked to/from a note.

        Args:
            note_id: Source note ID
            direction: "out" (links), "in" (backlinks), or "both"
            min_strength: Minimum link strength

        Returns:
            List of (note, strength) tuples
        """
        with self._lock:
            note = self._notes.get(note_id)
            if not note:
                return []

            results = []

            if direction in ("out", "both"):
                for linked_id in note.links:
                    strength = note.link_strengths.get(linked_id, 0.5)
                    if strength >= min_strength and linked_id in self._notes:
                        results.append((self._notes[linked_id], strength))

            if direction in ("in", "both"):
                for backlink_id in note.backlinks:
                    if backlink_id in self._notes and backlink_id not in [r[0].id for r in results]:
                        # Get strength from the linking note
                        linker = self._notes[backlink_id]
                        strength = linker.link_strengths.get(note_id, 0.5)
                        if strength >= min_strength:
                            results.append((self._notes[backlink_id], strength))

            return sorted(results, key=lambda x: x[1], reverse=True)

    # =========================================================================
    # BOX OPERATIONS (Soft Clustering)
    # =========================================================================

    def get_box(self, box_name: str) -> List[MemoryNote]:
        """Get all notes in a box."""
        with self._lock:
            note_ids = self._boxes.get(box_name, set())
            return [self._notes[nid] for nid in note_ids if nid in self._notes]

    def list_boxes(self) -> Dict[str, int]:
        """List all boxes with note counts."""
        with self._lock:
            return {name: len(ids) for name, ids in self._boxes.items()}

    def add_to_box(self, note_id: str, box_name: str):
        """Manually add a note to a box."""
        with self._lock:
            if note_id in self._notes:
                if box_name not in self._boxes:
                    self._boxes[box_name] = set()
                self._boxes[box_name].add(note_id)

                note = self._notes[note_id]
                if box_name not in note.boxes:
                    note.boxes.append(box_name)

    # =========================================================================
    # EVOLUTION (Memory Updates)
    # =========================================================================

    def _evolve_related(self, new_note: MemoryNote):
        """
        Evolve related notes when a new note is added.

        This implements the A-MEM memory evolution principle:
        when new knowledge is added, it can update the context
        and connections of related existing knowledge.

        Runs outside the main add() lock (involves LLM I/O), but
        re-acquires self._lock when mutating linked note attributes.
        """
        if not self.llm_func:
            return

        # Get strongly linked notes
        for linked_id in new_note.links:
            strength = new_note.link_strengths.get(linked_id, 0)
            if strength < 0.6:  # Only evolve strongly related
                continue

            # Snapshot linked note data under lock for the LLM prompt
            with self._lock:
                linked = self._notes.get(linked_id)
                if not linked:
                    continue
                linked_content = linked.content
                linked_context = linked.context
                linked_keywords = list(linked.keywords)

            # Update context to incorporate new relationship
            try:
                evolution_prompt = f"""A new memory has been added that relates to an existing memory.

Existing memory:
Content: {linked_content}
Context: {linked_context}
Keywords: {', '.join(linked_keywords)}

New related memory:
Content: {new_note.content}
Keywords: {', '.join(new_note.keywords)}

Update the context of the existing memory to reflect this new connection.
Keep it brief (1-2 sentences). Only output the updated context, nothing else."""

                updated_context = self.llm_func(evolution_prompt)
                if updated_context and len(updated_context) < 500:
                    with self._lock:
                        # Re-check note still exists after LLM call
                        linked = self._notes.get(linked_id)
                        if linked:
                            linked.context = updated_context.strip()
                            linked.updated_at = datetime.now().isoformat()
                            logger.debug(f"Evolved note {linked_id[:8]}")

            except Exception as e:
                logger.warning(f"Evolution failed for {linked_id}: {e}")

    # =========================================================================
    # ATTRIBUTE EXTRACTION
    # =========================================================================

    def _extract_attributes(self, note: MemoryNote):
        """
        Use LLM to extract structured attributes from content.

        Extracts: keywords, context summary, suggested tags
        """
        if not self.llm_func:
            return

        try:
            prompt = f"""Analyze this memory and extract structured attributes.

Memory content: {note.content}

Provide your response in this exact format:
KEYWORDS: word1, word2, word3, word4, word5
CONTEXT: Brief 1-sentence contextual description
TAGS: tag1, tag2, tag3

Guidelines:
- KEYWORDS: 3-7 key concepts/terms from the content
- CONTEXT: What this memory is about and why it matters
- TAGS: Categories this belongs to (e.g., coding, personal, learning, tool-use)"""

            response = self.llm_func(prompt)

            # Parse response
            for line in response.strip().split('\n'):
                line = line.strip()
                if line.startswith('KEYWORDS:'):
                    keywords = [k.strip() for k in line[9:].split(',') if k.strip()]
                    note.keywords = keywords[:10]
                elif line.startswith('CONTEXT:'):
                    note.context = line[8:].strip()[:300]
                elif line.startswith('TAGS:'):
                    extracted_tags = [t.strip().lower() for t in line[5:].split(',') if t.strip()]
                    # Merge with existing tags
                    note.tags = list(set(note.tags + extracted_tags))[:10]

        except Exception as e:
            logger.warning(f"Attribute extraction failed: {e}")
            # Fallback to simple extraction
            note.keywords = self._extract_keywords_simple(note.content)

    def _extract_keywords_simple(self, text: str) -> List[str]:
        """Simple keyword extraction without LLM."""
        # Stopwords
        stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'must', 'shall',
            'can', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
            'from', 'as', 'into', 'through', 'during', 'before', 'after',
            'above', 'below', 'between', 'under', 'again', 'further',
            'then', 'once', 'here', 'there', 'when', 'where', 'why',
            'how', 'all', 'each', 'few', 'more', 'most', 'other', 'some',
            'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
            'than', 'too', 'very', 'just', 'and', 'but', 'if', 'or',
            'because', 'until', 'while', 'this', 'that', 'these', 'those',
            'we', 'our', 'he', 'she', 'it'
        }

        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        words = [w for w in words if w not in stopwords]

        # Fallback: if all words were filtered, use original meaningful words
        if not words:
            words = [w for w in re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
                     if w not in {'the', 'a', 'an', 'is', 'are', 'was', 'were',
                                  'and', 'but', 'or', 'not', 'for', 'with'}]

        counts = Counter(words)
        return [word for word, _ in counts.most_common(7)]

    # =========================================================================
    # EMBEDDING & SIMILARITY
    # =========================================================================

    def _embed(self, text: str) -> Optional[np.ndarray]:
        """Generate embedding for text. Lazy-loads the model on first call."""
        if not self._embedder:
            # Lazy-load via shared singleton (avoids ~5s startup penalty)
            try:
                from aura.tools._shared_models import get_sentence_transformer
                self._embedder = get_sentence_transformer()
                logger.info(f"Loaded embedding model: {self._embedding_model_name}")
            except Exception as e:
                logger.warning(f"Failed to load embedding model: {e}")
                return None

        try:
            embedding = self._embedder.encode(text, convert_to_numpy=True, show_progress_bar=False)
            return embedding / np.linalg.norm(embedding)  # Normalize
        except Exception as e:
            logger.warning(f"Embedding failed: {e}")
            return None

    def _find_related(
        self,
        note: MemoryNote,
        embedding: Optional[np.ndarray]
    ) -> List[Tuple[str, float]]:
        """
        Find notes related to the given note.

        Returns list of (note_id, similarity) tuples.
        """
        if embedding is None or not self._embeddings:
            # Fallback: keyword overlap
            return self._find_related_by_keywords(note)

        similarities = []
        for other_id, other_embedding in self._embeddings.items():
            if other_id == note.id:
                continue

            # Cosine similarity (embeddings are normalized)
            similarity = float(np.dot(embedding, other_embedding))
            similarities.append((other_id, similarity))

        return sorted(similarities, key=lambda x: x[1], reverse=True)

    def _find_related_by_keywords(
        self,
        note: MemoryNote
    ) -> List[Tuple[str, float]]:
        """Fallback: find related notes by keyword overlap."""
        note_keywords = set(note.keywords)
        if not note_keywords:
            return []

        similarities = []
        for other_id, other_note in self._notes.items():
            if other_id == note.id:
                continue

            other_keywords = set(other_note.keywords)
            if not other_keywords:
                continue

            # Jaccard similarity
            intersection = len(note_keywords & other_keywords)
            union = len(note_keywords | other_keywords)
            similarity = intersection / union if union > 0 else 0

            if similarity > 0:
                similarities.append((other_id, similarity))

        return sorted(similarities, key=lambda x: x[1], reverse=True)

    def _search_by_embedding(
        self,
        query_embedding: np.ndarray,
        k: int,
        category: Optional[str],
        min_importance: float
    ) -> List[Tuple[MemoryNote, float]]:
        """Search by embedding similarity."""
        similarities = []

        for note_id, embedding in self._embeddings.items():
            note = self._notes.get(note_id)
            if not note:
                continue
            if category and note.category != category:
                continue
            if note.importance < min_importance:
                continue

            similarity = float(np.dot(query_embedding, embedding))
            similarities.append((note, similarity))

        return sorted(similarities, key=lambda x: x[1], reverse=True)[:k]

    def _search_by_keywords(
        self,
        query: str,
        k: int,
        category: Optional[str],
        min_importance: float
    ) -> List[Tuple[MemoryNote, float]]:
        """Fallback: search by keyword matching."""
        query_keywords = set(self._extract_keywords_simple(query))

        matches = []
        for note in self._notes.values():
            if category and note.category != category:
                continue
            if note.importance < min_importance:
                continue

            note_keywords = set(note.keywords)

            # Calculate overlap
            if query_keywords and note_keywords:
                intersection = len(query_keywords & note_keywords)
                similarity = intersection / len(query_keywords)
            else:
                # Check content
                query_lower = query.lower()
                content_lower = note.content.lower()
                similarity = 0.5 if query_lower in content_lower else 0.0

            if similarity > 0:
                matches.append((note, similarity))

        return sorted(matches, key=lambda x: x[1], reverse=True)[:k]

    def _find_exact_match(self, content: str) -> Optional[MemoryNote]:
        """Check if content already exists."""
        content_hash = hashlib.md5(content.encode()).hexdigest()
        for note in self._notes.values():
            if hashlib.md5(note.content.encode()).hexdigest() == content_hash:
                return note
        return None

    # =========================================================================
    # BOX ASSIGNMENT
    # =========================================================================

    def _assign_boxes(self, note: MemoryNote):
        """
        Assign note to boxes based on tags and category.

        Boxes are soft clusters - a note can belong to multiple.
        """
        # Category box
        if note.category:
            box_name = f"category:{note.category}"
            if box_name not in self._boxes:
                self._boxes[box_name] = set()
            self._boxes[box_name].add(note.id)
            if box_name not in note.boxes:
                note.boxes.append(box_name)

        # Tag boxes
        for tag in note.tags[:5]:  # Limit to 5 tag boxes
            box_name = f"tag:{tag}"
            if box_name not in self._boxes:
                self._boxes[box_name] = set()
            self._boxes[box_name].add(note.id)
            if box_name not in note.boxes:
                note.boxes.append(box_name)

    # =========================================================================
    # UTILITY
    # =========================================================================

    def _generate_id(self, content: str) -> str:
        """Generate unique ID for note."""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        content_hash = hashlib.md5(content[:100].encode()).hexdigest()[:8]
        return f"note_{timestamp}_{content_hash}"

    # =========================================================================
    # PERSISTENCE
    # =========================================================================

    def _load(self):
        """Load data from disk."""
        with self._lock:
            # Load notes
            if self.notes_file.exists():
                with open(self.notes_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                data = json.loads(line)
                                note = MemoryNote.from_dict(data)
                                self._notes[note.id] = note
                            except Exception as e:
                                logger.warning(f"Failed to load note: {e}")

            # Load embeddings
            if self.embeddings_file.exists():
                try:
                    data = np.load(self.embeddings_file, allow_pickle=False)
                    _cache = _BoundedEmbeddingCache(maxsize=10_000)
                    for k in data.files:
                        _cache.set(k, data[k])
                    self._embeddings = _cache
                except Exception as e:
                    logger.warning(f"Failed to load embeddings: {e}")

            # Load boxes
            if self.boxes_file.exists():
                try:
                    with open(self.boxes_file, 'r', encoding='utf-8') as f:
                        boxes_data = json.load(f)
                        self._boxes = {k: set(v) for k, v in boxes_data.items()}
                except Exception as e:
                    logger.warning(f"Failed to load boxes: {e}")

            # Reconstruct Qdrant collection from loaded notes+embeddings
            _orphaned_ids = []
            if HAS_QDRANT and self._qdrant_client is not None and self._embeddings:
                try:
                    points_to_upsert = []
                    for note_id, embedding in self._embeddings.items():
                        note = self._notes.get(note_id)
                        if note is None:
                            continue
                        vec = embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)
                        if not self._qdrant_dim:
                            self._ensure_qdrant_collection(len(vec))
                        if not self._qdrant_dim:
                            continue
                        tags = note.tags
                        if isinstance(tags, set):
                            tags = sorted(tags)
                        elif not isinstance(tags, list):
                            tags = list(tags) if tags else []
                        payload = {
                            "note_id": note_id,
                            "content": note.content[:500],
                            "keywords": note.keywords,
                            "tags": tags,
                            "category": note.category or "",
                            "importance": float(note.importance),
                            "created_at": note.created_at or "",
                        }
                        point_id = _deterministic_point_id(note_id)
                        points_to_upsert.append(PointStruct(id=point_id, vector=vec, payload=payload))
                    for note_id in self._notes:
                        if note_id not in self._embeddings:
                            _orphaned_ids.append(note_id)
                    if points_to_upsert and self._qdrant_dim:
                        # Batch upsert in chunks of 100
                        for _i in range(0, len(points_to_upsert), 100):
                            self._qdrant_client.upsert(
                                collection_name=self._qdrant_collection,
                                points=points_to_upsert[_i:_i+100]
                            )
                    _reconstructed = len(points_to_upsert)
                    _total = len(self._notes)
                    _orphaned = len(_orphaned_ids)
                    logger.info(
                        f"[A-MEM] Qdrant: {_reconstructed}/{_total} notes synced. "
                        f"{_orphaned} notes missing embeddings — not searchable via semantic search."
                    )
                except Exception as e:
                    logger.warning(f"[A-MEM] Qdrant reconstruction failed: {e}")

            logger.info(f"A-MEM loaded {len(self._notes)} notes, {len(self._boxes)} boxes")

            # Spawn background thread to re-embed orphaned notes (non-blocking)
            if _orphaned_ids and self._embedder is not None:
                def _reembed_orphans(_self, _ids):
                    _reembedded = 0
                    for _nid in _ids:
                        # Get content snapshot outside lock (avoid holding lock during embed)
                        with _self._lock:
                            _note_snap = _self._notes.get(_nid)
                            _content_snap = _note_snap.content if _note_snap is not None else None
                        if _content_snap is None:
                            continue
                        _emb = _self._embed(_content_snap)  # expensive — outside lock
                        if _emb is None:
                            continue
                        with _self._lock:
                            _note = _self._notes.get(_nid)  # moved inside lock
                            if _note is None:
                                continue  # note deleted while we computed embedding
                            _self._embeddings.set(_nid, _emb)
                            _note.has_embedding = True
                            if _self._qdrant_client is not None:
                                try:
                                    _tags = _note.tags
                                    if isinstance(_tags, set):
                                        _tags = sorted(_tags)
                                    elif not isinstance(_tags, list):
                                        _tags = list(_tags) if _tags else []
                                    _vec = _emb.tolist() if hasattr(_emb, 'tolist') else list(_emb)
                                    if _self._ensure_qdrant_collection(len(_vec)):
                                        _payload = {
                                            "note_id": _nid,
                                            "content": _note.content[:500],
                                            "keywords": _note.keywords,
                                            "tags": _tags,
                                            "category": _note.category or "",
                                            "importance": float(_note.importance),
                                            "created_at": _note.created_at or "",
                                        }
                                        _point_id = _deterministic_point_id(_nid)
                                        _self._qdrant_client.upsert(
                                            collection_name=_self._qdrant_collection,
                                            points=[PointStruct(id=_point_id, vector=_vec, payload=_payload)]
                                        )
                                except Exception as _e:
                                    logger.warning(f"[A-MEM] Re-embed failed for {_nid}: {_e}")
                            _reembedded += 1
                    if _reembedded > 0:
                        with _self._lock:
                            np.savez(_self.embeddings_file, **dict(_self._embeddings.items()))
                    logger.info(f"[A-MEM] Background re-embed complete: {_reembedded}/{len(_ids)} orphaned notes restored")

                threading.Thread(
                    target=_reembed_orphans,
                    args=(self, _orphaned_ids),
                    daemon=True
                ).start()

    def _append_note(self, note: MemoryNote):
        """Append note to JSONL file."""
        rotate_jsonl_if_needed(self.notes_file)
        with open(self.notes_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(note.to_dict()) + '\n')
            f.flush()
            os.fsync(f.fileno())

    def save(self):
        """Save all data to disk (atomic writes to prevent data loss on crash)."""
        with self._lock:
            # Save notes (atomic temp+rename)
            self._atomic_write_jsonl(
                self.notes_file,
                [note.to_dict() for note in self._notes.values()]
            )

            # Save embeddings
            if self._embeddings:
                np.savez(self.embeddings_file, **dict(self._embeddings.items()))

            # Save boxes (atomic temp+rename)
            boxes_data = {k: list(v) for k, v in self._boxes.items()}
            self._atomic_write_json(self.boxes_file, boxes_data)

            # Save stats (atomic temp+rename)
            stats = self.get_stats()
            self._atomic_write_json(self.stats_file, stats)

    def get_recent_notes(self, limit: int = 10) -> List["MemoryNote"]:
        """Return the most recently created notes (by created_at), newest first."""
        with self._lock:
            notes = sorted(
                self._notes.values(),
                key=lambda n: n.created_at or "",
                reverse=True,
            )
            return notes[:limit]

    def get_stats(self) -> Dict[str, Any]:
        """Get memory system statistics."""
        with self._lock:
            categories = {}
            total_links = 0

            for note in self._notes.values():
                categories[note.category] = categories.get(note.category, 0) + 1
                total_links += len(note.links)

            return {
                "total_notes": len(self._notes),
                "total_links": total_links,
                "total_boxes": len(self._boxes),
                "categories": categories,
                "has_embeddings": len(self._embeddings),
                "evolution_enabled": self.evolution_enabled,
                "link_threshold": self.link_threshold
            }

    def consolidate(self) -> Dict[str, Any]:
        """
        Consolidate memory: merge similar notes, prune weak links.

        Run periodically or during "dream" mode.
        """
        with self._lock:
            merged = 0
            pruned_links = 0

            # 1. Prune weak links
            for note in self._notes.values():
                weak_links = [
                    lid for lid, strength in note.link_strengths.items()
                    if strength < 0.2
                ]
                for lid in weak_links:
                    note.links.remove(lid) if lid in note.links else None
                    del note.link_strengths[lid]
                    pruned_links += 1

            # 2. Find near-duplicate notes (high similarity)
            if self._embedder and len(self._embeddings) > 1:
                note_ids = list(self._embeddings.keys())
                for i, id1 in enumerate(note_ids):
                    if id1 not in self._notes:
                        continue
                    emb1 = self._embeddings.get(id1)
                    if emb1 is None:
                        continue

                    for id2 in note_ids[i+1:]:
                        if id2 not in self._notes:
                            continue
                        emb2 = self._embeddings.get(id2)
                        if emb2 is None:
                            continue

                        similarity = float(np.dot(emb1, emb2))
                        if similarity > 0.95:  # Very similar
                            # Merge into the one with higher importance
                            note1 = self._notes[id1]
                            note2 = self._notes[id2]

                            if note1.importance >= note2.importance:
                                self._merge_notes(id1, id2)
                            else:
                                self._merge_notes(id2, id1)
                            merged += 1

            self.save()

            return {
                "merged_notes": merged,
                "pruned_links": pruned_links
            }

    def _merge_notes(self, keeper_id: str, remove_id: str):
        """Merge remove note into keeper note."""
        keeper = self._notes.get(keeper_id)
        remove = self._notes.get(remove_id)

        if not keeper or not remove:
            return

        # Merge attributes
        keeper.keywords = list(set(keeper.keywords + remove.keywords))[:10]
        keeper.tags = list(set(keeper.tags + remove.tags))[:10]
        keeper.access_count += remove.access_count
        keeper.importance = max(keeper.importance, remove.importance)

        # Merge links
        for link_id in remove.links:
            if link_id != keeper_id and link_id not in keeper.links:
                strength = remove.link_strengths.get(link_id, 0.5)
                keeper.add_link(link_id, strength)

        # Redirect backlinks
        for backlink_id in remove.backlinks:
            if backlink_id in self._notes:
                linker = self._notes[backlink_id]
                if remove_id in linker.links:
                    linker.links.remove(remove_id)
                    linker.add_link(keeper_id, linker.link_strengths.get(remove_id, 0.5))
                    del linker.link_strengths[remove_id]

        # Delete merged note
        self.delete(remove_id)


# Singleton instance
_amem_instance: Optional[AMEMSystem] = None
_amem_atexit_registered: bool = False


def _amem_atexit_save():
    """Save A-MEM data on interpreter shutdown to prevent data loss."""
    global _amem_instance
    if _amem_instance is not None:
        try:
            _amem_instance.save()
            logger.info("[A-MEM] atexit: data saved successfully")
        except Exception as e:
            logger.warning(f"[A-MEM] atexit: save failed: {e}")


def get_amem(llm_func: Optional[callable] = None) -> AMEMSystem:
    """Get or create the global A-MEM instance."""
    global _amem_instance, _amem_atexit_registered
    if _amem_instance is None:
        _amem_instance = AMEMSystem(llm_func=llm_func)
        if not _amem_atexit_registered:
            import atexit
            atexit.register(_amem_atexit_save)
            _amem_atexit_registered = True
    return _amem_instance


# Export
__all__ = [
    "AMEMSystem",
    "MemoryNote",
    "get_amem"
]
