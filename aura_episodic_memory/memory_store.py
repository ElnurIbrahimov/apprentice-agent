"""
Qdrant-based Memory Store for AURA Episodic Memory.

Uses Qdrant in embedded mode for zero-server, local-first operation.
Stores episodes as vectors with rich payload metadata for hybrid search.
"""

import hashlib
import json
import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable

from .episode import (
    Episode, EpisodeType, EpisodeQuery, EpisodeSearchResult,
    TemporalContext, EmotionalValence
)

# Check Qdrant availability
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance, VectorParams, PointStruct,
        Filter, FieldCondition, MatchValue, MatchAny, Range,
        FilterSelector, PayloadSchemaType
    )
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    QdrantClient = None

# sentence-transformers is lazy-loaded via aura.tools._shared_models.get_sentence_transformer()

logger = logging.getLogger(__name__)


def _episode_id_to_point_id(episode_id: str) -> int:
    """Convert an episode string ID to a stable 60-bit uint64 for Qdrant."""
    return int(hashlib.md5(episode_id.encode()).hexdigest()[:15], 16)


class EmbeddingModel:
    """Wrapper for sentence-transformers embedding model."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedding model.

        Args:
            model_name: HuggingFace model name for embeddings
        """
        self.model_name = model_name
        self._model = None
        self._dimension = None

    @property
    def model(self):
        """Lazy load model via shared singleton."""
        if self._model is None:
            from aura.tools._shared_models import get_sentence_transformer
            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = get_sentence_transformer()
            self._dimension = self._model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded. Dimension: {self._dimension}")
        return self._model

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        if self._dimension is None:
            _ = self.model  # Force load
        return self._dimension

    def embed(self, text: str) -> List[float]:
        """Generate embedding for text."""
        embedding = self.model.encode(text, convert_to_numpy=True, show_progress_bar=False)
        return embedding.tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return [e.tolist() for e in embeddings]


class EpisodicMemoryStore:
    """
    Qdrant-based storage for episodic memories.

    Uses embedded mode for zero-server operation. Supports:
    - Vector similarity search
    - Metadata filtering
    - Temporal queries
    - Hybrid retrieval
    """

    COLLECTION_NAME = "aura_episodes"

    def __init__(
        self,
        db_path: str,
        embedding_model: Optional[str] = "all-MiniLM-L6-v2",
        custom_embedder: Optional[Callable[[str], List[float]]] = None
    ):
        """
        Initialize episodic memory store.

        Args:
            db_path: Path to Qdrant database directory
            embedding_model: Name of sentence-transformers model
            custom_embedder: Optional custom embedding function
        """
        if not QDRANT_AVAILABLE:
            raise ImportError(
                "qdrant-client is required. Install with: "
                "pip install qdrant-client"
            )

        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)

        # Availability flag — set False on lock/access error so search() can skip
        self._available = True
        self._retry_after: float = 0.0
        self.client = None

        # Thread safety: Qdrant embedded mode (SQLite-backed) is not
        # safe for concurrent thread access. RLock allows reentrant calls
        # so a locked method can invoke another locked method.
        self._lock = threading.RLock()

        # Initialize Qdrant in embedded mode
        try:
            self.client = QdrantClient(path=str(self.db_path))
        except Exception as e:
            err_str = str(e).lower()
            if "lock" in err_str or "already accessed" in err_str or "locked" in err_str:
                logger.warning(f"[EpisodicMemory] Qdrant init failed due to lock/access conflict — store disabled: {e}")
            else:
                logger.warning(f"[EpisodicMemory] Qdrant init failed — store disabled: {e}")
            self._available = False
            self._retry_after = time.monotonic() + 300.0

        # Initialize embedding model
        if custom_embedder:
            self._embed = custom_embedder
            self._embedding_dim = len(custom_embedder("test"))
        else:
            self._embedder = EmbeddingModel(embedding_model)
            self._embed = self._embedder.embed
            self._embedding_dim = self._embedder.dimension

        # Ensure collection exists (only if client initialized successfully)
        if self._available:
            self._ensure_collection()

        # Auto-consolidation timer (fires every 24h)
        self._consolidation_timer: Optional[threading.Timer] = None
        self._schedule_consolidation()

        # Statistics
        self._stats = {
            "total_stored": 0,
            "total_retrieved": 0,
            "total_searches": 0
        }

        if self._available:
            logger.info(f"EpisodicMemoryStore initialized at {db_path}")
        else:
            logger.warning(f"EpisodicMemoryStore at {db_path} is unavailable (lock conflict or init error)")

    def _check_availability(self) -> bool:
        """Attempt reconnect after cooldown if previously unavailable."""
        if self._available:
            return True
        if time.monotonic() < self._retry_after:
            return False
        # Cooldown elapsed — try to reconnect
        try:
            self.client = QdrantClient(path=str(self.db_path))
            self._available = True
            self._ensure_collection()
            logger.info("[EpisodicMemory] Qdrant reconnected successfully")
            return True
        except Exception as e:
            logger.debug(f"[EpisodicMemory] Reconnect failed: {e}")
            self._retry_after = time.monotonic() + 300.0
            return False

    def _ensure_collection(self):
        """Create collection if it doesn't exist."""
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]

        if self.COLLECTION_NAME not in collection_names:
            self.client.create_collection(
                collection_name=self.COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=self._embedding_dim,
                    distance=Distance.COSINE
                )
            )
            logger.info(f"Created collection: {self.COLLECTION_NAME}")

    def store_episode(self, episode: Episode) -> str:
        """
        Store an episode in the database.

        Args:
            episode: Episode to store

        Returns:
            Episode ID
        """
        if not self._check_availability():
            return episode.id
        # Generate embedding if not present
        if episode.embedding is None:
            embed_text = self._create_embed_text(episode)
            episode.embedding = self._embed(embed_text)

        # Create payload from episode
        payload = episode.to_dict()

        # Create point
        point = PointStruct(
            id=_episode_id_to_point_id(episode.id),  # Qdrant needs positive int
            vector=episode.embedding,
            payload={
                **payload,
                "_episode_id": episode.id,  # Store original ID
                "_timestamp_unix": episode.temporal_context.timestamp.timestamp()
            }
        )

        with self._lock:
            # Upsert point
            self.client.upsert(
                collection_name=self.COLLECTION_NAME,
                points=[point]
            )

            self._stats["total_stored"] += 1

        logger.debug(f"Stored episode: {episode.id}")

        return episode.id

    def store_episodes_batch(self, episodes: List[Episode]) -> List[str]:
        """
        Store multiple episodes efficiently.

        Args:
            episodes: List of episodes to store

        Returns:
            List of episode IDs
        """
        if not self._check_availability():
            return [e.id for e in episodes]
        if not episodes:
            return []

        # Generate embeddings in batch
        texts_to_embed = []
        for episode in episodes:
            if episode.embedding is None:
                texts_to_embed.append(self._create_embed_text(episode))

        if texts_to_embed and hasattr(self, '_embedder'):
            embeddings = self._embedder.embed_batch(texts_to_embed)
            embed_idx = 0
            for episode in episodes:
                if episode.embedding is None:
                    episode.embedding = embeddings[embed_idx]
                    embed_idx += 1

        # Create points
        points = []
        for episode in episodes:
            payload = episode.to_dict()
            points.append(PointStruct(
                id=_episode_id_to_point_id(episode.id),
                vector=episode.embedding,
                payload={
                    **payload,
                    "_episode_id": episode.id,
                    "_timestamp_unix": episode.temporal_context.timestamp.timestamp()
                }
            ))

        with self._lock:
            # Upsert batch
            self.client.upsert(
                collection_name=self.COLLECTION_NAME,
                points=points
            )

            self._stats["total_stored"] += len(episodes)

        return [e.id for e in episodes]

    def _create_embed_text(self, episode: Episode) -> str:
        """Create text for embedding from episode."""
        parts = [episode.content]

        if episode.title:
            parts.insert(0, episode.title)

        if episode.entities_involved:
            parts.append(f"Entities: {', '.join(episode.entities_involved)}")

        if episode.tools_used:
            parts.append(f"Tools: {', '.join(episode.tools_used)}")

        return " | ".join(parts)

    def get_episode(self, episode_id: str) -> Optional[Episode]:
        """
        Retrieve episode by ID.

        Args:
            episode_id: Episode ID

        Returns:
            Episode or None if not found
        """
        if not self._available:
            return None
        with self._lock:
            # Search by payload filter
            results = self.client.scroll(
                collection_name=self.COLLECTION_NAME,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="_episode_id",
                            match=MatchValue(value=episode_id)
                        )
                    ]
                ),
                limit=1,
                with_vectors=True
            )

            points, _ = results
            if not points:
                return None

            point = points[0]
            self._stats["total_retrieved"] += 1

            episode = Episode.from_dict(point.payload, embedding=point.vector)
            episode.mark_accessed()

            # Update access metadata in DB
            self._update_episode_access(episode)

            return episode

    def _update_episode_access(self, episode: Episode):
        """Update episode access metadata in database."""
        with self._lock:
            self.client.set_payload(
                collection_name=self.COLLECTION_NAME,
                payload={
                    "access_count": episode.access_count,
                    "last_accessed": episode.last_accessed.isoformat() if episode.last_accessed else None
                },
                points=[_episode_id_to_point_id(episode.id)]
            )

    def search(
        self,
        query: EpisodeQuery,
        scorer: Optional[Callable] = None
    ) -> List[EpisodeSearchResult]:
        """
        Search for episodes matching query.

        Args:
            query: Search query with filters and weights
            scorer: Optional custom scoring function

        Returns:
            List of search results sorted by score
        """
        # Guard: skip search if store is unavailable (lock conflict or init error)
        if not self._check_availability():
            return []

        # Build filter conditions
        filter_conditions = self._build_filter(query)

        # Embed query text outside the lock (CPU-heavy, no Qdrant access)
        query_embedding = None
        if query.query_text:
            query_embedding = self._embed(query.query_text)

        with self._lock:
            self._stats["total_searches"] += 1

            # Perform search
            if query_embedding is not None:
                # Vector similarity search
                query_result = self.client.query_points(
                    collection_name=self.COLLECTION_NAME,
                    query=query_embedding,
                    query_filter=Filter(must=filter_conditions) if filter_conditions else None,
                    limit=query.limit * 2,  # Get extra for re-scoring
                    with_vectors=True
                )
                results = query_result.points
            else:
                # Scroll with filter only
                results, _ = self.client.scroll(
                    collection_name=self.COLLECTION_NAME,
                    scroll_filter=Filter(must=filter_conditions) if filter_conditions else None,
                    limit=query.limit * 2,
                    with_vectors=True
                )

        # Convert to episodes and score (no Qdrant access needed)
        search_results = []
        for point in results:
            episode = Episode.from_dict(
                point.payload,
                embedding=point.vector if hasattr(point, 'vector') else None
            )

            # Calculate score
            if scorer:
                score, breakdown = scorer(episode, query)
            else:
                score, breakdown = self._default_score(episode, query, point)

            if score >= query.min_score:
                search_results.append(EpisodeSearchResult(
                    episode=episode,
                    score=score,
                    score_breakdown=breakdown
                ))

        # Sort by score and limit
        search_results.sort(reverse=True)
        return search_results[:query.limit]

    def _build_filter(self, query: EpisodeQuery) -> List[FieldCondition]:
        """Build Qdrant filter from query."""
        conditions = []

        # Temporal filters
        if query.start_time:
            conditions.append(FieldCondition(
                key="_timestamp_unix",
                range=Range(gte=query.start_time.timestamp())
            ))

        if query.end_time:
            conditions.append(FieldCondition(
                key="_timestamp_unix",
                range=Range(lte=query.end_time.timestamp())
            ))

        if query.time_of_day:
            conditions.append(FieldCondition(
                key="temporal_context.time_of_day",
                match=MatchValue(value=query.time_of_day)
            ))

        if query.day_of_week:
            conditions.append(FieldCondition(
                key="temporal_context.day_of_week",
                match=MatchValue(value=query.day_of_week)
            ))

        # Type filter — use MatchAny so that multiple types are OR-combined,
        # not AND-combined (a single episode can only have one type).
        if query.episode_types:
            type_values = [t.value for t in query.episode_types]
            conditions.append(FieldCondition(
                key="episode_type",
                match=MatchAny(any=type_values)
            ))

        # Emotional valence filter
        if query.emotional_valence:
            conditions.append(FieldCondition(
                key="emotional_valence",
                match=MatchValue(value=query.emotional_valence.value)
            ))

        return conditions

    def _default_score(
        self,
        episode: Episode,
        query: EpisodeQuery,
        point: Any
    ) -> tuple[float, Dict[str, float]]:
        """Calculate default score for episode."""
        breakdown = {}

        # Recency score
        recency = episode.get_recency_score()
        breakdown["recency"] = recency

        # Importance score
        importance = episode.importance
        breakdown["importance"] = importance

        # Relevance score (from vector similarity if available)
        if hasattr(point, 'score'):
            relevance = point.score
        else:
            relevance = 0.5
        breakdown["relevance"] = relevance

        # Emotional congruence scoring
        emotional_congruence = 0.5
        if query.emotional_pad and query.emotional_weight > 0:
            valence_map = {
                EmotionalValence.POSITIVE: 1.0,
                EmotionalValence.NEGATIVE: -1.0,
                EmotionalValence.NEUTRAL: 0.0,
                EmotionalValence.MIXED: 0.0,
            }
            ep_valence = valence_map.get(episode.emotional_valence, 0.0)
            mood_pleasure = query.emotional_pad.get("pleasure", 0.0)
            # Congruence: same-sign valence and mood = boost
            emotional_congruence = 0.5 + 0.5 * (ep_valence * mood_pleasure)
            emotional_congruence = max(0.0, min(1.0, emotional_congruence))
        breakdown["emotional_congruence"] = emotional_congruence

        # Weighted combination
        ew = query.emotional_weight
        base_scale = 1.0 - ew
        score = (
            base_scale * (
                query.recency_weight * recency +
                query.importance_weight * importance +
                query.relevance_weight * relevance
            ) + ew * emotional_congruence
        )

        return score, breakdown

    def get_timeline(
        self,
        start_time: datetime,
        end_time: datetime,
        episode_types: Optional[List[EpisodeType]] = None,
        limit: int = 100
    ) -> List[Episode]:
        """
        Get episodes in chronological order within time range.

        Args:
            start_time: Start of time range
            end_time: End of time range
            episode_types: Optional type filter
            limit: Maximum episodes to return

        Returns:
            List of episodes sorted by timestamp
        """
        if not self._check_availability():
            return []
        conditions = [
            FieldCondition(
                key="_timestamp_unix",
                range=Range(
                    gte=start_time.timestamp(),
                    lte=end_time.timestamp()
                )
            )
        ]

        if episode_types:
            conditions.append(FieldCondition(
                key="episode_type",
                match=MatchAny(any=[etype.value for etype in episode_types])
            ))

        with self._lock:
            results, _ = self.client.scroll(
                collection_name=self.COLLECTION_NAME,
                scroll_filter=Filter(must=conditions),
                limit=limit
            )

        episodes = [Episode.from_dict(p.payload) for p in results]
        episodes.sort(key=lambda e: e.temporal_context.timestamp)

        return episodes

    def delete_episode(self, episode_id: str) -> bool:
        """
        Delete an episode.

        Args:
            episode_id: Episode ID to delete

        Returns:
            True if deleted, False if not found
        """
        if not self._check_availability():
            return False
        with self._lock:
            try:
                self.client.delete(
                    collection_name=self.COLLECTION_NAME,
                    points_selector=FilterSelector(
                        filter=Filter(
                            must=[FieldCondition(
                                key="_episode_id",
                                match=MatchValue(value=episode_id)
                            )]
                        )
                    )
                )
                return True
            except Exception as e:
                logger.error(f"Failed to delete episode {episode_id}: {e}")
                return False

    def get_statistics(self) -> Dict[str, Any]:
        """Get memory store statistics."""
        if not self._check_availability():
            return {"total_episodes": 0, "available": False, **self._stats}
        with self._lock:
            collection_info = self.client.get_collection(self.COLLECTION_NAME)

            return {
                "total_episodes": collection_info.points_count,
                "vector_dimension": self._embedding_dim,
                "collection_name": self.COLLECTION_NAME,
                **self._stats
            }

    def _schedule_consolidation(self) -> None:
        """Schedule the next consolidation run in 24 hours."""
        self._consolidation_timer = threading.Timer(86400, self._run_scheduled_consolidation)
        self._consolidation_timer.daemon = True
        self._consolidation_timer.start()

    def _run_scheduled_consolidation(self) -> None:
        """Run consolidation and reschedule."""
        try:
            from aura_episodic_memory.consolidation import MemoryConsolidator
            consolidator = MemoryConsolidator(self)
            consolidator.consolidate()
        except Exception as e:
            logger.warning(f"[EpisodicMemory] Scheduled consolidation failed: {e}")
        finally:
            self._schedule_consolidation()

    def close(self):
        """Close the database connection."""
        if self._consolidation_timer is not None:
            self._consolidation_timer.cancel()
            self._consolidation_timer = None
        with self._lock:
            if self.client:
                self.client.close()
                logger.info("EpisodicMemoryStore closed")
