"""Unified Memory Store — Phase 2 Memory Consolidation.

Single SQLite database with FTS5 full-text search and BLOB vector embeddings.
Replaces MemorySystem, A-MEM, Episodic, and RAG backends with one store.

Tables:
  memories     — all memory records (conversation, insight, fact, etc.)
  memories_fts — FTS5 virtual table auto-synced via triggers
  user_profile — JSON-serialized user model

Author: Aura Development Team
Created: 2026-03-16
"""

import atexit
import json
import logging
import math
import re
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]

# Import faiss-cpu (not faiss-gpu — GPU variant can hang on CUDA init).
# Falls back to numpy brute-force if neither is available.
faiss = None  # type: ignore[assignment]
_FAISS_AVAILABLE = False
try:
    import faiss as _faiss_mod
    # Validate with a no-op call to catch broken installs early
    _dummy = _faiss_mod.IndexFlatL2(1)
    faiss = _faiss_mod
    _FAISS_AVAILABLE = True
except Exception:
    pass

logger = logging.getLogger(__name__)

# Default DB path (overridable via Config)
_DEFAULT_DB_PATH = Path("data/aura_memory.db")


def _float32_to_blob(vec: np.ndarray) -> bytes:
    """Encode a numpy float32 array as a compact binary BLOB."""
    return vec.astype(np.float32).tobytes()


def _blob_to_float32(blob: bytes) -> np.ndarray:
    """Decode a binary BLOB back to a numpy float32 array."""
    if len(blob) % 4 != 0:
        raise ValueError(f"Embedding blob size {len(blob)} is not a multiple of 4 bytes")
    return np.frombuffer(blob, dtype=np.float32).copy()


@dataclass
class MemoryRecord:
    """A single memory record in the unified store."""
    id: str = ""
    content: str = ""
    title: str = ""
    source: str = "conversation"         # conversation, task_execution, learning, insight, dream, rag_chunk
    memory_type: str = "episodic"        # episodic, semantic, procedural, fact
    importance: float = 0.5
    # Memory record fields
    keywords: str = ""                   # comma-separated
    tags: str = ""                       # comma-separated
    category: str = ""
    boxes: str = ""                      # JSON list of linked note IDs
    links: str = ""                      # JSON list of related IDs
    # Episodic fields
    temporal_context: str = ""           # JSON: {timestamp, duration, ...}
    emotional_valence: str = "neutral"   # positive, negative, neutral, mixed
    emotional_pad: str = ""              # JSON: {pleasure, arousal, dominance}
    # FadeMem fields
    strength: float = 1.0
    decay_rate: float = 0.00206          # ~2 week half-life
    access_count: int = 0
    last_accessed: str = ""
    # Lifecycle
    lifecycle_state: str = "candidate"   # candidate, stable, summary, archived, forgotten
    user_id: str = "default_user"
    # Timestamps
    created_at: str = ""
    updated_at: str = ""
    # Embedding stored as BLOB (not in dataclass — handled separately)
    metadata: str = "{}"                 # JSON blob for anything else

    def __post_init__(self):
        if not self.id:
            self.id = uuid.uuid4().hex
        now = datetime.now().isoformat()
        if not self.created_at:
            self.created_at = now
        if not self.updated_at:
            self.updated_at = now
        if not self.last_accessed:
            self.last_accessed = now


# Pagination bounds for the FAISS-less brute-force semantic fallback.
# Peak memory during a scan is bounded by _BRUTE_FORCE_CHUNK_SIZE rows +
# the top-k heap, regardless of total table size — avoids loading a 100K+
# row table into RAM on FAISS-less installs.
_BRUTE_FORCE_CHUNK_SIZE = 1000
_BRUTE_FORCE_WARN_ROWS = 50_000


# Column order for INSERT (must match _CREATE_TABLE)
_COLUMNS = [
    "id", "content", "title", "source", "memory_type", "importance",
    "keywords", "tags", "category", "boxes", "links",
    "temporal_context", "emotional_valence", "emotional_pad",
    "strength", "decay_rate", "access_count", "last_accessed",
    "lifecycle_state", "user_id", "created_at", "updated_at",
    "metadata", "embedding",
]

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS memories (
    id                TEXT PRIMARY KEY,
    content           TEXT NOT NULL,
    title             TEXT NOT NULL DEFAULT '',
    source            TEXT NOT NULL DEFAULT 'conversation',
    memory_type       TEXT NOT NULL DEFAULT 'episodic',
    importance        REAL NOT NULL DEFAULT 0.5,
    keywords          TEXT NOT NULL DEFAULT '',
    tags              TEXT NOT NULL DEFAULT '',
    category          TEXT NOT NULL DEFAULT '',
    boxes             TEXT NOT NULL DEFAULT '[]',
    links             TEXT NOT NULL DEFAULT '[]',
    temporal_context  TEXT NOT NULL DEFAULT '',
    emotional_valence TEXT NOT NULL DEFAULT 'neutral',
    emotional_pad     TEXT NOT NULL DEFAULT '',
    strength          REAL NOT NULL DEFAULT 1.0,
    decay_rate        REAL NOT NULL DEFAULT 0.00206,
    access_count      INTEGER NOT NULL DEFAULT 0,
    last_accessed     TEXT NOT NULL DEFAULT '',
    lifecycle_state   TEXT NOT NULL DEFAULT 'candidate',
    user_id           TEXT NOT NULL DEFAULT 'default_user',
    created_at        TEXT NOT NULL,
    updated_at        TEXT NOT NULL,
    metadata          TEXT NOT NULL DEFAULT '{}',
    embedding         BLOB DEFAULT NULL
)
"""

_CREATE_FTS = """
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    content, title, keywords, tags, category,
    content='memories',
    content_rowid='rowid'
)
"""

_CREATE_TRIGGERS = [
    # After INSERT
    """CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
        INSERT INTO memories_fts(rowid, content, title, keywords, tags, category)
        VALUES (new.rowid, new.content, new.title, new.keywords, new.tags, new.category);
    END""",
    # After DELETE
    """CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
        INSERT INTO memories_fts(memories_fts, rowid, content, title, keywords, tags, category)
        VALUES ('delete', old.rowid, old.content, old.title, old.keywords, old.tags, old.category);
    END""",
    # After UPDATE
    """CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
        INSERT INTO memories_fts(memories_fts, rowid, content, title, keywords, tags, category)
        VALUES ('delete', old.rowid, old.content, old.title, old.keywords, old.tags, old.category);
        INSERT INTO memories_fts(rowid, content, title, keywords, tags, category)
        VALUES (new.rowid, new.content, new.title, new.keywords, new.tags, new.category);
    END""",
]

_CREATE_USER_PROFILE = """
CREATE TABLE IF NOT EXISTS user_profile (
    user_id    TEXT PRIMARY KEY,
    profile    TEXT NOT NULL DEFAULT '{}',
    updated_at TEXT NOT NULL
)
"""

_CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_memories_source ON memories(source)",
    "CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type)",
    "CREATE INDEX IF NOT EXISTS idx_memories_lifecycle ON memories(lifecycle_state)",
    "CREATE INDEX IF NOT EXISTS idx_memories_user ON memories(user_id)",
    "CREATE INDEX IF NOT EXISTS idx_memories_strength ON memories(strength)",
    "CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at)",
    "CREATE INDEX IF NOT EXISTS idx_memories_accessed ON memories(last_accessed)",
]


class MemoryStore:
    """Unified SQLite + FTS5 memory store.

    Thread-safe. Uses WAL mode for concurrent read/write.
    Embedding BLOBs stored as raw float32 arrays for fast numpy interop.
    """

    def __init__(self, db_path: Optional[str] = None):
        if db_path:
            self._db_path = Path(db_path)
        else:
            try:
                from aura.config import Config
                self._db_path = Path(getattr(Config, "AURA_MEMORY_DB_PATH", "") or _DEFAULT_DB_PATH)
            except (ImportError, AttributeError) as e:
                logger.debug(f"[MemoryStore] Config import failed, using default path: {e}")
                self._db_path = _DEFAULT_DB_PATH

        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()  # RLock to allow reentrant access (e.g. reinforce -> get)
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()
        atexit.register(self.close)

        # FAISS vector index (lazy-built on first search_semantic call)
        self._faiss_index: Optional[Any] = None  # faiss.IndexIDMap
        self._faiss_id_map: Dict[int, str] = {}  # faiss int64 id → memory_id string
        self._faiss_str_map: Dict[str, int] = {}  # memory_id string → faiss int64 id
        self._faiss_next_id: int = 0
        self._faiss_built = False
        self._faiss_lock = threading.Lock()  # Separate lock for FAISS index operations

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA cache_size=-8000")  # 8MB cache
        return self._conn

    def _init_db(self) -> None:
        with self._lock:
            conn = self._get_conn()
            conn.execute(_CREATE_TABLE)
            conn.execute(_CREATE_FTS)
            for trigger_sql in _CREATE_TRIGGERS:
                conn.execute(trigger_sql)
            conn.execute(_CREATE_USER_PROFILE)
            for idx_sql in _CREATE_INDEXES:
                conn.execute(idx_sql)
            conn.commit()

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def insert(
        self,
        record: MemoryRecord,
        embedding: Optional[np.ndarray] = None,
    ) -> str:
        """Insert a new memory record. Returns the record ID."""
        record.updated_at = datetime.now().isoformat()
        emb_blob = _float32_to_blob(embedding) if embedding is not None else None

        values = [
            record.id, record.content, record.title, record.source,
            record.memory_type, record.importance,
            record.keywords, record.tags, record.category,
            record.boxes, record.links,
            record.temporal_context, record.emotional_valence, record.emotional_pad,
            record.strength, record.decay_rate, record.access_count, record.last_accessed,
            record.lifecycle_state, record.user_id,
            record.created_at, record.updated_at,
            record.metadata, emb_blob,
        ]
        placeholders = ",".join(["?"] * len(_COLUMNS))
        sql = f"INSERT OR IGNORE INTO memories({','.join(_COLUMNS)}) VALUES({placeholders})"

        with self._lock:
            conn = self._get_conn()
            try:
                cur = conn.execute(sql, values)
                conn.commit()
                if cur.rowcount == 0:
                    logger.warning(f"[MemoryStore] INSERT OR IGNORE skipped for id={record.id} (duplicate)")
                    return record.id
            except Exception:
                conn.rollback()
                raise
        # Sync FAISS index only if the row was actually inserted
        if embedding is not None:
            self._faiss_add(record.id, embedding)
        return record.id

    def get(self, memory_id: str) -> Optional[MemoryRecord]:
        """Get a single memory by ID."""
        sql = f"SELECT {','.join(_COLUMNS)} FROM memories WHERE id=?"
        with self._lock:
            row = self._get_conn().execute(sql, (memory_id,)).fetchone()
        if not row:
            return None
        return self._row_to_record(row)

    def get_embedding(self, memory_id: str) -> Optional[np.ndarray]:
        """Get the embedding vector for a memory."""
        with self._lock:
            row = self._get_conn().execute(
                "SELECT embedding FROM memories WHERE id=?", (memory_id,)
            ).fetchone()
        if row and row[0]:
            return _blob_to_float32(row[0])
        return None

    # Fields allowed in update() — prevents SQL injection via kwargs keys
    _UPDATABLE_FIELDS = frozenset({
        "content", "title", "source", "memory_type", "importance",
        "keywords", "tags", "category", "boxes", "links",
        "temporal_context", "emotional_valence", "emotional_pad",
        "strength", "decay_rate", "access_count", "last_accessed",
        "lifecycle_state", "user_id", "metadata", "updated_at",
    })

    def update(self, memory_id: str, **kwargs) -> bool:
        """Update specific fields on a memory record."""
        if not kwargs:
            return False
        bad_fields = set(kwargs) - self._UPDATABLE_FIELDS
        if bad_fields:
            raise ValueError(f"Invalid update fields: {bad_fields}")
        kwargs["updated_at"] = datetime.now().isoformat()
        sets = ", ".join(f"{k}=?" for k in kwargs)
        vals = [*list(kwargs.values()), memory_id]
        sql = f"UPDATE memories SET {sets} WHERE id=?"
        with self._lock:
            conn = self._get_conn()
            try:
                cur = conn.execute(sql, vals)
                conn.commit()
            except Exception:
                conn.rollback()
                raise
        return cur.rowcount > 0

    def update_embedding(self, memory_id: str, embedding: np.ndarray) -> bool:
        """Update just the embedding BLOB for a memory."""
        blob = _float32_to_blob(embedding)
        with self._lock:
            conn = self._get_conn()
            try:
                cur = conn.execute(
                    "UPDATE memories SET embedding=?, updated_at=? WHERE id=?",
                    (blob, datetime.now().isoformat(), memory_id),
                )
                conn.commit()
            except Exception:
                conn.rollback()
                raise
        if cur.rowcount > 0:
            self._faiss_add(memory_id, embedding)
        return cur.rowcount > 0

    def delete(self, memory_id: str) -> bool:
        """Hard-delete a memory."""
        with self._lock:
            conn = self._get_conn()
            try:
                cur = conn.execute("DELETE FROM memories WHERE id=?", (memory_id,))
                conn.commit()
            except Exception:
                conn.rollback()
                raise
        if cur.rowcount > 0:
            self._faiss_remove(memory_id)
        return cur.rowcount > 0

    def touch(self, memory_id: str) -> None:
        """Update last_accessed and increment access_count."""
        now = datetime.now().isoformat()
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    "UPDATE memories SET last_accessed=?, access_count=access_count+1, updated_at=? WHERE id=?",
                    (now, now, memory_id),
                )
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    # Whitelist for safe list_paginated ordering — prevents SQL injection via order_by.
    _ORDER_BY_WHITELIST = frozenset({
        "created_at DESC", "created_at ASC",
        "updated_at DESC", "updated_at ASC",
        "importance DESC", "importance ASC",
        "strength DESC", "strength ASC",
        "access_count DESC", "access_count ASC",
    })

    def list_paginated(
        self,
        offset: int = 0,
        limit: int = 50,
        source_filter: Optional[str] = None,
        order_by: str = "created_at DESC",
    ) -> list[MemoryRecord]:
        """Return a paginated slice of memories, newest first by default.

        Used by the Memory Browser Mini App tab to render a timeline without
        needing a search query. Whitelists ``order_by`` to block SQL injection.
        """
        if order_by not in self._ORDER_BY_WHITELIST:
            order_by = "created_at DESC"
        limit = max(1, min(500, int(limit)))
        offset = max(0, int(offset))

        col_list = ",".join(_COLUMNS)
        if source_filter:
            sql = (
                f"SELECT {col_list} FROM memories WHERE source = ? "
                f"ORDER BY {order_by} LIMIT ? OFFSET ?"
            )
            params: tuple = (source_filter, limit, offset)
        else:
            sql = (
                f"SELECT {col_list} FROM memories "
                f"ORDER BY {order_by} LIMIT ? OFFSET ?"
            )
            params = (limit, offset)

        with self._lock:
            rows = self._get_conn().execute(sql, params).fetchall()
        return [self._row_to_record(r) for r in rows]

    def count_by_source(self, source_filter: Optional[str] = None) -> int:
        """Return the total number of memories, optionally filtered by source.

        Distinct from ``count(lifecycle_states, user_id)`` below, which filters
        by lifecycle and user instead of source.
        """
        with self._lock:
            if source_filter:
                row = self._get_conn().execute(
                    "SELECT COUNT(*) FROM memories WHERE source = ?",
                    (source_filter,),
                ).fetchone()
            else:
                row = self._get_conn().execute(
                    "SELECT COUNT(*) FROM memories"
                ).fetchone()
        return int(row[0]) if row else 0

    def list_sources(self) -> list[str]:
        """Return distinct source values in the store."""
        with self._lock:
            rows = self._get_conn().execute(
                "SELECT DISTINCT source FROM memories ORDER BY source"
            ).fetchall()
        return [r[0] for r in rows if r[0]]

    # ------------------------------------------------------------------
    # FAISS vector index
    # ------------------------------------------------------------------

    @property
    def _faiss_checkpoint_path(self) -> Path:
        return self._db_path.parent / "faiss.index"

    @property
    def _faiss_checkpoint_meta_path(self) -> Path:
        return self._db_path.parent / "faiss.index.meta.json"

    def _row_count_with_embedding(self) -> int:
        """Count memories that have an embedding — used as a checkpoint freshness key."""
        with self._lock:
            row = self._get_conn().execute(
                "SELECT COUNT(*) FROM memories WHERE embedding IS NOT NULL"
            ).fetchone()
        return int(row[0]) if row else 0

    def _save_faiss_checkpoint(self) -> None:
        """Persist the in-memory FAISS index so the next startup can skip the
        O(N) DB scan. Safe to call after a successful build — failures here
        are logged but not raised (the in-memory index is authoritative)."""
        if self._faiss_index is None:
            return
        try:
            from aura.paths import atomic_write_json
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self._faiss_index, str(self._faiss_checkpoint_path))
            meta = {
                "row_count": self._row_count_with_embedding(),
                "dim": int(self._faiss_index.d),
                "next_id": self._faiss_next_id,
                # id maps are required to rebuild the str<->int64 mapping
                "id_map": {str(k): v for k, v in self._faiss_id_map.items()},
            }
            atomic_write_json(self._faiss_checkpoint_meta_path, meta, indent=0)
            logger.debug("[MemoryStore] FAISS checkpoint saved (%d vectors)", meta["row_count"])
        except Exception as e:
            logger.debug(f"[MemoryStore] FAISS checkpoint save failed (non-fatal): {e}")

    def _load_faiss_checkpoint(self) -> bool:
        """Try to restore the FAISS index from disk. Returns True on success.

        The checkpoint is accepted only if the embedded-row count matches the
        current DB; otherwise the caller must rebuild to pick up
        insertions/deletions that happened while the process was down.
        """
        if not _FAISS_AVAILABLE or np is None:
            return False
        if not (self._faiss_checkpoint_path.exists() and self._faiss_checkpoint_meta_path.exists()):
            return False
        try:
            meta = json.loads(self._faiss_checkpoint_meta_path.read_text(encoding="utf-8"))
            expected_rows = int(meta.get("row_count", -1))
            actual_rows = self._row_count_with_embedding()
            if expected_rows != actual_rows:
                logger.info(
                    "[MemoryStore] FAISS checkpoint stale (%d saved vs %d current) — rebuilding",
                    expected_rows, actual_rows,
                )
                return False
            index = faiss.read_index(str(self._faiss_checkpoint_path))
            id_map = {int(k): v for k, v in meta.get("id_map", {}).items()}
            str_map = {v: k for k, v in id_map.items()}
            with self._faiss_lock:
                self._faiss_index = index
                self._faiss_id_map = id_map
                self._faiss_str_map = str_map
                self._faiss_next_id = int(meta.get("next_id", max(id_map.keys(), default=-1) + 1))
                self._faiss_built = True
            logger.info(
                "[MemoryStore] FAISS checkpoint loaded: %d vectors, dim=%d",
                actual_rows, int(meta.get("dim", 0)),
            )
            return True
        except Exception as e:
            logger.warning(f"[MemoryStore] FAISS checkpoint load failed, will rebuild: {e}")
            return False

    def _build_faiss_index(self) -> bool:
        """Build (or rebuild) the FAISS index from all embeddings in the DB.

        Tries to load a persisted checkpoint first (avoids the O(N) DB scan on
        every restart). Returns True if FAISS is available and the index was built.
        """
        if not _FAISS_AVAILABLE or np is None:
            return False

        # Fast path: restore from disk if fresh.
        if self._load_faiss_checkpoint():
            return True

        emb_idx = _COLUMNS.index("embedding")
        id_idx = _COLUMNS.index("id")

        with self._lock:
            rows = self._get_conn().execute(
                f"SELECT {','.join(_COLUMNS)} FROM memories WHERE embedding IS NOT NULL"
            ).fetchall()

        if not rows:
            # Empty index — detect dimension from first future insert
            with self._faiss_lock:
                self._faiss_built = True
            return True

        # Detect dimension from first row
        first_vec = _blob_to_float32(rows[0][emb_idx])
        dim = first_vec.shape[0]

        # IndexFlatIP = inner product on L2-normalized vectors = cosine similarity
        base_index = faiss.IndexFlatIP(dim)
        index = faiss.IndexIDMap(base_index)

        vectors = []
        ids = []
        id_map: Dict[int, str] = {}
        str_map: Dict[str, int] = {}
        next_id = 0

        for row in rows:
            blob = row[emb_idx]
            if not blob:
                continue
            vec = _blob_to_float32(blob)
            if vec.shape[0] != dim:
                continue
            norm = np.linalg.norm(vec)
            if norm < 1e-8:
                continue
            vec = vec / norm  # L2-normalize for cosine via IP

            mem_id = row[id_idx]
            faiss_id = next_id
            next_id += 1
            id_map[faiss_id] = mem_id
            str_map[mem_id] = faiss_id

            vectors.append(vec)
            ids.append(faiss_id)

        if vectors:
            mat = np.array(vectors, dtype=np.float32)
            index.add_with_ids(mat, np.array(ids, dtype=np.int64))

        with self._faiss_lock:
            self._faiss_index = index
            self._faiss_id_map = id_map
            self._faiss_str_map = str_map
            self._faiss_next_id = next_id
            self._faiss_built = True
        logger.info(f"[MemoryStore] FAISS index built: {len(vectors)} vectors, dim={dim}")
        # Persist so the next startup skips this O(N) scan. Failures are logged
        # and swallowed inside _save_faiss_checkpoint; don't leak them here.
        self._save_faiss_checkpoint()
        return True

    def _faiss_add(self, memory_id: str, embedding: np.ndarray) -> None:
        """Add or update a single vector in the FAISS index."""
        if not self._faiss_built or self._faiss_index is None:
            return
        with self._faiss_lock:
            # Remove old entry if updating
            if memory_id in self._faiss_str_map:
                self._faiss_remove_unlocked(memory_id)

            vec = embedding.astype(np.float32)
            norm = np.linalg.norm(vec)
            if norm < 1e-8:
                return
            vec = (vec / norm).reshape(1, -1)

            faiss_id = self._faiss_next_id
            self._faiss_next_id += 1
            self._faiss_id_map[faiss_id] = memory_id
            self._faiss_str_map[memory_id] = faiss_id
            self._faiss_index.add_with_ids(vec, np.array([faiss_id], dtype=np.int64))

    def _faiss_remove(self, memory_id: str) -> None:
        """Remove a vector from the FAISS index (acquires _faiss_lock)."""
        with self._faiss_lock:
            self._faiss_remove_unlocked(memory_id)

    def _faiss_remove_unlocked(self, memory_id: str) -> None:
        """Remove a vector from the FAISS index (caller must hold _faiss_lock)."""
        if not self._faiss_built or self._faiss_index is None:
            return
        faiss_id = self._faiss_str_map.pop(memory_id, None)
        if faiss_id is not None:
            self._faiss_id_map.pop(faiss_id, None)
            self._faiss_index.remove_ids(np.array([faiss_id], dtype=np.int64))

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search_semantic(
        self,
        query_embedding: np.ndarray,
        k: int = 20,
        lifecycle_states: Optional[List[str]] = None,
        user_id: Optional[str] = None,
    ) -> List[Tuple[MemoryRecord, float]]:
        """Cosine similarity search using FAISS (with brute-force fallback).

        Returns list of (record, similarity_score) sorted descending.
        Only considers memories with non-NULL embeddings.
        """
        # Try FAISS path first
        if _FAISS_AVAILABLE:
            if not self._faiss_built:
                self._build_faiss_index()
            if self._faiss_index is not None and self._faiss_index.ntotal > 0:
                return self._search_semantic_faiss(query_embedding, k, lifecycle_states, user_id)

        # Fallback: brute-force linear scan (O(N) — log warning for monitoring)
        logger.warning("[MemoryStore] FAISS unavailable or empty — using O(N) brute-force semantic search")
        return self._search_semantic_brute(query_embedding, k, lifecycle_states, user_id)

    def _search_semantic_faiss(
        self,
        query_embedding: np.ndarray,
        k: int,
        lifecycle_states: Optional[List[str]],
        user_id: Optional[str],
    ) -> List[Tuple[MemoryRecord, float]]:
        """FAISS-accelerated cosine search with post-filtering."""
        q = query_embedding.astype(np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm < 1e-8:
            return []
        q_unit = (q / q_norm).reshape(1, -1)

        states = set(lifecycle_states or ["candidate", "stable", "summary"])

        with self._faiss_lock:
            # Over-fetch to account for post-filtering (3x k, min 50)
            fetch_k = min(max(k * 3, 50), self._faiss_index.ntotal)
            scores, ids = self._faiss_index.search(q_unit, fetch_k)
            # Snapshot the id map under the lock
            id_map_snapshot = dict(self._faiss_id_map)

        results: List[Tuple[MemoryRecord, float]] = []
        for i in range(ids.shape[1]):
            faiss_id = int(ids[0][i])
            if faiss_id < 0:  # FAISS returns -1 for missing
                continue
            sim = float(scores[0][i])
            mem_id = id_map_snapshot.get(faiss_id)
            if not mem_id:
                continue
            record = self.get(mem_id)
            if not record:
                continue
            if record.lifecycle_state not in states:
                continue
            if user_id and record.user_id != user_id:
                continue
            results.append((record, sim))
            if len(results) >= k:
                break

        return results

    def _search_semantic_brute(
        self,
        query_embedding: np.ndarray,
        k: int,
        lifecycle_states: Optional[List[str]],
        user_id: Optional[str],
    ) -> List[Tuple[MemoryRecord, float]]:
        """Brute-force linear scan fallback (no FAISS).

        Paginated: pulls _BRUTE_FORCE_CHUNK_SIZE rows at a time, maintains a
        size-k min-heap of top similarities, discards chunk rows between
        iterations. Memory is bounded regardless of table size.
        """
        import heapq

        q = query_embedding.astype(np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm < 1e-8:
            return []
        q_unit = q / q_norm

        states = lifecycle_states or ["candidate", "stable", "summary"]
        state_placeholders = ",".join(["?"] * len(states))
        sql = f"""SELECT {','.join(_COLUMNS)} FROM memories
                  WHERE embedding IS NOT NULL
                  AND lifecycle_state IN ({state_placeholders})"""
        params_base: list = list(states)
        if user_id:
            sql += " AND user_id=?"
            params_base.append(user_id)
        sql += " ORDER BY rowid LIMIT ? OFFSET ?"

        emb_idx = _COLUMNS.index("embedding")
        # Min-heap of (similarity, tiebreak, record); pop min when full.
        heap: list[tuple[float, int, MemoryRecord]] = []
        tiebreak = 0
        offset = 0
        total = 0

        while True:
            with self._lock:
                rows = self._get_conn().execute(
                    sql, [*params_base, _BRUTE_FORCE_CHUNK_SIZE, offset]
                ).fetchall()
            if not rows:
                break

            for row in rows:
                emb_blob = row[emb_idx]
                if not emb_blob:
                    continue
                vec = _blob_to_float32(emb_blob)
                if vec.shape != q.shape:
                    continue
                vec_norm = np.linalg.norm(vec)
                if vec_norm < 1e-8:
                    continue
                sim = float(np.dot(q_unit, vec / vec_norm))
                tiebreak += 1
                if len(heap) < k:
                    heapq.heappush(heap, (sim, tiebreak, self._row_to_record(row)))
                elif sim > heap[0][0]:
                    heapq.heapreplace(heap, (sim, tiebreak, self._row_to_record(row)))

            total += len(rows)
            offset += _BRUTE_FORCE_CHUNK_SIZE
            if len(rows) < _BRUTE_FORCE_CHUNK_SIZE:
                break
            if total == _BRUTE_FORCE_WARN_ROWS:
                logger.warning(
                    "[MemoryStore] brute-force semantic scan crossed %d rows — "
                    "install faiss-cpu for scalable search",
                    total,
                )

        # Heap holds top-k by similarity; sort descending for output.
        return sorted(
            [(rec, sim) for (sim, _, rec) in heap],
            key=lambda x: x[1],
            reverse=True,
        )

    def search_bm25(
        self,
        query: str,
        k: int = 20,
        lifecycle_states: Optional[List[str]] = None,
        user_id: Optional[str] = None,
    ) -> List[Tuple[MemoryRecord, float]]:
        """FTS5 BM25 keyword search.

        Returns list of (record, bm25_score) sorted by relevance.
        BM25 scores are negative (more negative = more relevant), so we negate them.
        """
        states = lifecycle_states or ["candidate", "stable", "summary"]
        state_placeholders = ",".join(["?"] * len(states))

        # FTS5 match query — strip special FTS5 characters from each word
        words = []
        for w in query.split():
            fts_safe = re.sub(r'[*"(){}^\-:.+~\\]', '', w.strip()).lower()
            if fts_safe and fts_safe not in ("not", "and", "or"):
                words.append(fts_safe)
        if not words:
            return []
        match_expr = " OR ".join(f'"{w}"' for w in words[:10])  # Cap at 10 terms

        col_list = ", ".join(f"m.{c}" for c in _COLUMNS)
        sql = f"""SELECT {col_list}, bm25(memories_fts) as rank
                  FROM memories m
                  JOIN memories_fts ON memories_fts.rowid = m.rowid
                  WHERE memories_fts MATCH ?
                  AND m.lifecycle_state IN ({state_placeholders})"""
        params: list = [match_expr, *list(states)]
        if user_id:
            sql += " AND m.user_id=?"
            params.append(user_id)
        sql += " ORDER BY rank LIMIT ?"
        params.append(k)

        with self._lock:
            try:
                rows = self._get_conn().execute(sql, params).fetchall()
            except sqlite3.OperationalError as e:
                logger.debug(f"[MemoryStore] FTS5 query error: {e}")
                return []

        results: List[Tuple[MemoryRecord, float]] = []
        for row in rows:
            # row has all memory columns + rank at the end
            rank = row[-1]  # bm25 score (negative, more negative = better)
            record = self._row_to_record(row[:-1])  # exclude rank column
            score = -rank  # flip to positive (higher = better)
            results.append((record, score))
        return results

    def count(
        self,
        lifecycle_states: Optional[List[str]] = None,
        user_id: Optional[str] = None,
    ) -> int:
        """Count memories matching filters."""
        sql = "SELECT COUNT(*) FROM memories WHERE 1=1"
        params: list = []
        if lifecycle_states:
            placeholders = ",".join(["?"] * len(lifecycle_states))
            sql += f" AND lifecycle_state IN ({placeholders})"
            params.extend(lifecycle_states)
        if user_id:
            sql += " AND user_id=?"
            params.append(user_id)
        with self._lock:
            row = self._get_conn().execute(sql, params).fetchone()
        return row[0] if row else 0

    def get_recent(
        self,
        n: int = 10,
        source: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> List[MemoryRecord]:
        """Get N most recent memories."""
        sql = "SELECT " + ",".join(_COLUMNS) + " FROM memories WHERE lifecycle_state != 'forgotten'"
        params: list = []
        if source:
            sql += " AND source=?"
            params.append(source)
        if user_id:
            sql += " AND user_id=?"
            params.append(user_id)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(n)
        with self._lock:
            rows = self._get_conn().execute(sql, params).fetchall()
        return [self._row_to_record(r) for r in rows]

    def get_all_ids_with_embeddings(
        self,
        has_embedding: Optional[bool] = None,
    ) -> List[Tuple[str, bool]]:
        """Get (id, has_embedding) for all memories. Used by migration/re-embed."""
        sql = "SELECT id, CASE WHEN embedding IS NOT NULL THEN 1 ELSE 0 END FROM memories"
        if has_embedding is True:
            sql += " WHERE embedding IS NOT NULL"
        elif has_embedding is False:
            sql += " WHERE embedding IS NULL"
        with self._lock:
            rows = self._get_conn().execute(sql).fetchall()
        return [(r[0], bool(r[1])) for r in rows]

    # ------------------------------------------------------------------
    # Batch operations
    # ------------------------------------------------------------------

    def batch_insert(
        self,
        records: List[MemoryRecord],
        embeddings: Optional[List[Optional[np.ndarray]]] = None,
    ) -> int:
        """Insert multiple records in one transaction. Returns count inserted."""
        if not records:
            return 0
        placeholders = ",".join(["?"] * len(_COLUMNS))
        sql = f"INSERT OR IGNORE INTO memories({','.join(_COLUMNS)}) VALUES({placeholders})"

        rows_data = []
        for i, rec in enumerate(records):
            rec.updated_at = datetime.now().isoformat()
            emb = embeddings[i] if embeddings and i < len(embeddings) else None
            emb_blob = _float32_to_blob(emb) if emb is not None else None
            rows_data.append((
                rec.id, rec.content, rec.title, rec.source,
                rec.memory_type, rec.importance,
                rec.keywords, rec.tags, rec.category,
                rec.boxes, rec.links,
                rec.temporal_context, rec.emotional_valence, rec.emotional_pad,
                rec.strength, rec.decay_rate, rec.access_count, rec.last_accessed,
                rec.lifecycle_state, rec.user_id,
                rec.created_at, rec.updated_at,
                rec.metadata, emb_blob,
            ))

        with self._lock:
            conn = self._get_conn()
            try:
                conn.executemany(sql, rows_data)
                conn.commit()
            except Exception:
                conn.rollback()
                raise
        return len(rows_data)

    def batch_decay(self) -> int:
        """Apply exponential decay to all active memories in one SQL UPDATE.

        strength_new = strength * exp(-decay_rate * hours_since_last_access)

        IMPORTANT: After computing new strength, we also update last_accessed
        to NOW so that the next batch_decay call only decays for the elapsed
        interval, not from the original last_accessed. Without this, each call
        would re-apply decay over the full cumulative time on an already-decayed
        value, causing exponential double-decay (the root cause of aggressive
        memory forgetting).

        Returns number of rows affected.
        """
        now = datetime.now().isoformat()
        now_ts = datetime.now().timestamp()
        sql = """SELECT id, strength, decay_rate, last_accessed
                 FROM memories
                 WHERE lifecycle_state IN ('candidate', 'stable', 'summary')
                 AND strength > 0.01"""

        # Hold lock for the entire read-compute-write cycle to prevent
        # TOCTOU races where concurrent modifications between the read
        # and write phases could be overwritten with stale values.
        with self._lock:
            conn = self._get_conn()
            rows = conn.execute(sql).fetchall()

            if not rows:
                return 0

            updates: List[Tuple[float, str, str, str]] = []
            for mem_id, strength, decay_rate, last_accessed in rows:
                try:
                    la_ts = datetime.fromisoformat(last_accessed).timestamp()
                except (ValueError, TypeError):
                    la_ts = now_ts
                hours = max(0, (now_ts - la_ts) / 3600)
                new_strength = strength * math.exp(-decay_rate * hours)
                new_strength = max(0.0, min(1.0, new_strength))
                updates.append((new_strength, now, now, mem_id))

            try:
                conn.executemany(
                    "UPDATE memories SET strength=?, last_accessed=?, updated_at=? WHERE id=?",
                    updates,
                )
                conn.commit()
            except Exception:
                conn.rollback()
                raise
        return len(updates)

    def prune_forgotten(self, threshold: float = 0.05) -> int:
        """Mark memories with strength below threshold as 'forgotten'.
        Returns number of rows affected."""
        now = datetime.now().isoformat()
        with self._lock:
            conn = self._get_conn()
            try:
                cur = conn.execute(
                    """UPDATE memories SET lifecycle_state='forgotten', updated_at=?
                       WHERE strength < ? AND lifecycle_state NOT IN ('forgotten', 'archived')""",
                    (now, threshold),
                )
                conn.commit()
                count = cur.rowcount
            except Exception:
                conn.rollback()
                raise
        return count

    # ------------------------------------------------------------------
    # User Profile
    # ------------------------------------------------------------------

    def save_user_profile(self, user_id: str, profile_json: str) -> None:
        """Upsert user profile as JSON string."""
        now = datetime.now().isoformat()
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    "INSERT OR REPLACE INTO user_profile(user_id, profile, updated_at) VALUES(?,?,?)",
                    (user_id, profile_json, now),
                )
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def load_user_profile(self, user_id: str) -> Optional[str]:
        """Load user profile JSON string. Returns None if not found."""
        with self._lock:
            row = self._get_conn().execute(
                "SELECT profile FROM user_profile WHERE user_id=?", (user_id,)
            ).fetchone()
        return row[0] if row else None

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Get store statistics."""
        with self._lock:
            conn = self._get_conn()
            total = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            by_state = conn.execute(
                "SELECT lifecycle_state, COUNT(*) FROM memories GROUP BY lifecycle_state"
            ).fetchall()
            by_source = conn.execute(
                "SELECT source, COUNT(*) FROM memories GROUP BY source"
            ).fetchall()
            embedded = conn.execute(
                "SELECT COUNT(*) FROM memories WHERE embedding IS NOT NULL"
            ).fetchone()[0]
            profiles = conn.execute("SELECT COUNT(*) FROM user_profile").fetchone()[0]
        return {
            "total_memories": total,
            "by_lifecycle": dict(by_state),
            "by_source": dict(by_source),
            "embedded_count": embedded,
            "user_profiles": profiles,
            "db_path": str(self._db_path),
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _row_to_record(self, row: tuple) -> MemoryRecord:
        """Convert a database row to a MemoryRecord (embedding excluded)."""
        return MemoryRecord(
            id=row[0],
            content=row[1],
            title=row[2],
            source=row[3],
            memory_type=row[4],
            importance=row[5],
            keywords=row[6],
            tags=row[7],
            category=row[8],
            boxes=row[9],
            links=row[10],
            temporal_context=row[11],
            emotional_valence=row[12],
            emotional_pad=row[13],
            strength=row[14],
            decay_rate=row[15],
            access_count=row[16],
            last_accessed=row[17],
            lifecycle_state=row[18],
            user_id=row[19],
            created_at=row[20],
            updated_at=row[21],
            metadata=row[22],
            # embedding at index 23 is BLOB, not stored in dataclass
        )

    def close(self) -> None:
        """Close the SQLite connection."""
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                except sqlite3.Error:
                    pass
                self._conn = None


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_store_instance: Optional[MemoryStore] = None
_store_lock = threading.Lock()


def get_memory_store(db_path: Optional[str] = None) -> MemoryStore:
    """Get or create the global MemoryStore singleton."""
    global _store_instance
    if _store_instance is not None:
        if db_path and str(db_path) != str(_store_instance._db_path):
            logger.warning(
                "get_memory_store called with different path %s, "
                "returning existing store at %s",
                db_path, _store_instance._db_path,
            )
        return _store_instance
    with _store_lock:
        if _store_instance is None:
            _store_instance = MemoryStore(db_path)
    return _store_instance


__all__ = [
    "MemoryRecord",
    "MemoryStore",
    "get_memory_store",
]
