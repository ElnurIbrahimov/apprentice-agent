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
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]

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
        self._lock = threading.Lock()
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()
        atexit.register(self.close)

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
                conn.execute(sql, values)
                conn.commit()
            except Exception:
                conn.rollback()
                raise
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
        vals = list(kwargs.values()) + [memory_id]
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
        """Cosine similarity search on embedding BLOBs using chunked iteration.

        Returns list of (record, similarity_score) sorted descending.
        Only considers memories with non-NULL embeddings.
        """
        states = lifecycle_states or ["candidate", "stable", "summary"]
        state_placeholders = ",".join(["?"] * len(states))

        sql = f"""SELECT {','.join(_COLUMNS)} FROM memories
                  WHERE embedding IS NOT NULL
                  AND lifecycle_state IN ({state_placeholders})"""
        params: list = list(states)
        if user_id:
            sql += " AND user_id=?"
            params.append(user_id)

        # Pre-compute query unit vector
        q = query_embedding.astype(np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm < 1e-8:
            return []
        q_unit = q / q_norm

        emb_idx = _COLUMNS.index("embedding")
        scored: List[Tuple[MemoryRecord, float]] = []

        # Fetch all rows under lock to prevent cursor race conditions.
        # The cursor must not be used after the lock is released because
        # concurrent writes could invalidate it.
        with self._lock:
            cursor = self._get_conn().execute(sql, params)
            all_rows = cursor.fetchall()

        for row in all_rows:
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
            scored.append((self._row_to_record(row), sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

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
        params: list = [match_expr] + list(states)
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
            conn.execute(
                "INSERT OR REPLACE INTO user_profile(user_id, profile, updated_at) VALUES(?,?,?)",
                (user_id, profile_json, now),
            )
            conn.commit()

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
    "MemoryStore",
    "MemoryRecord",
    "get_memory_store",
]
