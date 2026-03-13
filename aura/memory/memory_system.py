"""Memory system using SQLite persistence with hybrid vector+text search.

Replaces the original JSON+Jaccard implementation.
Uses nomic-embed-text (via Ollama) for semantic similarity with Jaccard fallback.
Drop-in replacement: identical public API.
"""

import atexit
import json
import logging
import math
import re
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..config import Config

logger = logging.getLogger(__name__)


def _cosine(a: list, b: list) -> float:
    """Cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na * nb > 0.0 else 0.0


def _jaccard(query: str, text: str) -> float:
    """Jaccard word-overlap similarity (fallback when embeddings unavailable)."""
    def tok(s: str) -> set:
        return set(re.findall(r'\b\w+\b', s.lower()))
    qt, tt = tok(query), tok(text)
    if not qt or not tt:
        return 0.0
    return len(qt & tt) / len(qt | tt)


class MemorySystem:
    """SQLite-backed memory with nomic-embed-text semantic search (Jaccard fallback)."""

    _EMBED_MODEL = "nomic-embed-text:latest"

    def __init__(self, collection_name: Optional[str] = None):
        self.collection_name = collection_name or Config.MEMORY_COLLECTION_NAME
        Config.CHROMADB_PATH.mkdir(parents=True, exist_ok=True)
        self._db_path = Config.CHROMADB_PATH / f"{self.collection_name}.db"
        self._lock = threading.Lock()
        self._conn: Optional[sqlite3.Connection] = None
        self._EMBED_URL = getattr(Config, 'OLLAMA_EMBED_URL', 'http://localhost:11434/api/embeddings')
        self._init_db()
        self._migrate_json()
        atexit.register(self.close)

    # ── Internal DB helpers ───────────────────────────────────────────────────

    def _get_conn(self) -> sqlite3.Connection:
        """Return the shared SQLite connection (WAL mode for concurrency).

        Must only be called while holding self._lock.
        """
        if self._conn is None:
            self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
        return self._conn

    def _init_db(self) -> None:
        with self._lock:
            conn = self._get_conn()
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id        TEXT PRIMARY KEY,
                    content   TEXT NOT NULL,
                    type      TEXT NOT NULL DEFAULT 'episodic',
                    timestamp TEXT NOT NULL,
                    metadata  TEXT NOT NULL DEFAULT '{}',
                    embedding TEXT DEFAULT NULL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_type ON memories(type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ts   ON memories(timestamp)")
            conn.commit()

    def _migrate_json(self) -> None:
        """One-time migration from legacy JSON file → SQLite."""
        old_path = Config.CHROMADB_PATH / f"{self.collection_name}.json"
        if not old_path.exists():
            return
        try:
            data = json.loads(old_path.read_text(encoding="utf-8"))
            if not isinstance(data, list) or not data:
                old_path.rename(old_path.with_suffix(".json.bak"))
                return
            with self._lock:
                conn = self._get_conn()
                conn.executemany(
                    "INSERT OR IGNORE INTO memories(id,content,type,timestamp,metadata) VALUES(?,?,?,?,?)",
                    [
                        (
                            m.get("id", f"migrated_{i}"),
                            m.get("content", ""),
                            m.get("type", "episodic"),
                            m.get("timestamp", datetime.now().isoformat()),
                            json.dumps(m.get("metadata", {})),
                        )
                        for i, m in enumerate(data)
                    ],
                )
                conn.commit()
            old_path.rename(old_path.with_suffix(".json.bak"))
        except (json.JSONDecodeError, OSError, sqlite3.Error) as e:
            logger.warning(f"[MemorySystem] JSON migration failed: {e}")

    # ── Embedding ─────────────────────────────────────────────────────────────

    def _embed(self, text: str) -> Optional[list]:
        """Get embedding from Ollama nomic-embed-text (fast, local)."""
        try:
            import requests
            r = requests.post(
                self._EMBED_URL,
                json={"model": self._EMBED_MODEL, "prompt": text[:1000]},
                timeout=2,
            )
            if r.status_code == 200:
                return r.json().get("embedding")
        except Exception as e:
            logger.debug(f"[MemorySystem] Embedding failed: {e}")
        return None

    # ── Public API ────────────────────────────────────────────────────────────

    def remember(
        self,
        content: str,
        memory_type: str = "episodic",
        metadata: Optional[dict] = None,
    ) -> str:
        """Store a new memory. Returns the memory_id."""
        memory_id = f"{memory_type}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        embedding = self._embed(content)
        embedding_json = json.dumps(embedding) if embedding else None

        with self._lock:
            conn = self._get_conn()
            conn.execute(
                "INSERT OR REPLACE INTO memories(id,content,type,timestamp,metadata,embedding) VALUES(?,?,?,?,?,?)",
                (
                    memory_id,
                    content,
                    memory_type,
                    datetime.now().isoformat(),
                    json.dumps(metadata or {}),
                    embedding_json,
                ),
            )
            conn.commit()
        return memory_id

    def recall(
        self,
        query: str,
        n_results: Optional[int] = None,
        memory_type: Optional[str] = None,
    ) -> list[dict]:
        """Retrieve memories ranked by semantic (or Jaccard) similarity.

        Optimizations vs naive full-table scan:
        - Try embedding first; only fetch embedding column if available
        - When falling back to Jaccard, limit to most recent 500 rows
        """
        n_results = n_results or Config.MAX_MEMORY_RESULTS

        # Try to embed the query first so we know which columns to fetch
        query_vec = self._embed(query)

        if query_vec:
            # Semantic mode: fetch rows that have embeddings (best quality)
            sql = "SELECT id,content,type,timestamp,metadata,embedding FROM memories"
        else:
            # Jaccard fallback: only need text, skip embedding column, limit scan
            sql = "SELECT id,content,type,timestamp,metadata FROM memories"

        params: list = []
        wheres = []
        if memory_type:
            wheres.append("type=?")
            params.append(memory_type)
        if query_vec:
            wheres.append("embedding IS NOT NULL")

        if wheres:
            sql += " WHERE " + " AND ".join(wheres)

        if not query_vec:
            # Jaccard: limit to recent 500 to avoid O(N) on huge tables
            sql += " ORDER BY timestamp DESC LIMIT 500"

        with self._lock:
            rows = self._get_conn().execute(sql, params).fetchall()

        if not rows:
            if query_vec:
                # Semantic mode found no embedded rows — fall back to Jaccard
                query_vec = None
                sql = "SELECT id,content,type,timestamp,metadata FROM memories"
                params = []
                if memory_type:
                    sql += " WHERE type=?"
                    params.append(memory_type)
                sql += " ORDER BY timestamp DESC LIMIT 500"
                with self._lock:
                    rows = self._get_conn().execute(sql, params).fetchall()
            if not rows:
                return []

        scored = []
        if query_vec:
            for row_id, content, mem_type, ts, meta_str, emb_str in rows:
                try:
                    emb = json.loads(emb_str)
                    score = _cosine(query_vec, emb)
                except (json.JSONDecodeError, ValueError, ZeroDivisionError):
                    score = _jaccard(query, content)
                scored.append((score, row_id, content, mem_type, ts, meta_str))
        else:
            for row_id, content, mem_type, ts, meta_str in rows:
                score = _jaccard(query, content)
                scored.append((score, row_id, content, mem_type, ts, meta_str))

        scored.sort(key=lambda x: x[0], reverse=True)

        return [
            {
                "id": row_id,
                "content": content,
                "metadata": {
                    "type": mem_type,
                    "timestamp": ts,
                    **json.loads(meta_str or "{}"),
                },
                "distance": round(1.0 - score, 4),
            }
            for score, row_id, content, mem_type, ts, meta_str in scored[:n_results]
        ]

    def forget(self, memory_id: str) -> None:
        """Remove a specific memory by ID."""
        with self._lock:
            conn = self._get_conn()
            conn.execute("DELETE FROM memories WHERE id=?", (memory_id,))
            conn.commit()

    def get_recent(self, n: int = 10, memory_type: Optional[str] = None) -> list[dict]:
        """Get the N most recent memories."""
        sql = "SELECT id,content,type,timestamp,metadata FROM memories"
        params: list = []
        if memory_type:
            sql += " WHERE type=?"
            params.append(memory_type)
        sql += " ORDER BY timestamp DESC LIMIT ?"
        params.append(n)

        with self._lock:
            rows = self._get_conn().execute(sql, params).fetchall()

        return [
            {
                "id": row_id,
                "content": content,
                "metadata": {"type": mem_type, "timestamp": ts, **json.loads(meta_str or "{}")},
            }
            for row_id, content, mem_type, ts, meta_str in rows
        ]

    def clear(self) -> None:
        """Clear all memories."""
        with self._lock:
            conn = self._get_conn()
            conn.execute("DELETE FROM memories")
            conn.commit()

    def count(self) -> int:
        """Return total number of stored memories."""
        with self._lock:
            row = self._get_conn().execute("SELECT COUNT(*) FROM memories").fetchone()
        return row[0] if row else 0

    def close(self) -> None:
        """Explicitly close the SQLite connection."""
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                except sqlite3.Error:
                    pass
                self._conn = None

    def __del__(self):
        self.close()
