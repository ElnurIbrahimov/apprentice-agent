# aura/cli/activity_log.py
"""Activity log — SQLite-backed queryable interaction history."""
from __future__ import annotations

import json
import logging
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


_LOG_DB_PATH = Path.home() / ".aura" / "logs.db"


@dataclass
class LogEntry:
    """A single interaction log entry."""
    id: int
    timestamp: float
    session_id: str
    prompt: str
    response: str
    model: str
    tokens_in: int
    tokens_out: int
    cost: float
    tool_calls: int


class ActivityLog:
    """SQLite-backed activity log for queryable history."""

    def __init__(self, db_path: Optional[Path] = None):
        self._db_path = db_path or _LOG_DB_PATH
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        conn = sqlite3.connect(str(self._db_path), timeout=10)
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    session_id TEXT DEFAULT '',
                    prompt TEXT NOT NULL,
                    response TEXT DEFAULT '',
                    model TEXT DEFAULT '',
                    tokens_in INTEGER DEFAULT 0,
                    tokens_out INTEGER DEFAULT 0,
                    cost REAL DEFAULT 0.0,
                    tool_calls INTEGER DEFAULT 0
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_interactions_ts ON interactions(timestamp)
            """)
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS interactions_fts
                USING fts5(prompt, response)
            """)
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS interactions_ai AFTER INSERT ON interactions BEGIN
                    INSERT INTO interactions_fts(rowid, prompt, response) VALUES (new.id, new.prompt, new.response);
                END
            """)
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS interactions_ad AFTER DELETE ON interactions BEGIN
                    INSERT INTO interactions_fts(interactions_fts, rowid, prompt, response) VALUES('delete', old.id, old.prompt, old.response);
                END
            """)
            # Migration: add embedding column for semantic search
            try:
                conn.execute("ALTER TABLE interactions ADD COLUMN embedding BLOB DEFAULT NULL")
            except sqlite3.OperationalError:
                pass  # Column already exists
            conn.commit()
        finally:
            conn.close()

    def log(self, prompt: str, response: str = "", model: str = "",
            session_id: str = "", tokens_in: int = 0, tokens_out: int = 0,
            cost: float = 0.0, tool_calls: int = 0) -> int:
        """Log an interaction. Returns the row ID."""
        conn = sqlite3.connect(str(self._db_path), timeout=10)
        try:
            cursor = conn.execute(
                """INSERT INTO interactions
                   (timestamp, session_id, prompt, response, model, tokens_in, tokens_out, cost, tool_calls)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (time.time(), session_id, prompt[:10000], response[:50000], model,
                 tokens_in, tokens_out, cost, tool_calls),
            )
            row_id = cursor.lastrowid
            conn.commit()
            # Optionally compute and store embedding for semantic search
            try:
                import struct

                import ollama
                resp = ollama.embed(model="nomic-embed-text:latest", input=f"{prompt} {(response or '')[:500]}")
                if resp and "embeddings" in resp and resp["embeddings"]:
                    emb = resp["embeddings"][0]
                    blob = struct.pack(f'{len(emb)}f', *emb)
                    conn.execute("UPDATE interactions SET embedding = ? WHERE id = ?", (blob, row_id))
                    conn.commit()
            except Exception:
                logger.debug("Failed to store embedding for activity log entry", exc_info=True)
            return row_id
        except Exception as e:
            logger.warning("[ActivityLog] Failed to log interaction: %s", e)
            conn.rollback()
            return -1
        finally:
            conn.close()

    def search(self, query: str, limit: int = 20) -> List[Dict]:
        """Full-text search across prompts and responses."""
        conn = sqlite3.connect(str(self._db_path), timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            # Sanitize FTS query
            safe_query = " ".join(re.sub(r'[^\w\s]', '', query).split())
            if not safe_query:
                return []
            rows = conn.execute(
                """SELECT i.* FROM interactions i
                   JOIN interactions_fts f ON i.id = f.rowid
                   WHERE interactions_fts MATCH ?
                   ORDER BY i.timestamp DESC LIMIT ?""",
                (safe_query, limit),
            ).fetchall()
            return [dict(r) for r in rows]
        except Exception:
            logger.debug("Failed to perform FTS search in activity log", exc_info=True)
            return []
        finally:
            conn.close()

    def semantic_search(self, query: str, limit: int = 20) -> list[dict]:
        """Semantic search using embedding cosine similarity. Falls back to FTS5."""
        try:
            import math
            import struct

            import ollama

            resp = ollama.embed(model="nomic-embed-text:latest", input=query)
            if not resp or "embeddings" not in resp or not resp["embeddings"]:
                return self.search(query, limit)

            query_emb = resp["embeddings"][0]

            conn = sqlite3.connect(str(self._db_path), timeout=10)
            try:
                rows = conn.execute(
                    "SELECT id, timestamp, session_id, prompt, response, model, embedding "
                    "FROM interactions WHERE embedding IS NOT NULL "
                    "ORDER BY timestamp DESC LIMIT 500"
                ).fetchall()
            finally:
                conn.close()

            scored = []
            for row in rows:
                emb_blob = row[6]  # embedding column
                if not emb_blob:
                    continue
                n = len(emb_blob) // 4
                row_emb = struct.unpack(f'{n}f', emb_blob)

                # Cosine similarity
                dot = sum(a * b for a, b in zip(query_emb, row_emb))
                norm_q = math.sqrt(sum(a * a for a in query_emb))
                norm_r = math.sqrt(sum(a * a for a in row_emb))
                sim = dot / (norm_q * norm_r + 1e-8)

                if sim > 0.3:
                    scored.append((dict(zip(
                        ["id", "timestamp", "session_id", "prompt", "response", "model"],
                        row[:6]
                    )), sim))

            scored.sort(key=lambda x: -x[1])
            return [r for r, _ in scored[:limit]]
        except Exception:
            logger.debug("Semantic search failed, falling back to FTS5", exc_info=True)
            return self.search(query, limit)  # FTS5 fallback

    def get_recent(self, limit: int = 20) -> List[Dict]:
        """Get recent interactions."""
        conn = sqlite3.connect(str(self._db_path), timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT * FROM interactions ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]
        except Exception:
            logger.debug("Failed to fetch recent interactions from activity log", exc_info=True)
            return []
        finally:
            conn.close()

    def get_stats(self) -> Dict:
        """Get aggregate statistics."""
        conn = sqlite3.connect(str(self._db_path), timeout=10)
        try:
            row = conn.execute("""
                SELECT COUNT(*) as total,
                       SUM(tokens_in) as total_tokens_in,
                       SUM(tokens_out) as total_tokens_out,
                       SUM(cost) as total_cost,
                       SUM(tool_calls) as total_tool_calls
                FROM interactions
            """).fetchone()
            return {
                "total_interactions": row[0] or 0,
                "total_tokens_in": row[1] or 0,
                "total_tokens_out": row[2] or 0,
                "total_cost": row[3] or 0.0,
                "total_tool_calls": row[4] or 0,
            }
        except Exception:
            logger.debug("Failed to compute activity log stats", exc_info=True)
            return {"total_interactions": 0}
        finally:
            conn.close()

    def export_session(self, session_id: str, format: str = "markdown") -> str:
        """Export a session's interactions."""
        conn = sqlite3.connect(str(self._db_path), timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT * FROM interactions WHERE session_id = ? ORDER BY timestamp",
                (session_id,),
            ).fetchall()

            if format == "json":
                return json.dumps([dict(r) for r in rows], indent=2, default=str)
            else:  # markdown
                lines = [f"# Session: {session_id}", ""]
                for r in rows:
                    ts = time.strftime("%H:%M:%S", time.localtime(r["timestamp"]))
                    lines.append(f"## [{ts}] User")
                    lines.append(r["prompt"])
                    lines.append(f"\n## [{ts}] Aura ({r['model']})")
                    lines.append(r["response"])
                    lines.append("")
                return "\n".join(lines)
        except Exception:
            return ""
        finally:
            conn.close()
