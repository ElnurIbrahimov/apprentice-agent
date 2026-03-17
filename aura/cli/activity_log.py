# aura/cli/activity_log.py
"""Activity log — SQLite-backed queryable interaction history."""
from __future__ import annotations
import json
import sqlite3
import time
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass


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
        conn = sqlite3.connect(str(self._db_path))
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
                USING fts5(prompt, response, content=interactions, content_rowid=id)
            """)
            conn.commit()
        finally:
            conn.close()

    def log(self, prompt: str, response: str = "", model: str = "",
            session_id: str = "", tokens_in: int = 0, tokens_out: int = 0,
            cost: float = 0.0, tool_calls: int = 0) -> int:
        """Log an interaction. Returns the row ID."""
        conn = sqlite3.connect(str(self._db_path))
        try:
            cursor = conn.execute(
                """INSERT INTO interactions
                   (timestamp, session_id, prompt, response, model, tokens_in, tokens_out, cost, tool_calls)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (time.time(), session_id, prompt[:10000], response[:50000], model,
                 tokens_in, tokens_out, cost, tool_calls),
            )
            row_id = cursor.lastrowid
            # Update FTS
            conn.execute(
                "INSERT INTO interactions_fts(rowid, prompt, response) VALUES (?, ?, ?)",
                (row_id, prompt[:10000], response[:50000]),
            )
            conn.commit()
            return row_id
        except Exception:
            conn.rollback()
            return -1
        finally:
            conn.close()

    def search(self, query: str, limit: int = 20) -> List[Dict]:
        """Full-text search across prompts and responses."""
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        try:
            # Sanitize FTS query
            safe_query = " ".join(w for w in query.split() if w.isalnum())
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
            return []
        finally:
            conn.close()

    def get_recent(self, limit: int = 20) -> List[Dict]:
        """Get recent interactions."""
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT * FROM interactions ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]
        except Exception:
            return []
        finally:
            conn.close()

    def get_stats(self) -> Dict:
        """Get aggregate statistics."""
        conn = sqlite3.connect(str(self._db_path))
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
            return {"total_interactions": 0}
        finally:
            conn.close()

    def export_session(self, session_id: str, format: str = "markdown") -> str:
        """Export a session's interactions."""
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT * FROM interactions WHERE session_id = ? ORDER BY timestamp",
                (session_id,),
            ).fetchall()

            if format == "json":
                return json.dumps([dict(r) for r in rows], indent=2)
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
