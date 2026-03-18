"""
SQLite Persistence for AURA Proactive System.

Persists beliefs, decisions, events, daemon stats, and salience filter state
across restarts. Uses a single SQLite database with 6 tables.

Follows the existing _save_state()/_load_state() pattern used by
intrinsic_motivation.py, metacognition.py, theory_of_mind.py — but backed
by SQLite instead of JSON for efficient append + query + size caps.

All persistence calls are wrapped in try/except for graceful degradation:
if SQLite fails, the system works exactly as before (in-memory only).
"""

import json
import logging
import sqlite3
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger(__name__)


class ProactivePersistence:
    """Central SQLite persistence manager for the proactive subsystem.

    Thread-safe via threading.Lock (matching existing codebase pattern).
    Single database file at data/proactive/proactive_state.db.
    """

    # Row caps to prevent unbounded growth
    MAX_DECISIONS = 500
    MAX_ACTION_HISTORY = 200
    MAX_EVENT_LOG = 1000
    MAX_ACTIVITY_LOG = 2000

    def __init__(self, db_path: Optional[Path] = None):
        if db_path is None:
            base = Path(__file__).resolve().parent.parent.parent
            db_path = base / "data" / "proactive" / "proactive_state.db"

        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn: Optional[sqlite3.Connection] = None

        self._connect()
        self._create_tables()

        import atexit as _atexit
        _atexit.register(self.close)

        logger.info(f"[Persistence] Initialized at {self._db_path}")

    def _connect(self) -> None:
        """Open SQLite connection."""
        self._conn = sqlite3.connect(
            str(self._db_path),
            check_same_thread=False,
            timeout=5.0,
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")

    def _create_tables(self) -> None:
        """Create tables if they don't exist."""
        with self._lock:
            c = self._conn
            c.execute("""
                CREATE TABLE IF NOT EXISTS beliefs (
                    id INTEGER PRIMARY KEY,
                    user_busy REAL,
                    user_receptive REAL,
                    task_urgent REAL,
                    context_stable REAL,
                    uncertainty REAL,
                    updated_at TEXT
                )
            """)
            c.execute("""
                CREATE TABLE IF NOT EXISTS decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    action TEXT,
                    confidence REAL,
                    expected_free_energy REAL,
                    reasoning TEXT,
                    beliefs_snapshot TEXT,
                    created_at TEXT
                )
            """)
            c.execute("""
                CREATE TABLE IF NOT EXISTS daemon_state (
                    id INTEGER PRIMARY KEY,
                    state TEXT,
                    events_received INTEGER,
                    events_filtered INTEGER,
                    decisions_made INTEGER,
                    messages_sent INTEGER,
                    last_proactive_message_time REAL,
                    user_context TEXT,
                    updated_at TEXT
                )
            """)
            c.execute("""
                CREATE TABLE IF NOT EXISTS action_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    action TEXT,
                    taken_at TEXT
                )
            """)
            c.execute("""
                CREATE TABLE IF NOT EXISTS seen_events (
                    event_hash TEXT PRIMARY KEY,
                    first_seen REAL,
                    last_seen REAL
                )
            """)
            c.execute("""
                CREATE TABLE IF NOT EXISTS event_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT,
                    event_type TEXT,
                    priority INTEGER,
                    payload TEXT,
                    salience_score REAL,
                    created_at REAL
                )
            """)
            c.execute("""
                CREATE TABLE IF NOT EXISTS pymdp_state (
                    id INTEGER PRIMARY KEY,
                    state_json TEXT,
                    updated_at TEXT
                )
            """)
            c.execute("""
                CREATE TABLE IF NOT EXISTS activity_log (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp    REAL    NOT NULL,
                    category     TEXT    NOT NULL,
                    event_type   TEXT    NOT NULL,
                    summary      TEXT    NOT NULL,
                    payload_json TEXT,
                    duration_ms  INTEGER
                )
            """)
            c.execute(
                "CREATE INDEX IF NOT EXISTS idx_activity_ts ON activity_log(timestamp)"
            )
            c.commit()

    # ── Beliefs ──────────────────────────────────────────────────────

    def save_beliefs(self, beliefs: Any) -> None:
        """Save current belief state (single-row upsert, id=1)."""
        try:
            with self._lock:
                self._conn.execute(
                    """INSERT OR REPLACE INTO beliefs
                       (id, user_busy, user_receptive, task_urgent,
                        context_stable, uncertainty, updated_at)
                       VALUES (1, ?, ?, ?, ?, ?, ?)""",
                    (
                        beliefs.user_busy,
                        beliefs.user_receptive,
                        beliefs.task_urgent,
                        beliefs.context_stable,
                        beliefs.uncertainty,
                        datetime.now().isoformat(),
                    ),
                )
                self._conn.commit()
        except Exception as e:
            logger.warning(f"[Persistence] save_beliefs error: {e}")

    def load_beliefs(self) -> Optional[Dict[str, float]]:
        """Load belief state. Returns dict or None if no persisted state."""
        try:
            with self._lock:
                row = self._conn.execute(
                    "SELECT user_busy, user_receptive, task_urgent, "
                    "context_stable, uncertainty FROM beliefs WHERE id=1"
                ).fetchone()
            if row:
                return {
                    "user_busy": row[0],
                    "user_receptive": row[1],
                    "task_urgent": row[2],
                    "context_stable": row[3],
                    "uncertainty": row[4],
                }
        except Exception as e:
            logger.warning(f"[Persistence] load_beliefs error: {e}")
        return None

    # ── Decisions ────────────────────────────────────────────────────

    def save_decision(self, decision: Any, beliefs: Any) -> None:
        """Save a proactive decision with beliefs snapshot."""
        try:
            beliefs_json = json.dumps({
                "user_busy": beliefs.user_busy,
                "user_receptive": beliefs.user_receptive,
                "task_urgent": beliefs.task_urgent,
                "context_stable": beliefs.context_stable,
                "uncertainty": beliefs.uncertainty,
            })
            with self._lock:
                self._conn.execute(
                    """INSERT INTO decisions
                       (action, confidence, expected_free_energy,
                        reasoning, beliefs_snapshot, created_at)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        decision.action.value,
                        decision.confidence,
                        decision.expected_free_energy,
                        decision.reasoning,
                        beliefs_json,
                        datetime.now().isoformat(),
                    ),
                )
                # Trim to cap
                self._conn.execute(
                    """DELETE FROM decisions WHERE id NOT IN
                       (SELECT id FROM decisions ORDER BY id DESC LIMIT ?)""",
                    (self.MAX_DECISIONS,),
                )
                self._conn.commit()
        except Exception as e:
            logger.warning(f"[Persistence] save_decision error: {e}")

    def get_recent_decisions(self, limit: int = 50) -> List[dict]:
        """Get recent decisions (newest first)."""
        try:
            with self._lock:
                rows = self._conn.execute(
                    """SELECT action, confidence, expected_free_energy,
                              reasoning, beliefs_snapshot, created_at
                       FROM decisions ORDER BY id DESC LIMIT ?""",
                    (limit,),
                ).fetchall()
            return [
                {
                    "action": r[0],
                    "confidence": r[1],
                    "expected_free_energy": r[2],
                    "reasoning": r[3],
                    "beliefs_snapshot": json.loads(r[4]) if r[4] else {},
                    "created_at": r[5],
                }
                for r in rows
            ]
        except Exception as e:
            logger.warning(f"[Persistence] get_recent_decisions error: {e}")
            return []

    # ── Daemon state ─────────────────────────────────────────────────

    def save_daemon_state(
        self,
        stats: dict,
        user_context: dict,
        last_msg_time: float,
    ) -> None:
        """Save daemon operational state (single-row upsert, id=1)."""
        try:
            ctx_json = json.dumps(user_context)
            with self._lock:
                self._conn.execute(
                    """INSERT OR REPLACE INTO daemon_state
                       (id, state, events_received, events_filtered,
                        decisions_made, messages_sent,
                        last_proactive_message_time, user_context, updated_at)
                       VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        "running",
                        stats.get("events_received", 0),
                        stats.get("events_filtered", 0),
                        stats.get("decisions_made", 0),
                        stats.get("messages_sent", 0),
                        last_msg_time,
                        ctx_json,
                        datetime.now().isoformat(),
                    ),
                )
                self._conn.commit()
        except Exception as e:
            logger.warning(f"[Persistence] save_daemon_state error: {e}")

    def load_daemon_state(self) -> Optional[dict]:
        """Load daemon state. Returns dict or None."""
        try:
            with self._lock:
                row = self._conn.execute(
                    """SELECT events_received, events_filtered, decisions_made,
                              messages_sent, last_proactive_message_time,
                              user_context
                       FROM daemon_state WHERE id=1"""
                ).fetchone()
            if row:
                ctx = {}
                if row[5]:
                    try:
                        ctx = json.loads(row[5])
                    except (json.JSONDecodeError, TypeError):
                        pass
                return {
                    "events_received": row[0],
                    "events_filtered": row[1],
                    "decisions_made": row[2],
                    "messages_sent": row[3],
                    "last_proactive_message_time": row[4],
                    "user_context": ctx,
                }
        except Exception as e:
            logger.warning(f"[Persistence] load_daemon_state error: {e}")
        return None

    # ── Action history ───────────────────────────────────────────────

    def save_action(self, action: str, taken_at: datetime) -> None:
        """Append an action to history (capped at MAX_ACTION_HISTORY)."""
        try:
            with self._lock:
                self._conn.execute(
                    "INSERT INTO action_history (action, taken_at) VALUES (?, ?)",
                    (action, taken_at.isoformat()),
                )
                self._conn.execute(
                    """DELETE FROM action_history WHERE id NOT IN
                       (SELECT id FROM action_history ORDER BY id DESC LIMIT ?)""",
                    (self.MAX_ACTION_HISTORY,),
                )
                self._conn.commit()
        except Exception as e:
            logger.warning(f"[Persistence] save_action error: {e}")

    def load_action_history(self, limit: int = 100) -> List[Tuple[str, datetime]]:
        """Load action history (oldest first)."""
        try:
            with self._lock:
                rows = self._conn.execute(
                    """SELECT action, taken_at FROM action_history
                       ORDER BY id DESC LIMIT ?""",
                    (limit,),
                ).fetchall()
            result = []
            for action_str, taken_at_str in reversed(rows):
                try:
                    taken_at = datetime.fromisoformat(taken_at_str)
                except (ValueError, TypeError):
                    taken_at = datetime.now()
                result.append((action_str, taken_at))
            return result
        except Exception as e:
            logger.warning(f"[Persistence] load_action_history error: {e}")
            return []

    # ── Seen events (salience novelty tracking) ──────────────────────

    def save_seen_event(self, event_hash: str, timestamp: float) -> None:
        """Save or update a seen event hash."""
        try:
            with self._lock:
                self._conn.execute(
                    """INSERT INTO seen_events (event_hash, first_seen, last_seen)
                       VALUES (?, ?, ?)
                       ON CONFLICT(event_hash) DO UPDATE SET last_seen=?""",
                    (event_hash, timestamp, timestamp, timestamp),
                )
                self._conn.commit()
        except Exception as e:
            logger.warning(f"[Persistence] save_seen_event error: {e}")

    def load_seen_events(self, ttl: float = 3600.0) -> Dict[str, float]:
        """Load seen events within TTL. Cleans up expired entries."""
        try:
            cutoff = time.time() - ttl
            with self._lock:
                # Delete expired
                self._conn.execute(
                    "DELETE FROM seen_events WHERE last_seen < ?", (cutoff,)
                )
                self._conn.commit()
                rows = self._conn.execute(
                    "SELECT event_hash, last_seen FROM seen_events"
                ).fetchall()
            return {row[0]: row[1] for row in rows}
        except Exception as e:
            logger.warning(f"[Persistence] load_seen_events error: {e}")
            return {}

    # ── Event log ────────────────────────────────────────────────────

    def log_event(self, event: Any, salience_score: float) -> None:
        """Log a filtered event (capped at MAX_EVENT_LOG)."""
        try:
            payload_json = json.dumps(event.payload) if event.payload else "{}"
            priority_val = (
                event.priority.value
                if hasattr(event.priority, "value")
                else int(event.priority)
            )
            with self._lock:
                self._conn.execute(
                    """INSERT INTO event_log
                       (source, event_type, priority, payload,
                        salience_score, created_at)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        event.source,
                        event.event_type,
                        priority_val,
                        payload_json,
                        salience_score,
                        time.time(),
                    ),
                )
                self._conn.execute(
                    """DELETE FROM event_log WHERE id NOT IN
                       (SELECT id FROM event_log ORDER BY id DESC LIMIT ?)""",
                    (self.MAX_EVENT_LOG,),
                )
                self._conn.commit()
        except Exception as e:
            logger.warning(f"[Persistence] log_event error: {e}")

    def get_recent_events(self, limit: int = 50) -> List[dict]:
        """Get recent logged events (newest first)."""
        try:
            with self._lock:
                rows = self._conn.execute(
                    """SELECT source, event_type, priority, payload,
                              salience_score, created_at
                       FROM event_log ORDER BY id DESC LIMIT ?""",
                    (limit,),
                ).fetchall()
            return [
                {
                    "source": r[0],
                    "event_type": r[1],
                    "priority": r[2],
                    "payload": json.loads(r[3]) if r[3] else {},
                    "salience_score": r[4],
                    "created_at": r[5],
                }
                for r in rows
            ]
        except Exception as e:
            logger.warning(f"[Persistence] get_recent_events error: {e}")
            return []

    # ── pymdp state ───────────────────────────────────────────────────

    def save_pymdp_state(self, state: dict) -> None:
        """Save pymdp learned state (single-row upsert, id=1)."""
        try:
            state_json = json.dumps(state)
            with self._lock:
                self._conn.execute(
                    """INSERT OR REPLACE INTO pymdp_state
                       (id, state_json, updated_at)
                       VALUES (1, ?, ?)""",
                    (state_json, datetime.now().isoformat()),
                )
                self._conn.commit()
        except Exception as e:
            logger.warning(f"[Persistence] save_pymdp_state error: {e}")

    def load_pymdp_state(self) -> Optional[dict]:
        """Load pymdp learned state. Returns dict or None."""
        try:
            with self._lock:
                row = self._conn.execute(
                    "SELECT state_json FROM pymdp_state WHERE id=1"
                ).fetchone()
            if row and row[0]:
                return json.loads(row[0])
        except Exception as e:
            logger.warning(f"[Persistence] load_pymdp_state error: {e}")
        return None

    # ── Activity log ─────────────────────────────────────────────────

    def log_activity(self, category: str, event_type: str, summary: str,
                     payload: dict = None, duration_ms: int = None) -> None:
        """Append one activity event. Capped at MAX_ACTIVITY_LOG rows."""
        try:
            with self._lock:
                self._conn.execute(
                    "INSERT INTO activity_log"
                    " (timestamp, category, event_type, summary, payload_json, duration_ms)"
                    " VALUES (?, ?, ?, ?, ?, ?)",
                    (time.time(), category, event_type, summary,
                     json.dumps(payload) if payload else None, duration_ms)
                )
                self._conn.execute(
                    "DELETE FROM activity_log WHERE id NOT IN"
                    " (SELECT id FROM activity_log ORDER BY id DESC LIMIT ?)",
                    (self.MAX_ACTIVITY_LOG,)
                )
                self._conn.commit()
        except Exception as e:
            logger.warning(f"[Persistence] log_activity error: {e}")

    def get_activity_events(self, limit: int = 100, after: float = 0.0,
                            categories: list = None, before: float = None) -> list:
        """Return events newer than `after` timestamp, newest first.
        Optional `before` for paginating older events."""
        try:
            with self._lock:
                if categories:
                    ph = ",".join("?" * len(categories))
                    before_clause = " AND timestamp < ?" if before is not None else ""
                    params = [after, *categories]
                    if before is not None:
                        params.append(before)
                    params.append(limit)
                    rows = self._conn.execute(
                        f"SELECT id, timestamp, category, event_type, summary,"
                        f" payload_json, duration_ms FROM activity_log"
                        f" WHERE timestamp > ? AND category IN ({ph}){before_clause}"
                        f" ORDER BY timestamp DESC LIMIT ?",
                        params
                    ).fetchall()
                else:
                    before_clause = " AND timestamp < ?" if before is not None else ""
                    params = [after]
                    if before is not None:
                        params.append(before)
                    params.append(limit)
                    rows = self._conn.execute(
                        f"SELECT id, timestamp, category, event_type, summary,"
                        f" payload_json, duration_ms FROM activity_log"
                        f" WHERE timestamp > ?{before_clause}"
                        f" ORDER BY timestamp DESC LIMIT ?",
                        params
                    ).fetchall()
            return [
                {"id": r[0], "timestamp": r[1], "category": r[2],
                 "event_type": r[3], "summary": r[4],
                 "payload": json.loads(r[5]) if r[5] else None,
                 "duration_ms": r[6]}
                for r in rows
            ]
        except Exception as e:
            logger.warning(f"[Persistence] get_activity_events error: {e}")
            return []

    # ── Lifecycle ────────────────────────────────────────────────────

    def close(self) -> None:
        """Close the database connection."""
        try:
            with self._lock:
                if self._conn:
                    self._conn.close()
                    self._conn = None
            logger.info("[Persistence] Database closed")
        except Exception as e:
            logger.warning(f"[Persistence] close error: {e}")


# ── Singleton ────────────────────────────────────────────────────────

_persistence: Optional[ProactivePersistence] = None
_persistence_lock = threading.Lock()


def get_persistence() -> ProactivePersistence:
    """Get or create the global ProactivePersistence instance."""
    global _persistence
    if _persistence is None:
        with _persistence_lock:
            if _persistence is None:
                _persistence = ProactivePersistence()
    return _persistence
