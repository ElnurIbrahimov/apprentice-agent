"""Telegram Persistence Store — SQLite-backed state for TelegramBot.

Replaces in-memory dicts + JSON files with a single SQLite database.
Persists: user settings, document context, premium status, active chats,
group message cache, user locations, and per-user preferences.

Tables:
  user_settings    — language, keyboard, digest, model preferences
  premium_users    — payment tier, purchased_at, transaction_id
  active_chats     — tracked chat metadata
  document_context — uploaded doc text with TTL
  group_messages   — cached group messages for summarization
  user_locations   — last shared location per user
  inline_cache     — LRU inline query results

Author: Aura Development Team
Created: 2026-04-02
"""

import json
import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_DEFAULT_DB_PATH = Path("data/telegram_state.db")


class TelegramStore:
    """Thread-safe SQLite persistence for TelegramBot state."""

    def __init__(self, db_path: Optional[str] = None):
        self._db_path = Path(db_path) if db_path else _DEFAULT_DB_PATH
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()
        logger.info(f"[TelegramStore] Initialized at {self._db_path}")

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA cache_size=-4000")  # 4MB cache
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _init_db(self) -> None:
        with self._lock:
            conn = self._get_conn()
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS user_settings (
                    user_id TEXT PRIMARY KEY,
                    language TEXT DEFAULT 'en',
                    keyboard_enabled INTEGER DEFAULT 1,
                    digest_enabled INTEGER DEFAULT 0,
                    first_name TEXT,
                    username TEXT,
                    custom_model TEXT,
                    custom_instructions TEXT,
                    tts_enabled INTEGER DEFAULT 1,
                    created_at TEXT DEFAULT (datetime('now')),
                    updated_at TEXT DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS premium_users (
                    user_id TEXT PRIMARY KEY,
                    tier TEXT NOT NULL,
                    purchased_at TEXT,
                    transaction_id TEXT,
                    stars_amount INTEGER DEFAULT 0,
                    metadata TEXT DEFAULT '{}'
                );

                CREATE TABLE IF NOT EXISTS active_chats (
                    chat_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    first_name TEXT,
                    username TEXT,
                    started_at TEXT,
                    last_message TEXT,
                    chat_type TEXT DEFAULT 'private',
                    metadata TEXT DEFAULT '{}'
                );

                CREATE TABLE IF NOT EXISTS document_context (
                    user_id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    filename TEXT,
                    timestamp REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS group_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id TEXT NOT NULL,
                    user_id TEXT,
                    user_name TEXT,
                    text TEXT,
                    timestamp REAL NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_group_messages_chat
                    ON group_messages(chat_id, timestamp DESC);

                CREATE TABLE IF NOT EXISTS user_locations (
                    user_id TEXT PRIMARY KEY,
                    latitude REAL NOT NULL,
                    longitude REAL NOT NULL,
                    timestamp REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS inline_cache (
                    query_hash TEXT PRIMARY KEY,
                    query TEXT NOT NULL,
                    result TEXT NOT NULL,
                    timestamp REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS skill_state (
                    user_id TEXT PRIMARY KEY,
                    last_exchange TEXT DEFAULT '{}',
                    pending_action TEXT DEFAULT '{}',
                    create_state TEXT DEFAULT '{}'
                );

                CREATE TABLE IF NOT EXISTS digest_jobs (
                    chat_id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL,
                    enabled INTEGER DEFAULT 1
                );

                CREATE TABLE IF NOT EXISTS reaction_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    chat_id TEXT NOT NULL,
                    message_id INTEGER NOT NULL,
                    reactions TEXT NOT NULL,
                    sentiment TEXT NOT NULL,
                    timestamp REAL NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_reaction_feedback_user
                    ON reaction_feedback(user_id, timestamp DESC);
            """)
            conn.commit()

            # Migrate existing databases: add tts_enabled if missing
            try:
                conn.execute("ALTER TABLE user_settings ADD COLUMN tts_enabled INTEGER DEFAULT 1")
                conn.commit()
            except Exception:
                pass  # Column already exists

    def close(self):
        with self._lock:
            if self._conn:
                self._conn.close()
                self._conn = None

    # ==================== USER SETTINGS ====================

    def get_user_settings(self, user_id: str) -> dict:
        with self._lock:
            row = self._get_conn().execute(
                "SELECT * FROM user_settings WHERE user_id = ?", (user_id,)
            ).fetchone()
            return dict(row) if row else {}

    def set_user_setting(self, user_id: str, **kwargs) -> None:
        with self._lock:
            conn = self._get_conn()
            existing = conn.execute(
                "SELECT 1 FROM user_settings WHERE user_id = ?", (user_id,)
            ).fetchone()

            if existing:
                sets = ", ".join(f"{k} = ?" for k in kwargs)
                vals = list(kwargs.values()) + [user_id]
                conn.execute(
                    f"UPDATE user_settings SET {sets}, updated_at = datetime('now') WHERE user_id = ?",
                    vals,
                )
            else:
                cols = ["user_id"] + list(kwargs.keys())
                placeholders = ", ".join("?" for _ in cols)
                vals = [user_id] + list(kwargs.values())
                conn.execute(
                    f"INSERT INTO user_settings ({', '.join(cols)}) VALUES ({placeholders})",
                    vals,
                )
            conn.commit()

    def get_user_language(self, user_id: str) -> str:
        with self._lock:
            row = self._get_conn().execute(
                "SELECT language FROM user_settings WHERE user_id = ?", (user_id,)
            ).fetchone()
            return row["language"] if row else "en"

    def set_user_language(self, user_id: str, lang: str) -> None:
        self.set_user_setting(user_id, language=lang)

    def get_keyboard_enabled(self, user_id: str) -> bool:
        with self._lock:
            row = self._get_conn().execute(
                "SELECT keyboard_enabled FROM user_settings WHERE user_id = ?", (user_id,)
            ).fetchone()
            return bool(row["keyboard_enabled"]) if row else True

    def set_keyboard_enabled(self, user_id: str, enabled: bool) -> None:
        self.set_user_setting(user_id, keyboard_enabled=int(enabled))

    def get_tts_enabled(self, user_id: str) -> bool:
        with self._lock:
            row = self._get_conn().execute(
                "SELECT tts_enabled FROM user_settings WHERE user_id = ?", (user_id,)
            ).fetchone()
            return bool(row["tts_enabled"]) if row else True

    def set_tts_enabled(self, user_id: str, enabled: bool) -> None:
        self.set_user_setting(user_id, tts_enabled=int(enabled))

    # ==================== PREMIUM ====================

    def get_premium_users(self) -> Dict[str, dict]:
        with self._lock:
            rows = self._get_conn().execute("SELECT * FROM premium_users").fetchall()
            return {r["user_id"]: dict(r) for r in rows}

    def is_premium(self, user_id: str) -> bool:
        with self._lock:
            row = self._get_conn().execute(
                "SELECT 1 FROM premium_users WHERE user_id = ?", (user_id,)
            ).fetchone()
            return row is not None

    def set_premium(self, user_id: str, tier: str, transaction_id: str = "",
                    stars_amount: int = 0, metadata: dict = None) -> None:
        with self._lock:
            self._get_conn().execute(
                """INSERT OR REPLACE INTO premium_users
                   (user_id, tier, purchased_at, transaction_id, stars_amount, metadata)
                   VALUES (?, ?, datetime('now'), ?, ?, ?)""",
                (user_id, tier, transaction_id, stars_amount,
                 json.dumps(metadata or {})),
            )
            self._get_conn().commit()

    def remove_premium(self, user_id: str) -> None:
        with self._lock:
            self._get_conn().execute(
                "DELETE FROM premium_users WHERE user_id = ?", (user_id,)
            )
            self._get_conn().commit()

    # ==================== ACTIVE CHATS ====================

    def get_active_chats(self) -> Dict[str, dict]:
        with self._lock:
            rows = self._get_conn().execute("SELECT * FROM active_chats").fetchall()
            return {r["chat_id"]: dict(r) for r in rows}

    def upsert_active_chat(self, chat_id: str, user_id: str = "",
                           first_name: str = "", username: str = "",
                           chat_type: str = "private") -> None:
        with self._lock:
            self._get_conn().execute(
                """INSERT INTO active_chats (chat_id, user_id, first_name, username,
                   started_at, last_message, chat_type)
                   VALUES (?, ?, ?, ?, datetime('now'), datetime('now'), ?)
                   ON CONFLICT(chat_id) DO UPDATE SET
                   last_message = datetime('now'),
                   first_name = COALESCE(excluded.first_name, first_name),
                   username = COALESCE(excluded.username, username)""",
                (chat_id, user_id, first_name, username, chat_type),
            )
            self._get_conn().commit()

    # ==================== DOCUMENT CONTEXT ====================

    def get_doc_context(self, user_id: str, ttl: float = 1800.0) -> Optional[dict]:
        with self._lock:
            row = self._get_conn().execute(
                "SELECT text, filename, timestamp FROM document_context WHERE user_id = ?",
                (user_id,),
            ).fetchone()
            if not row:
                return None
            if time.time() - row["timestamp"] > ttl:
                self._get_conn().execute(
                    "DELETE FROM document_context WHERE user_id = ?", (user_id,)
                )
                self._get_conn().commit()
                return None
            return {"text": row["text"], "filename": row["filename"],
                    "timestamp": row["timestamp"]}

    def set_doc_context(self, user_id: str, text: str, filename: str) -> None:
        with self._lock:
            self._get_conn().execute(
                """INSERT OR REPLACE INTO document_context (user_id, text, filename, timestamp)
                   VALUES (?, ?, ?, ?)""",
                (user_id, text, filename, time.time()),
            )
            self._get_conn().commit()

    def clear_doc_context(self, user_id: str) -> None:
        with self._lock:
            self._get_conn().execute(
                "DELETE FROM document_context WHERE user_id = ?", (user_id,)
            )
            self._get_conn().commit()

    # ==================== GROUP MESSAGES ====================

    def add_group_message(self, chat_id: str, user_id: str, user_name: str,
                          text: str) -> None:
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                """INSERT INTO group_messages (chat_id, user_id, user_name, text, timestamp)
                   VALUES (?, ?, ?, ?, ?)""",
                (chat_id, user_id, user_name, text, time.time()),
            )
            # Keep only last 100 messages per group (more generous than 50 in-memory)
            conn.execute(
                """DELETE FROM group_messages WHERE chat_id = ? AND id NOT IN (
                   SELECT id FROM group_messages WHERE chat_id = ?
                   ORDER BY timestamp DESC LIMIT 100)""",
                (chat_id, chat_id),
            )
            conn.commit()

    def get_group_messages(self, chat_id: str, limit: int = 50) -> List[dict]:
        with self._lock:
            rows = self._get_conn().execute(
                """SELECT user_id, user_name, text, timestamp
                   FROM group_messages WHERE chat_id = ?
                   ORDER BY timestamp DESC LIMIT ?""",
                (chat_id, limit),
            ).fetchall()
            return [dict(r) for r in reversed(rows)]

    # ==================== USER LOCATIONS ====================

    def get_user_location(self, user_id: str) -> Optional[dict]:
        with self._lock:
            row = self._get_conn().execute(
                "SELECT latitude, longitude, timestamp FROM user_locations WHERE user_id = ?",
                (user_id,),
            ).fetchone()
            return dict(row) if row else None

    def set_user_location(self, user_id: str, lat: float, lon: float) -> None:
        with self._lock:
            self._get_conn().execute(
                """INSERT OR REPLACE INTO user_locations (user_id, latitude, longitude, timestamp)
                   VALUES (?, ?, ?, ?)""",
                (user_id, lat, lon, time.time()),
            )
            self._get_conn().commit()

    # ==================== INLINE CACHE ====================

    def get_inline_result(self, query_hash: str) -> Optional[str]:
        with self._lock:
            row = self._get_conn().execute(
                "SELECT result FROM inline_cache WHERE query_hash = ?", (query_hash,)
            ).fetchone()
            return row["result"] if row else None

    def set_inline_result(self, query_hash: str, query: str, result: str) -> None:
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                """INSERT OR REPLACE INTO inline_cache (query_hash, query, result, timestamp)
                   VALUES (?, ?, ?, ?)""",
                (query_hash, query, result, time.time()),
            )
            # Keep cache bounded at 200 entries
            conn.execute(
                """DELETE FROM inline_cache WHERE query_hash NOT IN (
                   SELECT query_hash FROM inline_cache ORDER BY timestamp DESC LIMIT 200)"""
            )
            conn.commit()

    # ==================== SKILL STATE ====================

    def get_skill_state(self, user_id: str) -> dict:
        with self._lock:
            row = self._get_conn().execute(
                "SELECT * FROM skill_state WHERE user_id = ?", (user_id,)
            ).fetchone()
            if not row:
                return {"last_exchange": {}, "pending_action": {}, "create_state": {}}
            return {
                "last_exchange": json.loads(row["last_exchange"]),
                "pending_action": json.loads(row["pending_action"]),
                "create_state": json.loads(row["create_state"]),
            }

    def set_skill_state(self, user_id: str, **kwargs) -> None:
        serialized = {k: json.dumps(v) for k, v in kwargs.items()}
        with self._lock:
            conn = self._get_conn()
            existing = conn.execute(
                "SELECT 1 FROM skill_state WHERE user_id = ?", (user_id,)
            ).fetchone()
            if existing:
                sets = ", ".join(f"{k} = ?" for k in serialized)
                conn.execute(
                    f"UPDATE skill_state SET {sets} WHERE user_id = ?",
                    list(serialized.values()) + [user_id],
                )
            else:
                defaults = {"last_exchange": "{}", "pending_action": "{}", "create_state": "{}"}
                defaults.update(serialized)
                conn.execute(
                    "INSERT INTO skill_state (user_id, last_exchange, pending_action, create_state) VALUES (?, ?, ?, ?)",
                    (user_id, defaults["last_exchange"], defaults["pending_action"], defaults["create_state"]),
                )
            conn.commit()

    # ==================== DIGEST JOBS ====================

    def get_digest_jobs(self) -> Dict[str, str]:
        with self._lock:
            rows = self._get_conn().execute(
                "SELECT chat_id, job_id FROM digest_jobs WHERE enabled = 1"
            ).fetchall()
            return {r["chat_id"]: r["job_id"] for r in rows}

    def set_digest_job(self, chat_id: str, job_id: str, enabled: bool = True) -> None:
        with self._lock:
            self._get_conn().execute(
                "INSERT OR REPLACE INTO digest_jobs (chat_id, job_id, enabled) VALUES (?, ?, ?)",
                (chat_id, job_id, int(enabled)),
            )
            self._get_conn().commit()

    # ==================== MIGRATION ====================

    def migrate_from_json(self, state_file: str = "data/messaging/telegram_state.json",
                          premium_file: str = "data/premium_users.json") -> int:
        """One-time migration from JSON files to SQLite. Returns count of migrated records."""
        count = 0
        # Migrate active chats
        state_path = Path(state_file)
        if state_path.exists():
            try:
                data = json.loads(state_path.read_text())
                for chat_id, info in data.get("active_chats", {}).items():
                    self.upsert_active_chat(
                        chat_id=chat_id,
                        user_id=info.get("user_id", ""),
                        first_name=info.get("first_name", ""),
                    )
                    count += 1
                logger.info(f"[TelegramStore] Migrated {count} active chats from JSON")
            except Exception as e:
                logger.warning(f"[TelegramStore] Failed to migrate state JSON: {e}")

        # Migrate premium users
        premium_path = Path(premium_file)
        if premium_path.exists():
            try:
                data = json.loads(premium_path.read_text())
                for user_id, info in data.items():
                    self.set_premium(
                        user_id=user_id,
                        tier=info.get("tier", "supporter"),
                        transaction_id=info.get("transaction_id", ""),
                        stars_amount=info.get("stars_amount", 0),
                    )
                    count += 1
                logger.info(f"[TelegramStore] Migrated premium users from JSON")
            except Exception as e:
                logger.warning(f"[TelegramStore] Failed to migrate premium JSON: {e}")

        return count

    # ==================== REACTION FEEDBACK ====================

    def save_reaction_feedback(self, user_id: str, chat_id: str, message_id: int,
                               reactions: str, sentiment: str):
        """Store a user's emoji reaction as feedback."""
        with self._lock:
            self._get_conn().execute(
                "INSERT INTO reaction_feedback (user_id, chat_id, message_id, reactions, sentiment, timestamp) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (user_id, chat_id, message_id, reactions, sentiment, time.time())
            )
            self._get_conn().commit()

    def get_reaction_stats(self, user_id: Optional[str] = None) -> dict:
        """Get reaction feedback statistics."""
        with self._lock:
            conn = self._get_conn()
            if user_id:
                rows = conn.execute(
                    "SELECT sentiment, COUNT(*) as cnt FROM reaction_feedback "
                    "WHERE user_id = ? GROUP BY sentiment", (user_id,)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT sentiment, COUNT(*) as cnt FROM reaction_feedback "
                    "GROUP BY sentiment"
                ).fetchall()
            return {row["sentiment"]: row["cnt"] for row in rows}
