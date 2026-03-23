"""Merkle hash-chain audit trail for agent actions.

Stolen from OpenFang: every agent action is cryptographically linked to
the previous one, forming an append-only tamper-evident chain.

If Aura gains real autonomy (Hands), provable action history is essential.
"""

import hashlib
import json
import logging
import os
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# Genesis hash (first entry in the chain)
_GENESIS_HASH = "0" * 64


@dataclass
class AuditEntry:
    """A single entry in the audit chain."""
    timestamp: float
    action_type: str          # "tool_call", "hand_event", "memory_write", "model_call", etc.
    action_data: str          # JSON-encoded action details
    agent_id: str             # Which agent/hand performed the action
    session_id: str           # Conversation/session context
    prev_hash: str            # Hash of previous entry (chain link)
    entry_hash: str = ""      # SHA-256 of this entry (computed on creation)
    entry_id: int = 0         # Auto-increment ID from SQLite

    def compute_hash(self) -> str:
        """Compute SHA-256 hash of this entry."""
        payload = f"{self.prev_hash}|{self.timestamp}|{self.action_type}|{self.action_data}|{self.agent_id}|{self.session_id}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class AuditChain:
    """Append-only Merkle hash-chain audit log.

    Every entry's hash depends on the previous entry's hash,
    creating a tamper-evident chain. If any entry is modified,
    all subsequent hashes become invalid.
    """

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            data_dir = os.environ.get("AURA_DATA_DIR", "data")
            os.makedirs(data_dir, exist_ok=True)
            db_path = os.path.join(data_dir, "audit_chain.db")

        self._db_path = db_path
        self._lock = threading.Lock()
        self._last_hash = _GENESIS_HASH
        self._init_db()
        self._load_last_hash()

    def _init_db(self):
        """Create audit table if it doesn't exist. No UPDATE/DELETE triggers."""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_chain (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    action_type TEXT NOT NULL,
                    action_data TEXT NOT NULL,
                    agent_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    prev_hash TEXT NOT NULL,
                    entry_hash TEXT NOT NULL UNIQUE
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_chain(timestamp)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_audit_agent ON audit_chain(agent_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_audit_session ON audit_chain(session_id)
            """)
            # Trigger: prevent UPDATE on audit_chain (append-only)
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS prevent_audit_update
                BEFORE UPDATE ON audit_chain
                BEGIN
                    SELECT RAISE(ABORT, 'audit_chain is append-only: UPDATE not allowed');
                END
            """)
            # Trigger: prevent DELETE on audit_chain (append-only)
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS prevent_audit_delete
                BEFORE DELETE ON audit_chain
                BEGIN
                    SELECT RAISE(ABORT, 'audit_chain is append-only: DELETE not allowed');
                END
            """)
            conn.commit()

    def _load_last_hash(self):
        """Load the hash of the most recent entry (for chaining)."""
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute(
                "SELECT entry_hash FROM audit_chain ORDER BY id DESC LIMIT 1"
            ).fetchone()
            if row:
                self._last_hash = row[0]

    def append(
        self,
        action_type: str,
        action_data: dict,
        agent_id: str = "main",
        session_id: str = "default",
    ) -> AuditEntry:
        """Append a new entry to the chain. Thread-safe."""
        with self._lock:
            entry = AuditEntry(
                timestamp=time.time(),
                action_type=action_type,
                action_data=json.dumps(action_data, default=str, ensure_ascii=False),
                agent_id=agent_id,
                session_id=session_id,
                prev_hash=self._last_hash,
            )
            entry.entry_hash = entry.compute_hash()

            with sqlite3.connect(self._db_path) as conn:
                cursor = conn.execute(
                    """INSERT INTO audit_chain
                       (timestamp, action_type, action_data, agent_id, session_id, prev_hash, entry_hash)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (entry.timestamp, entry.action_type, entry.action_data,
                     entry.agent_id, entry.session_id, entry.prev_hash, entry.entry_hash),
                )
                entry.entry_id = cursor.lastrowid
                conn.commit()

            self._last_hash = entry.entry_hash
            return entry

    def verify(self, limit: Optional[int] = None) -> tuple[bool, int, Optional[str]]:
        """Verify the integrity of the entire chain.

        Returns:
            (is_valid, entries_checked, error_message)
        """
        with sqlite3.connect(self._db_path) as conn:
            query = "SELECT id, timestamp, action_type, action_data, agent_id, session_id, prev_hash, entry_hash FROM audit_chain ORDER BY id ASC"
            if limit:
                query += f" LIMIT {int(limit)}"
            rows = conn.execute(query).fetchall()

        if not rows:
            return True, 0, None

        prev_hash = _GENESIS_HASH
        for row in rows:
            entry_id, ts, action_type, action_data, agent_id, session_id, stored_prev, stored_hash = row

            # Check chain link
            if stored_prev != prev_hash:
                return False, entry_id, f"Entry {entry_id}: prev_hash mismatch (expected {prev_hash[:16]}..., got {stored_prev[:16]}...)"

            # Recompute hash
            entry = AuditEntry(
                timestamp=ts,
                action_type=action_type,
                action_data=action_data,
                agent_id=agent_id,
                session_id=session_id,
                prev_hash=stored_prev,
            )
            computed = entry.compute_hash()
            if computed != stored_hash:
                return False, entry_id, f"Entry {entry_id}: hash mismatch (computed {computed[:16]}..., stored {stored_hash[:16]}...)"

            prev_hash = stored_hash

        return True, len(rows), None

    def tail(self, n: int = 20) -> list[AuditEntry]:
        """Get the last N entries."""
        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute(
                "SELECT id, timestamp, action_type, action_data, agent_id, session_id, prev_hash, entry_hash "
                "FROM audit_chain ORDER BY id DESC LIMIT ?",
                (n,),
            ).fetchall()

        entries = []
        for row in reversed(rows):
            entries.append(AuditEntry(
                entry_id=row[0],
                timestamp=row[1],
                action_type=row[2],
                action_data=row[3],
                agent_id=row[4],
                session_id=row[5],
                prev_hash=row[6],
                entry_hash=row[7],
            ))
        return entries

    def count(self) -> int:
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute("SELECT COUNT(*) FROM audit_chain").fetchone()
            return row[0] if row else 0

    def search(self, action_type: Optional[str] = None, agent_id: Optional[str] = None,
               since: Optional[float] = None, limit: int = 50) -> list[AuditEntry]:
        """Search audit entries with filters."""
        conditions = []
        params = []

        if action_type:
            conditions.append("action_type = ?")
            params.append(action_type)
        if agent_id:
            conditions.append("agent_id = ?")
            params.append(agent_id)
        if since:
            conditions.append("timestamp >= ?")
            params.append(since)

        where = " AND ".join(conditions) if conditions else "1=1"
        query = f"SELECT id, timestamp, action_type, action_data, agent_id, session_id, prev_hash, entry_hash FROM audit_chain WHERE {where} ORDER BY id DESC LIMIT ?"
        params.append(limit)

        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute(query, params).fetchall()

        return [
            AuditEntry(
                entry_id=r[0], timestamp=r[1], action_type=r[2], action_data=r[3],
                agent_id=r[4], session_id=r[5], prev_hash=r[6], entry_hash=r[7],
            )
            for r in reversed(rows)
        ]


# Global singleton
_chain: Optional[AuditChain] = None
_chain_lock = threading.Lock()


def get_audit_chain() -> AuditChain:
    global _chain
    if _chain is None:
        with _chain_lock:
            if _chain is None:
                _chain = AuditChain()
    return _chain
