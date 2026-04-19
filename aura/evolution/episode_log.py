"""
Skill Episode Log — records every live skill invocation so evolution can
later score them against real outcomes instead of LLM-invented rubrics.

Writes to the same SQLite DB the strategy bandit uses (`aura_meta.db`) so that
reactions, action-button events, and skill invocations all share a queryable
surface joinable by `request_id`.

The log is the data layer underneath GEPA's consolidation pass. Nothing here
proposes mutations — it only captures what was invoked with what text.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from aura.paths import AURA_DATA_DIR

logger = logging.getLogger(__name__)

_DEFAULT_USER_INPUT_CHARS = 500
_DEFAULT_RESPONSE_CHARS = 500


def _aura_meta_db_path() -> str:
    """Resolve the shared meta DB path, preferring env override used by the bandit."""
    env_dir = os.getenv("AURA_DATA_DIR")
    base = Path(env_dir) if env_dir else AURA_DATA_DIR
    base.mkdir(parents=True, exist_ok=True)
    return str(base / "aura_meta.db")


def procedure_hash(procedure_text: str) -> str:
    """Stable identifier for a procedure version — sha256 prefix."""
    return hashlib.sha256(procedure_text.encode("utf-8")).hexdigest()[:16]


@dataclass
class EpisodeRow:
    """One logged invocation; what the DB actually stores."""
    episode_id: int
    request_id: str
    skill_id: str
    procedure_hash: str
    user_input: str
    response: str
    tools_called: List[str]
    invoked_at: float


@dataclass
class OutcomeRow:
    """One outcome signal attached to an episode."""
    episode_id: int
    signal_kind: str
    score: float
    confidence: float
    recorded_at: float


@dataclass
class LabeledEpisode:
    """An episode plus its aggregated outcome signals — what evolution reads."""
    episode: EpisodeRow
    outcomes: List[OutcomeRow]

    def composite_score(self) -> Optional[float]:
        """Confidence-weighted mean, or None when the episode has no signal."""
        if not self.outcomes:
            return None
        total_w = sum(o.confidence for o in self.outcomes)
        if total_w <= 0:
            return None
        weighted = sum(o.score * o.confidence for o in self.outcomes)
        return max(0.0, min(1.0, weighted / total_w))


class SkillEpisodeLog:
    """Thread-safe SQLite-backed writer/reader for skill invocation episodes."""

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS skill_episodes (
        episode_id     INTEGER PRIMARY KEY AUTOINCREMENT,
        request_id     TEXT NOT NULL,
        skill_id       TEXT NOT NULL,
        procedure_hash TEXT NOT NULL,
        user_input     TEXT NOT NULL,
        response       TEXT NOT NULL,
        tools_called   TEXT,
        invoked_at     REAL NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_episodes_skill ON skill_episodes(skill_id, invoked_at DESC);
    CREATE INDEX IF NOT EXISTS idx_episodes_request ON skill_episodes(request_id);

    CREATE TABLE IF NOT EXISTS skill_episode_outcomes (
        episode_id     INTEGER NOT NULL,
        signal_kind    TEXT NOT NULL,
        score          REAL NOT NULL,
        confidence     REAL NOT NULL,
        recorded_at    REAL NOT NULL,
        FOREIGN KEY (episode_id) REFERENCES skill_episodes(episode_id)
    );
    CREATE INDEX IF NOT EXISTS idx_outcomes_episode ON skill_episode_outcomes(episode_id);
    """

    def __init__(self, db_path: Optional[str] = None):
        self._db_path = db_path or _aura_meta_db_path()
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=5.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def _init_db(self) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.executescript(self._SCHEMA)
                conn.commit()
            finally:
                conn.close()

    def log(
        self,
        *,
        request_id: str,
        skill_ids: List[str],
        user_input: str,
        response: str,
        procedures: Dict[str, str],
        tools_called: Optional[List[str]] = None,
        invoked_at: Optional[float] = None,
    ) -> List[int]:
        """Insert one row per invoked skill. Returns the new episode_ids.

        Safe to call from a background executor — catches all sqlite errors and
        logs them rather than propagating (never want logging failures to break
        a live conversation).
        """
        if not skill_ids or not request_id:
            return []

        ts = invoked_at or time.time()
        tools_json = json.dumps(tools_called) if tools_called else None
        truncated_input = user_input[:_DEFAULT_USER_INPUT_CHARS]
        truncated_response = response[:_DEFAULT_RESPONSE_CHARS]

        ids: List[int] = []
        try:
            with self._lock:
                conn = self._connect()
                try:
                    for skill_id in skill_ids:
                        proc = procedures.get(skill_id, "")
                        phash = procedure_hash(proc) if proc else ""
                        cur = conn.execute(
                            """
                            INSERT INTO skill_episodes
                              (request_id, skill_id, procedure_hash, user_input,
                               response, tools_called, invoked_at)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                request_id, skill_id, phash,
                                truncated_input, truncated_response,
                                tools_json, ts,
                            ),
                        )
                        if cur.lastrowid is not None:
                            ids.append(cur.lastrowid)
                    conn.commit()
                finally:
                    conn.close()
        except sqlite3.Error as e:
            logger.warning("skill episode log write failed: %s", e)
            return []

        return ids

    def add_outcome(
        self,
        *,
        episode_id: int,
        signal_kind: str,
        score: float,
        confidence: float,
        recorded_at: Optional[float] = None,
    ) -> bool:
        """Attach one outcome signal to an existing episode."""
        ts = recorded_at or time.time()
        score = max(0.0, min(1.0, float(score)))
        confidence = max(0.0, min(1.0, float(confidence)))
        try:
            with self._lock:
                conn = self._connect()
                try:
                    conn.execute(
                        """
                        INSERT INTO skill_episode_outcomes
                          (episode_id, signal_kind, score, confidence, recorded_at)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (episode_id, signal_kind, score, confidence, ts),
                    )
                    conn.commit()
                    return True
                finally:
                    conn.close()
        except sqlite3.Error as e:
            logger.warning("skill outcome write failed: %s", e)
            return False

    def episodes_for_request(self, request_id: str) -> List[EpisodeRow]:
        """All episodes logged under this request_id (one per matched skill)."""
        try:
            with self._lock:
                conn = self._connect()
                try:
                    rows = conn.execute(
                        """
                        SELECT episode_id, request_id, skill_id, procedure_hash,
                               user_input, response, tools_called, invoked_at
                        FROM skill_episodes
                        WHERE request_id = ?
                        ORDER BY invoked_at ASC
                        """,
                        (request_id,),
                    ).fetchall()
                finally:
                    conn.close()
        except sqlite3.Error as e:
            logger.warning("episodes_for_request read failed: %s", e)
            return []

        return [
            EpisodeRow(
                episode_id=r[0], request_id=r[1], skill_id=r[2],
                procedure_hash=r[3], user_input=r[4], response=r[5],
                tools_called=json.loads(r[6]) if r[6] else [],
                invoked_at=r[7],
            )
            for r in rows
        ]

    def labeled_episodes_for_skill(
        self,
        skill_id: str,
        window_days: int = 14,
        limit: int = 200,
    ) -> List[LabeledEpisode]:
        """Episodes for this skill within window, with attached outcomes.

        Only returns episodes that have at least one outcome signal — the
        caller wants labeled data for training/eval. Unlabeled episodes are
        filtered out here.
        """
        cutoff = time.time() - window_days * 86400.0
        try:
            with self._lock:
                conn = self._connect()
                try:
                    ep_rows = conn.execute(
                        """
                        SELECT DISTINCT e.episode_id, e.request_id, e.skill_id,
                               e.procedure_hash, e.user_input, e.response,
                               e.tools_called, e.invoked_at
                        FROM skill_episodes e
                        INNER JOIN skill_episode_outcomes o
                          ON o.episode_id = e.episode_id
                        WHERE e.skill_id = ?
                          AND e.invoked_at >= ?
                        ORDER BY e.invoked_at DESC
                        LIMIT ?
                        """,
                        (skill_id, cutoff, limit),
                    ).fetchall()

                    episodes = {
                        r[0]: EpisodeRow(
                            episode_id=r[0], request_id=r[1], skill_id=r[2],
                            procedure_hash=r[3], user_input=r[4], response=r[5],
                            tools_called=json.loads(r[6]) if r[6] else [],
                            invoked_at=r[7],
                        )
                        for r in ep_rows
                    }

                    if not episodes:
                        return []

                    qmarks = ",".join("?" * len(episodes))
                    out_rows = conn.execute(
                        f"""
                        SELECT episode_id, signal_kind, score, confidence, recorded_at
                        FROM skill_episode_outcomes
                        WHERE episode_id IN ({qmarks})
                        """,
                        list(episodes.keys()),
                    ).fetchall()
                finally:
                    conn.close()
        except sqlite3.Error as e:
            logger.warning("labeled_episodes_for_skill read failed: %s", e)
            return []

        by_episode: Dict[int, List[OutcomeRow]] = {eid: [] for eid in episodes}
        for r in out_rows:
            by_episode[r[0]].append(OutcomeRow(
                episode_id=r[0], signal_kind=r[1], score=r[2],
                confidence=r[3], recorded_at=r[4],
            ))

        return [
            LabeledEpisode(episode=ep, outcomes=by_episode.get(ep.episode_id, []))
            for ep in episodes.values()
        ]

    def count_for_skill(self, skill_id: str, window_days: int = 14) -> int:
        """Count of labeled (outcome-present) episodes for a skill in the window."""
        cutoff = time.time() - window_days * 86400.0
        try:
            with self._lock:
                conn = self._connect()
                try:
                    row = conn.execute(
                        """
                        SELECT COUNT(DISTINCT e.episode_id)
                        FROM skill_episodes e
                        INNER JOIN skill_episode_outcomes o
                          ON o.episode_id = e.episode_id
                        WHERE e.skill_id = ?
                          AND e.invoked_at >= ?
                        """,
                        (skill_id, cutoff),
                    ).fetchone()
                finally:
                    conn.close()
        except sqlite3.Error as e:
            logger.warning("count_for_skill read failed: %s", e)
            return 0
        return int(row[0]) if row else 0


# Module-level singleton — every caller sees the same DB handle + lock.
_log_singleton: Optional[SkillEpisodeLog] = None
_singleton_lock = threading.Lock()


def get_episode_log() -> SkillEpisodeLog:
    """Return the process-wide episode log. Lazy-init on first call."""
    global _log_singleton
    if _log_singleton is None:
        with _singleton_lock:
            if _log_singleton is None:
                _log_singleton = SkillEpisodeLog()
    return _log_singleton
