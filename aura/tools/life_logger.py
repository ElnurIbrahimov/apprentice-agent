"""Life Logger — unified timeline of everything that happens in your digital life.

Aggregates events from all AURA sources:
  - Clipboard memory (what you copied)
  - Obsidian notes (what you wrote)
  - GitHub commits (what you built)
  - Meetings (what you discussed)
  - Ambient audio (what was said)
  - Manual entries (anything you log)

"What happened Tuesday?" → AURA synthesizes a real answer from all sources.
"Find everything about 'authentication' this month" → cross-source search.

Storage: SQLite at data/life_log.db
"""

import json
import logging
import os
import sqlite3
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

DB_PATH = Path(os.getenv("AURA_DATA_DIR", "data")) / "life_log.db"
_db_lock = threading.Lock()

SOURCES = ["clipboard", "obsidian", "github", "meeting", "audio", "calendar", "task", "manual"]
DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def _get_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS life_events (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            source      TEXT    NOT NULL,
            event_type  TEXT    NOT NULL,
            title       TEXT    NOT NULL,
            description TEXT    DEFAULT '',
            tags        TEXT    DEFAULT '[]',
            timestamp   TEXT    NOT NULL,
            date        TEXT    NOT NULL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_date ON life_events(date)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_source ON life_events(source)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_title ON life_events(title)")
    conn.commit()
    return conn


class LifeLogger:

    def log_event(
        self,
        source: str,
        event_type: str,
        title: str,
        description: str = "",
        tags: Optional[List[str]] = None,
        timestamp: Optional[str] = None,
    ) -> Dict:
        """Add an event to the life timeline."""
        now_str = timestamp or datetime.now().isoformat()
        date_str = now_str[:10]
        try:
            with _db_lock:
                conn = _get_db()
                try:
                    conn.execute(
                        "INSERT INTO life_events (source, event_type, title, description, tags, timestamp, date) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (source, event_type, title[:200], description[:1000], json.dumps(tags or []), now_str, date_str),
                    )
                    conn.commit()
                finally:
                    conn.close()
            return {"success": True, "logged": title, "source": source, "date": date_str}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_day_summary(self, date: Optional[str] = None) -> Dict:
        """Get all events for a day — organized by source."""
        target = date or datetime.now().strftime("%Y-%m-%d")
        try:
            with _db_lock:
                conn = _get_db()
                try:
                    rows = conn.execute(
                        "SELECT * FROM life_events WHERE date = ? ORDER BY timestamp",
                        (target,),
                    ).fetchall()
                finally:
                    conn.close()

            events = [dict(r) for r in rows]
            by_source: Dict[str, List] = {}
            for e in events:
                by_source.setdefault(e["source"], []).append({
                    "type": e["event_type"],
                    "title": e["title"],
                    "description": e["description"],
                    "time": e["timestamp"][11:16],
                })

            timeline = [
                {"time": e["timestamp"][11:16], "source": e["source"], "title": e["title"]}
                for e in events
            ]

            return {
                "success": True,
                "date": target,
                "day_of_week": DAY_NAMES[datetime.strptime(target, "%Y-%m-%d").weekday()] if "-" in target else "",
                "total_events": len(events),
                "sources_active": list(by_source.keys()),
                "by_source": by_source,
                "timeline": timeline,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_week_summary(self, weeks_back: int = 0) -> Dict:
        """Summarize the past 7 days."""
        base = datetime.now() - timedelta(weeks=weeks_back)
        days = [(base - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(7)]
        try:
            with _db_lock:
                conn = _get_db()
                try:
                    placeholders = ",".join("?" for _ in days)
                    rows = conn.execute(
                        f"SELECT date, source, COUNT(*) as count FROM life_events WHERE date IN ({placeholders}) GROUP BY date, source ORDER BY date DESC",
                        days,
                    ).fetchall()
                    total_row = conn.execute(
                        f"SELECT COUNT(*) as n FROM life_events WHERE date IN ({placeholders})",
                        days,
                    ).fetchone()
                finally:
                    conn.close()

            by_day: Dict[str, Dict] = {}
            for r in rows:
                by_day.setdefault(r["date"], {})[r["source"]] = r["count"]

            return {
                "success": True,
                "week": [{"date": d, "day": DAY_NAMES[datetime.strptime(d, "%Y-%m-%d").weekday()], "sources": by_day.get(d, {}), "total": sum(by_day.get(d, {}).values())} for d in days],
                "total_events": total_row["n"] if total_row else 0,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def search_timeline(self, query: str, days: int = 30, source: Optional[str] = None) -> Dict:
        """Full-text search across the timeline."""
        since = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        q = f"%{query}%"
        try:
            with _db_lock:
                conn = _get_db()
                try:
                    if source:
                        rows = conn.execute(
                            "SELECT * FROM life_events WHERE date >= ? AND source = ? AND (title LIKE ? OR description LIKE ?) ORDER BY timestamp DESC LIMIT 50",
                            (since, source, q, q),
                        ).fetchall()
                    else:
                        rows = conn.execute(
                            "SELECT * FROM life_events WHERE date >= ? AND (title LIKE ? OR description LIKE ?) ORDER BY timestamp DESC LIMIT 50",
                            (since, q, q),
                        ).fetchall()
                finally:
                    conn.close()

            return {
                "success": True,
                "query": query,
                "since": since,
                "source_filter": source,
                "count": len(rows),
                "results": [
                    {
                        "time": r["timestamp"][:16],
                        "source": r["source"],
                        "type": r["event_type"],
                        "title": r["title"],
                        "description": r["description"][:150],
                    }
                    for r in rows
                ],
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def pull_from_sources(self, tools: Dict) -> Dict:
        """Pull recent events from other AURA tools into the timeline."""
        pulled: Dict[str, int] = {}

        # GitHub commits
        if "github" in tools:
            try:
                result = tools["github"].recent_commits(limit=20)
                for c in result.get("commits", []):
                    self.log_event(
                        source="github",
                        event_type="commit",
                        title=c.get("message", "commit")[:100],
                        description=c.get("sha", ""),
                        tags=["code"],
                    )
                    pulled["github"] = pulled.get("github", 0) + 1
            except Exception as e:
                logger.debug(f"[LifeLogger] GitHub pull failed: {e}")

        # Clipboard recent items
        if "clipboard" in tools:
            try:
                result = tools["clipboard"].list_recent(limit=20)
                for item in result.get("entries", []):
                    self.log_event(
                        source="clipboard",
                        event_type="copy",
                        title=item.get("content", "")[:80],
                        description=item.get("category", ""),
                        timestamp=item.get("timestamp"),
                    )
                    pulled["clipboard"] = pulled.get("clipboard", 0) + 1
            except Exception as e:
                logger.debug(f"[LifeLogger] Clipboard pull failed: {e}")

        # Meeting records
        meetings_dir = Path(os.getenv("AURA_DATA_DIR", "data")) / "meetings"
        if meetings_dir.exists():
            try:
                for path in sorted(meetings_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:10]:
                    with open(path) as f:
                        m = json.load(f)
                    self.log_event(
                        source="meeting",
                        event_type="meeting",
                        title=m.get("title", "Meeting"),
                        description=m.get("summary", "")[:200],
                        timestamp=m.get("started_at"),
                        tags=m.get("key_topics", []),
                    )
                    pulled["meeting"] = pulled.get("meeting", 0) + 1
            except Exception as e:
                logger.debug(f"[LifeLogger] Meeting pull failed: {e}")

        return {"success": True, "pulled": pulled, "total": sum(pulled.values())}

    def get_stats(self) -> Dict:
        """Overall life log statistics."""
        try:
            with _db_lock:
                conn = _get_db()
                try:
                    total = conn.execute("SELECT COUNT(*) as n FROM life_events").fetchone()["n"]
                    by_source = conn.execute(
                        "SELECT source, COUNT(*) as count FROM life_events GROUP BY source ORDER BY count DESC"
                    ).fetchall()
                    oldest = conn.execute("SELECT date FROM life_events ORDER BY timestamp LIMIT 1").fetchone()
                finally:
                    conn.close()

            return {
                "success": True,
                "total_events": total,
                "tracking_since": oldest["date"] if oldest else "no events",
                "by_source": {r["source"]: r["count"] for r in by_source},
            }
        except Exception as e:
            return {"success": False, "error": str(e)}


class LifeLoggerTool:
    """Unified life timeline — 'What happened Tuesday?' synthesizes notes, commits, meetings, clipboard into one answer."""

    name = "life_logger"
    description = "Unified life timeline — cross-source search and daily/weekly summaries across clipboard, notes, commits, meetings, audio."

    def __init__(self):
        self._logger = LifeLogger()
        self._tools: Dict = {}

    def set_tools(self, tools: Dict):
        """Inject sibling tool references for cross-source sync."""
        self._tools = tools

    def execute(self, action: str, **kwargs) -> Dict:
        a = action.lower().strip()

        if "log" in a or "add" in a or "record" in a:
            return self._logger.log_event(
                source=kwargs.get("source", "manual"),
                event_type=kwargs.get("event_type", "manual"),
                title=kwargs.get("title") or kwargs.get("text") or action,
                description=kwargs.get("description", ""),
                tags=kwargs.get("tags", []),
            )
        if "week" in a:
            return self._logger.get_week_summary(kwargs.get("weeks_back", 0))
        if "search" in a or "find" in a:
            return self._logger.search_timeline(
                kwargs.get("query") or action,
                kwargs.get("days", 30),
                kwargs.get("source"),
            )
        if "pull" in a or "sync" in a or "import" in a:
            return self._logger.pull_from_sources(self._tools)
        if "stat" in a:
            return self._logger.get_stats()

        # "what happened today / tuesday / yesterday"
        date = kwargs.get("date")
        if not date:
            # Try to parse day name from action
            for i, day in enumerate(DAY_NAMES):
                if day.lower() in a:
                    # Find the most recent occurrence of that day
                    today = datetime.now()
                    days_back = (today.weekday() - i) % 7
                    date = (today - timedelta(days=days_back)).strftime("%Y-%m-%d")
                    break
            if "yesterday" in a:
                date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

        return self._logger.get_day_summary(date)
