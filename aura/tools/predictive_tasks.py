"""Predictive Task Engine — learns your workflow patterns and proactively suggests next steps.

Tracks which tools you use at what times, discovers recurring patterns,
and predicts what you'll want to do before you ask.

'It's Friday afternoon — you usually run the weekly GitHub summary now.'

No external dependencies — SQLite only.
The task_scheduler integration means predictions can auto-trigger suggested tasks.

Storage: data/task_events.db
"""

import json
import logging
import os
import sqlite3
import threading
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

DB_PATH = Path(os.getenv("AURA_DATA_DIR", "data")) / "task_events.db"
_db_lock = threading.Lock()

DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def _get_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS task_events (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            tool        TEXT    NOT NULL,
            action      TEXT    NOT NULL,
            prompt_hint TEXT,
            hour        INTEGER NOT NULL,
            day_of_week INTEGER NOT NULL,
            timestamp   TEXT    NOT NULL
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_tool_time ON task_events(tool, hour, day_of_week)"
    )
    conn.commit()
    return conn


class PredictiveTaskEngine:
    """Track, analyze, and predict tool usage patterns."""

    def __init__(self):
        try:
            _get_db().close()
        except Exception as e:
            logger.warning(f"[PredictiveTasks] DB init error: {e}")

    # ------------------------------------------------------------------ #
    # Logging
    # ------------------------------------------------------------------ #

    def log_event(self, tool: str, action: str, prompt_hint: str = "") -> Dict:
        """Record a tool usage event for pattern learning."""
        if not tool:
            return {"success": False, "error": "tool name required"}
        now = datetime.now()
        try:
            with _db_lock:
                conn = _get_db()
                conn.execute(
                    "INSERT INTO task_events (tool, action, prompt_hint, hour, day_of_week, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
                    (tool, action, prompt_hint[:100], now.hour, now.weekday(), now.isoformat()),
                )
                conn.commit()
                conn.close()
            return {"success": True, "logged": f"{tool}.{action}", "at": now.strftime("%A %H:%M")}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ------------------------------------------------------------------ #
    # Analysis
    # ------------------------------------------------------------------ #

    def get_patterns(self, min_occurrences: int = 2) -> Dict:
        """Return recurring (tool, action, day, hour) patterns."""
        try:
            with _db_lock:
                conn = _get_db()
                rows = conn.execute(
                    """
                    SELECT tool, action, hour, day_of_week, COUNT(*) as count
                    FROM task_events
                    GROUP BY tool, action, hour, day_of_week
                    HAVING count >= ?
                    ORDER BY count DESC
                    LIMIT 50
                    """,
                    (min_occurrences,),
                ).fetchall()
                conn.close()

            patterns = [
                {
                    "tool": r["tool"],
                    "action": r["action"],
                    "hour": r["hour"],
                    "day": DAY_NAMES[r["day_of_week"]],
                    "occurrences": r["count"],
                    "label": f"{r['tool']}.{r['action']} — {DAY_NAMES[r['day_of_week']]} ~{r['hour']:02d}:00 ({r['count']}×)",
                }
                for r in rows
            ]
            return {"success": True, "count": len(patterns), "patterns": patterns}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_predictions(self, window_hours: int = 1) -> Dict:
        """Predict tasks relevant right now based on current time."""
        now = datetime.now()
        current_hour = now.hour
        current_dow = now.weekday()
        # Look ±window_hours around current time
        hours = list({(current_hour + i) % 24 for i in range(-window_hours, window_hours + 1)})
        placeholders = ",".join("?" for _ in hours)
        try:
            with _db_lock:
                conn = _get_db()
                # Day-specific predictions (higher confidence)
                day_rows = conn.execute(
                    f"""
                    SELECT tool, action, hour, COUNT(*) as count
                    FROM task_events
                    WHERE hour IN ({placeholders}) AND day_of_week = ?
                    GROUP BY tool, action, hour
                    HAVING count >= 2
                    ORDER BY count DESC LIMIT 5
                    """,
                    (*hours, current_dow),
                ).fetchall()
                # General time predictions (any day)
                any_rows = conn.execute(
                    f"""
                    SELECT tool, action, hour, COUNT(*) as count
                    FROM task_events
                    WHERE hour IN ({placeholders})
                    GROUP BY tool, action, hour
                    HAVING count >= 3
                    ORDER BY count DESC LIMIT 5
                    """,
                    hours,
                ).fetchall()
                conn.close()

            predictions = []
            seen = set()
            for r, is_day_specific in [(x, True) for x in day_rows] + [(x, False) for x in any_rows]:
                key = (r["tool"], r["action"])
                if key in seen:
                    continue
                seen.add(key)
                confidence = min(r["count"] / 8.0, 1.0) * (1.2 if is_day_specific else 1.0)
                confidence = min(confidence, 1.0)
                day_label = DAY_NAMES[current_dow] if is_day_specific else "most days"
                predictions.append(
                    {
                        "tool": r["tool"],
                        "action": r["action"],
                        "confidence": round(confidence, 2),
                        "occurrences": r["count"],
                        "suggestion": f"You usually run {r['tool']}.{r['action']} on {day_label} around {r['hour']:02d}:00",
                    }
                )
            predictions.sort(key=lambda x: x["confidence"], reverse=True)
            return {
                "success": True,
                "now": now.strftime("%A %H:%M"),
                "count": len(predictions),
                "predictions": predictions,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_stats(self) -> Dict:
        """Overall usage statistics."""
        try:
            with _db_lock:
                conn = _get_db()
                total = conn.execute("SELECT COUNT(*) as n FROM task_events").fetchone()["n"]
                top_tools = conn.execute(
                    "SELECT tool, COUNT(*) as count FROM task_events GROUP BY tool ORDER BY count DESC LIMIT 10"
                ).fetchall()
                recent = conn.execute(
                    "SELECT tool, action, timestamp FROM task_events ORDER BY id DESC LIMIT 10"
                ).fetchall()
                # Most active hour
                peak_hour = conn.execute(
                    "SELECT hour, COUNT(*) as count FROM task_events GROUP BY hour ORDER BY count DESC LIMIT 1"
                ).fetchone()
                conn.close()

            return {
                "success": True,
                "total_events": total,
                "peak_hour": f"{peak_hour['hour']:02d}:00" if peak_hour else "unknown",
                "top_tools": [{"tool": r["tool"], "uses": r["count"]} for r in top_tools],
                "recent": [{"tool": r["tool"], "action": r["action"], "at": r["timestamp"][:16]} for r in recent],
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_weekly_heatmap(self) -> Dict:
        """Usage heatmap: day × hour frequency."""
        try:
            with _db_lock:
                conn = _get_db()
                rows = conn.execute(
                    "SELECT day_of_week, hour, COUNT(*) as count FROM task_events GROUP BY day_of_week, hour"
                ).fetchall()
                conn.close()

            heatmap: Dict[str, Dict[int, int]] = {d: {} for d in DAY_NAMES}
            for r in rows:
                heatmap[DAY_NAMES[r["day_of_week"]]][r["hour"]] = r["count"]

            return {"success": True, "heatmap": heatmap}
        except Exception as e:
            return {"success": False, "error": str(e)}


class PredictiveTaskTool:
    """Learn workflow patterns and predict upcoming tasks — 'It's Friday, you usually run the weekly GitHub summary now.'"""

    name = "predictive_tasks"
    description = "Learn workflow patterns from tool usage and predict upcoming tasks — proactive suggestions before you ask."

    def __init__(self):
        self._engine = PredictiveTaskEngine()

    def execute(self, action: str, **kwargs) -> Dict:
        a = action.lower().strip()
        if "log" in a or "record" in a or "track" in a:
            return self._engine.log_event(
                kwargs.get("tool", ""),
                kwargs.get("action_name") or kwargs.get("tool_action", ""),
                kwargs.get("prompt", ""),
            )
        if "pattern" in a:
            return self._engine.get_patterns(kwargs.get("min_occurrences", 2))
        if "predict" in a or "suggest" in a or "now" in a:
            return self._engine.get_predictions(kwargs.get("window_hours", 1))
        if "stat" in a or "usage" in a or "history" in a:
            return self._engine.get_stats()
        if "heatmap" in a:
            return self._engine.get_weekly_heatmap()
        # Default: show predictions
        return self._engine.get_predictions()
