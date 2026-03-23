"""Predictive Task Engine — learns your workflow patterns and proactively suggests next steps.

Tracks which tools you use at what times, discovers recurring patterns,
and predicts what you'll want to do before you ask.

SOTA features: Markov chain transitions, exponential decay weighting, Bayesian
confidence (Laplace smoothing), user feedback loop, context-aware filtering,
combined prediction pipeline merging time + sequence + context + feedback signals.

No external dependencies — SQLite only.  Storage: data/task_events.db
"""

import logging
import os
import sqlite3
import threading
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Tuple

logger = logging.getLogger(__name__)

DB_PATH = Path(os.getenv("AURA_DATA_DIR", "data")) / "task_events.db"
_db_lock = threading.Lock()
DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def _get_db() -> sqlite3.Connection:
    """Open (and bootstrap) the task_events database."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("""CREATE TABLE IF NOT EXISTS task_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT, tool TEXT NOT NULL,
        action TEXT NOT NULL, prompt_hint TEXT, hour INTEGER NOT NULL,
        day_of_week INTEGER NOT NULL, timestamp TEXT NOT NULL)""")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_tool_time ON task_events(tool, hour, day_of_week)")
    # Additive migration: context_hint column
    try:
        conn.execute("ALTER TABLE task_events ADD COLUMN context_hint TEXT DEFAULT ''")
    except Exception:
        pass  # Column already exists
    # Markov transition tracking
    conn.execute("""CREATE TABLE IF NOT EXISTS tool_transitions (
        id INTEGER PRIMARY KEY AUTOINCREMENT, from_tool TEXT NOT NULL,
        from_action TEXT NOT NULL, to_tool TEXT NOT NULL,
        to_action TEXT NOT NULL, timestamp TEXT NOT NULL)""")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_transitions_from ON tool_transitions(from_tool, from_action)")
    # User feedback on predictions
    conn.execute("""CREATE TABLE IF NOT EXISTS prediction_feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT, prediction_id TEXT NOT NULL,
        tool TEXT NOT NULL, action TEXT NOT NULL,
        accepted INTEGER NOT NULL DEFAULT 0, timestamp TEXT NOT NULL)""")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_tool ON prediction_feedback(tool, action)")
    conn.commit()
    return conn


class PredictiveTaskEngine:
    """Track, analyze, and predict tool usage patterns."""

    def __init__(self):
        self._last_event: Optional[Tuple[str, str]] = None
        try:
            _get_db().close()
        except Exception as e:
            logger.warning(f"[PredictiveTasks] DB init error: {e}")

    # ---- Internal helpers ---- #

    def _record_transition(self, tool: str, action: str) -> None:
        """Record a Markov transition from the previous event to the current one."""
        if self._last_event is None:
            self._last_event = (tool, action)
            return
        from_tool, from_action = self._last_event
        try:
            with _db_lock:
                conn = _get_db()
                try:
                    conn.execute(
                        "INSERT INTO tool_transitions (from_tool,from_action,to_tool,to_action,timestamp) VALUES (?,?,?,?,?)",
                        (from_tool, from_action, tool, action, datetime.now().isoformat()))
                    conn.commit()
                finally:
                    conn.close()
        except Exception as e:
            logger.warning(f"[PredictiveTasks] Transition record error: {e}")
        self._last_event = (tool, action)

    @staticmethod
    def _apply_decay(rows: List[Dict], lambda_: float = 0.95) -> List[Tuple[Dict, float]]:
        """Exponential decay: weight = lambda_ ^ days_since_event. Returns [(row, weight)]."""
        now = datetime.now()
        result = []
        for row in rows:
            ts_str = row.get("timestamp") or row.get("ts") or ""
            try:
                days_ago = max((now - datetime.fromisoformat(ts_str)).total_seconds() / 86400.0, 0.0)
            except (ValueError, TypeError):
                days_ago = 30.0
            result.append((row, lambda_ ** days_ago))
        return result

    def _feedback_weight(self, tool: str, action: str) -> float:
        """Multiplier from feedback: >0.7 accept rate → 1.2, <0.3 → 0.5, else 1.0."""
        try:
            with _db_lock:
                conn = _get_db()
                try:
                    row = conn.execute(
                        "SELECT COUNT(*) as total, SUM(accepted) as acc FROM prediction_feedback WHERE tool=? AND action=?",
                        (tool, action)).fetchone()
                finally:
                    conn.close()
            total = row["total"] or 0
            acc = row["acc"] or 0
            if total < 3:
                return 1.0
            rate = acc / total
            return 1.2 if rate > 0.7 else (0.5 if rate < 0.3 else 1.0)
        except Exception:
            return 1.0

    # ---- Logging ---- #

    def log_event(self, tool: str, action: str, prompt_hint: str = "", context_hint: str = "") -> Dict:
        """Record a tool usage event for pattern learning."""
        if not tool:
            return {"success": False, "error": "tool name required"}
        now = datetime.now()
        try:
            with _db_lock:
                conn = _get_db()
                try:
                    conn.execute(
                        "INSERT INTO task_events (tool,action,prompt_hint,hour,day_of_week,timestamp,context_hint) VALUES (?,?,?,?,?,?,?)",
                        (tool, action, prompt_hint[:100], now.hour, now.weekday(), now.isoformat(), (context_hint or "")[:200]))
                    conn.commit()
                finally:
                    conn.close()
            self._record_transition(tool, action)
            return {"success": True, "logged": f"{tool}.{action}", "at": now.strftime("%A %H:%M")}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ---- Analysis ---- #

    def get_patterns(self, min_occurrences: int = 2) -> Dict:
        """Return recurring (tool, action, day, hour) patterns."""
        try:
            with _db_lock:
                conn = _get_db()
                try:
                    rows = conn.execute("""
                        SELECT tool, action, hour, day_of_week, COUNT(*) as count
                        FROM task_events GROUP BY tool, action, hour, day_of_week
                        HAVING count >= ? ORDER BY count DESC LIMIT 50""", (min_occurrences,)).fetchall()
                finally:
                    conn.close()
            patterns = [{
                "tool": r["tool"], "action": r["action"], "hour": r["hour"],
                "day": DAY_NAMES[r["day_of_week"]], "occurrences": r["count"],
                "label": f"{r['tool']}.{r['action']} — {DAY_NAMES[r['day_of_week']]} ~{r['hour']:02d}:00 ({r['count']}×)",
            } for r in rows]
            return {"success": True, "count": len(patterns), "patterns": patterns}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_predictions(self, window_hours: int = 1) -> Dict:
        """Predict tasks relevant right now using Bayesian confidence + decay + feedback."""
        now = datetime.now()
        current_hour, current_dow = now.hour, now.weekday()
        hours = list({(current_hour + i) % 24 for i in range(-window_hours, window_hours + 1)})
        ph = ",".join("?" for _ in hours)
        try:
            with _db_lock:
                conn = _get_db()
                try:
                    day_rows = conn.execute(f"""
                        SELECT tool, action, hour, timestamp, COUNT(*) as count FROM task_events
                        WHERE hour IN ({ph}) AND day_of_week=? GROUP BY tool, action, hour
                        HAVING count >= 2 ORDER BY count DESC LIMIT 10""", (*hours, current_dow)).fetchall()
                    any_rows = conn.execute(f"""
                        SELECT tool, action, hour, timestamp, COUNT(*) as count FROM task_events
                        WHERE hour IN ({ph}) GROUP BY tool, action, hour
                        HAVING count >= 3 ORDER BY count DESC LIMIT 10""", hours).fetchall()
                    total_in_window = conn.execute(
                        f"SELECT COUNT(*) as n FROM task_events WHERE hour IN ({ph})", hours).fetchone()["n"]
                    unique_tools = conn.execute(
                        f"SELECT COUNT(DISTINCT tool||'.'||action) as k FROM task_events WHERE hour IN ({ph})", hours).fetchone()["k"]
                    decay_rows = conn.execute(f"""
                        SELECT tool, action, timestamp FROM task_events
                        WHERE hour IN ({ph}) ORDER BY timestamp DESC LIMIT 500""", hours).fetchall()
                finally:
                    conn.close()
            # Decay-weighted counts per (tool, action)
            decay_counts: Dict[Tuple[str, str], float] = defaultdict(float)
            for row in decay_rows:
                key = (row["tool"], row["action"])
                try:
                    days_ago = max((now - datetime.fromisoformat(row["timestamp"])).total_seconds() / 86400.0, 0.0)
                except (ValueError, TypeError):
                    days_ago = 30.0
                decay_counts[key] += 0.95 ** days_ago
            # Bayesian: P(tool|hour) = (count + alpha) / (total + alpha * K)
            alpha, K, total = 1.0, max(unique_tools, 1), max(total_in_window, 1)
            predictions, seen = [], set()
            for r, is_day in [(x, True) for x in day_rows] + [(x, False) for x in any_rows]:
                key = (r["tool"], r["action"])
                if key in seen:
                    continue
                seen.add(key)
                decayed = decay_counts.get(key, r["count"])
                bayesian = (decayed + alpha) / (total + alpha * K)
                if is_day:
                    bayesian *= 1.2
                fb = self._feedback_weight(r["tool"], r["action"])
                confidence = round(min(bayesian * fb, 1.0), 3)
                day_label = DAY_NAMES[current_dow] if is_day else "most days"
                predictions.append({
                    "prediction_id": uuid.uuid4().hex[:12],
                    "tool": r["tool"], "action": r["action"],
                    "confidence": confidence, "occurrences": r["count"],
                    "suggestion": f"You usually run {r['tool']}.{r['action']} on {day_label} around {r['hour']:02d}:00",
                })
            predictions.sort(key=lambda x: x["confidence"], reverse=True)
            return {"success": True, "now": now.strftime("%A %H:%M"),
                    "count": len(predictions), "predictions": predictions[:10]}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_stats(self) -> Dict:
        """Overall usage statistics."""
        try:
            with _db_lock:
                conn = _get_db()
                try:
                    total = conn.execute("SELECT COUNT(*) as n FROM task_events").fetchone()["n"]
                    top_tools = conn.execute(
                        "SELECT tool, COUNT(*) as count FROM task_events GROUP BY tool ORDER BY count DESC LIMIT 10").fetchall()
                    recent = conn.execute(
                        "SELECT tool, action, timestamp FROM task_events ORDER BY id DESC LIMIT 10").fetchall()
                    peak_hour = conn.execute(
                        "SELECT hour, COUNT(*) as count FROM task_events GROUP BY hour ORDER BY count DESC LIMIT 1").fetchone()
                    transition_count = conn.execute("SELECT COUNT(*) as n FROM tool_transitions").fetchone()["n"]
                    feedback_count = conn.execute("SELECT COUNT(*) as n FROM prediction_feedback").fetchone()["n"]
                finally:
                    conn.close()
            return {
                "success": True, "total_events": total,
                "total_transitions": transition_count, "total_feedback": feedback_count,
                "peak_hour": f"{peak_hour['hour']:02d}:00" if peak_hour else "unknown",
                "top_tools": [{"tool": r["tool"], "uses": r["count"]} for r in top_tools],
                "recent": [{"tool": r["tool"], "action": r["action"], "at": r["timestamp"][:16]} for r in recent],
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_weekly_heatmap(self) -> Dict:
        """Usage heatmap: day x hour frequency."""
        try:
            with _db_lock:
                conn = _get_db()
                try:
                    rows = conn.execute(
                        "SELECT day_of_week, hour, COUNT(*) as count FROM task_events GROUP BY day_of_week, hour").fetchall()
                finally:
                    conn.close()
            heatmap: Dict[str, Dict[int, int]] = {d: {} for d in DAY_NAMES}
            for r in rows:
                heatmap[DAY_NAMES[r["day_of_week"]]][r["hour"]] = r["count"]
            return {"success": True, "heatmap": heatmap}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ---- Markov chain sequence predictions ---- #

    def _build_transition_matrix(self) -> Dict[Tuple[str, str], Dict[Tuple[str, str], float]]:
        """Build Markov transition probability matrix from recorded transitions."""
        try:
            with _db_lock:
                conn = _get_db()
                try:
                    rows = conn.execute("""SELECT from_tool, from_action, to_tool, to_action, COUNT(*) as cnt
                        FROM tool_transitions GROUP BY from_tool, from_action, to_tool, to_action""").fetchall()
                finally:
                    conn.close()
        except Exception:
            return {}
        totals: Dict[Tuple[str, str], int] = defaultdict(int)
        raw: Dict[Tuple[str, str], Dict[Tuple[str, str], int]] = defaultdict(lambda: defaultdict(int))
        for r in rows:
            src, dst = (r["from_tool"], r["from_action"]), (r["to_tool"], r["to_action"])
            raw[src][dst] += r["cnt"]
            totals[src] += r["cnt"]
        matrix: Dict[Tuple[str, str], Dict[Tuple[str, str], float]] = {}
        for src, dests in raw.items():
            t = totals[src]
            if t > 0:
                matrix[src] = {dst: cnt / t for dst, cnt in dests.items()}
        return matrix

    def get_sequence_predictions(self, last_tool: str = None, last_action: str = None) -> Dict:
        """Predict next tools based on Markov chain transitions."""
        if last_tool and last_action:
            state = (last_tool, last_action)
        elif self._last_event:
            state = self._last_event
        else:
            return {"success": True, "count": 0, "predictions": [], "note": "No previous event to base sequence on."}
        matrix = self._build_transition_matrix()
        transitions = matrix.get(state, {})
        if not transitions:
            return {"success": True, "from": f"{state[0]}.{state[1]}",
                    "count": 0, "predictions": [],
                    "note": f"No recorded transitions from {state[0]}.{state[1]}"}
        preds = []
        for (to_tool, to_action), prob in sorted(transitions.items(), key=lambda x: x[1], reverse=True):
            fb = self._feedback_weight(to_tool, to_action)
            preds.append({
                "prediction_id": uuid.uuid4().hex[:12],
                "tool": to_tool, "action": to_action,
                "probability": round(prob, 3), "confidence": round(min(prob * fb, 1.0), 3),
                "suggestion": f"After {state[0]}.{state[1]}, you usually run {to_tool}.{to_action}",
            })
        return {"success": True, "from": f"{state[0]}.{state[1]}",
                "count": len(preds), "predictions": preds[:10]}

    # ---- Feedback loop ---- #

    def record_feedback(self, prediction_id: str, tool: str, action: str, accepted: bool) -> Dict:
        """Record user feedback on a prediction (accepted/rejected)."""
        if not tool:
            return {"success": False, "error": "tool name required"}
        try:
            with _db_lock:
                conn = _get_db()
                try:
                    conn.execute(
                        "INSERT INTO prediction_feedback (prediction_id,tool,action,accepted,timestamp) VALUES (?,?,?,?,?)",
                        (prediction_id or uuid.uuid4().hex[:12], tool, action, 1 if accepted else 0, datetime.now().isoformat()))
                    conn.commit()
                finally:
                    conn.close()
            return {"success": True, "recorded": f"{tool}.{action}", "accepted": bool(accepted)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ---- Context-aware predictions ---- #

    def get_context_predictions(self, context_hint: str = "") -> Dict:
        """Filter predictions by context_hint. Falls back to general predictions if no context data."""
        if not context_hint:
            return self.get_predictions()
        now = datetime.now()
        current_hour = now.hour
        try:
            with _db_lock:
                conn = _get_db()
                try:
                    rows = conn.execute("""
                        SELECT tool, action, hour, COUNT(*) as count, MAX(timestamp) as latest_ts
                        FROM task_events WHERE context_hint=?
                        GROUP BY tool, action ORDER BY count DESC LIMIT 20""", (context_hint,)).fetchall()
                finally:
                    conn.close()
            if not rows:
                result = self.get_predictions()
                result["note"] = f"No data for context '{context_hint}', showing general predictions."
                return result
            predictions = []
            for r in rows:
                hour_diff = min(abs(r["hour"] - current_hour), 24 - abs(r["hour"] - current_hour))
                time_factor = max(1.0 - (hour_diff / 12.0), 0.2)
                try:
                    days_ago = max((now - datetime.fromisoformat(r["latest_ts"])).total_seconds() / 86400.0, 0.0)
                except (ValueError, TypeError):
                    days_ago = 30.0
                recency = 0.95 ** days_ago
                fb = self._feedback_weight(r["tool"], r["action"])
                confidence = min(time_factor * recency * fb * min(r["count"] / 5.0, 1.0), 1.0)
                predictions.append({
                    "prediction_id": uuid.uuid4().hex[:12],
                    "tool": r["tool"], "action": r["action"],
                    "confidence": round(confidence, 3), "occurrences": r["count"],
                    "context": context_hint,
                    "suggestion": f"In context '{context_hint}', you often run {r['tool']}.{r['action']}",
                })
            predictions.sort(key=lambda x: x["confidence"], reverse=True)
            return {"success": True, "context": context_hint, "now": now.strftime("%A %H:%M"),
                    "count": len(predictions), "predictions": predictions[:10]}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ---- Combined prediction pipeline ---- #

    def get_combined_predictions(self, context_hint: str = None) -> Dict:
        """Merge time + sequence + context predictions. Dedup by (tool, action), keep highest confidence."""
        now = datetime.now()
        time_result = self.get_predictions()
        seq_result = self.get_sequence_predictions()
        ctx_result = self.get_context_predictions(context_hint) if context_hint else {"predictions": []}
        merged: Dict[Tuple[str, str], Dict] = {}
        for source_label, result in [("time", time_result), ("sequence", seq_result), ("context", ctx_result)]:
            for p in result.get("predictions", []):
                key = (p["tool"], p["action"])
                conf = p.get("confidence", 0.0)
                if key not in merged or conf > merged[key].get("confidence", 0.0):
                    entry = dict(p)
                    entry["source"] = source_label
                    merged[key] = entry
        combined = sorted(merged.values(), key=lambda x: x.get("confidence", 0.0), reverse=True)
        return {
            "success": True, "now": now.strftime("%A %H:%M"),
            "count": len(combined), "predictions": combined[:15],
            "sources": {"time": time_result.get("count", 0),
                        "sequence": seq_result.get("count", 0),
                        "context": ctx_result.get("count", 0)},
        }


class PredictiveTaskTool:
    """Learn workflow patterns and predict upcoming tasks."""

    name = "predictive_tasks"
    description = "Learn workflow patterns from tool usage and predict upcoming tasks — proactive suggestions before you ask."

    def __init__(self):
        self._engine = PredictiveTaskEngine()

    def execute(self, action: str, **kwargs) -> Dict:
        """Route action strings to engine methods."""
        a = action.lower().strip()
        # --- Existing routes --- #
        if "log" in a or "record" in a or "track" in a:
            return self._engine.log_event(
                kwargs.get("tool", ""), kwargs.get("action_name") or kwargs.get("tool_action", ""),
                kwargs.get("prompt", ""), kwargs.get("context_hint", ""))
        if "pattern" in a:
            return self._engine.get_patterns(kwargs.get("min_occurrences", 2))
        if "predict" in a or "suggest" in a or "now" in a:
            return self._engine.get_predictions(kwargs.get("window_hours", 1))
        if "stat" in a or "usage" in a or "history" in a:
            return self._engine.get_stats()
        if "heatmap" in a:
            return self._engine.get_weekly_heatmap()
        # --- New routes --- #
        if "feedback" in a:
            return self._engine.record_feedback(
                kwargs.get("prediction_id", ""), kwargs.get("tool", ""),
                kwargs.get("action_name") or kwargs.get("tool_action", ""), kwargs.get("accepted", False))
        if a in ("sequence", "next", "after") or "sequence" in a or "next" in a or "after" in a:
            return self._engine.get_sequence_predictions(kwargs.get("last_tool"), kwargs.get("last_action"))
        if "context" in a:
            return self._engine.get_context_predictions(kwargs.get("context_hint", ""))
        if a in ("combined", "all") or "combined" in a:
            return self._engine.get_combined_predictions(kwargs.get("context_hint"))
        # Default: combined predictions (upgraded from simple predictions)
        return self._engine.get_combined_predictions(kwargs.get("context_hint"))
