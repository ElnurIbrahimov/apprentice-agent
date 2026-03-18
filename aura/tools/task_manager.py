"""Task Manager tool — World-class task and project tracking with SQLite backend.

Features:
- Rich task structure: P0-P3 priority, 5 statuses, due dates, tags, dependencies
- Views: filtered list, kanban board, timeline, overdue, stats
- Smart features: suggest_next (AI), decompose, auto-extract from conversations
- SQLite backend with full-text search and history tracking
- Backward compatible with old JSON data (auto-migrates)
"""

import json
import logging
import re
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

# Priority and status definitions
PRIORITIES = ["P0", "P1", "P2", "P3"]
PRIORITY_LABELS = {"P0": "critical", "P1": "high", "P2": "medium", "P3": "low"}
PRIORITY_FROM_LABEL = {v: k for k, v in PRIORITY_LABELS.items()}
PRIORITY_FROM_LABEL.update({p: p for p in PRIORITIES})  # P0->P0 etc.

STATUSES = ["todo", "in_progress", "blocked", "done", "cancelled"]

# Old-format compat mapping
LEGACY_STATUS_MAP = {
    "backlog": "todo",
    "review": "in_progress",
    "archived": "cancelled",
}
LEGACY_PRIORITY_MAP = {
    "critical": "P0",
    "high": "P1",
    "medium": "P2",
    "low": "P3",
}

DB_PATH = Path(__file__).parent.parent.parent / "data" / "task_manager.db"
LEGACY_JSON = Path(__file__).parent.parent.parent / "data" / "task_manager.json"

# SQL schema
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    description TEXT DEFAULT '',
    status TEXT DEFAULT 'todo' CHECK(status IN ('todo','in_progress','blocked','done','cancelled')),
    priority TEXT DEFAULT 'P2' CHECK(priority IN ('P0','P1','P2','P3')),
    project TEXT DEFAULT 'default',
    assignee TEXT DEFAULT '',
    deadline TEXT,
    tags TEXT DEFAULT '[]',  -- JSON array
    dependencies TEXT DEFAULT '[]',  -- JSON array of task IDs
    parent_id TEXT,  -- For subtasks
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT,
    blocked_reason TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS task_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL,
    field TEXT NOT NULL,
    old_value TEXT,
    new_value TEXT,
    changed_at TEXT NOT NULL,
    FOREIGN KEY (task_id) REFERENCES tasks(id) ON DELETE CASCADE
);

CREATE VIRTUAL TABLE IF NOT EXISTS tasks_fts USING fts5(
    task_id,
    title,
    description,
    tags,
    content='tasks',
    content_rowid='rowid'
);

-- Triggers to keep FTS in sync
CREATE TRIGGER IF NOT EXISTS tasks_ai AFTER INSERT ON tasks BEGIN
    INSERT INTO tasks_fts(task_id, title, description, tags)
    VALUES (new.id, new.title, new.description, new.tags);
END;

CREATE TRIGGER IF NOT EXISTS tasks_ad AFTER DELETE ON tasks BEGIN
    INSERT INTO tasks_fts(tasks_fts, task_id, title, description, tags)
    VALUES ('delete', old.id, old.title, old.description, old.tags);
END;

CREATE TRIGGER IF NOT EXISTS tasks_au AFTER UPDATE ON tasks BEGIN
    INSERT INTO tasks_fts(tasks_fts, task_id, title, description, tags)
    VALUES ('delete', old.id, old.title, old.description, old.tags);
    INSERT INTO tasks_fts(task_id, title, description, tags)
    VALUES (new.id, new.title, new.description, new.tags);
END;

CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
CREATE INDEX IF NOT EXISTS idx_tasks_priority ON tasks(priority);
CREATE INDEX IF NOT EXISTS idx_tasks_project ON tasks(project);
CREATE INDEX IF NOT EXISTS idx_tasks_deadline ON tasks(deadline);
CREATE INDEX IF NOT EXISTS idx_tasks_parent ON tasks(parent_id);
"""


def _normalize_priority(p: str) -> str:
    """Convert any priority format to P0-P3."""
    if not p:
        return "P2"
    p_upper = p.upper().strip()
    if p_upper in PRIORITIES:
        return p_upper
    return LEGACY_PRIORITY_MAP.get(p.lower().strip(), "P2")


def _normalize_status(s: str) -> str:
    """Convert any status format to the new set."""
    if not s:
        return "todo"
    s_lower = s.lower().strip()
    if s_lower in STATUSES:
        return s_lower
    return LEGACY_STATUS_MAP.get(s_lower, "todo")


def _priority_icon(p: str) -> str:
    return {"P0": "!!!", "P1": "!!", "P2": "!", "P3": "-"}.get(p, "!")


def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    """Convert a sqlite3.Row to a dict, parsing JSON fields."""
    d = dict(row)
    for jf in ("tags", "dependencies"):
        if jf in d and isinstance(d[jf], str):
            try:
                d[jf] = json.loads(d[jf])
            except (json.JSONDecodeError, TypeError):
                d[jf] = []
    return d


class TaskManagerTool:
    """World-class task and project manager with SQLite backend."""

    name = "task_manager"
    description = "Manage tasks with priorities (P0-P3), statuses, due dates, tags, dependencies, and AI features"

    # Keep old class attr for anything that might reference it
    TASKS_FILE = LEGACY_JSON

    def __init__(self):
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self._maybe_migrate_json()

    # -------------------------------------------------------------------
    #  Database layer
    # -------------------------------------------------------------------

    def _init_db(self):
        with self._conn() as conn:
            conn.executescript(SCHEMA_SQL)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _maybe_migrate_json(self):
        """One-time migration from legacy JSON file."""
        if not LEGACY_JSON.exists():
            return
        # Check if DB already has tasks
        with self._conn() as conn:
            count = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
            if count > 0:
                return  # Already migrated or has data

        try:
            with open(LEGACY_JSON, "r", encoding="utf-8") as f:
                old_tasks = json.load(f)
        except (json.JSONDecodeError, IOError):
            return

        if not old_tasks:
            return

        logger.info("Migrating %d tasks from JSON to SQLite...", len(old_tasks))
        with self._conn() as conn:
            for t in old_tasks:
                now = datetime.now().isoformat()
                # Handle subtasks: create parent, then children
                subtasks = t.get("subtasks", [])
                conn.execute(
                    """INSERT OR IGNORE INTO tasks
                       (id, title, description, status, priority, project, assignee,
                        deadline, tags, dependencies, parent_id, created_at, updated_at,
                        started_at, completed_at, blocked_reason)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        t.get("id", uuid.uuid4().hex[:8]),
                        t.get("title", "Untitled"),
                        t.get("description", ""),
                        _normalize_status(t.get("status", "todo")),
                        _normalize_priority(t.get("priority", "medium")),
                        t.get("project", "default"),
                        t.get("assignee", ""),
                        t.get("deadline"),
                        json.dumps(t.get("tags", [])),
                        json.dumps(t.get("dependencies", [])),
                        None,
                        t.get("created_at", now),
                        t.get("updated_at", now),
                        None,
                        t.get("completed_at"),
                        "",
                    ),
                )
                # Migrate subtasks as child tasks
                parent_id = t.get("id")
                for sub in subtasks:
                    sub_status = "done" if sub.get("done") else "todo"
                    conn.execute(
                        """INSERT OR IGNORE INTO tasks
                           (id, title, description, status, priority, project, assignee,
                            deadline, tags, dependencies, parent_id, created_at, updated_at,
                            started_at, completed_at, blocked_reason)
                           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (
                            sub.get("id", uuid.uuid4().hex[:8]),
                            sub.get("title", "Subtask"),
                            "",
                            sub_status,
                            _normalize_priority(t.get("priority", "medium")),
                            t.get("project", "default"),
                            "",
                            None,
                            "[]",
                            "[]",
                            parent_id,
                            now,
                            now,
                            None,
                            now if sub_status == "done" else None,
                            "",
                        ),
                    )

        # Rename old file so we don't re-migrate
        try:
            LEGACY_JSON.rename(LEGACY_JSON.with_suffix(".json.migrated"))
        except OSError:
            pass

        logger.info("Migration complete.")

    # -------------------------------------------------------------------
    #  Internal helpers
    # -------------------------------------------------------------------

    def _generate_id(self) -> str:
        return uuid.uuid4().hex[:8]

    def _record_change(self, conn, task_id: str, field: str, old_val: Any, new_val: Any):
        """Record a field change in task_history."""
        conn.execute(
            "INSERT INTO task_history (task_id, field, old_value, new_value, changed_at) VALUES (?,?,?,?,?)",
            (task_id, field, str(old_val) if old_val is not None else None,
             str(new_val) if new_val is not None else None,
             datetime.now().isoformat()),
        )

    def _get_task(self, conn, task_id: str) -> Optional[Dict[str, Any]]:
        row = conn.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
        return _row_to_dict(row) if row else None

    def _check_dependencies_met(self, conn, task_id: str) -> List[str]:
        """Return list of unfinished dependency task IDs."""
        task = self._get_task(conn, task_id)
        if not task:
            return []
        deps = task.get("dependencies", [])
        if not deps:
            return []
        placeholders = ",".join("?" * len(deps))
        rows = conn.execute(
            f"SELECT id, title, status FROM tasks WHERE id IN ({placeholders}) AND status NOT IN ('done','cancelled')",
            deps,
        ).fetchall()
        return [f"{r['id']} ({r['title']})" for r in rows]

    # -------------------------------------------------------------------
    #  Core CRUD
    # -------------------------------------------------------------------

    def add_task(self, title: str, project: str = "default", priority: str = "P2",
                 description: str = "", tags: List[str] = None, deadline: str = None,
                 assignee: str = "", dependencies: List[str] = None,
                 parent_id: str = None, blocked_reason: str = "") -> dict:
        """Create a new task."""
        if not title:
            return {"success": False, "error": "No title provided"}

        priority = _normalize_priority(priority)
        task_id = self._generate_id()
        now = datetime.now().isoformat()

        with self._conn() as conn:
            conn.execute(
                """INSERT INTO tasks
                   (id, title, description, status, priority, project, assignee,
                    deadline, tags, dependencies, parent_id, created_at, updated_at,
                    started_at, completed_at, blocked_reason)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    task_id, title, description, "todo", priority, project, assignee,
                    deadline, json.dumps(tags or []), json.dumps(dependencies or []),
                    parent_id, now, now, None, None, blocked_reason,
                ),
            )
            self._record_change(conn, task_id, "status", None, "todo")

        return {
            "success": True,
            "task_id": task_id,
            "title": title,
            "project": project,
            "priority": priority,
            "response": f"Task created [{task_id}]: {title} ({priority}, project: {project})"
                        + (f", due: {deadline}" if deadline else ""),
        }

    def update_task(self, task_id: str, **updates) -> dict:
        """Update one or more fields on a task."""
        if not task_id:
            return {"success": False, "error": "No task ID provided"}

        allowed = {"title", "description", "status", "priority", "project", "assignee",
                    "deadline", "tags", "dependencies", "parent_id", "blocked_reason"}

        with self._conn() as conn:
            task = self._get_task(conn, task_id)
            if not task:
                return {"success": False, "error": f"Task not found: {task_id}"}

            sets = []
            params = []
            for key, value in updates.items():
                if key not in allowed:
                    continue

                if key == "status":
                    value = _normalize_status(value)
                    if value not in STATUSES:
                        return {"success": False, "error": f"Invalid status: {value}. Use: {', '.join(STATUSES)}"}
                    # Check dependencies before moving to in_progress/done
                    if value in ("in_progress", "done"):
                        blockers = self._check_dependencies_met(conn, task_id)
                        if blockers:
                            return {
                                "success": False,
                                "error": f"Cannot move to '{value}': blocked by unfinished dependencies: {', '.join(blockers)}",
                            }

                if key == "priority":
                    value = _normalize_priority(value)

                if key in ("tags", "dependencies") and isinstance(value, list):
                    value = json.dumps(value)

                old_val = task.get(key)
                self._record_change(conn, task_id, key, old_val, value)
                sets.append(f"{key}=?")
                params.append(value)

            if not sets:
                return {"success": False, "error": "No valid fields to update"}

            # Auto-set timestamps
            now = datetime.now().isoformat()
            sets.append("updated_at=?")
            params.append(now)

            new_status = updates.get("status")
            if new_status:
                new_status = _normalize_status(new_status)
                if new_status == "done" and not task.get("completed_at"):
                    sets.append("completed_at=?")
                    params.append(now)
                if new_status == "in_progress" and not task.get("started_at"):
                    sets.append("started_at=?")
                    params.append(now)

            params.append(task_id)
            conn.execute(f"UPDATE tasks SET {', '.join(sets)} WHERE id=?", params)

        return {
            "success": True,
            "task_id": task_id,
            "updates": {k: v for k, v in updates.items() if k in allowed},
            "response": f"Task {task_id} updated: {', '.join(f'{k}={v}' for k,v in updates.items() if k in allowed)}",
        }

    def move_task(self, task_id: str, status: str) -> dict:
        """Change task status."""
        return self.update_task(task_id, status=status)

    def remove_task(self, task_id: str) -> dict:
        """Delete a task and its subtasks."""
        if not task_id:
            return {"success": False, "error": "No task ID provided"}

        with self._conn() as conn:
            task = self._get_task(conn, task_id)
            if not task:
                return {"success": False, "error": f"Task not found: {task_id}"}
            # Delete subtasks first
            conn.execute("DELETE FROM task_history WHERE task_id IN (SELECT id FROM tasks WHERE parent_id=?)", (task_id,))
            conn.execute("DELETE FROM tasks WHERE parent_id=?", (task_id,))
            conn.execute("DELETE FROM task_history WHERE task_id=?", (task_id,))
            conn.execute("DELETE FROM tasks WHERE id=?", (task_id,))

        return {"success": True, "removed_id": task_id,
                "response": f"Task {task_id} removed: {task.get('title', '')}"}

    # -------------------------------------------------------------------
    #  Subtasks
    # -------------------------------------------------------------------

    def add_subtask(self, task_id: str, title: str, priority: str = None) -> dict:
        """Add a subtask under a parent task."""
        with self._conn() as conn:
            parent = self._get_task(conn, task_id)
            if not parent:
                return {"success": False, "error": f"Parent task not found: {task_id}"}

        return self.add_task(
            title=title,
            project=parent.get("project", "default"),
            priority=priority or parent.get("priority", "P2"),
            parent_id=task_id,
        )

    def complete_subtask(self, task_id: str, subtask_id: str) -> dict:
        """Mark a subtask as done."""
        return self.move_task(subtask_id, "done")

    def list_subtasks(self, task_id: str) -> dict:
        """List all subtasks of a parent task."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM tasks WHERE parent_id=? ORDER BY created_at", (task_id,)
            ).fetchall()

        tasks = [_row_to_dict(r) for r in rows]
        formatted = []
        for t in tasks:
            icon = "[x]" if t["status"] == "done" else "[ ]"
            formatted.append(f"  {icon} [{t['id']}] {t['title']}")

        done = sum(1 for t in tasks if t["status"] == "done")
        return {
            "success": True,
            "count": len(tasks),
            "done": done,
            "tasks": tasks,
            "response": f"Subtasks of {task_id}: {done}/{len(tasks)} done\n" + "\n".join(formatted),
        }

    # -------------------------------------------------------------------
    #  Queries / Views
    # -------------------------------------------------------------------

    def list_tasks(self, project: str = None, status: str = None,
                   priority: str = None, tag: str = None,
                   include_subtasks: bool = False) -> dict:
        """List tasks with filters."""
        conditions = []
        params = []

        if not include_subtasks:
            conditions.append("parent_id IS NULL")

        if project:
            conditions.append("project=?")
            params.append(project)
        if status:
            status = _normalize_status(status)
            conditions.append("status=?")
            params.append(status)
        else:
            # Exclude cancelled by default
            conditions.append("status != 'cancelled'")
        if priority:
            priority = _normalize_priority(priority)
            conditions.append("priority=?")
            params.append(priority)
        if tag:
            conditions.append("tags LIKE ?")
            params.append(f'%"{tag}"%')

        where = " AND ".join(conditions) if conditions else "1=1"

        with self._conn() as conn:
            rows = conn.execute(
                f"""SELECT * FROM tasks WHERE {where}
                    ORDER BY
                        CASE priority WHEN 'P0' THEN 0 WHEN 'P1' THEN 1 WHEN 'P2' THEN 2 ELSE 3 END,
                        CASE status WHEN 'in_progress' THEN 0 WHEN 'blocked' THEN 1 WHEN 'todo' THEN 2 WHEN 'done' THEN 3 ELSE 4 END,
                        deadline ASC NULLS LAST""",
                params,
            ).fetchall()

        tasks = [_row_to_dict(r) for r in rows]
        formatted = []
        for t in tasks:
            icon = _priority_icon(t["priority"])
            dl = f" (due {t['deadline']})" if t.get("deadline") else ""
            tags_str = f" [{', '.join(t.get('tags', []))}]" if t.get("tags") else ""
            formatted.append(f"[{t['id']}] {icon} {t['priority']} [{t['status']}] {t['title']}{dl}{tags_str}")

        return {
            "success": True,
            "count": len(tasks),
            "tasks": tasks,
            "formatted": "\n".join(formatted) if formatted else "No tasks found",
            "response": f"{len(tasks)} task(s)\n" + "\n".join(formatted),
        }

    def board(self, project: str = None) -> dict:
        """Kanban board view grouped by status."""
        conditions = ["parent_id IS NULL"]
        params = []
        if project:
            conditions.append("project=?")
            params.append(project)

        where = " AND ".join(conditions)

        with self._conn() as conn:
            rows = conn.execute(
                f"""SELECT * FROM tasks WHERE {where}
                    ORDER BY CASE priority WHEN 'P0' THEN 0 WHEN 'P1' THEN 1 WHEN 'P2' THEN 2 ELSE 3 END""",
                params,
            ).fetchall()

        tasks = [_row_to_dict(r) for r in rows]

        board_data = {}
        for status in STATUSES:
            board_data[status] = [t for t in tasks if t["status"] == status]

        formatted = []
        for status in STATUSES:
            items = board_data[status]
            formatted.append(f"\n--- {status.upper()} ({len(items)}) ---")
            for t in items:
                icon = _priority_icon(t["priority"])
                dl = f" [due {t['deadline']}]" if t.get("deadline") else ""
                formatted.append(f"  [{t['id']}] {icon} {t['priority']} {t['title']}{dl}")
            if not items:
                formatted.append("  (empty)")

        return {
            "success": True,
            "board": board_data,
            "response": "\n".join(formatted),
        }

    def timeline(self, days: int = 30) -> dict:
        """Chronological view of tasks with due dates."""
        cutoff = (datetime.now() + timedelta(days=days)).isoformat()[:10]

        with self._conn() as conn:
            rows = conn.execute(
                """SELECT * FROM tasks
                   WHERE deadline IS NOT NULL AND deadline <= ? AND status NOT IN ('done','cancelled')
                   AND parent_id IS NULL
                   ORDER BY deadline ASC,
                       CASE priority WHEN 'P0' THEN 0 WHEN 'P1' THEN 1 WHEN 'P2' THEN 2 ELSE 3 END""",
                (cutoff,),
            ).fetchall()

        tasks = [_row_to_dict(r) for r in rows]
        today = datetime.now().date()

        formatted = [f"Timeline (next {days} days):"]
        current_date = None
        for t in tasks:
            try:
                dl = datetime.fromisoformat(t["deadline"]).date()
            except (ValueError, TypeError):
                continue
            date_str = t["deadline"][:10]
            if date_str != current_date:
                current_date = date_str
                overdue_tag = " ** OVERDUE **" if dl < today else ""
                days_away = (dl - today).days
                when = f"(today)" if days_away == 0 else f"({days_away}d)" if days_away > 0 else f"({abs(days_away)}d ago)"
                formatted.append(f"\n  {date_str} {when}{overdue_tag}")
            icon = _priority_icon(t["priority"])
            formatted.append(f"    [{t['id']}] {icon} {t['priority']} [{t['status']}] {t['title']}")

        if len(formatted) == 1:
            formatted.append("  No tasks with deadlines in this range.")

        return {
            "success": True,
            "count": len(tasks),
            "tasks": tasks,
            "response": "\n".join(formatted),
        }

    def overdue(self) -> dict:
        """List tasks past their deadline."""
        today = datetime.now().isoformat()[:10]

        with self._conn() as conn:
            rows = conn.execute(
                """SELECT * FROM tasks
                   WHERE deadline IS NOT NULL AND deadline < ? AND status NOT IN ('done','cancelled')
                   AND parent_id IS NULL
                   ORDER BY deadline ASC""",
                (today,),
            ).fetchall()

        tasks = [_row_to_dict(r) for r in rows]
        formatted = []
        for t in tasks:
            try:
                dl = datetime.fromisoformat(t["deadline"]).date()
                days_late = (datetime.now().date() - dl).days
            except (ValueError, TypeError):
                days_late = "?"
            formatted.append(f"[{t['id']}] {_priority_icon(t['priority'])} {t['priority']} OVERDUE ({days_late}d late): {t['title']} (due {t['deadline']})")

        return {
            "success": True,
            "count": len(tasks),
            "tasks": tasks,
            "response": f"{len(tasks)} overdue task(s)\n" + "\n".join(formatted) if tasks else "No overdue tasks",
        }

    def stats(self, project: str = None) -> dict:
        """Completion rate, avg time-to-done, priority distribution."""
        conditions = ["parent_id IS NULL"]
        params = []
        if project:
            conditions.append("project=?")
            params.append(project)
        where = " AND ".join(conditions)

        with self._conn() as conn:
            rows = conn.execute(f"SELECT * FROM tasks WHERE {where}", params).fetchall()

        tasks = [_row_to_dict(r) for r in rows]
        total = len(tasks)
        if total == 0:
            return {"success": True, "response": "No tasks found."}

        # Status distribution
        status_counts = {}
        for s in STATUSES:
            status_counts[s] = sum(1 for t in tasks if t["status"] == s)

        # Priority distribution
        priority_counts = {}
        for p in PRIORITIES:
            priority_counts[p] = sum(1 for t in tasks if t["priority"] == p)

        # Completion rate
        done = status_counts.get("done", 0)
        completion_rate = (done / total * 100) if total > 0 else 0

        # Average time to done
        durations = []
        for t in tasks:
            if t["status"] == "done" and t.get("completed_at") and t.get("created_at"):
                try:
                    created = datetime.fromisoformat(t["created_at"])
                    completed = datetime.fromisoformat(t["completed_at"])
                    durations.append((completed - created).total_seconds() / 3600)
                except (ValueError, TypeError):
                    pass
        avg_hours = sum(durations) / len(durations) if durations else 0

        # Overdue count
        today = datetime.now().isoformat()[:10]
        overdue_count = sum(
            1 for t in tasks
            if t.get("deadline") and t["deadline"] < today and t["status"] not in ("done", "cancelled")
        )

        # Blocked count
        blocked = status_counts.get("blocked", 0)

        formatted = [
            f"Task Stats{' (' + project + ')' if project else ''}:",
            f"  Total: {total}",
            f"  Completion: {done}/{total} ({completion_rate:.0f}%)",
            f"  Avg time-to-done: {avg_hours:.1f}h" if durations else "  Avg time-to-done: N/A",
            f"  Overdue: {overdue_count}",
            f"  Blocked: {blocked}",
            f"",
            f"  By Status:",
        ]
        for s in STATUSES:
            c = status_counts[s]
            bar = "#" * min(c, 30)
            formatted.append(f"    {s:15s} {c:3d} {bar}")
        formatted.append("")
        formatted.append("  By Priority:")
        for p in PRIORITIES:
            c = priority_counts[p]
            label = PRIORITY_LABELS[p]
            bar = "#" * min(c, 30)
            formatted.append(f"    {p} ({label:8s}) {c:3d} {bar}")

        return {
            "success": True,
            "total": total,
            "completion_rate": round(completion_rate, 1),
            "avg_hours_to_done": round(avg_hours, 1) if durations else None,
            "overdue": overdue_count,
            "blocked": blocked,
            "status_counts": status_counts,
            "priority_counts": priority_counts,
            "response": "\n".join(formatted),
        }

    def list_projects(self) -> dict:
        """List all projects with stats."""
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT project,
                    COUNT(*) as total,
                    SUM(CASE WHEN status='done' THEN 1 ELSE 0 END) as done,
                    SUM(CASE WHEN status='in_progress' THEN 1 ELSE 0 END) as active,
                    SUM(CASE WHEN status='blocked' THEN 1 ELSE 0 END) as blocked
                   FROM tasks WHERE parent_id IS NULL
                   GROUP BY project ORDER BY total DESC"""
            ).fetchall()

        projects = {}
        formatted = []
        for r in rows:
            projects[r["project"]] = {
                "total": r["total"], "done": r["done"],
                "in_progress": r["active"], "blocked": r["blocked"],
            }
            pct = (r["done"] / r["total"] * 100) if r["total"] > 0 else 0
            formatted.append(
                f"[{r['project']}] {r['total']} tasks ({r['done']} done={pct:.0f}%, {r['active']} active, {r['blocked']} blocked)"
            )

        return {
            "success": True,
            "projects": projects,
            "count": len(projects),
            "response": "\n".join(formatted) if formatted else "No projects",
        }

    # -------------------------------------------------------------------
    #  Search
    # -------------------------------------------------------------------

    def search_tasks(self, query: str) -> dict:
        """Full-text search across task titles, descriptions, and tags."""
        if not query:
            return {"success": False, "error": "No query provided"}

        # Use FTS5 for search
        with self._conn() as conn:
            try:
                fts_rows = conn.execute(
                    """SELECT task_id FROM tasks_fts WHERE tasks_fts MATCH ?
                       ORDER BY rank LIMIT 50""",
                    (query,),
                ).fetchall()
                task_ids = [r["task_id"] for r in fts_rows]
            except sqlite3.OperationalError:
                # FTS match syntax error — fall back to LIKE
                task_ids = []

            if task_ids:
                placeholders = ",".join("?" * len(task_ids))
                rows = conn.execute(
                    f"SELECT * FROM tasks WHERE id IN ({placeholders})", task_ids
                ).fetchall()
            else:
                # Fallback: LIKE search
                q = f"%{query}%"
                rows = conn.execute(
                    "SELECT * FROM tasks WHERE title LIKE ? OR description LIKE ? OR tags LIKE ? LIMIT 50",
                    (q, q, q),
                ).fetchall()

        tasks = [_row_to_dict(r) for r in rows]
        formatted = [f"[{t['id']}] {t['priority']} [{t['status']}] {t['title']}" for t in tasks]
        return {
            "success": True,
            "count": len(tasks),
            "tasks": tasks,
            "response": f"{len(tasks)} result(s) for '{query}':\n" + "\n".join(formatted),
        }

    # -------------------------------------------------------------------
    #  Task history
    # -------------------------------------------------------------------

    def task_history(self, task_id: str) -> dict:
        """Show the change history of a task."""
        with self._conn() as conn:
            task = self._get_task(conn, task_id)
            if not task:
                return {"success": False, "error": f"Task not found: {task_id}"}

            rows = conn.execute(
                "SELECT * FROM task_history WHERE task_id=? ORDER BY changed_at DESC LIMIT 50",
                (task_id,),
            ).fetchall()

        changes = [dict(r) for r in rows]
        formatted = [f"History for [{task_id}] {task.get('title', '')}:"]
        for c in changes:
            formatted.append(f"  {c['changed_at'][:16]} | {c['field']}: {c.get('old_value','(none)')} -> {c.get('new_value','(none)')}")

        return {
            "success": True,
            "task_id": task_id,
            "changes": changes,
            "response": "\n".join(formatted) if changes else f"No history for task {task_id}",
        }

    # -------------------------------------------------------------------
    #  Smart features
    # -------------------------------------------------------------------

    def suggest_next(self) -> dict:
        """Recommend which task to work on next.

        Scoring: priority weight + urgency (due date) + dependency-free bonus + not-blocked.
        """
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT * FROM tasks
                   WHERE status IN ('todo', 'in_progress') AND parent_id IS NULL
                   ORDER BY
                       CASE priority WHEN 'P0' THEN 0 WHEN 'P1' THEN 1 WHEN 'P2' THEN 2 ELSE 3 END,
                       deadline ASC NULLS LAST""",
            ).fetchall()

        tasks = [_row_to_dict(r) for r in rows]
        if not tasks:
            return {"success": True, "response": "No actionable tasks. All done!"}

        today = datetime.now().date()
        scored = []

        for t in tasks:
            score = 0.0

            # Priority weight (P0=40, P1=30, P2=20, P3=10)
            pweight = {"P0": 40, "P1": 30, "P2": 20, "P3": 10}
            score += pweight.get(t["priority"], 20)

            # In-progress bonus (already started)
            if t["status"] == "in_progress":
                score += 15

            # Urgency: days until deadline
            if t.get("deadline"):
                try:
                    dl = datetime.fromisoformat(t["deadline"]).date()
                    days_left = (dl - today).days
                    if days_left < 0:
                        score += 50  # Overdue = massive boost
                    elif days_left == 0:
                        score += 35  # Due today
                    elif days_left <= 3:
                        score += 25  # Due soon
                    elif days_left <= 7:
                        score += 15
                except (ValueError, TypeError):
                    pass

            # Dependency-free bonus
            deps = t.get("dependencies", [])
            if not deps:
                score += 10

            # Check if blocked by deps
            if deps:
                with self._conn() as conn:
                    blockers = self._check_dependencies_met(conn, t["id"])
                    if blockers:
                        score -= 100  # Can't work on this

            scored.append((score, t))

        scored.sort(key=lambda x: -x[0])

        formatted = ["Suggested task order:"]
        for i, (score, t) in enumerate(scored[:5]):
            icon = _priority_icon(t["priority"])
            dl = f" (due {t['deadline']})" if t.get("deadline") else ""
            status_tag = f" [IN PROGRESS]" if t["status"] == "in_progress" else ""
            formatted.append(f"  {i+1}. [{t['id']}] {icon} {t['priority']} {t['title']}{dl}{status_tag} (score: {score:.0f})")

        top = scored[0][1] if scored else None
        return {
            "success": True,
            "suggestion": top,
            "ranked": [{"score": s, "task": t} for s, t in scored[:5]],
            "response": "\n".join(formatted),
        }

    def decompose(self, task_id: str, subtasks: List[str] = None) -> dict:
        """Break a task into subtasks.

        If subtasks list is provided, creates them directly.
        Otherwise returns a suggestion structure for the AI to fill in.
        """
        with self._conn() as conn:
            task = self._get_task(conn, task_id)
            if not task:
                return {"success": False, "error": f"Task not found: {task_id}"}

        if subtasks:
            created = []
            for title in subtasks:
                result = self.add_subtask(task_id, title)
                if result.get("success"):
                    created.append(result)

            return {
                "success": True,
                "parent_id": task_id,
                "created": len(created),
                "response": f"Created {len(created)} subtasks under [{task_id}] {task['title']}",
            }

        # No subtasks provided — return the task details for AI to decompose
        return {
            "success": True,
            "needs_ai": True,
            "task": task,
            "response": f"Task [{task_id}]: {task['title']}\n"
                        f"Description: {task.get('description', '(none)')}\n"
                        f"Priority: {task['priority']}\n\n"
                        f"Provide a list of subtask titles to decompose this task. "
                        f"Call decompose('{task_id}', subtasks=['Step 1', 'Step 2', ...])",
        }

    def extract_tasks_from_text(self, text: str, project: str = "default") -> dict:
        """Auto-extract tasks from conversational text.

        Looks for patterns like:
        - "I need to ..."
        - "TODO: ..."
        - "We should ..."
        - "Don't forget to ..."
        - "Make sure to ..."
        - "Action item: ..."
        - Lines starting with "- [ ] "
        """
        patterns = [
            r"(?:I need to|I have to|I must|I should)\s+(.+?)(?:\.|$)",
            r"TODO:\s*(.+?)(?:\.|$)",
            r"(?:We should|We need to|We must)\s+(.+?)(?:\.|$)",
            r"(?:Don't forget to|Remember to|Make sure to)\s+(.+?)(?:\.|$)",
            r"(?:Action item|ACTION):\s*(.+?)(?:\.|$)",
            r"^- \[ \]\s*(.+?)$",
        ]

        extracted = []
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
            for m in matches:
                title = m.group(1).strip()
                if title and len(title) > 3 and title not in [e["title"] for e in extracted]:
                    extracted.append({"title": title})

        if not extracted:
            return {"success": True, "count": 0, "response": "No tasks found in text."}

        created = []
        for item in extracted:
            result = self.add_task(title=item["title"], project=project)
            if result.get("success"):
                created.append(result)

        formatted = [f"Extracted {len(created)} task(s) from text:"]
        for r in created:
            formatted.append(f"  [{r['task_id']}] {r['title']}")

        return {
            "success": True,
            "count": len(created),
            "tasks": created,
            "response": "\n".join(formatted),
        }

    # -------------------------------------------------------------------
    #  Backward-compatible helpers
    # -------------------------------------------------------------------

    def _load_tasks(self) -> List[Dict[str, Any]]:
        """Legacy compat: returns all tasks as list of dicts."""
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM tasks ORDER BY created_at").fetchall()
        return [_row_to_dict(r) for r in rows]

    def _save_tasks(self, tasks: List[Dict[str, Any]]) -> bool:
        """Legacy compat: no-op (SQLite is the source of truth)."""
        return True

    # -------------------------------------------------------------------
    #  Dispatch (extended)
    # -------------------------------------------------------------------

    def _extract_task_info(self, action: str) -> dict:
        result = {}
        # Title in quotes
        quote_match = re.search(r'["\']([^"\']+)["\']', action)
        if quote_match:
            result["title"] = quote_match.group(1)
        # Priority (P0-P3 or labels)
        p_match = re.search(r'\b(P[0-3])\b', action, re.IGNORECASE)
        if p_match:
            result["priority"] = p_match.group(1).upper()
        else:
            for label in ("critical", "high", "medium", "low"):
                if label in action.lower():
                    result["priority"] = LEGACY_PRIORITY_MAP[label]
                    break
        # Project
        proj_match = re.search(r'project:\s*(\S+)', action, re.IGNORECASE)
        if proj_match:
            result["project"] = proj_match.group(1)
        # Deadline
        dl_match = re.search(r'(?:deadline|due|by)\s*:?\s*(\d{4}-\d{2}-\d{2})', action, re.IGNORECASE)
        if dl_match:
            result["deadline"] = dl_match.group(1)
        # Tags
        tag_match = re.findall(r'#(\w+)', action)
        if tag_match:
            result["tags"] = tag_match
        return result

    def execute(self, action: str, **kwargs) -> dict:
        action_lower = action.lower().strip()

        # Board view
        if action_lower.startswith("board"):
            project = kwargs.get("project")
            if not project and len(action.split()) > 1:
                project = action.split(None, 1)[-1]
            return self.board(project=project)

        # Stats
        if action_lower in ("stats", "statistics"):
            return self.stats(project=kwargs.get("project"))

        # Timeline
        if action_lower.startswith("timeline"):
            days = kwargs.get("days", 30)
            return self.timeline(days=days)

        # Suggest next
        if action_lower in ("suggest", "suggest_next", "next", "what_next"):
            return self.suggest_next()

        # Projects
        if action_lower in ("projects", "list_projects"):
            return self.list_projects()

        # Overdue
        if action_lower in ("overdue", "past_due"):
            return self.overdue()

        # Task history
        if action_lower.startswith("history"):
            task_id = kwargs.get("task_id")
            if not task_id:
                m = re.search(r'\b([a-f0-9]{8})\b', action)
                task_id = m.group(1) if m else None
            if task_id:
                return self.task_history(task_id)
            return {"success": False, "error": "Provide task_id for history"}

        # Decompose
        if action_lower.startswith("decompose") or action_lower.startswith("break"):
            task_id = kwargs.get("task_id")
            subtasks = kwargs.get("subtasks")
            if not task_id:
                m = re.search(r'\b([a-f0-9]{8})\b', action)
                task_id = m.group(1) if m else None
            if task_id:
                return self.decompose(task_id, subtasks=subtasks)
            return {"success": False, "error": "Provide task_id to decompose"}

        # Extract from text
        if action_lower.startswith("extract"):
            text = kwargs.get("text", "")
            project = kwargs.get("project", "default")
            return self.extract_tasks_from_text(text, project=project)

        # List
        if action_lower.startswith("list") or action_lower.startswith("show"):
            return self.list_tasks(
                project=kwargs.get("project"),
                status=kwargs.get("status"),
                priority=kwargs.get("priority"),
                tag=kwargs.get("tag"),
            )

        # Search
        if action_lower.startswith("search") or action_lower.startswith("find"):
            query = kwargs.get("query") or (action.split(None, 1)[-1] if len(action.split()) > 1 else "")
            return self.search_tasks(query)

        # Remove
        if action_lower.startswith("remove") or action_lower.startswith("delete"):
            task_id = kwargs.get("task_id")
            if not task_id:
                m = re.search(r'\b([a-f0-9]{8})\b', action)
                task_id = m.group(1) if m else None
            if task_id:
                return self.remove_task(task_id)
            return {"success": False, "error": "No task ID specified"}

        # Move / status change
        if action_lower.startswith("move") or action_lower.startswith("status"):
            task_id = kwargs.get("task_id")
            status = kwargs.get("status")
            if not task_id or not status:
                parts = action.split()
                for p in parts:
                    if re.match(r'^[a-f0-9]{8}$', p):
                        task_id = task_id or p
                    elif _normalize_status(p.lower()) in STATUSES and p.lower() not in ("move", "status"):
                        status = status or p.lower()
            if task_id and status:
                return self.move_task(task_id, status)
            return {"success": False, "error": "Usage: move <task_id> <status>"}

        # Block
        if action_lower.startswith("block"):
            task_id = kwargs.get("task_id")
            reason = kwargs.get("reason", "")
            if not task_id:
                m = re.search(r'\b([a-f0-9]{8})\b', action)
                task_id = m.group(1) if m else None
            if task_id:
                return self.update_task(task_id, status="blocked", blocked_reason=reason)
            return {"success": False, "error": "No task ID specified"}

        # Done shortcut
        if action_lower.startswith("done") or action_lower.startswith("complete"):
            task_id = kwargs.get("task_id")
            if not task_id:
                m = re.search(r'\b([a-f0-9]{8})\b', action)
                task_id = m.group(1) if m else None
            if task_id:
                return self.move_task(task_id, "done")
            return {"success": False, "error": "No task ID specified"}

        # Add subtask
        if action_lower.startswith("subtask"):
            task_id = kwargs.get("task_id")
            title = kwargs.get("title")
            if not task_id or not title:
                m = re.search(r'\b([a-f0-9]{8})\b', action)
                task_id = task_id or (m.group(1) if m else None)
                title = title or re.sub(r'\b[a-f0-9]{8}\b', '', action.split(None, 1)[-1]).strip()
            if task_id and title:
                return self.add_subtask(task_id, title)
            return {"success": False, "error": "Usage: subtask <task_id> <title>"}

        # Default: add task
        title = kwargs.get("title")
        project = kwargs.get("project", "default")
        priority = kwargs.get("priority", "P2")
        description = kwargs.get("description", "")
        tags = kwargs.get("tags", [])
        deadline = kwargs.get("deadline")
        dependencies = kwargs.get("dependencies", [])

        if not title:
            extracted = self._extract_task_info(action)
            title = title or extracted.get("title")
            priority = extracted.get("priority", priority)
            project = extracted.get("project", project)
            deadline = extracted.get("deadline", deadline)
            tags = extracted.get("tags", tags)

        if not title:
            cleaned = re.sub(r'^add\s+', '', action, flags=re.IGNORECASE).strip()
            if cleaned and len(cleaned) > 2:
                title = cleaned

        if title:
            return self.add_task(
                title=title, project=project, priority=priority,
                description=description, tags=tags, deadline=deadline,
                dependencies=dependencies,
            )

        return {
            "success": False,
            "error": f"Could not parse: {action}. "
                     "Try: 'add <title>', 'list', 'board', 'timeline', 'stats', 'suggest', "
                     "'done <id>', 'move <id> <status>', 'search <query>', 'decompose <id>'"
        }


# Singleton
task_manager_tool = TaskManagerTool()
