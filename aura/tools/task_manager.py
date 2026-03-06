"""Task Manager tool — Kanban-style task and project tracking."""

import json
import re
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any


STATUSES = ["backlog", "todo", "in_progress", "review", "done", "archived"]
PRIORITIES = ["low", "medium", "high", "critical"]


@dataclass
class Task:
    """A task/todo item."""
    id: str
    title: str
    status: str = "todo"
    priority: str = "medium"
    project: str = "default"
    description: str = ""
    tags: List[str] = field(default_factory=list)
    assignee: str = ""
    deadline: Optional[str] = None       # ISO date
    created_at: str = ""
    updated_at: str = ""
    completed_at: Optional[str] = None
    subtasks: List[Dict[str, Any]] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)  # task IDs

    def __post_init__(self):
        now = datetime.now().isoformat()
        if not self.created_at:
            self.created_at = now
        if not self.updated_at:
            self.updated_at = now

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Task':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class TaskManagerTool:
    """Kanban-style task and project manager."""

    name = "task_manager"
    description = "Manage tasks, projects, and todo lists with priorities and deadlines"

    TASKS_FILE = Path(__file__).parent.parent.parent / "data" / "task_manager.json"

    def __init__(self):
        self._ensure_file()

    def _ensure_file(self):
        self.TASKS_FILE.parent.mkdir(parents=True, exist_ok=True)
        if not self.TASKS_FILE.exists():
            self._save_tasks([])

    def _load_tasks(self) -> List[Dict[str, Any]]:
        try:
            with open(self.TASKS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return []

    def _save_tasks(self, tasks: List[Dict[str, Any]]) -> bool:
        try:
            with open(self.TASKS_FILE, "w", encoding="utf-8") as f:
                json.dump(tasks, f, indent=4)
            return True
        except IOError:
            return False

    def _generate_id(self) -> str:
        return uuid.uuid4().hex[:8]

    # -- Core CRUD ----------------------------------------------------------

    def add_task(self, title: str, project: str = "default", priority: str = "medium",
                 description: str = "", tags: List[str] = None, deadline: str = None,
                 assignee: str = "") -> dict:
        if not title:
            return {"success": False, "error": "No title provided"}

        priority = priority.lower() if priority else "medium"
        if priority not in PRIORITIES:
            return {"success": False, "error": f"Invalid priority: {priority}. Use: {', '.join(PRIORITIES)}"}

        task = Task(
            id=self._generate_id(),
            title=title,
            project=project,
            priority=priority,
            description=description,
            tags=tags or [],
            deadline=deadline,
            assignee=assignee,
        )

        tasks = self._load_tasks()
        tasks.append(task.to_dict())
        self._save_tasks(tasks)

        return {
            "success": True,
            "task_id": task.id,
            "title": task.title,
            "project": task.project,
            "priority": task.priority,
            "response": f"Task created [{task.id}]: {title} (priority: {priority}, project: {project})"
        }

    def update_task(self, task_id: str, **updates) -> dict:
        if not task_id:
            return {"success": False, "error": "No task ID provided"}

        tasks = self._load_tasks()
        for i, t in enumerate(tasks):
            if t.get("id") == task_id:
                for key, value in updates.items():
                    if key in Task.__dataclass_fields__ and key not in ("id", "created_at"):
                        if key == "status" and value not in STATUSES:
                            return {"success": False, "error": f"Invalid status: {value}. Use: {', '.join(STATUSES)}"}
                        if key == "priority" and value not in PRIORITIES:
                            return {"success": False, "error": f"Invalid priority: {value}"}
                        t[key] = value
                t["updated_at"] = datetime.now().isoformat()
                if t.get("status") == "done" and not t.get("completed_at"):
                    t["completed_at"] = datetime.now().isoformat()
                tasks[i] = t
                self._save_tasks(tasks)
                return {"success": True, "task_id": task_id, "updates": updates,
                        "response": f"Task {task_id} updated"}
        return {"success": False, "error": f"Task not found: {task_id}"}

    def move_task(self, task_id: str, status: str) -> dict:
        return self.update_task(task_id, status=status.lower())

    def remove_task(self, task_id: str) -> dict:
        if not task_id:
            return {"success": False, "error": "No task ID provided"}
        tasks = self._load_tasks()
        original = len(tasks)
        tasks = [t for t in tasks if t.get("id") != task_id]
        if len(tasks) == original:
            return {"success": False, "error": f"Task not found: {task_id}"}
        self._save_tasks(tasks)
        return {"success": True, "removed_id": task_id, "response": f"Task {task_id} removed"}

    # -- Queries ------------------------------------------------------------

    def list_tasks(self, project: str = None, status: str = None,
                   priority: str = None, tag: str = None) -> dict:
        tasks = self._load_tasks()

        if project:
            tasks = [t for t in tasks if t.get("project", "").lower() == project.lower()]
        if status:
            tasks = [t for t in tasks if t.get("status", "").lower() == status.lower()]
        if priority:
            tasks = [t for t in tasks if t.get("priority", "").lower() == priority.lower()]
        if tag:
            tasks = [t for t in tasks if tag.lower() in [x.lower() for x in t.get("tags", [])]]

        # Exclude archived by default
        if not status:
            tasks = [t for t in tasks if t.get("status") != "archived"]

        # Sort: critical first, then by status order
        priority_order = {p: i for i, p in enumerate(reversed(PRIORITIES))}
        status_order = {s: i for i, s in enumerate(STATUSES)}
        tasks.sort(key=lambda t: (
            priority_order.get(t.get("priority", "medium"), 2),
            status_order.get(t.get("status", "todo"), 1),
        ))

        formatted = []
        for t in tasks:
            pri_icon = {"critical": "!!!", "high": "!!", "medium": "!", "low": "-"}.get(t.get("priority", "medium"), "!")
            deadline_str = f" (due {t['deadline']})" if t.get("deadline") else ""
            formatted.append(f"[{t['id']}] {pri_icon} [{t['status']}] {t['title']}{deadline_str}")

        return {
            "success": True,
            "count": len(tasks),
            "tasks": tasks,
            "formatted": "\n".join(formatted) if formatted else "No tasks found",
            "response": f"{len(tasks)} task(s)\n" + "\n".join(formatted)
        }

    def board(self, project: str = None) -> dict:
        """Kanban board view grouped by status."""
        tasks = self._load_tasks()
        if project:
            tasks = [t for t in tasks if t.get("project", "").lower() == project.lower()]

        board = {}
        for status in STATUSES:
            if status == "archived":
                continue
            board[status] = [t for t in tasks if t.get("status") == status]

        formatted = []
        for status, items in board.items():
            formatted.append(f"--- {status.upper()} ({len(items)}) ---")
            for t in items:
                pri_icon = {"critical": "!!!", "high": "!!", "medium": "!", "low": "-"}.get(t.get("priority"), "!")
                formatted.append(f"  [{t['id']}] {pri_icon} {t['title']}")
            if not items:
                formatted.append("  (empty)")

        return {
            "success": True,
            "board": board,
            "response": "\n".join(formatted)
        }

    def list_projects(self) -> dict:
        tasks = self._load_tasks()
        projects = {}
        for t in tasks:
            proj = t.get("project", "default")
            if proj not in projects:
                projects[proj] = {"total": 0, "done": 0, "in_progress": 0}
            projects[proj]["total"] += 1
            if t.get("status") == "done":
                projects[proj]["done"] += 1
            elif t.get("status") == "in_progress":
                projects[proj]["in_progress"] += 1

        formatted = []
        for name, stats in projects.items():
            formatted.append(f"[{name}] {stats['total']} tasks ({stats['done']} done, {stats['in_progress']} active)")

        return {
            "success": True,
            "projects": projects,
            "count": len(projects),
            "response": "\n".join(formatted) if formatted else "No projects"
        }

    def search_tasks(self, query: str) -> dict:
        if not query:
            return {"success": False, "error": "No query provided"}
        tasks = self._load_tasks()
        q = query.lower()
        matching = [t for t in tasks
                    if q in t.get("title", "").lower()
                    or q in t.get("description", "").lower()
                    or any(q in tag.lower() for tag in t.get("tags", []))]

        formatted = [f"[{t['id']}] [{t['status']}] {t['title']}" for t in matching]
        return {
            "success": True,
            "count": len(matching),
            "tasks": matching,
            "response": f"{len(matching)} result(s)\n" + "\n".join(formatted)
        }

    def overdue(self) -> dict:
        """List tasks past their deadline."""
        tasks = self._load_tasks()
        now = datetime.now().date()
        overdue = []
        for t in tasks:
            if t.get("deadline") and t.get("status") not in ("done", "archived"):
                try:
                    dl = datetime.fromisoformat(t["deadline"]).date()
                    if dl < now:
                        overdue.append(t)
                except (ValueError, TypeError):
                    pass

        formatted = [f"[{t['id']}] OVERDUE {t['deadline']}: {t['title']}" for t in overdue]
        return {
            "success": True,
            "count": len(overdue),
            "tasks": overdue,
            "response": f"{len(overdue)} overdue task(s)\n" + "\n".join(formatted) if overdue else "No overdue tasks"
        }

    # -- Subtasks -----------------------------------------------------------

    def add_subtask(self, task_id: str, title: str) -> dict:
        tasks = self._load_tasks()
        for i, t in enumerate(tasks):
            if t.get("id") == task_id:
                sub = {"id": self._generate_id(), "title": title, "done": False}
                t.setdefault("subtasks", []).append(sub)
                t["updated_at"] = datetime.now().isoformat()
                tasks[i] = t
                self._save_tasks(tasks)
                return {"success": True, "subtask_id": sub["id"],
                        "response": f"Subtask added to {task_id}: {title}"}
        return {"success": False, "error": f"Task not found: {task_id}"}

    def complete_subtask(self, task_id: str, subtask_id: str) -> dict:
        tasks = self._load_tasks()
        for i, t in enumerate(tasks):
            if t.get("id") == task_id:
                for sub in t.get("subtasks", []):
                    if sub.get("id") == subtask_id:
                        sub["done"] = True
                        t["updated_at"] = datetime.now().isoformat()
                        tasks[i] = t
                        self._save_tasks(tasks)
                        return {"success": True, "response": f"Subtask {subtask_id} completed"}
                return {"success": False, "error": f"Subtask not found: {subtask_id}"}
        return {"success": False, "error": f"Task not found: {task_id}"}

    # -- Dispatch -----------------------------------------------------------

    def _extract_task_info(self, action: str) -> dict:
        result = {}
        # Title in quotes
        quote_match = re.search(r'["\']([^"\']+)["\']', action)
        if quote_match:
            result["title"] = quote_match.group(1)
        # Priority
        for p in PRIORITIES:
            if p in action.lower():
                result["priority"] = p
                break
        # Project
        proj_match = re.search(r'project:\s*(\S+)', action, re.IGNORECASE)
        if proj_match:
            result["project"] = proj_match.group(1)
        # Deadline
        dl_match = re.search(r'(?:deadline|due|by)\s*:?\s*(\d{4}-\d{2}-\d{2})', action, re.IGNORECASE)
        if dl_match:
            result["deadline"] = dl_match.group(1)
        return result

    def execute(self, action: str, **kwargs) -> dict:
        action_lower = action.lower().strip()

        # Board view
        if action_lower.startswith("board"):
            project = kwargs.get("project")
            if not project and len(action.split()) > 1:
                project = action.split(None, 1)[-1]
            return self.board(project=project)

        # Projects
        if action_lower in ("projects", "list_projects"):
            return self.list_projects()

        # Overdue
        if action_lower in ("overdue", "past_due"):
            return self.overdue()

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
                    elif p.lower() in STATUSES:
                        status = status or p.lower()
            if task_id and status:
                return self.move_task(task_id, status)
            return {"success": False, "error": "Usage: move <task_id> <status>"}

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
        priority = kwargs.get("priority", "medium")
        description = kwargs.get("description", "")
        tags = kwargs.get("tags", [])
        deadline = kwargs.get("deadline")

        if not title:
            extracted = self._extract_task_info(action)
            title = title or extracted.get("title")
            priority = extracted.get("priority", priority)
            project = extracted.get("project", project)
            deadline = extracted.get("deadline", deadline)

        if not title:
            # Use action text as title if it looks like a task
            cleaned = re.sub(r'^add\s+', '', action, flags=re.IGNORECASE).strip()
            if cleaned and len(cleaned) > 2:
                title = cleaned

        if title:
            return self.add_task(title=title, project=project, priority=priority,
                                 description=description, tags=tags, deadline=deadline)

        return {
            "success": False,
            "error": f"Could not parse: {action}. "
                     "Try: 'add <title>', 'list', 'board', 'done <id>', 'move <id> <status>', 'search <query>'"
        }


# Singleton
task_manager_tool = TaskManagerTool()
