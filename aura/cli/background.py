# aura/cli/background.py
"""Background agent mode — run tasks asynchronously with notifications."""
from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

from rich.console import Console
from rich.table import Table


class TaskState(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class BackgroundTask:
    """A background agent task."""
    id: str
    prompt: str
    state: TaskState = TaskState.RUNNING
    result: str = ""
    error: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    iterations: int = 0
    thread: Optional[threading.Thread] = field(default=None, repr=False)
    _cancelled: bool = field(default=False, repr=False)

    @property
    def elapsed(self) -> float:
        end = self.end_time or time.time()
        return end - self.start_time if self.start_time else 0.0

    @property
    def elapsed_str(self) -> str:
        e = self.elapsed
        if e < 60:
            return f"{e:.0f}s"
        return f"{int(e // 60)}m{int(e % 60)}s"


class BackgroundManager:
    """Manages background agent tasks."""

    def __init__(self, max_tasks: int = 10):
        self._tasks: Dict[str, BackgroundTask] = {}
        self._lock = threading.Lock()
        self._max_tasks = max_tasks
        self._on_complete: Optional[Callable[[BackgroundTask], None]] = None

    def set_completion_callback(self, callback: Callable[[BackgroundTask], None]) -> None:
        """Set callback fired when a background task completes."""
        self._on_complete = callback

    def submit(
        self,
        prompt: str,
        execute_fn: Callable[[str], Dict],
    ) -> Optional[BackgroundTask]:
        """Submit a task to run in the background. Returns the task or None if at capacity."""
        # Auto-prune old completed tasks before adding new ones
        self.prune_completed()

        with self._lock:
            running = sum(1 for t in self._tasks.values() if t.state == TaskState.RUNNING)
            if running >= self._max_tasks:
                return None

        task_id = f"bg_{uuid.uuid4().hex[:6]}"
        task = BackgroundTask(
            id=task_id,
            prompt=prompt[:200],
            start_time=time.time(),
        )

        def _worker():
            try:
                result = execute_fn(prompt)
                if task._cancelled:
                    return  # Cancelled while running; don't overwrite state
                task.end_time = time.time()
                task.iterations = result.get("iterations", 0)
                if result.get("success"):
                    task.state = TaskState.COMPLETED
                    task.result = result.get("response", "")[:2000]
                else:
                    task.state = TaskState.FAILED
                    task.error = result.get("error", "Unknown error")[:500]
            except Exception as e:
                if task._cancelled:
                    return
                task.end_time = time.time()
                task.state = TaskState.FAILED
                task.error = str(e)[:500]

            if self._on_complete:
                try:
                    self._on_complete(task)
                except Exception as e:
                    logger.debug(f"[Background] on_complete callback failed for task {task_id}: {e}")

        thread = threading.Thread(target=_worker, daemon=True, name=f"bg-{task_id}")
        task.thread = thread

        with self._lock:
            self._tasks[task_id] = task

        thread.start()
        return task

    def get_task(self, task_id: str) -> Optional[BackgroundTask]:
        """Get a task by ID."""
        with self._lock:
            return self._tasks.get(task_id)

    def list_tasks(self) -> List[BackgroundTask]:
        """List all tasks, most recent first."""
        with self._lock:
            return sorted(self._tasks.values(), key=lambda t: t.start_time, reverse=True)

    def cancel(self, task_id: str) -> bool:
        """Mark a task as failed (thread cannot be killed, but state is updated)."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task and task.state == TaskState.RUNNING:
                task._cancelled = True  # Prevent worker from overwriting state
                task.state = TaskState.FAILED
                task.error = "Cancelled by user"
                task.end_time = time.time()
                return True
        return False

    def prune_completed(self, max_age: float = 3600) -> int:
        """Remove completed/failed tasks older than max_age seconds."""
        now = time.time()
        with self._lock:
            to_remove = [
                tid for tid, t in self._tasks.items()
                if t.state != TaskState.RUNNING and (now - t.end_time) > max_age
            ]
            for tid in to_remove:
                del self._tasks[tid]
            return len(to_remove)

    def shutdown(self, timeout: float = 5.0) -> None:
        """Cancel all running tasks and clean up."""
        with self._lock:
            for task_id, task in list(self._tasks.items()):
                if task.state == TaskState.RUNNING:
                    task._cancelled = True
                    task.state = TaskState.FAILED
                    task.error = "Shutdown"
                    task.end_time = time.time()

        # Give threads a moment to finish
        deadline = time.monotonic() + timeout
        for task_id, task in list(self._tasks.items()):
            remaining = deadline - time.monotonic()
            if remaining > 0 and task.thread and task.thread.is_alive():
                task.thread.join(timeout=remaining)

        with self._lock:
            self._tasks.clear()

    @property
    def running_count(self) -> int:
        with self._lock:
            return sum(1 for t in self._tasks.values() if t.state == TaskState.RUNNING)


def render_tasks_table(console: Console, tasks: List[BackgroundTask]) -> None:
    """Render background tasks as a Rich table."""
    if not tasks:
        console.print("[dim]No background tasks.[/dim]")
        return

    table = Table(title="Background Tasks", border_style="cyan")
    table.add_column("ID", style="dim", width=10)
    table.add_column("Task", min_width=30)
    table.add_column("Status", width=12)
    table.add_column("Time", width=8)

    state_styles = {
        TaskState.RUNNING: ("◉", "cyan"),
        TaskState.COMPLETED: ("●", "green"),
        TaskState.FAILED: ("✗", "red"),
    }

    for task in tasks[:20]:
        icon, style = state_styles.get(task.state, ("?", ""))
        table.add_row(
            task.id,
            task.prompt[:50],
            f"[{style}]{icon} {task.state.value}[/{style}]",
            task.elapsed_str,
        )

    console.print(table)


def notify_completion(task: BackgroundTask) -> None:
    """Send desktop notification when a background task completes."""
    try:
        import os
        if os.name == "nt":
            try:
                from winotify import Notification
                toast = Notification(
                    app_id="AURA",
                    title=f"Task {task.state.value}: {task.prompt[:40]}",
                    msg=task.result[:100] if task.result else task.error[:100],
                )
                toast.show()
                return
            except ImportError:
                pass
        # Fallback: bell character
        print("\a", end="", flush=True)
    except Exception as e:
        logger.debug(f"[Background] Task completion desktop notification failed: {e}")


def notify_operation_complete(operation: str, summary: str, success: bool = True) -> None:
    """Send desktop notification for any completed long operation."""
    try:
        import os
        if os.name == "nt":
            try:
                from winotify import Notification
                toast = Notification(
                    app_id="AURA",
                    title=f"{'Done' if success else 'Failed'}: {operation[:40]}",
                    msg=summary[:100],
                )
                toast.show()
                return
            except ImportError:
                pass
        # Fallback: terminal bell
        print("\a", end="", flush=True)
    except Exception as e:
        logger.debug(f"[Background] Operation complete desktop notification failed: {e}")


def create_background_indicator(manager: BackgroundManager) -> str:
    """Status bar indicator for running background tasks."""
    count = manager.running_count
    if count == 0:
        return ""
    return f"[cyan]⚡{count} bg[/cyan]"
