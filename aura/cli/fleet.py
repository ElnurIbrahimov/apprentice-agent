# aura/cli/fleet.py
"""Parallel sub-agent fleet — decompose tasks and run in parallel."""
from __future__ import annotations
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Callable
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.live import Live


class SubAgentStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


@dataclass
class SubTask:
    """A decomposed sub-task for parallel execution."""
    id: str
    description: str
    files: List[str] = field(default_factory=list)
    status: SubAgentStatus = SubAgentStatus.PENDING
    result: str = ""
    error: str = ""
    elapsed: float = 0.0
    iteration_count: int = 0


@dataclass
class FleetRun:
    """A fleet execution with multiple sub-tasks."""
    goal: str
    tasks: List[SubTask] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0

    @property
    def progress(self) -> str:
        done = sum(1 for t in self.tasks if t.status in (SubAgentStatus.DONE, SubAgentStatus.FAILED))
        return f"{done}/{len(self.tasks)}"

    @property
    def is_complete(self) -> bool:
        return all(t.status in (SubAgentStatus.DONE, SubAgentStatus.FAILED) for t in self.tasks)

    @property
    def elapsed(self) -> float:
        end = self.end_time or time.time()
        return end - self.start_time if self.start_time else 0.0


def render_fleet_dashboard(console: Console, fleet: FleetRun) -> None:
    """Render a live dashboard of fleet execution status."""
    table = Table(title=f"Fleet: {fleet.goal}", border_style="cyan", show_lines=True)
    table.add_column("#", style="dim", width=3)
    table.add_column("Task", min_width=30)
    table.add_column("Status", width=10)
    table.add_column("Time", width=8)
    table.add_column("Result", max_width=40)

    status_styles = {
        SubAgentStatus.PENDING: ("○", "dim"),
        SubAgentStatus.RUNNING: ("◉", "cyan"),
        SubAgentStatus.DONE: ("●", "green"),
        SubAgentStatus.FAILED: ("✗", "red"),
    }

    for task in fleet.tasks:
        icon, style = status_styles.get(task.status, ("?", ""))
        elapsed_str = f"{task.elapsed:.1f}s" if task.elapsed > 0 else "—"
        result_str = task.result[:40] if task.result else (task.error[:40] if task.error else "—")
        table.add_row(
            task.id[:3],
            task.description[:50],
            f"[{style}]{icon} {task.status.value}[/{style}]",
            elapsed_str,
            result_str,
        )

    footer = f"Progress: {fleet.progress} | Elapsed: {fleet.elapsed:.1f}s"
    console.print(Panel(table, subtitle=f"[dim]{footer}[/dim]", border_style="cyan"))


def parse_decomposition(response: str) -> List[SubTask]:
    """Parse LLM task decomposition into SubTask list."""
    import re
    tasks = []
    idx = 0
    for line in response.strip().splitlines():
        line = line.strip()
        match = re.match(r'^(?:\d+[\.\)]\s*|-\s*\[.\]\s*|-\s+)(.*)', line)
        if match:
            idx += 1
            desc = match.group(1).strip()
            desc = re.sub(r'\*\*(.*?)\*\*', r'\1', desc)  # strip bold
            files = re.findall(r'`([^`]+\.\w+)`', desc)
            tasks.append(SubTask(
                id=f"t{idx}",
                description=desc,
                files=files,
            ))
    return tasks


class FleetExecutor:
    """Executes sub-tasks in parallel using thread pool."""

    def __init__(self, max_workers: int = 3, budget_per_task: float = 1.0):
        self._max_workers = max_workers
        self._budget_per_task = budget_per_task
        self._lock = threading.Lock()

    def run(
        self,
        fleet: FleetRun,
        execute_fn: Callable[[str], Dict],
        on_update: Optional[Callable[[FleetRun], None]] = None,
    ) -> FleetRun:
        """Execute all tasks in parallel. execute_fn takes a prompt and returns result dict."""
        fleet.start_time = time.time()

        def _run_task(task: SubTask) -> SubTask:
            task.status = SubAgentStatus.RUNNING
            if on_update:
                on_update(fleet)
            start = time.time()
            try:
                result = execute_fn(task.description)
                task.elapsed = time.time() - start
                if result.get("success"):
                    task.status = SubAgentStatus.DONE
                    task.result = result.get("response", "")[:500]
                    task.iteration_count = result.get("iterations", 0)
                else:
                    task.status = SubAgentStatus.FAILED
                    task.error = result.get("error", "Unknown error")[:200]
            except Exception as e:
                task.elapsed = time.time() - start
                task.status = SubAgentStatus.FAILED
                task.error = str(e)[:200]
            if on_update:
                on_update(fleet)
            return task

        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            futures = {pool.submit(_run_task, task): task for task in fleet.tasks}
            for future in as_completed(futures):
                try:
                    future.result()  # propagate any unhandled exception
                except Exception:
                    pass

        fleet.end_time = time.time()
        return fleet


DECOMPOSITION_PROMPT = """Break this task into 2-5 independent sub-tasks that can be executed in parallel.

Task: {task}

Rules:
- Each sub-task should be independently executable
- Sub-tasks should not depend on each other's output
- Include file paths in backticks where relevant
- Keep each sub-task focused and specific

Respond with a numbered list:
1. [First sub-task]
2. [Second sub-task]
..."""
