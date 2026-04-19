# aura/cli/fleet.py
"""Parallel sub-agent fleet — decompose tasks and run in parallel."""
from __future__ import annotations

import threading
import time
from concurrent.futures import as_completed

from aura.pools import llm_pool
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table


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


def _build_fleet_renderable(fleet: FleetRun):
    """Build a Rich renderable for the fleet dashboard (used by both static and live)."""
    status_styles = {
        SubAgentStatus.PENDING: ("⏳", "dim"),
        SubAgentStatus.RUNNING: ("🔄", "bold cyan"),
        SubAgentStatus.DONE: ("✅", "bold green"),
        SubAgentStatus.FAILED: ("❌", "bold red"),
    }

    table = Table(
        show_header=True,
        header_style="bold white",
        border_style="cyan",
        show_lines=False,
        pad_edge=True,
        expand=True,
    )
    table.add_column("#", style="dim", width=4, justify="center")
    table.add_column("Task", min_width=30, ratio=3)
    table.add_column("Status", width=14, justify="center")
    table.add_column("Time", width=8, justify="right", style="dim")
    table.add_column("Result", max_width=40, ratio=2, no_wrap=True)

    for task in fleet.tasks:
        icon, style = status_styles.get(task.status, ("?", ""))
        elapsed_str = f"[cyan]{task.elapsed:.1f}s[/cyan]" if task.elapsed > 0 else "[dim]—[/dim]"

        if task.status == SubAgentStatus.FAILED and task.error:
            result_str = f"[red]{task.error[:40]}[/red]"
        elif task.result:
            result_str = f"[dim]{task.result[:40]}[/dim]"
        else:
            result_str = "[dim]—[/dim]"

        status_text = f"[{style}]{icon} {task.status.value}[/{style}]"

        # Highlight running tasks
        desc_style = "bold white" if task.status == SubAgentStatus.RUNNING else ""
        desc = f"[{desc_style}]{task.description[:50]}[/{desc_style}]" if desc_style else task.description[:50]

        table.add_row(task.id[:4], desc, status_text, elapsed_str, result_str)

    # Progress bar
    done_count = sum(1 for t in fleet.tasks if t.status in (SubAgentStatus.DONE, SubAgentStatus.FAILED))
    total = len(fleet.tasks) or 1
    pct = done_count / total
    bar_width = 30
    filled = int(pct * bar_width)
    bar = f"[green]{'━' * filled}[/green][dim]{'━' * (bar_width - filled)}[/dim]"

    failed_count = sum(1 for t in fleet.tasks if t.status == SubAgentStatus.FAILED)
    running_count = sum(1 for t in fleet.tasks if t.status == SubAgentStatus.RUNNING)

    status_parts = [f"[bold]{done_count}/{total}[/bold]"]
    if running_count > 0:
        status_parts.append(f"[cyan]{running_count} running[/cyan]")
    if failed_count > 0:
        status_parts.append(f"[red]{failed_count} failed[/red]")
    status_parts.append(f"[dim]{fleet.elapsed:.1f}s elapsed[/dim]")
    subtitle = f"{bar}  {' · '.join(status_parts)}"

    return Panel(
        table,
        title=f"[bold cyan]🚀 Fleet: {fleet.goal}[/bold cyan]",
        subtitle=subtitle,
        border_style="cyan",
        padding=(0, 1),
    )


def render_fleet_dashboard(console: Console, fleet: FleetRun) -> None:
    """Render a static snapshot of the fleet dashboard."""
    console.print(_build_fleet_renderable(fleet))


def run_fleet_live(
    console: Console,
    fleet: FleetRun,
    executor: "FleetExecutor",
    execute_fn: Callable[[str], Dict],
    refresh_rate: float = 4.0,
) -> FleetRun:
    """Run fleet execution with a Rich Live-updating dashboard.

    Args:
        console: Rich Console instance
        fleet: FleetRun with tasks to execute
        executor: FleetExecutor instance
        execute_fn: Function to execute each sub-task
        refresh_rate: Live display refresh rate (per second)

    Returns:
        Completed FleetRun
    """
    lock = threading.Lock()

    def _on_update(f: FleetRun) -> None:
        with lock:
            live.update(_build_fleet_renderable(f))

    with Live(
        _build_fleet_renderable(fleet),
        console=console,
        refresh_per_second=refresh_rate,
        transient=False,
    ) as live:
        result = executor.run(fleet, execute_fn, on_update=_on_update)
        live.update(_build_fleet_renderable(result))

    return result


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
                prompt = task.description
                if task.files:
                    file_context = "Relevant files: " + ", ".join(f"`{f}`" for f in task.files)
                    prompt = f"{file_context}\n\n{prompt}"
                result = execute_fn(prompt)
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

        pool = llm_pool()
        futures = {pool.submit(_run_task, task): task for task in fleet.tasks}
        for future in as_completed(futures):
            try:
                future.result()
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
