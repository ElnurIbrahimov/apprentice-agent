"""/cron — scheduled task management.

Create, list, pause, resume, and remove scheduled tasks.
Mirrors Hermes Agent's `hermes cron` subcommand pattern.

Tasks are stored in ~/.aura/cron_jobs.json and executed by the
aura_daemon.py background process.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

from ..context import get_ctx
from ..display import console
from .common import command, TIER_STABLE

logger = logging.getLogger(__name__)

_CRON_FILE = Path.home() / ".aura" / "cron_jobs.json"


def _load_jobs() -> list[dict]:
    """Load cron jobs from disk."""
    if not _CRON_FILE.exists():
        return []
    try:
        return json.loads(_CRON_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []


def _save_jobs(jobs: list[dict]) -> None:
    """Save cron jobs to disk atomically."""
    _CRON_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = _CRON_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(jobs, indent=2, default=str), encoding="utf-8")
    tmp.replace(_CRON_FILE)


def _parse_schedule(raw: str) -> str:
    """Parse a schedule string. Supports: 30m, 2h, daily, 0 9 * * *."""
    raw = raw.strip().lower()
    # Simple interval
    if raw.endswith("m") and raw[:-1].isdigit():
        return f"every {raw}"
    if raw.endswith("h") and raw[:-1].isdigit():
        return f"every {raw}"
    if raw == "daily":
        return "0 9 * * *"
    # Cron expression — pass through
    return raw


@command("/cron", "Manage scheduled tasks", tier=TIER_STABLE)
def handle_cron(agent: Any, arg: str, context: dict) -> Optional[str]:
    """Scheduled task management.

    Usage:
        /cron                        List all jobs
        /cron create <schedule> <prompt>   Create a job
        /cron pause <id>              Pause a job
        /cron resume <id>             Resume a job
        /cron remove <id>             Remove a job
        /cron run <id>                Trigger a job now
    """
    parts = (arg or "").strip().split(None, 2)
    sub = parts[0].lower() if parts else "list"

    if sub == "list" or sub == "":
        _list_jobs()
    elif sub == "create" and len(parts) >= 3:
        _create_job(parts[1], parts[2])
    elif sub == "pause" and len(parts) >= 2:
        _toggle_job(parts[1], enabled=False)
    elif sub == "resume" and len(parts) >= 2:
        _toggle_job(parts[1], enabled=True)
    elif sub == "remove" and len(parts) >= 2:
        _remove_job(parts[1])
    elif sub == "run" and len(parts) >= 2:
        _run_job(agent, parts[1])
    else:
        console.print("[dim]Usage: /cron [list|create SCHEDULE PROMPT|pause ID|resume ID|remove ID|run ID][/dim]")

    return None


def _list_jobs() -> None:
    """List all cron jobs."""
    from rich.table import Table
    from rich.panel import Panel

    jobs = _load_jobs()
    if not jobs:
        console.print("[dim]No scheduled jobs. Use /cron create <schedule> <prompt>[/dim]")
        return

    table = Table(box=None, padding=(0, 1), show_header=True, header_style="bold")
    table.add_column("ID", width=8)
    table.add_column("Schedule", width=18)
    table.add_column("Status", width=8, justify="center")
    table.add_column("Prompt", min_width=40)
    table.add_column("Last Run", width=12)

    for job in jobs:
        status = "[green]active[/green]" if job.get("enabled", True) else "[yellow]paused[/yellow]"
        last_run = job.get("last_run", 0)
        if last_run:
            elapsed = time.time() - last_run
            if elapsed < 3600:
                last_str = f"{int(elapsed / 60)}m ago"
            elif elapsed < 86400:
                last_str = f"{int(elapsed / 3600)}h ago"
            else:
                last_str = f"{int(elapsed / 86400)}d ago"
        else:
            last_str = "never"

        table.add_row(
            job.get("id", "?")[:8],
            job.get("schedule", ""),
            status,
            job.get("prompt", "")[:60],
            last_str,
        )

    console.print()
    console.print(Panel(
        table,
        title=f"[bold cyan]Cron Jobs  ({len(jobs)} total)[/bold cyan]",
        border_style="cyan",
        padding=(1, 2),
    ))
    console.print()


def _create_job(schedule: str, prompt: str) -> None:
    """Create a new cron job."""
    jobs = _load_jobs()
    job_id = f"job_{int(time.time())}_{len(jobs):03d}"
    parsed_schedule = _parse_schedule(schedule)
    jobs.append({
        "id": job_id,
        "schedule": parsed_schedule,
        "prompt": prompt,
        "enabled": True,
        "created_at": time.time(),
        "last_run": 0,
        "run_count": 0,
    })
    _save_jobs(jobs)
    console.print(f"[green]Created job {job_id}[/green]")
    console.print(f"  [dim]Schedule: {parsed_schedule}[/dim]")
    console.print(f"  [dim]Prompt: {prompt[:80]}[/dim]")


def _toggle_job(job_id: str, enabled: bool) -> None:
    """Pause or resume a job."""
    jobs = _load_jobs()
    for job in jobs:
        if job.get("id", "").startswith(job_id):
            job["enabled"] = enabled
            _save_jobs(jobs)
            action = "resumed" if enabled else "paused"
            console.print(f"[green]Job {job['id']} {action}.[/green]")
            return
    console.print(f"[red]Job '{job_id}' not found.[/red]")


def _remove_job(job_id: str) -> None:
    """Remove a cron job."""
    jobs = _load_jobs()
    new_jobs = [j for j in jobs if not j.get("id", "").startswith(job_id)]
    if len(new_jobs) == len(jobs):
        console.print(f"[red]Job '{job_id}' not found.[/red]")
        return
    _save_jobs(new_jobs)
    console.print(f"[green]Removed job matching '{job_id}'.[/green]")


def _run_job(agent: Any, job_id: str) -> None:
    """Trigger a job immediately."""
    jobs = _load_jobs()
    for job in jobs:
        if job.get("id", "").startswith(job_id):
            prompt = job.get("prompt", "")
            console.print(f"[cyan]Running job {job['id']}: {prompt[:60]}[/cyan]")
            # Update last_run
            job["last_run"] = time.time()
            job["run_count"] = job.get("run_count", 0) + 1
            _save_jobs(jobs)
            # Execute via agent
            try:
                result = agent.run(prompt)
                response = result.get("response", "") if result else ""
                if response:
                    console.print(f"[dim]{response[:200]}[/dim]")
            except Exception as e:
                console.print(f"[red]Job failed: {e}[/red]")
            return
    console.print(f"[red]Job '{job_id}' not found.[/red]")
