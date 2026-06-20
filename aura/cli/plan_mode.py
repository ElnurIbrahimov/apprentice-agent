# aura/cli/plan_mode.py
"""Editable plan mode — display, edit, and track plan execution.

Data classes and parsing logic live in aura.core.planner (shared with core).
This module provides Rich-based rendering and interactive approval.
"""
from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

# Re-export from core for backward compat
from aura.core.planner import (  # noqa: F401
    PLAN_GENERATION_PROMPT,
    ExecutionPlan,
    PlanStep,
    StepStatus,
    parse_plan_from_llm,
)


def render_plan(console: Console, plan: ExecutionPlan) -> None:
    """Render the plan with status indicators."""
    text = Text()
    text.append(f"Goal: {plan.goal}\n", style="bold")
    text.append(f"Approach: {plan.approach}\n\n", style="dim")

    for step in plan.steps:
        # Status icon
        icons = {
            StepStatus.PENDING: "○",
            StepStatus.RUNNING: "◉",
            StepStatus.DONE: "●",
            StepStatus.FAILED: "✗",
            StepStatus.SKIPPED: "◌",
        }
        colors = {
            StepStatus.PENDING: "dim",
            StepStatus.RUNNING: "cyan bold",
            StepStatus.DONE: "green",
            StepStatus.FAILED: "red",
            StepStatus.SKIPPED: "dim",
        }
        icon = icons.get(step.status, "○")
        color = colors.get(step.status, "dim")

        text.append(f"  {icon} ", style=color)
        text.append(f"Step {step.index}: ", style="bold" if step.status == StepStatus.RUNNING else "")
        text.append(step.description + "\n")

        if step.files:
            text.append(f"    Files: {', '.join(step.files)}\n", style="dim")
        if step.error:
            text.append(f"    ⚠ {step.error}\n", style="red")

    text.append(f"\nProgress: {plan.progress}", style="bold cyan")

    console.print(Panel(text, title="[bold cyan]Execution Plan[/bold cyan]", border_style="cyan", padding=(1, 2)))


def show_plan_approval(console: Console, plan: ExecutionPlan) -> str:
    """Render the plan and prompt for approval.

    Returns:
        'y' if approved, 'n' if cancelled, 'e' if user wants to edit.
    """
    render_plan(console, plan)
    console.print()
    console.print("[bold]Approve this plan?[/bold]  [green]y[/green]es / [red]n[/red]o / [yellow]e[/yellow]dit")
    try:
        choice = input("> ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return "n"
    if choice in ("y", "yes"):
        return "y"
    elif choice in ("e", "edit"):
        return "e"
    return "n"


def edit_plan_text(console: Console, plan_text: str) -> str:
    """Open the plan text in an editor for the user to modify.

    Falls back to inline editing if no editor is available.
    Returns the edited plan text.
    """
    import os
    import shlex
    import subprocess
    import tempfile
    from pathlib import Path

    editor_env = os.environ.get("EDITOR") or ("notepad" if os.name == "nt" else "nano")
    # Split so $EDITOR="code --wait" or $EDITOR="vim -O" works — previously the
    # entire string was passed as one argv element and we got ENOENT.
    try:
        editor_argv = shlex.split(editor_env, posix=(os.name != "nt"))
    except ValueError:
        editor_argv = [editor_env]
    if not editor_argv:
        editor_argv = ["notepad" if os.name == "nt" else "nano"]

    with tempfile.NamedTemporaryFile(suffix=".md", delete=False, mode="w", encoding="utf-8") as f:
        f.write(plan_text)
        tmp_path = f.name
    try:
        subprocess.run([*editor_argv, tmp_path], timeout=300, check=False)
        edited = Path(tmp_path).read_text(encoding="utf-8").strip()
        return edited if edited else plan_text
    except (FileNotFoundError, OSError) as e:
        console.print(f"[red]Editor failed: {e}[/red]")
        return plan_text
    finally:
        Path(tmp_path).unlink(missing_ok=True)
