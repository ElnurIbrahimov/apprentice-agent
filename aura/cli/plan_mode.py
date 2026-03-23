# aura/cli/plan_mode.py
"""Editable plan mode — generate, display, edit, and track plan execution."""
from __future__ import annotations
import re
from typing import List, Optional, Dict
from dataclasses import dataclass, field
from enum import Enum
from rich.console import Console
from rich.panel import Panel
from rich.text import Text


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class PlanStep:
    """A single step in an execution plan."""
    index: int
    description: str
    files: List[str] = field(default_factory=list)
    status: StepStatus = StepStatus.PENDING
    result: str = ""
    error: str = ""


@dataclass
class ExecutionPlan:
    """A structured execution plan with steps."""
    goal: str
    approach: str
    steps: List[PlanStep] = field(default_factory=list)

    @property
    def progress(self) -> str:
        done = sum(1 for s in self.steps if s.status in (StepStatus.DONE, StepStatus.SKIPPED))
        total = len(self.steps)
        return f"{done}/{total}"

    @property
    def is_complete(self) -> bool:
        return all(s.status in (StepStatus.DONE, StepStatus.SKIPPED, StepStatus.FAILED) for s in self.steps)

    @property
    def current_step(self) -> Optional[PlanStep]:
        for s in self.steps:
            if s.status == StepStatus.PENDING:
                return s
        return None

    def to_markdown(self) -> str:
        """Convert plan to Markdown checklist."""
        lines = [f"# Plan: {self.goal}", "", f"**Approach:** {self.approach}", ""]
        for step in self.steps:
            if step.status == StepStatus.DONE:
                checkbox = "[x]"
            elif step.status == StepStatus.RUNNING:
                checkbox = "[~]"
            elif step.status == StepStatus.FAILED:
                checkbox = "[!]"
            elif step.status == StepStatus.SKIPPED:
                checkbox = "[-]"
            else:
                checkbox = "[ ]"
            line = f"- {checkbox} **Step {step.index}:** {step.description}"
            if step.files:
                line += f" ({', '.join(step.files)})"
            lines.append(line)
            if step.error:
                lines.append(f"  - ⚠ {step.error}")
        return "\n".join(lines)


def parse_plan_from_llm(response: str) -> ExecutionPlan:
    """Parse an LLM-generated plan response into a structured ExecutionPlan."""
    lines = response.strip().splitlines()

    goal = ""
    approach = ""
    steps = []
    step_idx = 0

    for line in lines:
        line_stripped = line.strip()

        # Extract goal from first heading or "Goal:" line
        if not goal and (line_stripped.startswith("# ") or line_stripped.lower().startswith("goal:")):
            goal = re.sub(r'^#\s*|^goal:\s*', '', line_stripped, flags=re.IGNORECASE).strip()
            goal = re.sub(r'\*\*(.*?)\*\*', r'\1', goal)  # strip bold markdown
            continue

        # Extract approach
        if not approach and line_stripped.lower().startswith("approach:"):
            approach = re.sub(r'^approach:\s*', '', line_stripped, flags=re.IGNORECASE).strip()
            continue

        # Extract steps — look for numbered items or checkbox items
        step_match = re.match(r'^(?:\d+[\.\)]\s*|-\s*\[.\]\s*|-\s+)(.*)', line_stripped)
        if step_match:
            step_idx += 1
            desc = step_match.group(1).strip()
            # Remove bold markers
            desc = re.sub(r'\*\*(.*?)\*\*', r'\1', desc)

            # Extract file references
            files = re.findall(r'`([^`]+\.\w+)`', desc)

            steps.append(PlanStep(
                index=step_idx,
                description=desc,
                files=files,
            ))

    if not goal and steps:
        goal = "Execution Plan"
    if not approach:
        approach = "Step-by-step execution"

    return ExecutionPlan(goal=goal, approach=approach, steps=steps)


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


PLAN_GENERATION_PROMPT = """Analyze this task and create a step-by-step execution plan.

Task: {task}

Respond with a structured plan in this format:
# [Task Title]

Approach: [1-2 sentence approach description]

1. [First step description] (`file.py` if applicable)
2. [Second step description] (`file.py` if applicable)
3. [Continue as needed...]

Keep steps concrete and actionable. Include file paths where relevant. 5-10 steps max."""


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
    import tempfile
    import subprocess
    from pathlib import Path

    editor = os.environ.get("EDITOR", "notepad" if os.name == "nt" else "nano")
    with tempfile.NamedTemporaryFile(suffix=".md", delete=False, mode="w", encoding="utf-8") as f:
        f.write(plan_text)
        tmp_path = f.name
    try:
        subprocess.call([editor, tmp_path])
        edited = Path(tmp_path).read_text(encoding="utf-8").strip()
        return edited if edited else plan_text
    except (FileNotFoundError, OSError) as e:
        console.print(f"[red]Editor failed: {e}[/red]")
        return plan_text
    finally:
        Path(tmp_path).unlink(missing_ok=True)
