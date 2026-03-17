"""Syntax-highlighted inline diff viewer for the terminal."""
from __future__ import annotations
import difflib
from typing import Optional

from rich.panel import Panel
from rich.text import Text


def generate_diff(old: str, new: str, filename: str = "file") -> str:
    """Generate unified diff string between old and new content."""
    old_lines = old.splitlines(keepends=True)
    new_lines = new.splitlines(keepends=True)
    diff_lines = list(difflib.unified_diff(
        old_lines, new_lines,
        fromfile=f"a/{filename}", tofile=f"b/{filename}",
        lineterm="",
    ))
    if not diff_lines:
        return ""
    return "\n".join(diff_lines)


def diff_summary(old: str, new: str, filename: str = "file") -> str:
    """One-line summary: 'test.py (+3/-1)'."""
    old_lines = old.splitlines()
    new_lines = new.splitlines()
    added = removed = 0
    for line in difflib.unified_diff(old_lines, new_lines, lineterm=""):
        if line.startswith("+") and not line.startswith("+++"):
            added += 1
        elif line.startswith("-") and not line.startswith("---"):
            removed += 1
    if added == 0 and removed == 0:
        return f"{filename} (no changes)"
    return f"{filename} ([green]+{added}[/green]/[red]-{removed}[/red])"


def render_diff(old: str, new: str, filename: str = "file", context_lines: int = 3) -> Optional[Panel]:
    """Render a syntax-highlighted diff as a Rich Panel."""
    diff_text = generate_diff(old, new, filename)
    if not diff_text:
        return None

    text = Text()
    for line in diff_text.split("\n"):
        if line.startswith("+++") or line.startswith("---"):
            text.append(line + "\n", style="bold")
        elif line.startswith("@@"):
            text.append(line + "\n", style="cyan")
        elif line.startswith("+"):
            text.append(line + "\n", style="green")
        elif line.startswith("-"):
            text.append(line + "\n", style="red")
        else:
            text.append(line + "\n", style="dim")

    summary = diff_summary(old, new, filename)
    return Panel(text, title=f"[bold]{summary}[/bold]", border_style="dim", padding=(0, 1))


def render_diff_compact(old: str, new: str, filename: str = "file") -> str:
    """Compact one-line diff summary for tool call display."""
    return f"▸ [yellow]edit[/yellow] {diff_summary(old, new, filename)}"
