"""Rich-rendered diff display for file edit approval.

Shows syntax-highlighted unified diffs when the agentic loop
proposes file edits, and handles user approval flow.
"""

import difflib

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

console = Console()


def make_unified_diff(path: str, old: str, new: str) -> str:
    """Generate a unified diff string."""
    old_lines = old.splitlines(keepends=True)
    new_lines = new.splitlines(keepends=True)
    diff = difflib.unified_diff(
        old_lines, new_lines,
        fromfile=f"a/{path}",
        tofile=f"b/{path}",
        lineterm="",
    )
    return "\n".join(diff)


def show_diff(path: str, old: str, new: str) -> None:
    """Display a colored diff in the console."""
    diff_text = make_unified_diff(path, old, new)
    if not diff_text:
        console.print(f"  [dim]No changes in {path}[/dim]")
        return

    # Color the diff manually for better visibility
    colored = Text()
    for line in diff_text.split("\n"):
        if line.startswith("+++") or line.startswith("---"):
            colored.append(line + "\n", style="bold")
        elif line.startswith("@@"):
            colored.append(line + "\n", style="cyan")
        elif line.startswith("+"):
            colored.append(line + "\n", style="green")
        elif line.startswith("-"):
            colored.append(line + "\n", style="red")
        else:
            colored.append(line + "\n", style="dim")

    console.print(Panel(colored, title=f"[bold]{path}[/bold]", border_style="dim"))


def show_diff_and_confirm(path: str, old: str, new: str, trust_mode: bool = False) -> bool:
    """Show colored diff and ask for approval.

    Args:
        path: File path being edited
        old: Original file content
        new: New file content
        trust_mode: If True, show diff briefly but auto-approve

    Returns:
        True if approved, False if denied
    """
    show_diff(path, old, new)

    if trust_mode:
        console.print("  [dim]Auto-approved (trust mode)[/dim]")
        return True

    try:
        response = console.input("[bold]Apply? [y/n]: [/bold]").strip().lower()
        return response in ("y", "yes", "")
    except (EOFError, KeyboardInterrupt):
        return False


def show_tool_result_compact(tool_name: str, result: str, max_lines: int = 5) -> None:
    """Show a compact preview of a tool result."""
    lines = result.split("\n")
    if len(lines) <= max_lines:
        preview = result
    else:
        preview = "\n".join(lines[:max_lines]) + f"\n  ... ({len(lines) - max_lines} more lines)"

    console.print(f"  [dim]{preview}[/dim]", highlight=False)
