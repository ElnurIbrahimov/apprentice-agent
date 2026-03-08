"""Persistent status bar for AURA CLI — shows model, project, session info."""

import os
from pathlib import Path
from rich.text import Text


def build_status_bar(
    model: str = "auto",
    project_type: str = "",
    session_title: str = "",
    message_count: int = 0,
    width: int = 80,
    thinking: bool = False,
    elapsed: float = 0.0,
) -> Text:
    """Build a single-line status bar for the AURA CLI footer."""
    bar = Text()

    # Left: CWD + project type
    cwd = os.getcwd()
    home = str(Path.home())
    if cwd.startswith(home):
        cwd = "~" + cwd[len(home):]
    cwd = cwd.replace("\\", "/")
    if len(cwd) > 30:
        cwd = "..." + cwd[-27:]

    bar.append(f" {cwd}", style="dim white")
    if project_type and project_type != "unknown":
        bar.append(f" ({project_type})", style="dim cyan")

    bar.append("  │  ", style="dim")

    # Center: model or thinking indicator
    if thinking:
        bar.append(f"Thinking... ({elapsed:.1f}s)", style="bold yellow")
    else:
        model_short = model.replace(":cloud", "").replace(":latest", "")
        if len(model_short) > 25:
            model_short = model_short[:22] + "..."
        bar.append(model_short, style="bold cyan")

    bar.append("  │  ", style="dim")

    # Right: session info + hint
    if session_title:
        title = session_title[:20]
        bar.append(f'"{title}"', style="dim white")
        if message_count:
            bar.append(f" ({message_count} msgs)", style="dim")
    else:
        bar.append(f"{message_count} msgs", style="dim")

    # Keyboard hint
    hint = "  Ctrl+M model"
    remaining = width - bar.cell_len - len(hint) - 1
    if remaining > 0:
        bar.append(" " * remaining)
        bar.append("Ctrl+M", style="dim bold")
        bar.append(" model", style="dim")

    return bar
