"""
/history — scannable conversation timeline. Shows recent prompts and
responses with model, tokens, tools, and timestamps.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from .common import TIER_STABLE, command

logger = logging.getLogger(__name__)


@command("/history",  "Scannable conversation timeline",                    tier=TIER_STABLE)
def handle_history(agent: Any, arg: str, context: dict) -> Optional[str]:
    from rich.panel import Panel
    from rich.table import Table

    from ..context import get_ctx
    from ..display import console

    ctx = get_ctx()
    if not ctx or not ctx.agentic_loop:
        console.print("  [dim]No active session.[/dim]")
        return None

    loop = ctx.agentic_loop
    history = getattr(loop, "_conversation_history", []) or []

    if not history:
        console.print("  [dim]No conversation history yet.[/dim]")
        return None

    # Parse limit
    limit = 20
    if arg:
        try:
            limit = max(1, min(100, int(arg.strip())))
        except ValueError:
            pass

    # Build display
    entries = _group_into_turns(history, limit)

    table = Table(box=None, padding=(0, 1), show_header=True, header_style="bold")
    table.add_column("#", width=3, justify="right", style="dim")
    table.add_column("Role", width=6, style="bold")
    table.add_column("Content", min_width=50)
    table.add_column("Model", width=18, style="dim")
    table.add_column("Stats", width=14, style="dim", justify="right")

    for i, entry in enumerate(entries, 1):
        role_style = {
            "user": ("You", "cyan"),
            "assistant": ("Aura", "green"),
            "system": ("Sys", "yellow"),
            "tool": ("Tool", "magenta"),
        }.get(entry["role"], (entry["role"][:4].title(), "white"))

        content = entry["content"][:90].replace("\n", " ")
        if len(entry["content"]) > 90:
            content += "..."

        stats = ""
        if entry.get("model"):
            model_short = entry["model"].replace(":cloud", "").replace(":latest", "")
        else:
            model_short = ""

        if entry.get("tokens"):
            stats = f"{entry['tokens']} tok"
        if entry.get("tools"):
            stats = f"{stats} · {entry['tools']} calls" if stats else f"{entry['tools']} calls"

        table.add_row(
            str(i),
            f"[{role_style[1]}]{role_style[0]}[/{role_style[1]}]",
            content,
            model_short[:18] if model_short else "—",
            stats,
        )

    total_msgs = len(history)
    user_msgs = sum(1 for e in entries if e["role"] == "user")
    shown = len(entries)

    console.print()
    console.print(Panel(
        table,
        title="[bold cyan]📜 /history[/bold cyan]",
        subtitle=f"[dim]{shown} turns shown · {user_msgs} prompts · {total_msgs} total messages[/dim]",
        border_style="cyan",
        padding=(1, 2),
    ))
    console.print()

    return None


def _group_into_turns(history: list[dict], limit: int) -> list[dict]:
    """Group raw history into user/assistant pairs with metadata."""
    entries: list[dict] = []

    for msg in history:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if not content:
            continue

        entries.append({
            "role": role,
            "content": str(content)[:200],
            "model": msg.get("model", ""),
            "tokens": msg.get("tokens", 0),
            "tools": msg.get("tool_calls", 0),
        })

    # Return last N entries, newest last
    return entries[-limit:]
