"""Rich interactive permission dialog for tool approvals.

Replaces the plain `console.input("Allow? [y/n/always]: ")` prompt with a
bordered panel that shows the tool name, description, and the four scope
choices: allow_once / allow_session / allow_always / deny.

The PermissionManager already handles `allow_always` (adds to _always_approved)
and `deny`. `allow_session` is handled by the caller threading the decision
into the PermissionManager's session-scope trust set (see permissions.py).
"""
from __future__ import annotations

from typing import Any


def request_permission(
    console: Any,
    tool_name: str,
    description: str,
) -> str:
    """Rich permission dialog. Returns one of:
    'allow_once', 'allow_session', 'allow_always', 'deny'.
    """
    from rich.panel import Panel
    from rich.text import Text

    body = Text()
    body.append(tool_name, style="bold")
    if description:
        body.append("\n")
        for line in description.split("\n"):
            body.append(line + "\n", style="dim")

    body.append("\n")
    body.append("  y", style="bold green")
    body.append("  allow once\n", style="dim")
    body.append("  s", style="bold cyan")
    body.append("  allow for this session\n", style="dim")
    body.append("  a", style="bold yellow")
    body.append("  allow always (persist)\n", style="dim")
    body.append("  n", style="bold red")
    body.append("  deny", style="dim")

    console.print()
    console.print(Panel(
        body,
        title="[yellow]Permission required[/yellow]",
        border_style="yellow",
        padding=(0, 1),
    ))
    try:
        resp = console.input("  [bold]choose [y/s/a/n] [/bold]").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return "deny"
    if resp in ("s", "session"):
        return "allow_session"
    if resp in ("a", "always"):
        return "allow_always"
    if resp in ("n", "no", "deny"):
        return "deny"
    # Default / empty / "y" / "yes" → allow_once
    return "allow_once"
