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


def request_project_trust(
    console: Any,
    project_root: str,
    aura_md_path: str,
    hooks_preview: list,
) -> str:
    """Ask the user whether to trust a new/changed AURA.md in the current project.

    Returns 'trust', 'skip_hooks', or 'abort'. A returned 'trust' means the
    caller should persist (project_root, sha256) so a re-open skips this prompt.
    """
    from rich.panel import Panel
    from rich.text import Text

    body = Text()
    body.append("This project has an AURA.md that registers hooks.\n", style="bold")
    body.append("Hooks run as shell commands on events like ", style="dim")
    body.append("session_start, post_edit, post_tool_call", style="bold dim")
    body.append(".\n\n", style="dim")
    body.append("  Project: ", style="dim")
    body.append(project_root + "\n", style="white")
    body.append("  File:    ", style="dim")
    body.append(aura_md_path + "\n\n", style="white")

    if hooks_preview:
        body.append("Hooks defined:\n", style="bold")
        for h in hooks_preview[:10]:
            event = h.get("event", "?") if isinstance(h, dict) else "?"
            command = h.get("command", "?") if isinstance(h, dict) else str(h)
            if len(command) > 60:
                command = command[:60] + "…"
            body.append(f"  · {event}: ", style="cyan")
            body.append(f"{command}\n", style="dim")
        if len(hooks_preview) > 10:
            body.append(f"  … and {len(hooks_preview) - 10} more\n", style="dim")
        body.append("\n")

    body.append("  t", style="bold green")
    body.append("  trust this project (hooks will load; remembered)\n", style="dim")
    body.append("  s", style="bold yellow")
    body.append("  skip hooks (session only; don't load them)\n", style="dim")
    body.append("  n", style="bold red")
    body.append("  abort (exit without loading this project)", style="dim")

    console.print()
    console.print(Panel(
        body,
        title="[yellow]Untrusted project[/yellow]",
        border_style="yellow",
        padding=(0, 1),
    ))
    try:
        resp = console.input("  [bold]choose [t/s/n] [/bold]").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return "abort"
    if resp in ("t", "trust", "y", "yes"):
        return "trust"
    if resp in ("n", "abort", "no"):
        return "abort"
    # Default / empty / "s" → skip_hooks (safest default)
    return "skip_hooks"
