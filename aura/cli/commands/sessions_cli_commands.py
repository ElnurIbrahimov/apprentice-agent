"""/sessions and aura sessions — session management CLI.

List, browse, export, rename, delete, and prune sessions.
Mirrors Hermes Agent's `hermes sessions` subcommand.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Optional

from ..display import console, show_error
from .common import command, TIER_STABLE

logger = logging.getLogger(__name__)


@command("/sm", "Session management (list/export/rename/delete/stats/prune)", tier=TIER_STABLE)
def handle_sessions(agent: Any, arg: str, context: dict) -> Optional[str]:
    """Session management.

    Usage:
        /sessions              List recent sessions
        /sessions export ID    Export session to file
        /sessions rename ID T  Rename a session
        /sessions delete ID    Delete a session
        /sessions stats        Show session statistics
    """
    parts = (arg or "").strip().split(None, 2)
    sub = parts[0].lower() if parts else "list"

    if sub == "list" or sub == "":
        _list_sessions()
    elif sub == "export" and len(parts) >= 2:
        _export_session(parts[1].strip())
    elif sub == "rename" and len(parts) >= 3:
        _rename_session(parts[1].strip(), parts[2].strip())
    elif sub == "delete" and len(parts) >= 2:
        _delete_session(parts[1].strip())
    elif sub == "stats":
        _session_stats()
    elif sub == "prune" and len(parts) >= 2:
        _prune_sessions(parts[1].strip())
    else:
        console.print("[dim]Usage: /sessions [list|export ID|rename ID TITLE|delete ID|stats|prune DAYS][/dim]")

    return None


def _get_sessions_dir() -> Path:
    """Get the sessions directory (profile-aware)."""
    from aura.config_loader import get_aura_home
    profile = os.environ.get("AURA_PROFILE", "default")
    if profile == "default":
        return Path.cwd() / "data" / "agentic_sessions"
    return get_aura_home() / "profiles" / profile / "sessions"


def _load_session_summaries() -> list[dict]:
    """Load all session summaries from disk."""
    sessions_dir = _get_sessions_dir()
    if not sessions_dir.exists():
        return []

    sessions = []
    for d in sessions_dir.iterdir():
        if not d.is_dir():
            continue
        session_file = d / "session.json"
        if not session_file.exists():
            continue
        try:
            with open(session_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            sessions.append({
                "id": data.get("id", d.name),
                "title": data.get("title", "Untitled"),
                "model": data.get("model", ""),
                "created_at": data.get("created_at", 0),
                "updated_at": data.get("updated_at", 0),
                "message_count": data.get("stats", {}).get("message_count", 0),
                "tool_calls": data.get("stats", {}).get("tool_calls", 0),
                "path": str(d),
            })
        except (json.JSONDecodeError, OSError):
            continue

    sessions.sort(key=lambda x: x.get("updated_at", 0), reverse=True)
    return sessions


def _list_sessions() -> None:
    """List all sessions."""
    from rich.table import Table
    from rich.panel import Panel

    sessions = _load_session_summaries()
    if not sessions:
        console.print("[dim]No sessions found.[/dim]")
        return

    table = Table(box=None, padding=(0, 1), show_header=True, header_style="bold")
    table.add_column("#", width=4, justify="right")
    table.add_column("ID", style="cyan", width=20)
    table.add_column("Title", min_width=30)
    table.add_column("Msgs", width=5, justify="right")
    table.add_column("Model", width=15)
    table.add_column("Updated", width=12)

    for i, s in enumerate(sessions[:20], 1):
        updated = s.get("updated_at", 0)
        if updated:
            elapsed = time.time() - updated
            if elapsed < 3600:
                updated_str = f"{int(elapsed / 60)}m ago"
            elif elapsed < 86400:
                updated_str = f"{int(elapsed / 3600)}h ago"
            else:
                updated_str = f"{int(elapsed / 86400)}d ago"
        else:
            updated_str = "unknown"

        title = s.get("title", "Untitled")[:40]
        table.add_row(
            str(i),
            s["id"][:18],
            title,
            str(s.get("message_count", 0)),
            (s.get("model") or "auto")[:15],
            updated_str,
        )

    console.print()
    console.print(Panel(
        table,
        title=f"[bold cyan]Sessions  ({len(sessions)} total, showing {min(20, len(sessions))})[/bold cyan]",
        border_style="cyan",
        padding=(1, 2),
    ))
    console.print()


def _export_session(session_id: str) -> None:
    """Export a session to a JSON file."""
    sessions_dir = _get_sessions_dir()
    safe_id = re.sub(r"[^\w\-]", "_", session_id).strip("_")[:24]
    session_file = sessions_dir / session_id / "session.json"
    if not session_file.exists():
        console.print(f"[red]Session '{session_id}' not found.[/red]")
        return

    try:
        data = json.loads(session_file.read_text(encoding="utf-8"))
        outpath = Path.cwd() / f"session_{safe_id}_{int(time.time())}.json"
        outpath.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        console.print(f"[green]Exported to {outpath.name}[/green]")
    except Exception as e:
        console.print(f"[red]Export failed: {e}[/red]")


def _safe_session_path(sessions_dir: Path, session_id: str) -> Path | None:
    """Resolve a session id to a path within sessions_dir, blocking traversal.

    Returns None if session_id contains path-traversal characters or the
    resolved path would escape sessions_dir.
    """
    if not session_id or ".." in session_id or "/" in session_id or "\\" in session_id:
        return None
    target = (sessions_dir / session_id).resolve()
    try:
        target.relative_to(sessions_dir.resolve())
    except ValueError:
        return None
    return target


def _rename_session(session_id: str, title: str) -> None:
    """Rename a session."""
    sessions_dir = _get_sessions_dir()
    session_dir = _safe_session_path(sessions_dir, session_id)
    if session_dir is None:
        console.print(f"[red]Invalid session id: {session_id}[/red]")
        return
    session_file = session_dir / "session.json"
    if not session_file.exists():
        console.print(f"[red]Session '{session_id}' not found.[/red]")
        return

    try:
        data = json.loads(session_file.read_text(encoding="utf-8"))
        data["title"] = title
        session_file.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        console.print(f"[green]Renamed session to: {title}[/green]")
    except Exception as e:
        console.print(f"[red]Rename failed: {e}[/red]")


def _delete_session(session_id: str) -> None:
    """Delete a session."""
    import shutil
    sessions_dir = _get_sessions_dir()
    session_dir = _safe_session_path(sessions_dir, session_id)
    if session_dir is None:
        console.print(f"[red]Invalid session id: {session_id}[/red]")
        return
    if not session_dir.exists():
        console.print(f"[red]Session '{session_id}' not found.[/red]")
        return

    try:
        shutil.rmtree(session_dir)
        console.print(f"[green]Deleted session '{session_id}'.[/green]")
    except Exception as e:
        console.print(f"[red]Delete failed: {e}[/red]")


def _session_stats() -> None:
    """Show session statistics."""
    from rich.panel import Panel
    from rich.text import Text

    sessions = _load_session_summaries()
    if not sessions:
        console.print("[dim]No sessions to analyze.[/dim]")
        return

    total_msgs = sum(s.get("message_count", 0) for s in sessions)
    total_tools = sum(s.get("tool_calls", 0) for s in sessions)
    models = {}
    for s in sessions:
        m = s.get("model", "unknown")
        models[m] = models.get(m, 0) + 1

    text = Text()
    text.append(f"Total sessions: {len(sessions)}\n", style="bold")
    text.append(f"Total messages: {total_msgs}\n", style="cyan")
    text.append(f"Total tool calls: {total_tools}\n")
    text.append("\nModels used:\n", style="bold")
    for m, count in sorted(models.items(), key=lambda x: -x[1]):
        text.append(f"  {m or 'auto':<25} {count} sessions\n", style="dim")

    console.print()
    console.print(Panel(text, title="[bold cyan]Session Stats[/bold cyan]", border_style="cyan", padding=(1, 2)))
    console.print()


def _prune_sessions(days_str: str) -> None:
    """Delete sessions older than N days."""
    try:
        days = int(days_str)
    except ValueError:
        show_error("Days must be a number.")
        return

    sessions_dir = _get_sessions_dir().resolve()
    sessions = _load_session_summaries()
    cutoff = time.time() - (days * 86400)
    old = [s for s in sessions if s.get("updated_at", 0) < cutoff]

    if not old:
        console.print(f"[green]No sessions older than {days} days.[/green]")
        return

    import shutil
    count = 0
    for s in old:
        # Defensive: only delete paths that resolve under sessions_dir.
        try:
            target = Path(s["path"]).resolve()
            target.relative_to(sessions_dir)
        except (ValueError, OSError):
            continue
        try:
            shutil.rmtree(target)
            count += 1
        except Exception:
            pass

    console.print(f"[green]Pruned {count} sessions older than {days} days.[/green]")
