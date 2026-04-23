"""CLI subcommand: `aura worktree <name> [--branch B] [--remove]`."""
from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    from aura.cli.display import console
except ImportError:
    from rich.console import Console
    console = Console()


def _resolve_worktree_path(name: str) -> Path:
    root = Path(os.getcwd())
    return root / ".aura-worktrees" / name


def cmd_worktree(args: argparse.Namespace) -> int:
    from aura.tools.git_tool import git_tool

    name = (getattr(args, "worktree_name", "") or "").strip()
    remove = bool(getattr(args, "worktree_remove", False))
    open_new = bool(getattr(args, "worktree_open", False))
    branch = getattr(args, "worktree_branch", None) or name

    if not name and not getattr(args, "worktree_list", False):
        console.print("  Usage: aura worktree <name> [--branch B] [--open]")
        console.print("         aura worktree <name> --remove [--force]")
        console.print("         aura worktree --list")
        return 1

    if getattr(args, "worktree_list", False):
        r = git_tool.worktree_list()
        if not r.get("success"):
            console.print(f"[red]List failed:[/] {r.get('error')}")
            return 1
        from rich.table import Table
        t = Table(show_header=True, header_style="bold", box=None, padding=(0, 1))
        t.add_column("Path", style="cyan")
        t.add_column("Branch")
        t.add_column("HEAD", style="dim")
        for w in r.get("worktrees", []):
            t.add_row(w.get("path", ""), w.get("branch", ""), (w.get("head", "") or "")[:12])
        console.print(t)
        return 0

    path = _resolve_worktree_path(name)

    if remove:
        r = git_tool.worktree_remove(str(path), force=bool(getattr(args, "worktree_force", False)))
        if r.get("success"):
            console.print(f"  [green]Removed[/] {path}")
            return 0
        console.print(f"  [red]Remove failed:[/] {r.get('error')}")
        return 1

    path.parent.mkdir(parents=True, exist_ok=True)
    r = git_tool.worktree_add(str(path), branch=branch)
    if not r.get("success"):
        console.print(f"  [red]Create failed:[/] {r.get('error')}")
        return 1

    console.print(f"  [green]Worktree ready[/]  path=[cyan]{path}[/]  branch=[cyan]{branch}[/]")

    if open_new:
        try:
            if sys.platform.startswith("win"):
                subprocess.Popen(
                    ["cmd", "/c", "start", "cmd", "/k", "aura"],
                    cwd=str(path), close_fds=True,
                )
                console.print(f"  [dim]Opened a new session in {path}[/]")
            else:
                # Prior version spawned `sh -lc "aura"` with no terminal
                # window — aura ran as a background child of the current
                # shell and the user saw nothing. Probe common Linux
                # emulators via shutil.which; fall back to a helpful
                # message if none installed.
                import shutil as _shutil
                _terminals = [
                    ("x-terminal-emulator", ["-e", "aura"]),
                    ("gnome-terminal", ["--", "aura"]),
                    ("konsole", ["-e", "aura"]),
                    ("alacritty", ["-e", "aura"]),
                    ("kitty", ["aura"]),
                    ("wezterm", ["start", "aura"]),
                    ("xfce4-terminal", ["-e", "aura"]),
                    ("xterm", ["-e", "aura"]),
                ]
                launched = False
                for binary, args in _terminals:
                    if _shutil.which(binary):
                        subprocess.Popen([binary, *args], cwd=str(path))
                        launched = True
                        break
                if launched:
                    console.print(f"  [dim]Opened a new session in {path}[/]")
                else:
                    console.print(
                        "  [yellow]No terminal emulator found.[/] "
                        f"Run: [cyan]cd {path} && aura[/]"
                    )
        except Exception as e:
            console.print(f"  [yellow]Open failed:[/] {e}")
    else:
        console.print(f"  [dim]cd {path}  → then run `aura`[/]")
    return 0
