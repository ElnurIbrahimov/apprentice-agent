"""/personality, /completion, /catalog, /update — miscellaneous features.

Personality presets, shell completions, model catalog search, and
self-update. Mirrors Hermes Agent's various utility commands.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from ..display import console, show_error, show_info, show_success
from .common import command, TIER_STABLE

logger = logging.getLogger(__name__)


@command("/personality", "List/switch personality presets", tier=TIER_STABLE)
def handle_personality(agent: Any, arg: str, context: dict) -> Optional[str]:
    """List or switch personality presets.

    Usage:
        /personality              List all available
        /personality <name>        Switch to <name>
    """
    arg = (arg or "").strip()
    if not arg:
        _list_personalities()
    else:
        _switch_personality(arg)
    return None


def _list_personalities() -> None:
    from rich.table import Table
    from rich.panel import Panel

    from aura.personalities import list_personalities, get_active_personality

    all_p = list_personalities()
    active = get_active_personality()
    if not all_p:
        console.print("[dim]No personalities available.[/dim]")
        return

    table = Table(box=None, padding=(0, 1), show_header=True, header_style="bold")
    table.add_column("Personality", style="bold", width=15)
    table.add_column("Type", width=6)
    table.add_column("Preview", min_width=50)

    for p in all_p:
        marker = " [green]<-[/green]" if p["name"] == active else ""
        ptype = "[dim]custom[/dim]" if p["custom"] else "[cyan]built-in[/cyan]"
        table.add_row(
            p["name"] + marker,
            ptype,
            p["description"],
        )

    console.print()
    console.print(Panel(
        table,
        title=f"[bold cyan]Personalities  ({len(all_p)} available, active: {active})[/bold cyan]",
        border_style="cyan",
        padding=(1, 2),
    ))
    console.print("  [dim]Use /personality <name> to switch. Restart to take effect.[/dim]")
    console.print()


def _switch_personality(name: str) -> None:
    from aura.personalities import set_active_personality, get_personality_prompt

    if not get_personality_prompt(name):
        console.print(f"[red]Unknown personality: {name}[/red]")
        return
    if set_active_personality(name):
        console.print(f"[green]Switched to personality '{name}'.[/green]")
        console.print("[dim]Restart or /reset for changes to take effect.[/dim]")
    else:
        show_error("Failed to set personality. Check config.yaml.")


@command("/completion", "Generate shell completion script", tier=TIER_STABLE)
def handle_completion(agent: Any, arg: str, context: dict) -> Optional[str]:
    """Generate shell completion script.

    Usage:
        /completion bash           Print bash completion to stdout
        /completion zsh            Print zsh completion to stdout
        /completion powershell     Print PowerShell completion
    """
    from aura.completions import generate_completion

    shell = (arg or "").strip().lower()
    if not shell:
        console.print("[dim]Usage: /completion bash|zsh|powershell[/dim]")
        return None

    try:
        script = generate_completion(shell)
        print(script)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")

    return None


@command("/catalog", "Search remote model catalog", tier=TIER_STABLE)
def handle_catalog(agent: Any, arg: str, context: dict) -> Optional[str]:
    """Search the remote model catalog.

    Usage:
        /catalog [query]          Search for models (empty = list all)
    """
    from aura.model_catalog import search_catalog, list_catalog_models, is_catalog_enabled

    if not is_catalog_enabled():
        console.print("[dim]Model catalog disabled. Enable with:[/dim]")
        console.print("[dim]  aura config set model_catalog.enabled true[/dim]")
        return None

    query = (arg or "").strip()
    models = search_catalog(query) if query else list_catalog_models()
    if not models:
        console.print(f"[dim]No models found{' matching ' + repr(query) if query else ''}.[/dim]")
        return None

    from rich.table import Table
    table = Table(box=None, padding=(0, 1), show_header=True, header_style="bold")
    table.add_column("Model", style="bold cyan", min_width=30)
    table.add_column("Provider", width=12)
    table.add_column("Context", width=8, justify="right")

    for m in models[:30]:
        table.add_row(
            m.get("id", "?")[:50],
            m.get("provider", "?")[:12],
            f"{m.get('context_length', 0):,}" if m.get('context_length') else "?",
        )

    console.print()
    console.print(f"  [bold]Model Catalog[/bold]  ({len(models)} found)\n")
    console.print(table)
    console.print()
    return None


@command("/update", "Self-update Aura to latest version", tier=TIER_STABLE)
def handle_update(agent: Any, arg: str, context: dict) -> Optional[str]:
    """Pull latest from git and update pip dependencies."""
    from aura.self_update import run_update

    console.print("[cyan]Updating Aura...[/cyan]")
    result = run_update(backup=True, install_deps=True)

    if result["backup_path"]:
        console.print(f"  [dim]Backup: {result['backup_path']}[/dim]")

    if result["success"]:
        console.print(f"[green]{result['message']}[/green]")
    else:
        console.print(f"[red]{result['message']}[/red]")

    return None
