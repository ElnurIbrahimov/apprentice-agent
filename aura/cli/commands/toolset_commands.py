"""/toolsets — manage tool groups.

List, enable, and disable toolsets for the current platform.
Mirrors Hermes Agent's `hermes tools` interactive UI pattern.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from ..display import console, show_error, show_info, show_success
from .common import command, TIER_STABLE

logger = logging.getLogger(__name__)


@command("/toolsets", "List and manage tool groups", tier=TIER_STABLE)
def handle_toolsets(agent: Any, arg: str, context: dict) -> Optional[str]:
    """Show, enable, or disable toolsets.

    Usage:
        /toolsets              List all toolsets with status
        /toolset enable <name> Enable a toolset
        /toolset disable <name> Disable a toolset
    """
    parts = (arg or "").strip().split(None, 1)
    sub = parts[0].lower() if parts else "list"

    if sub == "list" or sub == "":
        _list_toolsets()
    elif sub == "enable" and len(parts) >= 2:
        _enable_toolset(parts[1].strip())
    elif sub == "disable" and len(parts) >= 2:
        _disable_toolset(parts[1].strip())
    else:
        console.print("[dim]Usage: /toolsets [list|enable <name>|disable <name>][/dim]")

    return None


def _list_toolsets() -> None:
    """Show a Rich table of all toolsets."""
    from rich.table import Table
    from rich.panel import Panel

    from aura.toolsets import list_toolsets

    toolsets = list_toolsets()
    if not toolsets:
        console.print("[dim]No toolsets defined.[/dim]")
        return

    table = Table(box=None, padding=(0, 1), show_header=True, header_style="bold")
    table.add_column("Toolset", style="bold", width=15)
    table.add_column("Status", width=8, justify="center")
    table.add_column("Tools", width=6, justify="right")
    table.add_column("Description", min_width=40)

    enabled_count = 0
    for ts in toolsets:
        if ts["enabled"]:
            status = "[green]\u2713[/green]"
            enabled_count += 1
        else:
            status = "[dim]\u2717[/dim]"

        table.add_row(
            ts["name"],
            status,
            str(ts["tool_count"]),
            ts["description"],
        )

    console.print()
    console.print(Panel(
        table,
        title=f"[bold cyan]Toolsets  ({enabled_count}/{len(toolsets)} enabled)[/bold cyan]",
        border_style="cyan",
        padding=(1, 2),
    ))
    console.print("  [dim]Use /toolset enable <name> or /toolset disable <name> to toggle.[/dim]")
    console.print("  [dim]Changes take effect on next session (/reset).[/dim]")
    console.print()


def _enable_toolset(name: str) -> None:
    """Enable a toolset in config.yaml."""
    from aura.toolsets import TOOLSETS

    if name not in TOOLSETS:
        console.print(f"[red]Unknown toolset: {name}[/red]")
        console.print(f"[dim]Available: {', '.join(sorted(TOOLSETS.keys()))}[/dim]")
        return

    from aura.config_loader import get_config_value, set_config_value

    # Get current enabled list
    enabled = get_config_value("toolsets.enabled", []) or []
    if name in enabled:
        console.print(f"[yellow]Toolset '{name}' is already enabled.[/yellow]")
        return

    # Remove from disabled if present
    disabled = get_config_value("toolsets.disabled", []) or []
    if name in disabled:
        disabled.remove(name)
        set_config_value("toolsets.disabled", disabled)

    enabled.append(name)
    set_config_value("toolsets.enabled", enabled)

    tool_count = len(TOOLSETS[name]["tools"])
    console.print(f"[green]Enabled toolset '{name}' ({tool_count} tools).[/green]")
    console.print("[dim]Use /reset for changes to take effect.[/dim]")


def _disable_toolset(name: str) -> None:
    """Disable a toolset in config.yaml."""
    from aura.toolsets import TOOLSETS

    if name not in TOOLSETS:
        console.print(f"[red]Unknown toolset: {name}[/red]")
        console.print(f"[dim]Available: {', '.join(sorted(TOOLSETS.keys()))}[/dim]")
        return

    # Don't allow disabling core
    if name == "core":
        show_error("Cannot disable 'core' toolset — it contains essential tools.")
        return

    from aura.config_loader import get_config_value, set_config_value

    # Get current disabled list
    disabled = get_config_value("toolsets.disabled", []) or []
    if name in disabled:
        console.print(f"[yellow]Toolset '{name}' is already disabled.[/yellow]")
        return

    # Remove from enabled if present
    enabled = get_config_value("toolsets.enabled", []) or []
    if name in enabled:
        enabled.remove(name)
        set_config_value("toolsets.enabled", enabled)

    disabled.append(name)
    set_config_value("toolsets.disabled", disabled)

    tool_count = len(TOOLSETS[name]["tools"])
    console.print(f"[green]Disabled toolset '{name}' ({tool_count} tools hidden).[/green]")
    console.print("[dim]Use /reset for changes to take effect.[/dim]")
