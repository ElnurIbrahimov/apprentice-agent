"""/status, /plugins, /lsp, /autoprune — operational commands.

Status aggregator, plugin management, LSP check, and session auto-prune.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from ..display import console
from .common import command, TIER_STABLE, TIER_BETA

logger = logging.getLogger(__name__)


@command("/status", "Show all component status", tier=TIER_STABLE)
def handle_status_all(agent: Any, arg: str, context: dict) -> Optional[str]:
    """Show unified status of all Aura components."""
    from rich.table import Table
    from rich.panel import Panel

    from aura.status_aggregator import get_full_status

    status = get_full_status()

    table = Table(box=None, padding=(0, 1), show_header=True, header_style="bold")
    table.add_column("Component", style="bold cyan", width=18)
    table.add_column("Status", min_width=50)

    for component, info in status.items():
        if isinstance(info, dict) and "error" in info:
            table.add_row(component, f"[red]error: {info['error']}[/red]")
            continue
        if component == "providers":
            cfg = info.get("configured", 0)
            total = info.get("total", 0)
            status_str = f"{cfg}/{total} configured"
            if info.get("names"):
                status_str += f"  [dim]({', '.join(info['names'][:5])}{'...' if len(info['names']) > 5 else ''})[/dim]"
        elif component == "models":
            status_str = f"default: [cyan]{info.get('default', '?')}[/cyan]"
        elif component == "profile":
            status_str = f"active: [cyan]{info.get('active', '?')}[/cyan]"
        elif component == "toolsets":
            status_str = f"{info.get('enabled', 0)}/{info.get('total', 0)} enabled"
        elif component == "cron":
            status_str = f"{info.get('active', 0)} active / {info.get('total', 0)} total"
        elif component == "sessions":
            status_str = f"{info.get('total', 0)} total"
        elif component == "activity":
            tokens = info.get("tokens_in", 0) + info.get("tokens_out", 0)
            status_str = f"{info.get('interactions', 0)} interactions, {tokens:,} tokens, ${info.get('cost', 0):.4f}"
        elif component == "security":
            mode = info.get("approvals_mode", "?")
            redact = "[green]on[/green]" if info.get("redact_secrets") else "[dim]off[/dim]"
            pii = "[green]on[/green]" if info.get("redact_pii") else "[dim]off[/dim]"
            status_str = f"approvals: {mode}, redact_secrets: {redact}, redact_pii: {pii}"
        elif component == "compression":
            status_str = f"{'enabled' if info.get('enabled') else 'disabled'}, threshold: {info.get('threshold', 0)*100:.0f}%"
        else:
            status_str = str(info)
        table.add_row(component, status_str)

    console.print()
    console.print(Panel(
        table,
        title="[bold cyan]Aura System Status[/bold cyan]",
        border_style="cyan",
        padding=(1, 2),
    ))
    console.print()
    return None


@command("/plugins", "List and manage plugins", tier=TIER_STABLE)
def handle_plugins_cmd(agent: Any, arg: str, context: dict) -> Optional[str]:
    """Plugin management.

    Usage:
        /plugins              List all plugins
        /plugins enable N     Enable a plugin
        /plugins disable N    Disable a plugin
        /plugins install URL  Install a plugin from git/local path
    """
    from aura.plugins import list_available_plugins, enable_plugin, disable_plugin, install_plugin

    parts = (arg or "").strip().split(None, 1)
    sub = parts[0].lower() if parts else "list"

    if sub == "list" or sub == "":
        plugins = list_available_plugins()
        if not plugins:
            console.print("[dim]No plugins found in ~/.aura/plugins/[/dim]")
            console.print("[dim]Add plugins by creating a directory with plugin.yaml[/dim]")
            return None
        for p in plugins:
            status = "[green]\u2713[/green]" if p["enabled"] else "[dim]\u2717[/dim]"
            console.print(f"  {status} [cyan]{p['name']:<20}[/cyan] v{p['version']:<10} [dim]{p['description']}[/dim]")
    elif sub == "enable" and len(parts) >= 2:
        if enable_plugin(parts[1]):
            console.print(f"[green]Enabled plugin '{parts[1]}'.[/green]")
        else:
            console.print(f"[red]Failed to enable plugin '{parts[1]}'.[/red]")
    elif sub == "disable" and len(parts) >= 2:
        if disable_plugin(parts[1]):
            console.print(f"[green]Disabled plugin '{parts[1]}'.[/green]")
        else:
            console.print(f"[red]Failed to disable plugin '{parts[1]}'.[/red]")
    elif sub == "install" and len(parts) >= 2:
        if install_plugin(parts[1]):
            console.print(f"[green]Installed plugin from {parts[1]}[/green]")
        else:
            console.print(f"[red]Failed to install plugin from {parts[1]}[/red]")
    else:
        console.print("[dim]Usage: /plugins [list|enable N|disable N|install URL][/dim]")
    return None


@command("/lsp", "Check LSP server configuration", tier=TIER_STABLE)
def handle_lsp(agent: Any, arg: str, context: dict) -> Optional[str]:
    """Show LSP status and configured servers."""
    from aura.lsp import (
        is_lsp_enabled, get_lsp_servers, get_lsp_wait_mode, get_lsp_wait_timeout,
        detect_language, get_server_command,
    )

    if not is_lsp_enabled():
        console.print("[dim]LSP disabled. Enable with:[/dim]")
        console.print("[dim]  aura config set lsp.enabled true[/dim]")
        return None

    console.print("  [bold cyan]LSP Status[/bold cyan]")
    console.print(f"  Wait mode: {get_lsp_wait_mode()}, timeout: {get_lsp_wait_timeout()}s\n")

    servers = get_lsp_servers()
    if not servers:
        console.print("  [dim]No LSP servers configured.[/dim]")
        console.print("  [dim]Example: aura config set lsp.servers.python 'pylsp --stdio'[/dim]")
        return None

    for lang, cmd in servers.items():
        console.print(f"  [green]\u2713[/green] [cyan]{lang:<12}[/cyan] {cmd}")

    # Test detection
    if arg.strip():
        lang = detect_language(arg.strip())
        if lang:
            cmd = get_server_command(lang)
            if cmd:
                console.print(f"\n  [bold]{arg.strip()}[/bold] \u2192 [cyan]{lang}[/cyan] \u2192 {cmd[0]}")
            else:
                console.print(f"\n  [yellow]No server configured for {lang}[/yellow]")
        else:
            console.print(f"\n  [dim]Could not detect language for {arg.strip()}[/dim]")

    return None


@command("/autoprune", "Run session auto-prune", tier=TIER_BETA)
def handle_autoprune(agent: Any, arg: str, context: dict) -> Optional[str]:
    """Run session auto-prune based on retention settings."""
    from aura.session_prune import run_auto_prune, is_auto_prune_enabled, get_retention_days

    if not is_auto_prune_enabled():
        console.print("[dim]Auto-prune disabled. Enable with:[/dim]")
        console.print("[dim]  aura config set sessions.auto_prune true[/dim]")
        console.print(f"  [dim](retention: {get_retention_days()} days)[/dim]")
        return None

    dry_run = "--dry-run" in (arg or "")
    result = run_auto_prune(dry_run=dry_run)
    action = "Would prune" if dry_run else "Pruned"
    console.print(f"[green]{action} {result['pruned']} sessions older than {get_retention_days()} days.[/green]")
    if result["errors"]:
        console.print(f"[yellow]{result['errors']} errors during prune.[/yellow]")
    return None


@command("/humanize", "Toggle human-like typing delay", tier=TIER_BETA)
def handle_humanize(agent: Any, arg: str, context: dict) -> Optional[str]:
    """Toggle humanize mode (natural typing delay)."""
    from aura.human_delay import get_human_delay_config

    cfg = get_human_delay_config()
    current_mode = cfg.get("mode", "off")
    new_mode = "natural" if current_mode == "off" else "off"

    from aura.config_loader import set_config_value
    if set_config_value("human_delay.mode", new_mode):
        if new_mode == "off":
            console.print("[green]Humanize mode disabled.[/green]")
        else:
            min_ms, max_ms = cfg.get("min_ms", 800), cfg.get("max_ms", 2500)
            console.print(f"[green]Humanize mode enabled ({min_ms}-{max_ms}ms delay).[/green]")
        console.print("[dim]Restart or /reset for changes to take effect.[/dim]")
    return None
