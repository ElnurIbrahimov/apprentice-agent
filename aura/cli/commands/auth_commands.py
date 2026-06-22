"""/auth — credential pool management commands.

List, add, remove, and reset API keys in the credential pool.
Mirrors Hermes Agent's `hermes auth list/add/remove/reset` pattern.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from ..display import console
from .common import command, TIER_STABLE

logger = logging.getLogger(__name__)


@command("/auth", "Manage credential pool (list/add/remove/reset)", tier=TIER_STABLE)
def handle_auth(agent: Any, arg: str, context: dict) -> Optional[str]:
    """Credential pool management.

    Usage:
        /auth list [provider]     List all keys with status
        /auth add                 Interactive: add a key to .env
        /auth remove <p> <idx>    Remove a key by provider + index
        /auth reset <provider>    Clear exhaustion status for a provider
    """
    parts = (arg or "").strip().split(None, 1)
    sub = parts[0].lower() if parts else "list"
    sub_arg = parts[1].strip() if len(parts) > 1 else ""

    if sub == "list":
        _auth_list(sub_arg)
    elif sub == "add":
        _auth_add()
    elif sub == "remove" and sub_arg:
        _auth_remove(sub_arg)
    elif sub == "reset" and sub_arg:
        _auth_reset(sub_arg)
    else:
        console.print("[dim]Usage: /auth [list [provider]|add|remove <provider> <idx>|reset <provider>][/dim]")

    return None


def _auth_list(provider_filter: str) -> None:
    """List all credentials with their status."""
    from rich.table import Table
    from rich.panel import Panel

    try:
        from aura.providers.credential_pool import get_pool
        pool = get_pool()
    except ImportError:
        console.print("[red]Credential pool not available.[/red]")
        return

    # Get all provider names from the pool
    all_providers = sorted(pool._pools.keys())

    if not all_providers:
        console.print("[dim]No credentials registered. Use /auth add to add API keys.[/dim]")
        return

    table = Table(box=None, padding=(0, 1), show_header=True, header_style="bold")
    table.add_column("Provider", style="bold", width=15)
    table.add_column("#", width=4, justify="right")
    table.add_column("Status", width=10)
    table.add_column("Key Preview", min_width=20)
    table.add_column("Cooldown", width=12)

    import time

    for provider_name in all_providers:
        if provider_filter and provider_filter.lower() not in provider_name.lower():
            continue

        ppool = pool._pools.get(provider_name)
        if not ppool or not ppool.keys:
            table.add_row(provider_name, "0", "[dim]no keys[/dim]", "", "")
            continue

        for i, key_entry in enumerate(ppool.keys):
            # Determine status
            now = time.time()
            if key_entry.is_available:
                status = "[green]available[/green]"
                cooldown = ""
            else:
                remaining = key_entry.cooldown_until - now
                if remaining > 60:
                    cooldown = f"{int(remaining / 60)}m"
                else:
                    cooldown = f"{int(remaining)}s"
                if key_entry.failure_count > 5:
                    status = "[red]exhausted[/red]"
                else:
                    status = "[yellow]cooling[/yellow]"

            # Mask key — show first 4 + last 4
            key = key_entry.key
            if len(key) > 12:
                preview = f"{key[:4]}...{key[-4:]}"
            else:
                preview = f"{key[:2]}..."

            table.add_row(
                provider_name if i == 0 else "",
                str(i + 1),
                status,
                preview,
                cooldown,
            )

    console.print()
    console.print(Panel(
        table,
        title="[bold cyan]Credential Pool[/bold cyan]",
        border_style="cyan",
        padding=(1, 2),
    ))


def _auth_add() -> None:
    """Interactive: prompt for provider + key, append to .env."""
    console.print("[bold]Add API Key[/bold]\n")

    # Show known providers
    try:
        from aura.providers.registry import PROVIDER_CONFIGS
        known = sorted(PROVIDER_CONFIGS.keys())
        console.print(f"[dim]Known providers: {', '.join(known)}[/dim]")
    except ImportError:
        known = []

    try:
        provider = console.input("  Provider name: ").strip().lower()
        if not provider:
            console.print("[dim]Cancelled.[/dim]")
            return

        key = console.input("  API key: ").strip()
        if not key:
            console.print("[dim]Cancelled.[/dim]")
            return

        # Determine env var name
        env_var = ""
        if provider in known:
            env_var = PROVIDER_CONFIGS[provider].get("env_var", "")
        if not env_var:
            env_var = f"{provider.upper()}_API_KEY"

        # Append to .env
        from aura.config_loader import get_env_path
        env_path = get_env_path()
        env_path.parent.mkdir(parents=True, exist_ok=True)

        # Check if key already exists
        existing = ""
        if env_path.exists():
            existing = env_path.read_text(encoding="utf-8")

        # Append or update
        lines = existing.splitlines() if existing else []
        found = False
        for i, line in enumerate(lines):
            if line.startswith(f"{env_var}="):
                lines[i] = f"{env_var}={key}"
                found = True
                break

        if not found:
            lines.append(f"{env_var}={key}")

        env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        # Reload the credential pool
        try:
            from aura.providers.credential_pool import get_pool
            get_pool().register(provider, env_var)
        except Exception:
            pass

        console.print(f"\n[green]Added {provider} key to {env_path}[/green]")
        console.print(f"[dim]Env var: {env_var}[/dim]")
        console.print("[dim]Restart or /reset for changes to take effect.[/dim]")

    except (EOFError, KeyboardInterrupt):
        console.print("\n[dim]Cancelled.[/dim]")


def _auth_remove(args: str) -> None:
    """Remove a key by provider + index."""
    parts = args.split()
    if len(parts) < 2:
        console.print("[dim]Usage: /auth remove <provider> <index>[/dim]")
        return

    provider = parts[0].lower()
    try:
        idx = int(parts[1]) - 1  # 1-indexed display
    except ValueError:
        console.print("[red]Index must be a number.[/red]")
        return

    try:
        from aura.providers.credential_pool import get_pool
        pool = get_pool()
        ppool = pool._pools.get(provider)
        if not ppool or idx < 0 or idx >= len(ppool.keys):
            console.print(f"[red]Invalid index for {provider}. Use /auth list {provider} to see indices.[/red]")
            return

        # Remove from env file
        env_var = ppool.env_var
        key_to_remove = ppool.keys[idx].key

        from aura.config_loader import get_env_path
        env_path = get_env_path()
        if env_path.exists():
            lines = env_path.read_text(encoding="utf-8").splitlines()
            # Find and remove the key from the comma-separated list
            for i, line in enumerate(lines):
                if line.startswith(f"{env_var}="):
                    value = line[len(env_var) + 1:]
                    keys = [k.strip() for k in value.replace(";", ",").split(",") if k.strip()]
                    keys = [k for k in keys if k != key_to_remove]
                    if keys:
                        lines[i] = f"{env_var}={','.join(keys)}"
                    else:
                        lines.pop(i)
                    break
            env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        # Reload pool
        ppool.load_from_env()

        console.print(f"[green]Removed key #{idx + 1} from {provider}.[/green]")

    except ImportError:
        console.print("[red]Credential pool not available.[/red]")
    except Exception as e:
        console.print(f"[red]Failed to remove key: {e}[/red]")


def _auth_reset(provider: str) -> None:
    """Clear exhaustion status for all keys of a provider."""
    try:
        from aura.providers.credential_pool import get_pool
        pool = get_pool()
        ppool = pool._pools.get(provider)
        if not ppool:
            console.print(f"[red]Provider '{provider}' not found in pool.[/red]")
            return

        import time
        count = 0
        for key_entry in ppool.keys:
            key_entry.cooldown_until = 0.0
            key_entry.failure_count = 0
            count += 1

        console.print(f"[green]Reset {count} key(s) for {provider}. All keys now available.[/green]")

    except ImportError:
        console.print("[red]Credential pool not available.[/red]")
    except Exception as e:
        console.print(f"[red]Failed to reset: {e}[/red]")
