"""/config and aura config — config file management commands.

Read and write ~/.aura/config.yaml from inside the CLI or from the shell.
Mirrors Hermes Agent's `hermes config` subcommand pattern.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from ..display import console
from .common import command, TIER_STABLE

logger = logging.getLogger(__name__)


@command("/config", "Show current configuration", tier=TIER_STABLE)
def handle_config(agent: Any, arg: str, context: dict) -> Optional[str]:
    """Show config, get a value, or set a value.

    Usage:
        /config              Show full config
        /config get key      Get a value (dotted path: model.default)
        /config set key val  Set a value (writes to config.yaml)
        /config path         Show config.yaml path
        /config edit         Open config.yaml in $EDITOR
    """
    parts = (arg or "").strip().split(None, 2)
    sub = parts[0].lower() if parts else "show"

    if sub == "show" or sub == "":
        _show_config()
    elif sub == "get" and len(parts) >= 2:
        _get_config_value(parts[1])
    elif sub == "set" and len(parts) >= 3:
        _set_config_value(parts[1], parts[2])
    elif sub == "path":
        _show_config_path()
    elif sub == "edit":
        _edit_config()
    else:
        console.print("[dim]Usage: /config [show|get KEY|set KEY VAL|path|edit][/dim]")

    return None


def _show_config() -> None:
    """Print the current config as YAML."""
    try:
        import yaml
        from aura.config_loader import load_config
        config = load_config(force=True)
        if not config:
            console.print("[dim]No config.yaml found. Defaults are in use.[/dim]")
            console.print(f"[dim]Create one at: {_config_path_str()}[/dim]")
            return
        console.print(yaml.safe_dump(config, default_flow_style=False, allow_unicode=True, sort_keys=False))
    except ImportError:
        _show_config_fallback()
    except Exception as e:
        console.print(f"[red]Failed to load config: {e}[/red]")


def _show_config_fallback() -> None:
    """Show config as dict repr when PyYAML is not available."""
    from aura.config_loader import load_config
    config = load_config(force=True)
    if not config:
        console.print("[dim]No config.yaml found. Defaults are in use.[/dim]")
        return
    import json
    console.print(json.dumps(config, indent=2, default=str))


def _get_config_value(key: str) -> None:
    """Get a single config value by dotted key path."""
    from aura.config_loader import get_config_value
    value = get_config_value(key)
    if value is None:
        console.print(f"[dim]Key '{key}' not set.[/dim]")
    else:
        console.print(f"[cyan]{key}[/cyan] = [green]{value}[/green]")


def _set_config_value(key: str, value: str) -> None:
    """Set a config value and write to config.yaml."""
    from aura.config_loader import set_config_value

    # Try to parse value as YAML (supports true/false, numbers, lists)
    parsed = _parse_value(value)

    success = set_config_value(key, parsed)
    if success:
        console.print(f"[green]Set {key} = {parsed}[/green]")
        console.print("[dim]Restart or /reset for changes to take effect.[/dim]")
    else:
        console.print(f"[red]Failed to set {key}. Check that PyYAML is installed.[/red]")


def _show_config_path() -> None:
    """Print the config.yaml file path."""
    console.print(f"[cyan]{_config_path_str()}[/cyan]")


def _edit_config() -> None:
    """Open config.yaml in $EDITOR."""
    import os
    import subprocess
    from aura.config_loader import get_config_path

    path = get_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("# Aura configuration\n", encoding="utf-8")

    editor = os.environ.get("EDITOR", "notepad" if os.name == "nt" else "vim")
    try:
        subprocess.run([editor, str(path)])
    except FileNotFoundError:
        console.print(f"[red]Editor '{editor}' not found. Set $EDITOR or edit manually:[/red]")
        console.print(f"  [dim]{path}[/dim]")
    except Exception as e:
        console.print(f"[red]Failed to open editor: {e}[/red]")
        console.print(f"  [dim]Edit manually: {path}[/dim]")


def _config_path_str() -> str:
    from aura.config_loader import get_config_path
    return str(get_config_path())


def _parse_value(raw: str) -> Any:
    """Parse a string value into the appropriate Python type.

    Handles: true/false, yes/no, numbers, comma-separated lists, and
    falls back to string for everything else.
    """
    raw = raw.strip()

    # Booleans
    if raw.lower() in ("true", "yes", "on"):
        return True
    if raw.lower() in ("false", "no", "off"):
        return False

    # Numbers
    try:
        if "." in raw:
            return float(raw)
        return int(raw)
    except ValueError:
        pass

    # Comma-separated list (if starts with [ or has commas)
    if raw.startswith("[") or "," in raw:
        # Strip brackets if present
        cleaned = raw.strip("[]")
        items = [item.strip().strip("'\"") for item in cleaned.split(",") if item.strip()]
        if items:
            return items

    # Plain string — strip quotes
    return raw.strip("'\"")
