"""Plugin system — composable plugin packages.

Mirrors Hermes Agent's plugins system:
  plugins:
    enabled:
      - web/ddgs
      - image_gen/fal
      - browser/browser_use
    disabled: []

Each plugin is a directory with a plugin.yaml manifest:
  name: web/ddgs
  description: Free DuckDuckGo search
  version: 1.0.0
  commands:
    - web_search_ddgs
"""
from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


PLUGINS_DIR = Path.home() / ".aura" / "plugins"


def get_plugins_config() -> dict:
    """Get the plugins config."""
    try:
        from aura.config_loader import get_config_value
        return get_config_value("plugins", {}) or {}
    except ImportError:
        return {}


def is_plugin_enabled(name: str) -> bool:
    """Check if a plugin is enabled."""
    cfg = get_plugins_config()
    enabled = cfg.get("enabled", [])
    disabled = cfg.get("disabled", [])

    if name in disabled:
        return False
    if not enabled:
        # No explicit enabled list — enable by default
        return True
    return name in enabled


def enable_plugin(name: str) -> bool:
    """Enable a plugin."""
    try:
        from aura.config_loader import set_config_value, get_config_value
        enabled = get_config_value("plugins.enabled", []) or []
        disabled = get_config_value("plugins.disabled", []) or []

        if name in disabled:
            disabled.remove(name)
            set_config_value("plugins.disabled", disabled)
        if name not in enabled:
            enabled.append(name)
            set_config_value("plugins.enabled", enabled)
        return True
    except ImportError:
        return False


def disable_plugin(name: str) -> bool:
    """Disable a plugin."""
    try:
        from aura.config_loader import set_config_value, get_config_value
        enabled = get_config_value("plugins.enabled", []) or []
        disabled = get_config_value("plugins.disabled", []) or []

        if name in enabled:
            enabled.remove(name)
            set_config_value("plugins.enabled", enabled)
        if name not in disabled:
            disabled.append(name)
            set_config_value("plugins.disabled", disabled)
        return True
    except ImportError:
        return False


def list_available_plugins() -> list[dict]:
    """List all plugins found in the plugins directory."""
    plugins = []
    if not PLUGINS_DIR.exists():
        return plugins

    for entry in sorted(PLUGINS_DIR.iterdir()):
        if not entry.is_dir():
            continue
        manifest = entry / "plugin.yaml"
        if not manifest.exists():
            continue
        try:
            import yaml
            data = yaml.safe_load(manifest.read_text(encoding="utf-8"))
            plugins.append({
                "name": data.get("name", entry.name),
                "description": data.get("description", ""),
                "version": data.get("version", "0.0.0"),
                "commands": data.get("commands", []),
                "path": str(entry),
                "enabled": is_plugin_enabled(data.get("name", entry.name)),
            })
        except (ImportError, Exception) as e:
            logger.debug(f"Plugin manifest parse failed: {e}")

    return plugins


def install_plugin(source: str) -> bool:
    """Install a plugin from a git URL or local path."""
    import shutil
    import re as _re
    import subprocess

    PLUGINS_DIR.mkdir(parents=True, exist_ok=True)

    def _safe_name(raw: str) -> str:
        """Sanitize a plugin name and reject path-traversal attempts."""
        safe = _re.sub(r"[^A-Za-z0-9_.-]", "_", raw).strip("._")[:60]
        if not safe or safe in (".", ".."):
            return ""
        # Verify the resolved path stays within PLUGINS_DIR.
        dest = (PLUGINS_DIR / safe).resolve()
        try:
            dest.relative_to(PLUGINS_DIR.resolve())
        except ValueError:
            return ""
        return safe

    if source.startswith("https://") or source.startswith("git@"):
        # Git clone
        raw_name = source.rstrip("/").split("/")[-1].replace(".git", "")
        plugin_name = _safe_name(raw_name)
        if not plugin_name:
            logger.error("Plugin install refused: unsafe name from URL")
            return False
        dest = PLUGINS_DIR / plugin_name
        try:
            subprocess.run(
                ["git", "clone", source, str(dest)],
                capture_output=True, text=True, timeout=60
            )
            console_log(f"Installed plugin '{plugin_name}'")
            return True
        except Exception as e:
            logger.error(f"Plugin install failed: {e}")
            return False
    elif Path(source).exists():
        raw_name = Path(source).name
        plugin_name = _safe_name(raw_name)
        if not plugin_name:
            logger.error("Plugin install refused: unsafe name from local path")
            return False
        dest = PLUGINS_DIR / plugin_name
        if dest.exists():
            return False
        try:
            shutil.copytree(source, dest)
            return True
        except Exception:
            return False
    return False


def console_log(msg: str) -> None:
    """Print a message to console if available."""
    try:
        from aura.cli.display import console
        console.print(msg)
    except ImportError:
        print(msg)
