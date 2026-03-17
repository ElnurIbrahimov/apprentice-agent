"""Theme system for AURA CLI — customizable color schemes."""
from __future__ import annotations
import json
import threading
from pathlib import Path
from typing import Dict, Optional
from dataclasses import dataclass, field, asdict


@dataclass
class AuraTheme:
    """A complete color theme for the CLI."""
    name: str
    # Banner colors (Rich style strings)
    banner_gradient: list = field(default_factory=lambda: ["cyan", "blue", "magenta"])
    # Status bar
    status_bg: str = ""
    status_fg: str = "white"
    status_accent: str = "cyan"
    # Response panel
    response_border: str = "cyan"
    response_header: str = "bold cyan"
    # Tool calls
    tool_color: str = "yellow"
    tool_success: str = "green"
    tool_error: str = "red"
    # Diff colors
    diff_added: str = "green"
    diff_removed: str = "red"
    diff_context: str = "dim"
    diff_header: str = "cyan"
    # General
    prompt_color: str = "cyan"
    info_color: str = "dim"
    warning_color: str = "yellow"
    error_color: str = "red"
    success_color: str = "green"
    # Code blocks
    code_theme: str = "monokai"  # Rich/Pygments syntax theme


# Built-in themes
THEMES: Dict[str, AuraTheme] = {
    "dark": AuraTheme(
        name="dark",
        banner_gradient=["cyan", "blue", "magenta"],
        response_border="cyan",
        code_theme="monokai",
    ),
    "light": AuraTheme(
        name="light",
        banner_gradient=["blue", "purple", "magenta"],
        status_fg="black",
        status_accent="blue",
        response_border="blue",
        response_header="bold blue",
        prompt_color="blue",
        code_theme="friendly",
    ),
    "monokai": AuraTheme(
        name="monokai",
        banner_gradient=["#a6e22e", "#66d9ef", "#ae81ff"],
        status_accent="#66d9ef",
        response_border="#66d9ef",
        response_header="bold #66d9ef",
        tool_color="#e6db74",
        prompt_color="#a6e22e",
        code_theme="monokai",
    ),
    "dracula": AuraTheme(
        name="dracula",
        banner_gradient=["#bd93f9", "#ff79c6", "#50fa7b"],
        status_accent="#bd93f9",
        response_border="#bd93f9",
        response_header="bold #bd93f9",
        tool_color="#f1fa8c",
        tool_success="#50fa7b",
        tool_error="#ff5555",
        prompt_color="#bd93f9",
        diff_added="#50fa7b",
        diff_removed="#ff5555",
        diff_header="#8be9fd",
        code_theme="dracula",
    ),
    "solarized": AuraTheme(
        name="solarized",
        banner_gradient=["#268bd2", "#2aa198", "#859900"],
        status_accent="#268bd2",
        response_border="#268bd2",
        response_header="bold #268bd2",
        tool_color="#b58900",
        tool_success="#859900",
        tool_error="#dc322f",
        prompt_color="#268bd2",
        code_theme="solarized-dark",
    ),
    "nord": AuraTheme(
        name="nord",
        banner_gradient=["#88c0d0", "#81a1c1", "#5e81ac"],
        status_accent="#88c0d0",
        response_border="#88c0d0",
        response_header="bold #88c0d0",
        tool_color="#ebcb8b",
        tool_success="#a3be8c",
        tool_error="#bf616a",
        prompt_color="#88c0d0",
        diff_added="#a3be8c",
        diff_removed="#bf616a",
        diff_header="#88c0d0",
        code_theme="nord-darker",
    ),
}

# Default theme
_current_theme: AuraTheme = THEMES["dark"]
_theme_lock = threading.Lock()
_THEMES_DIR = Path.home() / ".aura" / "themes"


def get_theme() -> AuraTheme:
    """Get the current active theme."""
    with _theme_lock:
        return _current_theme


def set_theme(name: str) -> bool:
    """Set the active theme by name. Returns True if successful."""
    global _current_theme

    # Check built-in themes
    if name in THEMES:
        with _theme_lock:
            _current_theme = THEMES[name]
        return True

    # Check custom themes
    custom = _load_custom_theme(name)
    if custom:
        with _theme_lock:
            _current_theme = custom
        return True

    return False


def list_themes() -> list:
    """List all available theme names (built-in + custom)."""
    names = list(THEMES.keys())
    # Add custom themes
    if _THEMES_DIR.exists():
        for f in _THEMES_DIR.glob("*.json"):
            name = f.stem
            if name not in names:
                names.append(name)
    return names


def _load_custom_theme(name: str) -> Optional[AuraTheme]:
    """Load a custom theme from ~/.aura/themes/."""
    # Reject path traversal attempts
    if "/" in name or "\\" in name or ".." in name:
        return None
    theme_file = _THEMES_DIR / f"{name}.json"
    try:
        theme_file.resolve().relative_to(_THEMES_DIR.resolve())
    except ValueError:
        return None
    if not theme_file.exists():
        return None
    try:
        data = json.loads(theme_file.read_text())
        data["name"] = name
        return AuraTheme(**{k: v for k, v in data.items() if k in AuraTheme.__dataclass_fields__})
    except (json.JSONDecodeError, TypeError, OSError):
        return None


def save_theme_preference(name: str) -> None:
    """Save theme preference to ~/.aura/config.json."""
    config_path = Path.home() / ".aura" / "config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config = {}
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    config["theme"] = name
    config_path.write_text(json.dumps(config, indent=2))


def load_theme_preference() -> str:
    """Load saved theme preference. Returns 'dark' if none saved."""
    config_path = Path.home() / ".aura" / "config.json"
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text())
            return config.get("theme", "dark")
        except (json.JSONDecodeError, OSError):
            pass
    return "dark"
