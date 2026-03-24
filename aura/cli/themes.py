"""Theme system for AURA CLI — customizable color schemes."""
from __future__ import annotations
import json
import threading
from pathlib import Path
from typing import Dict, Optional
from dataclasses import dataclass, field, asdict


@dataclass
class AuraTheme:
    """A complete color theme for the CLI with semantic color tokens."""
    name: str
    type: str = "dark"  # "dark" or "light"

    # Identity / brand
    accent: str = "#D777AF"         # Aura's signature pink-purple
    accent_dim: str = "#B0578F"     # Dimmer variant
    gradient: list = field(default_factory=lambda: ["#D777AF", "#B1B9F9", "#87D7D7"])

    # Text
    text_primary: str = "white"
    text_secondary: str = "#999999"
    text_muted: str = "#555555"
    text_accent: str = "#D7AFFF"

    # Status
    success: str = "#4EBA65"
    error: str = "#FF6B80"
    warning: str = "#FFC107"
    info: str = "#87AFFF"

    # Diff
    diff_added_bg: str = "#213A2B"
    diff_removed_bg: str = "#4A221D"
    diff_added_fg: str = "#38A660"
    diff_removed_fg: str = "#B3596B"
    diff_context: str = "dim"
    diff_header: str = "cyan"

    # UI elements
    border: str = "#505050"
    border_active: str = "#87AFFF"
    panel_bg: str = "#1a1a1a"
    input_bg: str = "#3F3F3F"

    # Tool calls
    tool_color: str = "#E6DB74"
    tool_success: str = "#4EBA65"
    tool_error: str = "#FF6B80"
    tool_pending: str = "#87AFFF"

    # Permission prompt
    permission_border: str = "#FFC107"
    permission_accent: str = "#B1B9F9"

    # Prompt
    prompt_color: str = "cyan"

    # Code blocks
    code_theme: str = "monokai"

    # Status bar
    status_bg: str = "#1a1a1a"
    status_fg: str = "#cccccc"

    # --- Backward compatibility aliases ---
    @property
    def banner_gradient(self):
        """Alias for gradient (backward compat)."""
        return self.gradient

    @property
    def response_border(self):
        """Alias for border_active (backward compat)."""
        return self.border_active

    @property
    def response_header(self):
        """Alias for bold border_active (backward compat)."""
        return f"bold {self.border_active}"

    @property
    def status_accent(self):
        """Alias for accent (backward compat)."""
        return self.accent

    @property
    def info_color(self):
        """Alias for text_muted (backward compat)."""
        return self.text_muted

    @property
    def warning_color(self):
        """Alias for warning (backward compat)."""
        return self.warning

    @property
    def error_color(self):
        """Alias for error (backward compat)."""
        return self.error

    @property
    def success_color(self):
        """Alias for success (backward compat)."""
        return self.success

    @property
    def diff_added(self):
        """Alias for diff_added_fg (backward compat)."""
        return self.diff_added_fg

    @property
    def diff_removed(self):
        """Alias for diff_removed_fg (backward compat)."""
        return self.diff_removed_fg


# Built-in themes
THEMES: Dict[str, AuraTheme] = {
    "dark": AuraTheme(
        name="dark", type="dark",
        accent="#D777AF", gradient=["cyan", "blue", "magenta"],
        code_theme="monokai",
    ),
    "light": AuraTheme(
        name="light", type="light",
        accent="#AF5F87", gradient=["blue", "#8B5CF6", "magenta"],
        text_primary="black", text_secondary="#555555", text_muted="#999999",
        diff_added_bg="#dafbe1", diff_removed_bg="#ffebe9",
        diff_added_fg="#1a7f37", diff_removed_fg="#cf222e",
        border="#d0d7de", border_active="#0969da", panel_bg="#f6f8fa",
        input_bg="#e4e4e4", status_bg="#f0f0f0", status_fg="#333333",
        prompt_color="blue", tool_color="#8B5CF6",
        success="#1a7f37", error="#cf222e", warning="#9a6700",
        code_theme="friendly",
    ),
    "monokai": AuraTheme(
        name="monokai", type="dark",
        accent="#a6e22e", gradient=["#a6e22e", "#66d9ef", "#ae81ff"],
        text_accent="#ae81ff", tool_color="#e6db74",
        prompt_color="#a6e22e", border_active="#66d9ef",
        success="#a6e22e", error="#f92672", warning="#e6db74",
        code_theme="monokai",
    ),
    "dracula": AuraTheme(
        name="dracula", type="dark",
        accent="#bd93f9", gradient=["#bd93f9", "#ff79c6", "#50fa7b"],
        text_accent="#bd93f9", tool_color="#f1fa8c",
        prompt_color="#bd93f9", border_active="#bd93f9",
        success="#50fa7b", error="#ff5555", warning="#f1fa8c", info="#8be9fd",
        diff_added_fg="#50fa7b", diff_removed_fg="#ff5555", diff_header="#8be9fd",
        code_theme="dracula",
    ),
    "solarized": AuraTheme(
        name="solarized", type="dark",
        accent="#268bd2", gradient=["#268bd2", "#2aa198", "#859900"],
        tool_color="#b58900", prompt_color="#268bd2", border_active="#268bd2",
        success="#859900", error="#dc322f", warning="#b58900", info="#268bd2",
        code_theme="solarized-dark",
    ),
    "nord": AuraTheme(
        name="nord", type="dark",
        accent="#88c0d0", gradient=["#88c0d0", "#81a1c1", "#5e81ac"],
        text_accent="#88c0d0", tool_color="#ebcb8b",
        prompt_color="#88c0d0", border_active="#88c0d0",
        success="#a3be8c", error="#bf616a", warning="#ebcb8b", info="#81a1c1",
        diff_added_fg="#a3be8c", diff_removed_fg="#bf616a", diff_header="#88c0d0",
        code_theme="nord-darker",
    ),
    "catppuccin": AuraTheme(
        name="catppuccin", type="dark",
        accent="#cba6f7", gradient=["#cba6f7", "#f5c2e7", "#94e2d5"],
        text_primary="#cdd6f4", text_secondary="#a6adc8", text_muted="#585b70",
        text_accent="#cba6f7", tool_color="#f9e2af",
        prompt_color="#cba6f7", border="#45475a", border_active="#cba6f7",
        panel_bg="#1e1e2e", input_bg="#313244", status_bg="#181825", status_fg="#cdd6f4",
        success="#a6e3a1", error="#f38ba8", warning="#f9e2af", info="#89b4fa",
        diff_added_bg="#1e3a2c", diff_removed_bg="#3e1e28",
        diff_added_fg="#a6e3a1", diff_removed_fg="#f38ba8",
        code_theme="monokai",
    ),
    "gruvbox": AuraTheme(
        name="gruvbox", type="dark",
        accent="#fe8019", gradient=["#fe8019", "#fabd2f", "#b8bb26"],
        text_primary="#ebdbb2", text_secondary="#a89984", text_muted="#665c54",
        text_accent="#fe8019", tool_color="#fabd2f",
        prompt_color="#fe8019", border="#504945", border_active="#fe8019",
        panel_bg="#1d2021", input_bg="#3c3836", status_bg="#1d2021", status_fg="#ebdbb2",
        success="#b8bb26", error="#fb4934", warning="#fabd2f", info="#83a598",
        diff_added_bg="#2a3a1e", diff_removed_bg="#3c1f1e",
        diff_added_fg="#b8bb26", diff_removed_fg="#fb4934",
        code_theme="gruvbox-dark",
    ),
    "tokyo-night": AuraTheme(
        name="tokyo-night", type="dark",
        accent="#7aa2f7", gradient=["#7aa2f7", "#bb9af7", "#7dcfff"],
        text_primary="#c0caf5", text_secondary="#9aa5ce", text_muted="#565f89",
        text_accent="#bb9af7", tool_color="#e0af68",
        prompt_color="#7aa2f7", border="#3b4261", border_active="#7aa2f7",
        panel_bg="#1a1b26", input_bg="#292e42", status_bg="#16161e", status_fg="#c0caf5",
        success="#9ece6a", error="#f7768e", warning="#e0af68", info="#7dcfff",
        diff_added_bg="#1a2a1e", diff_removed_bg="#2d1a24",
        diff_added_fg="#9ece6a", diff_removed_fg="#f7768e",
        code_theme="monokai",
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
        # Map legacy field names to new names
        _legacy_map = {
            "banner_gradient": "gradient",
            "diff_added": "diff_added_fg",
            "diff_removed": "diff_removed_fg",
            "error_color": "error",
            "warning_color": "warning",
            "success_color": "success",
            "info_color": "info",
            "response_border": "border_active",
        }
        for old_key, new_key in _legacy_map.items():
            if old_key in data and new_key not in data:
                data[new_key] = data.pop(old_key)
            elif old_key in data:
                del data[old_key]
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
