"""Keyboard shortcut registry with customization support.

Used by input.py to register keybindings. User overrides are loaded
from ~/.aura/keybindings.json.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

ACTION_CLEAR_SCREEN = "clear_screen"
ACTION_NEW_SESSION = "new_session"
ACTION_COMMAND_PALETTE = "command_palette"
ACTION_OPEN_EDITOR = "open_editor"
ACTION_REWIND = "rewind"
ACTION_CYCLE_PERMISSIONS = "cycle_permissions"
ACTION_MODEL_PICKER = "model_picker"
ACTION_SEARCH_HISTORY = "search_history"

DEFAULT_KEYBINDINGS: Dict[str, str] = {
    "ctrl+l": ACTION_CLEAR_SCREEN,
    "ctrl+n": ACTION_NEW_SESSION,
    "ctrl+k": ACTION_COMMAND_PALETTE,
    "ctrl+g": ACTION_OPEN_EDITOR,
    "ctrl+r": ACTION_SEARCH_HISTORY,
    "ctrl+z": ACTION_REWIND,
    "shift+tab": ACTION_CYCLE_PERMISSIONS,
    "alt+m": ACTION_MODEL_PICKER,
}

_KEYBINDINGS_PATH = Path.home() / ".aura" / "keybindings.json"


class KeybindingsRegistry:
    """Manages keyboard shortcuts with user customization."""

    def __init__(self, overrides: Optional[Dict[str, str]] = None):
        self._bindings: Dict[str, str] = dict(DEFAULT_KEYBINDINGS)
        if overrides is None:
            overrides = self._load_user_overrides()
        if overrides:
            self._bindings.update(overrides)

    def _load_user_overrides(self) -> Dict[str, str]:
        if _KEYBINDINGS_PATH.exists():
            try:
                return json.loads(_KEYBINDINGS_PATH.read_text())
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def get_action(self, key_combo: str) -> Optional[str]:
        """Get the action bound to a key combination."""
        return self._bindings.get(key_combo.lower())

    def get_key_for_action(self, action: str) -> Optional[str]:
        """Get the key combo bound to an action (first match)."""
        for key, act in self._bindings.items():
            if act == action:
                return key
        return None

    def all_bindings(self) -> Dict[str, str]:
        """Return all current bindings."""
        return dict(self._bindings)


def parse_key_to_pt(key_str: str) -> tuple:
    """Convert a human-readable key string to prompt_toolkit key args.

    Examples:
        "ctrl+l"     -> ("c-l",)
        "ctrl+n"     -> ("c-n",)
        "alt+m"      -> ("escape", "m")
        "shift+tab"  -> ("s-tab",)
        "ctrl+z"     -> ("c-z",)
    """
    parts = key_str.lower().strip().split("+")
    if len(parts) == 1:
        return (parts[0],)
    modifier = parts[0]
    key = parts[1]
    if modifier == "ctrl":
        return (f"c-{key}",)
    elif modifier == "alt":
        return ("escape", key)
    elif modifier == "shift":
        return (f"s-{key}",)
    return (key_str,)
