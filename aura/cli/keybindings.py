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

    Unsupported combos (e.g. ``ctrl+shift+k``) log a warning so users
    know their custom binding in ``~/.aura/keybindings.json`` didn't take
    effect, instead of silently handing prompt_toolkit a literal string
    it can't match.
    """
    import logging
    _log = logging.getLogger(__name__)

    parts = key_str.lower().strip().split("+")
    if len(parts) == 1:
        return (parts[0],)
    if len(parts) > 2:
        _log.warning(
            "keybinding '%s' uses an unsupported modifier combo; "
            "only single-modifier bindings are supported (ctrl+X, alt+X, shift+X)",
            key_str,
        )
        return (key_str,)  # prompt_toolkit will reject this, but we've warned
    modifier, key = parts
    if modifier == "ctrl":
        return (f"c-{key}",)
    elif modifier == "alt":
        return ("escape", key)
    elif modifier == "shift":
        return (f"s-{key}",)
    _log.warning(
        "keybinding '%s' has unrecognized modifier '%s'; "
        "expected ctrl/alt/shift", key_str, modifier,
    )
    return (key_str,)
