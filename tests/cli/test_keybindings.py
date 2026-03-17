"""Tests for keybindings registry."""
import pytest
from aura.cli.keybindings import KeybindingsRegistry, DEFAULT_KEYBINDINGS

def test_default_keybindings_exist():
    assert "ctrl+l" in DEFAULT_KEYBINDINGS
    assert "ctrl+n" in DEFAULT_KEYBINDINGS
    assert "ctrl+k" in DEFAULT_KEYBINDINGS
    assert "ctrl+g" in DEFAULT_KEYBINDINGS
    assert "escape escape" in DEFAULT_KEYBINDINGS
    assert "shift+tab" in DEFAULT_KEYBINDINGS

def test_registry_get_action():
    reg = KeybindingsRegistry()
    assert reg.get_action("ctrl+l") == "clear_screen"
    assert reg.get_action("ctrl+n") == "new_session"

def test_registry_custom_override():
    custom = {"ctrl+l": "new_session"}
    reg = KeybindingsRegistry(overrides=custom)
    assert reg.get_action("ctrl+l") == "new_session"

def test_registry_unknown_key():
    reg = KeybindingsRegistry()
    assert reg.get_action("ctrl+q") is None

def test_registry_get_key_for_action():
    reg = KeybindingsRegistry()
    assert reg.get_key_for_action("clear_screen") == "ctrl+l"

def test_registry_get_key_for_unknown_action():
    reg = KeybindingsRegistry()
    assert reg.get_key_for_action("nonexistent") is None

def test_registry_all_bindings():
    reg = KeybindingsRegistry()
    bindings = reg.all_bindings()
    assert len(bindings) == len(DEFAULT_KEYBINDINGS)
    assert bindings is not reg._bindings  # Should be a copy
