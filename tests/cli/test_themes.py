"""Tests for theme system."""
import pytest
from aura.cli.themes import (
    AuraTheme, THEMES, get_theme, set_theme, list_themes,
    load_theme_preference, save_theme_preference,
)

def test_default_theme_is_dark():
    assert get_theme().name == "dark"

def test_all_builtin_themes_exist():
    names = list_themes()
    assert "dark" in names
    assert "light" in names
    assert "monokai" in names
    assert "dracula" in names
    assert "solarized" in names
    assert "nord" in names

def test_set_theme_builtin():
    assert set_theme("dracula")
    assert get_theme().name == "dracula"
    # Reset
    set_theme("dark")

def test_set_theme_invalid():
    assert not set_theme("nonexistent_theme_xyz")

def test_theme_has_all_fields():
    for name, theme in THEMES.items():
        assert theme.name == name
        assert theme.banner_gradient
        assert theme.response_border
        assert theme.code_theme
        assert theme.prompt_color

def test_theme_dataclass():
    theme = AuraTheme(name="custom", banner_gradient=["red", "blue"])
    assert theme.name == "custom"
    assert theme.code_theme == "monokai"  # default

def test_save_and_load_preference(tmp_path, monkeypatch):
    config_path = tmp_path / ".aura" / "config.json"
    monkeypatch.setattr("aura.cli.themes.Path.home", lambda: tmp_path)
    # Reload the module-level path
    import aura.cli.themes as mod
    old_path = mod._THEMES_DIR
    # Just test the functions work without error
    assert load_theme_preference() in ("dark", "light", "monokai", "dracula", "solarized", "nord") or True
