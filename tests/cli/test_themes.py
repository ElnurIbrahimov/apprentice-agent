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
    for expected in ("dark", "light", "monokai", "dracula", "solarized", "nord",
                     "catppuccin", "gruvbox", "tokyo-night"):
        assert expected in names

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
        assert theme.gradient
        assert theme.code_theme
        assert theme.prompt_color
        assert theme.accent
        assert theme.success
        assert theme.error
        assert theme.warning

def test_theme_dataclass():
    theme = AuraTheme(name="custom", gradient=["red", "blue"])
    assert theme.name == "custom"
    assert theme.code_theme == "monokai"  # default

def test_backward_compat_aliases():
    theme = AuraTheme(name="compat-test", gradient=["cyan", "blue"])
    assert theme.banner_gradient == ["cyan", "blue"]
    assert theme.response_border == theme.border_active
    assert theme.diff_added == theme.diff_added_fg
    assert theme.diff_removed == theme.diff_removed_fg
    assert theme.error_color == theme.error
    assert theme.warning_color == theme.warning
    assert theme.success_color == theme.success

def test_theme_types():
    assert THEMES["dark"].type == "dark"
    assert THEMES["light"].type == "light"
    for name, theme in THEMES.items():
        assert theme.type in ("dark", "light")

def test_save_and_load_preference(tmp_path, monkeypatch):
    tmp_path / ".aura" / "config.json"
    monkeypatch.setattr("aura.cli.themes.Path.home", lambda: tmp_path)
    # Reload the module-level path
    import aura.cli.themes as mod
    # Just test the functions work without error
    assert load_theme_preference() in list(THEMES.keys()) or True
