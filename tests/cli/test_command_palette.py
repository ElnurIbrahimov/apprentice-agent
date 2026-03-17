"""Tests for command palette — fuzzy filtering, recency, categories."""
import pytest
from aura.cli.command_palette import (
    PaletteItem,
    _fuzzy_match,
    _sort_items,
    _usage_counts,
    record_usage,
    build_items_from_commands,
    build_palette,
)
from aura.cli.input import SLASH_COMMANDS


# ── PaletteItem creation ──────────────────────────────────────────────────

def test_palette_item_defaults():
    item = PaletteItem(label="/model")
    assert item.action == "/model"
    assert item.category == "command"
    assert item.description == ""


def test_palette_item_custom_action():
    item = PaletteItem(label="My file", action="open:myfile.py", category="file")
    assert item.action == "open:myfile.py"


# ── Fuzzy matching ─────────────────────────────────────────────────────────

def test_fuzzy_match_label_substring():
    item = PaletteItem(label="/model", description="View/set model")
    assert _fuzzy_match("mod", item) is True
    assert _fuzzy_match("odel", item) is True


def test_fuzzy_match_description_substring():
    item = PaletteItem(label="/clear", description="Clear conversation history")
    assert _fuzzy_match("conv", item) is True


def test_fuzzy_match_case_insensitive():
    item = PaletteItem(label="/Model", description="View model")
    assert _fuzzy_match("MODEL", item) is True
    assert _fuzzy_match("model", item) is True


def test_fuzzy_match_no_match():
    item = PaletteItem(label="/quit", description="Exit AURA")
    assert _fuzzy_match("model", item) is False


def test_empty_query_matches_all():
    """Empty filter string matches everything — _fuzzy_match is not called for empty,
    but we verify the filtering logic: empty input returns full list."""
    items = [PaletteItem(label="/a"), PaletteItem(label="/b")]
    # Empty query matches all via _fuzzy_match
    assert _fuzzy_match("", items[0]) is True
    assert _fuzzy_match("", items[1]) is True


# ── Recency sorting ───────────────────────────────────────────────────────

def test_recently_used_floats_to_top():
    _usage_counts.clear()
    items = [
        PaletteItem(label="/quit", description="Exit"),
        PaletteItem(label="/model", description="Model"),
        PaletteItem(label="/clear", description="Clear"),
    ]
    record_usage("/model")
    record_usage("/model")
    record_usage("/clear")
    sorted_items = _sort_items(items)
    assert sorted_items[0].label == "/model"
    assert sorted_items[1].label == "/clear"
    assert sorted_items[2].label == "/quit"
    _usage_counts.clear()


def test_no_usage_sorts_alphabetically():
    _usage_counts.clear()
    items = [
        PaletteItem(label="/zebra", category="command"),
        PaletteItem(label="/alpha", category="command"),
        PaletteItem(label="/middle", category="command"),
    ]
    sorted_items = _sort_items(items)
    assert [i.label for i in sorted_items] == ["/alpha", "/middle", "/zebra"]
    _usage_counts.clear()


# ── Item categories ────────────────────────────────────────────────────────

def test_build_palette_categories():
    items = build_palette(
        slash_commands=[("/model", "Set model")],
        recent_files=["main.py"],
        sessions=[{"id": "abc", "title": "Test session"}],
    )
    cats = {it.category for it in items}
    assert "command" in cats
    assert "file" in cats
    assert "session" in cats


def test_file_items_have_path_as_action():
    items = build_palette([], recent_files=["/tmp/foo.py"])
    assert items[0].action == "/tmp/foo.py"
    assert items[0].category == "file"


def test_session_items_action_format():
    items = build_palette([], sessions=[{"id": "s1", "title": "My Chat"}])
    assert items[0].action == "/sessions switch s1"
    assert items[0].category == "session"


# ── Build from SLASH_COMMANDS ──────────────────────────────────────────────

def test_build_items_from_slash_commands():
    items = build_items_from_commands(SLASH_COMMANDS)
    labels = {it.label for it in items}
    assert "/model" in labels
    assert "/quit" in labels
    assert "/clear" in labels
    assert all(it.category == "command" for it in items)
    assert len(items) == len(SLASH_COMMANDS)


def test_slash_command_descriptions_preserved():
    items = build_items_from_commands(SLASH_COMMANDS)
    model_item = next(it for it in items if it.label == "/model")
    assert "model" in model_item.description.lower()
