"""Tests for emotional context display."""
import pytest
from aura.cli.mood_display import (
    get_mood_emoji, get_mood_label, create_mood_indicator,
    render_mood_detail, format_dream_insight, format_break_suggestion,
)
from rich.console import Console
from io import StringIO

def test_mood_emoji_happy():
    emoji = get_mood_emoji(0.5, 0.0, 0.0)
    assert emoji == "\U0001f60a"

def test_mood_emoji_excited():
    emoji = get_mood_emoji(0.5, 0.5, 0.0)
    assert emoji == "\U0001f525"

def test_mood_emoji_calm():
    emoji = get_mood_emoji(0.5, -0.5, 0.0)
    assert emoji == "\U0001f60c"

def test_mood_emoji_neutral():
    emoji = get_mood_emoji(0.0, 0.0, 0.0)
    assert emoji == "\U0001f610"

def test_mood_label():
    assert get_mood_label(0.5, 0.5, 0.0) == "excited"
    assert get_mood_label(0.0, 0.0, 0.0) == "neutral"
    assert get_mood_label(-0.5, 0.0, 0.0) == "sad"

def test_mood_indicator_none():
    assert create_mood_indicator(None) == ""

def test_mood_indicator_with_state():
    state = {"mood": {"pad": {"pleasure": 0.5, "arousal": 0.1, "dominance": 0.0}}}
    indicator = create_mood_indicator(state)
    assert len(indicator) > 0

def test_render_mood_detail():
    state = {
        "mood": {"pad": {"pleasure": 0.3, "arousal": -0.2, "dominance": 0.1}},
        "neuromodulators": {"dopamine": 0.6, "serotonin": 0.5},
        "active_emotions": [{"name": "curious", "intensity": 0.7}],
    }
    console = Console(file=StringIO(), force_terminal=True, width=80)
    render_mood_detail(console, state)
    output = console.file.getvalue()
    assert "Pleasure" in output
    assert "dopamine" in output

def test_dream_insight():
    result = format_dream_insight("Pattern detected in your coding style")
    assert "Dream insight" in result

def test_break_suggestion_short_session():
    assert format_break_suggestion(30, 0.5) is None

def test_break_suggestion_long_session():
    result = format_break_suggestion(65, 0.8)
    assert result is not None
    assert "break" in result.lower()

def test_break_suggestion_very_long():
    result = format_break_suggestion(125, 0.3)
    assert result is not None
