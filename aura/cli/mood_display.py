# aura/cli/mood_display.py
"""Emotional context display — mood indicator, /mood command, dream insights."""
from __future__ import annotations

from typing import Dict, Optional

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

# PAD mood to emoji mapping (Pleasure, Arousal, Dominance)
_MOOD_EMOJIS = {
    "happy": "\U0001f60a",
    "excited": "\U0001f525",
    "calm": "\U0001f60c",
    "focused": "\U0001f3af",
    "curious": "\U0001f914",
    "tired": "\U0001f634",
    "neutral": "\U0001f610",
    "sad": "\U0001f614",
    "anxious": "\U0001f630",
    "confident": "\U0001f4aa",
}


def _classify_mood(pleasure: float, arousal: float, dominance: float) -> str:
    """Map PAD values (-1 to 1) to a mood label. Single source of truth for
    the PAD→mood mapping used by both ``get_mood_label`` and ``get_mood_emoji``."""
    if pleasure > 0.3 and arousal > 0.3:
        return "excited"
    elif pleasure > 0.3 and arousal < -0.3:
        return "calm"
    elif pleasure > 0.3:
        return "happy"
    elif pleasure < -0.3 and arousal > 0.3:
        return "anxious"
    elif pleasure < -0.3:
        return "sad"
    elif arousal < -0.3:
        return "tired"
    elif dominance > 0.3:
        return "confident"
    elif arousal > 0.1:
        return "curious"
    elif dominance > 0.1:
        return "focused"
    return "neutral"


def get_mood_emoji(pleasure: float, arousal: float, dominance: float) -> str:
    """Map PAD values (-1 to 1) to an emoji."""
    return _MOOD_EMOJIS.get(_classify_mood(pleasure, arousal, dominance),
                            _MOOD_EMOJIS["neutral"])


def get_mood_label(pleasure: float, arousal: float, dominance: float) -> str:
    """Map PAD values to a human-readable mood label."""
    return _classify_mood(pleasure, arousal, dominance)


def create_mood_indicator(emotional_state: Optional[Dict] = None) -> str:
    """Create a compact mood indicator for the status bar."""
    if not emotional_state:
        return ""

    mood = emotional_state.get("mood", {})
    pad = mood.get("pad", {})
    p = pad.get("pleasure", 0.0)
    a = pad.get("arousal", 0.0)
    d = pad.get("dominance", 0.0)

    emoji = get_mood_emoji(p, a, d)
    return f"{emoji}"


def render_mood_detail(console: Console, emotional_state: Dict) -> None:
    """Render detailed mood information for /mood command."""
    mood = emotional_state.get("mood", {})
    pad = mood.get("pad", {})
    p = pad.get("pleasure", 0.0)
    a = pad.get("arousal", 0.0)
    d = pad.get("dominance", 0.0)

    emoji = get_mood_emoji(p, a, d)
    label = get_mood_label(p, a, d)

    text = Text()
    text.append(f"  Mood: {emoji} {label.title()}\n\n", style="bold")

    # PAD values with visual bars
    def pad_bar(value: float, label: str) -> str:
        normalized = (value + 1) / 2  # -1..1 -> 0..1
        filled = int(normalized * 10)
        bar = "\u2588" * filled + "\u2591" * (10 - filled)
        return f"  {label:<12} [{bar}] {value:+.2f}"

    text.append(pad_bar(p, "Pleasure") + "\n")
    text.append(pad_bar(a, "Arousal") + "\n")
    text.append(pad_bar(d, "Dominance") + "\n")

    # Neuromodulators
    neuro = emotional_state.get("neuromodulators", {})
    if neuro:
        text.append("\n  Neuromodulators:\n", style="bold")
        for name, value in neuro.items():
            filled = max(0, min(10, int(value * 10))) if isinstance(value, (int, float)) else 5
            value_display = value if isinstance(value, (int, float)) else 0.0
            bar = "\u2588" * filled + "\u2591" * (10 - filled)
            text.append(f"  {name:<16} [{bar}] {value_display:.2f}\n", style="dim")

    # Active emotions
    emotions = emotional_state.get("active_emotions", [])
    if emotions:
        text.append(f"\n  Active emotions ({len(emotions)}):\n", style="bold")
        for em in emotions[:5]:
            name = em.get("name", em) if isinstance(em, dict) else str(em)
            intensity = em.get("intensity", 0.5) if isinstance(em, dict) else 0.5
            text.append(f"    \u2022 {name} ({intensity:.1f})\n")

    # Recent influences
    influences = emotional_state.get("recent_influences", emotional_state.get("triggers", []))
    if influences:
        text.append("\n  Recent influences:\n", style="bold")
        for inf in influences[:3]:
            text.append(f"    \u2192 {inf}\n", style="dim")

    console.print(Panel(text, title="[bold magenta]Emotional State[/bold magenta]", border_style="magenta"))


def format_dream_insight(insight: str) -> str:
    """Format a dream insight for display as a proactive suggestion."""
    return f"[dim magenta]\U0001f4ad Dream insight: {insight}[/dim magenta]"


def format_break_suggestion(session_duration_minutes: float, cognitive_load: float) -> Optional[str]:
    """Suggest a break if the session is long and cognitive load is high."""
    if session_duration_minutes > 60 and cognitive_load > 0.7:
        return "[dim yellow]\u2615 You've been at this for over an hour with high cognitive load. Consider a short break.[/dim yellow]"
    elif session_duration_minutes > 120:
        return "[dim yellow]\u2615 Session is over 2 hours. A break might help with focus.[/dim yellow]"
    return None
