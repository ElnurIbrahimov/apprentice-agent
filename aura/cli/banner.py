"""Banner and welcome line for AURA CLI.

Uses block characters (█ ▀ ▄) for a bold banner with Rich gradient colors.
Inspired by Gemini CLI's block-character approach.
"""
from __future__ import annotations

from rich.text import Text

# Gradient colors per row (cyan -> blue -> magenta)
_ROW_COLORS = [
    "bold cyan",
    "bold deep_sky_blue1",
    "bold dodger_blue1",
    "bold blue1",
    "bold dark_violet",
    "bold magenta",
]

# Block-character art for "AURA" — thick/bold style
# ONLY uses: █ (full block), ▀ (upper half), ▄ (lower half), and space
# Double-width strokes for maximum impact, 6 rows tall
_BANNER_LINES = [
    "  ▄████▄     ██    ██   ██████▄     ▄████▄  ",
    " ██▀  ▀██    ██    ██   ██   ▀██   ██▀  ▀██ ",
    " ██    ██    ██    ██   ██   ▄██   ██    ██ ",
    " ████████    ██    ██   ██████▀    ████████ ",
    " ██    ██    ▀██▄▄██▀   ██  ▀██   ██    ██ ",
    " ▀▀    ▀▀      ▀▀▀▀     ▀▀   ▀▀   ▀▀    ▀▀ ",
]


def _get_row_colors() -> list:
    """Get banner row colors from the active theme, falling back to defaults."""
    try:
        from aura.cli.themes import get_theme
        gradient = get_theme().banner_gradient
        if gradient and len(gradient) >= 2:
            # Expand gradient to 6 rows by cycling
            colors = []
            for i in range(6):
                colors.append(f"bold {gradient[i % len(gradient)]}")
            return colors
    except Exception:
        pass
    return _ROW_COLORS


def get_banner(width: int = 80) -> Text:
    """Return a Rich Text banner sized for the given terminal width."""
    text = Text()
    text.append("\n")

    banner_width = len(_BANNER_LINES[0])
    row_colors = _get_row_colors()

    if width >= banner_width + 4:
        # Full block-character banner with per-row gradient
        pad = " " * max(0, (width - banner_width) // 2 - 1)
        for i, line in enumerate(_BANNER_LINES):
            color = row_colors[i % len(row_colors)]
            text.append(pad)
            text.append(line, style=color)
            text.append("\n")
    elif width >= 20:
        # Compact styled text for narrow terminals
        pad = " " * max(0, (width - 10) // 2)
        text.append(pad)
        text.append("A", style="bold cyan")
        text.append(" U", style="bold dodger_blue1")
        text.append(" R", style="bold blue1")
        text.append(" A", style="bold magenta")
        text.append("\n")
    else:
        text.append("AURA", style="bold cyan")
        text.append("\n")

    return text


def get_welcome_line(version: str | None = None) -> Text:
    """Return the one-line welcome below the banner."""
    if version is None:
        try:
            from aura import __version__
            version = __version__
        except (ImportError, AttributeError):
            version = "4.3.0"
    t = Text()
    t.append(f"  v{version}", style="dim")
    t.append("  |  ", style="dim")
    t.append("/", style="bold cyan")
    t.append(" commands", style="dim")
    t.append("  |  ", style="dim")
    t.append("Alt+M", style="bold cyan")
    t.append(" model", style="dim")
    t.append("  |  ", style="dim")
    t.append("?", style="bold cyan")
    t.append(" help", style="dim")
    return t
