"""Banner and welcome line for AURA CLI.

Uses block characters (█ ▀ ▄) for a bold banner with Rich gradient colors.
Inspired by Gemini CLI's block-character approach.
"""

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


def get_banner(width: int = 80) -> Text:
    """Return a Rich Text banner sized for the given terminal width."""
    text = Text()
    text.append("\n")

    banner_width = len(_BANNER_LINES[0])

    if width >= banner_width + 4:
        # Full block-character banner with per-row gradient
        pad = " " * max(0, (width - banner_width) // 2 - 1)
        for i, line in enumerate(_BANNER_LINES):
            color = _ROW_COLORS[i % len(_ROW_COLORS)]
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


def get_welcome_line(version: str = "4.3.0") -> Text:
    """Return the one-line welcome below the banner."""
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
