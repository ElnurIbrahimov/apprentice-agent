"""Banner and welcome line for AURA CLI.

Uses Rich styled text instead of ASCII art for cross-platform compatibility.
"""

from rich.text import Text

# Gradient colors (cyan -> blue -> magenta)
_GRADIENT = ["cyan", "deep_sky_blue1", "dodger_blue1", "blue1", "dark_violet", "magenta"]


def get_banner(width: int = 80) -> Text:
    """Return a Rich Text banner sized for the given terminal width."""
    text = Text()
    text.append("\n")

    if width >= 50:
        # Spaced-out letters with gradient
        letters = "A   U   R   A"
        for i, ch in enumerate(letters):
            if ch == " ":
                text.append(" ")
            else:
                color = _GRADIENT[i % len(_GRADIENT)]
                text.append(ch, style=f"bold {color}")
        text.append("\n")

        # Gradient underline
        line_chars = "=" * min(len(letters), width - 4)
        for i, ch in enumerate(line_chars):
            color = _GRADIENT[i % len(_GRADIENT)]
            text.append(ch, style=color)
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
