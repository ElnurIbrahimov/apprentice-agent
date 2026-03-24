"""Banner for AURA CLI — gradient-colored header using theme colors."""
from __future__ import annotations

from rich.text import Text


def _apply_gradient(text: str, colors: list[str]) -> Text:
    """Apply a color gradient across *text* using *colors*.

    Each character is assigned a color by splitting the text into
    equal-sized segments -- one segment per color in the list.
    """
    if not colors:
        return Text(text, style="bold cyan")
    result = Text()
    n = len(text)
    if n == 0:
        return result
    num_colors = len(colors)
    for i, ch in enumerate(text):
        # Map character position to a color index
        idx = min(i * num_colors // n, num_colors - 1)
        result.append(ch, style=f"bold {colors[idx]}")
    return result


def get_banner(width: int = 80) -> Text:
    """Return an empty Text -- the banner is part of the welcome line."""
    return Text()


def get_welcome_line(version: str | None = None) -> Text:
    """Return a 1-line banner: gradient AURA + version + shortcut hints."""
    if version is None:
        try:
            from aura import __version__
            version = __version__
        except (ImportError, AttributeError):
            version = "4.6.0"

    # Get gradient colors from the active theme
    try:
        from aura.cli.themes import get_theme
        colors = get_theme().gradient
    except (ImportError, AttributeError):
        colors = ["cyan", "blue", "magenta"]

    t = Text("  ")
    t.append_text(_apply_gradient("AURA", colors))
    t.append(f" v{version}", style="dim")
    t.append("  \u2014  ", style="dim")
    t.append("/", style="bold cyan")
    t.append(" commands", style="dim")
    t.append("  \u2022  ", style="dim")
    t.append("Alt+M", style="bold cyan")
    t.append(" model", style="dim")
    t.append("  \u2022  ", style="dim")
    t.append("?", style="bold cyan")
    t.append(" help", style="dim")
    t.append("  \u2022  ", style="dim")
    t.append("Shift+Tab", style="bold cyan")
    t.append(" perms", style="dim")
    return t
