"""Banner for AURA CLI — clean single-line header."""
from __future__ import annotations

from rich.text import Text


def get_banner(width: int = 80) -> Text:
    """Return an empty Text — the banner is now part of the welcome line."""
    return Text()


def get_welcome_line(version: str | None = None) -> Text:
    """Return a clean 1-line banner+welcome: AURA v4.6.0 -- / commands ..."""
    if version is None:
        try:
            from aura import __version__
            version = __version__
        except (ImportError, AttributeError):
            version = "4.6.0"

    t = Text()
    t.append("  AURA", style="bold cyan")
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
    return t
