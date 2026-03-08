"""ASCII art banners for AURA CLI — 3 responsive sizes."""

from rich.text import Text

# Full banner for terminals >= 60 columns
BANNER_FULL = r"""
    ╔═══╗ ╔╗ ╔╗ ╔═══╗  ╔═══╗
    ║╔═╗║ ║║ ║║ ║╔═╗║  ║╔═╗║
    ║║ ║║ ║║ ║║ ║╔═╝╝  ║║ ║║
    ║╔═╗║ ║║ ║║ ║║╔═╗  ║╔═╗║
    ║║ ║║ ║╚═╝║ ║║║╚╗  ║║ ║║
    ╚╝ ╚╝ ╚═══╝ ╚╝╚═╝  ╚╝ ╚╝"""

# Compact banner for terminals 40-59 columns
BANNER_COMPACT = r"""
   ╔══╗╔╗╔╗╔══╗╔══╗
   ║╔╗║║║║║║╔═╝║╔╗║
   ║╔╗║║╚╝║║║╔╗║╔╗║
   ╚╝╚╝╚══╝╚╝╚╝╚╝╚╝"""

# Tiny banner for terminals < 40 columns
BANNER_TINY = "[bold cyan]◆ AURA[/bold cyan]"

# Gradient colors for the banner (cyan → blue → magenta)
_GRADIENT = ["cyan", "deep_sky_blue1", "dodger_blue1", "blue1", "dark_violet", "magenta"]


def get_banner(width: int = 80) -> Text:
    """Return a Rich Text banner sized for the given terminal width."""
    if width >= 60:
        raw = BANNER_FULL
    elif width >= 40:
        raw = BANNER_COMPACT
    else:
        return Text.from_markup(BANNER_TINY)

    lines = raw.strip("\n").split("\n")
    text = Text()

    for li, line in enumerate(lines):
        # Pick gradient color based on line position
        color = _GRADIENT[li % len(_GRADIENT)]
        text.append(line, style=f"bold {color}")
        text.append("\n")

    return text


def get_welcome_line(version: str = "4.3.0") -> Text:
    """Return the one-line welcome below the banner."""
    t = Text()
    t.append(f"  v{version}", style="dim")
    t.append("  │  ", style="dim")
    t.append("/", style="bold cyan")
    t.append(" commands", style="dim")
    t.append("  │  ", style="dim")
    t.append("Ctrl+M", style="bold cyan")
    t.append(" model", style="dim")
    t.append("  │  ", style="dim")
    t.append("?", style="bold cyan")
    t.append(" help", style="dim")
    return t
