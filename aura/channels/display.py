"""
Channel display — Rich terminal rendering for channel bridge messages.
Shows incoming channel messages and outgoing responses with visual distinction.
"""
from __future__ import annotations

import time
from typing import List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

# ---------------------------------------------------------------------------
# Channel metadata helpers
# ---------------------------------------------------------------------------

_CHANNEL_ICONS: dict[str, str] = {
    "telegram": "\U0001f4f1",   # mobile phone
    "extension": "\U0001f50c",  # electric plug
    "discord": "\U0001f4ac",    # speech balloon
    "whatsapp": "\U0001f4de",   # telephone receiver
    "slack": "\U0001f4bc",      # briefcase
}
_DEFAULT_ICON = "\U0001f4e8"  # incoming envelope

_CHANNEL_COLORS: dict[str, str] = {
    "telegram": "bright_magenta",
    "extension": "bright_blue",
    "discord": "bright_cyan",
    "whatsapp": "bright_green",
    "slack": "yellow",
}
_DEFAULT_COLOR = "white"

# Max characters shown before truncation
_MAX_MSG_LEN = 2000


def get_channel_icon(channel: str) -> str:
    """Return an emoji icon for a channel name."""
    return _CHANNEL_ICONS.get(channel.lower(), _DEFAULT_ICON)


def get_channel_color(channel: str) -> str:
    """Return a Rich color string for a channel name."""
    return _CHANNEL_COLORS.get(channel.lower(), _DEFAULT_COLOR)


def _get_console(console: Optional[Console] = None) -> Console:
    """Return the provided console or fall back to the global AURA one."""
    if console is not None:
        return console
    try:
        from aura.cli.display import console as _global_console
        return _global_console
    except ImportError:
        return Console(highlight=False)


def _timestamp_now() -> str:
    return time.strftime("%H:%M:%S")


def _truncate(text: str, limit: int = _MAX_MSG_LEN) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"... ({len(text)} chars)"


# ---------------------------------------------------------------------------
# Public display functions
# ---------------------------------------------------------------------------

def print_channel_message(
    channel: str,
    username: str,
    text: str,
    console: Optional[Console] = None,
) -> None:
    """Display an incoming message from a channel.

    Renders a Rich Panel with channel-coloured border, username, timestamp,
    and message body.  Safe to call from background threads.
    """
    con = _get_console(console)
    color = get_channel_color(channel)
    icon = get_channel_icon(channel)
    ts = _timestamp_now()

    title = Text()
    title.append(f" {icon} ", style=f"bold {color}")
    title.append(channel.capitalize(), style=f"bold {color}")

    subtitle = Text(ts, style="dim")

    body = Text()
    body.append(f"{username}: ", style="bold")
    body.append(_truncate(text))

    panel = Panel(
        body,
        title=title,
        title_align="left",
        subtitle=subtitle,
        subtitle_align="right",
        border_style=color,
        padding=(0, 1),
        expand=True,
    )
    con.print(panel)


def print_channel_response(
    channel: str,
    text: str,
    console: Optional[Console] = None,
) -> None:
    """Display an outgoing response being sent to a channel.

    Uses dimmer styling with a directional arrow to indicate outbound.
    """
    con = _get_console(console)
    color = get_channel_color(channel)
    icon = get_channel_icon(channel)

    title = Text()
    title.append(" -> ", style=f"dim {color}")
    title.append(f"{icon} ", style=f"dim {color}")
    title.append(channel.capitalize(), style=f"dim {color}")

    body = Text(_truncate(text), style="dim")

    panel = Panel(
        body,
        title=title,
        title_align="left",
        border_style=f"dim {color}",
        padding=(0, 1),
        expand=True,
    )
    con.print(panel)


def print_channel_status(
    channels: List[str],
    console: Optional[Console] = None,
) -> None:
    """Show which channels are currently active as a compact line."""
    con = _get_console(console)

    line = Text()
    line.append("  Channels: ", style="dim")
    for i, ch in enumerate(channels):
        if i > 0:
            line.append(" \u00b7 ", style="dim")  # middle dot separator
        color = get_channel_color(ch)
        icon = get_channel_icon(ch)
        line.append(f"{icon} ", style=color)
        line.append(ch.capitalize(), style=f"bold {color}")
    con.print(line)


def print_channel_notification(
    channel: str,
    username: str,
    preview: str,
    console: Optional[Console] = None,
) -> None:
    """Print a compact one-line notification (e.g. while user is typing).

    Format: [Telegram] Elnur: Can you check... (queued)
    """
    con = _get_console(console)
    color = get_channel_color(channel)

    # Truncate preview to ~50 chars for compactness
    short = preview if len(preview) <= 50 else preview[:47] + "..."

    line = Text()
    line.append(f"  [{channel.capitalize()}] ", style=f"bold {color}")
    line.append(f"{username}: ", style="bold")
    line.append(short, style="dim")
    line.append(" (queued)", style="dim italic")
    con.print(line)


def format_bridge_banner(channels: List[str]) -> Panel:
    """Return a Rich Panel showing the bridge startup banner.

    Callers should ``console.print()`` the returned Panel.
    """
    body = Text()
    body.append("AURA CLI Bridge\n", style="bold cyan")

    # Channel list
    ch_line = Text()
    ch_line.append("Channels: ", style="dim")
    for i, ch in enumerate(channels):
        if i > 0:
            ch_line.append(" \u00b7 ", style="dim")
        color = get_channel_color(ch)
        icon = get_channel_icon(ch)
        ch_line.append(f"{icon} ", style=color)
        ch_line.append(ch.capitalize(), style=f"bold {color}")
    body.append_text(ch_line)
    body.append("\n")

    body.append(
        "Type normally \u2014 channel messages will\n"
        "appear above your prompt.",
        style="dim",
    )

    return Panel(
        body,
        border_style="bold cyan",
        padding=(1, 2),
        expand=True,
    )
