"""Banner for AURA CLI — gradient block-art logo with theme colors."""
from __future__ import annotations

from rich.text import Text

# Compact 2-line block art — looks striking, renders on all terminals
_LOGO_LINES = [
    "▄▀█ █ █ █▀█ █▀█",
    "█▀█ █▄█ █▀▄ █▀█",
]

_STAR = "✦"


def _apply_gradient(text: str, colors: list[str]) -> Text:
    """Apply a color gradient across text, skipping spaces."""
    if not colors:
        return Text(text, style="bold cyan")
    result = Text()
    n = len(text)
    if n == 0:
        return result
    # Count non-space chars for gradient mapping
    solid_chars = [i for i, ch in enumerate(text) if ch != " "]
    num_solid = len(solid_chars)
    num_colors = len(colors)
    # Map each solid char to a gradient color
    color_map = {}
    for j, pos in enumerate(solid_chars):
        idx = min(j * num_colors // max(num_solid, 1), num_colors - 1)
        color_map[pos] = colors[idx]
    for i, ch in enumerate(text):
        if ch == " ":
            result.append(ch)
        else:
            result.append(ch, style=f"bold {color_map.get(i, colors[0])}")
    return result


def get_banner(width: int = 80) -> Text:
    """Return gradient block-art AURA logo."""
    try:
        from aura.cli.themes import get_theme
        colors = get_theme().gradient
    except (ImportError, AttributeError):
        colors = ["#D777AF", "#B1B9F9", "#87D7D7"]

    result = Text()
    for line in _LOGO_LINES:
        result.append("   ")
        result.append_text(_apply_gradient(line, colors))
        result.append("\n")
    return result


def get_welcome_line(version: str | None = None) -> Text:
    """Return full startup display: logo + info + shortcuts."""
    if version is None:
        try:
            from aura import __version__
            version = __version__
        except (ImportError, AttributeError):
            version = "dev"

    try:
        from aura.cli.themes import get_theme
        theme = get_theme()
        colors = theme.gradient
        accent = theme.accent
    except (ImportError, AttributeError):
        colors = ["#D777AF", "#B1B9F9", "#87D7D7"]
        accent = "#D777AF"

    result = Text()

    # Block-art logo with gradient
    for line in _LOGO_LINES:
        result.append("   ")
        result.append_text(_apply_gradient(line, colors))
        result.append("\n")

    # Info line: star + version + shortcut hints
    result.append("   ")
    result.append(_STAR, style=f"bold {accent}")
    result.append(f" v{version}", style="dim")
    result.append("  \u2014  ", style="dim")
    result.append("/", style=f"bold {accent}")
    result.append(" commands", style="dim")
    result.append("  \u00b7  ", style="dim")
    result.append("Alt+M", style=f"bold {accent}")
    result.append(" model", style="dim")
    result.append("  \u00b7  ", style="dim")
    result.append("?", style=f"bold {accent}")
    result.append(" help", style="dim")
    result.append("  \u00b7  ", style="dim")
    result.append("Shift+Tab", style=f"bold {accent}")
    result.append(" perms", style="dim")

    # Sandbox tier line (only shown when non-default)
    try:
        from aura.core.permissions import SandboxTier, get_sandbox_tier
        tier = get_sandbox_tier()
        if tier != SandboxTier.UNRESTRICTED:
            label = "read-only" if tier == SandboxTier.READ_ONLY else "workspace-write"
            warn_color = "yellow" if tier == SandboxTier.WORKSPACE_WRITE else "red"
            result.append("\n   ")
            result.append("\u2622", style=f"bold {warn_color}")
            result.append(f" sandbox: {label}", style=warn_color)
    except Exception:
        pass

    return result
