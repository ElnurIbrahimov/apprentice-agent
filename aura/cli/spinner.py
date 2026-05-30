"""Custom spinner for AURA CLI — shimmer animation + brand colors + elapsed timer."""
from __future__ import annotations

import math
import random
import time

from rich.text import Text

# Bounce animation frames (forward then reverse, like Claude Code / Codex)
_FRAMES = ["\u00b7", "\u2722", "*", "\u2736", "\u273b", "\u273d"]
_BOUNCE = _FRAMES + _FRAMES[-2:0:-1]  # [·, ✢, *, ✶, ✻, ✽, ✻, ✶, *, ✢]
FRAME_INTERVAL = 0.08  # 80ms per frame

# Shimmer sweep constants (cosine band animation like Claude Code)
_SHIMMER_PERIOD = 2.0   # seconds for one full sweep
_SHIMMER_BAND = 5.0     # half-width of highlight band in characters

# Default spinner verbs — practical, informative (phase-aware verbs in display.py
# override these once a specific tool is called)
SPINNER_VERBS = [
    "Thinking", "Processing", "Analyzing", "Working",
    "Reasoning", "Planning", "Generating",
]


def _get_accent_colors() -> tuple[str, str]:
    """Get accent and shimmer colors from theme."""
    try:
        from aura.cli.themes import get_theme
        theme = get_theme()
        return theme.accent, theme.accent_dim
    except (ImportError, AttributeError):
        from aura.cli.themes import AuraTheme
        _fb = AuraTheme(name="fallback")
        return _fb.accent, _fb.accent_dim


def _get_stall_colors() -> tuple[str, str]:
    """Get error colors for the stall state."""
    try:
        from aura.cli.themes import get_theme
        theme = get_theme()
        return theme.error, theme.error
    except (ImportError, AttributeError):
        from aura.cli.themes import AuraTheme
        _fb = AuraTheme(name="fallback")
        return _fb.error, _fb.error


def _shimmer_text(text: str, elapsed: float, base_color: str, shimmer_color: str) -> Text:
    """Apply a sweeping cosine-band shimmer highlight across text.

    Creates a glowing sweep effect where characters near the sweep position
    are highlighted with the shimmer color, fading to base color at edges.
    """
    result = Text()
    n = len(text)
    if n == 0:
        return result

    padding = 10
    total_width = n + padding * 2
    # Sweep position oscillates across the text
    pos = (elapsed % _SHIMMER_PERIOD) / _SHIMMER_PERIOD * total_width

    for i, ch in enumerate(text):
        i_pos = i + padding
        dist = abs(i_pos - pos)
        if dist <= _SHIMMER_BAND:
            # Cosine interpolation: 1.0 at center, 0.0 at edges
            intensity = 0.5 * (1.0 + math.cos(math.pi * dist / _SHIMMER_BAND))
        else:
            intensity = 0.0

        if intensity > 0.6:
            result.append(ch, style=f"bold {shimmer_color}")
        elif intensity > 0.2:
            result.append(ch, style=f"{base_color}")
        else:
            result.append(ch, style=f"dim {base_color}")

    return result


class AuraSpinner:
    """Animated spinner with shimmer verb + elapsed timer + token count.

    Implements __rich__() so Rich's Live can call it on each refresh tick.
    """

    def __init__(self, label: str | None = None, step: int | None = None):
        self._start = time.monotonic()
        self._verb = label or random.choice(SPINNER_VERBS)
        self._step = step
        self._stall_threshold = 30.0  # seconds before color shifts to warn
        self._tokens = 0
        self._fade_steps: int = 0
        # Parallel-tool counter. When >1, verb is replaced with "N tools running"
        # so the user sees that multiple tools are executing concurrently rather
        # than the spinner appearing stuck on a single label.
        self._parallel_count: int = 0

    def __rich__(self) -> Text:
        """Called by Rich Live on each refresh — returns fresh frame."""
        return self.render()

    def set_fade(self, steps: int = 3) -> None:
        """Enable a fade-in effect for the next N render calls."""
        self._fade_steps = steps

    def render(self) -> Text:
        """Render current spinner frame with shimmer verb and stats."""
        elapsed = time.monotonic() - self._start
        frame_idx = int(elapsed / FRAME_INTERVAL) % len(_BOUNCE)
        frame = _BOUNCE[frame_idx]
        accent, accent_dim = _get_accent_colors()

        # Color shifts toward the theme's error color when stalling — honors
        # user-selected themes instead of hardcoding a pink that clashes
        # outside the default look.
        if elapsed > self._stall_threshold:
            err, err_shimmer = _get_stall_colors()
            frame_style = f"bold {err}"
            verb_base = err
            verb_shimmer = err_shimmer
        else:
            frame_style = f"bold {accent}"
            verb_base = accent
            verb_shimmer = accent_dim

        # Apply dim during fade-in
        if self._fade_steps > 0:
            self._fade_steps -= 1
            frame_style = f"dim {frame_style}"

        t = Text("  ")
        t.append(frame, style=frame_style)
        t.append(" ", style="")

        # Step counter if in agentic loop
        if self._step is not None:
            t.append(f"Step {self._step} \u00b7 ", style="dim")

        # Shimmer effect on verb text. When running parallel tools, replace
        # the single verb with a count so users see that N tools are in flight.
        if self._parallel_count > 1:
            verb_text = f"{self._parallel_count} tools running..."
        else:
            verb_text = f"{self._verb}..."
        t.append_text(_shimmer_text(verb_text, elapsed, verb_base, verb_shimmer))

        # Elapsed timer
        elapsed_str = _format_elapsed(elapsed)
        t.append(f"  ({elapsed_str}", style="dim")

        # Token count (if tracking)
        if self._tokens > 0:
            if self._tokens >= 1000:
                tok_str = f"{self._tokens / 1000:.1f}K"
            else:
                tok_str = str(self._tokens)
            t.append(f" \u00b7 \u2193{tok_str} tokens", style="dim")

        t.append(")", style="dim")

        return t

    def update_verb(self, verb: str) -> None:
        """Change the displayed verb (e.g., when tool type changes)."""
        self._verb = verb

    def update_step(self, step: int) -> None:
        """Update the agentic step counter."""
        self._step = step

    def update_tokens(self, tokens: int) -> None:
        """Update the token count display."""
        self._tokens = tokens

    def set_parallel_count(self, n: int) -> None:
        """Set how many tools are running in parallel (0 or 1 = single label,
        >1 swaps the verb for "N tools running")."""
        self._parallel_count = max(0, int(n))


def _format_elapsed(seconds: float) -> str:
    """Format elapsed time compactly."""
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m"
