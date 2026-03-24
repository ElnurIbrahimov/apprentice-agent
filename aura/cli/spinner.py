"""Custom spinner for AURA CLI — bouncing Unicode frames + random verbs + elapsed timer."""
from __future__ import annotations

import random
import time
from rich.text import Text

# Bounce animation frames (forward then reverse, like Claude Code)
_FRAMES = ['\u00b7', '\u2722', '*', '\u2736', '\u273b', '\u273d']
_BOUNCE = _FRAMES + _FRAMES[-2:0:-1]  # [·, ✢, *, ✶, ✻, ✽, ✻, ✶, *, ✢]
FRAME_INTERVAL = 0.08  # 80ms per frame

# Aura-themed spinner verbs
SPINNER_VERBS = [
    "Thinking", "Perceiving", "Resonating", "Channeling", "Sensing",
    "Contemplating", "Manifesting", "Harmonizing", "Illuminating", "Attuning",
    "Synthesizing", "Crystallizing", "Weaving", "Flowing", "Pulsing",
    "Dreaming", "Evolving", "Awakening", "Transmuting", "Radiating",
    "Calibrating", "Decoding", "Unraveling", "Orchestrating", "Conjuring",
    "Computing", "Processing", "Analyzing", "Reflecting", "Exploring",
]


class AuraSpinner:
    """Animated spinner with verb + elapsed timer.

    Implements __rich__() so Rich's Live can call it on each refresh tick.
    """

    def __init__(self, label: str | None = None, step: int | None = None):
        self._start = time.monotonic()
        self._verb = label or random.choice(SPINNER_VERBS)
        self._step = step
        self._stall_threshold = 30.0  # seconds before color shifts to warn

    def __rich__(self) -> Text:
        """Called by Rich Live on each refresh — returns fresh frame."""
        return self.render()

    def render(self) -> Text:
        """Render current spinner frame with verb and elapsed time."""
        elapsed = time.monotonic() - self._start
        frame_idx = int(elapsed / FRAME_INTERVAL) % len(_BOUNCE)
        frame = _BOUNCE[frame_idx]

        # Color shifts toward red when stalling
        if elapsed > self._stall_threshold:
            frame_color = "red"
            verb_color = "red dim"
        else:
            frame_color = "cyan"
            verb_color = "dim"

        t = Text("  ")
        t.append(frame, style=f"bold {frame_color}")
        t.append(" ", style="")

        # Step counter if in agentic loop
        if self._step is not None:
            t.append(f"Step {self._step} \u00b7 ", style="dim")

        t.append(f"{self._verb}...", style=verb_color)

        # Elapsed timer
        elapsed_str = _format_elapsed(elapsed)
        t.append(f"  ({elapsed_str})", style="dim")

        return t

    def update_verb(self, verb: str) -> None:
        self._verb = verb

    def update_step(self, step: int) -> None:
        self._step = step


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
