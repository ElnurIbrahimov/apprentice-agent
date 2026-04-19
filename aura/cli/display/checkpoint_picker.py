"""Checkpoint picker UI + rewind-result display.

Extracted from the old monolithic ``display.py`` to keep the package tidy.
"""
from __future__ import annotations

from typing import Any, Optional

# Import the package as a module — attribute access at call time ensures
# tests that patch `aura.cli.display.console` see the mock, and keeps the
# submodule pluggable.
from aura.cli import display as _display


def show_checkpoint_picker(checkpoints: list[dict[str, Any]]) -> Optional[str]:
    """Display checkpoints in a styled numbered list."""
    import time as _time
    colors = _display._get_theme_colors()

    if not checkpoints:
        _display.show_info("No checkpoints available.")
        return None

    display = checkpoints[:10]

    _display.console.print()
    _display.console.print("  [bold]Checkpoints[/bold]")
    _display.console.print()

    now = _time.time()
    for i, cp in enumerate(display, 1):
        delta = now - cp.get("timestamp", now)
        if delta < 60:
            rel = f"{int(delta)}s ago"
        elif delta < 3600:
            rel = f"{int(delta / 60)}m ago"
        elif delta < 86400:
            rel = f"{int(delta / 3600)}h ago"
        else:
            rel = f"{int(delta / 86400)}d ago"

        label = cp.get("label", "") or "-"
        files = cp.get("files", [])
        file_names = ", ".join(f.get("backup_name", "?") for f in files[:3])
        if len(files) > 3:
            file_names += f" +{len(files) - 3}"

        num = str(i).rjust(2)
        _display.console.print(
            f"  [{colors['accent']}]{num}[/{colors['accent']}]"
            f"  [dim]{rel.ljust(10)}[/dim]  {label.ljust(24)}  [dim]{file_names}[/dim]"
        )

    _display.console.print()

    try:
        raw = _display.console.input("  [dim]Pick checkpoint # (or Enter to cancel): [/dim]")
        raw = raw.strip()
        if not raw:
            return None
        idx = int(raw) - 1
        if 0 <= idx < len(display):
            return display[idx]["id"]
        else:
            _display.show_error(f"Invalid selection: {raw}")
            return None
    except (ValueError, EOFError, KeyboardInterrupt):
        return None


def show_rewind_result(success: bool, checkpoint_id: str) -> None:
    """Display the result of a rewind operation."""
    colors = _display._get_theme_colors()
    if success:
        _display.console.print(f"  [{colors['success']}]\u2713 Rewound to checkpoint {checkpoint_id}[/{colors['success']}]")
    else:
        _display.show_error("Failed to rewind")
